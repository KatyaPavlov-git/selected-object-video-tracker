"""main.py — entry point for the ground-target-tracking pipeline.

Milestone 1: open + validate a video and display/iterate it.
Milestone 2: click ONE target point to track, draw
it, and play the clip with the point overlaid. Prints a summary banner once a
video is selected.
Milestone 3: automatically build a border-safe ROI / patch around the target,
draw the ROI rectangle during playback, and show the cropped patch in a separate
window.
Milestone 4: preprocessing pipelines (--show-preprocess shows raw vs processed).
Milestone 5: ORB/AKAZE feature extraction from the patch (count + low-texture warn).
Milestone 5.5: feature visualization (--show-keypoints overlays keypoints).
Milestone 6: continuous Lucas-Kanade optical-flow tracking (--method of).
The tracked point's pixel coordinates now UPDATE per frame to follow the same
real-world ground point as the scene moves; a moving ROI + trajectory are drawn.
--method fixed reproduces the pre-M6 constant-point behavior (regression guard).
Milestone 7: Kalman smoothing / short-term prediction (--method of_kalman, default) — wraps
the optical-flow tracker with a constant-velocity filter; --show-measurement draws
the raw pre-Kalman point for comparison.
Experiment logging: --save writes logs/run_NNN/{config.json, frame_log.csv,
output.mp4, stats.json}. Performance metrics (FPS / ms-per-frame / runtime) are
reported at the end of every run.

Video selection (no need to type long paths): if --video is omitted, the program
looks in a local `videos/` folder; auto-loads if there is exactly one clip, shows
a numbered menu if there are several, and otherwise opens a native file picker.

Run from the project root (note: this machine has `python3`, not `python`):

    python3 -m ground_target_tracking.main                       # pick + click
    python3 -m ground_target_tracking.main --video "<clip>.mp4"  # explicit override
    python3 -m ground_target_tracking.main --no-display --point 288,512 --save
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config, evaluation, experiment, preprocessing, session, trackers, utils
except ImportError:  # fallback: `python3 main.py` from inside the package folder
    import config
    import evaluation
    import experiment
    import preprocessing
    import session
    import trackers
    import utils


def _parse_point(text: str) -> utils.Point2D:
    """argparse type for --point 'x,y'."""
    try:
        xs, ys = text.split(",")
        return utils.Point2D(float(xs), float(ys))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"--point must be 'x,y' (got '{text}').")


def point_from_rc(i: float, j: float) -> utils.Point2D:
    """Map the assignment's manual [i],[j] indices (row, col) to pixel x/y."""
    return utils.Point2D(float(j), float(i))


def _parse_point_rc(text: str) -> utils.Point2D:
    """argparse type for --point-rc 'i,j' (row,col — the task.pdf convention)."""
    try:
        i_s, j_s = text.split(",")
        return point_from_rc(float(i_s), float(j_s))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            f"--point-rc must be 'i,j' (row,col indices; got '{text}').")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ground_target_tracking",
        description="Classical-CV ground-target tracking (Milestone 1: video loading).",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Path to the input video file (optional override). If omitted, the "
        f"program searches the '{config.VIDEOS_DIR}/' folder, then opens a file picker.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode: do not open a GUI window; just read frames.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=config.DEFAULT_MAX_FRAMES,
        help="Stop after N frames (0 = whole clip). Useful for quick checks.",
    )
    p.add_argument(
        "--init-frame",
        type=int,
        default=config.INIT_FRAME_INDEX,
        help="Frame index shown for target-point selection (default: %(default)s).",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=config.PATCH_SIZE,
        help="ROI/patch side length in pixels (default: %(default)s).",
    )
    pt = p.add_mutually_exclusive_group()
    pt.add_argument(
        "--point",
        type=_parse_point,
        default=None,
        help="Target point 'x,y' in full-resolution pixels; skips the interactive "
        "click (required in --no-display mode, unless --point-rc is given).",
    )
    pt.add_argument(
        "--point-rc",
        type=_parse_point_rc,
        default=None,
        dest="point_rc",
        help="Target point as manual indices 'i,j' (row,col — the assignment's "
        "[i],[j] convention). Equivalent to --point 'j,i'.",
    )
    p.add_argument(
        "--method",
        choices=("of", "of_kalman", "fixed"),
        default="of_kalman",
        help="Tracking method: 'of_kalman' = optical flow + Kalman smoothing/"
        "prediction (M7, default — the validated production profile); "
        "'of' = Lucas-Kanade optical flow only (M6); "
        "'fixed' = hold the point constant (pre-M6 baseline / regression guard).",
    )
    p.add_argument(
        "--detector",
        choices=("orb", "akaze"),
        default="orb",
        help="Feature detector for the M5 patch inspection (default: %(default)s).",
    )
    p.add_argument(
        "--show-preprocess",
        action="store_true",
        help="M4: show a 'raw | processed' window of the optical-flow preprocessing.",
    )
    p.add_argument(
        "--show-keypoints",
        action="store_true",
        help="M5.5: overlay the detected keypoints on the patch window.",
    )
    p.add_argument(
        "--show-measurement",
        action="store_true",
        help="M7: also draw the raw pre-Kalman measurement point (grey) under the "
        "filtered one, to visualize smoothing/lag (most useful with --method of_kalman).",
    )
    p.add_argument(
        "--overlay-mask",
        action="store_true",
        help="DEPRECATED no-op: overlay masking (HUD X diagonals + center "
        "crosshair excluded from seeding/scoring; display unchanged) is ON by "
        "default since 2026-07-08. Kept for compatibility.",
    )
    p.add_argument(
        "--no-overlay-mask",
        action="store_true",
        help="Disable the default HUD-overlay masking for this run (for "
        "sources where the fixed central-reticle model does not apply).",
    )
    p.add_argument(
        "--experimental-regional-motion",
        action="store_true",
        help="DEPRECATED no-op: regional local-motion support is ON by default "
        "since 2026-07-08 (the validated production profile). Kept for "
        "compatibility.",
    )
    p.add_argument(
        "--no-regional-motion",
        action="store_true",
        help="Disable the regional local-motion path for this run (falls back "
        "to the legacy ROI-only tracker update).",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Write an experiment run to logs/run_NNN/ (config.json, frame_log.csv, "
        "output.mp4, stats.json).",
    )
    p.add_argument(
        "--reacq",
        action="store_true",
        help="DEPRECATED no-op: reacquisition (LOST->search->reseed->probation) "
        "is ON by default since 2026-07-08. Kept for compatibility.",
    )
    p.add_argument(
        "--no-reacq",
        action="store_true",
        help="Disable reacquisition for this run: LOST becomes terminal "
        "(the pre-M9 freeze behavior).",
    )
    args = p.parse_args(argv)
    if args.point_rc is not None:
        args.point = args.point_rc     # downstream code consumes args.point only
    return args


def _prompt_menu(videos) -> str:
    """Print a numbered menu of videos and return the user's chosen path."""
    print(f"[video] Multiple videos found in '{config.VIDEOS_DIR}/':")
    for i, path in enumerate(videos, 1):
        print(f"   [{i}] {os.path.basename(path)}")
    while True:
        try:
            choice = input(f"Select a video [1-{len(videos)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise ValueError(
                "No selection (non-interactive input). Pass --video to choose explicitly."
            )
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            return videos[int(choice) - 1]
        print(f"   Invalid choice '{choice}'. Enter a number from 1 to {len(videos)}.")


def select_video(args: argparse.Namespace) -> str:
    """Resolve which video to open, following the selection priority.

    1. --video, if provided (local file path or http/rtsp URL).
    2. exactly one supported video in config.VIDEOS_DIR  -> auto-select.
    3. several supported videos in config.VIDEOS_DIR      -> numbered menu.
    4. no videos/ folder, or it has no supported videos   -> native file picker.
    """
    # 1. Explicit override.
    if args.video:
        if not utils.is_stream_url(args.video) and not os.path.isfile(args.video):
            raise ValueError(f"--video path not found: {args.video}")
        return args.video

    # 2-3. The local videos/ folder.
    videos = utils.list_videos()
    if len(videos) == 1:
        print(f"[video] Auto-selected the only clip in '{config.VIDEOS_DIR}/': "
              f"{os.path.basename(videos[0])}")
        return videos[0]
    if len(videos) > 1:
        return _prompt_menu(videos)

    # 4. Nothing found -> native file picker.
    print(f"[video] No supported videos in '{config.VIDEOS_DIR}/'. Opening file picker…")
    path = utils.pick_video_via_dialog()
    if not path:
        raise ValueError(
            "No video selected. Provide one with --video, or add clips to the "
            f"'{config.VIDEOS_DIR}/' folder."
        )
    print(f"[video] Selected: {path}")
    return path


def select_point_interactive(frame, init_frame_idx: int) -> utils.Point2D:
    """Milestone 2: let the user click ONE target point on the init frame.

    The frame is shown via fit_display (scaled to fit the screen); the click is
    mapped back to full-resolution coordinates. Click to place, Enter/Space to
    confirm, 'r' to re-pick, 'q'/Esc to cancel. Returns the point in full-res.
    """
    disp, scale = utils.fit_display(frame)
    win = f"{config.WINDOW_TITLE} - select target"
    state = {"pt": None}  # stored in full-resolution coords

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = utils.Point2D(x / scale, y / scale)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("[M2] Click the target point to track. "
          "Enter/Space = confirm, r = reset, q/Esc = cancel.")
    try:
        while True:
            canvas = disp.copy()
            utils.draw_hud(canvas, [
                f"M2 select target  (init frame {init_frame_idx})",
                "click point | Enter=confirm  r=reset  q=cancel",
            ])
            if state["pt"] is not None:
                marker = utils.Point2D(state["pt"].x * scale, state["pt"].y * scale)
                utils.draw_point(canvas, marker, label="target")
            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and state["pt"] is not None:  # Enter / Space
                return state["pt"]
            if key == ord("r"):
                state["pt"] = None
            elif key in (ord("q"), 27):  # q / Esc
                raise ValueError("Target-point selection cancelled.")
    finally:
        cv2.destroyWindow(win)


def resolve_target_point(args, init_frame, init_idx: int, meta) -> utils.Point2D:
    """Resolve the target point (M2): from --point, or an interactive click."""
    if args.point is not None:
        target = args.point
        print(f"[M2] Using target point from --point: ({target.x:.1f}, {target.y:.1f})")
    elif args.no_display:
        raise ValueError("--no-display requires --point (no GUI to click in headless mode).")
    else:
        target = select_point_interactive(init_frame, init_idx)
        print(f"[M2] Selected target point: ({target.x:.1f}, {target.y:.1f})")
    if not (0 <= target.x < meta.width and 0 <= target.y < meta.height):
        raise ValueError(
            f"Target point {target.as_int()} is outside the frame "
            f"({meta.width}x{meta.height})."
        )
    return target


def build_patch(args, init_frame, target) -> "utils.Patch":
    """Build the border-safe ROI/patch around the target and report it (M3)."""
    patch_size = max(config.MIN_PATCH_SIZE, args.patch_size)
    if patch_size != args.patch_size:
        print(f"[M3] patch size {args.patch_size} too small; using "
              f"config.MIN_PATCH_SIZE={config.MIN_PATCH_SIZE}.")
    patch = utils.extract_patch(init_frame, target, patch_size)
    x0, y0, x1, y1 = patch.bbox()
    aw, ah = patch.actual_size
    tx, ty = target.as_int()
    clipped = " (clipped at border)" if (aw, ah) != (patch_size, patch_size) else ""
    print("[M3] Built ROI / patch (local context around the fixed ground target):")
    print(f"     target point          : ({tx}, {ty})")
    print(f"     requested patch size  : {patch_size}x{patch_size}")
    print(f"     ROI top-left          : ({x0}, {y0})")
    print(f"     ROI bottom-right      : ({x1}, {y1})")
    print(f"     ROI bbox (x0,y0,x1,y1): ({x0}, {y0}, {x1}, {y1})")
    print(f"     actual patch size     : {aw}x{ah}{clipped}")
    return patch


def inspect_features(args, patch):
    """M5: extract + report features on the initial patch; return the keypoints.

    Diagnostic only — ORB/AKAZE here inspect texture, they do not track.
    """
    detector = trackers.build_detector(args.detector)
    keypoints, _ = trackers.detect_features(patch.image, detector, kind=args.detector)
    n = len(keypoints)
    print(f"[M5] {args.detector.upper()} keypoints in patch: {n} "
          f"(warn threshold {config.ORB_MIN_KEYPOINTS_WARN})")
    if n < config.ORB_MIN_KEYPOINTS_WARN:
        if getattr(config, "EXPERIMENTAL_REGIONAL_MOTION", False):
            # This counts ORB features in the IMMEDIATE 51x51 display patch only;
            # under the regional experiment the tracker discovers support in the
            # surrounding region, so "choose another point" would be misleading.
            print("[M5] NOTE (regional): few ORB features in the immediate 51x51 "
                  "display patch, but the regional-motion path (default ON) draws tracking "
                  "support from the surrounding clean scene; keeping the selection.")
        else:
            print("[M5] WARNING: selected patch has too few features. "
                  "Choose a more textured point.")
    return keypoints


def _cli_snapshot(args) -> dict:
    """JSON-serializable snapshot of the CLI args for the run config.json."""
    snap = {}
    for k, v in vars(args).items():
        snap[k] = [v.x, v.y] if isinstance(v, utils.Point2D) else v
    return snap


def draw_regional_overlay(img, inner, center, measure, predict, sres=None) -> None:
    """EXPERIMENTAL (Proposal A) display-only overlay. Draws onto the annotated
    COPY only; nothing here is read back by LK / NCC / reference / confidence.

    Surfaces BOTH questions so they are never conflated: the regional gate (is
    THIS frame's local motion well estimated?) and the M8 state (given
    confidence + hysteresis, what do we report?). They can legitimately differ
    — e.g. gate=PASS(measure) during the M8 recovery-hysteresis window shows as
    LOW_CONFIDENCE even though a real measurement was consumed."""
    diag = getattr(inner, "last_regional_diag", None)
    if diag is None or center is None:
        return
    cx, cy = int(round(center.x)), int(round(center.y))
    radius = int(diag.get("radius", 0) or 0)
    passed = bool(diag.get("gate_pass", False))
    col = (0, 255, 0) if passed else (0, 165, 255)
    if radius > 0:
        cv2.circle(img, (cx, cy), radius, col, 1, cv2.LINE_AA)  # support boundary
    pts = getattr(inner, "pts", None)
    if pts is not None:
        for p in pts.reshape(-1, 2):
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 2,
                       (0, 255, 0), -1)          # clean accepted support
    lines = [
        f"[REGIONAL] r={radius} clean={diag.get('n_clean', 0)} "
        f"inl={diag.get('n_inliers', '-')}/{diag.get('n_coherent', '-')} "
        f"ratio={diag.get('inlier_ratio', '-')} resid={diag.get('residual', '-')}",
        f"parallax={diag.get('parallax', '-')} banddiff={diag.get('band_disagree', '-')} "
        f"centroid_off={diag.get('centroid_offset', '-')} lever={diag.get('lever_arm_px', '-')}",
        f"gate={'PASS(measure)' if passed else 'REJECT(predict)'} "
        f"reason={diag.get('reason', '-')}   measure={measure} predict={predict}",
    ]
    if sres is not None:
        sig = sres.signals
        src = sres.result.source
        state = sres.state.name
        sim = sig.get("similarity")
        # explain a state that differs from the gate outcome
        note = ""
        if src == "predict":
            note = "Kalman coast (M7 short-term prediction)"
        elif passed and state == "LOW_CONFIDENCE" and \
                sres.confidence >= config.LOW_CONFIDENCE_BELOW:
            note = "recovering: M8 hysteresis (measurement IS used)"
        elif sim is None:
            note = "appearance unverifiable (ROI under overlay) -> sim neutral"
        lines.append(
            f"M8: conf={sres.confidence:.2f} = trk={sig.get('tracker_conf', '-')}"
            f" x sim={'None' if sim is None else sim} x edge="
            f"{'0.5' if sig.get('edge') else '1.0'}  state={state} src="
            f"{'MEASURE' if src == 'measure' else 'PREDICT'}"
            + (f"  [{note}]" if note else ""))
    y = img.shape[0] - 24 * len(lines) - 2
    for ln in lines:
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        y += 24


def reacq_selection_tier(sess) -> tuple:
    """Selection-time reacquirability tiers (submission guard).

    Evaluated only when reacquisition is enabled for the run. Reuses the
    session's existing init flags and the immutable reference's capability
    flags — no new thresholds. Returns (tier, message):

      1  no usable target appearance (dead-zone ROI, or a reference with
         neither capability) — reacquisition cannot re-identify the selection;
      2  descriptor cue absent but context usable — template-only recovery;
      0  silent.
    """
    reacq = getattr(sess, "_reacq", None)
    ref = getattr(reacq, "reference", None) if reacq is not None else None
    no_capability = (ref is None
                     or (not ref.has_descriptors and not ref.has_context))
    if sess.init_dead_zone or no_capability:
        return 1, ("[M9] WARNING: the selected point lies under the burned-in "
                   "HUD (or a textureless region); the coordinate is accepted "
                   "and tracked from surrounding clean-scene support (HUD "
                   "pixels are excluded from all evidence), but reacquisition "
                   "may be unable to re-identify this selection after a loss. "
                   "For reliable reacquisition choose a textured, off-HUD "
                   "object.")
    if not ref.has_descriptors:
        return 2, ("[M9] NOTE: no feature (descriptor) reference at this "
                   "selection; reacquisition is limited to template matching "
                   "(near-upright, moderate scale changes).")
    return 0, ""


def announce_reacq_selection(sess) -> int:
    """Warn + continue: print the tier message (if any) and return the tier.

    The selection is ALWAYS accepted — this function has no access to any
    picker and cannot re-prompt; it only informs the user of the selection's
    reacquirability contract before the run proceeds.
    """
    tier, msg = reacq_selection_tier(sess)
    if tier:
        print(msg)
    return tier


def run(args: argparse.Namespace) -> int:
    # Defaults promotion (2026-07-08): overlay masking + regional motion are ON
    # by default (the validated production profile). The legacy opt-in flags are
    # kept as harmless no-ops; the --no-* flags opt out per run.
    if args.overlay_mask:
        config.OVERLAY_MASK_ENABLED = True
    if getattr(args, "experimental_regional_motion", False):
        config.EXPERIMENTAL_REGIONAL_MOTION = True
    if getattr(args, "no_overlay_mask", False):
        config.OVERLAY_MASK_ENABLED = False
    if getattr(args, "no_regional_motion", False):
        config.EXPERIMENTAL_REGIONAL_MOTION = False
    if config.EXPERIMENTAL_REGIONAL_MOTION and not config.OVERLAY_MASK_ENABLED:
        print("[REGIONAL] regional motion enabled WITHOUT overlay masking: "
              "no overlay sensing installed; support may include overlay corners.")
    video_path = select_video(args)
    cap, meta = utils.open_video(video_path)
    print(utils.video_summary(meta))

    # M2: choose the ground target point on the init frame.
    init_idx = max(0, min(args.init_frame, max(0, meta.frame_count - 1)))
    init_frame = utils.get_frame(cap, init_idx)
    target = resolve_target_point(args, init_frame, init_idx, meta)

    # M3: build the border-safe ROI/patch around the target.
    patch = build_patch(args, init_frame, target)

    # M5: extract + report features on the initial patch (diagnostic).
    keypoints = inspect_features(args, patch)

    # Experiment logging (--save) + performance metrics (always).
    logger = None
    if args.save:
        logger = experiment.RunLogger()
        logger.save_config(_cli_snapshot(args))
        logger.open_video(meta.fps, (meta.width, meta.height))
        print(f"[save] writing experiment run to {logger.run_dir}/")
    perf = evaluation.PerfStats()
    of_pipeline = preprocessing.get_pipeline("of")  # for the --show-preprocess view

    # M6: build the tracker and initialize it on the selected point. The tracked
    # point's pixel coordinates now UPDATE per frame (optical flow) so it follows
    # the same real-world ground point as the scene moves. `--method fixed` keeps
    # the pre-M6 constant-point behavior as a regression guard.
    # M8: the tracker is driven through a TrackingSession — the control layer
    # that scores confidence and owns the TRACKING/LOW_CONFIDENCE/PREDICT/LOST
    # state; trackers themselves stay state-free.
    tracker = trackers.make_tracker(args.method, config)
    # G1 (2026-07-08): reacquisition is ON by default; --no-reacq opts out
    # per run (terminal-LOST, the pre-M9 behavior); the legacy --reacq opt-in
    # is a harmless no-op. None defers to config.REACQ_DECISION_ENABLED.
    enable_reacq = (False if getattr(args, "no_reacq", False)
                    else (True if args.reacq else None))
    sess = session.TrackingSession(tracker, config, enable_reacq=enable_reacq)
    sess.init(init_frame, target)
    if getattr(args, "no_reacq", False):
        print("[M9] reacquisition DISABLED for this run (--no-reacq): "
              "LOST is terminal.")
    # Selection-time reacquirability notice (Run-A postmortem, revised to
    # warn + continue): the selection is ALWAYS accepted — a point under the
    # burned-in HUD is a valid coordinate, tracked from surrounding clean
    # scene (HUD pixels stay excluded from all evidence). The notice only
    # states the reacquirability contract before the run proceeds.
    if sess._reacq_enabled:
        announce_reacq_selection(sess)
    _reacq_events_seen = 0
    # Init-on-overlay policy (M8): the selection is always accepted; report
    # honestly what the tracker can actually see there.
    regional_on = getattr(config, "EXPERIMENTAL_REGIONAL_MOTION", False)
    if sess.init_dead_zone and regional_on:
        # `init_dead_zone` describes ONLY the legacy 51x51 seed policy; it is
        # logging-only (never read by the state machine). Under the regional
        # experiment the surrounding clean scene is used, so the dead-zone
        # warning is materially misleading — report the truth instead.
        print(f"[M8/REGIONAL] NOTE: the immediate 51x51 ROI is not seedable under "
              f"the legacy policy ({sess.init_seedable:.0%} seedable), but "
              "the regional-motion path (default ON) is active: tracking uses clean "
              "scene support from the surrounding region. PREDICT is used only "
              "when the regional evidence-quality gates fail (not a permanent "
              "dead zone).")
    elif sess.init_dead_zone:
        print(f"[M8] WARNING: selected point lies in the HUD-overlay dead zone "
              f"(only {sess.init_seedable:.0%} of the ROI is seedable scene). "
              "No measurements are possible there; the tracker will coast "
              "(PREDICT/LOW_CONFIDENCE) until the target emerges from the overlay.")
    elif sess.init_on_overlay or sess.init_overlay_cov > 0:
        print(f"[M8] NOTE: selected target overlaps the HUD overlay "
              f"(ROI coverage {sess.init_overlay_cov:.0%}); tracking uses the "
              "surrounding clean scene support only.")
    current = target
    traj = [current]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    window = config.WINDOW_TITLE
    read = 0
    # G6 (2026-07-08): waitKey pacing is ADAPTIVE — the per-frame delay is
    # the source frame period MINUS the measured loop time, so playback
    # approaches source speed instead of stacking a fixed sleep on top of
    # processing (measured: waitKey(16) sleeps ~26 ms on macOS, playing the
    # 59.8 fps official sample at ~29 fps wall-clock). Display-only; every
    # measured timing boundary excludes waitKey.
    frame_period_ms = 1000.0 / (meta.fps or config.DEFAULT_FPS)
    tx, ty = target.as_int()
    show = not args.no_display
    # Effective debug-window switches: CLI flag OR the temporary config toggle
    # (config.DEBUG_SHOW_* are temporary visual-validation switches, removable later).
    show_preprocess = args.show_preprocess or config.DEBUG_SHOW_PREPROCESS
    show_keypoints = args.show_keypoints or config.DEBUG_SHOW_KEYPOINTS
    show_measurement = args.show_measurement
    # Precompute the zoomed patch view (the INITIAL patch). Shown INSIDE the loop,
    # AFTER the main/preprocess windows, so the small patch window is created last
    # and lands on top instead of hidden behind them (macOS window z-order).
    patch_view = None
    if show and config.SHOW_PATCH_WINDOW:
        pv = (utils.draw_keypoints(patch.image, keypoints)
              if show_keypoints else patch.image)
        patch_view = cv2.resize(pv, None, fx=config.PATCH_VIEW_ZOOM,
                                fy=config.PATCH_VIEW_ZOOM, interpolation=cv2.INTER_NEAREST)
    patch_win_placed = False

    # EXPERIMENTAL (Proposal A) display-only counters + inner-tracker handle.
    regional_on = getattr(config, "EXPERIMENTAL_REGIONAL_MOTION", False)
    regional_inner = getattr(sess.tracker, "inner", sess.tracker) if regional_on else None
    reg_measure = reg_predict = 0

    perf.start()
    try:
        for idx, frame in utils.read_frames(cap):
            t_loop = time.perf_counter()
            if args.max_frames and read >= args.max_frames:
                break
            read += 1
            fps_now = perf.summarize()["fps"]  # from completed frames (outside timing)

            perf.tic()
            # M8: the session scores the measurement, owns the state machine and
            # commits (or freezes) the point; `sres.state` is the honest status.
            sres = sess.step(frame)
            if sess._reacq_enabled and len(sess._reacq_events) > _reacq_events_seen:
                for ev in sess._reacq_events[_reacq_events_seen:]:
                    print(f"[M9-c] frame {read}: {ev}")
                _reacq_events_seen = len(sess._reacq_events)
            result = sres.result
            current = sres.point
            state = sres.state
            if regional_on:
                if result.source == "measure":
                    reg_measure += 1
                else:
                    reg_predict += 1
            if state not in (session.TrackState.LOST, session.TrackState.FEED_FROZEN):
                traj.append(current)  # LOST / FEED_FROZEN freeze the trajectory at the held point
            live_bbox = utils.clamp_roi(current.x, current.y, patch.size,
                                        meta.width, meta.height)
            cx, cy = current.as_int()

            annotated = None
            if show or logger is not None:
                box_color = (config.ROI_BOX_COLOR_LOST
                             if state in (session.TrackState.LOST,
                                          session.TrackState.FEED_FROZEN)
                             else config.ROI_BOX_COLOR if state is session.TrackState.TRACKING
                             else config.ROI_BOX_COLOR_LOW)
                annotated = frame.copy()
                utils.draw_trajectory(annotated, traj)
                utils.draw_roi(annotated, live_bbox, color=box_color)
                if show_measurement and result.raw_point is not None:
                    # raw pre-Kalman measurement (grey) under the filtered marker
                    utils.draw_point(annotated, result.raw_point,
                                     color=config.MEASUREMENT_POINT_COLOR, radius=3)
                utils.draw_point(annotated, current, label="target")
                hud_lines = [
                    f"[{args.method}]  frame {idx + 1}/{meta.frame_count}  "
                    f"pts={result.n_points}  conf={sres.confidence:.2f}  {state.value}",
                    f"target=({cx},{cy})  fps~{fps_now:.0f}   press 'q' to quit",
                ]
                if state is session.TrackState.LOST:
                    hud_lines.append(config.LOST_BANNER_TEXT)
                elif state is session.TrackState.FEED_FROZEN:
                    hud_lines.append(config.FEED_FROZEN_BANNER_TEXT)
                utils.draw_hud(annotated, hud_lines)
                if regional_on and regional_inner is not None:
                    draw_regional_overlay(annotated, regional_inner, current,
                                          reg_measure, reg_predict, sres)
            disp = combo = None
            if show:
                disp, _ = utils.fit_display(annotated)
                if show_preprocess:
                    proc_bgr = cv2.cvtColor(of_pipeline(frame), cv2.COLOR_GRAY2BGR)
                    rdisp, _ = utils.fit_display(frame)
                    pdisp, _ = utils.fit_display(proc_bgr)
                    combo = np.hstack([rdisp, pdisp])
            perf.toc()

            if logger is not None:
                logger.write_frame(annotated)
                logger.log_frame(
                    idx, current, n_points=result.n_points,
                    mean_error=(None if result.mean_error == float("inf") else result.mean_error),
                    source=result.source, step_ms=perf.last_ms(),
                    confidence=sres.confidence, state=state.name,
                )

            if show:
                cv2.imshow(window, disp)
                if combo is not None:
                    cv2.imshow("Preprocess: raw | processed", combo)
                if patch_view is not None:
                    # Shown last -> created on top of the other windows, not behind.
                    cv2.imshow(config.PATCH_WINDOW_TITLE, patch_view)
                    if not patch_win_placed:
                        cv2.moveWindow(config.PATCH_WINDOW_TITLE,
                                       config.PATCH_WINDOW_X, config.PATCH_WINDOW_Y)
                        patch_win_placed = True
                delay = max(1, int(frame_period_ms
                                   - (time.perf_counter() - t_loop) * 1000.0))
                if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                    print("[run] quit requested by user.")
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if logger is not None:
            # Finalize the mp4 (write the moov atom) + flush the CSV on EVERY exit
            # path, including exceptions/early quit. close() is idempotent, so the
            # normal-path logger.finish(stats) below still writes stats.json.
            logger.close()
    perf.stop()

    stats = perf.summarize()
    fx, fy = current.as_int()
    print(f"[perf] {perf.render()}")
    print(f"[run] method={args.method}  read {read} frame(s)  "
          f"start=({tx},{ty})  end=({fx},{fy})  patch={patch.size}px")
    if logger is not None:
        logger.finish(stats)
        print(f"[save] run complete: {logger.run_dir}/  "
              f"(config.json, frame_log.csv[{logger.n_rows} rows], output.mp4, stats.json)")
    return 0


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse error / --help
        return int(exc.code) if exc.code is not None else 2
    try:
        return run(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
