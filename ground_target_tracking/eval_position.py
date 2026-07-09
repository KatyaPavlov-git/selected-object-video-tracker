"""eval_position.py — headless position-accuracy runner (Stage 0, seed of M11).

Drives the REAL tracking pipeline (make_tracker -> TrackingSession -> step) with
no GUI and measures per-frame TARGET POSITION ERROR against an independent
reference, so a "confident but wrong" lock is reported as `accepted_wrong` instead
of passing as TRACKING. It changes NOTHING in tracking — it only observes the
committed points the session already produces.

Two modes:
  * real     — a video + a selected point; reference = AnnulusReference (nearby
               scene motion, HUD-overlay pixels excluded). Mirrors the validated
               long-run stepping (read frame 0 -> init -> step subsequent frames).
  * synthetic— warp frame 0 by a KNOWN transform ramp; reference = the analytic
               true point (ABSOLUTE ground truth; no HUD, no shared evidence).

Examples:
  python3 -m ground_target_tracking.eval_position real \
      --video videos/v8.mp4 --point 970,532 --method of_kalman \
      --overlay-mask --experimental-regional-motion --frames 750 \
      --baseline scratchpad/.../stab_long.json --out report.json
  python3 -m ground_target_tracking.eval_position synthetic \
      --video videos/v8.mp4 --point 300,300 --motion translate --dx 6 --frames 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.eval_position`
    from . import config, evaluation, preprocessing, session, trackers, utils
except ImportError:  # fallback: run from inside the package folder
    import config
    import evaluation
    import preprocessing  # noqa: F401  (kept for parity with main.py construction)
    import session
    import trackers
    import utils


def _parse_point(text: str) -> "utils.Point2D":
    xs, ys = text.split(",")
    return utils.Point2D(float(xs), float(ys))


def _make_session(method: str, overlay: bool, regional: bool,
                  frame0: np.ndarray, point: "utils.Point2D"):
    """Mirror main.py's construction exactly (config flag opt-ins included)."""
    config.OVERLAY_MASK_ENABLED = bool(overlay)
    config.EXPERIMENTAL_REGIONAL_MOTION = bool(regional)
    tracker = trackers.make_tracker(method, config)
    sess = session.TrackingSession(tracker, config)
    sess.init(frame0, point)
    inner = getattr(tracker, "inner", tracker)
    return sess, inner


def run_real(video: str, point: "utils.Point2D", method: str, overlay: bool,
             regional: bool, frames: int, tol_px: float) -> dict:
    """Drive the real pipeline on `video`; measure vs an AnnulusReference.

    Stepping matches the validated long-run: read frame 0, init on it, then step
    the subsequent frames (t=1..frames). End-to-end fps includes decode.
    """
    cap = cv2.VideoCapture(video)
    ok, f0 = cap.read()
    if not ok:
        raise ValueError(f"could not read frame 0 of {video}")
    sess, inner = _make_session(method, overlay, regional, f0, point)
    ref = evaluation.AnnulusReference(f0, point, config)
    acc = evaluation.PositionAccuracy(tol_px)
    clock = evaluation.WallClock()
    traj: List[Tuple[float, float]] = []
    reg_rows: List[dict] = []

    t = 0
    while t < frames:
        clock.tic()
        ok, frame = cap.read()
        if not ok:
            clock.toc()
            break
        t += 1
        res = sess.step(frame)            # the real, unmodified pipeline
        clock.toc()                       # decode + step timed end-to-end
        ref_pt, alive = ref.update(frame)  # independent, HUD-excluded cross-check
        acc.add(t, res.point, ref_pt, res.state.name, res.result.source, alive)
        traj.append((round(float(res.point.x), 1), round(float(res.point.y), 1)))
        d = getattr(inner, "last_regional_diag", None) or {}
        reg_rows.append({"t": t, "state": res.state.name,
                         "source": res.result.source,
                         "x": round(float(res.point.x), 1),
                         "y": round(float(res.point.y), 1),
                         "jump_vetoed": res.signals.get("jump_vetoed"),
                         "reg_gate": d.get("gate_pass"),
                         "reg_clean": d.get("n_clean"),
                         "reg_resid": d.get("residual")})
    cap.release()

    return {
        "mode": "real", "video": os.path.basename(video),
        "point": [point.x, point.y], "method": method,
        "overlay_mask": overlay, "regional": regional,
        "reference": "annulus (nearby scene, HUD-excluded; cross-check not GT)",
        "reference_ok_at_init": bool(ref.ok0),
        "reference_anchors": int(len(ref.anchors0)),
        "reference_anchors_on_overlay": ref.anchors_on_overlay(),
        "reference_overlay_px_in_roi": ref.overlay_px_in_roi,
        "accuracy": acc.summary(),
        "timing": clock.summary(),
        "trajectory": traj,
        "rows": reg_rows,
    }


def run_synthetic(video: str, point: "utils.Point2D", method: str, motion: str,
                  frames: int, dx: float, dy: float, zoom: float, rot: float,
                  tol_px: float) -> dict:
    """Absolute-GT run: warp frame 0 by a known ramp; reference = analytic point."""
    cap = cv2.VideoCapture(video)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame 0 of {video}")
    h, w = base.shape[:2]
    if motion == "translate":
        transforms = evaluation.translation_ramp(frames, dx, dy)
    elif motion == "zoom":
        transforms = evaluation.zoom_ramp(frames, zoom, w / 2.0, h / 2.0)
    elif motion == "rotate":
        transforms = evaluation.rotation_ramp(frames, rot, w / 2.0, h / 2.0)
    else:
        raise ValueError(f"unknown motion '{motion}'")
    seq = evaluation.synthetic_gt_sequence(base, point, transforms)

    # No HUD in the synthetic scene: overlay/regional off (absolute clean GT).
    sess, _ = _make_session(method, False, False, base, point)
    acc = evaluation.PositionAccuracy(tol_px)
    clock = evaluation.WallClock()
    traj: List[Tuple[float, float]] = []
    for t, (frame, true_pt) in enumerate(seq, start=1):
        clock.tic()
        res = sess.step(frame)
        clock.toc()
        acc.add(t, res.point, true_pt, res.state.name, res.result.source, None)
        traj.append((round(float(res.point.x), 1), round(float(res.point.y), 1)))

    return {
        "mode": "synthetic", "video": os.path.basename(video),
        "point": [point.x, point.y], "method": method, "motion": motion,
        "reference": "analytic warp (ABSOLUTE ground truth)",
        "dx": dx, "dy": dy, "zoom": zoom, "rot": rot,
        "accuracy": acc.summary(), "timing": clock.summary(), "trajectory": traj,
    }


def compare_baseline(rows: List[dict], baseline_path: str) -> dict:
    """Regression check: committed (x,y)/state/source vs a stab_long-style JSON."""
    base = json.load(open(baseline_path))
    brows = {r["t"]: r for r in base.get("rows", base)}
    keys = ["x", "y", "state", "source"]
    compared = 0
    mism = []
    for r in rows:
        b = brows.get(r["t"])
        if b is None:
            continue
        compared += 1
        diff = {k: (b.get(k), r.get(k)) for k in keys if b.get(k) != r.get(k)}
        if diff:
            mism.append({"t": r["t"], "diff": diff})
    return {"compared": compared, "mismatches": len(mism), "first": mism[:5]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_position",
                                 description="Stage-0 position-accuracy runner")
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("real")
    r.add_argument("--video", required=True)
    r.add_argument("--point", required=True, type=_parse_point)
    r.add_argument("--method", default="of_kalman")
    r.add_argument("--overlay-mask", action="store_true")
    r.add_argument("--experimental-regional-motion", action="store_true")
    r.add_argument("--frames", type=int, default=750)
    r.add_argument("--tol", type=float, default=25.0)
    r.add_argument("--baseline", default=None)
    r.add_argument("--out", default=None)

    s = sub.add_parser("synthetic")
    s.add_argument("--video", required=True)
    s.add_argument("--point", required=True, type=_parse_point)
    s.add_argument("--method", default="of_kalman")
    s.add_argument("--motion", default="translate",
                   choices=["translate", "zoom", "rotate"])
    s.add_argument("--frames", type=int, default=60)
    s.add_argument("--dx", type=float, default=6.0)
    s.add_argument("--dy", type=float, default=0.0)
    s.add_argument("--zoom", type=float, default=0.01)
    s.add_argument("--rot", type=float, default=0.5)
    s.add_argument("--tol", type=float, default=5.0)
    s.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.mode == "real":
        rep = run_real(args.video, args.point, args.method, args.overlay_mask,
                       args.experimental_regional_motion, args.frames, args.tol)
        if args.baseline:
            rep["baseline_check"] = compare_baseline(rep["rows"], args.baseline)
    else:
        rep = run_synthetic(args.video, args.point, args.method, args.motion,
                            args.frames, args.dx, args.dy, args.zoom, args.rot,
                            args.tol)

    printable = {k: v for k, v in rep.items() if k not in ("trajectory", "rows")}
    print(json.dumps(printable, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"[eval] full report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
