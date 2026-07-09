"""evaluation.py — engineering performance metrics (now) + accuracy metrics (later).

Implemented now (Batch 1): `PerfStats` — simple engineering metrics (FPS, average
processing time per frame, total runtime) collected from the first runs onward, so
we are not waiting until M11 to measure performance.

Added (Stage 0 — seed of Milestone 11): position-accuracy evaluation. The tracker
reports acceptance/state, but not whether the box is ON the target; these helpers
measure per-frame TARGET POSITION ERROR against an independent reference so a
"confident but wrong" lock (accepted-wrong) is caught. Three honesty tiers:
    - `synthetic_gt_sequence` — warp a frame by a KNOWN transform (absolute ground
      truth; no HUD, no shared evidence);
    - `AnnulusReference` — track scene anchors in an annulus around the point,
      EXCLUDING HUD-overlay pixels and using DIFFERENT points than the tracker's own
      support (a cross-check with accumulated LK error, NOT ground truth);
    - manual keyframe labels (supplied by the caller).
`PositionAccuracy` classifies each frame and `WallClock` measures END-TO-END fps
(decode+step+display) that `PerfStats` deliberately omits.

Still deferred:
    Impact measurement — Milestone 10 (retired).
    Multi-video accuracy comparison table — Milestone 11 (Stage 4).
"""
from __future__ import annotations

import statistics
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.eval_position`
    from . import config as _config
    from . import utils
except ImportError:  # fallback: run from inside the package folder
    import config as _config
    import utils


class PerfStats:
    """Collect lightweight per-run performance metrics.

    Usage in a frame loop:
        perf = PerfStats(); perf.start()
        for frame in ...:
            perf.tic()
            ...processing...
            perf.toc()
        perf.stop()
        print(perf.render()); stats = perf.summarize()

    FPS is computed from per-frame *processing* time (the tic/toc spans), so it is
    not skewed by interactive `waitKey` delays; `total_runtime_s` is wall-clock.
    """

    def __init__(self) -> None:
        self._frame_ms: List[float] = []
        self._t0: Optional[float] = None
        self._run_start: Optional[float] = None
        self._run_end: Optional[float] = None

    def start(self) -> None:
        self._run_start = time.perf_counter()

    def tic(self) -> None:
        self._t0 = time.perf_counter()

    def toc(self) -> None:
        if self._t0 is not None:
            self._frame_ms.append((time.perf_counter() - self._t0) * 1000.0)
            self._t0 = None

    def stop(self) -> None:
        self._run_end = time.perf_counter()

    def last_ms(self) -> Optional[float]:
        return self._frame_ms[-1] if self._frame_ms else None

    def summarize(self) -> dict:
        n = len(self._frame_ms)
        proc_s = sum(self._frame_ms) / 1000.0
        avg_ms = (sum(self._frame_ms) / n) if n else 0.0
        fps = (n / proc_s) if proc_s > 0 else 0.0
        if self._run_start is not None:
            end = self._run_end if self._run_end is not None else time.perf_counter()
            total = end - self._run_start
        else:
            total = proc_s
        return {
            "n_frames": n,
            "avg_ms_per_frame": round(avg_ms, 3),
            "fps": round(fps, 2),
            "total_runtime_s": round(total, 3),
        }

    def render(self) -> str:
        s = self.summarize()
        return (f"frames={s['n_frames']}  fps={s['fps']}  "
                f"avg={s['avg_ms_per_frame']} ms/frame  total={s['total_runtime_s']} s")


# =========================================================================== #
# Position-accuracy evaluation (Stage 0 — seed of Milestone 11)                #
# =========================================================================== #
# The tracker/session report acceptance and state, never whether the committed
# box is ON the target. These helpers measure per-frame TARGET POSITION ERROR
# against an INDEPENDENT reference, so a "confident but wrong" lock is caught as
# `accepted_wrong` instead of passing as TRACKING. Read-only: nothing here alters
# tracking; it observes committed points a caller supplies.


class WallClock:
    """End-to-end iteration timer (decode + step + display) — the span PerfStats
    omits. Use tic() at the top of a frame iteration and toc() at the bottom."""

    def __init__(self) -> None:
        self._spans: List[float] = []
        self._t: Optional[float] = None

    def tic(self) -> None:
        self._t = time.perf_counter()

    def toc(self) -> None:
        if self._t is not None:
            self._spans.append(time.perf_counter() - self._t)
            self._t = None

    def summary(self) -> dict:
        n = len(self._spans)
        s = sum(self._spans)
        return {
            "n_frames": n,
            "end_to_end_fps": round(n / s, 2) if s > 0 else 0.0,
            "avg_ms_per_iter": round(s / n * 1000.0, 3) if n else 0.0,
        }


def _affine(a, b, c, d, e, f) -> np.ndarray:
    return np.array([[a, b, c], [d, e, f]], np.float32)


def translation_ramp(n: int, dx: float, dy: float) -> List[np.ndarray]:
    """Cumulative pure-translation transforms t=1..n (absolute-GT synthetic case)."""
    return [_affine(1, 0, i * dx, 0, 1, i * dy) for i in range(1, n + 1)]


def zoom_ramp(n: int, per_frame: float, cx: float, cy: float) -> List[np.ndarray]:
    """Cumulative zoom about (cx, cy): scale s=(1+per_frame)^i each frame."""
    out = []
    for i in range(1, n + 1):
        s = (1.0 + per_frame) ** i
        out.append(_affine(s, 0, cx - s * cx, 0, s, cy - s * cy))
    return out


def rotation_ramp(n: int, deg_per_frame: float, cx: float, cy: float) -> List[np.ndarray]:
    """Cumulative rotation about (cx, cy) by deg_per_frame each frame."""
    out = []
    for i in range(1, n + 1):
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), deg_per_frame * i, 1.0)
        out.append(M.astype(np.float32))
    return out


def synthetic_gt_sequence(base_bgr: np.ndarray, point: "utils.Point2D",
                          transforms: Sequence[np.ndarray]
                          ) -> List[Tuple[np.ndarray, "utils.Point2D"]]:
    """Warp `base_bgr` by each 2x3 affine and apply the SAME transform to `point`.

    Returns [(frame, true_point), ...] — ABSOLUTE ground truth: the true target
    pixel is computed analytically, involving no HUD and no shared tracking
    evidence, so position error against it is exact.
    """
    h, w = base_bgr.shape[:2]
    p0 = np.array([point.x, point.y, 1.0], np.float64)
    seq: List[Tuple[np.ndarray, "utils.Point2D"]] = []
    for M in transforms:
        M = np.asarray(M, np.float32)
        warped = cv2.warpAffine(base_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        tp = M.astype(np.float64) @ p0
        seq.append((warped, utils.Point2D(float(tp[0]), float(tp[1]))))
    return seq


class AnnulusReference:
    """Independent position cross-check (NOT ground truth).

    Carries an estimate of the target by fitting a per-frame similarity to scene
    anchors in an annulus [inner_px, outer_px] around its OWN running estimate and
    applying it to that estimate — a chained frame-to-frame reference that
    re-seeds fresh anchors when support depletes, so it survives long aggressive
    zooms (a single-init annulus dies in ~100 frames on such footage). Deliberately:
      * seeds anchors EXCLUDING the fixed HUD overlay (dilated) — the white
        cursor / black X / letterbox never become reference evidence;
      * uses DIFFERENT points than the tracker's own support and never reads the
        tracker's committed point (genuinely independent).
    It shares evidence TYPE (LK on nearby scene) and ACCUMULATES frame-to-frame LK
    error, so it is a cross-check with stated, growing uncertainty — never absolute
    truth and never tracker-consensus. For absolute error use synthetic_gt_sequence.
    `update` returns (estimate|None, n_support); None when support is too thin.
    """

    def __init__(self, frame0_bgr: np.ndarray, point: "utils.Point2D", cfg=None,
                 inner_px: float = 45.0, outer_px: float = 100.0,
                 max_anchors: int = 14, roi_px: int = 221,
                 reseed_below: int = 6) -> None:
        cfg = cfg if cfg is not None else _config
        self.cfg = cfg
        self.inner_px, self.outer_px = float(inner_px), float(outer_px)
        self.max_anchors, self.roi_px = int(max_anchors), int(roi_px)
        self.reseed_below = int(reseed_below)
        self.overlay_mask = utils.build_overlay_mask(frame0_bgr, cfg)
        self._dil = (cv2.dilate(self.overlay_mask, np.ones((21, 21), np.uint8))
                     if self.overlay_mask is not None else None)
        self.prev = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
        self.est = np.array([point.x, point.y], np.float32)
        self.overlay_px_in_roi = 0  # set by the first _seed (for the report)
        self.anchors0 = self._seed(self.prev, self.est, record_roi=True)
        self.ok0 = len(self.anchors0) >= 4
        # Segment state: within a segment the fit is BASE->current (no per-frame
        # accumulation, accurate); error accrues only at the few re-seed
        # boundaries. seg_anchors0 = anchor positions at the segment base;
        # cur_pts = their current positions; seg_base_est = the estimate at the
        # segment base. They stay index-aligned as points are FB-culled.
        self.seg_base_est = self.est.copy()
        self.seg_anchors0 = self.anchors0.copy()
        self.cur_pts = self.anchors0.reshape(-1, 1, 2).copy()
        self.reseeds = 0
        self._lk = dict(winSize=tuple(cfg.LK_WIN_SIZE), maxLevel=cfg.LK_MAX_LEVEL,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                  30, 0.03))

    def _seed(self, gray: np.ndarray, center: np.ndarray,
              record_roi: bool = False) -> np.ndarray:
        """GFTT anchors in the annulus around `center`, EXCLUDING the overlay."""
        h, w = gray.shape[:2]
        gx0, gy0, gx1, gy1 = utils.clamp_roi(float(center[0]), float(center[1]),
                                             self.roi_px, w, h)
        m = np.full((gy1 - gy0, gx1 - gx0), 255, np.uint8)
        if self._dil is not None:
            roi_overlay = self._dil[gy0:gy1, gx0:gx1]
            if record_roi:
                self.overlay_px_in_roi = int((roi_overlay > 0).sum())
            m[roi_overlay > 0] = 0
        cs = cv2.goodFeaturesToTrack(gray[gy0:gy1, gx0:gx1], 200, 0.01, 5, mask=m)
        if cs is None or len(cs) == 0:
            return np.empty((0, 2), np.float32)
        cs = cs.reshape(-1, 2).astype(np.float32)
        cs[:, 0] += gx0
        cs[:, 1] += gy0
        d = np.hypot(cs[:, 0] - center[0], cs[:, 1] - center[1])
        return cs[(d >= self.inner_px) & (d <= self.outer_px)][:self.max_anchors]

    def anchors_on_overlay(self) -> int:
        """Count of INITIAL seed anchors on the overlay (must be 0 — a Stage-0
        self-check that the reference never uses HUD pixels)."""
        if self.overlay_mask is None or len(self.anchors0) == 0:
            return 0
        h, w = self.overlay_mask.shape[:2]
        xs = np.clip(self.anchors0[:, 0].astype(int), 0, w - 1)
        ys = np.clip(self.anchors0[:, 1].astype(int), 0, h - 1)
        return int((self.overlay_mask[ys, xs] > 0).sum())

    def update(self, frame_bgr: np.ndarray) -> Tuple[Optional["utils.Point2D"], int]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        est: Optional["utils.Point2D"] = None
        if self.cur_pts is not None and len(self.cur_pts) >= 4:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.cur_pts,
                                                  None, **self._lk)
            bk, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, nxt, None, **self._lk)
            fb = np.linalg.norm((self.cur_pts - bk).reshape(-1, 2), axis=1)
            keep = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb <= 1.0)
            self.seg_anchors0 = self.seg_anchors0[keep]  # stays index-aligned
            self.cur_pts = nxt[keep]
            if len(self.cur_pts) >= 4:
                # BASE(segment)->current similarity, applied to the segment-base
                # estimate: no per-frame accumulation within the segment.
                M, _ = cv2.estimateAffinePartial2D(
                    self.seg_anchors0, self.cur_pts.reshape(-1, 2),
                    method=cv2.RANSAC, ransacReprojThreshold=1.0)
                if M is not None and np.all(np.isfinite(M)):
                    self.est = (M[:, :2] @ self.seg_base_est + M[:, 2]).astype(np.float32)
                    est = utils.Point2D(float(self.est[0]), float(self.est[1]))
        support = 0 if self.cur_pts is None else len(self.cur_pts)
        if support < self.reseed_below:  # start a new segment around the estimate
            fresh = self._seed(gray, self.est)
            if len(fresh) >= 4:
                self.seg_base_est = self.est.copy()
                self.seg_anchors0 = fresh.copy()
                self.cur_pts = fresh.reshape(-1, 1, 2)
                self.reseeds += 1
                support = len(fresh)
        self.prev = gray
        return est, support


class PositionAccuracy:
    """Per-frame target-position error + acceptance classification.

    `tol_px` = "on target" radius. A frame is 'accepted' when it committed a real
    measurement (source=='measure' and state in TRACKING/LOW_CONFIDENCE). The key
    metric is `accepted_wrong`: the tracker accepted a point that is farther than
    tol from the independent reference — the "confident but wrong" class that
    acceptance-only checking cannot see.
    """

    def __init__(self, tol_px: float) -> None:
        self.tol = float(tol_px)
        self.rows: List[dict] = []

    def add(self, t: int, committed: "utils.Point2D",
            reference: Optional["utils.Point2D"], state: str, source: str,
            ref_alive: Optional[int] = None) -> None:
        err = None if reference is None else float(committed.dist(reference))
        accepted = (source == "measure"
                    and state in ("TRACKING", "LOW_CONFIDENCE"))
        on = None if err is None else bool(err <= self.tol)
        self.rows.append({
            "t": t, "err": None if err is None else round(err, 2),
            "state": state, "source": source, "accepted": accepted,
            "on_target": on, "ref_alive": ref_alive,
            "cx": round(committed.x, 1), "cy": round(committed.y, 1),
            "rx": None if reference is None else round(reference.x, 1),
            "ry": None if reference is None else round(reference.y, 1),
        })

    def summary(self) -> dict:
        errs = [r["err"] for r in self.rows if r["err"] is not None]
        es = sorted(errs)

        def pct(q: float):
            return es[min(len(es) - 1, int(q * len(es)))] if es else None

        acc_wrong = [r for r in self.rows
                     if r["accepted"] and r["on_target"] is False]
        acc_right = [r for r in self.rows
                     if r["accepted"] and r["on_target"] is True]
        return {
            "frames": len(self.rows),
            "frames_with_ref": len(errs),
            "tol_px": self.tol,
            "err_p50": round(statistics.median(errs), 1) if errs else None,
            "err_p90": round(pct(0.9), 1) if errs else None,
            "err_max": round(max(errs), 1) if errs else None,
            "accepted": sum(1 for r in self.rows if r["accepted"]),
            "accepted_on_target": len(acc_right),
            "accepted_wrong": len(acc_wrong),
            "accepted_wrong_frames": [r["t"] for r in acc_wrong][:25],
            "rejected_or_predict": sum(1 for r in self.rows if not r["accepted"]),
        }
