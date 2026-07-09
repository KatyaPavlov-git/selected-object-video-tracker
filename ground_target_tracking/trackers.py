"""trackers.py — feature extraction (M5) + the tracking-algorithm seam.

Milestone 5 (implemented here) is **feature extraction only** — it computes and
reports ORB/AKAZE keypoints/descriptors from the initial patch so we can verify we
are extracting good visual features before any tracking exists. ORB is NOT the
tracker; the M6 tracker will use optical-flow corners (goodFeaturesToTrack) with a
grid fallback, because a probe found ~0 ORB keypoints on a flat ground patch.

Implemented:
    class Tracker(ABC):  init(frame, point) / update(frame) -> TrackResult
    class FixedTracker(Tracker)            # M3 baseline (regression guard)
    class OpticalFlowTracker(Tracker)      # Milestone 6 — Lucas-Kanade
    class KalmanWrapper(Tracker)           # Milestone 7 — smoothing + short prediction
    make_tracker(method) -> Tracker        # 'fixed' | 'of' | 'of_kalman'

Trackers are state-free measurers: each update() also reports a [0..1]
`confidence` from its own signals (M8). The TRACKING/LOW_CONFIDENCE/PREDICT/LOST
decision lives in session.TrackingSession, which drives any of these trackers.
Reacquisition (Milestone 9) will be a separate Reacquirer component used by the
session while LOST — not another Tracker subclass.
"""
from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config, preprocessing, utils
except ImportError:  # fallback: run from inside the package folder
    import config
    import preprocessing
    import utils


# --------------------------------------------------------------------------- #
# Feature extraction (Milestone 5) — diagnostic, not the tracker
# --------------------------------------------------------------------------- #
def build_detector(kind: str = "orb", cfg=config):
    """Create a feature detector. 'orb' (default) or 'akaze'."""
    key = (kind or "orb").lower()
    if key == "orb":
        return cv2.ORB_create(nfeatures=cfg.ORB_N_FEATURES)
    if key == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"Unknown detector '{kind}' (expected orb|akaze).")


def detect_features(image: np.ndarray, detector=None, cfg=config,
                    kind: str = "orb") -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Detect keypoints + descriptors on a patch/frame.

    The image is run through the ORB feature pipeline (gray -> CLAHE) first —
    "preprocessing is part of the method". Returns (keypoints, descriptors); the
    descriptors are None when no keypoints are found.
    """
    if detector is None:
        detector = build_detector(kind, cfg)
    gray = preprocessing.orb_pipeline(cfg)(image)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return list(keypoints), descriptors


def feature_report(patch_img: np.ndarray, detector=None, cfg=config,
                   kind: str = "orb") -> dict:
    """Summarize features on the initial patch (count + low-texture flag)."""
    keypoints, _ = detect_features(patch_img, detector, cfg, kind)
    n = len(keypoints)
    return {
        "kind": kind,
        "n_keypoints": n,
        "warn_threshold": cfg.ORB_MIN_KEYPOINTS_WARN,
        "low_texture": n < cfg.ORB_MIN_KEYPOINTS_WARN,
    }


# --------------------------------------------------------------------------- #
# Tracking (Milestone 6) — the same real-world ground point, tracked over time
# --------------------------------------------------------------------------- #
class Tracker(abc.ABC):
    """Common tracker interface. `init` runs once on the init frame with the
    selected point; `update` runs once per subsequent frame and returns a
    TrackResult (the point's pixel coordinates may change as the scene moves)."""

    @abc.abstractmethod
    def init(self, frame_bgr: np.ndarray, point: "utils.Point2D") -> None: ...

    @abc.abstractmethod
    def update(self, frame_bgr: np.ndarray) -> "utils.TrackResult": ...


class FixedTracker(Tracker):
    """Holds the initial point constant — reproduces the pre-M6 fixed overlay.

    Kept as a regression baseline (`--method fixed`); it does no tracking.
    """

    def __init__(self, cfg=config) -> None:
        self.cfg = cfg
        self.point: Optional["utils.Point2D"] = None

    def init(self, frame_bgr, point) -> None:
        self.point = point

    def update(self, frame_bgr) -> "utils.TrackResult":
        return utils.TrackResult(point=self.point, ok=True, n_points=0,
                                 mean_error=0.0, source="fixed", raw_point=self.point)


class OpticalFlowTracker(Tracker):
    """Lucas-Kanade optical-flow tracker for the selected ground point (M6).

    Pipeline: seed corners in the ROI with goodFeaturesToTrack (grid fallback for
    low-texture patches) -> pyramidal LK forward+backward -> keep forward-backward
    consistent survivors -> move the point by the survivors' SIMILARITY TRANSFORM
    (translation + rotation + scale via estimateAffinePartial2D; median-translation
    fallback) -> re-detect when survivors deplete. Uses optical-flow corners (not
    ORB), which work inside small patches where ORB's 31px edge margin yields
    nothing. The similarity update was adopted after measurement: on slow-zoom/
    rotation aerial footage it cuts steady drift ~35% vs pure median translation.
    """

    name = "of"

    def __init__(self, cfg=config) -> None:
        self.cfg = cfg
        self.pipeline = preprocessing.get_pipeline("of", cfg)
        self.patch_size = cfg.PATCH_SIZE
        self.prev_gray: Optional[np.ndarray] = None
        self.pts: Optional[np.ndarray] = None  # (N,1,2) float32
        self.point: Optional["utils.Point2D"] = None
        self.ignore_mask: Optional[np.ndarray] = None  # 255 = never seed/trust here
        self._seed_mask: Optional[np.ndarray] = None   # ignore_mask dilated by LK half-window
        self._fine_overlap: Optional[np.ndarray] = None    # float32: window overlay fraction
        self._coarse_overlap: Optional[np.ndarray] = None  # float32: pyramid-support fraction
        # EXPERIMENTAL (Proposal A) — last regional-path diagnostics, for
        # display/logging ONLY. Never read by the default path. Never fed back
        # into LK / NCC / reference / confidence.
        self.last_regional_diag: Optional[dict] = None

    def set_ignore_mask(self, mask: Optional[np.ndarray]) -> None:
        """Install a fixed-overlay ignore mask (see utils.build_overlay_mask).

        calcOpticalFlowPyrLK cannot be masked, so exclusion is geometric: the
        mask is static, which lets everything expensive be precomputed here —
        a seed mask dilated by the LK half-window (no fresh seed's window may
        straddle the overlay) and per-pixel window-overlap maps used to cull
        survivors whose integration window is overlay-contaminated. The fine
        map matches LK_WIN_SIZE (pyramid level 0); the coarse map approximates
        upper-level support, where the bright crosshair survives pyramid blur.
        """
        self.ignore_mask = mask
        if mask is None:
            self._seed_mask = self._fine_overlap = self._coarse_overlap = None
            return
        d = int(self.cfg.OVERLAY_SEED_DILATE_PX)
        kernel = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
        self._seed_mask = cv2.dilate(mask, kernel) if d > 0 else mask
        on_overlay = (mask > 0).astype(np.float32)
        self._fine_overlap = cv2.boxFilter(on_overlay, -1, tuple(self.cfg.LK_WIN_SIZE),
                                           normalize=True)
        cs = int(self.cfg.OVERLAY_COARSE_SUPPORT_PX)
        self._coarse_overlap = cv2.boxFilter(on_overlay, -1, (cs, cs), normalize=True)

    def init(self, frame_bgr, point) -> None:
        self.point = point
        self.prev_gray = self.pipeline(frame_bgr)
        self.pts = self._seed_corners(self.prev_gray, point)

    def update(self, frame_bgr) -> "utils.TrackResult":
        # EXPERIMENTAL isolation boundary (Proposal A): when the flag is off
        # (default) this is a no-op and the existing path below runs unchanged.
        if getattr(self.cfg, "EXPERIMENTAL_REGIONAL_MOTION", False):
            return self._update_regional(frame_bgr)
        gray = self.pipeline(frame_bgr)
        redetected = False

        if self.pts is None or len(self.pts) == 0:
            self.pts = self._seed_corners(gray, self.point)
            self.prev_gray = gray
            n = 0 if self.pts is None else len(self.pts)
            return utils.TrackResult(point=self.point, ok=False, n_points=n,
                                     mean_error=float("inf"), redetected=True,
                                     raw_point=self.point, confidence=0.0)

        survivors, prev_surv, fb_err = self._lk_track(self.prev_gray, gray, self.pts)
        n_total = len(survivors)
        # Survivors whose LK integration window overlaps the fixed overlay are
        # not scene evidence — the overlay is static, so they pass the FB check
        # with the BEST scores while voting "zero motion". Cull them from the
        # vote, the confidence, and the retained point set; only clean
        # survivors count from here on.
        clean = self._clean_survivors(survivors)
        survivors, prev_surv, fb_err = survivors[clean], prev_surv[clean], fb_err[clean]
        n = len(survivors)
        overlay_frac = 1.0 - (n / n_total) if n_total else 0.0
        ok = n >= self.cfg.LK_MIN_TRACK_POINTS
        if overlay_frac > self.cfg.OVERLAY_MAX_POINT_FRACTION:
            ok = False  # measurement dominated by the overlay: reject outright
        mean_error = float(np.mean(fb_err)) if n > 0 else float("inf")
        confidence = self._confidence(n, mean_error)

        # Commit the motion update to the estimate ONLY when the measurement is
        # reliable. On a low-confidence frame we keep the last committed point
        # (and re-seed around it), so the caller's coast genuinely rejects the
        # bad measurement instead of the drift being baked into state.
        if ok:
            self.point, inliers = self._motion_update(prev_surv, survivors, self.point)
            retained = survivors
            # Prune retention to the RANSAC consensus, but never below the
            # reliability floor — over-pruning would only cause re-seed churn.
            if inliers is not None and inliers.sum() >= self.cfg.LK_MIN_TRACK_POINTS:
                retained = survivors[inliers]
            self.pts = retained.reshape(-1, 1, 2)

        # Re-detect (recentred on the committed point) when live points run low.
        n_live = len(self.pts) if ok else n
        if n_live < self.cfg.LK_REDETECT_BELOW:
            reseed = self._seed_corners(gray, self.point)
            if reseed is not None and len(reseed) > 0:
                self.pts = reseed
                redetected = True

        self.prev_gray = gray
        return utils.TrackResult(point=self.point, ok=ok, n_points=n,
                                 mean_error=mean_error, redetected=redetected,
                                 source="measure", raw_point=self.point,
                                 confidence=confidence)

    # ---- internals ----
    def _motion_update(self, prev_pts: np.ndarray, cur_pts: np.ndarray,
                       point: "utils.Point2D") -> Tuple["utils.Point2D", Optional[np.ndarray]]:
        """Move the tracked point by the survivors' frame-to-frame motion.

        Primary model: a similarity transform (translation + rotation + scale)
        fit robustly with estimateAffinePartial2D and applied to the point —
        pure translation cannot represent zoom/rotation, which biases the
        estimate off the target over time. Fallback: the original median
        translation, when there are too few survivors or the fit fails.

        Returns (new_point, inlier_mask): the RANSAC consensus as a boolean
        array aligned with the inputs (None for the median fallback). The
        threshold is explicit — OpenCV's 3.0px default keeps stationary points
        as inliers whenever scene motion is slower than ~3px/frame.
        """
        p = prev_pts.reshape(-1, 2).astype(np.float32)
        c = cur_pts.reshape(-1, 2).astype(np.float32)
        if len(p) >= self.cfg.LK_AFFINE_MIN_POINTS:
            M, inl = cv2.estimateAffinePartial2D(
                p, c, method=cv2.RANSAC,
                ransacReprojThreshold=float(self.cfg.LK_RANSAC_REPROJ_THRESHOLD))
            if M is not None and np.all(np.isfinite(M)):
                x = float(M[0, 0] * point.x + M[0, 1] * point.y + M[0, 2])
                y = float(M[1, 0] * point.x + M[1, 1] * point.y + M[1, 2])
                inliers = None if inl is None else inl.reshape(-1).astype(bool)
                return utils.Point2D(x, y), inliers
        disp = c - p
        return utils.Point2D(point.x + float(np.median(disp[:, 0])),
                             point.y + float(np.median(disp[:, 1]))), None

    def _clean_survivors(self, pts: np.ndarray) -> np.ndarray:
        """Boolean mask of survivors whose LK window is overlay-clean.

        A survivor is contaminated when the overlay covers more than
        OVERLAY_SURVIVOR_MAX_OVERLAP of its LK_WIN_SIZE window (level 0) or
        more than OVERLAY_COARSE_MAX_OVERLAP of the coarse pyramid-support
        probe — the single-pixel on/off test this replaces let a point 1px
        off the band vote with 90% of its window on the overlay.
        """
        if self._fine_overlap is None or pts is None or len(pts) == 0:
            return np.ones(0 if pts is None else len(pts), bool)
        p = pts.reshape(-1, 2)
        h, w = self._fine_overlap.shape[:2]
        xs = np.clip(p[:, 0].astype(int), 0, w - 1)
        ys = np.clip(p[:, 1].astype(int), 0, h - 1)
        return ((self._fine_overlap[ys, xs] <= self.cfg.OVERLAY_SURVIVOR_MAX_OVERLAP)
                & (self._coarse_overlap[ys, xs] <= self.cfg.OVERLAY_COARSE_MAX_OVERLAP))

    def _overlay_fraction(self, pts: np.ndarray) -> float:
        """Fraction of the given points judged overlay-contaminated (window-aware)."""
        if self.ignore_mask is None or pts is None or len(pts) == 0:
            return 0.0
        return 1.0 - float(np.mean(self._clean_survivors(pts)))

    def _confidence(self, n_survivors: int, mean_error: float) -> float:
        """Measurement quality in [0..1] from this tracker's own signals (M8).

        points_score: how many forward-backward survivors are still alive,
        normalized by CONF_POINTS_NORM. error_score: how far the mean FB error
        is below the acceptance threshold. Both degrade smoothly, so the session
        sees "getting worse" before the hard ok=False cutoff fires.
        """
        points_score = min(1.0, max(0.0, n_survivors / float(self.cfg.CONF_POINTS_NORM)))
        if mean_error == float("inf"):
            error_score = 0.0
        else:
            error_score = min(1.0, max(0.0, 1.0 - mean_error / float(self.cfg.LK_FB_ERROR_THRESHOLD)))
        return points_score * error_score

    def _seed_ignore(self) -> Optional[np.ndarray]:
        """Seed-exclusion mask: the dilated overlay (fresh seeds must keep their
        whole LK window off the overlay), falling back to the raw mask."""
        return self._seed_mask if self._seed_mask is not None else self.ignore_mask

    def _grid_points(self, center, shape) -> np.ndarray:
        h, w = shape[:2]
        x0, y0, x1, y1 = utils.clamp_roi(center.x, center.y, self.patch_size, w, h)
        step = max(2, int(self.cfg.GRID_SEED_STEP))
        xs = np.arange(x0 + step // 2, x1, step)
        ys = np.arange(y0 + step // 2, y1, step)
        if len(xs) == 0 or len(ys) == 0:
            return np.empty((0, 1, 2), np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        ignore = self._seed_ignore()
        if ignore is not None and len(pts):
            keep = ignore[pts[:, 1].astype(int), pts[:, 0].astype(int)] == 0
            pts = pts[keep]
        return pts.reshape(-1, 1, 2)

    def _seed_corners(self, gray, center) -> Optional[np.ndarray]:
        # GFTT computes its corner response over the WHOLE image regardless of
        # the mask (~8.6ms at 1080p), so detect on a padded ROI crop instead
        # and shift the corners back to frame coordinates.
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = utils.clamp_roi(center.x, center.y, self.patch_size, w, h)
        m = int(self.cfg.GFTT_CROP_MARGIN_PX)
        cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
        cx1, cy1 = min(w, x1 + m), min(h, y1 + m)
        crop = gray[cy0:cy1, cx0:cx1]
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 255
        ignore = self._seed_ignore()
        if ignore is not None:  # never seed on (or window-adjacent to) the overlay
            mask[ignore[cy0:cy1, cx0:cx1] > 0] = 0
        corners = cv2.goodFeaturesToTrack(
            crop, self.cfg.GFTT_MAX_CORNERS, self.cfg.GFTT_QUALITY_LEVEL,
            self.cfg.GFTT_MIN_DISTANCE, mask=mask,
        )
        if corners is not None:
            corners = corners.reshape(-1, 1, 2).astype(np.float32)
            corners[:, 0, 0] += cx0
            corners[:, 0, 1] += cy0
            pts = corners
        else:
            pts = np.empty((0, 1, 2), np.float32)
        if len(pts) < self.cfg.GFTT_MIN_SEED:  # low-texture fallback: add a grid
            grid = self._grid_points(center, gray.shape)
            pts = grid if len(pts) == 0 else (
                np.vstack([pts, grid]) if len(grid) else pts)
        return pts if len(pts) else None

    def _lk_track(self, prev_gray, gray, pts):
        lk = dict(winSize=tuple(self.cfg.LK_WIN_SIZE), maxLevel=self.cfg.LK_MAX_LEVEL,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk)
        if nxt is None:
            empty = np.empty((0, 1, 2), np.float32)
            return empty, empty, np.empty((0,), np.float32)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **lk)
        fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        keep = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb <= self.cfg.LK_FB_ERROR_THRESHOLD)
        return nxt[keep], pts[keep], fb[keep]

    # ------------------------------------------------------------------ #
    # EXPERIMENTAL — Proposal A: regional local-motion estimation
    # ------------------------------------------------------------------ #
    # Active ONLY when config.EXPERIMENTAL_REGIONAL_MOTION is True. The exact
    # selected coordinate stays the semantic target; clean scene support is
    # drawn from an enclosing neighborhood; a similarity model is fit to
    # clean-ONLY survivors and EVALUATED AT the selected coordinate. Overlay
    # pixels never seed and never enter the fit (dilated-mask seed exclusion +
    # window-overlap survivor culling). The clean-FRACTION veto is replaced by
    # evidence-quality gates. Self-contained: reuses _lk_track / _clean_survivors
    # / _seed_ignore but does not alter them or the default path.
    def _seed_region(self, gray, center, radius) -> Optional[np.ndarray]:
        """goodFeaturesToTrack over a (2*radius+1) box around `center`, excluding
        the dilated overlay. No grid fallback — regional support wants real
        corners; scarce corners are reported honestly as low support."""
        h, w = gray.shape[:2]
        side = 2 * int(radius) + 1
        x0, y0, x1, y1 = utils.clamp_roi(center.x, center.y, side, w, h)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        mask = np.full(crop.shape[:2], 255, np.uint8)
        ignore = self._seed_ignore()  # dilated overlay mask (None if no overlay)
        if ignore is not None:
            mask[ignore[y0:y1, x0:x1] > 0] = 0
        corners = cv2.goodFeaturesToTrack(
            crop, int(self.cfg.REGIONAL_GFTT_MAX_CORNERS),
            self.cfg.GFTT_QUALITY_LEVEL, self.cfg.GFTT_MIN_DISTANCE, mask=mask)
        if corners is None:
            return None
        corners = corners.reshape(-1, 1, 2).astype(np.float32)
        corners[:, 0, 0] += x0
        corners[:, 0, 1] += y0
        return corners

    def _regional_fit(self, prev_pts, cur_pts, point):
        """Similarity (4-DOF) RANSAC fit on clean support; transport `point`.

        4 DOF (translation+rotation+scale) is the minimal PHYSICALLY-justified
        model for aerial footage with slow zoom/rotation; affine's shear/
        independent-scale DOF have no basis for a rigid ground patch and would
        overfit + amplify lever-arm variance. Returns
        (new_point, inlier_mask, median_inlier_residual_px, M, inlier_thr_px) or
        (None, None, inf, None, thr)."""
        p = prev_pts.reshape(-1, 2).astype(np.float32)
        c = cur_pts.reshape(-1, 2).astype(np.float32)
        # motion-aware inlier threshold (LK error scales with displacement)
        med_motion = (float(np.median(np.linalg.norm(c - p, axis=1)))
                      if len(p) else 0.0)
        thr = max(float(self.cfg.LK_RANSAC_REPROJ_THRESHOLD),
                  float(self.cfg.REGIONAL_RANSAC_MOTION_FRAC) * med_motion)
        if len(p) < self.cfg.LK_AFFINE_MIN_POINTS:
            return None, None, float("inf"), None, thr
        M, inl = cv2.estimateAffinePartial2D(
            p, c, method=cv2.RANSAC, ransacReprojThreshold=thr)
        if M is None or not np.all(np.isfinite(M)):
            return None, None, float("inf"), None, thr
        inl = (inl.reshape(-1).astype(bool) if inl is not None
               else np.ones(len(p), bool))
        if inl.any():
            pred = (M[:, :2] @ p[inl].T).T + M[:, 2]
            resid = float(np.median(np.linalg.norm(c[inl] - pred, axis=1)))
        else:
            resid = float("inf")
        x = float(M[0, 0] * point.x + M[0, 1] * point.y + M[0, 2])
        y = float(M[1, 0] * point.x + M[1, 1] * point.y + M[1, 2])
        return utils.Point2D(x, y), inl, resid, M, thr

    def _update_regional(self, frame_bgr) -> "utils.TrackResult":
        cfg = self.cfg
        gray = self.pipeline(frame_bgr)
        if self.prev_gray is None or self.point is None:
            self.prev_gray = gray
            self.last_regional_diag = {"mode": "regional", "reason": "warmup",
                                       "gate_pass": False}
            return utils.TrackResult(point=self.point, ok=False, n_points=0,
                                     mean_error=float("inf"), source="measure",
                                     raw_point=self.point, confidence=0.0)
        center = self.point
        step = max(1, int(cfg.REGIONAL_RADIUS_STEP_PX))
        radius = int(cfg.REGIONAL_MIN_RADIUS_PX)
        prev_surv = surv = None
        n_seed = 0
        # (1) grow the support radius until the clean floor is met; STOP there.
        while radius <= int(cfg.REGIONAL_MAX_RADIUS_PX):
            region = self._seed_region(self.prev_gray, center, radius)
            n_seed = 0 if region is None else len(region)
            if region is not None and len(region) > 0:
                s_nxt, s_prev, _ = self._lk_track(self.prev_gray, gray, region)
                keep = self._clean_survivors(s_nxt)
                s_nxt, s_prev = s_nxt[keep], s_prev[keep]
                prev_surv, surv = s_prev, s_nxt  # keep best-so-far
                if len(s_nxt) >= int(cfg.REGIONAL_MIN_CLEAN_POINTS):
                    break
            radius += step
        diag = {"mode": "regional", "radius": radius, "n_seed": int(n_seed),
                "n_clean": 0 if surv is None else int(len(surv))}
        # (2) insufficient clean support even at MAX radius -> honest PREDICT.
        if surv is None or len(surv) < int(cfg.REGIONAL_MIN_CLEAN_POINTS):
            self.pts = (surv.reshape(-1, 1, 2)
                        if surv is not None and len(surv) else None)
            self.prev_gray = gray
            diag.update(reason="insufficient_clean_support", gate_pass=False)
            self.last_regional_diag = diag
            return utils.TrackResult(
                point=self.point, ok=False,
                n_points=0 if surv is None else int(len(surv)),
                mean_error=float("inf"), source="measure",
                raw_point=self.point, confidence=0.0)
        # (3) locality / parallax gate: near vs far band must transport the
        #     SELECTED point to the same place, else the support spans
        #     incompatible motion surfaces.
        pv = prev_surv.reshape(-1, 2)
        dist = np.hypot(pv[:, 0] - center.x, pv[:, 1] - center.y)
        near = dist < (radius * 0.5)
        coherent = np.ones(len(pv), bool)
        parallax = False
        band_disagree = None
        min_band = max(4, int(cfg.LK_AFFINE_MIN_POINTS))
        if near.sum() >= min_band and (~near).sum() >= min_band:
            pn, _, _, _, _ = self._regional_fit(prev_surv[near], surv[near], center)
            pf, _, _, _, _ = self._regional_fit(prev_surv[~near], surv[~near], center)
            if pn is not None and pf is not None:
                band_disagree = float(pn.dist(pf))
                if band_disagree > float(cfg.REGIONAL_BAND_AGREEMENT_PX):
                    parallax = True
                    coherent = near  # shrink to the nearest coherent band
        diag["parallax"] = bool(parallax)
        diag["band_disagree"] = (None if band_disagree is None
                                 else round(band_disagree, 3))
        if coherent.sum() < int(cfg.REGIONAL_MIN_CLEAN_POINTS):
            # parallax shrank support below the clean floor -> PREDICT honestly.
            self.pts = surv.reshape(-1, 1, 2)
            self.prev_gray = gray
            diag.update(reason="parallax_insufficient_coherent", gate_pass=False,
                        n_coherent=int(coherent.sum()))
            self.last_regional_diag = diag
            return utils.TrackResult(
                point=self.point, ok=False, n_points=int(coherent.sum()),
                mean_error=float("inf"), source="measure",
                raw_point=self.point, confidence=0.0)
        # (4) final similarity fit on the coherent clean support.
        pp, cc = prev_surv[coherent], surv[coherent]
        new_point, inl, resid, _, thr = self._regional_fit(pp, cc, center)
        n_coh = int(len(pp))
        n_inl = int(inl.sum()) if inl is not None else 0
        inl_ratio = (n_inl / n_coh) if n_coh else 0.0
        cen = pp.reshape(-1, 2).mean(0)
        centroid_off = float(np.hypot(cen[0] - center.x, cen[1] - center.y))
        # residual gate scales with the (motion-aware) inlier threshold so a fast
        # frame's inliers are not held to a slow frame's tolerance.
        resid_gate = max(float(cfg.REGIONAL_MAX_RESIDUAL_PX), 0.5 * thr)
        diag.update(n_coherent=n_coh, n_inliers=n_inl,
                    inlier_ratio=round(inl_ratio, 3),
                    residual=round(resid, 3) if np.isfinite(resid) else None,
                    inlier_thr=round(thr, 2))
        diag["centroid_offset"] = round(centroid_off, 1)
        # (5) evidence-quality gates REPLACE the clean-FRACTION veto here.
        gate = (new_point is not None
                and n_coh >= int(cfg.REGIONAL_MIN_CLEAN_POINTS)
                and n_inl >= int(cfg.REGIONAL_MIN_INLIERS)
                and inl_ratio >= float(cfg.REGIONAL_MIN_INLIER_RATIO)
                and np.isfinite(resid) and resid <= resid_gate)
        diag["gate_pass"] = bool(gate)
        if gate:
            # lever-arm monitor: similarity vs translation-only transport of the
            # point. The band gate already rejects the dangerous (parallax)
            # case; this is reported, not acted on (aerial zoom/rotation makes a
            # non-zero lever legitimate).
            med_t = np.median(cc.reshape(-1, 2) - pp.reshape(-1, 2), axis=0)
            p_trans = utils.Point2D(center.x + float(med_t[0]),
                                    center.y + float(med_t[1]))
            diag["lever_arm_px"] = round(float(new_point.dist(p_trans)), 2)
            self.point = new_point
            points_score = min(1.0, n_inl / float(cfg.CONF_POINTS_NORM))
            resid_score = max(0.0, 1.0 - resid / float(cfg.LK_FB_ERROR_THRESHOLD))
            confidence = points_score * resid_score
            ok = True
        else:
            diag.setdefault("reason", "gate_reject")
            confidence, ok = 0.0, False
        # reseed support in the CURRENT frame around the (possibly updated) point
        # for the next diagnostic / display; next frame re-detects from prev_gray.
        self.pts = self._seed_region(gray, self.point, radius)
        self.prev_gray = gray
        self.last_regional_diag = diag
        return utils.TrackResult(
            point=self.point, ok=ok, n_points=n_coh,
            mean_error=resid if np.isfinite(resid) else float("inf"),
            source="measure", raw_point=self.point, confidence=confidence)


class KalmanWrapper(Tracker):
    """Wrap an inner tracker with a constant-velocity Kalman filter (M7).

    State = (x, y, vx, vy). Each frame: predict, then correct with the inner
    measurement when it is reliable (ok); otherwise coast on the prediction for up
    to KALMAN_MAX_PREDICT_FRAMES frames (source="predict") before reporting ok=False.
    This SMOOTHS jitter and bridges brief gaps; it does NOT detect or reacquire the
    target (that is M8/M9). The returned point is the FILTERED estimate; `raw_point`
    carries the inner (pre-filter) measurement for --show-measurement.
    """

    def __init__(self, inner: Tracker, cfg=config) -> None:
        self.inner = inner
        self.cfg = cfg
        self.kf: Optional[cv2.KalmanFilter] = None
        self.point: Optional["utils.Point2D"] = None
        self._coast = 0
        self._last_conf = 1.0  # last measured confidence; decays while coasting (M8)

    def set_ignore_mask(self, mask) -> None:
        """Forward the fixed-overlay ignore mask to the inner tracker."""
        if hasattr(self.inner, "set_ignore_mask"):
            self.inner.set_ignore_mask(mask)

    def init(self, frame_bgr, point) -> None:
        self.inner.init(frame_bgr, point)
        self.kf = self._build_kf(point)
        self.point = point
        self._coast = 0
        self._last_conf = 1.0

    def _build_kf(self, point) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        # constant-velocity model (dt = 1 frame)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(self.cfg.KALMAN_PROCESS_NOISE)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(self.cfg.KALMAN_MEASUREMENT_NOISE)
        kf.errorCovPost = np.eye(4, dtype=np.float32) * float(self.cfg.KALMAN_INIT_ERROR_COV)
        kf.statePost = np.array([[point.x], [point.y], [0], [0]], np.float32)
        return kf

    def update(self, frame_bgr) -> "utils.TrackResult":
        pred = self.kf.predict()  # advance state by the velocity model
        measurement = self.inner.update(frame_bgr)
        if measurement.ok:
            z = np.array([[np.float32(measurement.point.x)],
                          [np.float32(measurement.point.y)]], np.float32)
            est = self.kf.correct(z)
            self._coast = 0
            source, ok = "measure", True
            confidence = measurement.confidence
            self._last_conf = confidence
        else:
            est = pred  # coast on the prediction
            self._coast += 1
            source = "predict"
            ok = self._coast <= self.cfg.KALMAN_MAX_PREDICT_FRAMES
            # A coasted point is only as trustworthy as the last measurement,
            # discounted for every frame flown blind (M8).
            confidence = self._last_conf * (float(self.cfg.CONF_COAST_DECAY) ** self._coast)
            # State ownership (Stage 1): M7 does NOT re-anchor the inner tracker
            # during a coast. Flying M6's anchor to the constant-velocity
            # prediction manufactured false locks under non-linear motion (radial
            # zoom): the prediction is wrong, the re-seeded corners land on
            # unrelated scene content, and the estimate snaps there. Instead M6
            # owns its own anchor — on a failed measurement it freezes at the last
            # reliable point and re-seeds around it (OpticalFlowTracker.update /
            # _update_regional). We only coast the filtered estimate here.
        self.point = utils.Point2D(float(est[0, 0]), float(est[1, 0]))
        return utils.TrackResult(
            point=self.point, ok=ok, n_points=measurement.n_points,
            mean_error=measurement.mean_error, redetected=measurement.redetected,
            source=source, raw_point=measurement.point, confidence=confidence,
        )


def make_tracker(method: str = "of", cfg=config) -> Tracker:
    """Build a tracker by method name. 'fixed' (M3 baseline) | 'of' (M6) | 'of_kalman' (M7)."""
    key = (method or "of").lower()
    if key == "fixed":
        return FixedTracker(cfg)
    if key == "of":
        return OpticalFlowTracker(cfg)
    if key == "of_kalman":
        return KalmanWrapper(OpticalFlowTracker(cfg), cfg)
    raise ValueError(f"Unknown method '{method}' (expected fixed|of|of_kalman).")
