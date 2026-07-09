"""reacquisition.py — Milestone 9 reacquisition, STAGE 2B (raw scoring only).

This module is the separate `Reacquirer` component the session will use while
LOST (see trackers.py header / AUDIT §2.2). It is NOT a Tracker and is NOT in
make_tracker. Stage 2B implements ONLY the raw-evidence surface — it produces
numbers, never verdicts:

    build_reference(frame, point, overlay_mask=None) -> ReferenceModel
        Build the IMMUTABLE identity reference once, at user-selection time.
    propose(frame, scales=None) -> list[RawCandidate]
        Raw multi-scale template peaks (scores + second-peak margins), NOT
        thresholded, NOT ranked into an acceptance. Deterministic & stateless:
        scales=None sweeps the full configured ladder; the session may pass a
        single scale per tick for round-robin throttling in Stage 2D.
    verify_at(frame, point) -> Optional[float]
        Best raw scale-tolerant identity NCC of the neighbourhood of `point`
        against the immutable context reference, or None when uninformative.

STAGE 2B DOES NOT: apply any accept/reject threshold, peak-margin gate,
GOOD/BAD classification, persistence, confirmation, or descriptor-based
proposing. All of those are Stage 2C decisions. The descriptor reference is
BUILT and stored here (raw data + a capability flag), but descriptor matching
and descriptor-generated candidates are deferred to Stage 2C.

OVERLAY-INDEPENDENT CORE: every function accepts `overlay_mask=None` and
degenerates exactly — validity ≡ all-valid, masked NCC ≡ TM_CCOEFF_NORMED,
keypoint filtering ≡ radius-only. The fixed HUD mask, when the caller supplies
one, only REMOVES pixels from evidence; it never adds or reconstructs any.

The masked-NCC helpers below deliberately REPLICATE the Stage-1 semantics in
session.py (`_masked_ncc` / `_masked_sweep_score` / `_valid_patch` /
`_masked_std`) rather than importing them, so closed Stage-1 code stays
untouched (accepted small duplication; see plan §14).
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config, preprocessing, trackers, utils
except ImportError:  # fallback: run from inside the package folder
    import config
    import preprocessing
    import trackers
    import utils


# --------------------------------------------------------------------------- #
# Raw data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawCandidate:
    """One raw template-proposal peak. NOT an acceptance — carries the raw
    score and the runner-up peak so Stage 2C can make the margin decision."""

    point: "utils.Point2D"     # selected-point coordinate, transported per plan §6
    raw_score: float           # best NCC at this scale (NOT thresholded)
    second_peak_score: float   # 2nd NMS peak at this scale (for the 2C margin gate)
    scale: float               # reference-scale ladder entry that produced it
    cue: str                   # "template" (descriptor cue is Stage 2C)


@dataclass(frozen=True)
class FeatureCandidate:
    """One raw ORB-feature proposal from a SINGLE fitted similarity transform
    (Stage 2B / M9-a). NOT an acceptance: it carries only numbers that
    BFMatcher+ratio and estimateAffinePartial2D genuinely produce. The existence
    of a candidate means "a transform could be fit from these correspondences",
    never "identity confirmed" — every accept/reject/routing decision (min
    inliers, min inlier ratio, residual limit, competing-candidate margin,
    immutable-reference confirmation, persistence, absence, refusal) is Stage 2C.

    Field classes:  D = raw decision evidence a Stage-2C gate consumes ·
    P = provenance ·  X = diagnostic / generation-context (no current gate
    consumes it; a KNN-pair/survivor count is provenance until a gate reads it).
    """

    point: "utils.Point2D"      # D: user-selected pixel (ref.point) transported through the fit M (full-res)
    rotation_deg: float         # D: degrees(atan2(M[1,0], M[0,0])) — estimated in-plane rotation
    scale: float                # D: hypot(M[0,0], M[1,0]) — estimated similarity scale
    n_inliers: int              # D: RANSAC inlier count
    inlier_ratio: float         # D: n_inliers / max(n_good, 1)
    residual: Optional[float]   # D: median reprojection residual of inliers (px); None when n_inliers == 0
    cue: str                    # P: "orb-feature"
    n_query_desc: int           # X: total query (reference) descriptors submitted to the matcher
    n_matches: int              # X: KNN pairs that had two neighbours (the ratio-testable set)
    n_good: int                 # X: ratio-test survivors (denominator of inlier_ratio)


@dataclass(frozen=True)
class LCReference:
    """LAST-CONFIDENT identity reference for the SIFT route (v8 design review).

    Built by build_lc_reference() from a session-owned snapshot taken while
    tracking was still trustworthy (state TRACKING, measured, clean ROI). It
    complements — never replaces — the immutable frame-0 ReferenceModel: the
    frame-0 model stays the identity anchor for the existing ORB/template
    routes; this one exists because the frame-0 view can be unusable (HUD-
    covered selection) or stale (opposite-side return). Same immutability
    pattern as ReferenceModel: frozen dataclass + read-only arrays.
    """

    point: "utils.Point2D"       # the tracked point at snapshot time (full-res)
    kp_xy: np.ndarray            # (N, 2) full-res SIFT keypoint coords (off-HUD)
    descriptors: np.ndarray      # (N, 128) float32 SIFT descriptors
    context: np.ndarray          # gray context window (diagnostics)
    context_offset: Tuple[float, float]  # point offset within the window

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v.flags.writeable = False


@dataclass(frozen=True)
class ReferenceModel:
    """The IMMUTABLE identity reference, written exactly once by
    build_reference() and never mutated (frozen dataclass + read-only arrays,
    enforced by __post_init__ and asserted by a unit test). Everything an
    acceptance later writes is separate; identity is anchored here.
    """

    point: "utils.Point2D"                 # the user-selected point (full-res)
    # Context-scale template (primary cue) + its point offset within the window:
    context: np.ndarray                    # gray context window (REACQ_TEMPLATE_SIZE)
    context_valid: Optional[np.ndarray]    # 255=observed; None when no overlay mask
    context_std: float                     # masked std over observed pixels
    context_offset: Tuple[float, float]    # (point - window_top_left), in reference px
    # Auxiliary 51px patch (probation aid where its own std is valid):
    patch: np.ndarray
    patch_valid: Optional[np.ndarray]
    patch_std: float
    patch_offset: Tuple[float, float]
    # Descriptor reference (BUILT here; matching deferred to Stage 2C):
    descriptors: Optional[np.ndarray]      # (N, D) kept ORB descriptors, or None
    kp_xy: Optional[np.ndarray]            # (N, 2) full-frame keypoint coords
    kp_offsets: Optional[np.ndarray]       # (N, 2) keypoint - point offsets
    # Capability flags (can a cue produce output at all — NOT accept/reject):
    has_context: bool
    has_descriptors: bool

    def __post_init__(self) -> None:
        # Freeze every stored ndarray in place so neither rebinding (frozen
        # dataclass) nor in-place writes can alter the identity reference.
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v.flags.writeable = False


# --------------------------------------------------------------------------- #
# Masked-NCC helpers (replicated Stage-1 semantics; overlay-independent)
# --------------------------------------------------------------------------- #
def _valid_from_mask(overlay_mask: Optional[np.ndarray],
                     x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
    """uint8 validity crop (255 = observed/overlay-free). None when unmasked."""
    if overlay_mask is None:
        return None
    return np.where(overlay_mask[y0:y1, x0:x1] > 0, 0, 255).astype(np.uint8)


def _masked_std(patch: np.ndarray, valid: Optional[np.ndarray]) -> float:
    """Std over observed pixels (all pixels when unmasked). Mirrors
    session._masked_std: overlay pixels are static and must not inflate std."""
    if patch is None or patch.size == 0:
        return 0.0
    if valid is None:
        return float(patch.std())
    vals = patch[valid > 0]
    return float(vals.std()) if vals.size else 0.0


def _masked_ncc(img: np.ndarray, templ: np.ndarray,
                valid: Optional[np.ndarray], min_valid_frac: float) -> Optional[float]:
    """Single-position NCC of two same-size patches over observed pixels.

    Mirrors session._masked_ncc. Returns None (uninformative) when too few
    pixels are observed or the score is numerically undefined. With valid=None
    (no overlay) it is exactly cv2.matchTemplate(TM_CCOEFF_NORMED).
    """
    if img.shape != templ.shape or img.size == 0:
        return None
    if valid is None:
        res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    else:
        if float(np.mean(valid > 0)) < min_valid_frac:
            return None
        res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED, mask=valid)
    sim = float(res[0, 0])
    if not np.isfinite(sim):
        return None
    return max(-1.0, min(1.0, sim))


def _top_two_peaks(response: np.ndarray, radius: int
                   ) -> Tuple[float, Tuple[int, int], float]:
    """Best peak (value, (x, y)) and the second NMS peak value.

    Non-finite cells are treated as -2 (below any real NCC), mirroring the
    Stage-1 sweep. The second peak is taken after suppressing a square
    neighbourhood of side (2*radius+1) around the best — this is what the 2C
    margin gate uses to reject a look-alike (two equal peaks -> ~0 margin).
    """
    resp = np.where(np.isfinite(response), response, -2.0).astype(np.float32)
    _, v1, _, loc1 = cv2.minMaxLoc(resp)
    x1, y1 = int(loc1[0]), int(loc1[1])
    r = max(1, int(radius))
    h, w = resp.shape[:2]
    resp[max(0, y1 - r):min(h, y1 + r + 1),
         max(0, x1 - r):min(w, x1 + r + 1)] = -2.0
    _, v2, _, _ = cv2.minMaxLoc(resp)
    return float(v1), (x1, y1), float(v2)


# --------------------------------------------------------------------------- #
# Reacquirer (Stage 2B: raw scoring only)
# --------------------------------------------------------------------------- #
class Reacquirer:
    """Raw reacquisition evidence over an immutable reference (Stage 2B).

    Holds the reference after build_reference(); propose()/verify_at() are pure
    functions of (reference, frame) — no hidden search state, no thresholds.
    """

    def __init__(self, cfg=config) -> None:
        self.cfg = cfg
        self._template = preprocessing.get_pipeline("template", cfg)
        self.reference: Optional[ReferenceModel] = None
        # SIFT last-confident route: reference + lazily created detector
        # (SIFT work happens ONLY when the session defers a build to recovery
        # entry — never on the healthy tracking path).
        self.lc_reference: Optional["LCReference"] = None
        self._sift = None
        # The fixed HUD overlay is screen-anchored (same pixels every frame),
        # so it is captured once at build_reference and reused for the query
        # side of verify_at. None = overlay-independent (no mask installed).
        self._overlay_mask: Optional[np.ndarray] = None
        # G2-a stripe-sweep accumulator (query side): one stripe per executed
        # recovery tick; reset on completion and on reference replacement.
        self._lc_sweep_state = None

    # ---- reference construction ---------------------------------------- #
    def build_reference(self, frame_bgr: np.ndarray, point: "utils.Point2D",
                        overlay_mask: Optional[np.ndarray] = None) -> ReferenceModel:
        """Build and store the immutable identity reference from the init frame.

        `overlay_mask` (255 = fixed HUD overlay) is supplied by the caller; when
        None the reference is built over all pixels (overlay-independent core).
        """
        h, w = frame_bgr.shape[:2]
        gray = self._template(frame_bgr)

        context, c_valid, c_off = self._window(gray, overlay_mask, point,
                                                int(self.cfg.REACQ_TEMPLATE_SIZE), w, h)
        c_std = _masked_std(context, c_valid)
        has_context = c_std >= float(self.cfg.REACQ_MIN_REF_STD)

        patch, p_valid, p_off = self._window(gray, overlay_mask, point,
                                             int(self.cfg.PATCH_SIZE), w, h)
        p_std = _masked_std(patch, p_valid)

        desc, kp_xy, kp_off, has_desc = self._descriptors(frame_bgr, overlay_mask, point)

        ref = ReferenceModel(
            point=utils.Point2D(float(point.x), float(point.y)),
            context=context, context_valid=c_valid, context_std=c_std, context_offset=c_off,
            patch=patch, patch_valid=p_valid, patch_std=p_std, patch_offset=p_off,
            descriptors=desc, kp_xy=kp_xy, kp_offsets=kp_off,
            has_context=has_context, has_descriptors=has_desc)
        self.reference = ref
        self._overlay_mask = overlay_mask
        return ref

    def _window(self, gray: np.ndarray, overlay_mask: Optional[np.ndarray],
                point: "utils.Point2D", size: int, w: int, h: int
                ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float]]:
        """Crop a border-safe window and store the point's offset WITHIN it
        (never assumed to be the centre — the init window clamps near borders)."""
        x0, y0, x1, y1 = utils.clamp_roi(point.x, point.y, size, w, h)
        win = gray[y0:y1, x0:x1].copy()
        valid = _valid_from_mask(overlay_mask, x0, y0, x1, y1)
        offset = (float(point.x) - x0, float(point.y) - y0)
        return win, valid, offset

    @staticmethod
    def _scaled_size(w: int, h: int, s: float) -> Tuple[int, int, float, float]:
        """Explicit resized dims + the ACTUAL per-axis scale factors derived from
        the real dimensions. Integer resize rounding makes these differ from the
        nominal `s`, so the map-back must use these, never `s` itself."""
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        return new_w, new_h, new_w / float(w), new_h / float(h)

    def _detect_scaled(self, frame_bgr: np.ndarray
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ORB on a consistently downscaled copy of the frame and return
        keypoint coordinates mapped BACK to FULL-RESOLUTION, with descriptors.

        Both the reference build (`_descriptors`) and the query proposal
        (`propose_features`) use this single path, so the descriptor basis
        (detect scale + orb_pipeline preprocessing) is identical on both sides —
        never a mixed full-res/downscaled basis. The coordinate map-back uses the
        actual per-axis factors from the real resized dimensions (robust to odd
        sizes / integer rounding). REACQ_FEAT_DETECT_SCALE == 1.0 degenerates to
        full-frame detection. Returns (xy_full (N,2) float32, descriptors), or
        (None, None) when nothing is detected."""
        h, w = frame_bgr.shape[:2]
        s = float(self.cfg.REACQ_FEAT_DETECT_SCALE)
        new_w, new_h, sx, sy = self._scaled_size(w, h, s)
        img = (frame_bgr if (new_w, new_h) == (w, h)
               else cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA))
        kps, desc = trackers.detect_features(img, kind="orb", cfg=self.cfg)
        if desc is None or not kps:
            return None, None
        xy = np.array([kp.pt for kp in kps], np.float32)
        xy[:, 0] /= sx                        # -> full-resolution coordinates
        xy[:, 1] /= sy
        return xy, desc

    def _offmask_xy(self, xy: np.ndarray, desc: np.ndarray, w: int, h: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """HUD-evidence contract (S4): drop FRAME-SIDE keypoints that sit on
        the overlay mask, so burned-in HUD pixels can never become match
        targets at search time. The reference side has always been filtered
        (`_descriptors` / `build_lc_reference`); this closes the query side.
        No-op when no overlay mask is installed."""
        mask = self._overlay_mask
        if mask is None or xy is None or not len(xy):
            return xy, desc
        ix = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
        iy = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
        keep = mask[iy, ix] == 0
        return xy[keep], desc[keep]

    def _descriptors(self, frame_bgr: np.ndarray, overlay_mask: Optional[np.ndarray],
                     point: "utils.Point2D"
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                Optional[np.ndarray], bool]:
        """ORB on a downscaled frame (E7/V2b: full-res detect exceeds budget),
        keypoint coords mapped back to full-res, kept to off-overlay keypoints
        within REACQ_KP_RADIUS_PX of the point (E1/E3: patch-level descriptors
        are empty; the reference must come from full-frame detection near point).
        Radius filter, overlay lookup, kp_xy and kp_offsets are all full-res."""
        xy, desc = self._detect_scaled(frame_bgr)
        if desc is None or xy is None:
            return None, None, None, False
        r = float(self.cfg.REACQ_KP_RADIUS_PX)
        h, w = frame_bgr.shape[:2]
        keep = []
        for i in range(len(xy)):
            kx, ky = float(xy[i, 0]), float(xy[i, 1])       # full-resolution coords
            if (kx - point.x) ** 2 + (ky - point.y) ** 2 > r * r:
                continue
            if overlay_mask is not None:
                ix = min(max(int(round(kx)), 0), w - 1)
                iy = min(max(int(round(ky)), 0), h - 1)
                if overlay_mask[iy, ix] > 0:
                    continue
            keep.append(i)
        if not keep:
            return None, None, None, False
        idx = np.array(keep, np.int64)
        kp_xy = xy[idx].copy()                               # full-res, frozen by ReferenceModel
        kp_off = kp_xy - np.array([point.x, point.y], np.float32)
        kept_desc = desc[idx].copy()
        has_desc = len(keep) >= int(self.cfg.REACQ_MIN_REF_KP)
        return kept_desc, kp_xy, kp_off, has_desc

    # ---- raw proposal (template cue only in 2B) ------------------------ #
    def propose(self, frame_bgr: np.ndarray,
                scales=None) -> List[RawCandidate]:
        """Raw multi-scale template peaks. One candidate per evaluated scale;
        no thresholding, no cross-scale ranking (both are Stage 2C). Returns []
        when there is no usable context reference (honest: nothing to propose)."""
        ref = self.reference
        if ref is None or not ref.has_context:
            return []
        p = float(self.cfg.REACQ_PYRAMID_SCALE)
        ladder = self.cfg.REACQ_SCALES if scales is None else scales
        gray = self._template(frame_bgr)
        small = cv2.resize(gray, None, fx=p, fy=p, interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        ox, oy = ref.context_offset
        out: List[RawCandidate] = []
        for s in ladder:
            f = float(s) * p                       # template scale in the half-res frame
            tmpl = cv2.resize(ref.context, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            th, tw = tmpl.shape[:2]
            if th < 2 or tw < 2 or th > sh or tw > sw:
                continue                            # scale does not fit this frame
            tmpl_valid = None
            if ref.context_valid is not None:
                tmpl_valid = cv2.resize(ref.context_valid, (tw, th),
                                        interpolation=cv2.INTER_NEAREST)
            response = self._sweep(small, tmpl, tmpl_valid)
            if response is None:
                continue
            radius = int(self.cfg.REACQ_PEAK_NMS_FRAC * min(th, tw))
            v1, (mx, my), v2 = _top_two_peaks(response, radius)
            # Transport coarse top-left -> selected-point coordinate (plan §6).
            pt = utils.Point2D(mx / p + ox * float(s), my / p + oy * float(s))
            out.append(RawCandidate(point=pt, raw_score=v1, second_peak_score=v2,
                                    scale=float(s), cue="template"))
        return out

    def _sweep(self, image: np.ndarray, tmpl: np.ndarray,
               tmpl_valid: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Full matchTemplate response map (TM_CCOEFF_NORMED). Masked when a
        validity mask is present and REACQ_COARSE_MASKED is on; otherwise the
        invalid template pixels are mean-filled (proposal-only reconstruction,
        plan §12) and an unmasked sweep is run (the perf escape hatch)."""
        if tmpl_valid is None or not getattr(self.cfg, "REACQ_COARSE_MASKED", True):
            if tmpl_valid is not None:
                tmpl = tmpl.copy()
                obs = tmpl_valid > 0
                if obs.any():
                    tmpl[~obs] = int(round(float(tmpl[obs].mean())))
            return cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
        return cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED, mask=tmpl_valid)

    # ---- raw feature proposal (ORB matching; M9-a) --------------------- #
    def propose_features(self, frame_bgr: np.ndarray) -> List["FeatureCandidate"]:
        """Raw ORB-feature proposal over the IMMUTABLE reference descriptors.

        Constructs AT MOST ONE similarity transform (BFMatcher NORM_HAMMING ->
        Lowe ratio test -> estimateAffinePartial2D RANSAC) and transports the
        stored selected pixel `ref.point` through it. Rotation/scale invariant
        natively (no template angle sweep). Returns the raw evidence only, NEVER
        a verdict; returns [] when there is nothing to fit:
          * the reference has no usable feature capability (`has_descriptors`),
          * the frame yields too few descriptors to match,
          * too few ratio-survivors to attempt a fit (REACQ_FEAT_MIN_MATCHES —
            a candidate-generation floor, not an acceptance gate), or
          * the RANSAC fit is degenerate / non-finite.

        Pure function of (immutable reference, frame): the reference arrays are
        only read (fancy-indexed copies), never mutated. The ratio cutoff, the
        min-match floor and the RANSAC threshold are candidate-GENERATION (recall)
        parameters — they change which raw geometry is offered, not whether
        identity is accepted (Stage 2C owns every accept/reject/routing decision).
        """
        ref = self.reference
        if ref is None or not ref.has_descriptors or ref.descriptors is None:
            return []
        n_query_desc = int(ref.descriptors.shape[0])
        xy, desc = self._detect_scaled(frame_bgr)          # downscaled detect; coords mapped to full-res
        if desc is None or xy is None or len(xy) < 2:
            return []
        h, w = frame_bgr.shape[:2]
        xy, desc = self._offmask_xy(xy, desc, w, h)        # S4: HUD never a match target
        if len(xy) < 2:
            return []
        # ORB descriptors are binary -> Hamming; knnMatch (k=2) for the ratio test
        # (crossCheck is incompatible with knnMatch and is not used).
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(ref.descriptors, desc, k=2)
        ratio = float(self.cfg.REACQ_FEAT_RATIO)
        src_idx: List[int] = []
        dst_idx: List[int] = []
        n_matches = 0
        for pair in knn:
            if len(pair) < 2:
                continue                       # only 2-neighbour pairs are ratio-testable
            n_matches += 1
            m, n = pair[0], pair[1]
            if m.distance < ratio * n.distance:
                src_idx.append(m.queryIdx)     # index into the immutable reference keypoints
                dst_idx.append(m.trainIdx)     # index into this frame's keypoints
        n_good = len(src_idx)
        if n_good < int(self.cfg.REACQ_FEAT_MIN_MATCHES):
            return []
        # Fancy indexing copies -> the immutable reference arrays are not touched.
        # Both src (reference) and dst (query) are FULL-RESOLUTION coordinates.
        src = ref.kp_xy[np.asarray(src_idx, np.int64)].astype(np.float32)
        dst = xy[np.asarray(dst_idx, np.int64)]
        M, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=float(self.cfg.REACQ_FEAT_RANSAC_THRESH))
        if M is None or not np.all(np.isfinite(M)):
            return []
        # Transport the SELECTED pixel (full-frame) through the full-frame->frame
        # similarity: src/dst are absolute coords, so M maps ref.point directly.
        px = float(M[0, 0] * ref.point.x + M[0, 1] * ref.point.y + M[0, 2])
        py = float(M[1, 0] * ref.point.x + M[1, 1] * ref.point.y + M[1, 2])
        rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        mask = None if inl is None else inl.reshape(-1).astype(bool)
        n_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = float(n_inliers) / float(max(n_good, 1))
        residual = self._inlier_residual(M, src, dst, mask)
        return [FeatureCandidate(
            point=utils.Point2D(px, py), rotation_deg=rotation_deg, scale=scale,
            n_inliers=n_inliers, inlier_ratio=inlier_ratio, residual=residual,
            cue="orb-feature", n_query_desc=n_query_desc,
            n_matches=n_matches, n_good=n_good)]

    # ---- SIFT last-confident route (v8 design review, 2026-07-07) -------- #
    def build_lc_reference(self, frame_bgr: np.ndarray, point: "utils.Point2D",
                           overlay_mask: Optional[np.ndarray] = None
                           ) -> Optional["LCReference"]:
        """Build/replace the SIFT last-confident reference from a trustworthy
        session snapshot (deferred to recovery entry — never on the healthy
        tracking path). Full-resolution SIFT (downscaled variants measured NOT
        viable on the v8 return); keypoints kept off-mask within
        REACQ_SIFT_KP_RADIUS_PX of the point, so HUD pixels never become
        feature evidence. Returns None (no capability) when the snapshot
        yields fewer keypoints than the identity floor could ever accept.
        """
        if self._sift is None:
            self._sift = cv2.SIFT_create()
        self._lc_sweep_state = None             # new reference -> fresh sweep
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or not kps:
            self.lc_reference = None
            return None
        h, w = frame_bgr.shape[:2]
        r = float(self.cfg.REACQ_SIFT_KP_RADIUS_PX)
        keep = []
        for i, kp in enumerate(kps):
            kx, ky = float(kp.pt[0]), float(kp.pt[1])
            if (kx - point.x) ** 2 + (ky - point.y) ** 2 > r * r:
                continue
            if overlay_mask is not None:
                ix = min(max(int(round(kx)), 0), w - 1)
                iy = min(max(int(round(ky)), 0), h - 1)
                if overlay_mask[iy, ix] > 0:
                    continue
            keep.append(i)
        if len(keep) < int(self.cfg.REACQ_SIFT_MIN_INLIERS):
            self.lc_reference = None
            return None
        kp_xy = np.float32([kps[i].pt for i in keep])
        context, _, offset = self._window(gray, overlay_mask, point,
                                          int(self.cfg.REACQ_TEMPLATE_SIZE), w, h)
        self.lc_reference = LCReference(
            point=utils.Point2D(float(point.x), float(point.y)),
            kp_xy=kp_xy,
            descriptors=np.ascontiguousarray(desc[keep]),
            context=context, context_offset=offset)
        return self.lc_reference

    def propose_features_lc(self, frame_bgr: np.ndarray) -> List["FeatureCandidate"]:
        """Raw SIFT-feature proposal over the last-confident reference.

        Mirrors propose_features: constructs AT MOST ONE similarity transform
        (BFMatcher NORM_L2 -> Lowe ratio REACQ_SIFT_RATIO ->
        estimateAffinePartial2D RANSAC) and transports the stored snapshot
        point through it. Full-resolution WHOLE-FRAME detect — the offline /
        diagnostic surface; the session's budgeted G2-a path accumulates the
        same evidence one stripe per tick via lc_sweep_step. Returns raw
        evidence only; [] when there is nothing to fit.
        """
        ref = self.lc_reference
        if ref is None:
            return []
        if self._sift is None:
            self._sift = cv2.SIFT_create()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < 2:
            return []
        xy = np.float32([kp.pt for kp in kps])
        h, w = gray.shape[:2]
        xy, desc = self._offmask_xy(xy, desc, w, h)        # S4: HUD never a match target
        return self._lc_fit(xy, desc)

    def _lc_fit(self, xy: np.ndarray, desc: Optional[np.ndarray]
                ) -> List["FeatureCandidate"]:
        """Shared SIFT-route fit core (G2-a): match query evidence (full-
        resolution coordinates, already HUD-filtered) against the LC reference
        and construct at most one similarity-transform candidate. Used by BOTH
        the whole-frame propose_features_lc and the per-tick stripe sweep, so
        the two paths cannot diverge in matching semantics. Reuses
        REACQ_FEAT_MIN_MATCHES / REACQ_FEAT_RANSAC_THRESH."""
        ref = self.lc_reference
        if ref is None or desc is None or len(xy) < 2:
            return []
        n_query_desc = int(ref.descriptors.shape[0])
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(ref.descriptors, desc, k=2)
        ratio = float(self.cfg.REACQ_SIFT_RATIO)
        src_idx: List[int] = []
        dst_idx: List[int] = []
        n_matches = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            n_matches += 1
            m, n = pair[0], pair[1]
            if m.distance < ratio * n.distance:
                src_idx.append(m.queryIdx)
                dst_idx.append(m.trainIdx)
        n_good = len(src_idx)
        if n_good < int(self.cfg.REACQ_FEAT_MIN_MATCHES):
            return []
        src = ref.kp_xy[np.asarray(src_idx, np.int64)].astype(np.float32)
        dst = xy[np.asarray(dst_idx, np.int64)]
        M, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=float(self.cfg.REACQ_FEAT_RANSAC_THRESH))
        if M is None or not np.all(np.isfinite(M)):
            return []
        px = float(M[0, 0] * ref.point.x + M[0, 1] * ref.point.y + M[0, 2])
        py = float(M[1, 0] * ref.point.x + M[1, 1] * ref.point.y + M[1, 2])
        rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        mask = None if inl is None else inl.reshape(-1).astype(bool)
        n_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = float(n_inliers) / float(max(n_good, 1))
        residual = self._inlier_residual(M, src, dst, mask)
        return [FeatureCandidate(
            point=utils.Point2D(px, py), rotation_deg=rotation_deg, scale=scale,
            n_inliers=n_inliers, inlier_ratio=inlier_ratio, residual=residual,
            cue="sift-lc", n_query_desc=n_query_desc,
            n_matches=n_matches, n_good=n_good)]

    def best_candidate_lc(self, frame_bgr: np.ndarray) -> "ReacqResult":
        """SIFT last-confident route decision (values only, no state). Mirrors
        the feature branch of best_candidate. Whole-frame surface (offline /
        diagnostic / stub-compatible); the session's budgeted path is
        lc_sweep_step."""
        if self.lc_reference is None:
            return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                               "sift:no-reference")
        cands = self.propose_features_lc(frame_bgr)
        return self._lc_result(frame_bgr, cands)

    def _lc_result(self, frame_bgr: np.ndarray,
                   cands: List["FeatureCandidate"]) -> "ReacqResult":
        """Classify SIFT-route candidates into a ReacqResult (shared by the
        whole-frame route and the stripe sweep). confirm_ncc carries the
        fitted-pose NCC against the IMMUTABLE frame-0 context as a DIAGNOSTIC
        (never a gate for this route — measured inverted on the v8 return);
        0.0 when unevaluable so downstream value objects stay total."""
        h, w = frame_bgr.shape[:2]
        if not cands:
            return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                               "sift:no-candidate")
        c = cands[0]
        if not (0 <= c.point.x < w and 0 <= c.point.y < h):
            return ReacqResult(Identity.NEUTRAL, c.point, c.scale, c.cue,
                               None, "sift:point-off-frame", feature=c)
        confirm = self.verify_at_pose(frame_bgr, c.point,
                                      c.rotation_deg, c.scale)
        confirm = 0.0 if confirm is None else confirm
        identity, reason = _classify_sift(c, self.cfg)
        return ReacqResult(identity, c.point, c.scale, c.cue, confirm,
                           reason, feature=c)

    # ---- G2-a budgeted stripe slicing (2026-07-08) ----------------------- #
    @staticmethod
    def _stripe_bounds(height: int, n: int, overlap: int
                       ) -> List[Tuple[int, int, int, int]]:
        """(y0, y1, core0, core1) rows for each of n horizontal stripes.
        Adjacent stripes overlap by `overlap` rows so border keypoints are not
        lost; the core band (dedupe ownership) tiles the height exactly, so
        the union over stripes is detection-complete without duplicates
        (measured identity-lossless on the v8 snapshot: 308 vs 307 disc kp)."""
        n = max(1, int(n))
        hs = max(1, height // n)
        out = []
        for s in range(n):
            core0 = s * hs
            core1 = (s + 1) * hs if s < n - 1 else height
            y0 = max(0, core0 - (overlap if s else 0))
            y1 = min(height, core1 + (overlap if s < n - 1 else 0))
            out.append((y0, y1, core0, core1))
        return out

    def _detect_stripe(self, gray: np.ndarray, bounds: Tuple[int, int, int, int],
                       scale: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """SIFT detect on ONE stripe; returns full-resolution keypoint
        coordinates (core-band owned only) + descriptors. `gray` is already at
        detect scale; `bounds` are rows at that scale; coordinates are mapped
        back to full resolution by 1/scale."""
        if self._sift is None:
            self._sift = cv2.SIFT_create()
        y0, y1, core0, core1 = bounds
        kps, desc = self._sift.detectAndCompute(gray[y0:y1], None)
        if desc is None or not kps:
            return np.zeros((0, 2), np.float32), None
        keep_xy, keep_i = [], []
        inv = 1.0 / float(scale)
        for i, kp in enumerate(kps):
            yy = kp.pt[1] + y0
            if core0 <= yy < core1:                 # core-band dedupe ownership
                keep_xy.append((kp.pt[0] * inv, yy * inv))
                keep_i.append(i)
        if not keep_i:
            return np.zeros((0, 2), np.float32), None
        return (np.float32(keep_xy),
                np.ascontiguousarray(desc[np.asarray(keep_i, np.int64)]))

    def new_lc_builder(self, frame_bgr: np.ndarray, point: "utils.Point2D",
                       overlay_mask: Optional[np.ndarray] = None
                       ) -> "LCStripedBuilder":
        """Incremental striped LC-reference construction (G2-a): the one-time
        ~100-240 ms whole-frame build is sliced into REACQ_SIFT_STRIPES
        full-resolution stripes, one per executed recovery tick. The builder
        assigns self.lc_reference on its final step (same radius / off-mask /
        floor rules as build_lc_reference)."""
        return LCStripedBuilder(self, frame_bgr, point, overlay_mask)

    def lc_sweep_step(self, frame_bgr: np.ndarray) -> Optional["ReacqResult"]:
        """G2-a budgeted SIFT query with HELD-FRAME sweep semantics
        (user-approved correction, 2026-07-08): the query frame is captured
        ONCE at sweep start and ALL stripes of the sweep detect on that same
        held frame — one stripe per executed tick at REACQ_SIFT_DETECT_SCALE
        (query side only; the reference-side downscale was measured
        non-viable), coordinates mapped to full resolution and HUD-filtered
        per stripe. The fit therefore carries whole-frame-consistent geometry
        (no cross-frame smear), delivered one sweep-length late. On the final
        stripe the accumulated evidence goes through the shared _lc_fit core
        and the classified ReacqResult is returned — evaluated against the
        HELD frame, whose geometry the fit describes; None on non-final
        stripes (no identity observation — persistence counts observations).
        """
        if self.lc_reference is None:
            self._lc_sweep_state = None
            return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                               "sift:no-reference")
        n = max(1, int(getattr(self.cfg, "REACQ_SIFT_STRIPES", 8)))
        scale = float(getattr(self.cfg, "REACQ_SIFT_DETECT_SCALE", 0.8))
        overlap = int(getattr(self.cfg, "REACQ_SIFT_STRIPE_OVERLAP_PX", 48))
        if self._lc_sweep_state is None:
            held = frame_bgr.copy()                        # sweep source frame
            gray = cv2.cvtColor(held, cv2.COLOR_BGR2GRAY)
            if abs(scale - 1.0) > 1e-9:
                gray = cv2.resize(gray, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
            self._lc_sweep_state = {"i": 0, "xy": [], "desc": [],
                                    "held": held, "gray": gray}
        gray = self._lc_sweep_state["gray"]
        held = self._lc_sweep_state["held"]
        bounds = self._stripe_bounds(gray.shape[0], n, overlap)
        i = self._lc_sweep_state["i"]
        xy, desc = self._detect_stripe(gray, bounds[i], scale)
        if desc is not None and len(xy):
            h, w = held.shape[:2]
            xy, desc = self._offmask_xy(xy, desc, w, h)    # S4: query side
            if desc is not None and len(xy):
                self._lc_sweep_state["xy"].append(xy)
                self._lc_sweep_state["desc"].append(desc)
        self._lc_sweep_state["i"] = i + 1
        if self._lc_sweep_state["i"] < n:
            return None                                    # sweep in progress
        if self._lc_sweep_state["xy"]:
            all_xy = np.concatenate(self._lc_sweep_state["xy"], axis=0)
            all_desc = np.concatenate(self._lc_sweep_state["desc"], axis=0)
        else:
            all_xy = np.zeros((0, 2), np.float32)
            all_desc = None
        self._lc_sweep_state = None                        # next sweep restarts
        cands = self._lc_fit(all_xy, all_desc)
        return self._lc_result(held, cands)

    @staticmethod
    def _inlier_residual(M: np.ndarray, src: np.ndarray, dst: np.ndarray,
                         mask: Optional[np.ndarray]) -> Optional[float]:
        """Median L2 reprojection residual (px) of the RANSAC inliers, or None
        when there are no inliers. Optional (rather than a NaN sentinel) matches
        the module's uninformative-result contract (`verify_at`, `_masked_ncc`)
        and is JSON-serialisable as null."""
        if mask is None or not bool(mask.any()):
            return None
        s = src[mask]
        d = dst[mask]
        proj = (s @ M[:, :2].T) + M[:, 2]      # apply the 2x3 similarity to inlier sources
        res = np.sqrt(((proj - d) ** 2).sum(axis=1))
        return float(np.median(res))

    # ---- raw point verification ---------------------------------------- #
    def verify_at(self, frame_bgr: np.ndarray,
                  point: "utils.Point2D") -> Optional[float]:
        """Best raw scale-tolerant identity NCC of the neighbourhood of `point`
        against the immutable context reference, or None when uninformative
        (no context cue, point off-frame, or too few observed pixels at every
        scale). Anchored AT the point via the stored offset — an internally
        misaligned region scores low rather than passing. NO threshold."""
        ref = self.reference
        if ref is None or not ref.has_context:
            return None
        h, w = frame_bgr.shape[:2]
        if not (0 <= point.x < w and 0 <= point.y < h):
            return None
        gray = self._template(frame_bgr)
        overlay_mask = self._overlay_mask
        min_frac = float(self.cfg.OVERLAY_SIM_MIN_VALID_FRAC)
        ox, oy = ref.context_offset
        best: Optional[float] = None
        for s in self.cfg.REACQ_SCALES:
            s = float(s)
            if abs(s - 1.0) < 1e-9:
                tmpl, tvalid, off = ref.context, ref.context_valid, (ox, oy)
            else:
                tmpl = cv2.resize(ref.context, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                tvalid = (None if ref.context_valid is None else
                          cv2.resize(ref.context_valid, (tmpl.shape[1], tmpl.shape[0]),
                                     interpolation=cv2.INTER_NEAREST))
                off = (ox * s, oy * s)
            score = self._aligned_ncc(gray, overlay_mask, point, tmpl, tvalid, off, min_frac)
            if score is not None and (best is None or score > best):
                best = score
        return best

    def _aligned_ncc(self, gray: np.ndarray, overlay_mask: Optional[np.ndarray],
                     point: "utils.Point2D", tmpl: np.ndarray,
                     tmpl_valid: Optional[np.ndarray], offset: Tuple[float, float],
                     min_frac: float) -> Optional[float]:
        """Single-alignment masked NCC of the frame window at `point` against
        `tmpl`, positioned so the selected point sits at `offset` within it.
        Crops both to their in-frame overlap (border-safe), unions the validity
        masks, then scores. None when the overlap is too small / under-observed.
        """
        th, tw = tmpl.shape[:2]
        h, w = gray.shape[:2]
        # Ideal (possibly out-of-frame) template top-left in the query frame.
        tlx = int(round(point.x - offset[0]))
        tly = int(round(point.y - offset[1]))
        # Overlap of [tlx, tlx+tw) x [tly, tly+th) with the frame.
        qx0, qy0 = max(0, tlx), max(0, tly)
        qx1, qy1 = min(w, tlx + tw), min(h, tly + th)
        if qx1 - qx0 < self.cfg.MIN_PATCH_SIZE or qy1 - qy0 < self.cfg.MIN_PATCH_SIZE:
            return None
        # Matching crop of the template (aligned by the same offset).
        rx0, ry0 = qx0 - tlx, qy0 - tly
        rx1, ry1 = rx0 + (qx1 - qx0), ry0 + (qy1 - qy0)
        t_crop = tmpl[ry0:ry1, rx0:rx1]
        q_crop = gray[qy0:qy1, qx0:qx1]
        valid = None
        q_valid = _valid_from_mask(overlay_mask, qx0, qy0, qx1, qy1)
        if q_valid is not None or tmpl_valid is not None:
            valid = np.full(t_crop.shape[:2], 255, np.uint8)
            if q_valid is not None:
                valid[q_valid == 0] = 0
            if tmpl_valid is not None:
                valid[tmpl_valid[ry0:ry1, rx0:rx1] == 0] = 0
        return _masked_ncc(q_crop, t_crop, valid, min_frac)

    # ---- raw fitted-pose verification (M9-b confirmation primitive) ----- #
    def verify_at_pose(self, frame_bgr: np.ndarray, point: "utils.Point2D",
                       rotation_deg: float, scale: float) -> Optional[float]:
        """Raw identity NCC of the query neighbourhood of `point` against the
        UNMODIFIED immutable context reference, evaluated at ONE fitted pose.

        The query is re-sampled to upright with a single warpAffine whose
        rotation/scale come verbatim from the candidate's fitted transform —
        no angle or scale search of any kind (this is NOT a rotated-template
        sweep; the pose is already known). Rationale: the upright verify_at
        collapses to noise (~0.03, measured) for valid rotations >= 15deg, so
        it cannot confirm rotated feature candidates; this primitive can.

        Validity: pixels sampled from outside the frame (and, when an overlay
        mask is installed, overlay pixels) are excluded; the reference's own
        context_valid is unioned in. Returns None when uninformative (no
        context cue, off-frame point, degenerate pose, or too few observed
        pixels) — NEVER a verdict; the threshold lives in Stage 2C.
        """
        ref = self.reference
        if ref is None or not ref.has_context:
            return None
        h, w = frame_bgr.shape[:2]
        if not (0 <= point.x < w and 0 <= point.y < h):
            return None
        s = float(scale)
        if not np.isfinite(s) or s <= 1e-6 or not np.isfinite(float(rotation_deg)):
            return None
        th, tw = ref.context.shape[:2]
        ox, oy = ref.context_offset
        a = np.radians(float(rotation_deg))
        ca, sa = float(np.cos(a) * s), float(np.sin(a) * s)
        # dst(x, y) samples the query at: point + s*R(rot) @ ((x, y) - (ox, oy))
        # (WARP_INVERSE_MAP: M maps output/reference coords -> query coords).
        M = np.array([[ca, -sa, point.x - (ca * ox - sa * oy)],
                      [sa,  ca, point.y - (sa * ox + ca * oy)]], np.float32)
        gray = self._template(frame_bgr)
        patch = cv2.warpAffine(gray, M, (tw, th),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # In-frame (and overlay-free) validity, warped through the same pose:
        if self._overlay_mask is None:
            vis = np.full((h, w), 255, np.uint8)
        else:
            vis = np.where(self._overlay_mask > 0, 0, 255).astype(np.uint8)
        valid = cv2.warpAffine(vis, M, (tw, th),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if ref.context_valid is not None:
            valid = np.where(ref.context_valid == 0, 0, valid).astype(np.uint8)
        return _masked_ncc(patch, ref.context, valid,
                           float(self.cfg.OVERLAY_SIM_MIN_VALID_FRAC))

    # ---- Stage 2C: per-frame decision (values only, no state) ----------- #
    def best_candidate(self, frame_bgr: np.ndarray) -> "ReacqResult":
        """Route by REFERENCE CAPABILITY, gather raw evidence, classify.

        Feature-capable reference -> feature route ONLY (a failed feature match
        is never silently re-routed to the template). Template fallback runs
        only for a low-texture reference that still has context capability.
        Neither capability -> NEUTRAL (conservative refusal; in particular
        low-texture + unsupported large rotation/scale is refused, since the
        near-upright template cue cannot support it and no rotated-template
        sweep exists). Pure function of (immutable reference, frame): returns
        values only — no acceptance latch, no tracker/session mutation.
        """
        ref = self.reference
        if ref is None:
            return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                               "no-reference")
        h, w = frame_bgr.shape[:2]
        if ref.has_descriptors:
            cands = self.propose_features(frame_bgr)
            if not cands:
                return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                                   "feature:no-candidate")
            c = cands[0]
            if not (0 <= c.point.x < w and 0 <= c.point.y < h):
                # Cannot be evaluated at an off-frame point -> no observation.
                return ReacqResult(Identity.NEUTRAL, c.point, c.scale, c.cue,
                                   None, "feature:point-off-frame", feature=c)
            confirm = self.verify_at_pose(frame_bgr, c.point,
                                          c.rotation_deg, c.scale)
            if confirm is None:
                return ReacqResult(Identity.NEUTRAL, c.point, c.scale, c.cue,
                                   None, "feature:confirm-unevaluable", feature=c)
            identity, reason = _classify_feature(c, confirm, self.cfg)
            return ReacqResult(identity, c.point, c.scale, c.cue, confirm,
                               reason, feature=c)
        if ref.has_context:
            cands = self.propose(frame_bgr)
            if not cands:
                return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                                   "template:no-candidate")
            best = max(cands, key=lambda t: t.raw_score)
            confirm = self.verify_at(frame_bgr, best.point)
            if confirm is None:
                return ReacqResult(Identity.NEUTRAL, best.point, best.scale,
                                   best.cue, None,
                                   "template:confirm-unevaluable", template=best)
            identity, reason = _classify_template(best, confirm, self.cfg)
            return ReacqResult(identity, best.point, best.scale, best.cue,
                               confirm, reason, template=best)
        return ReacqResult(Identity.NEUTRAL, None, None, None, None,
                           "no-capability")


class LCStripedBuilder:
    """G2-a incremental striped construction of the SIFT last-confident
    reference. One step() = FULL-RESOLUTION SIFT detect on ONE stripe of the
    stored snapshot (overlapped stripes, core-band dedupe — measured
    identity-lossless vs the whole-frame build: 308 vs 307 disc keypoints; the
    reference side is never downscaled, which was measured non-viable). On the
    final stripe the accumulated keypoints pass the SAME radius / off-mask /
    floor rules as build_lc_reference, and the completed LCReference (or None
    on a floor failure) is assigned to the owning Reacquirer. step() returns
    True once the build has finished."""

    def __init__(self, reacq: "Reacquirer", frame_bgr: np.ndarray,
                 point: "utils.Point2D",
                 overlay_mask: Optional[np.ndarray] = None) -> None:
        self._reacq = reacq
        self._gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._point = point
        self._mask = overlay_mask
        n = max(1, int(getattr(reacq.cfg, "REACQ_SIFT_STRIPES", 8)))
        overlap = int(getattr(reacq.cfg, "REACQ_SIFT_STRIPE_OVERLAP_PX", 48))
        self._bounds = Reacquirer._stripe_bounds(self._gray.shape[0], n,
                                                 overlap)
        self._i = 0
        self._xy: List[np.ndarray] = []
        self._desc: List[np.ndarray] = []

    @property
    def done(self) -> bool:
        return self._i >= len(self._bounds)

    def step(self) -> bool:
        if self.done:
            return True
        xy, desc = self._reacq._detect_stripe(self._gray,
                                              self._bounds[self._i], 1.0)
        if desc is not None and len(xy):
            self._xy.append(xy)
            self._desc.append(desc)
        self._i += 1
        if not self.done:
            return False
        self._finish()
        return True

    def _finish(self) -> None:
        reacq, point = self._reacq, self._point
        h, w = self._gray.shape[:2]
        reacq._lc_sweep_state = None            # new reference -> fresh sweep
        if not self._xy:
            reacq.lc_reference = None
            return
        xy = np.concatenate(self._xy, axis=0)
        desc = np.concatenate(self._desc, axis=0)
        r = float(reacq.cfg.REACQ_SIFT_KP_RADIUS_PX)
        keep = []
        for i in range(len(xy)):
            kx, ky = float(xy[i, 0]), float(xy[i, 1])
            if (kx - point.x) ** 2 + (ky - point.y) ** 2 > r * r:
                continue
            if self._mask is not None:
                ix = min(max(int(round(kx)), 0), w - 1)
                iy = min(max(int(round(ky)), 0), h - 1)
                if self._mask[iy, ix] > 0:
                    continue
            keep.append(i)
        if len(keep) < int(reacq.cfg.REACQ_SIFT_MIN_INLIERS):
            reacq.lc_reference = None
            return
        keep_a = np.asarray(keep, np.int64)
        context, _, offset = reacq._window(
            self._gray, self._mask, point,
            int(reacq.cfg.REACQ_TEMPLATE_SIZE), w, h)
        reacq.lc_reference = LCReference(
            point=utils.Point2D(float(point.x), float(point.y)),
            kp_xy=np.ascontiguousarray(xy[keep_a]),
            descriptors=np.ascontiguousarray(desc[keep_a]),
            context=context, context_offset=offset)


# --------------------------------------------------------------------------- #
# Stage 2C decision layer (M9-b): identity verdicts + Option-C persistence.
#
# OWNERSHIP: this layer owns every accept/refuse THRESHOLD and VERDICT over the
# raw Stage-2B evidence. It returns VALUE OBJECTS only: no tracker.init(), no
# session/state mutation, no LOST declaration, no probation — those are Stage
# 2D (M9-c), which also owns the HypothesisTracker INSTANCE lifecycle. The
# HypothesisTracker CLASS is defined and unit-tested here.
# --------------------------------------------------------------------------- #
class Identity(enum.Enum):
    """Per-frame identity verdict over the best available raw candidate.

    MATCH      candidate present and EVERY identity gate passes.
    AMBIGUOUS  a candidate is PRESENT but fails any identity gate (weak
               geometry, out-of-envelope scale, present-but-low confirmation,
               repeated-structure margin). Clears persistence: a prior
               hypothesis must never coast through a present-but-failed
               candidate.
    NEUTRAL    genuinely no usable observation: no candidate, no reference
               capability, or the candidate cannot be evaluated (off-frame
               point / confirmation unavailable). Persistence may freeze
               through a BOUNDED number of these.
    """

    MATCH = "match"
    AMBIGUOUS = "ambiguous"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class ReacqResult:
    """One frame's decision output (a value object; the session acts on it in
    Stage 2D). Evidence is carried in typed frozen fields — the candidate
    dataclasses are themselves frozen, so the result is immutable throughout."""

    identity: Identity
    point: Optional["utils.Point2D"]      # best candidate's location (diagnostic on non-MATCH)
    scale: Optional[float]                # candidate scale estimate
    cue: Optional[str]                    # "orb-feature" | "template" | None
    confirm_ncc: Optional[float]          # immutable-reference confirmation at the point
    reason: str                           # gate provenance of the verdict
    feature: Optional[FeatureCandidate] = None   # raw feature evidence (when that cue ran)
    template: Optional[RawCandidate] = None      # raw template evidence (when that cue ran)


@dataclass(frozen=True)
class AcceptedHypothesis:
    """One-shot acceptance event from HypothesisTracker: emitted EXACTLY ONCE
    when a compatible hypothesis first reaches the persistence floor. A value
    object only — reseeding/probation from it is Stage 2D's decision."""

    point: "utils.Point2D"
    scale: float
    cue: str
    confirm_ncc: float
    streak: int


def _classify_feature(cand: FeatureCandidate, confirm_ncc: float, cfg
                      ) -> Tuple[Identity, str]:
    """Identity gates for a PRESENT, evaluable feature candidate. Any failed
    gate -> AMBIGUOUS (present-but-failed is never NEUTRAL). Caller has already
    handled the no-candidate / unevaluable NEUTRAL cases."""
    if cand.n_inliers < int(cfg.REACQ_MIN_INLIERS):
        return Identity.AMBIGUOUS, "feature:inliers-low"
    if cand.inlier_ratio < float(cfg.REACQ_MIN_INLIER_RATIO):
        return Identity.AMBIGUOUS, "feature:inlier-ratio-low"
    if cand.residual is None or cand.residual > float(cfg.REACQ_MAX_RESIDUAL_PX):
        return Identity.AMBIGUOUS, "feature:residual-high"
    if not (float(cfg.REACQ_FEAT_SCALE_MIN) <= cand.scale
            <= float(cfg.REACQ_FEAT_SCALE_MAX)):
        return Identity.AMBIGUOUS, "feature:scale-out-of-envelope"
    if confirm_ncc < float(cfg.REACQ_CONFIRM_MIN_NCC_FEAT):
        # Present-but-low confirmation is the false-lock / decoy signature
        # (mirrors RECOVER_MIN_SIM semantics) -> refuse, clear persistence.
        return Identity.AMBIGUOUS, "feature:confirm-low"
    return Identity.MATCH, "feature:all-gates-pass"


def _classify_sift(cand: FeatureCandidate, cfg) -> Tuple[Identity, str]:
    """Identity gates for a PRESENT, on-frame SIFT last-confident candidate.

    By the v8 design review (2026-07-07): inlier_ratio and confirm-NCC do NOT
    gate this route — both were measured INVERTED on the true rotated return
    (ratio 0.32–0.42 vs spurious small-sample fits at 0.5+; confirm 0.04–0.16
    for an opposite-side view). Identity rests on the RANSAC geometry here
    plus Option-C persistence and probation downstream. Any failed gate ->
    AMBIGUOUS (present-but-failed is never NEUTRAL)."""
    if cand.n_inliers < int(cfg.REACQ_SIFT_MIN_INLIERS):
        return Identity.AMBIGUOUS, "sift:inliers-low"
    if cand.residual is None or cand.residual > float(cfg.REACQ_MAX_RESIDUAL_PX):
        return Identity.AMBIGUOUS, "sift:residual-high"
    if not (float(cfg.REACQ_FEAT_SCALE_MIN) <= cand.scale
            <= float(cfg.REACQ_FEAT_SCALE_MAX)):
        return Identity.AMBIGUOUS, "sift:scale-out-of-envelope"
    return Identity.MATCH, "sift:all-gates-pass"


def _classify_template(cand: RawCandidate, confirm_ncc: float, cfg
                       ) -> Tuple[Identity, str]:
    """Identity gates for a PRESENT, evaluable template candidate (fallback is
    near-upright by construction; the discrete REACQ_SCALES ladder is its scale
    envelope). The second NMS peak is GENUINE competing evidence, so the margin
    gate refuses repeated structure."""
    if cand.raw_score < float(cfg.REACQ_TEMPLATE_MIN_NCC):
        return Identity.AMBIGUOUS, "template:raw-low"
    if (cand.raw_score - cand.second_peak_score) < float(cfg.REACQ_TEMPLATE_MIN_MARGIN):
        return Identity.AMBIGUOUS, "template:margin-low"
    if confirm_ncc < float(cfg.REACQ_CONFIRM_MIN_NCC_TMPL):
        return Identity.AMBIGUOUS, "template:confirm-low"
    return Identity.MATCH, "template:all-gates-pass"


class HypothesisTracker:
    """Option-C persistence over per-frame ReacqResults (M9-b class; the
    Stage-2D session owns the INSTANCE lifecycle).

    Transition contract (approved):
      MATCH, compatible    -> streak += 1 (same cue, move <= MAX_MOVE_PX,
                              relative scale deviation <= SCALE_TOL)
      MATCH, incompatible  -> REPLACE with a fresh streak=1 hypothesis
                              (unrelated candidates never accumulate)
      AMBIGUOUS            -> clear immediately
      NEUTRAL              -> freeze (streak kept) for at most
                              REACQ_PERSIST_MAX_NEUTRAL consecutive frames;
                              one more clears. Two MATCHes never accumulate
                              across a longer no-evidence interval.

    ONE-SHOT ACCEPTANCE: AcceptedHypothesis is emitted exactly once, when a
    compatible hypothesis FIRST reaches streak == REACQ_PERSIST_N; the
    hypothesis is then latched (continues tracking compatibility) and further
    compatible MATCH frames emit nothing. Replacement, AMBIGUOUS, or an
    exceeded NEUTRAL gap resets the latch. Stage 2D never needs to dedupe.
    """

    def __init__(self, cfg=config) -> None:
        self.cfg = cfg
        self._point: Optional["utils.Point2D"] = None
        self._scale: Optional[float] = None
        self._cue: Optional[str] = None
        self._streak = 0
        self._neutral_gap = 0
        self._accepted = False

    # -- read-only introspection (for tests / 2D diagnostics) -------------- #
    @property
    def streak(self) -> int:
        return self._streak

    @property
    def accepted(self) -> bool:
        return self._accepted

    @property
    def has_hypothesis(self) -> bool:
        return self._point is not None

    def clear(self) -> None:
        self._point = None
        self._scale = None
        self._cue = None
        self._streak = 0
        self._neutral_gap = 0
        self._accepted = False

    def update(self, result: "ReacqResult") -> Optional[AcceptedHypothesis]:
        if result.identity is Identity.AMBIGUOUS:
            self.clear()
            return None
        if result.identity is Identity.NEUTRAL:
            if self._point is not None:
                self._neutral_gap += 1
                if self._neutral_gap > int(self.cfg.REACQ_PERSIST_MAX_NEUTRAL):
                    self.clear()
            return None
        # MATCH — join the current hypothesis or replace it.
        compatible = False
        if self._point is not None and result.point is not None:
            move = result.point.dist(self._point)
            a, b = float(result.scale), float(self._scale)
            scale_dev = abs(a - b) / max(min(abs(a), abs(b)), 1e-6)
            compatible = (result.cue == self._cue
                          and move <= float(self.cfg.REACQ_PERSIST_MAX_MOVE_PX)
                          and scale_dev <= float(self.cfg.REACQ_PERSIST_SCALE_TOL))
        if compatible:
            self._streak += 1
        else:
            self._streak = 1
            self._accepted = False          # a NEW hypothesis re-arms the one-shot
        self._point = result.point
        self._scale = float(result.scale)
        self._cue = result.cue
        self._neutral_gap = 0
        if not self._accepted and self._streak >= int(self.cfg.REACQ_PERSIST_N):
            self._accepted = True           # latch: no re-emission on later frames
            return AcceptedHypothesis(point=result.point, scale=float(result.scale),
                                      cue=str(result.cue),
                                      confirm_ncc=float(result.confirm_ncc),
                                      streak=self._streak)
        return None


# --------------------------------------------------------------------------- #
# Phase 2 — bounded SEARCHING scheduler (descriptor-free template path ONLY).
#
# Replaces the synchronous full-frame five-scale template sweep (measured
# ~125 ms/frame @1080p — the Run A stall) with:
#   * GLOBAL-SCAN: an incremental tiled sweep — per SEARCHING frame at most
#     REACQ_SCAN_UNITS_PER_FRAME atomic units (one unit = one scale over one
#     full-width stripe), preserving rolling full-frame x scale coverage;
#   * a bounded, spatially-distinct pending-candidate list (R2): pre-gate
#     peaks (>= REACQ_TEMPLATE_MIN_NCC) queue for verification, duplicates
#     collapse, stale entries expire, rejected sites cool down;
#   * VERIFY-FIRST (S1): at most ONE candidate per frame is re-searched on the
#     CURRENT live frame around its discovery coordinate (A2: age-scaled,
#     capped window — never a blind verify of a stale coordinate) and then
#     judged by the EXISTING verify_at + _classify_template gates, unchanged.
#
# OWNERSHIP: like HypothesisTracker, the CLASS lives here and the session owns
# the INSTANCE lifecycle (fresh per recovery episode). The Reacquirer itself
# stays stateless except its immutable reference — every piece of mutable
# search state (scan cursor, pending list, cooldowns, calibration) lives here.
#
# TIMING CONTRACT: work QUANTITY per frame is deterministically bounded
# (<= REACQ_SCAN_UNITS_PER_FRAME units + <= 1 re-search + <= 1 verify_at);
# wall-clock time is empirically calibrated (EMA unit cost vs
# REACQ_SCAN_BUDGET_MS) and measured per frame; occasional runtime overruns
# may still occur and are LOGGED (scan_overrun diagnostic), never fatal.
# --------------------------------------------------------------------------- #
@dataclass
class PendingCandidate:
    """One pre-gate scan peak awaiting FRESH live verification (R2 policy).
    Carries only discovery metadata — never an identity verdict."""

    point: "utils.Point2D"   # full-res discovery (or last verified) coordinate
    scale: float             # scale-ladder entry that produced the peak
    ncc: float               # raw peak NCC (ordering key; NOT an acceptance)
    frame_idx: int           # scheduler frame the peak was (re)discovered on
    hot: bool = False        # True = re-enqueued from a MATCH: verified first,
                             # so the persistence streak builds without gaps


class TemplateScanScheduler:
    """Bounded per-frame scan/verify scheduler over a Reacquirer's immutable
    template reference (see the block comment above for the full contract)."""

    def __init__(self, reacquirer: "Reacquirer", cfg=config) -> None:
        self._rq = reacquirer
        self.cfg = cfg
        self._units: Optional[List[Tuple[float, int, int]]] = None  # (scale, y_img0, y_img1)
        self._tmpl: dict = {}                 # scale -> (tmpl, tmpl_valid), resized ONCE
        self._plane: Optional[Tuple[int, int]] = None  # half-res (H, W) the units were built for
        self._cursor = 0                      # next unit in the rolling cycle
        self._cycles = 0                      # completed full coverage cycles
        self._frame_idx = 0                   # executed SEARCHING evaluations seen
        self._pending: List[PendingCandidate] = []
        self._cooldown: List[Tuple[float, float, int]] = []  # (x, y, expires_at_frame)
        self._unit_ms: Optional[float] = None  # EMA-calibrated atomic-unit cost
        self._verify_only_last = False         # instrumented S1 safety fallback
        self._no_progress = 0                  # consecutive frames with 0 scan units

    # ---- per-frame entry point ------------------------------------------ #
    def step(self, frame_bgr: np.ndarray) -> Tuple["ReacqResult", dict]:
        """One SEARCHING frame: VERIFY-FIRST when candidates are pending
        (except immediately after a verify-only fallback frame — the next
        frame MUST advance the global scan), otherwise GLOBAL-SCAN. Returns
        (ReacqResult for HypothesisTracker, instrumentation dict)."""
        t0 = time.perf_counter()
        self._frame_idx += 1
        cfg = self.cfg
        budget = float(cfg.REACQ_SCAN_BUDGET_MS)
        p = float(cfg.REACQ_PYRAMID_SCALE)
        gray = self._rq._template(frame_bgr)
        small = cv2.resize(gray, None, fx=p, fy=p, interpolation=cv2.INTER_AREA)
        if self._units is None or self._plane != small.shape[:2]:
            self._build_units(small.shape[:2])
        self._expire()
        verify_frame = (bool(self._pending) and bool(self._units)
                        and not self._verify_only_last)
        units_done = 0
        verify_only = False
        cooldown_applied = False
        age = None
        win = None
        if verify_frame:
            # -- VERIFY-FIRST: one candidate, fresh live re-search + verify -- #
            self._pending.sort(key=lambda c: (c.hot, c.ncc), reverse=True)
            cand = self._pending.pop(0)       # strongest-first (hot continuation wins)
            age = self._frame_idx - cand.frame_idx
            win = int(cfg.REACQ_RESEARCH_WIN_PX if age <= 1
                      else cfg.REACQ_RESEARCH_WIN_CAP_PX)
            raw = self._research(small, cand.point, win)
            if raw is None:
                # Re-search miss: the peak is gone on the live frame. Drop the
                # candidate; NO identity observation is fabricated (NEUTRAL).
                rr = ReacqResult(Identity.NEUTRAL, None, None, "template", None,
                                 "template-scan:research-miss")
            else:
                confirm = self._rq.verify_at(frame_bgr, raw.point)
                if confirm is None:
                    rr = ReacqResult(Identity.NEUTRAL, raw.point, raw.scale,
                                     raw.cue, None,
                                     "template:confirm-unevaluable", template=raw)
                else:
                    identity, reason = _classify_template(raw, confirm, cfg)
                    rr = ReacqResult(identity, raw.point, raw.scale, raw.cue,
                                     confirm, reason, template=raw)
                    if identity is Identity.MATCH:
                        # LOCAL-CONFIRM: re-enqueue at the VERIFIED location so
                        # the next frame re-verifies it and the persistence
                        # streak builds (observation #1 was fed just now).
                        self._add(raw.point, float(raw.scale),
                                  float(raw.raw_score), hot=True)
                    else:               # AMBIGUOUS -> rejected site cools down
                        self._cooldown.append(
                            (float(raw.point.x), float(raw.point.y),
                             self._frame_idx + int(cfg.REACQ_REJECT_COOLDOWN_FRAMES)))
                        cooldown_applied = True
            # Trailing global unit — only when the calibrated budget permits.
            elapsed = (time.perf_counter() - t0) * 1000.0
            if self._unit_ms is not None and elapsed + self._unit_ms <= budget:
                self._run_unit(small)
                units_done = 1
            else:
                verify_only = True      # S1 safety fallback (instrumented);
                                        # the NEXT frame is forced to scan
        else:
            # -- GLOBAL-SCAN: bounded incremental units --------------------- #
            u_max = max(1, int(cfg.REACQ_SCAN_UNITS_PER_FRAME))
            u_min = max(1, int(cfg.REACQ_SCAN_MIN_UNITS))
            while self._units and units_done < u_max:
                if units_done >= u_min and self._unit_ms is not None:
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    if elapsed + self._unit_ms > budget:
                        break           # min progress done; don't start a unit
                                        # the calibrated budget can't absorb
                self._run_unit(small)
                units_done += 1
            rr = ReacqResult(Identity.NEUTRAL, None, None, "template", None,
                             "template-scan:scanning")
        self._verify_only_last = verify_only
        self._no_progress = self._no_progress + 1 if units_done == 0 else 0
        m9_ms = (time.perf_counter() - t0) * 1000.0
        diag = {
            "scan_mode": "VERIFY" if verify_frame else "SCAN",
            "scan_units": units_done,
            "scan_cycle_units": self._cursor,
            "scan_cycle_total": 0 if self._units is None else len(self._units),
            "scan_cycles_done": self._cycles,
            "scan_pending": len(self._pending),
            "scan_cand_age": age,
            "scan_win_px": win,
            "scan_cooldown_applied": cooldown_applied,
            "scan_m9_ms": round(m9_ms, 2),
            "scan_overrun": bool(m9_ms > budget),
            "scan_verify_only": verify_only,
            "scan_no_progress_streak": self._no_progress,
        }
        return rr, diag

    # ---- atomic unit: one scale over one full-width stripe ---------------- #
    def _build_units(self, plane_hw: Tuple[int, int]) -> None:
        """Stripe grid: for each fitting ladder scale, REACQ_SCAN_STRIPES
        full-width stripes whose RESPONSE rows tile [0, H_resp) exactly — one
        complete cycle evaluates every template position of every fitting
        scale once (rolling full-frame x scale coverage; the image rows of
        adjacent stripes overlap by template-height-1, a pure halo, so no
        position is ever skipped). Templates are resized ONCE and cached."""
        H, W = plane_hw
        cfg = self.cfg
        p = float(cfg.REACQ_PYRAMID_SCALE)
        ref = self._rq.reference
        self._units = []
        self._tmpl = {}
        n = max(1, int(cfg.REACQ_SCAN_STRIPES))
        for s in cfg.REACQ_SCALES:
            s = float(s)
            f = s * p
            tmpl = cv2.resize(ref.context, None, fx=f, fy=f,
                              interpolation=cv2.INTER_AREA)
            th, tw = tmpl.shape[:2]
            if th < 2 or tw < 2 or th > H or tw > W:
                continue                # scale does not fit this frame
            tvalid = None
            if ref.context_valid is not None:
                tvalid = cv2.resize(ref.context_valid, (tw, th),
                                    interpolation=cv2.INTER_NEAREST)
            self._tmpl[s] = (tmpl, tvalid)
            h_resp = H - th + 1
            stripe = max(1, -(-h_resp // n))          # ceil(h_resp / n)
            y = 0
            while y < h_resp:
                y_end = min(y + stripe, h_resp)
                self._units.append((s, y, y_end + th - 1))
                y = y_end
        self._plane = (H, W)
        self._cursor = 0

    def _run_unit(self, small: np.ndarray) -> None:
        """ONE atomic unit: (masked) matchTemplate of one scale over one
        stripe; a peak clearing the pre-gate (REACQ_TEMPLATE_MIN_NCC, the
        existing threshold) enters the pending list. The unit's wall cost
        feeds the EMA calibration used by every budget check."""
        t0 = time.perf_counter()
        scale, y0, y1 = self._units[self._cursor]
        tmpl, tvalid = self._tmpl[scale]
        resp = self._rq._sweep(small[y0:y1], tmpl, tvalid)
        if resp is not None:
            radius = int(self.cfg.REACQ_PEAK_NMS_FRAC * min(tmpl.shape[:2]))
            v1, (mx, my), _ = _top_two_peaks(resp, radius)
            if v1 >= float(self.cfg.REACQ_TEMPLATE_MIN_NCC):
                p = float(self.cfg.REACQ_PYRAMID_SCALE)
                ox, oy = self._rq.reference.context_offset
                pt = utils.Point2D(mx / p + ox * scale,
                                   (y0 + my) / p + oy * scale)
                self._add(pt, scale, float(v1), hot=False)
        self._cursor += 1
        if self._cursor >= len(self._units):
            self._cursor = 0
            self._cycles += 1
        cost = (time.perf_counter() - t0) * 1000.0
        self._unit_ms = (cost if self._unit_ms is None
                         else 0.7 * self._unit_ms + 0.3 * cost)

    # ---- fresh live local re-search (A2 anchor) --------------------------- #
    def _research(self, small: np.ndarray, center: "utils.Point2D",
                  win: int) -> Optional[RawCandidate]:
        """Live re-search around the candidate's discovery coordinate: the
        same multi-scale (masked) matchTemplate, restricted to an age-scaled,
        hard-capped window on the CURRENT frame. Motion is absorbed by the
        window margin (proposes WHERE to look); identity is proven only by
        the verify_at + classification gates afterwards. Returns the best
        window candidate (with the window's second NMS peak for the existing
        margin gate) when it clears the pre-gate; None = miss."""
        cfg = self.cfg
        p = float(cfg.REACQ_PYRAMID_SCALE)
        H, W = small.shape[:2]
        half = win // 2
        x0 = min(max(0, int(round(center.x * p)) - half), max(0, W - win))
        y0 = min(max(0, int(round(center.y * p)) - half), max(0, H - win))
        x1, y1 = min(W, x0 + win), min(H, y0 + win)
        window = small[y0:y1, x0:x1]
        ox, oy = self._rq.reference.context_offset
        best = None
        for s, (tmpl, tvalid) in self._tmpl.items():
            th, tw = tmpl.shape[:2]
            if th > (y1 - y0) or tw > (x1 - x0):
                continue                # this scale does not fit the window
            resp = self._rq._sweep(window, tmpl, tvalid)
            if resp is None:
                continue
            radius = int(cfg.REACQ_PEAK_NMS_FRAC * min(th, tw))
            v1, (mx, my), v2 = _top_two_peaks(resp, radius)
            if best is None or v1 > best[0]:
                best = (v1, v2, x0 + mx, y0 + my, s)
        if best is None or best[0] < float(cfg.REACQ_TEMPLATE_MIN_NCC):
            return None
        v1, v2, gx, gy, s = best
        pt = utils.Point2D(gx / p + ox * s, gy / p + oy * s)
        return RawCandidate(point=pt, raw_score=float(v1),
                            second_peak_score=float(v2), scale=float(s),
                            cue="template")

    # ---- bounded candidate list (R2 policy) ------------------------------- #
    def _add(self, point: "utils.Point2D", scale: float, ncc: float,
             hot: bool = False) -> bool:
        """Enqueue a pre-gate peak: cooldown-blocked sites are refused;
        spatially-nearby duplicates collapse to one entry (stronger kept; a
        hot continuation always refreshes its site's coordinate); when full,
        a spatially distinct newcomer replaces the weakest only if stronger.
        Bounded at REACQ_CAND_QUEUE entries — never an unbounded queue."""
        cfg = self.cfg
        sep = float(cfg.REACQ_CAND_MIN_SEP_PX)
        for (x, y, exp) in self._cooldown:
            if (exp > self._frame_idx
                    and (point.x - x) ** 2 + (point.y - y) ** 2 <= sep * sep):
                return False
        for c in self._pending:
            if c.point.dist(point) <= sep:      # same site
                if hot or ncc >= c.ncc:
                    c.point, c.scale, c.ncc = point, scale, ncc
                    c.frame_idx = self._frame_idx
                    c.hot = c.hot or hot
                return True
        if len(self._pending) < int(cfg.REACQ_CAND_QUEUE):
            self._pending.append(
                PendingCandidate(point, scale, ncc, self._frame_idx, hot))
            return True
        weakest = min(self._pending, key=lambda c: c.ncc)
        if ncc > weakest.ncc:                   # distinct + stronger: replace
            self._pending.remove(weakest)
            self._pending.append(
                PendingCandidate(point, scale, ncc, self._frame_idx, hot))
            return True
        return False

    def _expire(self) -> None:
        """Drop pending candidates past REACQ_CAND_TTL_FRAMES and cooldown
        entries past their expiry (both measured in executed evaluations)."""
        ttl = int(self.cfg.REACQ_CAND_TTL_FRAMES)
        self._pending = [c for c in self._pending
                         if (self._frame_idx - c.frame_idx) <= ttl]
        self._cooldown = [e for e in self._cooldown if e[2] > self._frame_idx]
