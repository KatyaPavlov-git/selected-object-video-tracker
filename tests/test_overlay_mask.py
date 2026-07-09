"""Unit tests for the fixed HUD-overlay handling (mask-complete estimation).

Covers the mask geometry (frame-centered HUD model, independent pillarbox
detection), every estimation stage that consumes the mask (dilated seed
exclusion, window-overlap survivor culling, RANSAC-pruned motion vote),
the masked NCC similarity (scale + translation tolerance preserved), the
adaptive-reference overlay gate, the init-on-overlay policy, and the session
overlay telemetry.
"""
import sys
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")

from ground_target_tracking import utils
from ground_target_tracking.session import TrackingSession, TrackState
from ground_target_tracking.trackers import OpticalFlowTracker, Tracker


def make_cfg(enabled=True):
    return types.SimpleNamespace(
        OVERLAY_MASK_ENABLED=enabled,
        OVERLAY_DIAG_THICKNESS_PX=8,
        OVERLAY_X_SLOPE=4.0 / 3.0,
        OVERLAY_CROSSHAIR_RADIUS_PX=20,
        OVERLAY_LETTERBOX_MAX_INTENSITY=12,
        OVERLAY_LETTERBOX_MARGIN_PX=4,
        OVERLAY_MAX_POINT_FRACTION=0.5,
        # mask-complete estimation
        OVERLAY_SEED_DILATE_PX=10,
        OVERLAY_SURVIVOR_MAX_OVERLAP=0.10,
        OVERLAY_COARSE_SUPPORT_PX=85,
        OVERLAY_COARSE_MAX_OVERLAP=0.50,
        LK_RANSAC_REPROJ_THRESHOLD=1.0,
        GFTT_CROP_MARGIN_PX=16,
        # tracker bits used by the seeding helpers
        PATCH_SIZE=41,
        GRID_SEED_STEP=8,
        GFTT_MAX_CORNERS=50,
        GFTT_QUALITY_LEVEL=0.01,
        GFTT_MIN_DISTANCE=3,
        GFTT_MIN_SEED=8,
        LK_WIN_SIZE=(21, 21),
        LK_MAX_LEVEL=3,
        LK_FB_ERROR_THRESHOLD=1.0,
        LK_MIN_TRACK_POINTS=6,
        LK_REDETECT_BELOW=10,
        LK_AFFINE_MIN_POINTS=3,
        CONF_POINTS_NORM=20,
        USE_GAUSSIAN_BLUR=False,
        USE_CLAHE=False,
        GAMMA=1.0,
        PREPROC_RESIZE_SCALE=1.0,
        GAUSSIAN_KSIZE=5,
        GAUSSIAN_SIGMA=0,
        CLAHE_CLIP_LIMIT=2.0,
        CLAHE_TILE_GRID=(8, 8),
    )


def make_session_cfg():
    """Session config (StubTracker-driven tests) with the overlay enabled."""
    cfg = make_cfg(enabled=True)
    cfg.MIN_PATCH_SIZE = 5
    cfg.LOW_CONFIDENCE_BELOW = 0.40
    cfg.LOST_CONFIDENCE_BELOW = 0.15
    cfg.MIN_PATCH_SIMILARITY = 0.30
    cfg.LOST_AFTER_N_BAD = 4
    cfg.RECOVER_MARGIN = 0.05
    cfg.RECOVER_N = 3
    cfg.RECOVER_MIN_SIM = 0.50
    cfg.EDGE_MARGIN_PX = 8
    cfg.EDGE_PENALTY = 0.5
    cfg.MAX_JUMP_PX = 42
    cfg.JUMP_VETO_MAX_FRAMES = 3
    cfg.REF_UPDATE_MIN_SIM = 0.50
    cfg.REF_UPDATE_EVERY = 3
    cfg.SIM_SCALES = (0.8, 1.0, 1.25)
    cfg.OVERLAY_SIM_MIN_VALID_FRAC = 0.30
    cfg.OVERLAY_SIM_MASKED_MIN_CONTAM = 0.005
    cfg.REF_UPDATE_MAX_OVERLAY_FRAC = 0.35
    cfg.OVERLAY_INIT_MIN_SEEDABLE = 0.10
    return cfg


def diag1_x(cfg, w, h, y):
    """Model diag1 (top-left -> bottom-right) x at row y (frame-centered)."""
    return w / 2.0 + cfg.OVERLAY_X_SLOPE * (y - h / 2.0)


def perp_dist_diag1(cfg, w, h, x, y):
    a = cfg.OVERLAY_X_SLOPE
    return abs(x - diag1_x(cfg, w, h, y)) * np.cos(np.arctan(a))


def letterboxed_frame(w=600, h=400, bar=80, textured=False):
    """Content area flanked by black letterbox bars (mid-gray or noise)."""
    if textured:
        rng = np.random.default_rng(7)
        f = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        f = np.full((h, w, 3), 128, np.uint8)
    f[:, :bar] = 0
    f[:, w - bar:] = 0
    return f


class TestBuildOverlayMask(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(utils.build_overlay_mask(letterboxed_frame(),
                                                   make_cfg(enabled=False)))

    def test_marks_diagonals_crosshair_and_letterbox(self):
        cfg = make_cfg()
        f = letterboxed_frame()
        m = utils.build_overlay_mask(f, cfg)
        self.assertEqual(m.shape, (400, 600))
        self.assertTrue((m[:, :80] == 255).all(), "letterbox bars not masked")
        self.assertEqual(m[200, 300], 255, "frame center (crosshair) not masked")
        # a point on the frame-centered model diagonal
        x = int(round(diag1_x(cfg, 600, 400, 100)))
        self.assertEqual(m[100, x], 255, "model diagonal not masked")
        # an off-overlay content point stays usable
        self.assertEqual(m[60, 400], 0, "clean content pixel wrongly masked")

    def test_no_letterbox_falls_back_to_full_frame(self):
        cfg = make_cfg()
        f = np.full((400, 600, 3), 128, np.uint8)
        m = utils.build_overlay_mask(f, cfg)
        x0 = int(round(diag1_x(cfg, 600, 400, 0)))
        self.assertEqual(m[0, x0], 255, "diagonal top endpoint not masked")
        self.assertEqual(m[0, 0], 0,
                         "frame corner masked: X must be frame-centered, "
                         "not corner-to-corner")
        self.assertEqual(m[200, 300], 255)    # crosshair
        self.assertEqual(m[40, 450], 0)       # clean pixel


class TestHudGeometryModel(unittest.TestCase):
    """The HUD X/crosshair anchor to the FRAME center, independent of
    pillarboxes (the geometry defect found on the official sample video)."""

    def make_v8_like(self):
        """1920x1080 mid-gray content with genuine asymmetric pillarboxes."""
        f = np.full((1080, 1920, 3), 128, np.uint8)
        f[:, :239] = 0
        f[:, 1773:] = 0
        return f

    def test_pillarboxes_masked_but_hud_frame_centered(self):
        cfg = make_cfg()
        cfg.OVERLAY_CROSSHAIR_RADIUS_PX = 45
        m = utils.build_overlay_mask(self.make_v8_like(), cfg)
        self.assertTrue((m[:, :239] == 255).all(), "left pillarbox not masked")
        self.assertTrue((m[:, 1773:] == 255).all(), "right pillarbox not masked")
        # crosshair disc centered at the FRAME center (960,540), not the
        # content-box center (1006,540): x=920 is inside the frame-centered
        # disc and outside the content-centered one; x=1045 is the reverse.
        self.assertEqual(m[540, 920], 255, "disc not centered on the frame")
        self.assertEqual(m[540, 1045], 0, "disc still centered on content box")

    def test_model_diagonal_covers_real_line_locations(self):
        cfg = make_cfg()
        cfg.OVERLAY_CROSSHAIR_RADIUS_PX = 45
        m = utils.build_overlay_mask(self.make_v8_like(), cfg)
        for y in (100, 300, 700, 900, 1000):
            x1 = int(round(diag1_x(cfg, 1920, 1080, y)))
            x2 = int(round(1920 / 2.0 - cfg.OVERLAY_X_SLOPE * (y - 540)))
            self.assertEqual(m[y, x1], 255, f"diag1 uncovered at y={y}")
            self.assertEqual(m[y, x2], 255, f"diag2 uncovered at y={y}")
        # the measured real-line location from the geometry diagnosis
        # (test datum, not an implementation rule)
        self.assertEqual(m[736, 1218], 255,
                         "known real-line pixel (1218,736) not covered")

    def test_geometry_scales_with_resolution(self):
        cfg = make_cfg()
        for (w, h) in ((1920, 1080), (1280, 720)):
            f = np.full((h, w, 3), 128, np.uint8)
            m = utils.build_overlay_mask(f, cfg)
            self.assertEqual(m[h // 2, w // 2], 255)
            for y in (h // 6, 5 * h // 6):
                x = int(round(diag1_x(cfg, w, h, y)))
                self.assertEqual(m[y, x], 255, f"{w}x{h}: diag1 uncovered y={y}")

    def test_mask_area_decomposition_no_scene_inflation(self):
        cfg = make_cfg()
        cfg.OVERLAY_CROSSHAIR_RADIUS_PX = 45
        full = utils.build_overlay_mask(self.make_v8_like(), cfg)
        hud_only = utils.build_overlay_mask(
            np.full((1080, 1920, 3), 128, np.uint8), cfg)
        stripes_frac = (239 + cfg.OVERLAY_LETTERBOX_MARGIN_PX
                        + (1920 - 1773) + cfg.OVERLAY_LETTERBOX_MARGIN_PX) / 1920.0
        hud_frac = float((hud_only > 0).mean())
        full_frac = float((full > 0).mean())
        self.assertLess(hud_frac, 0.04, "HUD-only mask inflates over scene")
        self.assertGreater(hud_frac, 0.015)
        self.assertLess(full_frac, stripes_frac + hud_frac + 0.005,
                        "mask exceeds pillarboxes + HUD geometry")


def hud_scene(w=1920, h=1080, cfg=None, seed=11, line_value=15, blur=1.5):
    """Textured scene with a thin dark HUD X drawn at the MODEL geometry."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 256, (h, w), dtype=np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    s = cfg.OVERLAY_X_SLOPE
    cx, cy = w / 2.0, h / 2.0
    cv2.line(g, (int(round(cx - s * cy)), 0),
             (int(round(cx + s * (h - 1 - cy))), h - 1), line_value, 3)
    cv2.line(g, (int(round(cx + s * cy)), 0),
             (int(round(cx - s * (h - 1 - cy))), h - 1), line_value, 3)
    cv2.circle(g, (int(cx), int(cy)), 30, 255, 2)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class TestSeedsAndWindowsVsRealLine(unittest.TestCase):
    """Seeding and culling must exclude the REAL HUD line location."""

    def setUp(self):
        self.cfg = make_cfg()
        self.cfg.OVERLAY_CROSSHAIR_RADIUS_PX = 45
        self.frame = hud_scene(cfg=self.cfg)
        self.mask = utils.build_overlay_mask(self.frame, self.cfg)
        self.tracker = OpticalFlowTracker(self.cfg)
        self.tracker.set_ignore_mask(self.mask)
        # a point ON the model diagonal, away from the crosshair
        self.on_line = utils.Point2D(
            float(round(diag1_x(self.cfg, 1920, 1080, 736))), 736.0)

    def test_seeds_avoid_real_line(self):
        gray = self.tracker.pipeline(self.frame)
        pts = self.tracker._seed_corners(gray, self.on_line)
        self.assertIsNotNone(pts, "on-line init with clean support must seed")
        for x, y in pts.reshape(-1, 2):
            d = perp_dist_diag1(self.cfg, 1920, 1080, x, y)
            self.assertGreater(d, 6.0,
                               f"seed ({x:.0f},{y:.0f}) sits on the real line")

    def test_online_survivors_culled_offline_kept(self):
        on = [[self.on_line.x, self.on_line.y]]
        off = [[self.on_line.x + 60.0, self.on_line.y]]  # ~36px perp away
        pts = np.array(on + off, np.float32).reshape(-1, 1, 2)
        clean = self.tracker._clean_survivors(pts)
        self.assertFalse(clean[0], "on-line survivor not culled")
        self.assertTrue(clean[1], "clean survivor near the line over-culled")


class TestSurvivorCulling(unittest.TestCase):
    """Window-overlap culling against a simple vertical band (trivial geometry)."""

    def setUp(self):
        self.cfg = make_cfg()
        self.mask = np.zeros((300, 400), np.uint8)
        self.mask[:, 190:210] = 255  # 20px-wide static band
        self.tracker = OpticalFlowTracker(self.cfg)
        self.tracker.set_ignore_mask(self.mask)

    def _clean(self, x, y=150.0):
        pts = np.array([[x, y]], np.float32).reshape(-1, 1, 2)
        return bool(self.tracker._clean_survivors(pts)[0])

    def test_window_overlap_geometry(self):
        self.assertTrue(self._clean(150.0), "far point wrongly culled")
        self.assertFalse(self._clean(200.0), "on-band point kept")
        # off the band but with >10% of the 21x21 window on it
        self.assertFalse(self._clean(185.0), "window-straddling point kept")
        # window clear at level 0, coarse overlap under the 0.5 threshold
        self.assertTrue(self._clean(170.0), "near point over-culled")

    def test_overlay_fraction_is_window_aware(self):
        on = [[200.0, 150.0]] * 3
        off = [[100.0, 150.0]]
        pts = np.array(on + off, np.float32).reshape(-1, 1, 2)
        self.assertAlmostEqual(self.tracker._overlay_fraction(pts), 0.75)
        self.assertEqual(self.tracker._overlay_fraction(None), 0.0)
        self.tracker.set_ignore_mask(None)
        self.assertEqual(self.tracker._overlay_fraction(pts), 0.0)

    def test_update_tracks_true_motion_next_to_static_band(self):
        """Scene shifts +2px/frame under a static dark band; the tracked point
        must follow the scene, not the band, and retained points stay clean."""
        rng = np.random.default_rng(11)
        base = rng.integers(0, 256, (300, 400), dtype=np.uint8)

        def frame_at(t):
            scene = np.roll(base, 2 * t, axis=1)
            scene = scene.copy()
            scene[:, 190:210] = 20  # static overlay band, drawn every frame
            return cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

        self.tracker.init(frame_at(0), utils.Point2D(150.0, 150.0))
        for t in range(1, 16):
            out = self.tracker.update(frame_at(t))
        gt_x = 150.0 + 2 * 15  # = 180: ROI now abuts the band
        self.assertLess(abs(out.point.x - gt_x), 3.0,
                        f"tracked x={out.point.x:.1f}, expected ~{gt_x}")
        self.assertLess(abs(out.point.y - 150.0), 3.0)
        clean = self.tracker._clean_survivors(self.tracker.pts)
        self.assertTrue(clean.all(), "contaminated survivors retained in self.pts")


class TestMotionUpdateRansac(unittest.TestCase):
    def test_static_contaminants_excluded_from_inliers(self):
        cfg = make_cfg()
        tracker = OpticalFlowTracker(cfg)
        rng = np.random.default_rng(3)
        movers = rng.uniform(20, 280, (30, 2)).astype(np.float32)
        static = rng.uniform(20, 280, (10, 2)).astype(np.float32)
        prev = np.vstack([movers, static]).reshape(-1, 1, 2)
        cur = np.vstack([movers + (3.0, 2.0), static]).reshape(-1, 1, 2)
        point, inl = tracker._motion_update(prev, cur, utils.Point2D(50.0, 50.0))
        self.assertIsNotNone(inl, "RANSAC inlier mask discarded")
        self.assertTrue(inl[:30].all(), "true movers rejected as outliers")
        self.assertFalse(inl[30:].any(),
                         "zero-motion contaminants accepted as inliers")
        self.assertLess(abs(point.x - 53.0), 0.2)
        self.assertLess(abs(point.y - 52.0), 0.2)


class TestKalmanDoesNotReanchorInnerOnCoast(unittest.TestCase):
    def test_inner_anchor_frozen_during_coast(self):
        # Stage 1 state-ownership: M7 must NOT re-anchor M6 onto the Kalman
        # prediction during a coast. The inner tracker owns its anchor and
        # freezes it (re-seeding around its own last point). So inner.init is
        # called exactly ONCE — at KalmanWrapper.init — and never again while
        # coasting, regardless of the predict budget.
        from ground_target_tracking.trackers import KalmanWrapper

        class DeadInner(Tracker):
            def __init__(self):
                self.init_points = []

            def init(self, frame_bgr, point):
                self.init_points.append((point.x, point.y))

            def update(self, frame_bgr):
                return utils.TrackResult(point=utils.Point2D(50.0, 50.0),
                                         ok=False, n_points=0,
                                         mean_error=float("inf"),
                                         confidence=0.0, source="measure")

        cfg = make_cfg()
        cfg.KALMAN_PROCESS_NOISE = 0.1
        cfg.KALMAN_MEASUREMENT_NOISE = 0.1
        cfg.KALMAN_INIT_ERROR_COV = 1.0
        cfg.KALMAN_MAX_PREDICT_FRAMES = 5
        cfg.CONF_COAST_DECAY = 0.85
        inner = DeadInner()
        kw = KalmanWrapper(inner, cfg)
        frame = np.zeros((100, 100, 3), np.uint8)
        kw.init(frame, utils.Point2D(50.0, 50.0))
        for _ in range(cfg.KALMAN_MAX_PREDICT_FRAMES + 3):
            kw.update(frame)
        self.assertEqual(len(inner.init_points), 1,
                         "M7 must not re-anchor the inner tracker during a coast")
        self.assertEqual(inner.init_points[0], (50.0, 50.0),
                         "the only init is the original anchor from KalmanWrapper.init")


class StubTracker(Tracker):
    def __init__(self, results):
        self.results = list(results)
        self.i = 0

    def init(self, frame_bgr, point):
        pass

    def update(self, frame_bgr):
        r = self.results[min(self.i, len(self.results) - 1)]
        self.i += 1
        return r


def res(x, y, ok=True, conf=0.9):
    return utils.TrackResult(point=utils.Point2D(x, y), ok=ok, n_points=20,
                             mean_error=0.1, source="measure", confidence=conf)


class SessionOverlayBase(unittest.TestCase):
    """Shared geometry: 600x400 textured letterboxed frame, model-geometry
    mask (thickness 8, disc r20, dilate 10, frame center (300,200)).

      clean_p (300, 40) — >120px from both diagonals and the crosshair
      diag_p  (250,170) — ~6px perp from diag1: ROI overlaps the band
                          (coverage between the mode threshold and the
                          reference gate), the on-line dwell analog
      cross_p (300,200) — crosshair center: the disc dominates the ROI
    """

    def setUp(self):
        self.cfg = make_session_cfg()
        self.frame = letterboxed_frame(textured=True)
        self.clean_p = utils.Point2D(300.0, 40.0)
        self.diag_p = utils.Point2D(250.0, 170.0)
        self.cross_p = utils.Point2D(300.0, 200.0)

    def make_session(self, point, results=None):
        sess = TrackingSession(StubTracker(results or [res(point.x, point.y)]),
                               self.cfg)
        sess.init(self.frame, point)
        return sess

    def static_overlay_transfer(self, target_frame, src_frame, sess):
        """Stamp the (static) overlay-region pixels back after a scene warp."""
        out = target_frame.copy()
        out[sess._overlay > 0] = src_frame[sess._overlay > 0]
        return out


class TestSessionMaskedAppearance(SessionOverlayBase):
    def test_precondition_coverages(self):
        sess = self.make_session(self.clean_p)
        cov_diag = sess._overlay_coverage(self.diag_p, self.frame.shape)
        cov_cross = sess._overlay_coverage(self.cross_p, self.frame.shape)
        self.assertGreater(cov_diag, self.cfg.OVERLAY_SIM_MASKED_MIN_CONTAM)
        self.assertLess(cov_diag, self.cfg.REF_UPDATE_MAX_OVERLAY_FRAC)
        self.assertGreater(cov_cross, self.cfg.REF_UPDATE_MAX_OVERLAY_FRAC)
        self.assertEqual(sess._overlay_coverage(self.clean_p, self.frame.shape), 0.0)

    def test_masked_similarity_ignores_overlay_pixels(self):
        sess = self.make_session(self.diag_p)
        corrupted = self.frame.copy()
        corrupted[sess._overlay > 0] = np.random.default_rng(5).integers(
            0, 256, (int((sess._overlay > 0).sum()), 3), dtype=np.uint8)
        sim = sess._patch_similarity(corrupted, self.diag_p)
        self.assertIsNotNone(sim)
        self.assertGreater(sim, 0.95,
                           "overlay-pixel corruption leaked into masked NCC")
        # sanity: corrupting VALID pixels must depress similarity
        corrupted2 = self.frame.copy()
        corrupted2[sess._overlay == 0] = 128
        sim2 = sess._patch_similarity(corrupted2, self.diag_p)
        self.assertTrue(sim2 is None or sim2 < 0.5,
                        "scene-pixel corruption not detected")

    def _plain_sim(self, frame, point):
        """Similarity with the overlay disabled (control path)."""
        cfg_off = make_session_cfg()
        cfg_off.OVERLAY_MASK_ENABLED = False
        sess_off = TrackingSession(StubTracker([res(point.x, point.y)]), cfg_off)
        sess_off.init(self.frame, point)
        return sess_off._patch_similarity(frame, point)

    def test_tiny_contamination_keeps_unmasked_scoring(self):
        """A speck of contamination below the mode threshold must score
        byte-identically to the unmasked path (no mode cliff), and just above
        the threshold the masked score must stay close, not collapse."""
        sess = self.make_session(self.clean_p)
        x0, y0, _, _ = utils.clamp_roi(self.clean_p.x, self.clean_p.y,
                                       self.cfg.PATCH_SIZE, 600, 400)
        # inject a 2x2 speck (4/1681 = 0.0024 <= 0.005) into the live mask
        sess._overlay[y0:y0 + 2, x0:x0 + 2] = 255
        sim_speck = sess._patch_similarity(self.frame, self.clean_p)
        sim_plain = self._plain_sim(self.frame, self.clean_p)
        self.assertAlmostEqual(sim_speck, sim_plain, places=6,
                               msg="tiny contamination flipped the scoring mode")
        # 5x5 = 25/1681 = 0.015 > threshold: masked mode on, score comparable
        sess._overlay[y0:y0 + 5, x0:x0 + 5] = 255
        sim_masked = sess._patch_similarity(self.frame, self.clean_p)
        self.assertIsNotNone(sim_masked)
        self.assertLess(abs(sim_masked - sim_plain), 0.05,
                        "masked mode meaningfully diverges on identical content")

    def test_masked_path_preserves_scale_tolerance(self):
        sess = self.make_session(self.diag_p)
        h, w = self.frame.shape[:2]
        M = cv2.getRotationMatrix2D((self.diag_p.x, self.diag_p.y), 0, 1.25)
        zoomed = cv2.warpAffine(self.frame, M, (w, h))
        zoomed = self.static_overlay_transfer(zoomed, self.frame, sess)
        sim = sess._patch_similarity(zoomed, self.diag_p)
        self.assertIsNotNone(sim)
        self.assertGreater(sim, 0.5,
                           "masked path lost the scale sweep (1.25x zoom)")

    def test_masked_path_preserves_translation_tolerance(self):
        """The correct oracle is the CLEAN-SCENE control: the masked score on
        a shifted scene with a static overlay band must match the unmasked
        score on the same shift WITHOUT the static band (pure scene). Naive
        parity against the unmasked score on the static-band frame would be
        wrong — there the band correlates with itself and inflates the score,
        which is precisely the artifact masking removes."""
        sess = self.make_session(self.diag_p)
        shifted_clean = np.roll(self.frame, (3, 4), axis=(0, 1))
        shifted_static = self.static_overlay_transfer(shifted_clean, self.frame,
                                                      sess)
        sim_masked = sess._patch_similarity(shifted_static, self.diag_p)
        sim_control = self._plain_sim(shifted_clean, self.diag_p)
        sim_inflated = self._plain_sim(shifted_static, self.diag_p)
        self.assertIsNotNone(sim_masked)
        self.assertIsNotNone(sim_control)
        self.assertLess(abs(sim_masked - sim_control), 0.05,
                        f"masked score {sim_masked:.3f} deviates from the "
                        f"clean-scene control {sim_control:.3f}")
        self.assertLessEqual(sim_masked, sim_inflated + 0.02,
                             "masked path enjoys static-band self-correlation")

    def test_similarity_neutral_when_overlay_dominates(self):
        sess = self.make_session(self.clean_p)
        self.assertIsNone(sess._patch_similarity(self.frame, self.cross_p))

    def test_reference_updates_allowed_under_moderate_contamination(self):
        """A target ON the diagonal (~20% coverage) must keep adapting."""
        sess = self.make_session(self.diag_p)
        ref0 = sess._ref_patch.copy()
        # legitimately evolved appearance: brightness shift (NCC-invariant),
        # overlay pixels stamped back static
        evolved = cv2.convertScaleAbs(self.frame, alpha=1.0, beta=15)
        evolved = self.static_overlay_transfer(evolved, self.frame, sess)
        for _ in range(self.cfg.REF_UPDATE_EVERY):
            sess._maybe_update_reference(evolved, similarity=0.9,
                                         edge_near=False, jump_vetoed=False,
                                         measured_ok=True, tracker_conf=0.9)
        self.assertFalse(np.array_equal(sess._ref_patch, ref0),
                         "moderate contamination wrongly froze the reference")
        self.assertIsNotNone(sess._ref_valid,
                             "contaminated snapshot must store its validity")

    def test_reference_never_dominated_by_overlay(self):
        sess = self.make_session(self.clean_p)
        ref0 = sess._ref_patch.copy()
        sess.point = self.cross_p  # overlay-heavy ROI (crosshair disc)
        for _ in range(self.cfg.REF_UPDATE_EVERY + 2):
            sess._maybe_update_reference(self.frame, similarity=0.9,
                                         edge_near=False, jump_vetoed=False,
                                         measured_ok=True, tracker_conf=0.9)
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "overlay-heavy snapshot became the reference")

    def test_known_occlusion_freezes_lost_streak(self):
        # Bad frames while the point sits inside the dilated overlay zone are
        # KNOWN occlusion: the LOST streak must freeze, not advance...
        bad = res(self.cross_p.x, self.cross_p.y, ok=False, conf=0.05)
        sess = self.make_session(self.cross_p, [bad])
        for _ in range(3 * self.cfg.LOST_AFTER_N_BAD):
            out = sess.step(self.frame)
        self.assertIsNot(out.state, TrackState.LOST,
                         "LOST declared during known overlay occlusion")
        # ...while the same evidence on clean ground still reaches LOST.
        bad2 = res(self.clean_p.x, self.clean_p.y, ok=False, conf=0.05)
        sess2 = self.make_session(self.clean_p, [bad2])
        for _ in range(self.cfg.LOST_AFTER_N_BAD):
            out2 = sess2.step(self.frame)
        self.assertIs(out2.state, TrackState.LOST)

    def test_overlay_coverage_signal(self):
        sess = self.make_session(self.diag_p)
        out = sess.step(self.frame)
        self.assertIn("overlay", out.signals)
        self.assertGreater(out.signals["overlay"], 0.0)
        sess2 = self.make_session(self.clean_p)
        self.assertEqual(sess2.step(self.frame).signals["overlay"], 0.0)


class TestInitOnOverlayPolicy(unittest.TestCase):
    """Policy B: selections on the HUD are accepted; support decides honesty."""

    def setUp(self):
        self.cfg = make_cfg()
        # real proportions: 51px ROI vs 12px band + r45 disc (as on the sample
        # video, where an on-line click measures ~0.19 seedable and the
        # crosshair disc 0.000)
        self.cfg.OVERLAY_CROSSHAIR_RADIUS_PX = 45
        self.cfg.OVERLAY_DIAG_THICKNESS_PX = 12
        # session-side keys for TrackingSession
        for k, v in vars(make_session_cfg()).items():
            if not hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.cfg.PATCH_SIZE = 51
        self.frame = hud_scene(cfg=self.cfg)

    def test_init_on_line_with_support_tracks_from_clean_seeds(self):
        on_line = utils.Point2D(
            float(round(diag1_x(self.cfg, 1920, 1080, 736))), 736.0)
        sess = TrackingSession(OpticalFlowTracker(self.cfg), self.cfg)
        sess.init(self.frame, on_line)
        self.assertFalse(sess.init_dead_zone)
        self.assertGreater(sess.init_overlay_cov, 0.0)
        self.assertTrue(sess.init_on_overlay)
        out = sess.step(self.frame)  # static scene: measurement must exist
        self.assertGreaterEqual(out.result.n_points, self.cfg.LK_MIN_TRACK_POINTS,
                                "on-line init with clean support failed to seed")
        self.assertTrue(out.result.ok)

    def test_init_at_crosshair_dead_zone_fabricates_nothing(self):
        center = utils.Point2D(960.0, 540.0)
        sess = TrackingSession(OpticalFlowTracker(self.cfg), self.cfg)
        sess.init(self.frame, center)
        self.assertTrue(sess.init_dead_zone)
        self.assertLess(sess.init_seedable, self.cfg.OVERLAY_INIT_MIN_SEEDABLE)
        for _ in range(10):
            out = sess.step(self.frame)
            self.assertFalse(out.result.ok, "dead zone produced a measurement")
            self.assertEqual(out.result.n_points, 0,
                             "dead zone fabricated LK points")
            self.assertIsNot(out.state, TrackState.TRACKING,
                             "dead zone reported as TRACKING")


class TestIdentityMaskRC1(unittest.TestCase):
    """RC1: the geometry-faithful IDENTITY (stroke) mask restores M8 local
    appearance identity under the HUD that the conservative SEED (filled-disc)
    mask destroyed. The seed mask (kind='seed', default) is UNCHANGED and still
    used for seeding/ignore/occlusion; only M8 similarity uses the identity mask.
    """

    def setUp(self):
        self.cfg = make_session_cfg()
        self.cfg.OVERLAY_CROSSHAIR_RADIUS_PX = 45   # PRODUCTION disc reproduces the bug
        self.cfg.PATCH_SIZE = 51                    # production-scale appearance patch
        self.frame = letterboxed_frame(textured=True)
        self.h, self.w = self.frame.shape[:2]
        self.center = utils.Point2D(self.w / 2.0, self.h / 2.0)   # under the crosshair

    def _valid_frac(self, mask, pt):
        x0, y0, x1, y1 = utils.clamp_roi(pt.x, pt.y, self.cfg.PATCH_SIZE, self.w, self.h)
        return float((mask[y0:y1, x0:x1] == 0).mean())

    def test_seed_disc_erases_patch_but_identity_preserves_scene(self):
        seed = utils.build_overlay_mask(self.frame, self.cfg)              # kind='seed'
        ident = utils.build_overlay_mask(self.frame, self.cfg, kind="identity")
        self.assertEqual(self._valid_frac(seed, self.center), 0.0,
                         "seed disc must fully mask the under-HUD patch (the RC1 bug)")
        self.assertGreaterEqual(self._valid_frac(ident, self.center),
                                self.cfg.OVERLAY_SIM_MIN_VALID_FRAC,
                                "identity mask must keep >= the similarity gate valid")

    def test_identity_masks_actual_strokes_only(self):
        ident = utils.build_overlay_mask(self.frame, self.cfg, kind="identity")
        icx, icy = int(self.w // 2), int(self.h // 2)
        s = self.cfg.OVERLAY_X_SLOPE
        # actual strokes remain masked:
        self.assertEqual(ident[icy, icx], 255, "centre dot not masked")
        yd = icy - 80
        self.assertEqual(ident[yd, int(round(icx + s * (yd - icy)))], 255,
                         "diagonal stroke not masked")
        self.assertEqual(ident[icy - 15, icx], 255, "vertical tick not masked")
        # scene BETWEEN the strokes stays valid (>= gate); do NOT require the
        # literal centre pixel to be valid.
        self.assertGreaterEqual(self._valid_frac(ident, self.center),
                                self.cfg.OVERLAY_SIM_MIN_VALID_FRAC)

    def test_session_under_hud_has_verifiable_identity(self):
        sess = TrackingSession(StubTracker([res(self.center.x, self.center.y)]), self.cfg)
        sess.init(self.frame, self.center)
        self.assertGreater(sess._ref_std, 1e-6,
                           "identity mask must give a non-vacuous reference (RC1)")
        self.assertIsNotNone(sess._patch_similarity(self.frame, self.center),
                             "similarity must be a real value under the HUD (RC1)")

    def test_seed_mask_still_dead_zone_for_seeding(self):
        # seeding safety unchanged: the seed disc still fully covers the patch.
        sess = TrackingSession(StubTracker([res(self.center.x, self.center.y)]), self.cfg)
        sess.init(self.frame, self.center)
        self.assertEqual(self._valid_frac(sess._overlay, self.center), 0.0)
        self.assertTrue(sess.init_dead_zone, "seed-based dead-zone telemetry unchanged")


if __name__ == "__main__":
    unittest.main()
