"""Stage 2B unit tests — RAW-SCORE CORRECTNESS ONLY (no verdicts).

Every assertion here is about a number or a capability flag the reacquirer
COMPUTES: coordinate transport, offset geometry, immutability, capability
honesty, verify_at monotonicity / None-cases, overlay-independence. Nothing
here asserts an accept/reject outcome — thresholds, peak-margin gates and
GOOD/BAD classification are Stage 2C and must NOT appear.

Self-contained config (like tests/test_session.py) so the tests are immune to
production retuning; the scale ladder is kept at the production values so the
scale round-trip can land on a real ladder entry.

Run from the repository root:
    python3 -m unittest tests.test_reacquisition -v
"""
import math
import sys
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import reacquisition as R
from ground_target_tracking import utils


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        PATCH_SIZE=21,
        MIN_PATCH_SIZE=5,
        REACQ_TEMPLATE_SIZE=81,
        REACQ_PYRAMID_SCALE=0.5,
        REACQ_SCALES=(0.6, 0.8, 1.0, 1.3, 1.7),
        REACQ_MIN_REF_STD=2.0,
        REACQ_KP_RADIUS_PX=100,
        REACQ_MIN_REF_KP=10,
        REACQ_PEAK_NMS_FRAC=0.5,
        REACQ_COARSE_MASKED=True,
        OVERLAY_SIM_MIN_VALID_FRAC=0.30,
        # preprocessing (template_pipeline reads these):
        GAMMA=1.0, USE_CLAHE=False, USE_GAUSSIAN_BLUR=False,
        OVERLAY_MASK_ENABLED=False,
        ORB_N_FEATURES=500,
        # M9-a feature-proposal candidate-generation params (experimental, uncalibrated):
        REACQ_FEAT_RATIO=0.75,
        REACQ_FEAT_MIN_MATCHES=6,
        REACQ_FEAT_RANSAC_THRESH=3.0,
        REACQ_FEAT_DETECT_SCALE=0.66,   # V2b: downscaled detect; feature tests exercise this path
        # M9-b Stage-2C decision params (test-local, self-contained — immune to
        # production recalibration; values chosen so the deterministic scenes
        # below sit clearly on their intended side of each gate):
        REACQ_MIN_INLIERS=8,
        REACQ_MIN_INLIER_RATIO=0.45,
        REACQ_MAX_RESIDUAL_PX=2.0,
        REACQ_FEAT_SCALE_MIN=0.5,
        REACQ_FEAT_SCALE_MAX=2.0,
        REACQ_CONFIRM_MIN_NCC_FEAT=0.35,
        REACQ_TEMPLATE_MIN_NCC=0.60,
        REACQ_TEMPLATE_MIN_MARGIN=0.10,
        REACQ_CONFIRM_MIN_NCC_TMPL=0.50,
        REACQ_PERSIST_N=3,
        REACQ_PERSIST_MAX_MOVE_PX=30.0,
        REACQ_PERSIST_SCALE_TOL=0.25,
        REACQ_PERSIST_MAX_NEUTRAL=2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def textured(h, w, seed, blur=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (blur, blur), 0)


def warp(frame, M, border=cv2.BORDER_REFLECT):
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, M.astype(np.float32), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=border)


class ReacqTestBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.frame = textured(480, 640, 7)
        self.p = utils.Point2D(400.0, 240.0)

    def built(self, frame=None, point=None, overlay_mask=None, cfg=None):
        rq = R.Reacquirer(cfg or self.cfg)
        rq.build_reference(frame if frame is not None else self.frame,
                           point or self.p, overlay_mask)
        return rq

    def best(self, cands):
        return max(cands, key=lambda c: c.raw_score)


class TestImmutability(ReacqTestBase):
    def test_reference_arrays_are_read_only(self):
        ref = self.built().reference
        for arr in (ref.context, ref.patch):
            self.assertFalse(arr.flags.writeable, "reference array must be read-only")
        with self.assertRaises(ValueError):
            ref.context[0, 0] = 0  # in-place write must fail

    def test_reference_bytes_unchanged_after_calls(self):
        rq = self.built()
        before = rq.reference.context.tobytes()
        for _ in range(5):
            rq.propose(self.frame)
            rq.verify_at(self.frame, self.p)
        self.assertEqual(rq.reference.context.tobytes(), before,
                         "propose/verify_at must not mutate the reference")


class TestCoordinateContract(ReacqTestBase):
    def test_self_match_top_peak_on_the_point(self):
        rq = self.built()
        best = self.best(rq.propose(self.frame))
        self.assertAlmostEqual(best.point.x, self.p.x, delta=2.0)
        self.assertAlmostEqual(best.point.y, self.p.y, delta=2.0)
        self.assertEqual(best.scale, 1.0)

    def test_translation_roundtrip(self):
        rq = self.built()
        dx, dy = -60.0, 40.0
        moved = warp(self.frame, np.array([[1, 0, dx], [0, 1, dy]]))
        best = self.best(rq.propose(moved))
        self.assertAlmostEqual(best.point.x, self.p.x + dx, delta=2.5)
        self.assertAlmostEqual(best.point.y, self.p.y + dy, delta=2.5)
        self.assertEqual(best.scale, 1.0)

    def test_scale_roundtrip_reports_ladder_entry(self):
        rq = self.built()
        z = 1.3
        cx, cy = self.p.x, self.p.y            # zoom about the point: it stays fixed
        M = np.array([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
        zoomed = warp(self.frame, M)
        best = self.best(rq.propose(zoomed))
        self.assertAlmostEqual(best.point.x, self.p.x, delta=4.0)
        self.assertAlmostEqual(best.point.y, self.p.y, delta=4.0)
        self.assertEqual(best.scale, 1.3, "winning ladder scale should match the zoom")

    def test_offset_stored_for_border_clamped_init(self):
        # A point near the right border: the init window clamps, so the point's
        # offset within the window is NOT the centre. verify_at must still
        # self-match, which only works if the offset is stored, not assumed.
        edge = utils.Point2D(620.0, 240.0)     # 20px from the 640 border
        rq = self.built(point=edge)
        ref = rq.reference
        # clamp_roi SHRINKS the window at the border (does not shift), so the
        # actual window is narrower than REACQ_TEMPLATE_SIZE and the point sits
        # RIGHT of the window centre: offset must exceed actual_width / 2.
        self.assertLess(ref.context.shape[1], self.cfg.REACQ_TEMPLATE_SIZE,
                        "the border window should be clamped narrower")
        self.assertGreater(ref.context_offset[0], ref.context.shape[1] / 2.0,
                           "clamped window must record an off-centre point offset")
        # The only thing that makes a self-match work at a clamped edge is the
        # stored offset being correct:
        self.assertAlmostEqual(rq.verify_at(self.frame, edge), 1.0, delta=1e-3)


class TestCapabilityHonesty(ReacqTestBase):
    def test_textured_init_enables_context(self):
        ref = self.built().reference
        self.assertTrue(ref.has_context)
        self.assertGreaterEqual(ref.context_std, self.cfg.REACQ_MIN_REF_STD)

    def test_flat_init_disables_context_and_yields_nothing(self):
        flat = np.full((480, 640, 3), 128, np.uint8)
        rq = self.built(frame=flat, point=utils.Point2D(320.0, 240.0))
        self.assertFalse(rq.reference.has_context)
        self.assertEqual(rq.propose(flat), [], "flat reference must propose nothing")
        self.assertIsNone(rq.verify_at(flat, utils.Point2D(320.0, 240.0)))

    def test_fully_masked_init_disables_context(self):
        mask = np.full((480, 640), 255, np.uint8)   # everything is overlay
        rq = self.built(overlay_mask=mask)
        self.assertFalse(rq.reference.has_context,
                         "a window with no observed pixels cannot be a reference")
        self.assertIsNone(rq.verify_at(self.frame, self.p))

    def test_masked_keypoints_excluded_from_reference(self):
        # HUD pixels must never become feature evidence: with a band mask
        # through the point, every KEPT reference keypoint lies off the mask,
        # and the context validity zeroes exactly the masked window pixels.
        mask = np.zeros((480, 640), np.uint8)
        mask[230:251, :] = 255                    # band through p=(400,240)
        rq = self.built(overlay_mask=mask)
        ref = rq.reference
        self.assertIsNotNone(ref.kp_xy,
                             "textured scene should keep off-band keypoints")
        for kx, ky in ref.kp_xy:
            ix = min(max(int(round(float(kx))), 0), 639)
            iy = min(max(int(round(float(ky))), 0), 479)
            self.assertEqual(mask[iy, ix], 0,
                             f"kept keypoint ({kx:.1f},{ky:.1f}) lies on the mask")
        x0, y0, x1, y1 = utils.clamp_roi(self.p.x, self.p.y,
                                         self.cfg.REACQ_TEMPLATE_SIZE, 640, 480)
        win_mask = mask[y0:y1, x0:x1]
        self.assertTrue((ref.context_valid[win_mask > 0] == 0).all(),
                        "masked pixels must be invalid in the context reference")
        self.assertTrue((ref.context_valid[win_mask == 0] == 255).all(),
                        "clean pixels must stay valid in the context reference")

    def test_descriptor_capability_flag(self):
        ref = self.built().reference
        self.assertEqual(ref.has_descriptors, ref.kp_xy is not None
                         and len(ref.kp_xy) >= self.cfg.REACQ_MIN_REF_KP)
        # descriptors are BUILT in 2B but propose is template-only:
        self.assertTrue(all(c.cue == "template" for c in self.built().propose(self.frame)))


class TestVerifyRaw(ReacqTestBase):
    def test_verify_none_off_frame(self):
        rq = self.built()
        self.assertIsNone(rq.verify_at(self.frame, utils.Point2D(-5.0, 240.0)))
        self.assertIsNone(rq.verify_at(self.frame, utils.Point2D(640.0, 240.0)))

    def test_verify_monotonic_vs_corruption(self):
        rq = self.built()
        scores = []
        for sigma in (0.0, 15.0, 35.0, 70.0):
            f = self.frame.astype(np.float32)
            rng = np.random.default_rng(1)
            noise = rng.normal(0, sigma, f.shape).astype(np.float32)
            corrupted = np.clip(f + noise, 0, 255).astype(np.uint8)
            scores.append(rq.verify_at(corrupted, self.p))
        for a, b in zip(scores, scores[1:]):
            self.assertGreaterEqual(a + 1e-6, b,
                                    f"verify_at should not rise with corruption: {scores}")

    def test_verify_low_but_not_none_on_unrelated_content(self):
        # No thresholding: an unrelated (but observed) neighbourhood still
        # yields a RAW number, not None. 2C decides if it is good enough.
        rq = self.built()
        other = textured(480, 640, 999)
        s = rq.verify_at(other, self.p)
        self.assertIsNotNone(s, "observed content must return a raw score, not None")
        self.assertLess(s, 0.5, "unrelated content should score low (but is not gated here)")


class TestOverlayIndependence(ReacqTestBase):
    def test_masked_ncc_degenerates_to_plain_matchtemplate(self):
        a = textured(41, 41, 3)[:, :, 0]
        b = textured(41, 41, 8)[:, :, 0]
        plain = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
        got = R._masked_ncc(a, b, None, 0.30)
        self.assertAlmostEqual(got, plain, places=5,
                               msg="valid=None must equal unmasked NCC")

    def test_masked_hole_reference_still_scores(self):
        # A validity hole in the reference (a stripe of overlay) must not kill
        # the score: enough observed pixels remain to self-match.
        mask = np.zeros((480, 640), np.uint8)
        mask[235:245, :] = 255                 # a thin horizontal overlay stripe
        rq = self.built(overlay_mask=mask)
        self.assertTrue(rq.reference.has_context)
        s = rq.verify_at(self.frame, self.p)
        self.assertIsNotNone(s)
        self.assertGreater(s, 0.8, "masked self-match should still score high")


class TestNoThresholding(ReacqTestBase):
    def test_propose_returns_candidates_regardless_of_quality(self):
        # 2B must not gate: even a frame whose best match is weak returns one
        # RawCandidate per fitting scale, each carrying a finite raw score.
        rq = self.built()
        weak = textured(480, 640, 424242)      # unrelated scene
        cands = rq.propose(weak)
        self.assertTrue(len(cands) >= 1, "propose must not drop candidates by score")
        for c in cands:
            self.assertTrue(np.isfinite(c.raw_score))
            self.assertTrue(np.isfinite(c.second_peak_score))
            self.assertEqual(c.cue, "template")

    def test_single_scale_argument_is_honored(self):
        rq = self.built()
        cands = rq.propose(self.frame, scales=[1.0])
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].scale, 1.0)


class TestDistractorSceneBuilder(unittest.TestCase):
    """Construction sanity for the Amendment-A1 in-frame scene (built in 2B,
    consumed by the 2D integration benchmark). No session, no verdicts here."""

    def _import(self):
        try:
            from synthetic_scenes import build_distractor_scene
        except ImportError:
            from tests.synthetic_scenes import build_distractor_scene
        return build_distractor_scene

    def test_scene_is_well_formed(self):
        build = self._import()
        frames, gt = build(n_frames=30)
        self.assertEqual(len(frames), 30)
        self.assertEqual(len(gt), 30)
        self.assertEqual(frames[0].shape, (480, 640, 3))
        # target GT is constant and always in-frame (an always-visible target):
        self.assertTrue(all(p.x == gt[0].x and p.y == gt[0].y for p in gt))
        self.assertTrue(0 <= gt[0].x < 640 and 0 <= gt[0].y < 480)
        # the sweep actually changes the frame content (distractor moves):
        self.assertFalse(np.array_equal(frames[0], frames[-1]))


# =========================================================================== #
# M9-a: raw ORB feature proposer (propose_features) — RAW EVIDENCE ONLY.
# No assertion here checks an accept/reject verdict, routing, margin, or
# persistence: those are Stage 2C. Every assertion is about a number/geometry
# the proposer COMPUTES (transport, rotation/scale recovery, capability, RANSAC
# rejection, immutability). Angles are specific measured cases, NOT a
# universal-360 recall claim.
# =========================================================================== #
class FeatureBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.frame = textured(480, 640, 11)   # strongly textured -> ample ORB keypoints
        self.p = utils.Point2D(320.0, 240.0)  # central: keypoint support stays interior under warp
        self.rq = R.Reacquirer(self.cfg)
        self.rq.build_reference(self.frame, self.p)

    def only(self, cands):
        self.assertEqual(len(cands), 1, "single-fit baseline yields exactly one candidate")
        return cands[0]

    def expect_pt(self, Mw):
        return (Mw[0, 0] * self.p.x + Mw[0, 1] * self.p.y + Mw[0, 2],
                Mw[1, 0] * self.p.x + Mw[1, 1] * self.p.y + Mw[1, 2])


class TestFeatureCapability(FeatureBase):
    def test_reference_has_feature_capability(self):
        self.assertTrue(self.rq.reference.has_descriptors)
        self.assertIsNotNone(self.rq.reference.descriptors)

    def test_no_descriptors_yields_no_candidates(self):
        flat = np.full((480, 640, 3), 128, np.uint8)
        rq = R.Reacquirer(self.cfg)
        rq.build_reference(flat, utils.Point2D(320.0, 240.0))
        self.assertFalse(rq.reference.has_descriptors)
        self.assertEqual(rq.propose_features(self.frame), [],
                         "no feature capability -> [] cleanly")

    def test_insufficient_frame_descriptors_no_crash(self):
        flat = np.full((480, 640, 3), 128, np.uint8)
        self.assertEqual(self.rq.propose_features(flat), [],
                         "flat query frame -> no candidate, no crash")

    def test_unrelated_scene_yields_no_candidate(self):
        # Measured deterministic: the ratio test leaves too few survivors on an
        # unrelated scene, so no fit is attempted (generation floor, not a verdict).
        unrelated = textured(480, 640, 424242)
        self.assertEqual(self.rq.propose_features(unrelated), [],
                         "unrelated content -> too few good matches -> []")


class TestFeatureTransport(FeatureBase):
    def test_self_match_transports_the_point(self):
        c = self.only(self.rq.propose_features(self.frame))
        self.assertAlmostEqual(c.point.x, self.p.x, delta=1.0)
        self.assertAlmostEqual(c.point.y, self.p.y, delta=1.0)
        self.assertAlmostEqual(c.scale, 1.0, delta=0.02)
        self.assertAlmostEqual(c.rotation_deg, 0.0, delta=1.0)
        self.assertEqual(c.cue, "orb-feature")

    def test_translation_transport(self):
        for (dx, dy) in [(-50.0, 30.0), (70.0, -40.0)]:
            Mw = np.array([[1, 0, dx], [0, 1, dy]], float)
            c = self.only(self.rq.propose_features(
                warp(self.frame, Mw, cv2.BORDER_REFLECT_101)))
            ex, ey = self.expect_pt(Mw)
            self.assertAlmostEqual(c.point.x, ex, delta=2.0, msg=f"dx={dx} dy={dy}")
            self.assertAlmostEqual(c.point.y, ey, delta=2.0, msg=f"dx={dx} dy={dy}")

    def test_rotation_recovered_including_beyond_40_and_180(self):
        # These SPECIFIC angles are recovered natively (no template angle sweep).
        # This is not a universal-360 recall claim.
        for ang in (15, 45, 90, 135, 200, 300):
            Mw = cv2.getRotationMatrix2D((self.p.x, self.p.y), ang, 1.0)
            c = self.only(self.rq.propose_features(
                warp(self.frame, Mw, cv2.BORDER_REFLECT_101)))
            ex, ey = self.expect_pt(Mw)
            exp_rot = math.degrees(math.atan2(Mw[1, 0], Mw[0, 0]))
            self.assertAlmostEqual(c.point.x, ex, delta=3.0, msg=f"ang={ang}")
            self.assertAlmostEqual(c.point.y, ey, delta=3.0, msg=f"ang={ang}")
            self.assertAlmostEqual(c.rotation_deg, exp_rot, delta=2.0, msg=f"ang={ang}")
            self.assertAlmostEqual(c.scale, 1.0, delta=0.05, msg=f"ang={ang}")

    def test_scale_recovered(self):
        for z in (0.8, 1.3):
            Mw = cv2.getRotationMatrix2D((self.p.x, self.p.y), 0, z)
            c = self.only(self.rq.propose_features(
                warp(self.frame, Mw, cv2.BORDER_REFLECT_101)))
            ex, ey = self.expect_pt(Mw)
            self.assertAlmostEqual(c.point.x, ex, delta=3.0, msg=f"z={z}")
            self.assertAlmostEqual(c.point.y, ey, delta=3.0, msg=f"z={z}")
            self.assertAlmostEqual(c.scale, z, delta=0.05, msg=f"z={z}")

    def test_combined_translation_rotation_scale(self):
        Mw = cv2.getRotationMatrix2D((self.p.x, self.p.y), 25, 1.2)
        Mw[0, 2] += -30
        Mw[1, 2] += 20
        c = self.only(self.rq.propose_features(
            warp(self.frame, Mw, cv2.BORDER_REFLECT_101)))
        ex, ey = self.expect_pt(Mw)
        exp_rot = math.degrees(math.atan2(Mw[1, 0], Mw[0, 0]))
        self.assertAlmostEqual(c.point.x, ex, delta=3.0)
        self.assertAlmostEqual(c.point.y, ey, delta=3.0)
        self.assertAlmostEqual(c.rotation_deg, exp_rot, delta=2.0)
        self.assertAlmostEqual(c.scale, 1.2, delta=0.05)

    def test_transport_uses_selected_point_not_window_centre(self):
        # Border-clamped selected point: the context-window centre differs from
        # the stored point, so a (wrong) window-centre transport would land
        # elsewhere. The feature path transports ref.point directly.
        pe = utils.Point2D(620.0, 240.0)
        rq = R.Reacquirer(self.cfg)
        rq.build_reference(self.frame, pe)
        ref = rq.reference
        self.assertTrue(ref.has_descriptors)
        win_centre_x = ref.point.x - ref.context_offset[0] + ref.context.shape[1] / 2.0
        self.assertLess(win_centre_x, ref.point.x - 5.0,
                        "border clamp must put the window centre well left of the point")
        # dx kept small: at detect scale 0.66 the ORB edge margin thins the
        # right-border keypoints, so a large translation drops below the min-match
        # floor. -10 retains a stable fit while keeping a clean 10px window-vs-point
        # separation to distinguish the two transport hypotheses.
        dx = -10.0
        c = self.only(rq.propose_features(
            warp(self.frame, np.array([[1, 0, dx], [0, 1, 0]], float), cv2.BORDER_REFLECT_101)))
        self.assertAlmostEqual(c.point.x, pe.x + dx, delta=2.0,
                               msg="transport must follow the stored selected pixel")
        self.assertGreater(abs(c.point.x - (win_centre_x + dx)), 5.0,
                           "transport must NOT follow the window centre")


class TestFeatureRansacAndImmutability(FeatureBase):
    def test_ransac_rejects_geometric_decoy(self):
        dx = -50.0
        q = warp(self.frame, np.array([[1, 0, dx], [0, 1, 0]], float),
                 cv2.BORDER_REFLECT_101).copy()
        # Paste a copy of the reference neighbourhood far away -> a geometric decoy
        # that generates matches inconsistent with the dominant translation.
        sx0, sy0 = int(self.p.x) - 70, int(self.p.y) - 70
        patch = self.frame[sy0:sy0 + 140, sx0:sx0 + 140].copy()
        q[40:40 + 140, 60:60 + 140] = patch
        c = self.only(self.rq.propose_features(q))
        self.assertAlmostEqual(c.point.x, self.p.x + dx, delta=2.0,
                               msg="RANSAC must lock the dominant (true) transform, not the decoy")
        self.assertLess(c.n_inliers, c.n_good,
                        "some geometrically inconsistent (decoy) matches must be rejected")

    def test_reference_immutable_after_repeated_propose_features(self):
        ref = self.rq.reference
        d0, k0 = ref.descriptors.tobytes(), ref.kp_xy.tobytes()
        p0 = (ref.point.x, ref.point.y)
        for _ in range(5):
            self.rq.propose_features(
                warp(self.frame, np.array([[1, 0, 5], [0, 1, 5]], float), cv2.BORDER_REFLECT_101))
        self.assertEqual(ref.descriptors.tobytes(), d0, "descriptors must not change")
        self.assertEqual(ref.kp_xy.tobytes(), k0, "keypoint coords must not change")
        self.assertEqual((ref.point.x, ref.point.y), p0, "selected point must not change")
        self.assertFalse(ref.descriptors.flags.writeable)

    def test_candidate_is_raw_evidence_only(self):
        c = self.only(self.rq.propose_features(self.frame))
        for f in ("point", "rotation_deg", "scale", "n_inliers", "inlier_ratio",
                  "residual", "cue", "n_query_desc", "n_matches", "n_good"):
            self.assertTrue(hasattr(c, f), f"raw field {f} must exist")
        # No verdict / routing field may exist on a 2B raw candidate:
        self.assertFalse(hasattr(c, "accepted"))
        self.assertFalse(hasattr(c, "verdict"))
        self.assertFalse(hasattr(c, "is_identity"))
        self.assertEqual(c.cue, "orb-feature")
        self.assertTrue(np.isfinite(c.inlier_ratio))
        self.assertGreaterEqual(c.n_inliers, 0)


class TestFeatureResidualContract(unittest.TestCase):
    """The Optional[float] residual contract, tested directly on the helper so
    the None branch is exercised deterministically (a returned fit rarely has 0
    inliers)."""

    def test_residual_none_when_no_inliers(self):
        M = np.array([[1, 0, 3], [0, 1, -2]], float)
        src = np.array([[10, 10], [20, 30]], np.float32)
        dst = np.array([[13, 8], [23, 28]], np.float32)
        self.assertIsNone(R.Reacquirer._inlier_residual(M, src, dst, None))
        self.assertIsNone(R.Reacquirer._inlier_residual(
            M, src, dst, np.array([False, False])))

    def test_residual_matches_reprojection(self):
        # Pure translation (3,-2), perfectly consistent points -> residual 0.
        M = np.array([[1, 0, 3], [0, 1, -2]], float)
        src = np.array([[10, 10], [20, 30], [5, 40]], np.float32)
        dst = src + np.array([3, -2], np.float32)
        r = R.Reacquirer._inlier_residual(M, src, dst, np.array([True, True, True]))
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 0.0, places=4)


# =========================================================================== #
# M9-a V2b: downscaled ORB detection with coords mapped back to full-resolution.
# =========================================================================== #
class TestDownscaleDetectPath(unittest.TestCase):
    """ORB detection runs on a frame downscaled by REACQ_FEAT_DETECT_SCALE with
    keypoint coordinates mapped BACK to full resolution. These exercise that
    path directly: actual (not nominal) resize ratios, ref/query full-res
    round-trip, odd dimensions, and the small-target capability floor."""

    def setUp(self):
        self.cfg = make_cfg()   # REACQ_FEAT_DETECT_SCALE = 0.66

    def only_one(self, cands):
        self.assertEqual(len(cands), 1, "single-fit baseline yields exactly one candidate")
        return cands[0]

    def test_scaled_size_uses_actual_ratios_not_nominal(self):
        # Odd dimensions: integer resize rounding makes the true per-axis factor
        # differ from the nominal 0.66; the map-back MUST use the actual factor.
        w, h, s = 641, 483, 0.66
        new_w, new_h, sx, sy = R.Reacquirer._scaled_size(w, h, s)
        self.assertEqual(new_w, round(w * s))
        self.assertEqual(new_h, round(h * s))
        self.assertNotAlmostEqual(sx, s, places=4,
                                  msg="odd dims: actual sx must differ from nominal 0.66")
        self.assertAlmostEqual(sx, new_w / w, places=12)
        self.assertAlmostEqual(sy, new_h / h, places=12)

    def test_scale_1p0_degenerates_to_full_frame(self):
        # REACQ_FEAT_DETECT_SCALE == 1.0 must map back with factor 1 (no shift).
        for dim in ((640, 480), (641, 483)):
            nw, nh, sx, sy = R.Reacquirer._scaled_size(dim[0], dim[1], 1.0)
            self.assertEqual((nw, nh), dim)
            self.assertEqual((sx, sy), (1.0, 1.0))

    def test_reference_query_fullres_roundtrip_at_0p66(self):
        # Self-propose: ref and query share the SAME full-res basis, so the fit is
        # ~identity -> scale ~1.0 and the point transports to itself. A downscaled
        # query basis (coords NOT mapped back) would instead show scale ~0.66.
        frame = textured(480, 640, 11)
        p = utils.Point2D(320.0, 240.0)
        rq = R.Reacquirer(self.cfg)
        rq.build_reference(frame, p)
        c = self.only_one(rq.propose_features(frame))
        self.assertAlmostEqual(c.scale, 1.0, delta=0.02,
                               msg="ref/query share full-res basis -> scale ~1.0, not ~0.66")
        self.assertAlmostEqual(c.point.x, p.x, delta=1.0)
        self.assertAlmostEqual(c.point.y, p.y, delta=1.0)

    def test_odd_frame_dimensions_transport(self):
        # Odd H and W: only correct if the real resize ratios are used (not /0.66).
        frame = textured(483, 641, 11)
        p = utils.Point2D(321.0, 243.0)
        rq = R.Reacquirer(self.cfg)
        rq.build_reference(frame, p)
        c = self.only_one(rq.propose_features(frame))
        self.assertAlmostEqual(c.point.x, p.x, delta=1.5)
        self.assertAlmostEqual(c.point.y, p.y, delta=1.5)
        self.assertAlmostEqual(c.scale, 1.0, delta=0.02)

    def test_small_target_48px_capability_and_accuracy(self):
        # 48px textured target on a low-texture background at detect scale 0.66.
        # Measured reliable at the test resolution (floor ~40px; 24px loses
        # rotation, 32px loses scale -- documented in the M9-a report). This
        # asserts (1) CAPABILITY (a candidate is produced; keypoint floor cleared)
        # and (2) GEOMETRIC ACCURACY (transport within tolerance). It deliberately
        # does NOT assert any inlier/ratio acceptance threshold -- (3) identity
        # acceptance is Stage 2C (M9-b), not M9-a.
        s = 48
        rng = np.random.default_rng(2)
        patch = cv2.GaussianBlur(rng.integers(0, 256, (s, s, 3), np.uint8), (3, 3), 0)
        scene = np.full((480, 640, 3), 128, np.uint8)
        scene[240 - s // 2:240 + s // 2, 320 - s // 2:320 + s // 2] = patch
        p = utils.Point2D(320.0, 240.0)
        rq = R.Reacquirer(self.cfg)
        rq.build_reference(scene, p)
        self.assertTrue(rq.reference.has_descriptors,
                        "48px target must clear the keypoint capability floor")
        cs = self.only_one(rq.propose_features(scene))       # capability + accuracy: self
        self.assertAlmostEqual(cs.point.x, p.x, delta=1.5)
        self.assertAlmostEqual(cs.point.y, p.y, delta=1.5)
        Mw = cv2.getRotationMatrix2D((p.x, p.y), 30, 1.0)     # capability + accuracy: rotation
        cr = self.only_one(rq.propose_features(warp(scene, Mw, cv2.BORDER_REFLECT_101)))
        ex = Mw[0, 0] * p.x + Mw[0, 1] * p.y + Mw[0, 2]
        ey = Mw[1, 0] * p.x + Mw[1, 1] * p.y + Mw[1, 2]
        self.assertAlmostEqual(cr.point.x, ex, delta=3.0)
        self.assertAlmostEqual(cr.point.y, ey, delta=3.0)


# =========================================================================== #
# GOLDEN EQUIVALENCE: locks the UNCHANGED template path against commit 806564c.
# =========================================================================== #
class TestGoldenTemplateEquivalence(unittest.TestCase):
    """The reacquisition.py template path (build_reference / propose / verify_at)
    is byte-identical to commit 806564c (verified at M9-a authoring time:
    `git diff 806564c HEAD -- ground_target_tracking/reacquisition.py` is empty).
    The golden constants below were generated from that pristine path with
    make_cfg() on the fixed seeds used here, reviewed, and embedded so this test
    needs NO git access at runtime. M9-a only ADDS FeatureCandidate/
    propose_features; if any future edit alters the template path these fail.

    Tolerances: matchTemplate (TM_CCOEFF_NORMED) is not guaranteed bit-identical
    across BLAS builds, so scores are checked to places=5; coarse-peak-derived
    points land on pixel-quantised values (places=3); structure (count / order /
    scale / cue) is exact. Values are machine/OpenCV-specific golden references.
    """

    GOLD_S1 = [
        (0.6, 264.0, 84.0, 0.190787, 0.185861, "template"),
        (0.8, 114.0, 362.0, 0.167954, 0.148895, "template"),
        (1.0, 400.0, 240.0, 0.999992, 0.120830, "template"),
        (1.3, 112.0, 366.0, 0.090986, 0.088732, "template"),
        (1.7, 412.0, 322.0, 0.080815, 0.073324, "template"),
    ]
    GOLD_S2 = [
        (0.6, 606.0, 34.0, 0.201203, 0.190776, "template"),
        (0.8, 54.0, 402.0, 0.167950, 0.148900, "template"),
        (1.0, 340.0, 280.0, 1.000000, 0.120356, "template"),
        (1.3, 576.0, 218.0, 0.097655, 0.090986, "template"),
        (1.7, 352.0, 362.0, 0.080815, 0.080089, "template"),
    ]

    def setUp(self):
        self.cfg = make_cfg()
        self.frame = textured(480, 640, 7)
        self.p = utils.Point2D(400.0, 240.0)
        self.rq = R.Reacquirer(self.cfg)
        self.rq.build_reference(self.frame, self.p)

    def _assert_matches(self, cands, gold):
        self.assertEqual(len(cands), len(gold), "candidate count/order locked")
        for c, g in zip(cands, gold):
            self.assertEqual(c.scale, g[0], "scale must match")
            self.assertAlmostEqual(c.point.x, g[1], places=3)
            self.assertAlmostEqual(c.point.y, g[2], places=3)
            self.assertAlmostEqual(c.raw_score, g[3], places=5)
            self.assertAlmostEqual(c.second_peak_score, g[4], places=5)
            self.assertEqual(c.cue, g[5], "cue must match")

    def test_propose_base_matches_806564c(self):
        self._assert_matches(self.rq.propose(self.frame), self.GOLD_S1)

    def test_propose_translation_matches_806564c(self):
        moved = warp(self.frame, np.array([[1, 0, -60], [0, 1, 40]], float))
        self._assert_matches(self.rq.propose(moved), self.GOLD_S2)

    def test_single_scale_matches_806564c(self):
        cands = self.rq.propose(self.frame, scales=[1.0])
        self._assert_matches(cands, [self.GOLD_S1[2]])

    def test_verify_and_degenerate_match_806564c(self):
        self.assertAlmostEqual(self.rq.verify_at(self.frame, self.p), 0.999994, places=5)
        self.assertIsNone(self.rq.verify_at(self.frame, utils.Point2D(-5.0, 240.0)))
        self.assertIsNone(self.rq.verify_at(self.frame, utils.Point2D(640.0, 240.0)))
        self.assertAlmostEqual(self.rq.verify_at(textured(480, 640, 999), self.p),
                               0.039799, places=5)
        flat = np.full((480, 640, 3), 128, np.uint8)
        rqf = R.Reacquirer(self.cfg)
        rqf.build_reference(flat, utils.Point2D(320.0, 240.0))
        self.assertEqual(rqf.propose(flat), [])
        self.assertIsNone(rqf.verify_at(flat, utils.Point2D(320.0, 240.0)))


# =========================================================================== #
# M9-b: Stage 2C decision layer — routing, gates, confirmation, persistence.
# Verdicts (MATCH/AMBIGUOUS/NEUTRAL) are decision-layer VALUE outputs; nothing
# here mutates a tracker or session, and acceptance means only that an
# AcceptedHypothesis object was returned.
# =========================================================================== #
def gaussian_blob(h, w, cx, cy, sx, sy, amp=110, angle_deg=0.0):
    """Smooth (corner-free) anisotropic Gaussian blob on a flat background:
    a deterministic LOW-TEXTURE target — ORB finds almost no keypoints
    (has_descriptors False) while the context std stays high (has_context)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.radians(angle_deg)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    g = amp * np.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))
    img = np.clip(128 + g, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class DecisionBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.frame = textured(480, 640, 11)
        self.p = utils.Point2D(320.0, 240.0)
        self.rq = R.Reacquirer(self.cfg)
        self.rq.build_reference(self.frame, self.p)

    def blob_rq(self, frame):
        rq = R.Reacquirer(self.cfg)
        rq.build_reference(frame, self.p)
        return rq


class TestVerifyAtPose(DecisionBase):
    """Raw fitted-pose confirmation primitive (no thresholds here)."""

    def test_rotated_candidate_confirms_where_upright_cannot(self):
        q = warp(self.frame, cv2.getRotationMatrix2D((self.p.x, self.p.y), 45, 1.0),
                 cv2.BORDER_REFLECT_101)
        c = self.rq.propose_features(q)[0]
        upright = self.rq.verify_at(q, c.point)
        posed = self.rq.verify_at_pose(q, c.point, c.rotation_deg, c.scale)
        self.assertLess(upright, 0.2, "upright NCC collapses on a 45deg rotation")
        self.assertGreater(posed, 0.5, "fitted-pose NCC must confirm the same candidate")

    def test_wrong_location_scores_low(self):
        s = self.rq.verify_at_pose(self.frame, utils.Point2D(100.0, 100.0), 0.0, 1.0)
        self.assertIsNotNone(s)
        self.assertLess(s, 0.2)

    def test_none_off_frame_and_degenerate_pose(self):
        self.assertIsNone(self.rq.verify_at_pose(
            self.frame, utils.Point2D(-3.0, 100.0), 0.0, 1.0))
        self.assertIsNone(self.rq.verify_at_pose(
            self.frame, self.p, 0.0, 0.0))        # zero scale
        self.assertIsNone(self.rq.verify_at_pose(
            self.frame, self.p, float("nan"), 1.0))

    def test_none_without_context_reference(self):
        flat = np.full((480, 640, 3), 128, np.uint8)
        rq = self.blob_rq(flat)
        self.assertIsNone(rq.verify_at_pose(flat, self.p, 0.0, 1.0))


class TestDecisionRouting(DecisionBase):
    """Capability routing + corrected MATCH/AMBIGUOUS/NEUTRAL semantics."""

    def test_feature_capable_self_is_match(self):
        r = self.rq.best_candidate(self.frame)
        self.assertIs(r.identity, R.Identity.MATCH)
        self.assertEqual(r.cue, "orb-feature")
        self.assertEqual(r.reason, "feature:all-gates-pass")
        self.assertIsNotNone(r.feature)
        self.assertIsNone(r.template)

    def test_feature_match_survives_large_rotation(self):
        q = warp(self.frame, cv2.getRotationMatrix2D((self.p.x, self.p.y), 45, 1.0),
                 cv2.BORDER_REFLECT_101)
        r = self.rq.best_candidate(q)
        self.assertIs(r.identity, R.Identity.MATCH,
                      "fitted-pose confirmation must not reject valid rotation")

    def test_feature_failure_never_falls_back_to_template(self):
        # Feature-capable reference + unrelated frame: NEUTRAL (no candidate),
        # and the template cue must NOT have been consulted.
        r = self.rq.best_candidate(textured(480, 640, 777))
        self.assertIs(r.identity, R.Identity.NEUTRAL)
        self.assertEqual(r.reason, "feature:no-candidate")
        self.assertIsNone(r.template, "no silent template fallback")
        self.assertIsNone(r.cue)

    def test_strong_geometry_failed_confirm_is_ambiguous(self):
        # Ring keypoints (radius 41..100) stay intact -> strong geometry; the
        # confirm window content is replaced -> present-but-low confirmation.
        q = self.frame.copy()
        q[240 - 40:240 + 41, 320 - 40:320 + 41] = textured(81, 81, 999)
        r = self.rq.best_candidate(q)
        self.assertIs(r.identity, R.Identity.AMBIGUOUS)
        self.assertEqual(r.reason, "feature:confirm-low")
        self.assertGreaterEqual(r.feature.n_inliers, self.cfg.REACQ_MIN_INLIERS,
                                "geometry itself must have been strong")

    def test_no_reference_is_neutral(self):
        r = R.Reacquirer(self.cfg).best_candidate(self.frame)
        self.assertIs(r.identity, R.Identity.NEUTRAL)
        self.assertEqual(r.reason, "no-reference")

    def test_low_texture_blob_routes_to_template_match(self):
        blob = gaussian_blob(480, 640, 320, 240, 18, 18)
        rq = self.blob_rq(blob)
        self.assertFalse(rq.reference.has_descriptors, "blob must be below the kp floor")
        self.assertTrue(rq.reference.has_context)
        r = rq.best_candidate(blob)
        self.assertIs(r.identity, R.Identity.MATCH)
        self.assertEqual(r.cue, "template")
        moved = warp(blob, np.array([[1, 0, 60], [0, 1, -40]], float))
        r2 = rq.best_candidate(moved)
        self.assertIs(r2.identity, R.Identity.MATCH)
        self.assertAlmostEqual(r2.point.x, 380.0, delta=3.0)
        self.assertAlmostEqual(r2.point.y, 200.0, delta=3.0)

    def test_repeated_structure_is_ambiguous(self):
        rep = np.maximum(gaussian_blob(480, 640, 320, 240, 18, 18),
                         gaussian_blob(480, 640, 500, 240, 18, 18))
        rq = self.blob_rq(rep)
        r = rq.best_candidate(rep)
        self.assertIs(r.identity, R.Identity.AMBIGUOUS,
                      "two identical peaks must refuse via the margin gate")
        self.assertEqual(r.reason, "template:margin-low")

    def test_low_texture_large_rotation_refuses(self):
        # Elongated smooth bar rotated 90deg: near-upright template cannot
        # support it -> conservative refusal (never MATCH), no angle sweep.
        bar = gaussian_blob(480, 640, 320, 240, 40, 8)
        rq = self.blob_rq(bar)
        self.assertFalse(rq.reference.has_descriptors)
        r = rq.best_candidate(gaussian_blob(480, 640, 320, 240, 40, 8, angle_deg=90))
        self.assertIsNot(r.identity, R.Identity.MATCH)

    def test_no_capability_is_neutral(self):
        flat = np.full((480, 640, 3), 128, np.uint8)
        rq = self.blob_rq(flat)
        r = rq.best_candidate(flat)
        self.assertIs(r.identity, R.Identity.NEUTRAL)
        self.assertEqual(r.reason, "no-capability")

    def test_absent_target_with_decoy_never_accepts(self):
        # True target absent; a high-similarity decoy (0.95 blend of the true
        # patch) sits elsewhere. Measured behavior at detect scale 0.66: the
        # proposer yields no candidate (paste-boundary corruption) -> NEUTRAL;
        # if a candidate ever arises, the confirm/persistence gates own it.
        # Either way: never MATCH, and the tracker must never emit.
        tgt = self.frame[240 - 60:240 + 60, 320 - 60:320 + 60].copy()
        decoy = (0.95 * tgt + 0.05 * textured(120, 120, 77)).astype(np.uint8)
        absent = textured(480, 640, 501, blur=5).copy()
        absent[300 - 60:300 + 60, 480 - 60:480 + 60] = decoy
        ht = R.HypothesisTracker(self.cfg)
        for _ in range(5):
            r = self.rq.best_candidate(absent)
            self.assertIsNot(r.identity, R.Identity.MATCH)
            self.assertIsNone(ht.update(r), "absence must never emit an acceptance")

    def test_best_candidate_mutates_nothing(self):
        ref = self.rq.reference
        ctx0 = ref.context.tobytes()
        desc0 = ref.descriptors.tobytes()
        attrs0 = set(vars(self.rq).keys())
        for q in (self.frame, textured(480, 640, 777)):
            self.rq.best_candidate(q)
        self.assertEqual(ref.context.tobytes(), ctx0)
        self.assertEqual(ref.descriptors.tobytes(), desc0)
        self.assertEqual(set(vars(self.rq).keys()), attrs0,
                         "best_candidate must add no hidden state")


class TestClassifyGates(unittest.TestCase):
    """Pure gate logic on constructed candidates (each gate individually).
    Present-but-failed is AMBIGUOUS — never NEUTRAL, never MATCH."""

    def setUp(self):
        self.cfg = make_cfg()

    def fc(self, **kw):
        base = dict(point=utils.Point2D(100.0, 100.0), rotation_deg=0.0, scale=1.0,
                    n_inliers=20, inlier_ratio=0.9, residual=0.5, cue="orb-feature",
                    n_query_desc=50, n_matches=40, n_good=25)
        base.update(kw)
        return R.FeatureCandidate(**base)

    def tc(self, **kw):
        base = dict(point=utils.Point2D(100.0, 100.0), raw_score=0.9,
                    second_peak_score=0.3, scale=1.0, cue="template")
        base.update(kw)
        return R.RawCandidate(**base)

    def test_feature_gates(self):
        C = self.cfg
        cases = [
            (self.fc(n_inliers=7), 0.9, R.Identity.AMBIGUOUS, "feature:inliers-low"),
            (self.fc(inlier_ratio=0.44), 0.9, R.Identity.AMBIGUOUS, "feature:inlier-ratio-low"),
            (self.fc(residual=2.5), 0.9, R.Identity.AMBIGUOUS, "feature:residual-high"),
            (self.fc(residual=None), 0.9, R.Identity.AMBIGUOUS, "feature:residual-high"),
            (self.fc(scale=2.5), 0.9, R.Identity.AMBIGUOUS, "feature:scale-out-of-envelope"),
            (self.fc(scale=0.4), 0.9, R.Identity.AMBIGUOUS, "feature:scale-out-of-envelope"),
            (self.fc(), 0.2, R.Identity.AMBIGUOUS, "feature:confirm-low"),
            (self.fc(), 0.9, R.Identity.MATCH, "feature:all-gates-pass"),
        ]
        for cand, confirm, want_id, want_reason in cases:
            got_id, got_reason = R._classify_feature(cand, confirm, C)
            self.assertIs(got_id, want_id, got_reason)
            self.assertEqual(got_reason, want_reason)

    def test_template_gates(self):
        C = self.cfg
        cases = [
            (self.tc(raw_score=0.5), 0.9, R.Identity.AMBIGUOUS, "template:raw-low"),
            (self.tc(raw_score=0.9, second_peak_score=0.85), 0.9,
             R.Identity.AMBIGUOUS, "template:margin-low"),
            (self.tc(), 0.3, R.Identity.AMBIGUOUS, "template:confirm-low"),
            (self.tc(), 0.9, R.Identity.MATCH, "template:all-gates-pass"),
        ]
        for cand, confirm, want_id, want_reason in cases:
            got_id, got_reason = R._classify_template(cand, confirm, C)
            self.assertIs(got_id, want_id, got_reason)
            self.assertEqual(got_reason, want_reason)


class TestHypothesisTracker(unittest.TestCase):
    """Option-C persistence with bounded NEUTRAL gap and ONE-SHOT acceptance."""

    def setUp(self):
        self.cfg = make_cfg()          # PERSIST_N=3, MAX_NEUTRAL=2
        self.ht = R.HypothesisTracker(self.cfg)

    def m(self, x=100.0, y=100.0, scale=1.0, cue="orb-feature", confirm=0.9):
        return R.ReacqResult(R.Identity.MATCH, utils.Point2D(x, y), scale, cue,
                             confirm, "test")

    def amb(self):
        return R.ReacqResult(R.Identity.AMBIGUOUS, utils.Point2D(0.0, 0.0), 1.0,
                             "orb-feature", 0.1, "test")

    def neu(self):
        return R.ReacqResult(R.Identity.NEUTRAL, None, None, None, None, "test")

    # -- the five required one-shot proofs ---------------------------------- #
    def test_1_no_emission_before_N(self):
        self.assertIsNone(self.ht.update(self.m(100, 100)))
        self.assertIsNone(self.ht.update(self.m(103, 101)))
        self.assertEqual(self.ht.streak, 2)
        self.assertFalse(self.ht.accepted)

    def test_2_exactly_one_emission_at_N(self):
        self.ht.update(self.m(100, 100))
        self.ht.update(self.m(103, 101))
        acc = self.ht.update(self.m(105, 99))
        self.assertIsInstance(acc, R.AcceptedHypothesis)
        self.assertEqual(acc.streak, 3)
        self.assertAlmostEqual(acc.point.x, 105.0)

    def test_3_no_repeated_emission_after_N(self):
        for i in range(3):
            out = self.ht.update(self.m(100 + i, 100))
        self.assertIsNotNone(out)
        for i in range(5):                       # N+1 .. N+5: latched, no re-emit
            self.assertIsNone(self.ht.update(self.m(104 + i, 100)))
        self.assertTrue(self.ht.accepted)
        self.assertEqual(self.ht.streak, 8)

    def test_4_incompatible_hypothesis_can_later_accumulate_and_emit_once(self):
        for i in range(3):
            self.ht.update(self.m(100 + i, 100))
        self.assertTrue(self.ht.accepted)
        # far-away MATCH replaces (streak=1, latch re-armed):
        self.assertIsNone(self.ht.update(self.m(400, 300)))
        self.assertEqual(self.ht.streak, 1)
        self.assertFalse(self.ht.accepted)
        self.assertIsNone(self.ht.update(self.m(402, 301)))
        acc = self.ht.update(self.m(404, 299))
        self.assertIsInstance(acc, R.AcceptedHypothesis)
        self.assertIsNone(self.ht.update(self.m(405, 299)), "one-shot after replacement too")

    def test_5_ambiguous_and_expired_neutral_reset_the_one_shot(self):
        for i in range(3):
            self.ht.update(self.m(100 + i, 100))
        self.assertTrue(self.ht.accepted)
        self.ht.update(self.amb())               # clears hypothesis AND latch
        self.assertFalse(self.ht.has_hypothesis)
        self.assertFalse(self.ht.accepted)
        outs = [self.ht.update(self.m(100 + i, 100)) for i in range(3)]
        self.assertEqual([o is not None for o in outs], [False, False, True],
                         "after AMBIGUOUS reset, acceptance re-emits exactly once at N")
        # expired NEUTRAL gap also resets:
        for _ in range(self.cfg.REACQ_PERSIST_MAX_NEUTRAL + 1):
            self.ht.update(self.neu())
        self.assertFalse(self.ht.has_hypothesis)
        self.assertFalse(self.ht.accepted)
        outs = [self.ht.update(self.m(200 + i, 200)) for i in range(3)]
        self.assertEqual([o is not None for o in outs], [False, False, True])

    # -- bounded NEUTRAL and accumulation rules ----------------------------- #
    def test_neutral_within_bound_freezes_streak(self):
        self.ht.update(self.m(100, 100))
        self.ht.update(self.m(102, 100))
        for _ in range(self.cfg.REACQ_PERSIST_MAX_NEUTRAL):   # gap = 2 (allowed)
            self.assertIsNone(self.ht.update(self.neu()))
        self.assertEqual(self.ht.streak, 2, "frozen, not cleared")
        acc = self.ht.update(self.m(104, 101))
        self.assertIsInstance(acc, R.AcceptedHypothesis,
                              "MATCH after a bounded gap continues the streak")

    def test_neutral_beyond_bound_clears(self):
        self.ht.update(self.m(100, 100))
        self.ht.update(self.m(102, 100))
        for _ in range(self.cfg.REACQ_PERSIST_MAX_NEUTRAL + 1):
            self.ht.update(self.neu())
        self.assertFalse(self.ht.has_hypothesis)
        self.assertIsNone(self.ht.update(self.m(104, 101)))
        self.assertIsNone(self.ht.update(self.m(105, 101)))
        self.assertEqual(self.ht.streak, 2,
                         "two MATCHes must NOT have accumulated across the long gap")

    def test_unrelated_candidates_never_accumulate(self):
        for i in range(6):                        # alternate two far-apart points
            pt = (100.0, 100.0) if i % 2 == 0 else (400.0, 300.0)
            self.assertIsNone(self.ht.update(self.m(*pt)))
            self.assertEqual(self.ht.streak, 1)

    def test_scale_jump_replaces(self):
        self.ht.update(self.m(100, 100, scale=1.0))
        self.ht.update(self.m(101, 100, scale=1.0))
        self.assertIsNone(self.ht.update(self.m(102, 100, scale=1.5)))
        self.assertEqual(self.ht.streak, 1, "scale-incompatible MATCH must replace")

    def test_cue_switch_replaces(self):
        self.ht.update(self.m(100, 100, cue="orb-feature"))
        self.assertIsNone(self.ht.update(self.m(101, 100, cue="template")))
        self.assertEqual(self.ht.streak, 1)

    def test_neutral_without_hypothesis_is_noop(self):
        for _ in range(5):
            self.assertIsNone(self.ht.update(self.neu()))
        self.assertFalse(self.ht.has_hypothesis)

    def test_decision_flag_default_on(self):
        # G1 (2026-07-08, user-approved): reacquisition is the delivered
        # default, enabled on the 9-pick acceptance grid + synthetic GT +
        # false-lock-floor evidence; --no-reacq restores terminal-LOST.
        from ground_target_tracking import config as prod_config
        self.assertTrue(prod_config.REACQ_DECISION_ENABLED,
                        "reacquisition is the delivered default since G1")


if __name__ == "__main__":
    unittest.main()
