"""Unit tests for the Stage-0 position-accuracy evaluation harness (stdlib only).

Self-contained and synthetic (no video files): they validate the MEASUREMENT
math, not tracker quality. Real-video measurement lives in eval_position.py.

Critical regression coverage: the AnnulusReference must NEVER seed anchors on the
HUD overlay (white cursor / black X / letterbox) — proven by
`test_annulus_excludes_overlay`.

Run from the repository root:
    python3 -m unittest discover tests -v
"""
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import config as gconfig
from ground_target_tracking import evaluation, utils


def _textured(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Trackable BGR texture: blurred random noise (lots of stable corners)."""
    rng = np.random.RandomState(seed)
    g = rng.randint(0, 256, (h, w), np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class TestSyntheticGroundTruth(unittest.TestCase):
    def test_true_point_follows_known_translation(self):
        base = _textured(300, 300, seed=1)
        pt = utils.Point2D(150, 150)
        seq = evaluation.synthetic_gt_sequence(
            base, pt, evaluation.translation_ramp(10, 4.0, 2.0))
        self.assertEqual(len(seq), 10)
        _, last_true = seq[-1]
        # frame t=10 shifted the content by (40, 20); true point moves with it.
        self.assertAlmostEqual(last_true.x, 150 + 40, delta=0.01)
        self.assertAlmostEqual(last_true.y, 150 + 20, delta=0.01)

    def test_perfect_committed_gives_zero_error(self):
        base = _textured(200, 200, seed=2)
        pt = utils.Point2D(100, 100)
        seq = evaluation.synthetic_gt_sequence(
            base, pt, evaluation.translation_ramp(15, 3.0, -1.0))
        acc = evaluation.PositionAccuracy(tol_px=2.0)
        for t, (_, true_pt) in enumerate(seq, start=1):
            acc.add(t, true_pt, true_pt, "TRACKING", "measure")  # perfect tracker
        s = acc.summary()
        self.assertEqual(s["err_max"], 0.0)
        self.assertEqual(s["accepted_wrong"], 0)
        self.assertEqual(s["accepted_on_target"], 15)


class TestAnnulusReference(unittest.TestCase):
    def test_recovers_known_translation(self):
        base = _textured(400, 400, seed=3)
        pt = utils.Point2D(200, 200)
        ref = evaluation.AnnulusReference(base, pt, gconfig)
        self.assertTrue(ref.ok0)
        self.assertGreaterEqual(len(ref.anchors0), 4)
        # translate the whole scene by a known (dx, dy)
        dx, dy = 6.0, 3.0
        M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
        shifted = cv2.warpAffine(base, M, (400, 400), borderMode=cv2.BORDER_REFLECT)
        est, alive = ref.update(shifted)
        self.assertIsNotNone(est)
        self.assertGreaterEqual(alive, 4)
        self.assertAlmostEqual(est.x, 200 + dx, delta=3.0)
        self.assertAlmostEqual(est.y, 200 + dy, delta=3.0)

    def test_thin_support_returns_none(self):
        flat = np.zeros((200, 200, 3), np.uint8)  # no corners -> no anchors
        ref = evaluation.AnnulusReference(flat, utils.Point2D(100, 100), gconfig)
        est, alive = ref.update(flat)
        self.assertIsNone(est)
        self.assertEqual(alive, 0)


class TestHudExclusionRegression(unittest.TestCase):
    """The reference must not use HUD pixels (cursor / X / letterbox) as evidence."""

    def setUp(self):
        self._prev = gconfig.OVERLAY_MASK_ENABLED
        gconfig.OVERLAY_MASK_ENABLED = True
        self.addCleanup(setattr, gconfig, "OVERLAY_MASK_ENABLED", self._prev)

    def test_annulus_excludes_overlay(self):
        base = _textured(600, 800, seed=4)  # bright everywhere -> no letterbox
        # point offset from centre so the 221px ROI overlaps the X + crosshair.
        pt = utils.Point2D(520, 300)
        ref = evaluation.AnnulusReference(base, pt, gconfig)
        # non-vacuous: there really were overlay pixels inside the seeding ROI...
        self.assertGreater(ref.overlay_px_in_roi, 0)
        # ...and NONE of the seeded anchors landed on the overlay.
        self.assertEqual(ref.anchors_on_overlay(), 0)
        self.assertTrue(ref.ok0)  # still found clean scene anchors to track


class TestPositionAccuracy(unittest.TestCase):
    def test_flags_accepted_wrong(self):
        acc = evaluation.PositionAccuracy(tol_px=10.0)
        # accepted a point 141px from the reference -> accepted_wrong
        acc.add(1, utils.Point2D(100, 100), utils.Point2D(200, 200),
                "TRACKING", "measure")
        # accepted a point 1.4px from the reference -> accepted, on target
        acc.add(2, utils.Point2D(100, 100), utils.Point2D(101, 101),
                "LOW_CONFIDENCE", "measure")
        # a predict frame is not an accepted measurement
        acc.add(3, utils.Point2D(100, 100), utils.Point2D(300, 300),
                "PREDICT", "predict")
        s = acc.summary()
        self.assertEqual(s["accepted"], 2)
        self.assertEqual(s["accepted_wrong"], 1)
        self.assertEqual(s["accepted_on_target"], 1)
        self.assertEqual(s["accepted_wrong_frames"], [1])
        self.assertEqual(s["rejected_or_predict"], 1)

    def test_none_reference_is_not_counted(self):
        acc = evaluation.PositionAccuracy(tol_px=5.0)
        acc.add(1, utils.Point2D(0, 0), None, "TRACKING", "measure")
        s = acc.summary()
        self.assertEqual(s["frames"], 1)
        self.assertEqual(s["frames_with_ref"], 0)
        self.assertIsNone(s["err_max"])


class TestWallClock(unittest.TestCase):
    def test_reports_fps(self):
        clk = evaluation.WallClock()
        for _ in range(3):
            clk.tic()
            clk.toc()
        s = clk.summary()
        self.assertEqual(s["n_frames"], 3)
        self.assertGreater(s["end_to_end_fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
