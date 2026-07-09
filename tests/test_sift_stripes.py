"""Unit tests for the G2-a budgeted SIFT stripe slicing (2026-07-08).

The whole-frame SIFT route measured 110-350 ms per tick at 1080p (and the
one-time reference build 100-240 ms) — over the 33.3 ms real-time budget
while LOST. G2-a slices the reference build into FULL-RESOLUTION stripes
(measured identity-lossless) and the query into per-tick stripes at
REACQ_SIFT_DETECT_SCALE, fitting once per completed sweep through the same
match core as the whole-frame surface. These tests pin the stripe geometry,
the builder/sweep contracts, and the striped-vs-whole-frame equivalence.
"""
import sys
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import config as prod_config, utils
from ground_target_tracking import reacquisition as R


def make_cfg(**over):
    cfg = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                   for k in dir(prod_config) if k.isupper()})
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def textured(h, w, seed=7):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(
        rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (3, 3), 0)


class TestStripeBounds(unittest.TestCase):
    def test_cores_tile_height_exactly(self):
        for h, n, ov in ((1080, 8, 48), (480, 8, 48), (200, 3, 10), (9, 4, 2)):
            b = R.Reacquirer._stripe_bounds(h, n, ov)
            self.assertEqual(len(b), n)
            self.assertEqual(b[0][2], 0, "first core starts at row 0")
            self.assertEqual(b[-1][3], h, "last core ends at the height")
            for (_, _, _, c1), (_, _, c0n, _) in zip(b, b[1:]):
                self.assertEqual(c1, c0n, "cores tile without gap/overlap")

    def test_overlap_extends_detection_windows(self):
        b = R.Reacquirer._stripe_bounds(400, 4, 20)
        for i, (y0, y1, c0, c1) in enumerate(b):
            if i > 0:
                self.assertEqual(y0, c0 - 20, "leading overlap")
            if i < len(b) - 1:
                self.assertEqual(y1, c1 + 20, "trailing overlap")
            self.assertGreaterEqual(y1, c1)
            self.assertLessEqual(y0, c0)


class TestStripedBuilder(unittest.TestCase):
    def test_striped_build_matches_whole_frame_reference(self):
        frame = textured(480, 640)
        pt = utils.Point2D(320.0, 240.0)
        whole = R.Reacquirer(make_cfg())
        ref_w = whole.build_lc_reference(frame, pt, None)
        self.assertIsNotNone(ref_w)

        striped = R.Reacquirer(make_cfg())
        builder = striped.new_lc_builder(frame, pt, None)
        n = int(striped.cfg.REACQ_SIFT_STRIPES)
        for _ in range(n - 1):
            self.assertFalse(builder.step(), "not done before the final stripe")
            self.assertIsNone(striped.lc_reference)
        self.assertTrue(builder.step(), "final stripe completes the build")
        ref_s = striped.lc_reference
        self.assertIsNotNone(ref_s)
        # Identity-lossless within a small border-effect tolerance (measured
        # 308 vs 307 disc keypoints on the v8 snapshot).
        nw, ns = len(ref_w.kp_xy), len(ref_s.kp_xy)
        self.assertLessEqual(abs(ns - nw), max(3, int(0.15 * nw)),
                             f"striped {ns} vs whole {nw} keypoints")

    def test_flat_snapshot_fails_the_floor(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        striped = R.Reacquirer(make_cfg())
        builder = striped.new_lc_builder(frame, utils.Point2D(320.0, 240.0),
                                         None)
        while not builder.step():
            pass
        self.assertIsNone(striped.lc_reference,
                          "flat snapshot cannot clear the keypoint floor")

    def test_mask_filters_reference_keypoints(self):
        frame = textured(480, 640, seed=11)
        pt = utils.Point2D(320.0, 240.0)
        mask = np.zeros((480, 640), np.uint8)
        mask[:, 320:] = 255                     # right half is "HUD"
        striped = R.Reacquirer(make_cfg())
        builder = striped.new_lc_builder(frame, pt, mask)
        while not builder.step():
            pass
        ref = striped.lc_reference
        if ref is not None:                     # floor may or may not survive
            self.assertTrue(np.all(ref.kp_xy[:, 0] < 320),
                            "no reference keypoint on the masked half")


class TestLcSweep(unittest.TestCase):
    def test_observation_only_on_final_stripe_and_self_match(self):
        frame = textured(480, 640)
        pt = utils.Point2D(320.0, 240.0)
        rq = R.Reacquirer(make_cfg())
        self.assertIsNotNone(rq.build_lc_reference(frame, pt, None))
        n = int(rq.cfg.REACQ_SIFT_STRIPES)
        for i in range(n - 1):
            self.assertIsNone(rq.lc_sweep_step(frame),
                              f"stripe {i} must not yield an observation")
        rr = rq.lc_sweep_step(frame)
        self.assertIsNotNone(rr, "final stripe yields the sweep observation")
        self.assertIs(rr.identity, R.Identity.MATCH,
                      f"self-match must pass the gates ({rr.reason})")
        self.assertLess(abs(rr.point.x - pt.x) + abs(rr.point.y - pt.y), 4.0,
                        "self-match transports the reference point")
        self.assertAlmostEqual(rr.scale, 1.0, delta=0.05)
        # The sweep restarts: the next call is stripe 0 again (no observation).
        self.assertIsNone(rq.lc_sweep_step(frame))

    def test_no_reference_is_immediate_neutral(self):
        rq = R.Reacquirer(make_cfg())
        rr = rq.lc_sweep_step(textured(480, 640))
        self.assertIsNotNone(rr)
        self.assertIs(rr.identity, R.Identity.NEUTRAL)
        self.assertEqual(rr.reason, "sift:no-reference")


if __name__ == "__main__":
    unittest.main()
