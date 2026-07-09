"""Phase-2 bounded SEARCHING scheduler — deterministic tests.

Covers the approved R2 (top-3 spatially-distinct candidate list + TTL +
rejected-site cooldown), A2 (fresh live re-search, age-scaled capped window)
and S1 (verify-first + bounded global-scan progress, single verify-only
fallback frame) contracts, plus the session integration invariants:
scheduler state resets between recovery episodes, the feature-reference path
is unchanged, and the old one-frame full-frame five-scale sweep (propose /
best_candidate) is unreachable from template-path SEARCHING.

Synthetic scenes only (seeded noise textures, no-overlay core path); no test
reads or calibrates anything from v8. Timing-related assertions check only
the WORK/instrumentation contract (bounded units, overrun flag present),
never wall-clock values — work is deterministically bounded, wall-clock is
merely measured.
"""
import sys
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")
from ground_target_tracking import config as prod_config
from ground_target_tracking import reacquisition as R
from ground_target_tracking import trackers, utils
from ground_target_tracking.session import TrackingSession, TrackState


def snapshot_cfg(**overrides):
    """Production-config snapshot as a namespace (tests never mutate the
    shared config module)."""
    ns = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                  for k in dir(prod_config) if k.isupper()})
    ns.OVERLAY_MASK_ENABLED = False            # no-overlay core path
    ns.EXPERIMENTAL_REGIONAL_MOTION = False
    ns.FEED_FROZEN_ENABLED = False             # deterministic-seed (identical) synthetic
                                               # frames (not a real feed) -> detector off
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def template_cfg(**overrides):
    """Config forcing the descriptor-free TEMPLATE path (huge keypoint floor
    -> has_descriptors False; context capability intact)."""
    return snapshot_cfg(REACQ_MIN_REF_KP=10 ** 6, **overrides)


def _textured(h, w, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


REF_POINT = utils.Point2D(320.0, 240.0)


def make_scheduler(cfg, seed=3):
    """Real Reacquirer + reference on a textured init frame; returns
    (scheduler, reacquirer, init_frame, BGR context window to paste)."""
    frame0 = _textured(480, 640, seed)
    rq = R.Reacquirer(cfg)
    ref = rq.build_reference(frame0, REF_POINT)
    assert ref.has_context and not ref.has_descriptors
    x0, y0, x1, y1 = utils.clamp_roi(REF_POINT.x, REF_POINT.y,
                                     int(cfg.REACQ_TEMPLATE_SIZE), 640, 480)
    patch = frame0[y0:y1, x0:x1].copy()
    return R.TemplateScanScheduler(rq, cfg), rq, frame0, patch


def paste(frame, patch, x, y, scale=1.0, blur=0):
    """Paste (optionally rescaled/blurred) reference content at top-left
    (x, y); returns the expected transported candidate point."""
    p = patch if scale == 1.0 else cv2.resize(patch, None, fx=scale, fy=scale,
                                              interpolation=cv2.INTER_AREA)
    if blur:
        p = cv2.GaussianBlur(p, (blur, blur), 0)
    h, w = p.shape[:2]
    frame[y:y + h, x:x + w] = p
    half = patch.shape[0] / 2.0
    return utils.Point2D(x + half * scale, y + half * scale)


class TestCandidateList(unittest.TestCase):
    """R2 policy at the _add/_expire level (pure list logic)."""

    def setUp(self):
        self.cfg = template_cfg()
        self.sched, self.rq, self.frame0, self.patch = make_scheduler(self.cfg)
        self.sched._frame_idx = 10

    def _pt(self, x, y):
        return utils.Point2D(float(x), float(y))

    def test_nearby_duplicates_collapse_stronger_kept(self):
        self.assertTrue(self.sched._add(self._pt(100, 100), 1.0, 0.7))
        self.assertTrue(self.sched._add(self._pt(130, 100), 1.0, 0.9))  # same site
        self.assertEqual(len(self.sched._pending), 1)
        self.assertAlmostEqual(self.sched._pending[0].ncc, 0.9)
        self.assertAlmostEqual(self.sched._pending[0].point.x, 130.0)
        # weaker duplicate does NOT displace the stronger observation
        self.sched._add(self._pt(110, 100), 1.0, 0.65)
        self.assertAlmostEqual(self.sched._pending[0].ncc, 0.9)

    def test_queue_bounded_and_replacement(self):
        for i, ncc in enumerate((0.70, 0.80, 0.90)):
            self.sched._add(self._pt(50 + 200 * i, 100), 1.0, ncc)
        self.assertEqual(len(self.sched._pending), int(self.cfg.REACQ_CAND_QUEUE))
        # weaker distinct newcomer refused
        self.assertFalse(self.sched._add(self._pt(50, 400), 1.0, 0.65))
        self.assertEqual(len(self.sched._pending), 3)
        # stronger distinct newcomer replaces the weakest (0.70)
        self.assertTrue(self.sched._add(self._pt(50, 400), 1.0, 0.75))
        self.assertEqual(len(self.sched._pending), 3)
        self.assertNotIn(0.70, [c.ncc for c in self.sched._pending])

    def test_ttl_expiry(self):
        self.sched._add(self._pt(100, 100), 1.0, 0.9)
        self.sched._frame_idx += int(self.cfg.REACQ_CAND_TTL_FRAMES) + 1
        self.sched._expire()
        self.assertEqual(len(self.sched._pending), 0)

    def test_rejected_site_cooldown_blocks_reenqueue(self):
        exp = self.sched._frame_idx + int(self.cfg.REACQ_REJECT_COOLDOWN_FRAMES)
        self.sched._cooldown.append((100.0, 100.0, exp))
        self.assertFalse(self.sched._add(self._pt(120, 100), 1.0, 0.95))
        self.assertEqual(len(self.sched._pending), 0)
        # distinct site unaffected
        self.assertTrue(self.sched._add(self._pt(400, 300), 1.0, 0.8))
        # cooldown expires -> site usable again
        self.sched._frame_idx = exp
        self.sched._expire()
        self.assertTrue(self.sched._add(self._pt(120, 100), 1.0, 0.95))


class TestSchedulerFrames(unittest.TestCase):
    """step()-level behavior on synthetic scenes (real primitives)."""

    def setUp(self):
        self.cfg = template_cfg()
        self.sched, self.rq, self.frame0, self.patch = make_scheduler(self.cfg)
        self.bg = _textured(480, 640, seed=99)      # background w/o the target

    def test_two_candidates_one_frame_both_retained(self):
        # 0.6-scaled pastes in stripes 0 and 1 of the FIRST ladder scale ->
        # both discovered by frame 1's three units, both retained.
        f = self.bg.copy()
        paste(f, self.patch, 60, 30, scale=0.6)
        paste(f, self.patch, 420, 100, scale=0.6)
        rr, diag = self.sched.step(f)
        self.assertEqual(diag["scan_mode"], "SCAN")
        self.assertEqual(diag["scan_pending"], 2)
        self.assertIs(rr.identity, R.Identity.NEUTRAL)

    def test_strongest_false_rejected_then_second_verified(self):
        f = self.bg.copy()
        pa = paste(f, self.patch, 60, 30, scale=0.6)             # strong (false)
        pb = paste(f, self.patch, 420, 100, scale=0.6, blur=3)   # weaker (true)
        orig = self.rq.verify_at
        self.rq.verify_at = (lambda fr, pt:
                             0.0 if pt.dist(pa) < 60 else orig(fr, pt))
        self.sched.step(f)                                        # discover both
        self.assertEqual(len(self.sched._pending), 2)
        self.assertGreater(max(c.ncc for c in self.sched._pending),
                           min(c.ncc for c in self.sched._pending))
        rr2, diag2 = self.sched.step(f)                           # verify strongest
        self.assertEqual(diag2["scan_mode"], "VERIFY")
        self.assertIs(rr2.identity, R.Identity.AMBIGUOUS)
        self.assertTrue(diag2["scan_cooldown_applied"])
        rr3, diag3 = self.sched.step(f)                           # then the second
        self.assertEqual(diag3["scan_mode"], "VERIFY")
        self.assertIs(rr3.identity, R.Identity.MATCH)
        self.assertLess(rr3.point.dist(pb), 8.0)

    def test_at_most_one_verify_per_frame(self):
        f = self.bg.copy()
        paste(f, self.patch, 60, 30, scale=0.6)
        paste(f, self.patch, 420, 100, scale=0.6)
        calls = []
        orig = self.rq.verify_at
        self.rq.verify_at = lambda fr, pt: calls.append(1) or orig(fr, pt)
        for _ in range(10):
            n0 = len(calls)
            self.sched.step(f)
            self.assertLessEqual(len(calls) - n0, 1)

    def test_verify_uses_fresh_live_research_after_motion(self):
        f1 = self.bg.copy()
        paste(f1, self.patch, 200, 150)                     # discovered here
        # drive scan frames until the site is found
        found = False
        for _ in range(12):
            _, diag = self.sched.step(f1)
            if diag["scan_pending"] > 0:
                found = True
                break
        self.assertTrue(found)
        # target MOVES before the verify frame -> verification must re-search
        # the LIVE frame and match at the moved location, not the stale one
        f2 = self.bg.copy()
        moved = paste(f2, self.patch, 230, 175)             # +30, +25 px
        rr, diag = self.sched.step(f2)
        self.assertEqual(diag["scan_mode"], "VERIFY")
        self.assertIs(rr.identity, R.Identity.MATCH)
        self.assertLess(rr.point.dist(moved), 8.0)

    def test_age_scaled_window_caps_at_max(self):
        f = self.bg.copy()
        target = paste(f, self.patch, 200, 150)
        self.sched.step(f)                                  # calibrate a unit
        self.sched._add(target, 1.0, 0.9)
        self.sched._pending[0].frame_idx = self.sched._frame_idx - 5  # old
        _, diag = self.sched.step(f)
        self.assertEqual(diag["scan_mode"], "VERIFY")
        self.assertEqual(diag["scan_win_px"],
                         int(self.cfg.REACQ_RESEARCH_WIN_CAP_PX))
        # fresh candidate (age <= 1) uses the base window
        self.sched._verify_only_last = False
        self.sched._add(target, 1.0, 0.9)
        self.sched._pending[0].frame_idx = self.sched._frame_idx
        _, diag2 = self.sched.step(f)
        if diag2["scan_mode"] == "VERIFY":                  # (not a forced scan)
            self.assertEqual(diag2["scan_win_px"],
                             int(self.cfg.REACQ_RESEARCH_WIN_PX))

    def test_false_candidates_cannot_stall_coverage(self):
        # every verification refuses (confirm 0.0) -> repeated false peaks;
        # global rolling coverage must still complete a full cycle, with
        # never two consecutive zero-progress frames.
        f = self.bg.copy()
        paste(f, self.patch, 60, 30, scale=0.6)
        paste(f, self.patch, 420, 100, scale=0.6)
        paste(f, self.patch, 200, 300)
        self.rq.verify_at = lambda fr, pt: 0.0
        total = None
        zero_streak_max = 0
        for _ in range(80):
            _, diag = self.sched.step(f)
            total = diag["scan_cycle_total"]
            zero_streak_max = max(zero_streak_max,
                                  diag["scan_no_progress_streak"])
            if diag["scan_cycles_done"] >= 1:
                break
        self.assertGreaterEqual(self.sched._cycles, 1)
        self.assertGreater(total, 0)
        self.assertLessEqual(zero_streak_max, 1)

    def test_full_cycle_completes_and_units_bounded(self):
        # pure background: no pre-gate peaks -> pure scan; the cycle covers
        # every (scale, stripe) unit; per-frame units never exceed the bound.
        for i in range(60):
            _, diag = self.sched.step(self.bg)
            self.assertLessEqual(diag["scan_units"],
                                 int(self.cfg.REACQ_SCAN_UNITS_PER_FRAME))
            self.assertGreaterEqual(diag["scan_units"],
                                    int(self.cfg.REACQ_SCAN_MIN_UNITS))
            if diag["scan_cycles_done"] >= 1:
                break
        self.assertGreaterEqual(self.sched._cycles, 1)
        # 5 fitting scales x REACQ_SCAN_STRIPES stripes at 640x480
        self.assertEqual(diag["scan_cycle_total"],
                         5 * int(self.cfg.REACQ_SCAN_STRIPES))

    def test_verify_only_fallback_never_two_consecutive(self):
        # zero budget forces the verify-only fallback on every verify frame;
        # the scheduler must interleave scan progress (never two consecutive
        # zero-unit frames) and instrument the fallback.
        cfg = template_cfg(REACQ_SCAN_BUDGET_MS=0.0)
        sched, rq, frame0, patch = make_scheduler(cfg)
        f = _textured(480, 640, seed=99)
        paste(f, patch, 200, 150)
        saw_fallback = False
        prev_zero = False
        for _ in range(20):
            _, diag = sched.step(f)
            zero = diag["scan_units"] == 0
            self.assertFalse(prev_zero and zero,
                             "two consecutive zero-progress frames")
            saw_fallback = saw_fallback or diag["scan_verify_only"]
            self.assertTrue(diag["scan_overrun"] or diag["scan_m9_ms"] == 0.0)
            prev_zero = zero
        self.assertTrue(saw_fallback)

    def test_timing_overrun_logged_without_crash(self):
        cfg = template_cfg(REACQ_SCAN_BUDGET_MS=0.0)
        sched, rq, frame0, patch = make_scheduler(cfg)
        rr, diag = sched.step(_textured(480, 640, seed=99))
        self.assertTrue(diag["scan_overrun"])              # logged
        self.assertIs(rr.identity, R.Identity.NEUTRAL)     # and non-fatal


class TestSessionIntegration(unittest.TestCase):
    """Wiring invariants: routing, episode resets, feature path unchanged,
    old full-frame sweep unreachable."""

    def _lost_session(self, cfg, frame0):
        sess = TrackingSession(trackers.make_tracker("of_kalman", cfg), cfg,
                               enable_reacq=True)
        sess.init(frame0, REF_POINT)
        sess.state = TrackState.LOST          # unit-level recovery entry
        return sess

    def test_template_path_uses_scheduler_and_old_sweep_unreachable(self):
        cfg = template_cfg()
        frame0 = _textured(480, 640, seed=3)
        sess = self._lost_session(cfg, frame0)

        def _boom(*a, **k):
            raise AssertionError("old full-frame path reached")

        sess._reacq.propose = _boom            # the five-scale full-frame sweep
        sess._reacq.best_candidate = _boom     # and its decision wrapper
        bg = _textured(480, 640, seed=99)
        for _ in range(5):
            out = sess.step(bg)
        self.assertIsNotNone(sess._scan)
        self.assertEqual(out.state, TrackState.LOST)
        self.assertIn("scan_mode", out.signals)          # instrumentation flows
        self.assertIn("scan_m9_ms", out.signals)

    def test_feature_reference_path_runs_feature_first(self):
        # Interleaved-routing revision: a feature-capable reference now ALSO
        # gets the bounded scan scheduler (the template route is no longer
        # structurally blocked by descriptors), but the first executed tick
        # still runs the feature route exclusively — one route per tick; the
        # scan is reachable only after a feature NEUTRAL (test_reacq_routing).
        cfg = snapshot_cfg()                   # production keypoint floor
        frame0 = _textured(480, 640, seed=3)
        sess = self._lost_session(cfg, frame0)
        if not sess._reacq.reference.has_descriptors:
            self.skipTest("scene yielded no descriptor capability")
        calls = []
        orig = sess._reacq.best_candidate
        sess._reacq.best_candidate = lambda fr: calls.append(1) or orig(fr)
        out = sess.step(_textured(480, 640, seed=99))
        self.assertIsNotNone(sess._scan)       # scheduler built (routing fix)
        self.assertEqual(len(calls), 1)        # feature-first best_candidate
        self.assertEqual(out.signals["reacq_route"], "feature")
        self.assertNotIn("scan_mode", out.signals)  # no scan step this tick

    def test_scheduler_resets_between_episodes(self):
        cfg = template_cfg()
        frame0 = _textured(480, 640, seed=3)
        sess = self._lost_session(cfg, frame0)
        out = sess.step(_textured(480, 640, seed=99))    # enters recovery
        scan1 = sess._scan
        self.assertIsNotNone(scan1)
        scan1._frame_idx = 123                            # mark episode state
        # probation failure -> fresh episode -> fresh scheduler state
        sess._fail_probation(out.result, None, False, "test")
        self.assertIsNot(sess._scan, scan1)
        self.assertEqual(sess._scan._frame_idx, 0)
        # public reset clears it entirely
        sess.reset()
        self.assertIsNone(sess._scan)


if __name__ == "__main__":
    unittest.main()
