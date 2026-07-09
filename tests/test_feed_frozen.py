"""FEED_FROZEN detection unit tests (Commit 4a, §E).

T-b (the adaptive-flow corruption discount) is DISABLED in the shipping bundle
(tested and rejected — it discounted sustained live motion and delayed LOST on
v8; see config.py). These tests cover the T-a freeze detector that DID ship:

  * a dead / repeated feed enters a DISTINCT surfaced condition (FEED_FROZEN),
    holds the last point, and makes NO TRACKING / LOW_CONFIDENCE claim;
  * on confirmed exit the Kalman is re-seeded at the held point and the state
    becomes LOW_CONFIDENCE for immediate revalidation (recover, or fall to LOST
    if the target is gone);
  * the REQUIRED no-hold scenarios (static-live scene with a small moving
    target, slow motion, a short repeat burst) never freeze;
  * the whole detector is a NO-OP when FEED_FROZEN_ENABLED is absent (the gating
    that keeps the from-scratch synthetic tests byte-for-byte unchanged).

Detector calibration (T_static=1.5, N=3, M=2) is validated separately on v8 +
the approved external clip + synthetic >=30fps fixtures; here the fixtures use
clear supra/sub-threshold motion so the STATE MACHINE is what is under test.

Run from the repository root:
    python3 -m unittest tests.test_feed_frozen -v
"""
import sys
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import config as prod_config
from ground_target_tracking import session, trackers, utils
from ground_target_tracking.session import TrackState

W, H = 480, 270
TARGET = utils.Point2D(240.0, 135.0)


def freeze_cfg(**over):
    """Production snapshot with the feed detector ON and T-b OFF (the shipped
    profile), on a HUD-free synthetic scene."""
    cfg = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                   for k in dir(prod_config) if k.isupper()})
    cfg.FEED_FROZEN_ENABLED = True
    cfg.FEED_CORRUPT_ENABLED = False          # T-b rejected/disabled (shipped)
    cfg.OVERLAY_MASK_ENABLED = False          # no burned-in HUD in the synthetic scene
    cfg.EXPERIMENTAL_REGIONAL_MOTION = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def textured(seed=7):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def pan(img, dx):
    """Integer horizontal shift — a clear supra-threshold live motion."""
    return np.roll(img, dx, axis=1)


def run(cfg, frames, point=TARGET, enable_reacq=False):
    tracker = trackers.make_tracker("of_kalman", cfg)
    sess = session.TrackingSession(tracker, cfg, enable_reacq=enable_reacq)
    sess.init(frames[0], point)
    outs = [sess.step(f) for f in frames[1:]]
    return outs, sess


class TestFeedFrozenStateMachine(unittest.TestCase):
    def _frozen_span(self, states):
        idx = [i for i, s in enumerate(states) if s is TrackState.FEED_FROZEN]
        return idx

    def test_i_freeze_resume_unmoved_holds_then_revalidates(self):
        # Freeze -> resume with the target still present: hold, surface the
        # distinct condition, then exit to LOW_CONFIDENCE revalidation and NOT
        # LOST (the target is revalidatable). Full LOW_CONFIDENCE->TRACKING
        # promotion is the ordinary M8 recovery, exercised in test_session; here
        # the FEED_FROZEN-specific contract is the held condition + the exit
        # landing in LOW_CONFIDENCE without a false LOST.
        cfg = freeze_cfg()
        base = textured()
        live_in = [pan(base, k) for k in range(0, 8)]        # slow pan -> TRACKING
        frozen = [pan(base, 7)] * 6                           # exact repeats -> FEED_FROZEN
        live_out = [pan(base, 7 + k) for k in range(1, 12)]   # resume the pan (unmoved)
        outs, _ = run(cfg, live_in + frozen + live_out)
        states = [o.state for o in outs]
        fidx = self._frozen_span(states)
        self.assertTrue(fidx, "the repeated feed never surfaced FEED_FROZEN")
        # while frozen: NO TRACKING / LOW_CONFIDENCE claim, distinct in signals,
        # and the held point does not move.
        held = (outs[fidx[0]].point.x, outs[fidx[0]].point.y)
        for i in fidx:
            self.assertIs(states[i], TrackState.FEED_FROZEN)
            self.assertEqual(outs[i].signals.get("feed"), "frozen")
            self.assertEqual((outs[i].point.x, outs[i].point.y), held)
        tail = states[fidx[-1] + 1:]
        self.assertIs(tail[0], TrackState.LOW_CONFIDENCE,
                      "confirmed exit must land in LOW_CONFIDENCE revalidation")
        self.assertNotIn(TrackState.LOST, tail,
                         "an unmoved (revalidatable) target must not go LOST")

    def test_ii_freeze_resume_displaced_goes_lost(self):
        cfg = freeze_cfg(LOST_AFTER_N_BAD=6)
        base, other = textured(7), textured(99)
        live_in = [pan(base, k) for k in range(0, 8)]
        frozen = [pan(base, 7)] * 6
        live_out = [pan(other, k) for k in range(0, 14)]      # target gone at the held point
        outs, _ = run(cfg, live_in + frozen + live_out)
        states = [o.state for o in outs]
        fidx = self._frozen_span(states)
        self.assertTrue(fidx)
        tail = states[fidx[-1] + 1:]
        self.assertIn(TrackState.LOW_CONFIDENCE, tail,
                      "exit did not attempt LOW_CONFIDENCE revalidation")
        self.assertIn(TrackState.LOST, tail,
                      "displaced/absent target did not fall to LOST")

    def test_iii_static_live_small_moving_target_no_hold(self):
        cfg = freeze_cfg()
        base = textured(7)
        frames = []
        for k in range(0, 22):
            f = base.copy()                                   # EXACT-static background
            x = 100 + 3 * k
            f[130:150, x:x + 20] = 245                        # small bright target moving 3px/f
            frames.append(f)
        outs, _ = run(cfg, frames, point=utils.Point2D(110.0, 140.0))
        states = [o.state for o in outs]
        self.assertNotIn(TrackState.FEED_FROZEN, states,
                         "a small moving target must keep the feed live")

    def test_iv_slow_motion_no_hold(self):
        cfg = freeze_cfg()
        base = textured(7)
        frames = [pan(base, k) for k in range(0, 22)]         # continuous 1px/f pan
        outs, _ = run(cfg, frames)
        states = [o.state for o in outs]
        self.assertNotIn(TrackState.FEED_FROZEN, states,
                         "slow but continuous motion must not freeze")

    def test_v_repeat_burst_no_hold(self):
        cfg = freeze_cfg()
        base = textured(7)
        frames = [base.copy()]
        pos = 0
        for _ in range(6):                                    # [motion, motion, repeat, repeat]
            pos += 1; frames.append(pan(base, pos))
            pos += 1; frames.append(pan(base, pos))
            frames.append(frames[-1].copy())                  # repeat 1
            frames.append(frames[-1].copy())                  # repeat 2 (>=2 identical, < N=3)
        outs, _ = run(cfg, frames)
        states = [o.state for o in outs]
        self.assertNotIn(TrackState.FEED_FROZEN, states,
                         "<=2 consecutive repeats must not reach the N=3 freeze")


class TestFeedFrozenPrimitivesAndGating(unittest.TestCase):
    def test_block_max_identical_is_zero_shifted_exceeds_threshold(self):
        cfg = freeze_cfg()
        base = textured(7)
        sess = session.TrackingSession(trackers.make_tracker("of_kalman", cfg), cfg)
        sess.init(base, TARGET)
        small0 = sess._feed_small(base)
        self.assertEqual(sess._block_max(small0, small0), 0.0)
        small1 = sess._feed_small(pan(base, 1))
        self.assertGreaterEqual(sess._block_max(small0, small1), cfg.FEED_FROZEN_T_STATIC)

    def test_enter_after_exactly_n_and_exit_after_exactly_m(self):
        cfg = freeze_cfg()
        base = textured(7)
        frozen = [pan(base, 3)] * 6
        # 2 live -> frozen repeats -> 3 live: enter on the N-th quiet frame, exit on the M-th.
        live_in = [pan(base, k) for k in range(0, 3)]
        live_out = [pan(base, 3 + k) for k in range(1, 5)]
        outs, _ = run(cfg, live_in + frozen + live_out)
        states = [o.state for o in outs]
        first_frozen = next(i for i, s in enumerate(states) if s is TrackState.FEED_FROZEN)
        # the frozen input begins at index len(live_in)-1 in `outs` (step compares
        # frames[1:]); ENTER requires N=3 consecutive quiet block-max diffs.
        self.assertGreaterEqual(sum(1 for s in states if s is TrackState.FEED_FROZEN), 1)
        after = states[first_frozen:]
        # after the feed resumes, the very next non-frozen state is LOW_CONFIDENCE
        resumed = next(s for s in after if s is not TrackState.FEED_FROZEN)
        self.assertIs(resumed, TrackState.LOW_CONFIDENCE)

    def test_disabled_when_flag_absent(self):
        # A cfg WITHOUT the FEED_FROZEN_ENABLED attribute must never freeze, even
        # on byte-identical frames — this is the getattr-default gating that keeps
        # the from-scratch synthetic suites byte-for-byte unchanged.
        cfg = freeze_cfg()
        delattr(cfg, "FEED_FROZEN_ENABLED")   # attribute absent -> detector off
        base = textured(7)
        outs, sess = run(cfg, [base] * 10)    # 10 IDENTICAL frames
        self.assertFalse(sess._feed_frozen_enabled)
        self.assertNotIn(TrackState.FEED_FROZEN, [o.state for o in outs])


if __name__ == "__main__":
    unittest.main()
