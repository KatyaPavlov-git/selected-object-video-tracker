"""Run C — M8 identity-unknown policy (similarity=None) tests.

Contract under test (session.step, live path only):
  * similarity None + INFORMATIVE reference = identity UNKNOWN:
      - can never RESET an existing bad streak (hold);
      - beyond LOST_UNKNOWN_SIM_AFTER_N consecutive frames it becomes loss
        evidence itself -> honest LOST within a bounded number of frames;
  * one brief None frame changes nothing;
  * numeric similarity behavior (good resets / low is bad) is unchanged;
  * a flat/absent reference (low-texture target — identity never measurable)
    keeps FULL legacy None-neutrality (no escalation, resets allowed);
  * known-overlay-occlusion suspension keeps precedence over escalated
    unknown frames (explained unknown != unexplained identity loss);
  * edge penalty and jump veto semantics are untouched.

Similarity is scripted by stubbing session._patch_similarity — the policy is
about how the VALUE is judged, not how it is computed.
"""
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, ".")
from ground_target_tracking import utils
from ground_target_tracking.session import TrackingSession, TrackState

try:
    from test_session import StubTracker, make_cfg, res
except ImportError:
    from tests.test_session import StubTracker, make_cfg, res


N_UNKNOWN = 3          # small structural bound for fast tests
N_BAD = 4              # matches make_cfg().LOST_AFTER_N_BAD


class UnknownSimBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.cfg.LOST_UNKNOWN_SIM_AFTER_N = N_UNKNOWN
        assert self.cfg.LOST_AFTER_N_BAD == N_BAD
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        self.p0 = utils.Point2D(100.0, 100.0)

    def make_session(self, results, sims):
        """Session whose per-frame similarity is scripted (last value repeats).
        The noise init frame gives an informative reference (_ref_std > 0)."""
        sess = TrackingSession(StubTracker(results), self.cfg)
        sess.init(self.frame, self.p0)
        self.assertGreater(sess._ref_std, 1e-6)      # informative reference
        seq = list(sims)
        sess._patch_similarity = (
            lambda frame, point: seq.pop(0) if len(seq) > 1 else seq[0])
        return sess

    def drive(self, sess, n):
        return [sess.step(self.frame) for _ in range(n)]


class TestUnknownPolicy(UnknownSimBase):
    def test_isolated_none_frame_harmless(self):
        # good, None, good ... : no bad frames, streaks stay 0, TRACKING kept.
        sess = self.make_session([res(conf=0.9)], [0.9, None, 0.9])
        outs = self.drive(sess, 6)
        self.assertTrue(all(o.state is TrackState.TRACKING for o in outs))
        self.assertEqual(sess._bad_streak, 0)
        self.assertEqual(sess._unknown_streak, 0)    # cleared by numeric frames

    def test_none_below_bound_with_high_confidence_not_bad(self):
        # exactly N_UNKNOWN None frames: nothing counts as bad yet (crit. 2/3).
        sess = self.make_session([res(conf=0.9)], [None])
        outs = self.drive(sess, N_UNKNOWN)
        self.assertEqual(sess._bad_streak, 0)
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs))

    def test_persistent_unknown_reaches_lost_within_bound(self):
        # all-None + HIGH tracker/regional confidence: frames beyond the bound
        # are loss evidence -> LOST at exactly N_UNKNOWN + LOST_AFTER_N_BAD.
        sess = self.make_session([res(conf=0.9)], [None])
        outs = self.drive(sess, N_UNKNOWN + N_BAD)
        self.assertTrue(all(o.state is not TrackState.LOST
                            for o in outs[:-1]), "LOST fired early")
        self.assertIs(outs[-1].state, TrackState.LOST,
                      "identity unknown for the full bound must declare LOST")

    def test_numeric_low_similarity_unchanged(self):
        # present-but-wrong appearance stays the ordinary bad path (M8 as-is).
        sess = self.make_session([res(conf=0.9)], [0.0])
        outs = self.drive(sess, N_BAD)
        self.assertIs(outs[-1].state, TrackState.LOST)
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs[:-1]))

    def test_none_holds_streak_and_never_resets_it(self):
        # bad, bad, None(hold), bad, bad -> LOST on frame 5. Under the legacy
        # rule the None frame (evidence = conf = 0.9) RESET the streak and
        # LOST never fired (the measured Run C failure).
        sess = self.make_session([res(conf=0.9)], [0.0, 0.0, None, 0.0, 0.0])
        outs = self.drive(sess, 5)
        self.assertEqual(sess._bad_streak, N_BAD)
        self.assertIs(outs[-1].state, TrackState.LOST)
        self.assertIs(outs[2].state, TrackState.LOW_CONFIDENCE)  # held, not reset

    def test_recovery_numeric_good_clears_both_streaks(self):
        # None frames then a real numeric-good frame: unknown counter and the
        # bad streak both clear (identity re-confirmed).
        sess = self.make_session([res(conf=0.9)], [0.0, 0.0, None, 0.9])
        self.drive(sess, 4)
        self.assertEqual(sess._unknown_streak, 0)
        self.assertEqual(sess._bad_streak, 0)

    def test_flat_reference_keeps_legacy_neutrality(self):
        # low-texture target: identity was NEVER measurable -> no escalation,
        # no held streak, no LOST from None alone, ever.
        sess = self.make_session([res(conf=0.9)], [None])
        sess._ref_std = 0.0                       # the ref-flat capability gate
        outs = self.drive(sess, 4 * (N_UNKNOWN + N_BAD))
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs))
        self.assertEqual(sess._unknown_streak, 0)
        self.assertEqual(sess._bad_streak, 0)

    def test_known_occlusion_suspension_takes_precedence(self):
        # escalated-unknown frames on the dilated overlay FREEZE the streak
        # (explained unknown) -> no LOST while HUD-occluded.
        sess = self.make_session([res(conf=0.9)], [None])
        ovl = np.zeros((200, 200), np.uint8)
        ovl[90:111, 90:111] = 255                 # covers the committed point
        sess._overlay_dil = ovl
        outs = self.drive(sess, 4 * (N_UNKNOWN + N_BAD))
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs))
        self.assertEqual(sess._bad_streak, 0)     # frozen, never advanced

    def test_edge_penalty_semantics_unchanged(self):
        # near-border point with good numeric similarity: edge penalizes the
        # displayed confidence, but is NOT loss evidence (streak stays 0).
        sess = self.make_session([res(x=6.0, y=6.0, conf=0.9)], [0.9])
        outs = self.drive(sess, 6)
        self.assertEqual(sess._bad_streak, 0)
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs))
        self.assertTrue(all(o.signals["edge"] for o in outs))

    def test_jump_veto_semantics_unchanged(self):
        # a teleporting candidate is vetoed -> evidence 0 -> bad (as before);
        # the unknown policy does not interfere with the veto path.
        sess = self.make_session([res(x=180.0, y=180.0, conf=0.9)], [0.9])
        out = sess.step(self.frame)
        self.assertTrue(out.signals["jump_vetoed"])
        self.assertEqual(sess._bad_streak, 1)


if __name__ == "__main__":
    unittest.main()
