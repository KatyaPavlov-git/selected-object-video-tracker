"""Stage 2D/2E benchmark: the exit-absence-RETURN reacquisition contract on the
REAL pipeline. The 2E complement to tests/test_benchmark_exit_return.py, whose
2A form documents the A4 gap (terminal LOST through the return) and reserves this
slot: "Stage 2D replaces the return-phase assertions with the reacquisition
contract."

Same controlled scene + analytic ground truth + visually similar decoy
(tests/synthetic_scenes.py), no-overlay core path. Reacquisition is exercised via
the per-instance enable_reacq flag; the PRODUCTION default
(config.REACQ_DECISION_ENABLED) is NOT read or changed here.

Contract:
  enabled  : exit -> LOST -> SEARCH -> reacquire -> PROBATION -> TRACKING, with
             EXACTLY ONE REACQ_INIT in the episode, the committed point back
             within REACQ_RETURN_TOL_PX of ground truth, and the run ending
             TRACKING;
  disabled : terminal LOST through absence AND return, with zero recovery events
             and the committed point frozen (the preserved 2A contract).

Run from the repository root:
    python3 -m unittest tests.test_benchmark_reacq_return -v
"""
import sys
import types
import unittest

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import config as prod_config
from ground_target_tracking import evaluation, trackers
from ground_target_tracking.session import TrackingSession, TrackState
try:  # discovery start-dir on sys.path (python3 -m unittest discover tests)
    from synthetic_scenes import build_scene, exit_return_schedule, fully_visible
except ImportError:  # direct module run from the repo root
    from tests.synthetic_scenes import build_scene, exit_return_schedule, fully_visible

REACQ_RETURN_TOL_PX = 5.0   # committed-vs-GT tolerance after reacquisition
TAIL_AFTER_RETURN = 10      # frames past first-fully-visible-return before judging


def snapshot_cfg(**overrides):
    """Full production-config snapshot as a namespace: production values drive
    behavior, but tests never mutate the shared config module."""
    ns = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                  for k in dir(prod_config) if k.isupper()})
    ns.OVERLAY_MASK_ENABLED = False            # no-overlay core path
    ns.EXPERIMENTAL_REGIONAL_MOTION = False
    ns.FEED_FROZEN_ENABLED = False             # synthetic absence holds IDENTICAL frames
                                               # (not a real feed) -> feed-death detector off
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def run(cfg, enable):
    """One full exit-return pass; returns (rows, last_visible_t, first_return_t,
    events). rows carry state, committed point, per-frame GT error and source."""
    base, target, _decoy = build_scene()
    h, w = base.shape[:2]
    seq = evaluation.synthetic_gt_sequence(base, target, exit_return_schedule())
    vis = [fully_visible(gt, w, h) for _, gt in seq]
    gap0 = vis.index(False)
    last_visible_t = gap0
    first_return_t = vis.index(True, gap0) + 1

    tracker = trackers.make_tracker("of_kalman", cfg)
    sess = TrackingSession(tracker, cfg, enable_reacq=enable)
    sess.init(base, target)
    rows = []
    for t, (frame, gt) in enumerate(seq, start=1):
        out = sess.step(frame)
        err = ((out.point.x - gt.x) ** 2 + (out.point.y - gt.y) ** 2) ** 0.5
        rows.append({"t": t, "state": out.state, "x": out.point.x,
                     "y": out.point.y, "err": err, "source": out.result.source,
                     "visible": vis[t - 1]})
    return rows, last_visible_t, first_return_t, list(sess._reacq_events)


def first_lost_t(rows):
    for r in rows:
        if r["state"] is TrackState.LOST:
            return r["t"]
    return None


class TestExitReturnReacquire2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = snapshot_cfg(LOST_OFFFRAME_HARDENING=True)   # matches the 2A cfg
        cls.rows, cls.last_vis, cls.first_ret, cls.events = run(cls.cfg, enable=True)
        cls.rows_off, _, cls.first_ret_off, cls.events_off = run(cls.cfg, enable=False)
        cls.names = [e["event"] for e in cls.events]

    # ---- enabled: the full recovery cycle ------------------------------- #
    def test_exit_reaches_lost_after_last_visible(self):
        lost_t = first_lost_t(self.rows)
        self.assertIsNotNone(lost_t, "enabled run never reached LOST on exit")
        self.assertGreater(lost_t, self.last_vis,
                           "LOST must not fire while the target is visible")

    def test_recovery_cycle_events_in_order(self):
        # RECOVERY_ENTER -> REACQ_INIT -> PROBATION_START -> PROBATION_OK, in order
        for name in ("RECOVERY_ENTER", "REACQ_INIT", "PROBATION_START", "PROBATION_OK"):
            self.assertIn(name, self.names, f"missing recovery event {name}")
        self.assertLess(self.names.index("RECOVERY_ENTER"),
                        self.names.index("REACQ_INIT"))
        self.assertLess(self.names.index("REACQ_INIT"),
                        self.names.index("PROBATION_OK"))

    def test_exactly_one_reacq_init_in_episode(self):
        inits = [e for e in self.events if e["event"] == "REACQ_INIT"]
        self.assertEqual(len(inits), 1,
                         f"expected exactly one REACQ_INIT, got {len(inits)}")
        self.assertEqual(inits[0]["episode"], 1, "reacquire must be episode 1")

    def test_committed_point_returns_within_tolerance(self):
        tail = [r for r in self.rows if r["t"] >= self.first_ret + TAIL_AFTER_RETURN]
        self.assertTrue(tail, "scene has no post-return tail to judge")
        errs = sorted(r["err"] for r in tail)
        median = errs[len(errs) // 2]
        self.assertLessEqual(median, REACQ_RETURN_TOL_PX,
                             f"committed point not back on target after reacquire: "
                             f"median tail err {median:.2f}px > {REACQ_RETURN_TOL_PX}px")

    def test_run_ends_tracking(self):
        self.assertIs(self.rows[-1]["state"], TrackState.TRACKING,
                      "reacquired run must end in TRACKING")

    # ---- disabled: the preserved 2A terminal-LOST contract -------------- #
    def test_disabled_stays_terminal_lost_through_return(self):
        lost_t = first_lost_t(self.rows_off)
        self.assertIsNotNone(lost_t, "disabled run never reached LOST")
        after = [r for r in self.rows_off if r["t"] >= lost_t]
        self.assertTrue(all(r["state"] is TrackState.LOST for r in after),
                        "disabled: LOST must be terminal through absence AND return")
        frozen = {(r["x"], r["y"]) for r in after}
        self.assertEqual(len(frozen), 1,
                         "disabled: the committed point must stay frozen while LOST")
        self.assertGreaterEqual(self.rows_off[-1]["t"], self.first_ret_off,
                                "scene must actually contain a return phase")

    def test_disabled_emits_no_recovery_events(self):
        self.assertEqual(self.events_off, [],
                         "disabled path must emit no recovery events")


if __name__ == "__main__":
    unittest.main()
