"""Stage 2A benchmark: the exit-absence-return contract on the REAL pipeline.

Drives make_tracker("of_kalman") -> TrackingSession over a controlled scene
(tests/synthetic_scenes.py) with analytic ground truth and a visually similar
decoy, on the no-overlay core path (plan §8).

2A form — no reacquisition exists yet, so the contract here is:
  * visible phase: >=90% TRACKING after warm-up, zero accepted-wrong frames;
  * with LOST_OFFFRAME_HARDENING on: LOST within LOST_AFTER_N_BAD + 10 frames
    of the last fully-visible frame;
  * absence and beyond: 100% LOST with the committed point frozen — including
    the return phase, which DOCUMENTS the A4 gap (Stage 2D replaces the
    return-phase assertions with the reacquisition contract);
  * without the hardening flag: LOST arrives later or never (today's behavior,
    kept as the comparison row).

Run from the repository root:
    python3 -m unittest tests.test_benchmark_exit_return -v
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

WARMUP = 10          # frames before the visible-phase TRACKING ratio is judged
LOST_SLACK = 10      # contract: LOST <= LOST_AFTER_N_BAD + LOST_SLACK after exit
ACCEPT_TOL_PX = 5.0  # accepted-wrong tolerance (existing synthetic default)


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


def run_benchmark(cfg):
    """One full pass; returns (rows, last_visible_t, first_return_t)."""
    base, target, _decoy = build_scene()
    h, w = base.shape[:2]
    seq = evaluation.synthetic_gt_sequence(base, target,
                                           exit_return_schedule())
    vis = [fully_visible(gt, w, h) for _, gt in seq]
    gap0 = vis.index(False)                     # first not-fully-visible
    last_visible_t = gap0                       # t is 1-based; vis[] 0-based
    first_return_t = vis.index(True, gap0) + 1

    tracker = trackers.make_tracker("of_kalman", cfg)
    # Stage 2A is the NO-REACQUISITION baseline (terminal LOST, frozen point).
    # G1 (2026-07-08) made reacquisition the delivered default, so this
    # benchmark now pins the --no-reacq path explicitly.
    sess = TrackingSession(tracker, cfg, enable_reacq=False)
    sess.init(base, target)
    rows = []
    for t, (frame, gt) in enumerate(seq, start=1):
        out = sess.step(frame)
        err = ((out.point.x - gt.x) ** 2 + (out.point.y - gt.y) ** 2) ** 0.5
        rows.append({"t": t, "state": out.state, "x": out.point.x,
                     "y": out.point.y, "err": err,
                     "source": out.result.source, "visible": vis[t - 1]})
    return rows, last_visible_t, first_return_t


def first_lost_t(rows):
    for r in rows:
        if r["state"] is TrackState.LOST:
            return r["t"]
    return None


class TestExitReturnBenchmark2A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_on = snapshot_cfg(LOST_OFFFRAME_HARDENING=True)
        cls.rows, cls.last_vis, cls.first_ret = run_benchmark(cls.cfg_on)
        cls.rows_off, _, _ = run_benchmark(snapshot_cfg())

    def test_visible_phase_tracks_the_target(self):
        vis_rows = [r for r in self.rows
                    if WARMUP < r["t"] <= self.last_vis]
        tracking = sum(r["state"] is TrackState.TRACKING for r in vis_rows)
        self.assertGreaterEqual(tracking / len(vis_rows), 0.90,
                                "visible phase must be >=90% TRACKING")
        accepted_wrong = [r for r in vis_rows
                          if r["source"] == "measure"
                          and r["state"] in (TrackState.TRACKING,
                                             TrackState.LOW_CONFIDENCE)
                          and r["err"] > ACCEPT_TOL_PX]
        self.assertEqual(accepted_wrong, [],
                         f"confident-but-wrong frames in the visible phase: "
                         f"{[(r['t'], round(r['err'], 1)) for r in accepted_wrong]}")

    def test_exit_reaches_lost_within_contract_window(self):
        lost_t = first_lost_t(self.rows)
        self.assertIsNotNone(lost_t, "hardened run never reached LOST")
        budget = self.last_vis + self.cfg_on.LOST_AFTER_N_BAD + LOST_SLACK
        self.assertLessEqual(lost_t, budget,
                             f"LOST at t={lost_t}, contract window ends {budget}")
        self.assertGreater(lost_t, self.last_vis,
                           "LOST must not fire while the target is visible")

    def test_absence_and_return_stay_lost_with_point_frozen(self):
        # 2A form: LOST is terminal, so every frame from entry on is LOST and
        # the committed point never moves. The return-phase half of this
        # assertion DOCUMENTS the A4 gap and is replaced in Stage 2D by the
        # reacquisition contract.
        lost_t = first_lost_t(self.rows)
        after = [r for r in self.rows if r["t"] >= lost_t]
        self.assertTrue(all(r["state"] is TrackState.LOST for r in after),
                        "2A: LOST must be terminal through absence AND return")
        frozen = {(r["x"], r["y"]) for r in after}
        self.assertEqual(len(frozen), 1,
                         "the committed point must stay frozen while LOST")
        self.assertGreaterEqual(self.rows[-1]["t"], self.first_ret,
                                "scene must actually contain a return phase")

    def test_hardening_makes_lost_entry_earlier_than_today(self):
        lost_on = first_lost_t(self.rows)
        lost_off = first_lost_t(self.rows_off)
        if lost_off is None:
            return  # today's behavior: never LOST — the defect in its pure form
        self.assertLess(lost_on, lost_off,
                        "hardening must not delay LOST entry vs today")


if __name__ == "__main__":
    unittest.main()
