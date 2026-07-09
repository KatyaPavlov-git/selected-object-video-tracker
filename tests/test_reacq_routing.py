"""M9 interleaved-routing tests (submission-hardening).

The routing contract under test (session-level; reacquisition.py untouched):
  * exactly ONE route executes per executed recovery tick;
  * on a feature-capable + context-capable reference, the bounded template
    scan runs ONLY after the feature route's last result was NEUTRAL
    (strict alternation) — the template route is no longer structurally
    blocked merely because descriptors exist;
  * a feature AMBIGUOUS is a refusal: it never triggers the template route
    and clears the hypothesis (conservative acceptance preserved);
  * acceptance gates/thresholds are unchanged — an AMBIGUOUS/NEUTRAL stream
    through the unblocked route can never re-seed the tracker.

Driven with scripted stubs (deterministic); the REAL HypothesisTracker runs
inside the session. The scan-factory CONDITION is tested against the real
_new_scan; the routed sessions use a scripted StubScan via the factory seam.

Run from the repository root:
    python3 -m unittest discover tests -v
"""
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import reacquisition, utils
from ground_target_tracking.session import TrackingSession, TrackState

try:  # discovery adds tests/ to sys.path; direct module runs use the package
    from test_session import (StubTracker, make_reacq_cfg, res,
                              rr_ambiguous, rr_match, rr_neutral)
except ImportError:
    from tests.test_session import (StubTracker, make_reacq_cfg, res,
                                    rr_ambiguous, rr_match, rr_neutral)


class StubReacqRef:
    """Scripted Reacquirer WITH a capability-flagged reference, so the session
    routing sees a feature-capable + context-capable identity model."""

    def __init__(self, script, has_context=True, has_descriptors=True):
        self.script = list(script)
        self.i = 0
        self.calls = 0
        self.reference = types.SimpleNamespace(has_context=has_context,
                                               has_descriptors=has_descriptors)

    def build_reference(self, frame_bgr, point, overlay_mask=None):
        pass

    def best_candidate(self, frame_bgr):
        self.calls += 1
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return r


class StubScan:
    """Scripted TemplateScanScheduler stand-in: step() replays ReacqResults
    (repeating the last) with a minimal instrumentation dict."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = 0

    def step(self, frame_bgr):
        self.calls += 1
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return r, {"scan_mode": "SCAN", "scan_units": 1}


class RoutedSession(TrackingSession):
    """Session whose _new_scan returns the test's scripted scan. The factory
    CONDITION (context capability) is covered by TestScanFactory below."""

    def __init__(self, *args, scan_factory=None, **kwargs):
        self._scan_factory = scan_factory
        super().__init__(*args, **kwargs)

    def _new_scan(self):
        return self._scan_factory() if self._scan_factory is not None else None


def rr_scan_match(x=120.0, y=100.0, confirm=0.9):
    return reacquisition.ReacqResult(reacquisition.Identity.MATCH,
                                     utils.Point2D(x, y), 1.0, "template",
                                     confirm, "stub-scan")


def rr_scan_ambiguous():
    return reacquisition.ReacqResult(reacquisition.Identity.AMBIGUOUS,
                                     utils.Point2D(120.0, 100.0), 1.0,
                                     "template", 0.1, "stub-scan:confirm-low")


def rr_scan_neutral():
    return reacquisition.ReacqResult(reacquisition.Identity.NEUTRAL, None,
                                     None, "template", None,
                                     "template-scan:scanning")


class RoutingBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_reacq_cfg()
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        self.p0 = utils.Point2D(100.0, 100.0)

    def build(self, feat_script, scan_script, cfg=None):
        cfg = cfg or self.cfg
        self.scan = StubScan(scan_script)
        self.reacq = StubReacqRef(feat_script)
        sess = RoutedSession(StubTracker([res(conf=0.05, ok=False)]), cfg,
                             enable_reacq=True, reacquirer=self.reacq,
                             scan_factory=lambda: self.scan)
        sess.init(self.frame, self.p0)
        for _ in range(cfg.LOST_AFTER_N_BAD):     # M8 declares LOST
            out = sess.step(self.frame)
        assert out.state is TrackState.LOST
        return sess

    def tick(self, sess):
        """One executed recovery tick; returns (route, feature_calls_delta,
        scan_calls_delta, SessionResult)."""
        f0, s0 = self.reacq.calls, self.scan.calls
        out = sess.step(self.frame)
        return (out.signals.get("reacq_route"),
                self.reacq.calls - f0, self.scan.calls - s0, out)


class TestScanFactory(unittest.TestCase):
    """The REAL _new_scan condition: any context-capable reference gets the
    bounded scheduler — descriptors no longer disqualify the template route."""

    def _sess(self, has_context, has_descriptors):
        cfg = make_reacq_cfg()
        sess = TrackingSession(
            StubTracker([res()]), cfg, enable_reacq=True,
            reacquirer=StubReacqRef([rr_neutral()], has_context=has_context,
                                    has_descriptors=has_descriptors))
        rng = np.random.default_rng(42)
        sess.init(rng.integers(0, 256, (200, 200, 3), dtype=np.uint8),
                  utils.Point2D(100.0, 100.0))
        return sess

    def test_scheduler_built_for_descriptor_capable_reference(self):
        sess = self._sess(has_context=True, has_descriptors=True)
        self.assertIsInstance(sess._new_scan(),
                              reacquisition.TemplateScanScheduler,
                              "descriptor capability must not block the "
                              "template scheduler")

    def test_scheduler_built_for_descriptor_free_reference(self):
        sess = self._sess(has_context=True, has_descriptors=False)
        self.assertIsInstance(sess._new_scan(),
                              reacquisition.TemplateScanScheduler)

    def test_no_scheduler_without_context_capability(self):
        sess = self._sess(has_context=False, has_descriptors=True)
        self.assertIsNone(sess._new_scan(),
                          "no context cue -> nothing for the scan to sweep")


class TestInterleavedRouting(RoutingBase):
    def test_feature_neutral_interleaves_scan_ticks(self):
        sess = self.build([rr_neutral()], [rr_scan_neutral()])
        routes = [self.tick(sess)[0] for _ in range(6)]
        self.assertEqual(routes, ["feature", "template-scan"] * 3,
                         "NEUTRAL feature results must strictly alternate "
                         "with bounded scan ticks")
        self.assertEqual(self.reacq.calls, 3)
        self.assertEqual(self.scan.calls, 3)

    def test_feature_candidate_suppresses_scan(self):
        # A feature route that keeps producing candidates (MATCH here, with
        # persistence raised so acceptance never fires) runs exclusively.
        cfg = make_reacq_cfg(REACQ_PERSIST_N=99)
        sess = self.build([rr_match()], [rr_scan_neutral()], cfg=cfg)
        routes = [self.tick(sess)[0] for _ in range(6)]
        self.assertEqual(routes, ["feature"] * 6)
        self.assertEqual(self.scan.calls, 0,
                         "scan must not run while the feature route yields "
                         "candidates")

    def test_feature_ambiguous_never_routes_to_template(self):
        # AMBIGUOUS = present-but-failed: a refusal, not an absence — it must
        # not open the template route, and it clears the hypothesis.
        sess = self.build([rr_ambiguous()], [rr_scan_match()])
        for _ in range(6):
            route, _, ds, out = self.tick(sess)
            self.assertEqual(route, "feature")
            self.assertEqual(ds, 0, "AMBIGUOUS fell through to the template")
            self.assertIs(out.state, TrackState.LOST)
        self.assertFalse(sess._hypo.has_hypothesis,
                         "AMBIGUOUS must clear persistence")
        self.assertEqual(sess.tracker.init_calls, 1, "no reseed on refusals")

    def test_one_route_per_tick_bound(self):
        # The real-time work bound: every executed tick runs EXACTLY one
        # route, across feature-NEUTRAL, scan, and feature-reclaim phases.
        feat = [rr_neutral(), rr_neutral(), rr_match(), rr_neutral()]
        sess = self.build(feat, [rr_scan_neutral()],
                          cfg=make_reacq_cfg(REACQ_PERSIST_N=99))
        for k in range(10):
            _, df, ds, _ = self.tick(sess)
            self.assertEqual(df + ds, 1,
                             f"tick {k}: expected exactly one route, got "
                             f"feature={df} scan={ds}")

    def test_template_match_streak_builds_to_probation(self):
        # Template MATCHes via the unblocked scan build a persistence streak
        # across the interleaved feature-NEUTRAL gaps (gap 1 <= MAX_NEUTRAL)
        # and re-seed the tracker exactly once into PROBATION.
        sess = self.build([rr_neutral()], [rr_scan_match()])
        states = []
        for _ in range(2 * self.cfg.REACQ_PERSIST_N):  # F,S alternation
            states.append(self.tick(sess)[3].state)
        self.assertEqual(sess.tracker.init_calls, 2,
                         "exactly one reseed from the template streak")
        init_events = [e for e in sess._reacq_events if e["event"] == "REACQ_INIT"]
        self.assertEqual(len(init_events), 1)
        self.assertEqual(init_events[0]["cue"], "template")
        self.assertIs(states[-1], TrackState.LOW_CONFIDENCE,
                      "acceptance must enter probation (public LOW_CONFIDENCE)")

    def test_false_acceptance_unchanged_on_unblocked_route(self):
        # Conservative acceptance survives the unblock: a scan stream of
        # present-but-failed (AMBIGUOUS) or absent (NEUTRAL) template results
        # can never re-seed, however long it runs.
        for scan_script in ([rr_scan_ambiguous()], [rr_scan_neutral()]):
            sess = self.build([rr_neutral()], scan_script)
            for _ in range(12):
                out = sess.step(self.frame)
                self.assertIs(out.state, TrackState.LOST)
            self.assertEqual(sess.tracker.init_calls, 1,
                             "sub-threshold template evidence re-seeded the "
                             "tracker")
            self.assertEqual(
                [e for e in sess._reacq_events if e["event"] == "REACQ_INIT"],
                [])


if __name__ == "__main__":
    unittest.main()
