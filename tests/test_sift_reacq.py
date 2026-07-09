"""SIFT last-confident reacquisition route (v8 design review, 2026-07-07).

Covers the four contract layers of the new route:
  * SNAPSHOT gate — the session keeps ONE cheap last-confident frame copy,
    refreshed only on trustworthy frames (TRACKING + measured + clean ROI),
    throttled, reset on init, and never captured while reacq is disabled;
  * DEFERRED build — no SIFT work before LOST; the reference is built exactly
    once per snapshot, at recovery entry;
  * CLASSIFICATION — floor-10 inliers / residual / scale-envelope gates; by
    the measured v8 evidence there is deliberately NO inlier-ratio gate and
    confirm-NCC never gates (both inverted on the true rotated return);
  * ROUTING + persistence — the SIFT tick preempts exactly its cadence slots
    (one route per executed tick), feeds its OWN HypothesisTracker so
    interleaved NEUTRAL results never burn its gap budget, and acceptance
    flows through the existing reseed/probation path; sub-floor evidence can
    never re-seed.

The end-to-end fixture mirrors the v8 geometry: an EVIDENCE-FREE flat center
(the crosshair-masked selection analog) surrounded by clean textured context,
a hard exit to unrelated content, and a return rotated ~160 deg at 1.2x —
the regime where templates and ORB are measured dead and only the SIFT
context match can reconstruct the selected coordinate.

Run from the repository root:
    python3 -m unittest discover tests -v
"""
import sys
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import config as prod_config
from ground_target_tracking import reacquisition as R
from ground_target_tracking import trackers, utils
from ground_target_tracking.session import TrackingSession, TrackState

try:  # discovery adds tests/ to sys.path; direct module runs use the package
    from test_session import (StubTracker, make_reacq_cfg, res, rr_neutral)
except ImportError:
    from tests.test_session import (StubTracker, make_reacq_cfg, res,
                                    rr_neutral)


def rr_sift(identity, x=120.0, y=100.0, reason="stub"):
    return R.ReacqResult(identity, utils.Point2D(x, y), 1.0, "sift-lc",
                         0.05, reason)


class StubReacqLC:
    """Scripted Reacquirer with the SIFT-route surface: build_lc_reference
    installs a (stub) lc_reference; best_candidate / best_candidate_lc replay
    their scripts and count calls."""

    def __init__(self, feat_script, lc_script):
        self.feat = list(feat_script)
        self.lc = list(lc_script)
        self.i = 0
        self.j = 0
        self.calls = 0        # best_candidate (existing routes)
        self.lc_calls = 0     # best_candidate_lc (SIFT route)
        self.build_lc_calls = 0
        self.lc_reference = None

    def build_reference(self, frame_bgr, point, overlay_mask=None):
        pass

    def build_lc_reference(self, frame_bgr, point, overlay_mask=None):
        self.build_lc_calls += 1
        self.lc_reference = types.SimpleNamespace(point=point)
        return self.lc_reference

    def best_candidate(self, frame_bgr):
        self.calls += 1
        r = self.feat[min(self.i, len(self.feat) - 1)]
        self.i += 1
        return r

    def best_candidate_lc(self, frame_bgr):
        self.lc_calls += 1
        r = self.lc[min(self.j, len(self.lc) - 1)]
        self.j += 1
        return r


class SiftSessionBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_reacq_cfg(REACQ_SIFT_EVERY=3)
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        self.p0 = utils.Point2D(100.0, 100.0)

    def build(self, tracker_script, feat_script, lc_script, cfg=None,
              enable=True):
        cfg = cfg or self.cfg
        self.stub = StubReacqLC(feat_script, lc_script)
        sess = TrackingSession(StubTracker(tracker_script), cfg,
                               enable_reacq=enable, reacquirer=self.stub)
        sess.init(self.frame, self.p0)
        return sess

    def good_then_bad(self, n_good=None):
        """Tracker script: enough strong frames to capture a snapshot, then
        sustained failure toward LOST."""
        n_good = n_good or self.cfg.REF_UPDATE_EVERY
        return [res(conf=0.9)] * n_good + [res(conf=0.05, ok=False)] * 50


class TestSnapshotGate(SiftSessionBase):
    def test_snapshot_after_warmup_then_refreshes_every_frame(self):
        sess = self.build(self.good_then_bad(n_good=20), [rr_neutral()],
                          [rr_neutral()])
        for _ in range(self.cfg.REF_UPDATE_EVERY - 1):
            sess.step(self.frame)
        self.assertIsNone(sess._lc_frame, "snapshot before the warm-up count")
        sess.step(self.frame)
        self.assertIsNotNone(sess._lc_frame)
        self.assertEqual((sess._lc_point.x, sess._lc_point.y),
                         (self.p0.x, self.p0.y))
        # Warm-up satisfied: every further qualifying frame refreshes the
        # copy, so the snapshot is always the LAST trustworthy view.
        first = sess._lc_frame
        sess.step(self.frame)
        self.assertIsNot(sess._lc_frame, first,
                         "post-warm-up qualifying frames must refresh")

    def test_no_snapshot_when_reacq_disabled(self):
        sess = self.build([res(conf=0.9)], [rr_neutral()], [rr_neutral()],
                          enable=False)
        for _ in range(self.cfg.REF_UPDATE_EVERY * 2):
            sess.step(self.frame)
        self.assertIsNone(sess._lc_frame)

    def test_predict_frames_never_qualify(self):
        sess = self.build([res(conf=0.9, source="predict", ok=True)],
                          [rr_neutral()], [rr_neutral()])
        for _ in range(self.cfg.REF_UPDATE_EVERY * 2):
            sess.step(self.frame)
        self.assertIsNone(sess._lc_frame, "a coasted prediction is not "
                          "trustworthy snapshot material")

    def test_vetoed_frame_resets_the_streak(self):
        far = 100.0 + self.cfg.MAX_JUMP_PX + 5.0
        script = ([res(conf=0.9)] * (self.cfg.REF_UPDATE_EVERY - 1)
                  + [res(x=far, y=100.0, conf=0.9)]
                  + [res(conf=0.9)] * (self.cfg.REF_UPDATE_EVERY - 1))
        sess = self.build(script, [rr_neutral()], [rr_neutral()])
        for _ in range(2 * self.cfg.REF_UPDATE_EVERY - 1):
            sess.step(self.frame)
        self.assertIsNone(sess._lc_frame,
                          "the throttle must not survive a vetoed frame")

    def test_edge_frames_never_qualify(self):
        sess = TrackingSession(StubTracker([res(x=15.0, y=100.0, conf=0.9)]),
                               self.cfg, enable_reacq=True,
                               reacquirer=StubReacqLC([rr_neutral()],
                                                      [rr_neutral()]))
        sess.init(self.frame, utils.Point2D(15.0, 100.0))
        for _ in range(self.cfg.REF_UPDATE_EVERY * 2):
            sess.step(self.frame)
        self.assertIsNone(sess._lc_frame)

    def test_init_clears_the_snapshot(self):
        sess = self.build(self.good_then_bad(), [rr_neutral()], [rr_neutral()])
        for _ in range(self.cfg.REF_UPDATE_EVERY):
            sess.step(self.frame)
        self.assertIsNotNone(sess._lc_frame)
        sess.init(self.frame, self.p0)
        self.assertIsNone(sess._lc_frame, "a new target keeps no old snapshot")


class TestDeferredBuild(SiftSessionBase):
    def test_built_once_at_recovery_entry_only(self):
        sess = self.build(self.good_then_bad(), [rr_neutral()], [rr_neutral()])
        for _ in range(self.cfg.REF_UPDATE_EVERY):
            sess.step(self.frame)
        self.assertEqual(self.stub.build_lc_calls, 0,
                         "no SIFT work on the healthy tracking path")
        for _ in range(self.cfg.LOST_AFTER_N_BAD + 1):
            sess.step(self.frame)
        self.assertIs(sess.state, TrackState.LOST)
        self.assertEqual(self.stub.build_lc_calls, 1,
                         "built exactly once at recovery entry")
        for _ in range(6):
            sess.step(self.frame)
        self.assertEqual(self.stub.build_lc_calls, 1,
                         "the same snapshot is never rebuilt")

    def test_no_build_without_a_snapshot(self):
        sess = self.build([res(conf=0.05, ok=False)], [rr_neutral()],
                          [rr_neutral()])
        for _ in range(self.cfg.LOST_AFTER_N_BAD + 4):
            sess.step(self.frame)
        self.assertIs(sess.state, TrackState.LOST)
        self.assertEqual(self.stub.build_lc_calls, 0)


class TestClassifySift(unittest.TestCase):
    def cand(self, inl=12, ratio=0.1, resid=1.0, scale=1.0):
        return R.FeatureCandidate(
            point=utils.Point2D(50.0, 50.0), rotation_deg=160.0, scale=scale,
            n_inliers=inl, inlier_ratio=ratio, residual=resid, cue="sift-lc",
            n_query_desc=100, n_matches=40, n_good=30)

    def setUp(self):
        self.cfg = make_reacq_cfg(REACQ_SIFT_MIN_INLIERS=10,
                                  REACQ_MAX_RESIDUAL_PX=2.0,
                                  REACQ_FEAT_SCALE_MIN=0.5,
                                  REACQ_FEAT_SCALE_MAX=2.0)

    def test_all_gates_pass_ignores_ratio_and_confirm(self):
        # inlier_ratio 0.1 and any confirm value must NOT block: both axes
        # were measured inverted on the true v8 rotated return.
        identity, reason = R._classify_sift(self.cand(inl=10, ratio=0.1),
                                            self.cfg)
        self.assertIs(identity, R.Identity.MATCH)
        self.assertEqual(reason, "sift:all-gates-pass")

    def test_inlier_floor(self):
        identity, reason = R._classify_sift(self.cand(inl=9), self.cfg)
        self.assertIs(identity, R.Identity.AMBIGUOUS)
        self.assertEqual(reason, "sift:inliers-low")

    def test_residual_gate(self):
        identity, reason = R._classify_sift(self.cand(resid=3.0), self.cfg)
        self.assertIs(identity, R.Identity.AMBIGUOUS)
        self.assertEqual(reason, "sift:residual-high")

    def test_scale_envelope(self):
        identity, reason = R._classify_sift(self.cand(scale=2.5), self.cfg)
        self.assertIs(identity, R.Identity.AMBIGUOUS)
        self.assertEqual(reason, "sift:scale-out-of-envelope")


class TestSiftRouting(SiftSessionBase):
    def to_lost(self, sess):
        for _ in range(self.cfg.REF_UPDATE_EVERY + self.cfg.LOST_AFTER_N_BAD):
            out = sess.step(self.frame)
        assert out.state is TrackState.LOST
        return out

    def tick(self, sess):
        f0, l0 = self.stub.calls, self.stub.lc_calls
        out = sess.step(self.frame)
        return (out.signals.get("reacq_route"), self.stub.calls - f0,
                self.stub.lc_calls - l0, out)

    def test_sift_route_owns_ticks_while_reference_exists(self):
        # G2-a (2026-07-08): while an LC reference exists, the SIFT route owns
        # EVERY executed tick (on the real Reacquirer each tick is one bounded
        # stripe; on this stub surface it is one whole-frame evaluation). The
        # retired REACQ_SIFT_EVERY cadence no longer interleaves other routes.
        sess = self.build(self.good_then_bad(),
                          [rr_neutral()], [rr_sift(R.Identity.NEUTRAL)])
        self.to_lost(sess)
        for _ in range(6):
            route, df, dl, _ = self.tick(sess)
            self.assertEqual(route, "sift-lc")
            self.assertEqual(df, 0, "no other route while LC exists")
            self.assertEqual(dl, 1, "exactly one evaluation per executed tick")

    def test_sift_persistence_latches_and_reseeds_once(self):
        # G2-a semantics: the stub surface yields one identity observation per
        # executed tick, so the separate HypothesisTracker latches after
        # exactly REACQ_PERSIST_N compatible MATCHes and re-seeds ONCE into
        # probation on that tick.
        sess = self.build(self.good_then_bad(),
                          [rr_neutral()],
                          [rr_sift(R.Identity.MATCH,
                                   reason="sift:all-gates-pass")])
        self.to_lost(sess)
        states = []
        for _ in range(self.cfg.REACQ_PERSIST_N):
            states.append(self.tick(sess)[3].state)
        self.assertEqual(sess.tracker.init_calls, 2,
                         "exactly one reseed from the SIFT streak")
        inits = [e for e in sess._reacq_events if e["event"] == "REACQ_INIT"]
        self.assertEqual(len(inits), 1)
        self.assertEqual(inits[0]["cue"], "sift-lc")
        self.assertIs(states[-1], TrackState.LOW_CONFIDENCE,
                      "acceptance must enter probation")
        self.assertTrue(any(e["event"] == "PROBATION_START"
                            for e in sess._reacq_events))

    def test_subfloor_evidence_never_reseeds(self):
        for lc_script in ([rr_sift(R.Identity.AMBIGUOUS,
                                   reason="sift:inliers-low")],
                          [rr_sift(R.Identity.NEUTRAL,
                                   reason="sift:no-candidate")]):
            sess = self.build(self.good_then_bad(), [rr_neutral()], lc_script)
            self.to_lost(sess)
            for _ in range(12):
                out = sess.step(self.frame)
                self.assertIs(out.state, TrackState.LOST)
            self.assertEqual(sess.tracker.init_calls, 1)
            self.assertEqual([e for e in sess._reacq_events
                              if e["event"] == "REACQ_INIT"], [])

    def test_no_sift_route_without_lc_reference(self):
        # No snapshot -> no LC reference -> the SIFT route never preempts.
        sess = self.build([res(conf=0.05, ok=False)], [rr_neutral()],
                          [rr_sift(R.Identity.MATCH)])
        for _ in range(self.cfg.LOST_AFTER_N_BAD):
            sess.step(self.frame)
        for _ in range(6):
            route, _, dl, _ = self.tick(sess)
            self.assertNotEqual(route, "sift-lc")
            self.assertEqual(dl, 0)


class StubReacqSweep(StubReacqLC):
    """Sweep-surface stub (G2-a): exposes new_lc_builder (completes in one
    tick) and lc_sweep_step (one scripted observation per call — real sweep
    mechanics are covered by test_sift_stripes), so the session's accept-time
    current-frame localization branch runs. best_candidate_lc plays the
    VERIFICATION script and counts the expensive whole-frame fits."""

    def __init__(self, sweep_script, verify_script):
        super().__init__([rr_neutral()], verify_script)
        self.sweep = list(sweep_script)
        self.k = 0
        self.sweep_calls = 0

    def new_lc_builder(self, frame_bgr, point, overlay_mask=None):
        self.build_lc_calls += 1
        self.lc_reference = types.SimpleNamespace(point=point)

        class _OneTickBuilder:
            def step(_self):
                return True
        return _OneTickBuilder()

    def lc_sweep_step(self, frame_bgr):
        self.sweep_calls += 1
        r = self.sweep[min(self.k, len(self.sweep) - 1)]
        self.k += 1
        return r


class TestAcceptTimeLocalization(SiftSessionBase):
    """Option-i acceptance-localization semantics (user-adopted 2026-07-08):
    sweep observations drive persistence only; ONE whole-frame fit on the
    CURRENT frame gates + localizes the accept; a FAILED verification clears
    the hypothesis latch and requires a completely fresh persistence sequence
    before the next expensive fit — never a per-tick retry."""

    def build_sweep(self, sweep_script, verify_script):
        self.stub = StubReacqSweep(sweep_script, verify_script)
        sess = TrackingSession(StubTracker(self.good_then_bad()), self.cfg,
                               enable_reacq=True, reacquirer=self.stub)
        sess.init(self.frame, self.p0)
        return sess

    def to_lost(self, sess):
        for _ in range(self.cfg.REF_UPDATE_EVERY + self.cfg.LOST_AFTER_N_BAD):
            out = sess.step(self.frame)
        assert out.state is TrackState.LOST
        return out

    def test_failed_verification_clears_latch_and_needs_fresh_sequence(self):
        n = self.cfg.REACQ_PERSIST_N
        sess = self.build_sweep(
            [rr_sift(R.Identity.MATCH, reason="sift:all-gates-pass")],
            [rr_sift(R.Identity.AMBIGUOUS, reason="sift:inliers-low"),
             rr_sift(R.Identity.MATCH, x=140.0,
                     reason="sift:all-gates-pass")])
        self.to_lost(sess)
        sess.step(self.frame)                     # builder tick (no observation)
        for _ in range(n):                        # fresh streak -> one-shot #1
            sess.step(self.frame)
        self.assertEqual(self.stub.lc_calls, 1, "one fit at the one-shot")
        self.assertEqual([e for e in sess._reacq_events
                          if e["event"] == "REACQ_INIT"], [],
                         "failed verification must not accept")
        self.assertEqual(sess.tracker.init_calls, 1)
        self.assertEqual(sess._hypo_sift.streak, 0,
                         "failed verification clears the latch/streak")
        for _ in range(n - 1):                    # not yet a fresh sequence
            sess.step(self.frame)
        self.assertEqual(self.stub.lc_calls, 1,
                         "no expensive fit before a fresh full sequence")
        sess.step(self.frame)                     # fresh sequence completes
        self.assertEqual(self.stub.lc_calls, 2)
        inits = [e for e in sess._reacq_events if e["event"] == "REACQ_INIT"]
        self.assertEqual(len(inits), 1)
        self.assertEqual(inits[0]["point"], [140.0, 100.0],
                         "reseed uses the CURRENT-frame fit point, not the "
                         "sweep point")
        self.assertEqual(sess.tracker.init_calls, 2)

    def test_no_per_tick_expensive_retry_after_failure(self):
        n = self.cfg.REACQ_PERSIST_N
        sess = self.build_sweep(
            [rr_sift(R.Identity.MATCH, reason="sift:all-gates-pass")],
            [rr_sift(R.Identity.AMBIGUOUS, reason="sift:inliers-low")])
        self.to_lost(sess)
        sess.step(self.frame)                     # builder tick
        n_obs = 4 * n
        for _ in range(n_obs):
            out = sess.step(self.frame)
            self.assertIs(out.state, TrackState.LOST)
        self.assertEqual(self.stub.sweep_calls, n_obs)
        self.assertEqual(self.stub.lc_calls, n_obs // n,
                         "expensive fits only at one-shot completions, "
                         "never per tick")
        self.assertEqual([e for e in sess._reacq_events
                          if e["event"] == "REACQ_INIT"], [])
        self.assertEqual(sess.tracker.init_calls, 1)


class TestRotatedReturnEndToEnd(unittest.TestCase):
    """Real pipeline (of_kalman + session + real Reacquirer/SIFT) on the v8
    analog: evidence-free flat center + clean textured ring, exit to
    unrelated content, return rotated 160 deg at 1.2x. Templates and ORB are
    measured dead in this regime; only the SIFT context match can reconstruct
    the selected coordinate."""

    def synth_cfg(self):
        cfg = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                       for k in dir(prod_config)
                                       if k.isupper()})
        cfg.REACQ_SIFT_EVERY = 2      # fast test cadence
        cfg.REACQ_SEARCH_EVERY = 1
        cfg.REF_UPDATE_EVERY = 3
        cfg.LOST_AFTER_N_BAD = 10     # faster LOST entry (structural, test-local)
        # The evidence-free center is trackable only via regional scene
        # support — the same profile the real v8 crosshair run uses.
        cfg.EXPERIMENTAL_REGIONAL_MOTION = True
        # This synthetic scene has NO burned-in HUD: pin the overlay mask OFF
        # (the --no-overlay-mask analog) so the default-ON fixed central
        # reticle model (2026-07-08 defaults promotion) does not swallow the
        # scene's central evidence geometry.
        cfg.OVERLAY_MASK_ENABLED = False
        # Synthetic exit/return scene holds static frames during absence (not a
        # real feed) -> the feed-death detector is off here (Commit 4a).
        cfg.FEED_FROZEN_ENABLED = False
        return cfg

    def test_full_arc_tracking_lost_reacq_tracking(self):
        cfg = self.synth_cfg()
        rng = np.random.default_rng(7)
        base = cv2.GaussianBlur(
            rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (3, 3), 0)
        p0 = utils.Point2D(360.0, 260.0)
        # Evidence-free center (the crosshair-masked selection analog): the
        # whole 51px patch is flat -> M8 similarity is None (neutral), and
        # identity can only come from the surrounding clean context.
        base[229:292, 329:392] = 128
        rot = cv2.getRotationMatrix2D((320.0, 240.0), 160.0, 1.2)
        returned = cv2.warpAffine(base, rot, (640, 480),
                                  borderMode=cv2.BORDER_REFLECT)
        gt = (rot @ np.array([p0.x, p0.y, 1.0])).ravel()
        # Absence = continuously CHANGING unrelated content (like the real v8
        # absence): a static noise frame would let the regional tracker
        # positionally re-lock while identity stays neutral (flat reference).
        absent = [cv2.GaussianBlur(
            rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (3, 3), 0)
            for _ in range(45)]

        tracker = trackers.make_tracker("of_kalman", cfg)
        sess = TrackingSession(tracker, cfg, enable_reacq=True)
        sess.init(base, p0)

        phases = [[base] * 12, absent, [returned] * 60]
        states, points, frame_no = [], [], 0
        phase_starts = {}
        for tag, frames in enumerate(phases):
            phase_starts[tag] = frame_no + 1
            for frame in frames:
                frame_no += 1
                out = sess.step(frame)
                states.append((frame_no, out.state))
                points.append((frame_no, out.point.x, out.point.y))

        # Phase 1: tracked + snapshot captured.
        self.assertTrue(any(s is TrackState.TRACKING
                            for f, s in states if f < phase_starts[1]),
                        "phase 1 must reach TRACKING")
        # Phase 2: LOST, and no reacquisition during absence.
        self.assertTrue(any(s is TrackState.LOST
                            for f, s in states if f < phase_starts[2]),
                        "absence must reach LOST")
        inits = [e for e in sess._reacq_events if e["event"] == "REACQ_INIT"]
        self.assertEqual(len(inits), 1, f"expected exactly one REACQ_INIT, "
                         f"got {inits}")
        self.assertEqual(inits[0]["cue"], "sift-lc")
        ix, iy = inits[0]["point"]
        self.assertLess(np.hypot(ix - gt[0], iy - gt[1]), 8.0,
                        f"reseed point ({ix},{iy}) too far from GT "
                        f"({gt[0]:.1f},{gt[1]:.1f})")
        # Return phase: back to TRACKING and locked near GT.
        ret_states = [s for f, s in states if f >= phase_starts[2]]
        self.assertIn(TrackState.TRACKING, ret_states,
                      "must return to TRACKING after reacquisition")
        run = best = 0
        for s in ret_states:
            run = run + 1 if s is TrackState.TRACKING else 0
            best = max(best, run)
        self.assertGreaterEqual(best, 15,
                                "re-lock must hold for a sustained run")
        last = points[-1]
        self.assertLess(np.hypot(last[1] - gt[0], last[2] - gt[1]), 12.0,
                        "committed point must stay near the returned target")


class TestSnapshotFreeze(SiftSessionBase):
    """Per-episode stable-reference lifecycle (v8 nearby-selection
    instability fix, 2026-07-07): the snapshot refreshes during the
    segment's FIRST sustained-trustworthy window and freezes at that
    window's first SUSTAINED quality break — REF_UPDATE_EVERY consecutive
    non-qualifying frames, reusing the existing warm-up constant (no new
    threshold). A fresh learning cycle starts only after PROBATION_OK."""

    def freeze_cfg(self):
        # long LOST fuse: a sustained dip must freeze without entering LOST
        return make_reacq_cfg(REACQ_SIFT_EVERY=3, LOST_AFTER_N_BAD=30)

    def test_sustained_break_freezes_the_snapshot(self):
        cfg = self.freeze_cfg()
        w = cfg.REF_UPDATE_EVERY
        script = ([res(conf=0.9)] * (w + 2) + [res(conf=0.2)] * w
                  + [res(conf=0.9)] * (w + 4))
        sess = self.build(script, [rr_neutral()], [rr_neutral()], cfg=cfg)
        for _ in range(w + 2):
            sess.step(self.frame)
        frozen = sess._lc_frame
        self.assertIsNotNone(frozen)
        for _ in range(w):                     # sustained quality break
            sess.step(self.frame)
        self.assertTrue(sess._lc_frozen)
        for _ in range(w + 4):                 # the track re-qualifies...
            sess.step(self.frame)
        self.assertGreaterEqual(sess._lc_qualify_streak, w,
                                "the re-qualified streak must be long enough "
                                "that a refresh WOULD have fired")
        self.assertIs(sess._lc_frame, frozen,
                      "re-qualified frames must NOT replace a frozen snapshot")

    def test_short_dip_resumes_refreshing(self):
        # The non-qualifying gap includes the M8 TRACKING re-entry lag
        # (~2 frames), so a "short" dip needs a threshold larger than
        # dip + lag: with REF_UPDATE_EVERY=6, a 2-frame dip gaps ~4 frames.
        cfg = make_reacq_cfg(REACQ_SIFT_EVERY=3, LOST_AFTER_N_BAD=30,
                             REF_UPDATE_EVERY=6)
        w = cfg.REF_UPDATE_EVERY
        script = ([res(conf=0.9)] * (w + 2) + [res(conf=0.2)] * 2
                  + [res(conf=0.9)] * (2 * w + 4))
        sess = self.build(script, [rr_neutral()], [rr_neutral()], cfg=cfg)
        for _ in range(w + 2):
            sess.step(self.frame)
        before = sess._lc_frame
        self.assertIsNotNone(before)
        for _ in range(2):                     # dip + re-entry lag < threshold
            sess.step(self.frame)
        self.assertFalse(sess._lc_frozen)
        for _ in range(2 * w + 4):
            sess.step(self.frame)
        self.assertFalse(sess._lc_frozen)
        self.assertIsNot(sess._lc_frame, before,
                         "a short dip must not end the learning window")

    def test_probation_ok_starts_a_new_learning_cycle(self):
        cfg = make_reacq_cfg(REACQ_SIFT_EVERY=3)
        w = cfg.REF_UPDATE_EVERY
        match = rr_sift(R.Identity.MATCH)
        script = ([res(conf=0.9)] * w
                  + [res(conf=0.05, ok=False)] * (cfg.LOST_AFTER_N_BAD + 1)
                  + [res(x=120.0, y=100.0, conf=0.9)] * 40)
        sess = self.build(script, [rr_neutral()], [match], cfg=cfg)
        for _ in range(w):
            sess.step(self.frame)
        frozen = sess._lc_frame
        self.assertIsNotNone(frozen)
        for _ in range(30):                    # LOST -> sift accept -> probation
            sess.step(self.frame)
            if any(e["event"] == "PROBATION_OK" for e in sess._reacq_events):
                break
        self.assertTrue(any(e["event"] == "PROBATION_OK"
                            for e in sess._reacq_events))
        self.assertFalse(sess._lc_frozen, "PROBATION_OK must reopen learning")
        for _ in range(w + 2):                 # the validated segment re-warms
            sess.step(self.frame)
        self.assertIsNot(sess._lc_frame, frozen,
                         "the validated segment must establish a NEW snapshot")
        self.assertFalse(sess._lc_built,
                         "a fresh snapshot invalidates the built reference")

    def test_fallback_reference_reused_without_rebuild(self):
        """If the post-reacquisition segment never re-qualifies, the next
        episode reuses the previously BUILT reference (no rebuild)."""
        # REF_UPDATE_EVERY=6: the ~5 leftover good frames after PROBATION_OK
        # can never re-warm a new snapshot, so the old reference must be the
        # fallback. The tracker script is consumed on EVERY frame (including
        # SEARCHING), hence the explicit filler for the search ticks.
        cfg = make_reacq_cfg(REF_UPDATE_EVERY=6)
        w = cfg.REF_UPDATE_EVERY
        match = rr_sift(R.Identity.MATCH)
        # G2-a recalibration: acceptance now lands after REACQ_PERSIST_N(2)
        # consecutive ticks (observation per tick on the stub surface), so the
        # post-PROBATION_OK leftover must stay < REF_UPDATE_EVERY good frames:
        # 10 - 2 (search) - REACQ_PROBATION_N(3) = 5 < 6 -> no new snapshot.
        script = ([res(conf=0.9)] * w
                  + [res(conf=0.05, ok=False)] * cfg.LOST_AFTER_N_BAD
                  + [res(x=120.0, y=100.0, conf=0.9)] * 10
                  + [res(x=120.0, y=100.0, conf=0.05, ok=False)] * 40)
        sess = self.build(script, [rr_neutral()], [match], cfg=cfg)
        seen_ok = False
        for _ in range(60):
            sess.step(self.frame)
            evs = sess._reacq_events
            if not seen_ok:
                seen_ok = any(e["event"] == "PROBATION_OK" for e in evs)
            if seen_ok and sum(1 for e in evs
                               if e["event"] == "RECOVERY_ENTER") >= 2:
                break
        self.assertTrue(seen_ok)
        self.assertGreaterEqual(sum(1 for e in sess._reacq_events
                                    if e["event"] == "RECOVERY_ENTER"), 2)
        self.assertEqual(self.stub.build_lc_calls, 1,
                         "no new snapshot -> episode-1 reference is reused")

    def test_init_clears_freeze_state(self):
        cfg = self.freeze_cfg()
        w = cfg.REF_UPDATE_EVERY
        script = [res(conf=0.9)] * (w + 1) + [res(conf=0.2)] * w
        sess = self.build(script, [rr_neutral()], [rr_neutral()], cfg=cfg)
        for _ in range(2 * w + 1):
            sess.step(self.frame)
        self.assertTrue(sess._lc_frozen)
        sess.init(self.frame, self.p0)
        self.assertFalse(sess._lc_frozen)
        self.assertEqual(sess._lc_break_len, 0)
        self.assertFalse(sess._lc_seg_fresh)
        self.assertIsNone(sess._lc_frame)


class TestFrameSideMasking(unittest.TestCase):
    """S4 (HUD-evidence contract, 2026-07-07): FRAME-SIDE keypoints on the
    overlay mask are never match targets, on BOTH feature routes. The
    reference side has always been mask-filtered; these tests isolate the
    query side by swapping the installed mask under a fixed reference."""

    def setUp(self):
        self.cfg = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                            for k in dir(prod_config)
                                            if k.isupper()})
        self.cfg.OVERLAY_MASK_ENABLED = True
        rng = np.random.default_rng(11)
        self.scene = cv2.GaussianBlur(
            rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (3, 3), 0)
        self.p = utils.Point2D(320.0, 240.0)
        self.clear = np.zeros((480, 640), np.uint8)
        self.full = np.full((480, 640), 255, np.uint8)

    def test_sift_query_keypoints_on_mask_are_dropped(self):
        reac = R.Reacquirer(self.cfg)
        reac.build_reference(self.scene, self.p, self.clear)
        self.assertIsNotNone(
            reac.build_lc_reference(self.scene, self.p, self.clear))
        self.assertEqual(len(reac.propose_features_lc(self.scene)), 1,
                         "control: the identical frame must match")
        reac._overlay_mask = self.full      # every frame-side kp is on-mask
        self.assertEqual(reac.propose_features_lc(self.scene), [],
                         "masked frame-side keypoints must never be matched")

    def test_orb_query_keypoints_on_mask_are_dropped(self):
        reac = R.Reacquirer(self.cfg)
        reac.build_reference(self.scene, self.p, self.clear)
        self.assertTrue(reac.reference.has_descriptors)
        self.assertEqual(len(reac.propose_features(self.scene)), 1,
                         "control: the identical frame must match")
        reac._overlay_mask = self.full
        self.assertEqual(reac.propose_features(self.scene), [],
                         "masked frame-side keypoints must never be matched")


class TestSnapshotLifecycleEndToEnd(unittest.TestCase):
    """Real pipeline (of_kalman + session + real Reacquirer/SIFT) on the
    fixture's synthetic scene family — the per-episode stable-reference
    lifecycle contract (2026-07-07, user-approved):

      * TWO loss/return episodes reacquire correctly, the second from a
        reference RE-LEARNED after the first PROBATION_OK;
      * the APPROVED LIMITATION stays visible: a hard tracking stall
        (>= warm-up frames) during a photometric appearance morph freezes
        the pre-stall snapshot, so a return matching only the POST-stall
        appearance is honestly terminal-LOST (never a false lock). The
        rolling-refresh policy this replaced handled that case but made
        nearby HUD-covered v8 selections diverge (reacquire vs never) —
        the measured trade was explicitly accepted. If a future change
        makes the dip case reacquire, update AUDIT/docs consciously."""

    W, H = 640, 480
    P0 = utils.Point2D(360.0, 260.0)

    def synth_cfg(self):
        cfg = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                       for k in dir(prod_config)
                                       if k.isupper()})
        cfg.REACQ_SIFT_EVERY = 2
        cfg.REACQ_SEARCH_EVERY = 1
        cfg.REF_UPDATE_EVERY = 3
        cfg.LOST_AFTER_N_BAD = 10
        cfg.EXPERIMENTAL_REGIONAL_MOTION = True
        # HUD-free synthetic scene: pin the overlay mask OFF (see the same
        # pin in TestRotatedReturnEndToEnd.synth_cfg).
        cfg.OVERLAY_MASK_ENABLED = False
        # Synthetic exit/return scene holds static frames during absence (not a
        # real feed) -> the feed-death detector is off here (Commit 4a).
        cfg.FEED_FROZEN_ENABLED = False
        return cfg

    def flat(self, img):
        img = img.copy()
        img[229:292, 329:392] = 128
        return img

    def noise(self, rng):
        return cv2.GaussianBlur(
            rng.integers(0, 256, (self.H, self.W, 3), dtype=np.uint8),
            (3, 3), 0)

    def blend(self, a, b, alpha):
        return self.flat(cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0))

    def warp(self, img, m):
        return cv2.warpAffine(img, m, (self.W, self.H),
                              borderMode=cv2.BORDER_REFLECT)

    def transport(self, m, p):
        v = (m @ np.array([p.x, p.y, 1.0])).ravel()
        return float(v[0]), float(v[1])

    def run_arc(self, phases):
        cfg = self.synth_cfg()
        tracker = trackers.make_tracker("of_kalman", cfg)
        sess = TrackingSession(tracker, cfg, enable_reacq=True)
        sess.init(phases[0][1][0], self.P0)
        states = []
        for _, frames in phases:
            for fr in frames:
                out = sess.step(fr)
                states.append(out.state)
        return states, [dict(e) for e in sess._reacq_events]

    def test_two_episode_relearned_reference(self):
        rng = np.random.default_rng(7)
        n1 = self.flat(self.noise(rng))
        n2 = self.noise(rng)
        n3 = self.noise(rng)
        absent1 = [self.noise(rng) for _ in range(45)]
        absent2 = [self.noise(rng) for _ in range(45)]
        morph1 = [self.blend(n1, n2, a) for a in np.linspace(0.0, 0.8, 24)]
        m1 = cv2.getRotationMatrix2D((320.0, 240.0), 160.0, 1.2)
        gt1 = self.transport(m1, self.P0)
        ret1 = self.warp(self.blend(n1, n2, 0.8), m1)
        n3rot = self.warp(self.flat(n3), m1)
        morph2 = [cv2.addWeighted(ret1, 1.0 - a, n3rot, a, 0.0)
                  for a in np.linspace(0.0, 0.7, 24)]
        m2 = cv2.getRotationMatrix2D((300.0, 250.0), -35.0, 0.9)
        gt2 = self.transport(m2, utils.Point2D(*gt1))
        phases = [("track", morph1), ("absent", absent1),
                  ("return1", [ret1] * 30), ("morph2", morph2),
                  ("absent2", absent2),
                  ("return2", [self.warp(morph2[-1], m2)] * 60)]
        states, events = self.run_arc(phases)
        inits = [e for e in events if e["event"] == "REACQ_INIT"]
        oks = [e for e in events if e["event"] == "PROBATION_OK"]
        self.assertEqual(len(inits), 2,
                         f"expected exactly two REACQ_INITs, got {inits}")
        self.assertGreaterEqual(len(oks), 2)
        for e, gt in zip(inits, (gt1, gt2)):
            ix, iy = e["point"]
            self.assertLess(np.hypot(ix - gt[0], iy - gt[1]), 8.0,
                            f"episode landing ({ix},{iy}) too far from GT {gt}")

    def test_dip_morph_late_return_stays_lost(self):
        """APPROVED SAFETY REGRESSION (visible by design — see class doc)."""
        rng = np.random.default_rng(7)
        n1 = self.flat(self.noise(rng))
        n2 = self.noise(rng)
        _n3 = self.noise(rng)
        absent1 = [self.noise(rng) for _ in range(45)]
        part1 = [self.blend(n1, n2, a) for a in np.linspace(0.0, 0.3, 10)]
        dip = [self.noise(rng) for _ in range(2)]       # hard tracking stall
        part2 = [self.blend(n1, n2, a) for a in np.linspace(0.36, 0.8, 12)]
        m1 = cv2.getRotationMatrix2D((320.0, 240.0), 160.0, 1.2)
        late = self.warp(self.blend(n1, n2, 0.8), m1)
        phases = [("track", part1 + dip + part2), ("absent", absent1),
                  ("return", [late] * 60)]
        states, events = self.run_arc(phases)
        self.assertIn(TrackState.LOST, states, "the arc must reach LOST")
        inits = [e for e in events if e["event"] == "REACQ_INIT"]
        self.assertEqual(inits, [],
                         "the approved limitation is HONEST terminal LOST — "
                         "no reacquisition and, critically, no false lock")


if __name__ == "__main__":
    unittest.main()
