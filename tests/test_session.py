"""Unit tests for the M8 TrackingSession state machine (stdlib unittest only).

The session is driven with a scripted stub tracker and synthetic noise frames,
so every confidence signal is controlled:
  - tracker confidence comes straight from the script,
  - patch similarity is 1.0 while we feed the init frame and ~0 on a new
    noise frame (NCC against the stored reference patch),
  - edge / jump behavior is forced through the scripted points.

Run from the repository root:
    python3 -m unittest discover tests -v
"""
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import utils
from ground_target_tracking.session import SessionResult, TrackingSession, TrackState
from ground_target_tracking.trackers import Tracker


def make_cfg():
    """Self-contained M8 config so tests are immune to production re-tuning."""
    return types.SimpleNamespace(
        PATCH_SIZE=21,
        MIN_PATCH_SIZE=5,
        LOW_CONFIDENCE_BELOW=0.40,
        LOST_CONFIDENCE_BELOW=0.15,
        MIN_PATCH_SIMILARITY=0.30,
        LOST_AFTER_N_BAD=4,
        RECOVER_MARGIN=0.05,
        RECOVER_N=3,
        RECOVER_MIN_SIM=0.50,
        EDGE_MARGIN_PX=8,
        EDGE_PENALTY=0.5,
        MAX_JUMP_PX=42,
        JUMP_VETO_MAX_FRAMES=3,
        REF_UPDATE_MIN_SIM=0.50,
        REF_UPDATE_EVERY=3,
        SIM_SCALES=(0.8, 1.0, 1.25),
        OVERLAY_MASK_ENABLED=False,
    )


class StubTracker(Tracker):
    """Replays a scripted list of TrackResults; repeats the last one forever."""

    def __init__(self, results):
        self.results = list(results)
        self.i = 0
        self.init_calls = 0

    def init(self, frame_bgr, point):
        self.init_calls += 1

    def update(self, frame_bgr):
        r = self.results[min(self.i, len(self.results) - 1)]
        self.i += 1
        return r


def res(x=100.0, y=100.0, ok=True, conf=0.9, source="measure", n=20, err=0.1):
    return utils.TrackResult(point=utils.Point2D(x, y), ok=ok, n_points=n,
                             mean_error=err, source=source, confidence=conf)


class SessionTestBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        self.other_frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        self.p0 = utils.Point2D(100.0, 100.0)

    def make_session(self, results):
        sess = TrackingSession(StubTracker(results), self.cfg)
        sess.init(self.frame, self.p0)
        return sess


class TestStatesAndTransitions(SessionTestBase):
    def test_stays_tracking_on_confident_measurements(self):
        sess = self.make_session([res(conf=0.9)])
        for _ in range(10):
            out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.TRACKING)
        self.assertGreaterEqual(out.confidence, self.cfg.LOW_CONFIDENCE_BELOW)

    def test_degraded_confidence_enters_low_confidence(self):
        sess = self.make_session([res(conf=0.9), res(conf=0.25)])
        sess.step(self.frame)
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)

    def test_recovery_requires_hysteresis(self):
        # 1 degraded frame, then strong frames: TRACKING only after RECOVER_N.
        sess = self.make_session([res(conf=0.25)] + [res(conf=0.9)] * 10)
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for k in range(self.cfg.RECOVER_N - 1):
            out = sess.step(self.frame)
            self.assertIs(out.state, TrackState.LOW_CONFIDENCE,
                          f"recovered too early (good frame #{k + 1})")
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.TRACKING)

    def test_lost_after_consecutive_bad_frames(self):
        sess = self.make_session([res(conf=0.05, ok=False)])
        states = [sess.step(self.frame).state for _ in range(self.cfg.LOST_AFTER_N_BAD)]
        self.assertIs(states[-1], TrackState.LOST)
        self.assertTrue(all(s is not TrackState.LOST for s in states[:-1]),
                        "went LOST before the streak completed")

    def test_lost_is_terminal_and_point_frozen(self):
        bad = [res(conf=0.05, ok=False)] * self.cfg.LOST_AFTER_N_BAD
        good_far = [res(x=150.0, y=150.0, conf=0.95)] * 5
        sess = self.make_session(bad + good_far)
        for _ in range(self.cfg.LOST_AFTER_N_BAD):
            out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOST)
        frozen = (out.point.x, out.point.y)
        for _ in range(5):  # strong measurements must NOT revive the session (M9's job)
            out = sess.step(self.frame)
            self.assertIs(out.state, TrackState.LOST)
            self.assertEqual((out.point.x, out.point.y), frozen)

    def test_predict_state_while_kalman_coasts(self):
        sess = self.make_session([res(conf=0.9),
                                  res(ok=True, conf=0.6, source="predict", n=0)])
        sess.step(self.frame)
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.PREDICT)

    def test_coast_exhaustion_accumulates_to_lost_not_instantly(self):
        # Exhausted coasting is bad-frame evidence, not an instant LOST: a brief
        # full dropout (camera shake) must be able to recover.
        sess = self.make_session([res(conf=0.9),
                                  res(ok=False, conf=0.1, source="predict", n=0)])
        sess.step(self.frame)
        for k in range(self.cfg.LOST_AFTER_N_BAD - 1):
            out = sess.step(self.frame)
            self.assertIs(out.state, TrackState.LOW_CONFIDENCE,
                          f"left LOW_CONFIDENCE too early (bad frame #{k + 1})")
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOST)

    def test_brief_dropout_recovers_without_lost(self):
        dropout = [res(ok=False, conf=0.1, source="predict", n=0)] * (
            self.cfg.LOST_AFTER_N_BAD - 2)
        sess = self.make_session([res(conf=0.9)] + dropout + [res(conf=0.9)] * 10)
        outs = [sess.step(self.frame) for _ in range(len(dropout) + 11)]
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs),
                        "a dropout shorter than the streak must not become LOST")
        self.assertIs(outs[-1].state, TrackState.TRACKING)


class TestSignals(SessionTestBase):
    def test_jump_veto_rejects_the_frame_but_keeps_the_point(self):
        sess = self.make_session([res(conf=0.9),
                                  res(x=190.0, y=190.0, conf=0.9),  # 127px jump > 42
                                  res(conf=0.9)])
        sess.step(self.frame)
        out = sess.step(self.frame)
        self.assertTrue(out.signals["jump_vetoed"])
        self.assertEqual(out.confidence, 0.0)
        self.assertEqual((out.point.x, out.point.y), (100.0, 100.0),
                         "a vetoed jump must not move the committed point")
        self.assertIsNot(out.state, TrackState.LOST, "one vetoed frame must not mean LOST")

    def test_persistent_jump_is_accepted_as_real_motion(self):
        # A displacement that persists past JUMP_VETO_MAX_FRAMES is a fast pan /
        # shake, not a glitch: the session must re-trust the tracker instead of
        # veto-spiraling into LOST. In a real pan the object travels with the
        # frame, so its appearance at the new location still matches the
        # reference — model that by pasting the object patch at the new spot.
        panned = self.frame.copy()
        panned[170:191, 170:191] = self.frame[90:111, 90:111]
        far = [res(x=180.0, y=180.0, conf=0.9)] * 10
        sess = self.make_session([res(conf=0.9)] + far)
        sess.step(self.frame)
        outs = [sess.step(panned) for _ in range(self.cfg.JUMP_VETO_MAX_FRAMES + 1)]
        self.assertTrue(all(o.signals["jump_vetoed"] for o in outs[:-1]))
        self.assertFalse(outs[-1].signals["jump_vetoed"],
                         "persistent motion was never accepted")
        self.assertEqual((outs[-1].point.x, outs[-1].point.y), (180.0, 180.0))
        self.assertIsNot(outs[-1].state, TrackState.LOST)

    def test_similarity_collapse_drives_lost_despite_confident_tracker(self):
        # The official-sample-video false-lock class: tracker is confident, but the ROI content
        # no longer resembles the reference patch -> similarity kills it.
        sess = self.make_session([res(conf=0.95)])
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.TRACKING)
        for _ in range(self.cfg.LOST_AFTER_N_BAD):
            out = sess.step(self.other_frame)  # unrelated content at the same point
        self.assertIs(out.state, TrackState.LOST)

    def test_edge_penalty_reduces_confidence(self):
        sess = self.make_session([res(conf=0.9), res(x=5.0, y=100.0, conf=0.9)])
        sess.step(self.frame)
        out = sess.step(self.frame)
        self.assertTrue(out.signals["edge"])
        self.assertLessEqual(out.confidence,
                             0.9 * self.cfg.EDGE_PENALTY + 1e-9)

    def test_similarity_is_one_on_the_init_frame(self):
        sess = self.make_session([res(conf=0.9)])
        out = sess.step(self.frame)
        self.assertIsNotNone(out.signals["similarity"])
        self.assertGreater(out.signals["similarity"], 0.99)

    def test_scaled_object_keeps_similarity_alive(self):
        # A legitimate scale change (drone approach) must not collapse the
        # similarity signal: the object rendered at 1.25x its reference size
        # should still score well via the scale-tolerant NCC.
        import cv2
        ref_region = self.frame[90:111, 90:111]          # the 21px init patch
        upscaled = cv2.resize(ref_region, (26, 26), interpolation=cv2.INTER_AREA)
        scaled_frame = self.frame.copy()
        scaled_frame[87:113, 87:113] = upscaled          # object now 1.25x bigger
        sess = self.make_session([res(conf=0.9)])
        out = sess.step(scaled_frame)
        self.assertIsNotNone(out.signals["similarity"])
        self.assertGreater(out.signals["similarity"], 0.5,
                           "scale-tolerant similarity failed on a 1.25x object")
        # ...while unrelated content still scores low at every scale:
        sess2 = self.make_session([res(conf=0.9)])
        out2 = sess2.step(self.other_frame)
        self.assertLess(out2.signals["similarity"], 0.3)

    def test_flat_reference_makes_similarity_neutral(self):
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        sess = TrackingSession(StubTracker([res(conf=0.9)]), self.cfg)
        sess.init(flat, self.p0)
        out = sess.step(flat)
        self.assertIsNone(out.signals["similarity"])
        self.assertIs(out.state, TrackState.TRACKING)


class TestAdaptiveReference(SessionTestBase):
    """The conservative reference-update gate (approved M8 design change)."""

    def setUp(self):
        super().setUp()
        # A blend of the init frame and unrelated noise: similar enough to pass
        # the update gate (NCC ~0.8 vs init), different enough to detect a swap.
        self.blend = (0.6 * self.frame + 0.4 * self.other_frame).astype(np.uint8)

    def test_reference_updates_after_throttle_on_strong_frames(self):
        sess = self.make_session([res(conf=0.9)])
        ref0 = sess._ref_patch.copy()
        for _ in range(self.cfg.REF_UPDATE_EVERY - 1):
            sess.step(self.blend)
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "updated before the throttle count was reached")
        sess.step(self.blend)
        self.assertFalse(np.array_equal(sess._ref_patch, ref0),
                         "reference did not follow strong TRACKING frames")

    def test_reference_never_updates_from_dissimilar_content(self):
        sess = self.make_session([res(conf=0.9)])
        ref0 = sess._ref_patch.copy()
        for _ in range(self.cfg.REF_UPDATE_EVERY * 3):
            sess.step(self.other_frame)  # confident tracker, but NCC ~0 < gate
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "junk content leaked into the reference")

    # The old contract test ("never updates outside TRACKING state") is split:
    # the gate is now keyed to the POSITIONAL axis, so positional degradation
    # still freezes the reference, while identity decay alone no longer blocks
    # its own repair (the self-freeze fix).

    def test_reference_frozen_under_positional_degradation(self):
        # tracker_conf < LOW_CONFIDENCE_BELOW: no positional trust -> the
        # reference must never update, whatever the similarity says.
        sess = self.make_session([res(conf=0.2)])
        ref0 = sess._ref_patch.copy()
        out = None
        for k in range(self.cfg.REF_UPDATE_EVERY * 3):
            out = sess.step(self.blend)
            self.assertEqual(out.signals["ref_gate"], "low-tracker-conf")
            self.assertEqual(out.signals["ref_staleness"], k + 1,
                             "staleness must grow while the gate blocks")
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "reference updated under positional degradation")

    def test_reference_updates_under_identity_decay_with_positional_confidence(self):
        # The self-freeze scenario: identity decay drops the COMBINED state to
        # LOW_CONFIDENCE and (with a strict recovery floor) blocks recovery, so
        # under the old state-keyed gate the reference could never follow the
        # appearance and the session froze permanently. The positional-axis
        # gate must keep updating (tracker_conf high, sim >= REF_UPDATE_MIN_SIM
        # vs the CURRENT ref) — and that update is what unlocks recovery.
        self.cfg.RECOVER_MIN_SIM = 0.99      # identity axis blocks recovery
        sess = self.make_session([res(conf=0.2)] + [res(conf=0.9)] * 20)
        ref0 = sess._ref_patch.copy()
        out = sess.step(self.blend)          # weak frame -> LOW_CONFIDENCE
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for _ in range(self.cfg.REF_UPDATE_EVERY):
            out = sess.step(self.blend)      # decayed identity, confident tracker
            self.assertIsNot(out.state, TrackState.TRACKING,
                             "precondition: combined state must stay degraded")
            self.assertGreaterEqual(out.signals["similarity"],
                                    self.cfg.REF_UPDATE_MIN_SIM,
                                    "precondition: continuity floor must hold")
        self.assertFalse(np.array_equal(sess._ref_patch, ref0),
                         "self-freeze: reference did not follow identity decay "
                         "despite positional confidence")
        self.assertEqual(out.signals["ref_gate"], "updated")
        self.assertEqual(out.signals["ref_staleness"], 0)
        for _ in range(self.cfg.RECOVER_N):  # sim vs updated ref ~1.0 now
            out = sess.step(self.blend)
        self.assertIs(out.state, TrackState.TRACKING,
                      "the reference update must unlock recovery")

    def test_gradual_appearance_change_never_freezes(self):
        # Legitimate slow drift: the reference keeps re-snapshotting through
        # the throttle, staleness stays bounded, and the session never decays
        # toward LOST while the tracker is confident.
        sess = self.make_session([res(conf=0.9)])
        ref0 = sess._ref_patch.copy()
        max_staleness = 0
        for k in range(1, 25):
            alpha = min(0.9, 0.03 * k)      # ~3% more unrelated content per frame
            drift = ((1.0 - alpha) * self.frame
                     + alpha * self.other_frame).astype(np.uint8)
            out = sess.step(drift)
            self.assertIsNot(out.state, TrackState.LOST)
            max_staleness = max(max_staleness, out.signals["ref_staleness"])
        self.assertFalse(np.array_equal(sess._ref_patch, ref0),
                         "reference froze during legitimate gradual change")
        self.assertLessEqual(max_staleness, self.cfg.REF_UPDATE_EVERY,
                             "staleness must stay bounded by the throttle "
                             "while every frame qualifies")

    def test_poisoning_still_blocked(self):
        # Every anti-poisoning guard survives the positional-axis gate; each
        # blocked frame reports its reason and the reference stays bit-identical.
        # (a) dissimilar content — continuity floor:
        sess = self.make_session([res(conf=0.9)])
        ref0 = sess._ref_patch.copy()
        for _ in range(self.cfg.LOST_AFTER_N_BAD - 1):
            out = sess.step(self.other_frame)
            self.assertEqual(out.signals["ref_gate"], "low-sim")
        self.assertTrue(np.array_equal(sess._ref_patch, ref0))
        # (b) edge proximity (target initialized near the border — no jump):
        sess = TrackingSession(StubTracker([res(x=15.0, y=100.0, conf=0.9)]),
                               self.cfg)
        sess.init(self.frame, utils.Point2D(15.0, 100.0))
        ref0 = sess._ref_patch.copy()
        out = sess.step(self.frame)
        self.assertEqual(out.signals["ref_gate"], "edge")
        self.assertTrue(np.array_equal(sess._ref_patch, ref0))
        # (c) jump veto:
        sess = self.make_session([res(conf=0.9),
                                  res(x=100.0 + self.cfg.MAX_JUMP_PX + 5.0,
                                      y=100.0, conf=0.9)])
        sess.step(self.frame)
        ref0 = sess._ref_patch.copy()
        out = sess.step(self.frame)
        self.assertEqual(out.signals["ref_gate"], "veto")
        self.assertTrue(np.array_equal(sess._ref_patch, ref0))
        # (d) coasted prediction is never reference material:
        sess = self.make_session([res(conf=0.9, source="predict", ok=True)])
        ref0 = sess._ref_patch.copy()
        out = sess.step(self.frame)
        self.assertEqual(out.signals["ref_gate"], "not-measured")
        self.assertTrue(np.array_equal(sess._ref_patch, ref0))

    def test_occlusion_freezes_reference_not_poisons(self):
        # A brief occlusion (unrelated ROI content) blocks updates and grows
        # staleness; on emergence the gate resumes and snapshots again.
        sess = self.make_session([res(conf=0.9)])
        ref0 = sess._ref_patch.copy()
        for _ in range(self.cfg.LOST_AFTER_N_BAD - 2):   # occluded, short of LOST
            out = sess.step(self.other_frame)
            self.assertEqual(out.signals["ref_gate"], "low-sim")
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "occluder content leaked into the reference")
        staleness_during = out.signals["ref_staleness"]
        self.assertGreater(staleness_during, 0)
        out = None
        # Every clean frame qualifies, so the REF_UPDATE_EVERY-th one snapshots
        # (loop length lands exactly on an update frame).
        for _ in range(2 * self.cfg.REF_UPDATE_EVERY):
            out = sess.step(self.frame)                  # target re-emerges
        self.assertEqual(out.signals["ref_gate"], "updated",
                         "updates must resume after the occlusion clears")
        self.assertEqual(out.signals["ref_staleness"], 0)

    def test_lost_behavior_unchanged(self):
        # LOST semantics are untouched by the gate change: the reference and
        # the committed point stay frozen once LOST is declared.
        sess = self.make_session([res(conf=0.05, ok=False)])
        ref0 = sess._ref_patch.copy()
        for _ in range(self.cfg.LOST_AFTER_N_BAD):
            out = sess.step(self.other_frame)
        self.assertIs(out.state, TrackState.LOST)
        frozen = (out.point.x, out.point.y)
        for _ in range(5):
            out = sess.step(self.other_frame)
        self.assertIs(out.state, TrackState.LOST)
        self.assertEqual((out.point.x, out.point.y), frozen)
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "reference changed after LOST")

    def test_non_qualifying_frame_resets_the_throttle(self):
        sess = self.make_session(
            [res(conf=0.9)] * (self.cfg.REF_UPDATE_EVERY - 1) +
            [res(conf=0.2)] +
            [res(conf=0.9)] * (self.cfg.REF_UPDATE_EVERY - 1))
        ref0 = sess._ref_patch.copy()
        for _ in range(2 * self.cfg.REF_UPDATE_EVERY - 1):
            sess.step(self.blend)
        self.assertTrue(np.array_equal(sess._ref_patch, ref0),
                        "throttle survived a non-qualifying frame")


class TestRecoveryRevalidation(SessionTestBase):
    """Stage 1: return to TRACKING requires a RE-VALIDATED reliable measurement,
    not merely N consecutive strong frames (which a manufactured lock satisfies)."""

    def test_unreliable_measure_does_not_count_toward_recovery(self):
        # A stream of high-CONFIDENCE but UNRELIABLE (ok=False) measure frames
        # must not promote back to TRACKING — the old count-only path would.
        # Confidence stays high (similarity ~1.0 vs the init frame) so it never
        # goes LOST either: it must sit honestly in LOW_CONFIDENCE.
        sess = self.make_session([res(conf=0.25)] +
                                 [res(conf=0.9, ok=False)] * 10)
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for _ in range(self.cfg.RECOVER_N + 4):
            out = sess.step(self.frame)
            self.assertIs(out.state, TrackState.LOW_CONFIDENCE,
                          "unreliable measurements must not promote to TRACKING")

    def test_low_similarity_blocks_recovery(self):
        # The false-lock signature: reliable, confident measurements whose ROI
        # content does NOT match the reference well enough. First drop to
        # LOW_CONFIDENCE (one weak frame), then feed strong blend frames whose
        # similarity is below RECOVER_MIN_SIM: recovery must stay blocked even
        # though the combined confidence clears the strong/recover threshold.
        # Revised for the positional-axis update gate: the continuity floor is
        # raised alongside, otherwise the reference would LEGITIMATELY adapt to
        # the blend (sim ~0.8 >= REF_UPDATE_MIN_SIM with a confident tracker)
        # and then recover — the documented, accepted adaptation path. This
        # test isolates the recovery gate: content the reference cannot adopt
        # must never re-validate to TRACKING.
        self.cfg.RECOVER_MIN_SIM = 0.99
        self.cfg.REF_UPDATE_MIN_SIM = 0.99
        blend = (0.6 * self.frame + 0.4 * self.other_frame).astype(np.uint8)
        sess = self.make_session([res(conf=0.1)] + [res(conf=0.95)] * 12)
        out = sess.step(blend)                       # weak frame -> LOW_CONFIDENCE
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for _ in range(self.cfg.RECOVER_N + 4):
            out = sess.step(blend)                   # strong, but similarity < gate
            self.assertIs(out.state, TrackState.LOW_CONFIDENCE,
                          "low-similarity content must not re-validate to TRACKING")

    def test_neutral_similarity_flat_target_can_recover(self):
        # Guard the neutral-pass refinement: a flat/low-texture target has
        # similarity None (NCC undefined). None must be treated as neutral so a
        # genuine strong measured re-lock still recovers — a naive "similarity
        # must exceed the gate" would strand flat targets in LOW_CONFIDENCE.
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        sess = TrackingSession(
            StubTracker([res(conf=0.2)] + [res(conf=0.9)] * 10), self.cfg)
        sess.init(flat, self.p0)
        out = sess.step(flat)
        self.assertIsNone(out.signals["similarity"])
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for _ in range(self.cfg.RECOVER_N):
            out = sess.step(flat)
        self.assertIs(out.state, TrackState.TRACKING,
                      "a flat target (neutral similarity) must still recover")


class TestO1UnknownNotConfirmation(SessionTestBase):
    """O1 (user-adopted 2026-07-08): similarity None while the REFERENCE is
    informative (sim_unknown) is neutral for the re-validation streak — it
    neither advances nor resets it. Only a real similarity >= RECOVER_MIN_SIM
    advances; the legacy flat-reference carve-out (reference uninformative)
    keeps None as confirming. Measured basis: unknown-as-confirming certified
    TRACKING on ambient desert for 9 reads while the target was absent."""

    def scripted_sim(self, sess, sims):
        seq = list(sims)

        def fake(frame_bgr, point, _seq=seq):
            return _seq.pop(0) if _seq else None
        sess._patch_similarity = fake

    def drop_then(self, results):
        # One degraded frame drops TRACKING -> LOW_CONFIDENCE, then `results`.
        return [res(conf=0.25)] + results

    def test_unknown_does_not_advance_the_streak(self):
        n = self.cfg.RECOVER_N
        sess = self.make_session(self.drop_then([res(conf=0.9)] * (3 * n)))
        # init consumes no similarity; step 1 drops to LOW_CONFIDENCE.
        self.scripted_sim(sess, [0.2] + [None] * (3 * n))
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for _ in range(3 * n):
            out = sess.step(self.frame)
            self.assertIs(out.state, TrackState.LOW_CONFIDENCE,
                          "unknown frames must not certify recovery")

    def test_unknown_holds_but_does_not_reset_the_streak(self):
        n = self.cfg.RECOVER_N          # 3
        sess = self.make_session(self.drop_then([res(conf=0.9)] * 10))
        # drop, then real/real/None/real: the None HOLDS the 2-streak, the
        # final real frame completes RECOVER_N=3 -> TRACKING.
        self.scripted_sim(sess, [0.2, 0.9, 0.9, None, 0.9])
        sess.step(self.frame)                       # LOW_CONFIDENCE
        sess.step(self.frame)                       # streak 1
        sess.step(self.frame)                       # streak 2
        out = sess.step(self.frame)                 # None -> hold
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        out = sess.step(self.frame)                 # streak 3 -> TRACKING
        self.assertIs(out.state, TrackState.TRACKING,
                      "an unknown frame must not reset the streak")

    def test_flat_reference_carve_out_keeps_none_confirming(self):
        n = self.cfg.RECOVER_N
        flat = np.full((200, 200, 3), 128, np.uint8)
        sess = TrackingSession(StubTracker(
            self.drop_then([res(conf=0.9)] * (n + 2))), self.cfg)
        sess.init(flat, self.p0)                    # flat ref: _ref_std ~ 0
        out = sess.step(flat)
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        for _ in range(n):
            out = sess.step(flat)
        self.assertIs(out.state, TrackState.TRACKING,
                      "identity-incapable reference keeps legacy neutrality")


class TestDecisionBaseline(SessionTestBase):
    """Stage 1: the jump-veto / recovery baseline is the last MEASURED-accepted
    point, distinct from the (possibly coasting) display point."""

    def test_coast_prediction_does_not_move_the_veto_baseline(self):
        # A within-budget coast GLIDES the display point (small predicted step),
        # but the veto/recovery baseline must stay at the last MEASURED point so
        # a later measurement is judged against the verified location, not the
        # drifted prediction. Geometry: glide 100->110 (<MAX_JUMP_PX), then a
        # measure at 100+MAX_JUMP_PX+3 is >MAX_JUMP_PX from the baseline (100)
        # yet only MAX_JUMP_PX-7 from the glided display point (110): the split
        # is what makes it a veto.
        glide_x = 110.0
        far_x = 100.0 + self.cfg.MAX_JUMP_PX + 3.0
        sess = self.make_session([
            res(conf=0.9),                                            # measure @100
            res(x=glide_x, y=100.0, ok=True, conf=0.6, source="predict"),  # coast
            res(x=far_x, y=100.0, conf=0.9),                          # real measure
        ])
        sess.step(self.frame)                                        # baseline @100
        coast = sess.step(self.frame)                                # PREDICT, glide
        self.assertIs(coast.state, TrackState.PREDICT)
        self.assertEqual((coast.point.x, coast.point.y), (glide_x, 100.0),
                         "display point should glide with the within-budget coast")
        self.assertEqual((sess._accepted_point.x, sess._accepted_point.y),
                         (100.0, 100.0),
                         "coast prediction must not move the measured baseline")
        out = sess.step(self.frame)                                  # far real measure
        self.assertTrue(out.signals["jump_vetoed"],
                        "teleport judged against the verified baseline is vetoed")


class TestResultShape(SessionTestBase):
    def test_session_result_carries_inner_result_and_signals(self):
        sess = self.make_session([res(conf=0.9, n=17)])
        out = sess.step(self.frame)
        self.assertIsInstance(out, SessionResult)
        self.assertEqual(out.result.n_points, 17)
        for key in ("tracker_conf", "similarity", "edge", "jump_vetoed", "bad_streak"):
            self.assertIn(key, out.signals)


class TestOffFrameHardening(SessionTestBase):
    """Stage 2A: a point outside the frame is unobservable and must count as
    loss evidence when LOST_OFFFRAME_HARDENING is on — including when the
    measurement claims ok/confident (the regional off-frame defect) and when
    border-clamping would land the point on the HUD overlay (suspension
    freeze). Every OTHER test in this file runs on a config WITHOUT the flag
    (getattr default False), which is itself the proof that absent-flag
    behavior is unchanged.

    Flat frames keep similarity None (neutral) throughout — the same
    sim-None regime measured on the official sample video, where evidence
    reduces to tracker confidence alone.
    """

    def setUp(self):
        super().setUp()
        self.flat = np.full((200, 200, 3), 128, dtype=np.uint8)

    def _walk_off_bottom(self, hold_y=240.0):
        # Measured-ok, confident steps of +20px (< MAX_JUMP_PX: never vetoed)
        # walking off the 200px frame; the last result repeats forever.
        steps = [res(y=100.0 + 20.0 * k, conf=0.9) for k in range(1, 8)]
        steps[-1] = res(y=hold_y, conf=0.9)
        return steps

    def _flat_session(self, results, cfg):
        sess = TrackingSession(StubTracker(results), cfg)
        sess.init(self.flat, self.p0)
        return sess

    def test_off_frame_measured_ok_never_lost_today(self):
        # Defect #1 documented (flag OFF = today): confident "measured"
        # frames at an off-frame point are never bad — the session keeps
        # reporting TRACKING on a coordinate below the image forever.
        sess = self._flat_session(self._walk_off_bottom(), self.cfg)
        outs = [sess.step(self.flat) for _ in range(12)]
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs),
                        "expected today's behavior: off-frame never reaches LOST")
        self.assertIs(outs[-1].state, TrackState.TRACKING)
        self.assertGreaterEqual(outs[-1].point.y, 200.0,
                                "the walk should have left the frame")

    def test_off_frame_measured_ok_reaches_lost_with_hardening(self):
        self.cfg.LOST_OFFFRAME_HARDENING = True
        self.cfg.LOST_OFFFRAME_MARGIN_PX = 0
        sess = self._flat_session(self._walk_off_bottom(), self.cfg)
        # Steps 1-4 in-frame (y=120..180); step 5 (y=200) is the first
        # off-frame frame; LOST after LOST_AFTER_N_BAD consecutive ones.
        outs = [sess.step(self.flat) for _ in range(4 + self.cfg.LOST_AFTER_N_BAD)]
        self.assertTrue(all(o.state is not TrackState.LOST
                            for o in outs[:-1]),
                        "went LOST before the off-frame streak completed")
        self.assertIs(outs[-1].state, TrackState.LOST)
        frozen = (outs[-1].point.x, outs[-1].point.y)
        for _ in range(4):
            out = sess.step(self.flat)
            self.assertIs(out.state, TrackState.LOST)
            self.assertEqual((out.point.x, out.point.y), frozen)

    def test_off_frame_bypasses_overlay_suspension(self):
        # Defect #2: with the whole border inside the dilated overlay, a bad
        # off-frame stream freezes the streak today (clamped point lands on
        # overlay); the hardening must bypass the suspension.
        coast_dead = [res(conf=0.9),
                      res(x=100.0, y=250.0, ok=False, conf=0.05,
                          source="predict", n=0)]
        sess_off = self._flat_session(coast_dead, self.cfg)
        sess_off._overlay_dil = np.full((200, 200), 255, np.uint8)
        outs = [sess_off.step(self.flat) for _ in range(12)]
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs),
                        "expected today's freeze: overlay suspension blocks LOST")

        self.cfg.LOST_OFFFRAME_HARDENING = True
        sess_on = self._flat_session(coast_dead, self.cfg)
        sess_on._overlay_dil = np.full((200, 200), 255, np.uint8)
        sess_on.step(self.flat)                       # good frame
        for k in range(self.cfg.LOST_AFTER_N_BAD - 1):
            out = sess_on.step(self.flat)
            self.assertIsNot(out.state, TrackState.LOST,
                             f"went LOST too early (bad frame #{k + 1})")
        out = sess_on.step(self.flat)
        self.assertIs(out.state, TrackState.LOST,
                      "off-frame must bypass the known-occlusion suspension")

    def test_margin_keeps_near_border_points_in_frame(self):
        self.cfg.LOST_OFFFRAME_HARDENING = True
        self.cfg.LOST_OFFFRAME_MARGIN_PX = 30
        near = [res(y=100.0 + 20.0 * k, conf=0.9) for k in range(1, 7)]  # ...220
        sess = self._flat_session(near, self.cfg)
        outs = [sess.step(self.flat) for _ in range(12)]
        self.assertTrue(all(o.state is not TrackState.LOST for o in outs),
                        "a point within the margin band must not count as off-frame")
        far = self._walk_off_bottom(hold_y=260.0)     # beyond 200 + 30
        sess2 = self._flat_session(far, self.cfg)
        outs2 = [sess2.step(self.flat) for _ in range(7 + self.cfg.LOST_AFTER_N_BAD)]
        self.assertIs(outs2[-1].state, TrackState.LOST,
                      "beyond the margin band the hardening must engage")


# =========================================================================== #
# M9-c: reacquisition integration into the session lifecycle.
# Driven with a StubReacq (scripted ReacqResults) + StubTracker, so LOST ->
# search -> reseed -> probation is fully deterministic. The REAL M9-b
# HypothesisTracker runs inside the session (not stubbed). Nothing here changes
# the production default; every reacq test passes enable_reacq=True explicitly.
# =========================================================================== #
from ground_target_tracking.reacquisition import Identity, ReacqResult


def make_reacq_cfg(**over):
    cfg = make_cfg()
    cfg.LOST_AFTER_N_BAD = 4
    # M9-b HypothesisTracker (short for fast tests):
    cfg.REACQ_PERSIST_N = 2
    cfg.REACQ_PERSIST_MAX_MOVE_PX = 30.0
    cfg.REACQ_PERSIST_SCALE_TOL = 0.25
    cfg.REACQ_PERSIST_MAX_NEUTRAL = 2
    # M9-c integration:
    cfg.REACQ_SEARCH_EVERY = 1
    cfg.REACQ_PROBATION_N = 3
    cfg.REACQ_PROBATION_MAX_FRAMES = 6
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class StubReacq:
    """Scripted Reacquirer: build_reference is a no-op; best_candidate replays
    a list of ReacqResults (repeating the last). Counts executed evaluations."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = 0
        self.ref_built = False

    def build_reference(self, frame_bgr, point, overlay_mask=None):
        self.ref_built = True

    def best_candidate(self, frame_bgr):
        self.calls += 1
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return r


def rr_match(x=100.0, y=100.0, scale=1.0, cue="orb-feature", confirm=0.9):
    return ReacqResult(Identity.MATCH, utils.Point2D(x, y), scale, cue, confirm, "stub")


def rr_neutral():
    return ReacqResult(Identity.NEUTRAL, None, None, None, None, "stub")


def rr_ambiguous():
    return ReacqResult(Identity.AMBIGUOUS, utils.Point2D(0.0, 0.0), 1.0,
                       "orb-feature", 0.1, "stub")


class ReacqSessionBase(unittest.TestCase):
    def setUp(self):
        self.cfg = make_reacq_cfg()
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        self.p0 = utils.Point2D(100.0, 100.0)

    def build(self, tracker_script, reacq_script, cfg=None, enable=True):
        cfg = cfg or self.cfg
        sess = TrackingSession(StubTracker(tracker_script), cfg,
                               enable_reacq=enable, reacquirer=StubReacq(reacq_script))
        sess.init(self.frame, self.p0)
        return sess

    def to_lost(self, sess, n=None):
        """Step bad frames until LOST (M8 owns this)."""
        n = n or self.cfg.LOST_AFTER_N_BAD
        out = None
        for _ in range(n):
            out = sess.step(self.frame)
        assert out.state is TrackState.LOST, "expected M8 to declare LOST"
        return out

    def events(self, sess, name):
        return [e for e in sess._reacq_events if e["event"] == name]


class TestReacqDisabled(ReacqSessionBase):
    def test_disabled_flag_preserves_existing_behavior(self):
        sess = self.build([res(conf=0.05, ok=False)], [rr_match()], enable=False)
        stub = sess._injected_reacq
        for _ in range(self.cfg.LOST_AFTER_N_BAD + 6):
            out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOST, "LOST stays terminal when disabled")
        self.assertIsNone(sess._reacq, "no Reacquirer built when disabled")
        self.assertEqual(stub.calls, 0, "best_candidate never called when disabled")
        self.assertEqual(sess._reacq_events, [], "no recovery events when disabled")
        self.assertEqual(sess.tracker.init_calls, 1, "no reseed when disabled")


class TestRecoveryEntry(ReacqSessionBase):
    def test_sustained_failure_enters_recovery_once(self):
        sess = self.build([res(conf=0.05, ok=False)], [rr_neutral()])
        self.to_lost(sess)
        sess.step(self.frame)                     # first LOST frame -> recovery entry
        sess.step(self.frame)
        self.assertEqual(len(self.events(sess, "RECOVERY_ENTER")), 1)
        self.assertEqual(sess._reacq_episode, 1)

    def test_recovery_creates_one_hypothesis_tracker(self):
        sess = self.build([res(conf=0.05, ok=False)], [rr_neutral()])
        self.to_lost(sess)
        sess.step(self.frame)
        h1 = sess._hypo
        self.assertIsNotNone(h1)
        for _ in range(4):
            sess.step(self.frame)
        self.assertIs(sess._hypo, h1, "HypothesisTracker not recreated each SEARCHING frame")

    def test_no_reacquisition_during_healthy_tracking(self):
        sess = self.build([res(conf=0.9)], [rr_match()])
        for _ in range(10):
            out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.TRACKING)
        self.assertEqual(sess._injected_reacq.calls, 0,
                         "best_candidate must not run during healthy tracking")


class TestReacqInit(ReacqSessionBase):
    def test_match_without_persistence_does_not_init(self):
        # one MATCH then NEUTRAL forever -> streak never reaches PERSIST_N=2.
        sess = self.build([res(conf=0.05, ok=False)], [rr_match(), rr_neutral()])
        self.to_lost(sess)
        for _ in range(8):
            sess.step(self.frame)
        self.assertEqual(sess.tracker.init_calls, 1, "no reseed without persistence")
        self.assertEqual(self.events(sess, "REACQ_INIT"), [])

    def test_accepted_hypothesis_calls_tracker_init_once(self):
        sess = self.build([res(conf=0.05, ok=False)] * 4 + [res(conf=0.9)],
                          [rr_match()])
        self.to_lost(sess)
        sess.step(self.frame)                     # SEARCHING eval1 (streak 1)
        out = sess.step(self.frame)               # eval2 (streak 2) -> accept + reseed
        self.assertEqual(sess.tracker.init_calls, 2, "exactly one reseed init")
        self.assertEqual(len(self.events(sess, "REACQ_INIT")), 1)
        self.assertIs(out.state, TrackState.LOW_CONFIDENCE)
        self.assertIs(sess._reacq_phase, __import__("ground_target_tracking.session",
                      fromlist=["_Phase"])._Phase.PROBATION)

    def test_continued_frames_do_not_repeat_init(self):
        sess = self.build([res(conf=0.05, ok=False)] * 4 + [res(conf=0.9)],
                          [rr_match()])
        self.to_lost(sess)
        for _ in range(10):
            sess.step(self.frame)
        self.assertEqual(sess.tracker.init_calls, 2,
                         "one-shot latch + phase change prevent repeat init")
        self.assertEqual(len(self.events(sess, "REACQ_INIT")), 1)

    def test_offframe_acceptance_does_not_init(self):
        sess = self.build([res(conf=0.05, ok=False)],
                          [rr_match(x=999.0, y=999.0)])   # off a 200x200 frame
        self.to_lost(sess)
        for _ in range(6):
            sess.step(self.frame)
        self.assertEqual(sess.tracker.init_calls, 1, "off-frame acceptance must not init")
        self.assertTrue(self.events(sess, "INIT_SKIPPED"))

    def test_no_session_mutation_from_raw_neutral_results(self):
        sess = self.build([res(conf=0.05, ok=False)], [rr_neutral()])
        self.to_lost(sess)
        frozen = sess.point
        for _ in range(6):
            out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOST)
        self.assertEqual((out.point.x, out.point.y), (frozen.x, frozen.y),
                         "committed point stays frozen under NEUTRAL search")
        self.assertEqual(sess.tracker.init_calls, 1)


class TestProbation(ReacqSessionBase):
    def _to_reseed(self, tracker_tail):
        sess = self.build([res(conf=0.05, ok=False)] * 4 + tracker_tail, [rr_match()])
        self.to_lost(sess)
        sess.step(self.frame)                     # eval1
        sess.step(self.frame)                     # eval2 -> reseed
        assert sess.tracker.init_calls == 2
        return sess

    def test_successful_init_enters_probation(self):
        sess = self._to_reseed([res(conf=0.9)])
        self.assertIs(sess.state, TrackState.LOW_CONFIDENCE)
        self.assertTrue(self.events(sess, "PROBATION_START"))

    def test_probation_success_returns_to_tracking(self):
        sess = self._to_reseed([res(conf=0.9)])   # healthy forever
        outs = [sess.step(self.frame) for _ in range(self.cfg.REACQ_PROBATION_N)]
        self.assertIs(outs[-1].state, TrackState.TRACKING)
        self.assertTrue(self.events(sess, "PROBATION_OK"))
        self.assertIs(sess._reacq_phase, __import__("ground_target_tracking.session",
                      fromlist=["_Phase"])._Phase.NONE)

    def test_probation_failure_returns_to_recovery(self):
        sess = self._to_reseed([res(conf=0.05, ok=False)])   # bad immediately
        out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOST)
        self.assertTrue(self.events(sess, "PROBATION_FAIL"))
        self.assertEqual(sess._reacq_episode, 2, "failure starts a fresh episode")

    def test_reseed_does_not_overwrite_ref_patch(self):
        ref_bytes = None
        sess = self.build([res(conf=0.05, ok=False)] * 4 + [res(conf=0.9)], [rr_match()])
        ref_bytes = sess._ref_patch.tobytes()
        self.to_lost(sess)
        sess.step(self.frame)
        sess.step(self.frame)                     # reseed here
        self.assertEqual(sess.tracker.init_calls, 2)
        self.assertEqual(sess._ref_patch.tobytes(), ref_bytes,
                         "reseed must NOT re-snapshot _ref_patch (no circular validation)")
        sess.step(self.frame)                     # a probation frame
        self.assertEqual(sess._ref_patch.tobytes(), ref_bytes,
                         "probation must NOT rebuild the M8 reference")

    def test_probation_deadline_bounds_low_confidence_band_stall(self):
        # Persistent LOW_CONFIDENCE-band measurements (not bad, not healthy) must
        # NOT stall probation forever: the total-frame deadline fails it.
        # evidence in [0.15,0.40): conf 0.30 on the init frame (sim ~1.0) ->
        # evidence 0.30 (not bad), confidence 0.30 (< 0.40 -> not healthy).
        sess = self._to_reseed([res(conf=0.30, ok=True, source="measure")])
        for _ in range(self.cfg.REACQ_PROBATION_MAX_FRAMES + 1):
            out = sess.step(self.frame)
        self.assertIs(out.state, TrackState.LOST, "deadline must bound the stall")
        fails = self.events(sess, "PROBATION_FAIL")
        self.assertTrue(fails and fails[-1]["reason"] == "probation:deadline")

    def test_probation_success_precedes_deadline_same_frame(self):
        # PROBATION_N == MAX_FRAMES, all healthy: on the final frame the healthy
        # count reaches N AND elapsed reaches MAX simultaneously -> SUCCESS wins.
        cfg = make_reacq_cfg(REACQ_PROBATION_N=3, REACQ_PROBATION_MAX_FRAMES=3)
        sess = TrackingSession(StubTracker([res(conf=0.05, ok=False)] * 4 + [res(conf=0.9)]),
                               cfg, enable_reacq=True, reacquirer=StubReacq([rr_match()]))
        sess.init(self.frame, self.p0)
        self.to_lost(sess, cfg.LOST_AFTER_N_BAD)
        sess.step(self.frame); sess.step(self.frame)   # reseed
        self.assertEqual(sess.tracker.init_calls, 2)
        outs = [sess.step(self.frame) for _ in range(3)]
        self.assertIs(outs[-1].state, TrackState.TRACKING,
                      "success takes precedence over the deadline on a tie")
        self.assertTrue(self.events(sess, "PROBATION_OK"))
        self.assertEqual(self.events(sess, "PROBATION_FAIL"), [])


class TestEpisodeReset(ReacqSessionBase):
    def test_recovery_episode_reset_prevents_stale_hypothesis(self):
        sess = self._to_reseed_fail()
        # after a probation failure we are SEARCHING in episode 2 with a fresh
        # HypothesisTracker; the OLD accepted hypothesis cannot re-init.
        self.assertEqual(sess._reacq_episode, 2)
        self.assertEqual(sess.tracker.init_calls, 2, "old hypothesis did not re-init")

    def _to_reseed_fail(self):
        sess = self.build([res(conf=0.05, ok=False)] * 4 + [res(conf=0.9)]
                          + [res(conf=0.05, ok=False)], [rr_match()])
        self.to_lost(sess)
        sess.step(self.frame); sess.step(self.frame)   # reseed (init_calls 2)
        # next frame: probation sees a bad frame -> fail -> episode 2
        sess.step(self.frame)
        return sess

    def test_new_target_reset_clears_state(self):
        sess = self.build([res(conf=0.05, ok=False)], [rr_match()])
        self.to_lost(sess)
        sess.step(self.frame)                     # in recovery, episode 1
        self.assertEqual(sess._reacq_episode, 1)
        sess.init(self.frame, utils.Point2D(120.0, 120.0))   # NEW target
        self.assertEqual(sess._reacq_episode, 0)
        self.assertIsNone(sess._hypo)
        self.assertIs(sess._reacq_phase, __import__("ground_target_tracking.session",
                      fromlist=["_Phase"])._Phase.NONE)


class TestSearchCadence(ReacqSessionBase):
    def test_skipped_cadence_frames_leave_hypotracker_untouched(self):
        # SEARCH_EVERY=2 with MAX_NEUTRAL=0: if skipped frames were fed NEUTRAL,
        # the zero neutral-gap would clear the streak after every MATCH and
        # acceptance would NEVER happen. It DOES happen -> skips are untouched,
        # and persistence is counted in executed evaluations (=2), not frames.
        cfg = make_reacq_cfg(REACQ_SEARCH_EVERY=2, REACQ_PERSIST_MAX_NEUTRAL=0,
                             REACQ_PERSIST_N=2)
        sess = TrackingSession(StubTracker([res(conf=0.05, ok=False)] * 4 + [res(conf=0.9)]),
                               cfg, enable_reacq=True, reacquirer=StubReacq([rr_match()]))
        sess.init(self.frame, self.p0)
        self.to_lost(sess, cfg.LOST_AFTER_N_BAD)
        # search ticks: 1 skip, 2 eval1, 3 skip, 4 eval2 -> accept on the 4th SEARCHING frame
        for _ in range(4):
            sess.step(self.frame)
        self.assertEqual(sess.tracker.init_calls, 2, "accepted despite skips (untouched)")
        self.assertEqual(sess._injected_reacq.calls, 2,
                         "best_candidate ran only on executed evaluations, not every frame")


if __name__ == "__main__":
    unittest.main()
