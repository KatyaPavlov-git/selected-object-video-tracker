"""session.py — tracking confidence + lost-state control layer (Milestone 8).

This module is the CONTROL layer that sits above the per-frame data path
(preprocess -> track -> filter). Trackers measure; the TrackingSession decides
how much to trust the measurement and exposes an explicit state:

    TRACKING        confident measurement — the box means what it says
    LOW_CONFIDENCE  still measuring, but degraded — treat with suspicion
    PREDICT         no reliable measurement; Kalman coasting within budget
    LOST            tracking is no longer trustworthy — the system says so
                    (terminal in M8; Milestone 9 adds reacquisition here)

Confidence is a product of simple, explainable signals:
    tracker confidence  (survivor count + forward-backward error; coast decay)
  x patch similarity    (NCC of the current ROI vs the reference patch)
  x edge penalty        (ROI clamped to / near the frame border)
with a jump veto (a one-frame displacement larger than MAX_JUMP_PX is not
committed — unless it persists for JUMP_VETO_MAX_FRAMES consecutive frames,
which means real fast motion rather than a glitch). State transitions use
consecutive-frame counters (hysteresis) so a single noisy frame cannot flip
the state, and LOST requires a sustained run of bad frames.

The reference patch is CONSERVATIVELY ADAPTIVE: a fixed init-frame template
decays against legitimate viewpoint/scale/lighting evolution (measured on the
official sample video: NCC 0.96 -> ~0.1 in ~100 frames while tracking was
still correct), so the reference is re-snapshotted — but ONLY through a strict
gate (positional trust: tracker_conf >= LOW_CONFIDENCE_BELOW + measurement ok
+ similarity >= REF_UPDATE_MIN_SIM + no edge penalty + no jump veto, throttled
to every REF_UPDATE_EVERY qualifying frame). The gate is keyed to the
POSITIONAL axis, not the combined TrackState: the combined state embeds
similarity, so keying on it let identity decay block the only mechanism that
restores identity (self-freeze). Positionally-degraded content still never
becomes the reference, which is what keeps the false-lock detection intact.

State lives ONLY here — trackers stay state-free and report per-frame quality;
the session owns history (streaks, last credible point, coast budget outcome).
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config, utils
    from . import reacquisition as reacq
    from .trackers import Tracker
except ImportError:  # fallback: run from inside the package folder
    import config
    import utils
    import reacquisition as reacq
    from trackers import Tracker


class TrackState(enum.Enum):
    """Explicit tracking state exposed to the UI / logs (M8). PUBLIC contract —
    unchanged by M9-c (reacquisition rides under LOST/LOW_CONFIDENCE)."""

    TRACKING = "TRACKING"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PREDICT = "PREDICT"
    LOST = "LOST"
    FEED_FROZEN = "FEED_FROZEN"   # dead/repeated input feed — held, distinct (Commit 4a)


class _Phase(enum.Enum):
    """INTERNAL M9-c reacquisition phase (not a public TrackState). Maps to a
    public state for reporting: NONE -> whatever M8 says; SEARCHING -> LOST;
    PROBATION -> LOW_CONFIDENCE. See plan §8 (REACQUIRED is an internal flag)."""

    NONE = "none"            # normal M8 operation; no reacquisition running
    SEARCHING = "searching"  # M8-LOST + reacquisition active (public LOST)
    PROBATION = "probation"  # re-seeded, proving before TRACKING (public LOW_CONFIDENCE)


@dataclass
class SessionResult:
    """One frame's session outcome (state + committed point + diagnostics).

    `point` is the session-committed estimate: the tracker's point while it is
    trusted, or the frozen last-credible point once LOST. `confidence` is the
    combined session confidence in [0..1]; `signals` carries the raw components
    (tracker_conf / similarity / edge / jump_vetoed / bad_streak) for the HUD,
    logging and tests. `result` is the inner tracker's TrackResult.
    """

    state: TrackState
    point: "utils.Point2D"
    confidence: float
    result: "utils.TrackResult"
    signals: dict = field(default_factory=dict)


class TrackingSession:
    """Wrap a Tracker with confidence scoring and a lost-state machine (M8).

    The session does NOT alter how the tracker measures (M1-M7 unchanged); it
    decides, per frame, whether the measurement deserves trust, commits or
    rejects it, and reports an honest TrackState. Once LOST it stops committing
    tracker output and freezes the last credible point — recovery from LOST is
    Milestone 9 (reacquisition) and is intentionally absent here.
    """

    def __init__(self, tracker: Tracker, cfg=config, enable_reacq=None,
                 reacquirer=None) -> None:
        self.tracker = tracker
        self.cfg = cfg
        # M9-c reacquisition gate. Production default stays False
        # (cfg.REACQ_DECISION_ENABLED); `enable_reacq` is the per-run override
        # (main.py --reacq / tests). When disabled, EVERY M9-c branch below is
        # skipped and the session behaves byte-for-byte as the M8 baseline.
        self._reacq_enabled = bool(getattr(cfg, "REACQ_DECISION_ENABLED", False)
                                   if enable_reacq is None else enable_reacq)
        self._injected_reacq = reacquirer     # test-injected Reacquirer (else built in init)
        self._reacq: Optional["reacq.Reacquirer"] = None      # immutable identity reference
        self._hypo: Optional["reacq.HypothesisTracker"] = None  # session-owned, per episode
        # Phase 2: bounded SEARCHING scheduler for the descriptor-free template
        # path (session-owned instance, fresh per recovery episode — mirrors
        # the HypothesisTracker ownership). None on the feature path.
        self._scan: Optional["reacq.TemplateScanScheduler"] = None
        self._reacq_phase = _Phase.NONE
        self._reacq_episode = 0               # generation id; guards stale acceptance
        self._probation_healthy = 0
        self._probation_elapsed = 0
        self._search_tick = 0                 # cadence counter (evaluations, not frames)
        # Interleaved routing (feature-capable + context-capable reference):
        # exactly ONE route per executed tick; the scan runs only after the
        # feature route's last result was NEUTRAL (strict alternation).
        self._last_feat_neutral = False       # last feature evaluation was NEUTRAL
        self._route_flip = False              # True -> the next executed tick scans
        # SIFT last-confident route: the session keeps ONE cheap snapshot
        # (frame copy + point) per recovery episode, captured during the
        # segment's FIRST sustained-trustworthy window and FROZEN at that
        # window's first sustained break (v8 nearby-selection instability
        # fix, 2026-07-07); the SIFT reference build is DEFERRED to recovery
        # entry so the healthy tracking path never pays SIFT cost.
        self._hypo_sift: Optional["reacq.HypothesisTracker"] = None
        self._exec_tick = 0                   # executed recovery evaluations (cadence base)
        self._lc_frame: Optional[np.ndarray] = None
        self._lc_point: Optional["utils.Point2D"] = None
        self._lc_qualify_streak = 0
        self._lc_built = False                # SIFT ref built for the current snapshot
        self._lc_builder = None               # G2-a striped build in progress (LOST ticks)
        self._accept_fit_log = []             # accept-time fit diagnostics
                                              # (option i; bounded)
        self._lc_frozen = False               # snapshot frozen for this episode
        self._lc_break_len = 0                # consecutive non-qualifying frames
        self._lc_seg_fresh = False            # a snapshot was captured THIS segment
        self._probation_positional = False    # sift-lc reseed: positional-only probation
        self._reacq_events: list = []         # observable recovery events (diagnostic)
        self.state = TrackState.TRACKING
        self.point: Optional["utils.Point2D"] = None
        # Decision baseline (Stage 1): the last MEASURED-accepted point, distinct
        # from the display point. `self.point` may glide on a Kalman coast, but a
        # gap/prediction must not move the baseline the jump veto and recovery
        # measure against — only a real accepted measurement writes this.
        self._accepted_point: Optional["utils.Point2D"] = None
        self._overlay: Optional[np.ndarray] = None    # SEED mask (255=overlay): seeding/ignore/occlusion
        self._overlay_dil: Optional[np.ndarray] = None  # dilated: known-occlusion zone
        # RC1: geometry-faithful STROKE mask for M8 LOCAL APPEARANCE only (the
        # seed disc erased the whole 51px patch at a crosshair selection). Used
        # solely for _ref_valid + _patch_similarity's current-patch validity.
        self._overlay_identity: Optional[np.ndarray] = None
        self._ref_patch: Optional[np.ndarray] = None  # gray reference patch (adaptive)
        self._ref_valid: Optional[np.ndarray] = None  # 255 where _ref_patch is overlay-free
        self._ref_std = 0.0
        self._ref_qualify_streak = 0  # consecutive qualifying frames toward the update throttle
        self._ref_staleness = 0       # live-path frames since the last reference snapshot
        self._bad_streak = 0   # consecutive frames judged bad (toward LOST)
        self._good_streak = 0  # consecutive strong frames (toward recovery)
        self._veto_streak = 0  # consecutive jump-vetoed frames (escape hatch for real motion)
        self._unknown_streak = 0  # consecutive identity-UNKNOWN frames (similarity None
                                  # while the reference is informative) — Run C policy
        # init-on-overlay telemetry (set by init(); see policy comment there)
        self.init_overlay_cov = 0.0
        self.init_on_overlay = False
        self.init_seedable = 1.0
        self.init_dead_zone = False
        # ---- Feed-health: FEED_FROZEN detection + T-b corruption discount ----
        # (Commit 4a, §E). A NO-OP unless cfg.FEED_FROZEN_ENABLED — absent on the
        # from-scratch test SimpleNamespaces, so those sessions are byte-for-byte
        # the pre-4a behavior (mirrors the REACQ_DECISION_ENABLED gating).
        self._feed_frozen_enabled = bool(getattr(cfg, "FEED_FROZEN_ENABLED", False))
        self._feed_corrupt_enabled = (self._feed_frozen_enabled
                                      and bool(getattr(cfg, "FEED_CORRUPT_ENABLED", True)))
        self._ff_w = int(getattr(cfg, "FEED_FROZEN_SMALL_W", 480))
        self._ff_h = int(getattr(cfg, "FEED_FROZEN_SMALL_H", 270))
        self._ff_block = int(getattr(cfg, "FEED_FROZEN_BLOCK_PX", 16))
        self._ff_t_static = float(getattr(cfg, "FEED_FROZEN_T_STATIC", 1.5))
        self._ff_enter_n = int(getattr(cfg, "FEED_FROZEN_ENTER_N", 3))
        self._ff_exit_m = int(getattr(cfg, "FEED_FROZEN_EXIT_M", 2))
        self._corrupt_ratio = float(getattr(cfg, "FEED_CORRUPT_FLOW_RATIO", 3.0))
        self._corrupt_floor = float(getattr(cfg, "FEED_CORRUPT_FLOW_FLOOR", 1.0))
        self._corrupt_window = int(getattr(cfg, "FEED_CORRUPT_FLOW_WINDOW", 60))
        self._corrupt_warmup = int(getattr(cfg, "FEED_CORRUPT_FLOW_WARMUP", 15))
        self._feed_frozen = False              # currently in the FEED_FROZEN condition
        self._prev_small = None                # previous 480x270 gray (block-max / flow-std)
        self._frozen_block_mask = None         # 17x30 bool: OSD blocks excluded from block-max
        self._static_run = 0                   # consecutive quiet frames (toward ENTER)
        self._motion_run = 0                   # consecutive motion frames (toward EXIT)
        self._flow_norm_window: list = []      # rolling live flow-std (non-discounted)
        self._corrupt_discounts = 0            # reported count of T-b discounts (this run)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def init(self, frame_bgr: np.ndarray, point: "utils.Point2D") -> None:
        """Initialize the tracker and capture the grayscale reference patch."""
        # Fixed HUD-overlay ignore mask (opt-in): installed BEFORE tracker init
        # so even the initial corner seeding avoids the overlay lines. The
        # session keeps it too — the appearance signals (NCC similarity,
        # adaptive reference) must exclude overlay pixels as well.
        self._overlay = utils.build_overlay_mask(frame_bgr, self.cfg)
        # RC1: identity (appearance) mask — strokes only, scene between them kept.
        self._overlay_identity = utils.build_overlay_mask(frame_bgr, self.cfg,
                                                          kind="identity")
        self._overlay_dil = None
        if self._overlay is not None:
            if hasattr(self.tracker, "set_ignore_mask"):
                self.tracker.set_ignore_mask(self._overlay)
            # Dilated zone where measurements are structurally compromised
            # (matches the tracker's seed exclusion) — used by the
            # known-occlusion LOST suspension in step().
            d = int(self.cfg.OVERLAY_SEED_DILATE_PX)
            kernel = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
            self._overlay_dil = cv2.dilate(self._overlay, kernel) if d > 0 else self._overlay
        # Init-on-overlay policy (explicit, policy B): a click on a HUD line is
        # ACCEPTED — the underlying ground point is usually valid and most of
        # the ROI is clean scene. Seeding/estimation use only non-overlay
        # support (dilated exclusion); these fields tell the caller what the
        # selection looks like so it can warn the user. When the seedable
        # fraction is below OVERLAY_INIT_MIN_SEEDABLE (e.g. the crosshair
        # disc), the selection is a declared DEAD ZONE: nothing is fabricated,
        # the tracker simply has no measurements and the state machine reports
        # PREDICT/LOW_CONFIDENCE (never TRACKING) until the target emerges.
        self.init_overlay_cov = self._overlay_coverage(point, frame_bgr.shape)
        self.init_on_overlay = self._on_dilated_overlay(point)
        self.init_seedable = 1.0
        self.init_dead_zone = False
        if self._overlay_dil is not None:
            h, w = frame_bgr.shape[:2]
            x0, y0, x1, y1 = utils.clamp_roi(point.x, point.y,
                                             self.cfg.PATCH_SIZE, w, h)
            roi = self._overlay_dil[y0:y1, x0:x1]
            self.init_seedable = float((roi == 0).mean()) if roi.size else 0.0
            self.init_dead_zone = (self.init_seedable
                                   < self.cfg.OVERLAY_INIT_MIN_SEEDABLE)
        self.tracker.init(frame_bgr, point)
        self.point = point
        self._accepted_point = point
        self.state = TrackState.TRACKING
        self._bad_streak = 0
        self._good_streak = 0
        self._veto_streak = 0
        self._unknown_streak = 0
        self._ref_qualify_streak = 0
        self._ref_staleness = 0
        self._lc_frame = None                 # new target: no stale snapshot
        self._lc_builder = None               # discard any in-progress striped build
        self._lc_point = None
        self._lc_qualify_streak = 0
        self._lc_built = False
        self._lc_frozen = False
        self._lc_break_len = 0
        self._lc_seg_fresh = False
        self._ref_patch = self._gray_patch(frame_bgr, point)
        self._ref_valid = self._valid_patch_identity(point, frame_bgr.shape)
        # A near-uniform reference has no texture for NCC to verify against;
        # flag it so similarity degrades to a neutral signal instead of noise.
        self._ref_std = self._masked_std(self._ref_patch, self._ref_valid)
        # M9-c: build the IMMUTABLE identity reference ONCE, at selection time,
        # and clear any recovery state (new target). Only when reacquisition is
        # enabled — when disabled nothing here runs and there are no side effects.
        self._reset_recovery()
        if self._reacq_enabled:
            self._reacq = self._injected_reacq or reacq.Reacquirer(self.cfg)
            self._reacq.build_reference(frame_bgr, point, self._overlay)
        else:
            self._reacq = None
        # Feed-health (Commit 4a): precompute the OSD block-mask (from the same
        # overlay the pipeline uses) and seed the previous-frame buffer with the
        # init frame so block-max is defined from the first step().
        self._feed_frozen = False
        self._static_run = 0
        self._motion_run = 0
        self._flow_norm_window = []
        self._corrupt_discounts = 0
        self._prev_small = None
        self._frozen_block_mask = None
        if self._feed_frozen_enabled:
            self._frozen_block_mask = self._build_frozen_block_mask()
            self._prev_small = self._feed_small(frame_bgr)

    def step(self, frame_bgr: np.ndarray) -> SessionResult:
        """Advance one frame: measure, score, decide state, commit or freeze."""
        # --- Feed-health gate (Commit 4a, §E): operates on the RAW input feed
        # BEFORE the tracker, orthogonal to the tracking state. A frozen or
        # corruption-discounted frame short-circuits here (the tracker is NOT
        # stepped, so no Kalman correction / streak movement / reference update).
        if self._feed_frozen_enabled:
            fr = self._feed_health_step(frame_bgr)
            if fr is not None:
                return fr

        result = self.tracker.update(frame_bgr)

        # M9-c PROBATION (public LOW_CONFIDENCE): a re-seeded tracker proving
        # itself. Its own phase, so the M8 bad-streak/LOST machinery is never
        # touched. Only runs when reacquisition is enabled.
        if self._reacq_enabled and self._reacq_phase is _Phase.PROBATION:
            return self._probation_step(frame_bgr, result)

        if self.state is TrackState.LOST:
            # M9-c: while enabled, LOST is no longer terminal — reacquisition runs
            # (SEARCHING). Recovery ENTRY (new episode + fresh HypothesisTracker)
            # happens on the first LOST frame; M8 stays the SOLE declarer of LOST.
            if self._reacq_enabled and self._reacq is not None:
                if self._reacq_phase is _Phase.NONE:
                    self._enter_recovery()
                return self._recovery_step(frame_bgr, result)
            # Terminal in M8 (disabled path — UNCHANGED): keep reporting honestly,
            # never re-trust the tracker.
            return SessionResult(state=TrackState.LOST, point=self.point,
                                 confidence=0.0, result=result,
                                 signals={"tracker_conf": result.confidence,
                                          "similarity": None, "edge": False,
                                          "jump_vetoed": False, "overlay": None,
                                          "bad_streak": self._bad_streak})

        h, w = frame_bgr.shape[:2]
        candidate = result.point
        self._ref_staleness += 1   # live-path frames since the last snapshot

        # --- session-level signals ------------------------------------- #
        # Jump veto with an escape hatch: a single-frame teleport is rejected,
        # but displacement that PERSISTS for several consecutive frames is real
        # motion (fast pan / camera shake) — trust the tracker again, otherwise
        # the gap to the frozen point only grows and the veto never releases.
        jump_vetoed = self._jump_exceeded(candidate)
        if jump_vetoed:
            self._veto_streak += 1
            if self._veto_streak > self.cfg.JUMP_VETO_MAX_FRAMES:
                jump_vetoed = False
                self._veto_streak = 0
        else:
            self._veto_streak = 0
        similarity = self._patch_similarity(frame_bgr, candidate)
        edge_near = self._near_edge(candidate, w, h)
        overlay_cov = self._overlay_coverage(candidate, frame_bgr.shape)

        sim_score = 1.0 if similarity is None else min(1.0, max(0.0, similarity))
        edge_penalty = float(self.cfg.EDGE_PENALTY) if edge_near else 1.0
        tracker_conf = min(1.0, max(0.0, float(result.confidence)))
        confidence = 0.0 if jump_vetoed else tracker_conf * sim_score * edge_penalty

        # --- lost decision (bad-frame streak) ---------------------------- #
        # A frame is bad when the loss EVIDENCE — measurement quality x aligned
        # appearance — is below the lost threshold. The edge penalty is
        # deliberately excluded here: border proximity is a RISK (shown as
        # LOW_CONFIDENCE via `confidence`), not evidence the object is gone —
        # an object grazing the frame edge and coming back must survive as
        # LOW_CONFIDENCE. A true exit still trips this: the survivors collapse
        # (tracker confidence -> 0), re-seeded junk kills similarity, and
        # exhausted Kalman coasting counts as bad frames as well (but never
        # declares LOST instantly, so a brief dropout can recover).
        # Identity-unknown policy (Run C): similarity None while the REFERENCE
        # is informative means identity could not be evaluated THIS frame
        # (e.g. the clamped border-sliver patch) — unknown, not confirmed.
        # Unknown never testifies that tracking is healthy: it cannot reset an
        # existing bad streak (hold below), and once it persists beyond
        # LOST_UNKNOWN_SIM_AFTER_N consecutive frames it becomes loss evidence
        # itself (measured on the official sample: border-sliver None frames
        # with regional confidence up to 0.92 reset a 10-frame bad streak and
        # deferred LOST indefinitely). A flat/absent reference (low-texture
        # target — identity never measurable by design) keeps the full legacy
        # neutrality: no counter, no escalation, resets allowed.
        sim_unknown = (similarity is None and self._ref_patch is not None
                       and self._ref_std >= 1e-6)
        self._unknown_streak = self._unknown_streak + 1 if sim_unknown else 0
        coast_exhausted = (result.source == "predict") and (not result.ok)
        evidence = 0.0 if jump_vetoed else tracker_conf * sim_score
        bad = coast_exhausted or evidence < self.cfg.LOST_CONFIDENCE_BELOW
        if (sim_unknown and self._unknown_streak
                > int(getattr(self.cfg, "LOST_UNKNOWN_SIM_AFTER_N", 20))):
            bad = True                   # persistent unknown = loss evidence
        # Off-frame hardening (Stage 2A): a point outside the frame is
        # unobservable — no patch, no similarity, no scene support — so the
        # frame is loss evidence no matter how confident the measurement
        # claims to be, and the known-occlusion suspension below must not
        # apply (an off-frame point is gone, not HUD-occluded; clamping it
        # onto a border overlay pixel would freeze the streak forever).
        off_frame = False
        if getattr(self.cfg, "LOST_OFFFRAME_HARDENING", False):
            m = float(getattr(self.cfg, "LOST_OFFFRAME_MARGIN_PX", 0))
            off_frame = (candidate is None
                         or not (-m <= candidate.x < w + m
                                 and -m <= candidate.y < h + m))
            bad = bad or off_frame
        # Known-occlusion suspension: while the committed point lies inside the
        # dilated overlay, measurements are structurally compromised by the
        # overlay (windows clipped, similarity partial), so bad frames FREEZE
        # the LOST streak instead of advancing it — a crossing target must
        # resurface as LOW_CONFIDENCE, not LOST. The streak is frozen, never
        # reset: history against a real loss ending on the overlay is kept.
        if bad and (off_frame or not self._on_dilated_overlay(candidate)):
            self._bad_streak += 1
        elif not bad and not sim_unknown:
            self._bad_streak = 0
        # (not-bad + unknown -> the streak HOLDS: an unevaluable frame can
        #  neither advance nor erase the accumulated loss evidence.)

        if self._bad_streak >= self.cfg.LOST_AFTER_N_BAD:
            self.state = TrackState.LOST
            self._good_streak = 0
            return SessionResult(state=TrackState.LOST, point=self.point,
                                 confidence=confidence, result=result,
                                 signals=self._signals(tracker_conf, similarity,
                                                       edge_near, jump_vetoed,
                                                       overlay_cov))

        # --- state among the live states -------------------------------- #
        coasting = (result.source == "predict") and result.ok
        strong = confidence >= self.cfg.LOW_CONFIDENCE_BELOW
        if coasting:
            new_state = TrackState.PREDICT
            self._good_streak = 0
        elif strong:
            if self.state is TrackState.TRACKING:
                new_state = TrackState.TRACKING
            else:
                # Recovery requires a RE-VALIDATED reliable measurement THIS frame
                # (Stage 1) — not merely N strong frames, which a manufactured
                # lock also satisfies. "Reliable now" = a real measure (not a
                # coast/predict), accepted (not vetoed), that still matches the
                # reference. similarity None (flat/low-texture target) is neutral,
                # mirroring the confidence path; a present-but-low similarity is
                # the false-lock signature and blocks recovery. The RECOVER_N
                # hysteresis is kept, but only re-validated frames advance it.
                measured_now = (result.source == "measure" and result.ok
                                and not jump_vetoed)
                # O1 (user-adopted 2026-07-08): similarity None counts as
                # neutral-confirming ONLY when the reference itself is
                # uninformative (sim_unknown False — the legacy flat/low-
                # texture carve-out). When the reference IS informative, an
                # unevaluable frame is UNKNOWN: it neither advances nor
                # resets the re-validation streak — measured on the official
                # sample, unknown-as-confirming let 9 frames of coherent
                # ambient desert texture certify TRACKING while the target
                # was absent (reads 113-121, all picks; simulation proof:
                # O1 removes that interval with every other milestone
                # byte-identical).
                sim_ok = ((similarity is not None
                           and similarity >= self.cfg.RECOVER_MIN_SIM)
                          or (similarity is None and not sim_unknown))
                revalidated = (measured_now and sim_ok
                               and confidence >= (self.cfg.LOW_CONFIDENCE_BELOW +
                                                  self.cfg.RECOVER_MARGIN))
                if revalidated:
                    self._good_streak += 1
                elif not (similarity is None and sim_unknown):
                    self._good_streak = 0
                # (unknown holds the streak — mirroring the bad-streak hold)
                new_state = (TrackState.TRACKING
                             if self._good_streak >= self.cfg.RECOVER_N
                             else TrackState.LOW_CONFIDENCE)
        else:
            new_state = TrackState.LOW_CONFIDENCE
            self._good_streak = 0
        self.state = new_state

        # --- commit ------------------------------------------------------ #
        # Commit the tracker's point when this frame produced a usable estimate:
        # a reliable measurement, or a within-budget Kalman prediction. A vetoed
        # jump is never committed; the point then stays at the last estimate.
        if result.ok and not jump_vetoed:
            self.point = candidate                     # display / glide (may coast)
            if result.source == "measure":             # baseline: real measures only
                self._accepted_point = candidate

        # --- conservative reference update ------------------------------- #
        # The reference may follow the target's gradual appearance change, but
        # ONLY through the strict gate below — junk content on degraded frames
        # can never become the reference, so false-lock detection survives.
        ref_gate = self._maybe_update_reference(
            frame_bgr, similarity, edge_near, jump_vetoed,
            measured_ok=(result.ok and result.source != "predict"),
            tracker_conf=tracker_conf)

        # --- SIFT last-confident snapshot (v8 design review) -------------- #
        # While the frame is trustworthy on the POSITIONAL axis, keep ONE
        # cheap frame copy for the deferred SIFT reference. Deliberately NO
        # similarity condition (a HUD-covered selection has similarity None
        # for the whole run) and NO ROI-overlay conjunct: that guard protects
        # the M8 patch reference, which stores ROI PIXELS as evidence — the
        # LC reference stores only per-keypoint evidence, and its HUD
        # exclusion is enforced at build time (build_lc_reference discards
        # every keypoint on the mask and returns None below the clean-keypoint
        # floor). Measured on v8: the trustworthy point sits inside a masked
        # band while its 150px ring context is clean and sufficient — an ROI
        # pre-condition blocked every qualifying frame for zero added safety.
        if self._reacq_enabled:
            lc_ok = (self.state is TrackState.TRACKING
                     and result.ok and result.source == "measure"
                     and not edge_near and not jump_vetoed)
            if lc_ok:
                self._lc_qualify_streak += 1
                self._lc_break_len = 0
                # WARM-UP, then refresh, then FREEZE: the first snapshot
                # requires REF_UPDATE_EVERY consecutive qualifying frames (a
                # single flukey TRACKING frame can never become the
                # reference); every further qualifying frame refreshes the
                # copy UNTIL the segment's first SUSTAINED quality break
                # (REF_UPDATE_EVERY consecutive non-qualifying frames), after
                # which the snapshot is frozen for the episode. Rationale
                # (2026-07-07, measured on v8): once quality has broken, the
                # track is suspect — later "re-qualified" views are the
                # degrading pre-loss tail whose match margins flicker at the
                # accept floor, and refreshing into them made nearby
                # selections diverge (reacquire vs never). The first
                # sustained window's view carries fat margins; freezing there
                # is deterministic across nearby picks. A fresh learning
                # cycle starts only after a probation-validated
                # reacquisition (_succeed_probation).
                if (not self._lc_frozen
                        and self._lc_qualify_streak >= self.cfg.REF_UPDATE_EVERY):
                    self._lc_frame = frame_bgr.copy()
                    self._lc_point = self.point
                    self._lc_built = False    # a fresh snapshot invalidates the built ref
                    self._lc_seg_fresh = True
            else:
                self._lc_qualify_streak = 0
                if self._lc_seg_fresh and not self._lc_frozen:
                    self._lc_break_len += 1
                    if self._lc_break_len >= int(self.cfg.REF_UPDATE_EVERY):
                        self._lc_frozen = True

        return SessionResult(state=self.state, point=self.point,
                             confidence=confidence, result=result,
                             signals=self._signals(tracker_conf, similarity,
                                                   edge_near, jump_vetoed,
                                                   overlay_cov,
                                                   ref_gate=ref_gate))

    # ------------------------------------------------------------------ #
    # M9-c reacquisition integration (runs ONLY while enabled AND M8-LOST)
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Public reset hook (session reset / video restart / end-of-stream):
        clear ALL recovery state so no hypothesis survives across episodes.
        Does not rebuild the immutable reference or touch M8 tracking state."""
        self._reset_recovery()
        # feed-health runtime state (the OSD block-mask stays; init() rebuilds it)
        self._feed_frozen = False
        self._static_run = 0
        self._motion_run = 0
        self._prev_small = None
        self._flow_norm_window = []

    def _reset_recovery(self) -> None:
        self._hypo = None
        self._hypo_sift = None
        self._scan = None
        self._reacq_phase = _Phase.NONE
        self._reacq_episode = 0
        self._probation_healthy = 0
        self._probation_elapsed = 0
        self._search_tick = 0
        self._exec_tick = 0
        self._last_feat_neutral = False
        self._route_flip = False
        self._probation_positional = False

    def _new_scan(self) -> Optional["reacq.TemplateScanScheduler"]:
        """Phase 2: a FRESH bounded template-scan scheduler for this episode —
        for ANY context-capable reference. On a descriptor-free reference it
        is the sole route (unchanged); on a feature-capable reference it is
        interleaved by _recovery_step ONLY while the feature route returns
        NEUTRAL (the template route is no longer structurally blocked merely
        because descriptors exist). The no-capability refusal keeps the
        unchanged best_candidate route."""
        # getattr: injected test doubles may not expose a reference at all —
        # they (like the feature path) keep the unchanged best_candidate route.
        ref = getattr(self._reacq, "reference", None)
        if ref is not None and ref.has_context:
            return reacq.TemplateScanScheduler(self._reacq, self.cfg)
        return None

    def _event(self, name, **detail) -> None:
        """Record an observable recovery event (diagnostic + the smoke output)."""
        self._reacq_events.append({"event": name, "episode": self._reacq_episode,
                                   **detail})
        if len(self._reacq_events) > 256:      # bounded
            del self._reacq_events[:-256]

    def _enter_recovery(self) -> None:
        """First LOST frame of an episode: bump the episode id, build a FRESH
        session-owned HypothesisTracker, clear any stale latch/probation."""
        self._reacq_episode += 1
        self._hypo = reacq.HypothesisTracker(self.cfg)
        self._hypo_sift = reacq.HypothesisTracker(self.cfg)
        self._scan = self._new_scan()
        self._probation_healthy = 0
        self._probation_elapsed = 0
        self._search_tick = 0
        self._exec_tick = 0
        self._last_feat_neutral = False
        self._route_flip = False
        # Deferred SIFT last-confident reference: built ONCE per snapshot,
        # starting on the first LOST frame. G2-a (2026-07-08): the real
        # Reacquirer slices the ~100-240 ms whole-frame build into
        # REACQ_SIFT_STRIPES full-resolution stripes, one per executed
        # recovery tick, so no single LOST frame pays the whole cost. Stub /
        # legacy reacquirers without the builder surface keep the one-shot
        # build.
        if self._lc_frame is not None and not self._lc_built:
            if hasattr(self._reacq, "new_lc_builder"):
                self._lc_builder = self._reacq.new_lc_builder(
                    self._lc_frame, self._lc_point, self._overlay)
            elif hasattr(self._reacq, "build_lc_reference"):
                self._reacq.build_lc_reference(self._lc_frame, self._lc_point,
                                               self._overlay)
            self._lc_built = True
        self._reacq_phase = _Phase.SEARCHING
        self._event("RECOVERY_ENTER")

    def _recovery_step(self, frame_bgr, result) -> "SessionResult":
        """SEARCHING (public LOST): run reacquisition at the bounded cadence and
        feed EXECUTED evaluations into the session-owned HypothesisTracker. The
        committed point stays frozen. On a valid one-shot AcceptedHypothesis,
        re-seed once and enter PROBATION.

        Route selection (exactly ONE route per executed tick — the per-frame
        work bound the real-time budget is judged against):
          * no scheduler (feature-only / no-capability reference): the
            unchanged best_candidate route;
          * descriptor-free reference: the bounded scan scheduler (unchanged);
          * BOTH capabilities: feature-first; while the feature route's last
            result was NEUTRAL (nothing to evaluate — never an AMBIGUOUS
            refusal), executed ticks strictly ALTERNATE feature / scan, so the
            template route is reachable without ever stacking two routes into
            one frame."""
        self._search_tick += 1
        every = max(1, int(self.cfg.REACQ_SEARCH_EVERY))
        rr = None
        accepted = None
        scan_diag = None
        route = None
        if self._search_tick % every == 0:            # an EXECUTED evaluation
            self._exec_tick += 1
            lc_building = self._lc_builder is not None
            lc_ready = getattr(self._reacq, "lc_reference", None) is not None
            if lc_building or lc_ready:
                # SIFT last-confident route (G2-a, 2026-07-08): while an LC
                # reference exists (or is being built) it owns EVERY executed
                # tick, but each tick performs only ONE bounded stripe
                # (~12-18 ms @1080p — inside the 33.3 ms budget) instead of
                # the retired every-Nth whole-frame evaluation (110-350 ms).
                # Build stripes and non-final sweep stripes yield NO identity
                # observation, so the route's HypothesisTracker still counts
                # observations, exactly as the old cadence semantics did.
                route = "sift-lc"
                if lc_building:
                    if self._lc_builder.step():
                        self._lc_builder = None      # build finished (or floor-failed)
                    rr = None
                elif hasattr(self._reacq, "lc_sweep_step"):
                    rr = self._reacq.lc_sweep_step(frame_bgr)
                else:
                    # Stub / legacy surface: whole-frame evaluation per tick.
                    rr = self._reacq.best_candidate_lc(frame_bgr)
                if rr is not None:
                    accepted = self._hypo_sift.update(rr)
                    # Accept-time current-frame localization (option i,
                    # user-adopted 2026-07-08): the sweep's held-frame fits
                    # serve ONLY as persistence evidence — their geometry is
                    # one sweep-length stale, and reseeding from it measured
                    # 52-81 px checkpoint errors. When the UNCHANGED
                    # persistence gate first becomes ready to accept, ONE
                    # whole-frame fit on the CURRENT frame must independently
                    # pass the UNCHANGED identity gates and alone supplies
                    # the reseed localization (~105-120 ms, once per
                    # persistence one-shot; measured rolling-1s throughput
                    # stays >=37 processed frames). On a failed verification
                    # the hypothesis latch is CLEARED: no acceptance, sweeps
                    # resume, and another expensive fit is possible only
                    # after a completely fresh persistence sequence — never
                    # on a per-tick cadence.
                    if (accepted is not None
                            and hasattr(self._reacq, "lc_sweep_step")):
                        _t0 = time.perf_counter()
                        rr_now = self._reacq.best_candidate_lc(frame_bgr)
                        _ms = (time.perf_counter() - _t0) * 1e3
                        ok_now = rr_now.identity is reacq.Identity.MATCH
                        self._accept_fit_log.append(
                            {"exec_tick": self._exec_tick,
                             "ms": round(_ms, 2), "passed": ok_now,
                             "reason": rr_now.reason})
                        if len(self._accept_fit_log) > 64:   # bounded
                            del self._accept_fit_log[:-64]
                        if ok_now:
                            accepted = reacq.AcceptedHypothesis(
                                point=rr_now.point, scale=rr_now.scale,
                                cue=rr_now.cue,
                                confirm_ncc=(0.0 if rr_now.confirm_ncc is None
                                             else rr_now.confirm_ncc),
                                streak=accepted.streak)
                        else:
                            self._hypo_sift.clear()
                            accepted = None
            else:
                if self._scan is None:
                    # Feature path / no-capability: unchanged best_candidate route.
                    route = "feature"
                    rr = self._reacq.best_candidate(frame_bgr)
                elif not getattr(getattr(self._reacq, "reference", None),
                                 "has_descriptors", False):
                    # Descriptor-free template path (unchanged): bounded incremental
                    # scan + verify-first — the old one-frame full-frame sweep is
                    # unreachable from here.
                    route = "template-scan"
                    rr, scan_diag = self._scan.step(frame_bgr)
                elif self._last_feat_neutral and self._route_flip:
                    # Interleave: the feature route found nothing last tick — give
                    # this tick to the bounded template scan, return next tick.
                    route = "template-scan"
                    rr, scan_diag = self._scan.step(frame_bgr)
                    self._route_flip = False
                else:
                    route = "feature"
                    rr = self._reacq.best_candidate(frame_bgr)
                    self._last_feat_neutral = (rr.identity is reacq.Identity.NEUTRAL)
                    self._route_flip = self._last_feat_neutral
                accepted = self._hypo.update(rr)      # None until the one-shot fires
        # else SKIPPED frame -> HypothesisTracker UNTOUCHED (persistence counts
        # evaluations, not frames; a low cadence never spends the neutral gap).
        if accepted is not None and self._try_accept(frame_bgr, accepted):
            return SessionResult(state=self.state, point=self.point,   # now PROBATION
                                 confidence=0.0, result=result,
                                 signals=self._recovery_signals(rr, ran=True,
                                                                scan_diag=scan_diag,
                                                                route=route))
        return SessionResult(state=TrackState.LOST, point=self.point,
                             confidence=0.0, result=result,
                             signals=self._recovery_signals(rr, ran=(rr is not None),
                                                            scan_diag=scan_diag,
                                                            route=route))

    def _try_accept(self, frame_bgr, accepted) -> bool:
        """Validate a one-shot AcceptedHypothesis, then re-seed EXACTLY once.
        Off-frame / non-finite / degenerate-ROI acceptances do NOT initialize;
        the hypothesis is cleared so a fresh valid one can form."""
        h, w = frame_bgr.shape[:2]
        p = accepted.point
        if p is None or not (np.isfinite(p.x) and np.isfinite(p.y)):
            return self._skip_init("init-skip:non-finite")
        if not (0 <= p.x < w and 0 <= p.y < h):
            return self._skip_init("init-skip:off-frame")
        x0, y0, x1, y1 = utils.clamp_roi(p.x, p.y, self.cfg.PATCH_SIZE, w, h)
        if (x1 - x0) < self.cfg.MIN_PATCH_SIZE or (y1 - y0) < self.cfg.MIN_PATCH_SIZE:
            return self._skip_init("init-skip:degenerate-roi")
        self._reseed(frame_bgr, p, accepted)
        return True

    def _skip_init(self, reason) -> bool:
        self._event("INIT_SKIPPED", reason=reason)
        self._reacq_episode += 1                  # fresh episode: never retry a bad accept
        self._hypo = reacq.HypothesisTracker(self.cfg)
        self._hypo_sift = reacq.HypothesisTracker(self.cfg)
        self._scan = self._new_scan()             # scheduler state resets with the episode
        self._exec_tick = 0
        self._last_feat_neutral = False
        self._route_flip = False
        return False

    def _reseed(self, frame_bgr, point, accepted) -> None:
        """Re-initialize the tracker at the accepted point ONCE and enter
        probation. Updates ONLY tracker-local operational state + the committed
        point + M8 streaks/state. The immutable identity reference AND the M8
        appearance reference (_ref_patch/_ref_valid/_ref_std) are NOT touched —
        probation must validate against evidence independent of the candidate.

        sift-lc acceptances (approved 2026-07-07): probation judges POSITIONAL
        health only — the appearance axis cannot testify for an opposite-side
        return (measured 0.04-0.16 similarity on the TRUE v8 target), so the
        probation decision treats similarity as neutral for this route while
        still reporting it in signals. ORB/template probation is unchanged."""
        self._probation_positional = (accepted.cue == "sift-lc")
        self.tracker.init(frame_bgr, point)       # the single tracker.init()
        self.point = point
        self._accepted_point = point
        self._bad_streak = 0
        self._good_streak = 0
        self._veto_streak = 0
        self._unknown_streak = 0
        self._ref_qualify_streak = 0
        self._ref_staleness = 0
        self._lc_qualify_streak = 0
        self.state = TrackState.LOW_CONFIDENCE
        self._reacq_phase = _Phase.PROBATION
        self._probation_healthy = 0
        self._probation_elapsed = 0
        self._event("REACQ_INIT",
                    point=[round(float(point.x), 2), round(float(point.y), 2)],
                    cue=accepted.cue, confirm=round(float(accepted.confirm_ncc), 4),
                    scale=round(float(accepted.scale), 4), streak=int(accepted.streak))
        self._event("PROBATION_START")

    def _probation_step(self, frame_bgr, result) -> "SessionResult":
        """PROBATION (public LOW_CONFIDENCE): score the re-seeded tracker with
        the EXISTING M8 signals (no new appearance thresholds), validating
        against the INDEPENDENT pre-LOST reference. Needs REACQ_PROBATION_N
        healthy frames to succeed; a bad frame fails immediately; the
        REACQ_PROBATION_MAX_FRAMES deadline bounds it. SUCCESS is checked before
        the deadline (a genuine completion wins a same-frame tie)."""
        h, w = frame_bgr.shape[:2]
        candidate = result.point
        similarity = (self._patch_similarity(frame_bgr, candidate)
                      if candidate is not None else None)
        edge_near = self._near_edge(candidate, w, h) if candidate is not None else True
        sim_score = 1.0 if similarity is None else min(1.0, max(0.0, similarity))
        if self._probation_positional:
            # sift-lc reseed (approved): appearance cannot testify for an
            # opposite-side return — similarity stays in signals as a
            # diagnostic but is NEUTRAL for the probation decision,
            # mirroring the existing flat/masked-reference neutrality.
            sim_score = 1.0
        edge_penalty = float(self.cfg.EDGE_PENALTY) if edge_near else 1.0
        tracker_conf = min(1.0, max(0.0, float(result.confidence)))
        confidence = tracker_conf * sim_score * edge_penalty
        evidence = tracker_conf * sim_score
        off_frame = (candidate is None
                     or not (0 <= candidate.x < w and 0 <= candidate.y < h))
        coast_exhausted = (result.source == "predict") and (not result.ok)
        bad = (off_frame or coast_exhausted
               or evidence < float(self.cfg.LOST_CONFIDENCE_BELOW))
        healthy = ((not bad) and confidence >= float(self.cfg.LOW_CONFIDENCE_BELOW)
                   and result.source == "measure")
        if result.ok and not off_frame:           # commit a usable estimate
            self.point = candidate
            if result.source == "measure":
                self._accepted_point = candidate
        self._probation_elapsed += 1
        if bad:
            return self._fail_probation(result, similarity, edge_near, "probation:bad-frame")
        if healthy:
            self._probation_healthy += 1
        if self._probation_healthy >= int(self.cfg.REACQ_PROBATION_N):
            return self._succeed_probation(frame_bgr, result, similarity,
                                           edge_near, confidence)
        if self._probation_elapsed >= int(self.cfg.REACQ_PROBATION_MAX_FRAMES):
            return self._fail_probation(result, similarity, edge_near, "probation:deadline")
        self.state = TrackState.LOW_CONFIDENCE
        sig = self._signals(tracker_conf, similarity, edge_near, False, 0.0)
        sig.update(reacq_phase=self._reacq_phase.name,
                   probation_healthy=self._probation_healthy,
                   probation_elapsed=self._probation_elapsed)
        return SessionResult(state=self.state, point=self.point,
                             confidence=confidence, result=result, signals=sig)

    def _succeed_probation(self, frame_bgr, result, similarity, edge_near,
                           confidence) -> "SessionResult":
        self._event("PROBATION_OK", healthy=self._probation_healthy,
                    elapsed=self._probation_elapsed)
        if self._probation_positional:
            # sift-lc re-anchor (approved 2026-07-07): after floor-10 geometry,
            # 3-fold persistence and REACQ_PROBATION_N healthy positional
            # frames — the same evidence that flips the state to TRACKING —
            # adopt the probation-validated patch as the M8 working reference.
            # The stale pre-LOST view would otherwise testify against the true
            # target forever (measured 0.11-0.16 on the v8 opposite-side
            # return; the continuity floor legally blocks adaptation). The
            # immutable frame-0 identity reference stays untouched.
            new_ref = self._gray_patch(frame_bgr, self.point)
            new_valid = self._valid_patch_identity(self.point, frame_bgr.shape)
            new_std = self._masked_std(new_ref, new_valid)
            if new_ref is not None and new_std >= 1e-6:
                self._ref_patch = new_ref
                self._ref_valid = new_valid
                self._ref_std = new_std
                self._ref_qualify_streak = 0
                self._ref_staleness = 0
        self._probation_positional = False
        # Fresh snapshot-learning cycle (per-episode reference lifecycle):
        # the probation-validated track may now establish the NEXT episode's
        # stable snapshot; the current one remains the fallback until the new
        # segment's first sustained-trustworthy window replaces it.
        self._lc_frozen = False
        self._lc_break_len = 0
        self._lc_seg_fresh = False
        self._reacq_phase = _Phase.NONE
        self._hypo = None
        self.state = TrackState.TRACKING
        self._good_streak = 0
        self._bad_streak = 0
        sig = self._signals(min(1.0, max(0.0, float(result.confidence))),
                            similarity, edge_near, False, 0.0)
        sig.update(reacq_phase=self._reacq_phase.name)
        return SessionResult(state=self.state, point=self.point,
                             confidence=confidence, result=result, signals=sig)

    def _fail_probation(self, result, similarity, edge_near, reason) -> "SessionResult":
        """Probation failure -> re-enter SEARCHING (public LOST) with a FRESH
        episode/HypothesisTracker. The committed point stays frozen; the
        immutable reference is intact for the next search."""
        self._event("PROBATION_FAIL", reason=reason, healthy=self._probation_healthy,
                    elapsed=self._probation_elapsed)
        self._probation_positional = False
        self.state = TrackState.LOST
        self._reacq_episode += 1
        self._hypo = reacq.HypothesisTracker(self.cfg)
        self._hypo_sift = reacq.HypothesisTracker(self.cfg)
        self._scan = self._new_scan()             # scheduler state resets with the episode
        self._reacq_phase = _Phase.SEARCHING
        self._probation_healthy = 0
        self._probation_elapsed = 0
        self._search_tick = 0
        self._exec_tick = 0
        self._last_feat_neutral = False
        self._route_flip = False
        sig = self._signals(min(1.0, max(0.0, float(result.confidence))),
                            similarity, edge_near, False, 0.0)
        sig.update(reacq_phase=self._reacq_phase.name, probation_fail=reason)
        return SessionResult(state=TrackState.LOST, point=self.point,
                             confidence=0.0, result=result, signals=sig)

    def _recovery_signals(self, rr, ran, scan_diag=None, route=None) -> dict:
        """SEARCHING diagnostics — preserve the ReacqResult evidence
        (cue/point/scale/confirm/reason) for logging and tests. `scan_diag`
        carries the Phase-2 scheduler instrumentation (mode, units, cycle
        progress, pending, window, timing/overrun, verify-only fallback);
        `route` records which single route this executed tick ran."""
        sig = {"tracker_conf": 0.0, "similarity": None, "edge": False,
               "jump_vetoed": False, "overlay": None, "bad_streak": self._bad_streak,
               "reacq_phase": self._reacq_phase.name, "reacq_ran": bool(ran),
               "reacq_route": route}
        if rr is not None:
            hypo = self._hypo_sift if route == "sift-lc" else self._hypo
            sig.update(
                reacq_identity=rr.identity.name, reacq_reason=rr.reason, reacq_cue=rr.cue,
                reacq_point=(None if rr.point is None
                             else [round(float(rr.point.x), 2), round(float(rr.point.y), 2)]),
                reacq_scale=(None if rr.scale is None else round(float(rr.scale), 4)),
                reacq_confirm=(None if rr.confirm_ncc is None else round(float(rr.confirm_ncc), 4)),
                reacq_streak=(hypo.streak if hypo is not None else 0))
        if scan_diag is not None:
            sig.update(scan_diag)
        return sig

    # ------------------------------------------------------------------ #
    # feed-health: FEED_FROZEN detection + T-b corruption discount (Commit 4a)
    # ------------------------------------------------------------------ #
    def _feed_health_step(self, frame_bgr) -> Optional["SessionResult"]:
        """Run the block-max freeze detector (every state) and the T-b
        corruption discount (live path only). Returns a SessionResult to
        SHORT-CIRCUIT step() (frozen hold, confirmed exit, or corruption
        discount), or None to let normal processing proceed."""
        small = self._feed_small(frame_bgr)
        prev = self._prev_small
        if prev is None:                       # first step: nothing to diff against
            self._prev_small = small
            return None
        block_max = self._block_max(prev, small)
        # T-b flow-std is computed ONLY on the live path (never LOST / recovery /
        # probation / frozen) — this keeps the ~7 ms Farneback off the LOST-phase
        # real-time budget and keeps the rolling norm genuinely "live".
        live_path = (self.state is not TrackState.LOST
                     and self._reacq_phase is _Phase.NONE
                     and not self._feed_frozen)
        flow_std = (self._flow_std(prev, small)
                    if (self._feed_corrupt_enabled and live_path) else None)
        self._prev_small = small
        # --- T-a freeze state machine (applies in EVERY state) --------------- #
        if self._feed_frozen:
            self._motion_run = self._motion_run + 1 if block_max >= self._ff_t_static else 0
            if self._motion_run >= self._ff_exit_m:
                return self._exit_feed_frozen(frame_bgr, block_max)
            return self._frozen_result(block_max)
        self._static_run = self._static_run + 1 if block_max < self._ff_t_static else 0
        if self._static_run >= self._ff_enter_n:
            return self._enter_feed_frozen(block_max)
        # --- T-b corruption discount (live path only) ----------------------- #
        if flow_std is not None:
            norm = (float(np.median(self._flow_norm_window))
                    if len(self._flow_norm_window) >= self._corrupt_warmup else None)
            if norm is not None and flow_std >= max(self._corrupt_ratio * norm,
                                                    self._corrupt_floor):
                self._corrupt_discounts += 1
                return self._corruption_result(block_max, flow_std, norm)
            # non-corrupt live frame contributes to the rolling live-norm
            self._flow_norm_window.append(flow_std)
            if len(self._flow_norm_window) > self._corrupt_window:
                del self._flow_norm_window[0]
        return None

    def _feed_small(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self._ff_w, self._ff_h),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    def _block_max(self, prev_small: np.ndarray, cur_small: np.ndarray) -> float:
        """Max over 16x16 blocks of the per-block MEAN-abs frame diff, OSD blocks
        excluded (the calibrated T-a detector definition)."""
        bs, sw, sh = self._ff_block, self._ff_w, self._ff_h
        padh = ((sh + bs - 1) // bs) * bs
        padw = ((sw + bs - 1) // bs) * bs
        d = np.abs(cur_small - prev_small)
        if padh != sh or padw != sw:
            d = np.pad(d, ((0, padh - sh), (0, padw - sw)), mode="edge")
        blk = d.reshape(padh // bs, bs, padw // bs, bs).mean(axis=(1, 3))
        if self._frozen_block_mask is not None:
            blk = np.where(self._frozen_block_mask, -1.0, blk)
        return float(blk.max())

    @staticmethod
    def _flow_std(prev_small: np.ndarray, cur_small: np.ndarray) -> float:
        """Global flow-inconsistency: std of the dense Farneback flow magnitude
        (the T-b corruption signal). Params match the 2026-07-08 calibration."""
        flow = cv2.calcOpticalFlowFarneback(
            prev_small.astype(np.uint8), cur_small.astype(np.uint8),
            None, 0.5, 2, 15, 3, 5, 1.1, 0)
        return float(np.sqrt((flow * flow).sum(-1)).std())

    def _build_frozen_block_mask(self) -> Optional[np.ndarray]:
        """17x30 bool block-mask: True where a 16x16 block overlaps the OSD
        overlay (excluded from block-max). None when no overlay is installed."""
        if self._overlay is None:
            return None
        bs, sw, sh = self._ff_block, self._ff_w, self._ff_h
        padh = ((sh + bs - 1) // bs) * bs
        padw = ((sw + bs - 1) // bs) * bs
        m = cv2.resize(self._overlay, (sw, sh), interpolation=cv2.INTER_NEAREST)
        if padh != sh or padw != sw:
            m = np.pad(m, ((0, padh - sh), (0, padw - sw)), mode="edge")
        return (m.reshape(padh // bs, bs, padw // bs, bs) > 0).any(axis=(1, 3))

    def _frozen_result(self, block_max: float) -> "SessionResult":
        """FEED_FROZEN: hold the last point, suspend all evidence advancement."""
        r = utils.TrackResult(point=self.point, ok=False, n_points=0,
                              mean_error=float("inf"), source="frozen",
                              raw_point=self.point, confidence=0.0)
        return SessionResult(state=TrackState.FEED_FROZEN, point=self.point,
                             confidence=0.0, result=r,
                             signals=self._feed_signals("frozen", block_max=block_max))

    def _enter_feed_frozen(self, block_max: float) -> "SessionResult":
        self._feed_frozen = True
        self._motion_run = 0
        return self._frozen_result(block_max)

    def _exit_feed_frozen(self, frame_bgr, block_max: float) -> "SessionResult":
        """Confirmed exit (M motion frames): zero the Kalman velocity + re-seed at
        the HELD point, then LOW_CONFIDENCE for immediate revalidation. The
        bad-streak is PRESERVED (resumes from its suspended value); good / veto /
        unknown reset for a clean re-validation. Any in-progress recovery is
        abandoned (revalidate live; a genuinely gone target re-LOSTs and
        re-enters reacquisition)."""
        self._feed_frozen = False
        self._static_run = 0
        self._motion_run = 0
        held = self.point
        if held is not None:
            self.tracker.init(frame_bgr, held)   # KF rebuilt with zero velocity
            self._accepted_point = held
        self._good_streak = 0
        self._veto_streak = 0
        self._unknown_streak = 0
        self._reacq_phase = _Phase.NONE
        self.state = TrackState.LOW_CONFIDENCE
        r = utils.TrackResult(point=held, ok=False, n_points=0,
                              mean_error=float("inf"), source="frozen",
                              raw_point=held, confidence=0.0)
        return SessionResult(state=TrackState.LOW_CONFIDENCE, point=held,
                             confidence=0.0, result=r,
                             signals=self._feed_signals("exit", block_max=block_max))

    def _corruption_result(self, block_max: float, flow_std: float,
                           norm: float) -> "SessionResult":
        """T-b: an INERT frame — state + point held, tracker NOT stepped, no
        streak / Kalman / reference movement. Never FEED_FROZEN, never LOST."""
        r = utils.TrackResult(point=self.point, ok=False, n_points=0,
                              mean_error=float("inf"), source="corrupt-skip",
                              raw_point=self.point, confidence=0.0)
        return SessionResult(state=self.state, point=self.point, confidence=0.0,
                             result=r,
                             signals=self._feed_signals("corrupt-discount",
                                                        block_max=block_max,
                                                        flow_std=flow_std,
                                                        flow_norm=norm))

    def _feed_signals(self, feed: str, block_max=None, flow_std=None,
                      flow_norm=None) -> dict:
        return {"tracker_conf": 0.0, "similarity": None, "edge": False,
                "jump_vetoed": False, "overlay": None,
                "bad_streak": self._bad_streak, "feed": feed,
                "feed_block_max": None if block_max is None else round(block_max, 3),
                "feed_static_run": self._static_run,
                "feed_motion_run": self._motion_run,
                "feed_flow_std": None if flow_std is None else round(flow_std, 3),
                "feed_flow_norm": None if flow_norm is None else round(flow_norm, 3),
                "corrupt_discounts": self._corrupt_discounts}

    # ------------------------------------------------------------------ #
    # conservative adaptive reference
    # ------------------------------------------------------------------ #
    def _maybe_update_reference(self, frame_bgr, similarity, edge_near,
                                jump_vetoed, measured_ok: bool,
                                tracker_conf: float) -> str:
        """Re-snapshot the reference patch through the strict update gate.

        Gate (ALL must hold): positional trust this frame (tracker_conf >=
        LOW_CONFIDENCE_BELOW — the POSITIONAL axis, deliberately not the
        combined TrackState, which embeds similarity and would let identity
        decay freeze its own repair), a reliable measurement (not a coasted
        prediction), similarity to the CURRENT reference >= REF_UPDATE_MIN_SIM
        (continuity floor), no edge penalty, no jump veto. Qualifying frames
        are throttled: the snapshot happens on every REF_UPDATE_EVERY-th
        consecutive qualifying frame. A non-qualifying frame resets the
        throttle, so degradation stalls adaptation instantly.

        Returns the gate outcome for signals/tests: "updated" or the first
        blocking condition ("not-measured" | "veto" | "edge" |
        "low-tracker-conf" | "sim-unknown" | "low-sim" | "overlay" |
        "throttle" | "flat-content").
        """
        gate = None
        if not measured_ok:
            gate = "not-measured"
        elif jump_vetoed:
            gate = "veto"
        elif edge_near:
            gate = "edge"
        elif tracker_conf < self.cfg.LOW_CONFIDENCE_BELOW:
            gate = "low-tracker-conf"
        elif similarity is None:
            gate = "sim-unknown"
        elif similarity < self.cfg.REF_UPDATE_MIN_SIM:
            gate = "low-sim"
        # The reference must never be DOMINATED by the static overlay.
        # Snapshots carry their validity mask and masked NCC never compares
        # overlay pixels, so a majority-valid reference is safe — the gate
        # only rejects overlay-heavy patches (e.g. the crosshair disc), while
        # a target sitting ON a diagonal (~0.30 ROI coverage) keeps adapting.
        elif (self._overlay is not None
                and (self._overlay_coverage(self.point, frame_bgr.shape)
                     > self.cfg.REF_UPDATE_MAX_OVERLAY_FRAC)):
            gate = "overlay"
        if gate is not None:
            self._ref_qualify_streak = 0
            return gate
        self._ref_qualify_streak += 1
        if self._ref_qualify_streak < self.cfg.REF_UPDATE_EVERY:
            return "throttle"
        new_ref = self._gray_patch(frame_bgr, self.point)
        new_valid = self._valid_patch_identity(self.point, frame_bgr.shape)
        new_std = self._masked_std(new_ref, new_valid)
        if new_ref is None or new_std < 1e-6:
            return "flat-content"  # never degrade the reference to empty/flat
        self._ref_patch = new_ref
        self._ref_valid = new_valid
        self._ref_std = new_std
        self._ref_qualify_streak = 0
        self._ref_staleness = 0
        return "updated"

    # ------------------------------------------------------------------ #
    # signals
    # ------------------------------------------------------------------ #
    def _signals(self, tracker_conf, similarity, edge_near, jump_vetoed,
                 overlay_cov=0.0, ref_gate=None) -> dict:
        # ref_gate is the update-gate outcome on frames where the gate ran
        # (None on paths that never evaluate it, e.g. probation).
        return {"tracker_conf": round(tracker_conf, 4),
                "similarity": None if similarity is None else round(similarity, 4),
                "edge": edge_near, "jump_vetoed": jump_vetoed,
                "overlay": round(overlay_cov, 4),
                "bad_streak": self._bad_streak,
                "ref_staleness": self._ref_staleness,
                "ref_gate": ref_gate}

    def _gray_patch(self, frame_bgr: np.ndarray, point: "utils.Point2D") -> Optional[np.ndarray]:
        patch = utils.extract_patch(frame_bgr, point, self.cfg.PATCH_SIZE)
        if patch.image.size == 0:
            return None
        return cv2.cvtColor(patch.image, cv2.COLOR_BGR2GRAY)

    def _valid_patch(self, point: "utils.Point2D",
                     frame_shape) -> Optional[np.ndarray]:
        """uint8 validity patch aligned with _gray_patch: 255 = overlay-free.

        None when no overlay mask is installed (everything valid).
        """
        if self._overlay is None or point is None:
            return None
        h, w = frame_shape[:2]
        x0, y0, x1, y1 = utils.clamp_roi(point.x, point.y, self.cfg.PATCH_SIZE, w, h)
        return np.where(self._overlay[y0:y1, x0:x1] > 0, 0, 255).astype(np.uint8)

    def _valid_patch_identity(self, point: "utils.Point2D",
                              frame_shape) -> Optional[np.ndarray]:
        """RC1: validity patch from the IDENTITY (stroke-only) mask — 255 where
        the appearance patch is stroke-free scene. Used only for the M8 reference
        (_ref_valid) and _patch_similarity's current-patch validity, so a
        crosshair-region selection keeps usable scene pixels between the strokes
        (the seed disc erased all of them). Everything else keeps the seed mask.
        """
        if self._overlay_identity is None or point is None:
            return None
        h, w = frame_shape[:2]
        x0, y0, x1, y1 = utils.clamp_roi(point.x, point.y, self.cfg.PATCH_SIZE, w, h)
        return np.where(self._overlay_identity[y0:y1, x0:x1] > 0, 0, 255).astype(np.uint8)

    def _overlay_coverage(self, point: "utils.Point2D", frame_shape) -> float:
        """Fraction of the ROI at `point` covered by the fixed overlay."""
        valid = self._valid_patch(point, frame_shape)
        if valid is None or valid.size == 0:
            return 0.0
        return 1.0 - float(np.mean(valid > 0))

    @staticmethod
    def _masked_std(patch: Optional[np.ndarray],
                    valid: Optional[np.ndarray]) -> float:
        """Std of the patch over overlay-free pixels (all pixels when unmasked).

        Overlay pixels are static structure — including them would let a flat
        scene patch look textured to the near-uniform-reference guard.
        """
        if patch is None:
            return 0.0
        if valid is None:
            return float(patch.std())
        vals = patch[valid > 0]
        return float(vals.std()) if vals.size else 0.0

    def _patch_similarity(self, frame_bgr: np.ndarray,
                          point: "utils.Point2D") -> Optional[float]:
        """NCC of the current ROI against the reference patch, in [-1..1].

        Near borders the clamped patch shrinks; the comparison is ALIGNED — the
        reference is cropped to the region corresponding to the visible part of
        the ROI (using the clamp offsets), so a half-visible object at the edge
        is compared against the matching half of the reference instead of an
        arbitrary top-left crop. Returns None when the signal is uninformative
        (no/flat reference, sliver patch) — treated as neutral, never as bad.
        """
        if self._ref_patch is None or self._ref_std < 1e-6:
            return None
        cur = self._gray_patch(frame_bgr, point)
        if cur is None:
            return None
        # Overlay-aware comparison: masked scoring runs over overlay-free
        # pixels only (union of both validity masks) but searches the SAME
        # alignment space as the normal path — the full scale sweep plus
        # sliding-position tolerance. The two modes differ only in how pixels
        # are scored, never in robustness. Raw NCC near the overlay lies in
        # BOTH directions: shared static overlay pixels inflate it (endorsing
        # a lock onto the overlay), and a legitimate crossing depresses it
        # (manufacturing false LOST evidence). Masked scoring engages only
        # above OVERLAY_SIM_MASKED_MIN_CONTAM invalid pixels, so a stray
        # masked pixel cannot flip the scoring mode (below the threshold the
        # two scores agree to <0.02, measured).
        cur_valid = self._valid_patch_identity(point, frame_bgr.shape)
        ref_valid = self._ref_valid

        def _invalid_frac(v):
            return 0.0 if v is None else 1.0 - float(np.mean(v > 0))

        masked = (self._overlay is not None
                  and max(_invalid_frac(cur_valid), _invalid_frac(ref_valid))
                  > self.cfg.OVERLAY_SIM_MASKED_MIN_CONTAM)
        # Clamp offsets: how far the patch's top-left was pushed in from where
        # an unclamped patch would start (mirrors utils.clamp_roi arithmetic).
        h, w = frame_bgr.shape[:2]
        half = self.cfg.PATCH_SIZE // 2
        icx, icy = int(round(point.x)), int(round(point.y))
        x0, y0, _, _ = utils.clamp_roi(point.x, point.y, self.cfg.PATCH_SIZE, w, h)
        dx = max(0, x0 - (icx - half))
        dy = max(0, y0 - (icy - half))
        ref = self._ref_patch
        region = ref[min(dy, ref.shape[0]):, min(dx, ref.shape[1]):]
        ch = min(cur.shape[0], region.shape[0])
        cw = min(cur.shape[1], region.shape[1])
        if ch < self.cfg.MIN_PATCH_SIZE or cw < self.cfg.MIN_PATCH_SIZE:
            return None
        clamped = (cur.shape != ref.shape) or (dx > 0 or dy > 0)
        if clamped:
            # Border sliver: single-scale, corner-aligned comparison only.
            if masked:
                valid = np.full((ch, cw), 255, np.uint8)
                if cur_valid is not None:
                    valid[cur_valid[:ch, :cw] == 0] = 0
                if ref_valid is not None:
                    rv = ref_valid[min(dy, ref_valid.shape[0]):,
                                   min(dx, ref_valid.shape[1]):]
                    valid[rv[:ch, :cw] == 0] = 0
                return self._masked_ncc(cur[:ch, :cw], region[:ch, :cw], valid)
            res = cv2.matchTemplate(cur[:ch, :cw], region[:ch, :cw],
                                    cv2.TM_CCOEFF_NORMED)
            sim = float(res[0, 0])
            return sim if np.isfinite(sim) else 0.0
        # Full patch: scale-tolerant NCC — the best score across reference
        # scales wins, so a legitimate scale change (approach/descent) does not
        # permanently depress similarity, while wrong content stays low at
        # every scale. Size-mismatched comparisons slide the smaller image over
        # the larger one (mild translation tolerance comes for free). Masked
        # mode keeps the identical sweep: each scale PROPOSES an alignment by
        # sliding with the template's own validity mask, then the winning
        # alignment is re-scored exactly over the UNION of both validity masks
        # (bounded: one extra single-position call per scale).
        best = None
        for s in self.cfg.SIM_SCALES:
            scaled = (ref if abs(s - 1.0) < 1e-9 else
                      cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            ref_fits = (scaled.shape[0] <= cur.shape[0] and
                        scaled.shape[1] <= cur.shape[1])
            if masked:
                sv = (None if ref_valid is None else
                      cv2.resize(ref_valid, (scaled.shape[1], scaled.shape[0]),
                                 interpolation=cv2.INTER_NEAREST))
            else:
                sv = None
            if ref_fits:
                big, small, big_v, small_v = cur, scaled, cur_valid, sv
            else:
                big, small, big_v, small_v = scaled, cur, sv, cur_valid
            if small.shape[0] < self.cfg.MIN_PATCH_SIZE or small.shape[1] < self.cfg.MIN_PATCH_SIZE:
                continue
            if big.shape[0] < small.shape[0] or big.shape[1] < small.shape[1]:
                continue
            if masked:
                sim = self._masked_sweep_score(big, small, big_v, small_v)
                if sim is None:
                    continue
            else:
                res = cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED)
                sim = float(res.max())
            if np.isfinite(sim) and (best is None or sim > best):
                best = sim
        return best

    def _masked_sweep_score(self, big: np.ndarray, small: np.ndarray,
                            big_valid: Optional[np.ndarray],
                            small_valid: Optional[np.ndarray]) -> Optional[float]:
        """One scale's masked score with full sliding tolerance.

        Proposal: slide `small` over `big` with the template's own validity
        mask (matchTemplate cannot express the image-side mask while sliding).
        Verification: re-score the winning alignment exactly, over the UNION
        of both validity masks. Returns None when the winning alignment has
        too few valid pixels or the score is numerically undefined.
        """
        res = cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED,
                                mask=small_valid)
        res = np.where(np.isfinite(res), res, -2.0)
        _, _, _, loc = cv2.minMaxLoc(res)
        x, y = int(loc[0]), int(loc[1])
        sh, sw = small.shape[:2]
        win = big[y:y + sh, x:x + sw]
        union = np.full((sh, sw), 255, np.uint8)
        if small_valid is not None:
            union[small_valid == 0] = 0
        if big_valid is not None:
            bv = big_valid[y:y + sh, x:x + sw]
            union[bv == 0] = 0
        return self._masked_ncc(win, small, union)

    def _masked_ncc(self, img: np.ndarray, templ: np.ndarray,
                    valid: np.ndarray) -> Optional[float]:
        """Zero-mean NCC of two same-size patches over valid pixels only.

        Returns None (neutral) when too few pixels are overlay-free or when
        the valid region is flat (NCC undefined) — overlay proximity is a
        RISK, not loss evidence, mirroring the frame-edge semantics.
        """
        if float(np.mean(valid > 0)) < self.cfg.OVERLAY_SIM_MIN_VALID_FRAC:
            return None
        res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED, mask=valid)
        sim = float(res[0, 0])
        if not np.isfinite(sim):
            return None
        return max(-1.0, min(1.0, sim))

    def _on_dilated_overlay(self, point: Optional["utils.Point2D"]) -> bool:
        """True when the point sits inside the dilated overlay zone."""
        if self._overlay_dil is None or point is None:
            return False
        h, w = self._overlay_dil.shape[:2]
        x = min(max(int(round(point.x)), 0), w - 1)
        y = min(max(int(round(point.y)), 0), h - 1)
        return bool(self._overlay_dil[y, x] > 0)

    def _near_edge(self, point: "utils.Point2D", frame_w: int, frame_h: int) -> bool:
        x0, y0, x1, y1 = utils.clamp_roi(point.x, point.y, self.cfg.PATCH_SIZE,
                                         frame_w, frame_h)
        m = int(self.cfg.EDGE_MARGIN_PX)
        return (x0 <= m or y0 <= m or x1 >= frame_w - m or y1 >= frame_h - m)

    def _jump_exceeded(self, candidate: "utils.Point2D") -> bool:
        # Measure the teleport against the last MEASURED-accepted point, not the
        # (possibly coasted) display point — a drifting prediction must not widen
        # the veto gate. Fall back to self.point if no measurement has committed.
        ref = self._accepted_point if self._accepted_point is not None else self.point
        if ref is None or candidate is None:
            return False
        return ref.dist(candidate) > float(self.cfg.MAX_JUMP_PX)
