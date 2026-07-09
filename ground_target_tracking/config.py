"""config.py — single source of truth for all configurable parameters.

Every tunable value in the project lives here so the whole system can be
experimented with by editing one file, instead of hunting for magic numbers
across modules. All other modules import their parameters from here:

    from . import config
    patch = utils.extract_patch(frame, point, config.PATCH_SIZE)

Sections are ordered to mirror the pipeline (Video -> ROI -> ... -> Evaluation).
Parameters for milestones that are not implemented yet are included as
well-documented placeholders so the structure is ready to grow incrementally;
they are annotated with the milestone that will consume them.

This module imports only third-party libraries (cv2) — never project-local
modules — so it is the deepest dependency sink and can never cause a cycle.
"""
from __future__ import annotations

import cv2

# =========================================================================== #
# Video                                                                        #
# =========================================================================== #
DEFAULT_FPS = 30.0          # fallback playback fps when a clip reports 0 fps
DEFAULT_MAX_FRAMES = 0      # process this many frames (0 = the whole clip)
INIT_FRAME_INDEX = 0        # frame shown for target selection (M2)

# Video input / selection (used when --video is omitted)
VIDEOS_DIR = "videos"       # folder searched (relative to the working dir) for clips
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# Experiment logging (a `--save` run writes LOGS_DIR/<prefix>NNN/ ...)
LOGS_DIR = "logs"           # root folder for per-run experiment directories
RUN_DIR_PREFIX = "run_"     # run sub-folder prefix -> run_001, run_002, ...

# =========================================================================== #
# ROI / Patch                                                                  #
# =========================================================================== #
PATCH_SIZE = 51             # side length (px) of the square patch around the point (M3)
ROI_SIZE = PATCH_SIZE       # alias: the ROI and the patch are the same square here
MIN_PATCH_SIZE = 5          # guard: smallest sensible patch side length

# =========================================================================== #
# Display                                                                      #
# =========================================================================== #
MAX_DISPLAY_HEIGHT = 900    # tall frames are downscaled to <= this for on-screen windows
                            # (this is the effective interactive window height cap)
WINDOW_TITLE = "Ground Target Tracking"
PATCH_WINDOW_TITLE = "Selected Patch"
PATCH_VIEW_ZOOM = 4         # integer nearest-neighbour zoom for the patch window (display only)
PATCH_WINDOW_X = 540        # patch-window on-screen position (px from left) so it isn't
PATCH_WINDOW_Y = 60         # hidden behind the larger main / preprocess windows

# =========================================================================== #
# Preprocessing                                  (Milestone 4 — placeholders)  #
# =========================================================================== #
# "Preprocessing is part of the tracking method": different methods will read
# different subsets of these. No HSV / color segmentation by design.
USE_GAUSSIAN_BLUR = False
GAUSSIAN_KSIZE = 5
GAUSSIAN_SIGMA = 0
USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
GAMMA = 1.0                 # 1.0 = no gamma correction
PREPROC_RESIZE_SCALE = 1.0  # 1.0 = no resize

# =========================================================================== #
# Feature Extraction / ORB                       (Milestone 5 — placeholders)  #
# =========================================================================== #
ORB_N_FEATURES = 500
ORB_MIN_KEYPOINTS_WARN = 10  # warn if the initial patch yields fewer keypoints

# =========================================================================== #
# Optical Flow (Lucas-Kanade)                    (Milestone 6 — placeholders)  #
# =========================================================================== #
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
GFTT_MAX_CORNERS = 50        # goodFeaturesToTrack: max corners inside the patch
GFTT_QUALITY_LEVEL = 0.01
GFTT_MIN_DISTANCE = 3
LK_FB_ERROR_THRESHOLD = 1.0  # forward-backward consistency threshold (px)
LK_AFFINE_MIN_POINTS = 3     # min survivors to fit the similarity-transform update
                             # (below this, fall back to median translation)
LK_RANSAC_REPROJ_THRESHOLD = 1.0  # RANSAC inlier threshold (px) for the similarity fit.
                                  # OpenCV's default (3.0) admits static points as inliers
                                  # under slow scene motion; 1.0 matches LK_FB_ERROR_THRESHOLD.
GFTT_CROP_MARGIN_PX = 16     # pad around the seed ROI when cropping for GFTT (the corner
                             # response is otherwise computed over the WHOLE frame: ~8.6ms
                             # at 1080p even with an ROI mask, vs ~0.02ms on the crop)
LK_MIN_TRACK_POINTS = 6      # below this, flag low confidence (ok=False)
LK_REDETECT_BELOW = 10       # re-run goodFeaturesToTrack when live points drop below this
GFTT_MIN_SEED = 8            # below this many GFTT corners, add a grid-seed fallback
GRID_SEED_STEP = 10          # grid-seed lattice spacing (px) inside the ROI (low-texture fallback)

# =========================================================================== #
# Fixed HUD-overlay ignore mask (drone OSD: X diagonals + center crosshair)    #
# =========================================================================== #
# The official sample video burns a static overlay into every frame; its lines
# are strong corners/edges that GFTT/LK would otherwise latch onto. The mask
# excludes overlay pixels from feature SEEDING and penalizes measurements whose
# survivors sit on it. Display is never altered. DEFAULT ON since 2026-07-08
# (defaults promotion): this is the profile every piece of v8 acceptance
# evidence validates; measured harmless on clean footage for off-center picks
# (zero overlay coverage at init). Opt out per run with --no-overlay-mask.
# NOTE: the X/crosshair geometry is a FIXED central model (only the letterbox
# part is content-adaptive), so centered picks on clean videos inherit the
# central masking — covered by the external-clip validation gate.
# GEOMETRY MODEL (verified on the sample video, frames 0/100/400/800): the OSD
# anchors the X and crosshair to the FULL-FRAME centre — NOT to the content box
# between the pillarbox bars, which is asymmetric on this footage. Pillarbox
# detection and HUD geometry are therefore independent (see build_overlay_mask).
OVERLAY_MASK_ENABLED = True        # default ON (2026-07-08); disable with --no-overlay-mask
OVERLAY_DIAG_THICKNESS_PX = 12     # ignore band width along each X diagonal
OVERLAY_X_SLOPE = 4.0 / 3.0        # diagonal dx/dy, dimensionless: x = cx +/- SLOPE*(y-cy).
                                   # Measured 1.3321..1.3337 (RANSAC line fits, RMS<=0.55px)
                                   # across sampled frames; 4/3 = a 4:3 OSD reticle centred
                                   # in the frame. Model error <=~2.2px perp, absorbed by
                                   # the 6px band half-width.
OVERLAY_CROSSHAIR_RADIUS_PX = 45   # ignore disc around the frame center
# Identity (M8 local appearance-similarity) overlay mask — geometry-faithful:
# masks ONLY the actual crosshair STROKES (thin diagonals + "+" ticks + centre
# dot), NOT the clean scene between them, so a crosshair-region selection keeps
# usable appearance pixels for similarity. Separate from the conservative SEED
# disc above (OVERLAY_CROSSHAIR_RADIUS_PX), which is UNCHANGED and still used for
# LK seeding / ignore / dilation / known-occlusion. Reticle geometry measured on
# the sample video: dark diagonal ~3-5px at <=3px offset; "+" ticks ~2px thick,
# reach ~22px; centre dot ~1px. (RC1: the 45px filled disc erased the whole 51px
# appearance patch for a centre selection -> _ref_std=0 -> similarity None.)
OVERLAY_IDENTITY_DIAG_THICKNESS_PX = 8  # identity diagonal band (covers dark line+offset; << seed 12)
OVERLAY_TICK_GAP_PX = 5                 # "+" tick starts this many px from centre
OVERLAY_TICK_LEN_PX = 17                # "+" tick bar length (reach = gap+len ~= 22px)
OVERLAY_TICK_THICKNESS_PX = 3           # "+" tick bar thickness
OVERLAY_CENTER_DOT_RADIUS_PX = 3        # small centre-marker disc
OVERLAY_IDENTITY_DILATE_PX = 3          # small anti-alias safety dilation of the strokes
OVERLAY_LETTERBOX_MAX_INTENSITY = 12  # columns darker than this = letterbox bars
OVERLAY_LETTERBOX_MARGIN_PX = 6    # extra band masked at the content/bar boundary
OVERLAY_MAX_POINT_FRACTION = 0.5   # above this fraction of overlay-contaminated survivors,
                                   # the measurement is rejected as unreliable

# Mask-complete estimation (window-aware exclusion; all no-ops without a mask).
# LK cannot be masked (calcOpticalFlowPyrLK has no mask parameter), so survivors
# whose INTEGRATION WINDOW overlaps the overlay are excluded from the motion
# vote and from retention instead. The overlap maps are precomputed once at
# set_ignore_mask (the mask is static).
OVERLAY_SEED_DILATE_PX = 10        # dilate the seed-exclusion mask by the LK half-window
                                   # so no fresh seed's 21x21 window straddles the overlay
OVERLAY_SURVIVOR_MAX_OVERLAP = 0.10  # cull a survivor when more than this fraction of its
                                     # LK_WIN_SIZE window lies on the overlay (synthetic
                                     # ground-truth sweep: 0.20 -> 13.7px mean crossing
                                     # error, 0.10 -> 10.9, 0.05 -> 6.6; 0.10 balances
                                     # drag against depletion on low-texture ROIs)
OVERLAY_COARSE_SUPPORT_PX = 85     # probe size approximating pyramid-level support
                                   # (LK_WIN_SIZE x 2^2); catches coarse-level pull near
                                   # the crosshair disc that the fine window test misses
OVERLAY_COARSE_MAX_OVERLAP = 0.50  # cull when the coarse window is majority-overlay
                                   # (conservative: only fires in/near the crosshair disc)
OVERLAY_SIM_MIN_VALID_FRAC = 0.30  # below this fraction of overlay-free patch pixels the
                                   # NCC similarity is uninformative -> neutral (None),
                                   # mirroring the frame-edge "risk, not loss" semantics
OVERLAY_SIM_MASKED_MIN_CONTAM = 0.005  # switch NCC to masked scoring only above this
                                       # invalid-pixel fraction (~13px of a 51px patch).
                                       # Below it masked and unmasked scores agree to <0.02
                                       # (measured sweep on the sample video), so a single
                                       # stray masked pixel cannot flip the scoring mode.
REF_UPDATE_MAX_OVERLAY_FRAC = 0.35 # refuse adaptive-reference snapshots whose ROI overlay
                                   # coverage exceeds this. Snapshots carry their validity
                                   # mask and masked NCC never compares overlay pixels, so
                                   # a majority-valid reference is safe; the threshold must
                                   # admit a target ON a diagonal (~0.30 ROI coverage) while
                                   # still rejecting crosshair-dominated patches (~0.6+).
OVERLAY_INIT_MIN_SEEDABLE = 0.10   # init policy: below this seedable (non-dilated-overlay)
                                   # fraction of the ROI the selection is a HUD dead zone -
                                   # accepted, but reported as occluded (no fabricated
                                   # measurements). Measured on the sample video (51px ROI):
                                   # a click ON a diagonal leaves ~0.19 seedable and yields
                                   # 9-12 actual seeds; the crosshair disc leaves 0.000 and
                                   # yields none - 0.10 separates the two with ~2x margin.
# Known-occlusion LOST suspension: while the committed point lies INSIDE the
# OVERLAY_SEED_DILATE_PX-dilated overlay, bad frames freeze (never reset) the
# LOST streak — measurements there are structurally compromised by the overlay,
# so low evidence means "cannot see", not "gone". A crossing target resurfaces
# as LOW_CONFIDENCE and recovers; LOST keeps meaning "actually gone". Known
# limitation: a target that truly disappears while the estimate sits on the
# overlay stays LOW_CONFIDENCE (with signals['overlay'] raised) until M9.

# =========================================================================== #
# Kalman Filter                                  (Milestone 7 — placeholders)  #
# =========================================================================== #
KALMAN_PROCESS_NOISE = 0.1   # retuned with the similarity update: cuts transient
                             # filter lag ~4x with no downside on any GT metric
KALMAN_MEASUREMENT_NOISE = 1e-1
KALMAN_MAX_PREDICT_FRAMES = 15  # how long to coast on prediction before reporting ok=False
                                # (~0.25 s at 60 fps footage; was 8, retuned with M8)
KALMAN_INIT_ERROR_COV = 1.0    # initial posterior error-covariance scale (state uncertainty at init)

# =========================================================================== #
# Feature Matching / Reacquisition               (Milestone 9 — placeholders)  #
# =========================================================================== #
FM_RATIO_TEST = 0.75
FM_MIN_MATCHES = 10
FM_USE_HOMOGRAPHY = True

# =========================================================================== #
# Reacquisition — Stage 2B OPERATIONAL parameters       (Milestone 9)          #
# =========================================================================== #
# These configure HOW raw reacquisition evidence is COMPUTED (window sizes,
# pyramid, scale ladder, capability floors). They are NOT decision constants:
# no accept/reject threshold, no peak-margin, no classification lives here.
# All accept/reject/classification constants belong to Stage 2C. Every value
# below is an INITIAL EXPERIMENTAL VALUE, calibrated in Stage 2C.
REACQ_TEMPLATE_SIZE = 153      # side (px) of the context reference window (E2: 51px is vacuous on v8)
REACQ_PYRAMID_SCALE = 0.5      # coarse-search downscale for the proposal sweep (E7: half-res fits budget)
REACQ_SCALES = (0.6, 0.8, 1.0, 1.3, 1.7)  # reference-scale ladder (return-time scale may differ, E4)
REACQ_MIN_REF_STD = 2.0        # capability floor: masked std below this -> context cue disabled (flat/vacuous)
REACQ_KP_RADIUS_PX = 100       # keep full-frame ORB keypoints within this radius of the selected point
REACQ_MIN_REF_KP = 10          # capability floor: fewer kept keypoints -> descriptor cue disabled.
                               # Independent Stage-2B value; deliberately NOT aliased to FM_MIN_MATCHES
                               # so M9 calibration stays decoupled from the closed Stage-1 config.
REACQ_PEAK_NMS_FRAC = 0.5      # NMS radius = this * scaled-template side, for distinct raw peaks
REACQ_COARSE_MASKED = True     # masked coarse sweep (False = mean-fill invalid + unmasked, perf escape hatch)

# --- Stage 2B feature-proposal (M9-a) CANDIDATE-GENERATION parameters -------- #
# These govern HOW raw ORB-feature geometry is CONSTRUCTED (match -> ratio-test
# -> RANSAC similarity fit) so propose_features() can emit a raw FeatureCandidate.
# They are candidate-generation / RECALL knobs, NOT identity-acceptance thresholds:
# loosening them yields more/looser raw candidates, tightening yields fewer; NONE
# of them accepts a candidate as identity. Every accept/reject/routing decision
# (min inliers, min inlier ratio, residual limit, competing-candidate margin,
# immutable-reference confirmation, persistence, absence handling, conservative
# refusal) belongs to Stage 2C (M9-b), which may also re-calibrate the three
# values below purely as recall knobs. Each is an INITIAL EXPERIMENTAL,
# UNCALIBRATED value; no value here is a validated identity threshold.
REACQ_FEAT_RATIO = 0.75        # Lowe ratio-test cutoff for KNN correspondences (recall knob)
REACQ_FEAT_MIN_MATCHES = 6     # experimental structural robustness floor of ratio-survivors, chosen
                               # BEFORE attempting the fit (NOT the mathematical minimum required by
                               # estimateAffinePartial2D, NOT an acceptance gate) -> below it no fit
                               # is attempted and propose_features() returns [].
REACQ_FEAT_RANSAC_THRESH = 3.0 # px reprojection tolerance for the RANSAC similarity fit (mirrors the
                               # LK default; a generation param, re-calibratable in Stage 2C)
REACQ_FEAT_DETECT_SCALE = 0.66 # experimental operational CANDIDATE-GENERATION detect scale: ORB runs on a
                               # frame downscaled by this factor (both reference AND query) and keypoint
                               # coords are mapped back to full-resolution, to keep the ORB primitive within
                               # the per-frame budget across scene texture. NOT an identity threshold, NOT a
                               # V8-calibrated value; a recall/runtime knob, re-calibratable in Stage 2C.

# =========================================================================== #
# Reacquisition — Stage 2C DECISION parameters              (Milestone 9-b)    #
# =========================================================================== #
# IDENTITY-DECISION thresholds owned by the Stage 2C decision layer
# (reacquisition.classify_* / best_candidate / HypothesisTracker). Unlike the
# REACQ_FEAT_* generation knobs above, these decide accept/refuse. Every value
# below is CALIB-selected on synthetic seeds 0..19 and validated FROZEN on the
# disjoint HELD-OUT seeds 100..119 — NEVER on v8 (report-only).
#
# CALIBRATION OUTCOME (M9-b): under these frozen values, true-positive coverage
# is 200/200 feature + 20/20 border + 40/40 template on BOTH CALIB and HELD-OUT
# with accepted_wrong == 0; wrong-location/drift, unrelated absence, low-blend
# (<=0.5) decoys and all template negatives are refused. The zero-false-accept
# gate is NOT met for the NEAR-COPY capability-floor classes: a self-copy decoy
# at blend >= ~0.65 of the exact target patch (true target absent) reproduces
# the immutable descriptors AND the immutable context, so it is indistinguishable
# from the true target on every available axis (inliers/ratio/residual/confirm).
# This is a documented appearance-only limitation, NOT hidden -> REACQ_DECISION_
# ENABLED stays False (see plan R11). Persistence mitigates transient/inconsistent
# hypotheses only, NOT a persistent high-similarity copy.
#
# Feature-cue identity gates (candidates from propose_features):
REACQ_MIN_INLIERS = 12           # min RANSAC inlier count for identity
REACQ_MIN_INLIER_RATIO = 0.5     # min inliers / ratio-test survivors
REACQ_MAX_RESIDUAL_PX = 2.0      # max median reprojection residual (full-res px)
REACQ_FEAT_SCALE_MIN = 0.5       # accepted fitted-scale envelope (mirrors the
REACQ_FEAT_SCALE_MAX = 2.0       #   REACQ_SCALES design range, continuous)
REACQ_CONFIRM_MIN_NCC_FEAT = 0.20  # min fitted-pose NCC vs the immutable context
                                   # (verify_at_pose; pose-warp interpolation loss
                                   # makes this DELIBERATELY separate from _TMPL.
                                   # Separates drift/wrong-loc (<=0.037) from
                                   # positives (>=0.33); cannot separate near-copies)
# Template-fallback identity gates (candidates from propose; near-upright only):
REACQ_TEMPLATE_MIN_NCC = 0.60    # min raw peak NCC
REACQ_TEMPLATE_MIN_MARGIN = 0.60 # min (raw - second_peak): repeated structure refusal
REACQ_CONFIRM_MIN_NCC_TMPL = 0.60  # min upright verify_at at the peak point
# Option-C persistence (STRUCTURAL design values, not appearance thresholds):
REACQ_PERSIST_N = 3              # consecutive compatible MATCH frames before the
                                 # one-shot AcceptedHypothesis is emitted
REACQ_PERSIST_MAX_MOVE_PX = 30.0 # max inter-evaluation move to join a hypothesis
REACQ_PERSIST_SCALE_TOL = 0.25   # max relative scale deviation to join
REACQ_PERSIST_MAX_NEUTRAL = 2    # NEUTRAL (no-observation) frames a hypothesis may
                                 # freeze through; one more clears it (bounded gap)
# --------------------------------------------------------------------------- #
# SIFT last-confident route (v8 design review, 2026-07-07). A SECOND, mutable-
# by-snapshot reference captured while tracking is still trustworthy, matched
# with SIFT during LOST. Calibrated on the measured v8 return (rotation
# ~150-180deg, 10-14 inliers) vs the full absence sweep (max 7 inliers
# anywhere). These are SIFT-ROUTE-ONLY values: the ORB/template gates above
# are deliberately untouched. inlier_ratio and confirm-NCC gates are OMITTED
# for this route by design — both were measured INVERTED on the v8 return
# (true-return ratio 0.32-0.42 vs spurious small-sample fits at 0.5+; true
# confirm 0.04-0.16 because the return is an opposite-side view); identity
# rests on RANSAC geometry + Option-C persistence + probation instead.
REACQ_SIFT_KP_RADIUS_PX = 150   # keep SIFT keypoints within this radius of the point
                                # (150 -> 271 kp on the v8 snapshot; 100 was too sparse)
REACQ_SIFT_RATIO = 0.8          # Lowe ratio for the SIFT route (0.75 starves the
                                # rotated return: 7-10 inliers vs floor 10)
REACQ_SIFT_MIN_INLIERS = 10     # identity floor (v8: return 10-14, absence max 7)
REACQ_SIFT_EVERY = 10           # RETIRED 2026-07-08 (G2-a): the session no longer
                                # cadences whole-frame SIFT ticks; the stripe sweep
                                # below governs. Constant kept for compatibility.
# --- G2-a budgeted stripe slicing (2026-07-08) ----------------------------- #
# Whole-frame SIFT ticks measured 110-350 ms at 1080p (and the one-time
# reference build 100-240 ms) — over the 33.3 ms real-time budget while LOST.
# The route is re-shaped into per-tick stripes: the REFERENCE builds striped at
# FULL resolution (measured identity-lossless: striped-union disc keypoints
# 308 vs 307 whole-frame), the QUERY accumulates one stripe per executed tick
# at 0.8 detect scale (measured: 12-18 ms/stripe; return inliers 18-29 vs
# floor 10; absence <=3; ref-side downscale is NOT viable — 76 disc kp, 4-9
# inliers — which is why only the query side is scaled). One fit per completed
# sweep; identity gates unchanged.
REACQ_SIFT_STRIPES = 8            # stripes per sweep (per-stripe ~12-18 ms @1080p)
REACQ_SIFT_DETECT_SCALE = 0.8     # query-side detect scale (reference stays 1.0)
REACQ_SIFT_STRIPE_OVERLAP_PX = 48 # stripe overlap at detect scale (border kp safety)

# Master gate — DEFAULT ON since 2026-07-08 (G1, user-approved decision):
# the enable evidence is the 9-pick official-sample acceptance grid (loss ->
# sift-lc reacquisition -> sustained lock; zero absence-window accepts), the
# synthetic 1080p exit/return benchmark (reacquires to exact GT), the measured
# false-lock floor (junk 57-113px off-target, all rejected), and the G2-a
# budgeted search (LOST-phase full-op mean ~25 ms; min rolling-1s throughput
# 37 frames). Loss stays honestly declared; recovery is now the delivered
# default. Disable per run with --no-reacq (the old --reacq opt-in is a
# compatibility no-op).
REACQ_DECISION_ENABLED = True

# =========================================================================== #
# Reacquisition — Stage 2D SESSION-INTEGRATION parameters   (Milestone 9-c)    #
# =========================================================================== #
# STRUCTURAL recovery/probation controls owned by the TrackingSession
# integration (session.py). They are NOT appearance thresholds and NOT a
# recalibration of the M9-b decision gates: probation reuses the existing M8
# confidence signals. Reacquisition runs ONLY while the M8-declared LOST state
# is active AND REACQ_DECISION_ENABLED (or the per-run --reacq override) is on.
REACQ_SEARCH_EVERY = 1         # run best_candidate() every Nth LOST frame (>=1). Skipped
                               # frames leave HypothesisTracker UNTOUCHED — persistence is
                               # measured in executed evaluations, never in skipped frames.
REACQ_PROBATION_N = 5          # healthy post-reseed frames required to return to TRACKING
REACQ_PROBATION_MAX_FRAMES = 15  # total post-reseed frames allowed before probation TIMES OUT
                               # (>= REACQ_PROBATION_N). Bounds probation even under persistent
                               # LOW_CONFIDENCE-band tracking. SUCCESS takes precedence over the
                               # deadline if both trigger on the same frame.

# --- Phase 2: bounded SEARCHING scheduler (descriptor-free template path) ---- #
# STRUCTURAL scheduling parameters for the incremental tiled scan + candidate
# list + verify-first scheduling (TemplateScanScheduler). They bound WORK
# QUANTITY per SEARCHING frame; wall-clock time is empirically calibrated and
# measured, occasional overruns may occur and are logged, non-fatal. None of
# these is an identity-acceptance threshold: the pre-gate reuses
# REACQ_TEMPLATE_MIN_NCC and verification reuses the existing verify_at +
# _classify_template gates unchanged.
REACQ_SCAN_UNITS_PER_FRAME = 3   # max atomic scan units on a scan-only SEARCHING frame
REACQ_SCAN_MIN_UNITS = 1         # min scan units on a scan frame (guaranteed rolling progress)
REACQ_CAND_QUEUE = 3             # max pending candidates (bounded, spatially distinct)
REACQ_CAND_MIN_SEP_PX = 100.0    # full-res px: closer candidates collapse to one site
                                 # (stronger kept); also the rejected-site cooldown radius
REACQ_CAND_TTL_FRAMES = 30       # unverified candidates expire after this many SEARCHING frames
REACQ_REJECT_COOLDOWN_FRAMES = 60  # rejected (AMBIGUOUS) sites are not re-enqueued for this long
REACQ_RESEARCH_WIN_PX = 192      # half-res px: live re-search window side (candidate age <= 1)
REACQ_RESEARCH_WIN_CAP_PX = 256  # half-res px: window side for older candidates (hard cap)
REACQ_SCAN_BUDGET_MS = 16.0      # empirical per-frame M9 work TARGET the scheduler sizes work
                                 # against (EMA-calibrated unit cost). NOT a guarantee: an
                                 # overrun frame is logged and adapted to, never fatal.
REACQ_SCAN_STRIPES = 6           # full-width stripes per scale = atomic-unit granularity
                                 # (5 scales x 6 ~= 30 units/cycle at 1080p, ~5 ms/unit measured)

# =========================================================================== #
# Confidence / Lost-state machine                              (Milestone 8)   #
# =========================================================================== #
# State thresholds on the combined confidence (tracker x similarity x edge):
LOW_CONFIDENCE_BELOW = 0.40    # below this -> LOW_CONFIDENCE
LOST_CONFIDENCE_BELOW = 0.15   # below this (persistently) -> LOST
MIN_PATCH_SIMILARITY = 0.30    # reserved for the M9 reacquisition verify gate (similarity
                               # already contributes to LOST via the combined confidence)

# Tracker-level confidence (OpticalFlowTracker / KalmanWrapper):
CONF_POINTS_NORM = 20          # survivor count that maps to a full points-score of 1.0
CONF_COAST_DECAY = 0.85        # per-frame confidence decay while Kalman coasts (predict)

# Scale-tolerant similarity: NCC is evaluated with the reference resized by each
# of these factors and the best score wins. Lets similarity (and therefore the
# existing recovery hysteresis) survive a legitimate scale change (approach /
# descent) without weakening junk detection — wrong content scores low at every
# scale. Applied only when the ROI is full-size (border patches stay 1.0-only).
SIM_SCALES = (0.8, 1.0, 1.25)

# Conservative adaptive reference (similarity template follows the target's
# legitimate appearance change; the strict gate keeps junk from ever becoming
# the reference — updates happen ONLY on strong TRACKING frames):
REF_UPDATE_MIN_SIM = 0.50      # qualifying frame must still match the current reference this well
REF_UPDATE_EVERY = 5           # update at most every Nth consecutive qualifying frame

# Session-level state machine:
LOST_AFTER_N_BAD = 20          # consecutive bad frames (combined conf) before LOST (~1/3 s @60fps)
LOST_UNKNOWN_SIM_AFTER_N = 20  # identity-unknown policy (Run C): consecutive frames similarity
                               # may stay None (uninformative CURRENT patch, e.g. border sliver,
                               # while the reference IS informative) before further unknown frames
                               # count as loss evidence. Unknown frames also never RESET an
                               # existing bad streak (unknown != confirmed). Structural bound,
                               # mirrors LOST_AFTER_N_BAD; pure-unknown LOST latency is the sum
                               # of the two (~40 frames). Flat/absent reference (low-texture
                               # target) keeps full legacy None-neutrality — no escalation.
RECOVER_MARGIN = 0.05          # hysteresis: recovery needs conf >= LOW_CONFIDENCE_BELOW + margin
RECOVER_N = 5                  # ...for this many consecutive frames
RECOVER_MIN_SIM = 0.50         # NCC appearance re-validation required to promote back to
                               # TRACKING (Stage 1): a real measure must still match the
                               # reference. similarity None (flat/low-texture) is neutral;
                               # a present-but-low score is the false-lock signature -> no
                               # recovery. Mirrors REF_UPDATE_MIN_SIM; independently tunable.
EDGE_MARGIN_PX = 48            # ROI within this distance of the frame border -> edge penalty
EDGE_PENALTY = 0.5             # confidence multiplier applied near/at the border
MAX_JUMP_PX = 2 * PATCH_SIZE   # one-frame displacement above this -> measurement vetoed
JUMP_VETO_MAX_FRAMES = 3       # consecutive vetoes before accepting the motion as real (pan/shake)

# LOST-entry correctness for off-frame exits (Stage 2A). A candidate point
# outside the frame is UNOBSERVABLE — there is no patch, no similarity, no
# scene support at it — so with the flag on such a frame counts as loss
# evidence (bad) regardless of how confident the measurement claims to be,
# and the known-occlusion suspension does not apply (an off-frame point is
# gone, not HUD-occluded; border-clamping it onto an overlay pixel froze the
# LOST streak forever — measured on the official sample: the final 197 frames
# reported TRACKING at y≈1166 on a 1080-row frame). Default OFF until the
# Stage-2E evidence gate; enabled per-run/per-test during Stage 2.
LOST_OFFFRAME_HARDENING = False
LOST_OFFFRAME_MARGIN_PX = 0    # a point this many px beyond the border still counts as in-frame

# =========================================================================== #
# Feed-health: FEED_FROZEN detection + corruption discount   (Commit 4a, §E)   #
# =========================================================================== #
# A DISTINCT surfaced condition for a DEAD / REPEATED input feed (signal loss),
# separate from LOST (target gone) and LOW_CONFIDENCE (degraded track). While
# frozen the last point is HELD, evidence advancement is SUSPENDED (no scoring,
# streaks, reference/snapshot, or reacquisition search), and the ONLY surfaced
# label is FEED_FROZEN_BANNER_TEXT — never a TRACKING/LOW_CONFIDENCE claim on a
# non-changing feed. On confirmed exit the Kalman is zeroed + re-seeded at the
# held point and the state becomes LOW_CONFIDENCE for immediate revalidation.
#
# T-a freeze detector: block-max = max over 16x16 blocks (OSD-masked) of the
# per-block MEAN-abs frame diff at 480x270 (INTER_AREA); < T_static for N
# consecutive frames ENTERS, >= T_static for M consecutive EXITS. T_static
# CALIBRATED 2026-07-08 on v8 + the approved external clip (white squirrel,
# 1080p59.94, CC BY-SA 4.0) + synthetic >=30fps fixtures:
#   frozen (v8 tail): 61% exactly 0, p95 0.71 ; live worst-sustained-3
#   (quietest 3-consecutive window): v8 4.70 / clip 7.23 ; required no-hold
#   small-moving-target worst-sustained-3 = 1.91 ; slow-motion 3.60.
#   => VALIDATED SAFE INTERVAL (0.711, 1.908). T_static = 1.5 enters on a true
#   frozen feed and does NOT false-hold v8/clip live footage, the small-moving-
#   target fixture, or the slow-motion fixture. 2.0 is REJECTED — it false-holds
#   the required small-moving-target no-hold case (1.91 < 2.0). INHERENT,
#   DOCUMENTED, HARMLESS LIMITATION: a genuinely MOTIONLESS live scene (only
#   sensor noise, nothing moving) is indistinguishable from a frozen feed by
#   frame differencing and will hold — holding is harmless when nothing
#   (including the target) is moving. The block aggregation is MEAN-per-block
#   (robust); this calibration is specific to that definition.
FEED_FROZEN_ENABLED = True        # delivered default ON (2026-07-08); absent/False -> no-op
FEED_FROZEN_T_STATIC = 1.5        # block-max threshold (validated gap (0.711, 1.908))
FEED_FROZEN_ENTER_N = 3           # consecutive quiet frames (block-max < T_static) to enter
FEED_FROZEN_EXIT_M = 2            # consecutive motion frames (block-max >= T_static) to exit
FEED_FROZEN_SMALL_W = 480         # frame-diff downscale width  (INTER_AREA)
FEED_FROZEN_SMALL_H = 270         # frame-diff downscale height
FEED_FROZEN_BLOCK_PX = 16         # block side for the per-block mean-abs-diff
FEED_FROZEN_BANNER_TEXT = "FEED FROZEN - holding last position"  # HUD banner (distinct)
# --- T-b corruption discount (adaptive 3x rolling live-flow-std) — REJECTED -- #
# TESTED AND REJECTED from the shipping bundle (user ruling, 2026-07-08): the
# adaptive 3x rolling-live-flow-std discount would mark a live-path frame INERT
# (no streak/Kalman/reference movement) when its Farneback flow-std exceeds
# RATIO x the rolling live-norm AND a floor. It caused a REAL v8 grid regression:
# excluding discounted frames from the norm froze the norm at the early slow-
# drone baseline, so the sustained faster motion of the house EXIT (reads ~45-206,
# 130 discounts) stayed above 3x and was discounted continuously -> the LOST
# bad-streak could not accumulate -> first LOST slipped 235 -> 489 and episode-1
# reacquisition never fired. It is DISABLED by default and NOT redesigned in this
# bundle; the code path stays behind the flag for reproducibility of the
# rejection. MAX_JUMP_PX (jump veto) remains the backstop against corrupt-frame
# measurement jumps. The constants below are dormant while the flag is False.
FEED_CORRUPT_ENABLED = False      # DISABLED (rejected — see above); T-b never runs
FEED_CORRUPT_FLOW_RATIO = 3.0     # discount when flow-std >= RATIO x rolling live-norm ...
FEED_CORRUPT_FLOW_FLOOR = 1.0     # ... AND >= this absolute floor (protects low-motion scenes)
FEED_CORRUPT_FLOW_WINDOW = 60     # rolling window (frames) for the live-norm median
FEED_CORRUPT_FLOW_WARMUP = 15     # live frames before the discount may fire (norm established)

# =========================================================================== #
# Evaluation                          (HIT_THRESHOLD used at M3; rest at M11)  #
# =========================================================================== #
HIT_THRESHOLD = 50.0        # px; impact is a HIT if distance(target, impact) <= this
# The following are emitted at M11 once tracking exists:
# REPORT_FPS = True
# REPORT_LOST_FRAMES = True

# =========================================================================== #
# Visualization                                                                #
# =========================================================================== #
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
BANNER_WIDTH = 32                   # width of the console summary banner rule
HUD_TEXT_COLOR = (0, 255, 0)        # BGR — green
HUD_TEXT_THICKNESS = 1
HUD_OUTLINE_THICKNESS = 3           # dark outline behind HUD text for legibility
TARGET_POINT_COLOR = (0, 0, 255)    # BGR — red: the selected ground target      (M2)
TARGET_POINT_RADIUS = 5             #                                            (M2)
ROI_BOX_COLOR = (0, 255, 255)       # BGR — yellow: ROI while TRACKING           (M3)
ROI_BOX_THICKNESS = 2               #                                            (M3)
ROI_BOX_COLOR_LOW = (0, 165, 255)   # BGR — orange: ROI in LOW_CONFIDENCE/PREDICT (M8)
ROI_BOX_COLOR_LOST = (0, 0, 255)    # BGR — red: ROI at the last credible point in LOST (M8)
LOST_BANNER_TEXT = "LOST - target left view / tracking unreliable"  # HUD banner (M8)
TRAJECTORY_COLOR = (255, 0, 0)      # BGR — blue: the tracked trajectory polyline (M6)
TRAJECTORY_THICKNESS = 2            #                                            (M6)
MEASUREMENT_POINT_COLOR = (180, 180, 180)  # BGR — grey: raw pre-Kalman point (--show-measurement) (M7)

# =========================================================================== #
# Debug                                                                        #
# =========================================================================== #
DEBUG = False               # verbose logging
SHOW_PATCH_WINDOW = True    # show the cropped patch in a separate window        (M3)

# --- TEMPORARY debug switches for visual validation (remove before final cleanup) ---
# Flip these so a plain Run of main.py (no CLI flags) pops up the inspection
# windows. They OR with the --show-preprocess / --show-keypoints CLI flags.
# Set both back to False to disable; they are not part of the final workflow.
DEBUG_SHOW_PREPROCESS = False  # TEMP (M4):   show the "raw | processed" window
DEBUG_SHOW_KEYPOINTS = False   # TEMP (M5.5): overlay keypoints on the patch window

# =========================================================================== #
# Regional local-motion estimation (Proposal A) — PRODUCTION DEFAULT           #
# =========================================================================== #
# Promoted to the delivered default on 2026-07-08 (defaults promotion): this
# is the validated profile behind all v8 acceptance evidence, and it is
# load-bearing for HUD-covered selections (measured: without it, a masked
# center pick never declares LOST — it coasts frozen in LOW_CONFIDENCE for the
# whole video). When this flag is False the tracker's update() path is
# byte-for-byte the legacy behavior: none of the constants below are read.
# Disable per-run with --no-regional-motion (pairs with --no-overlay-mask,
# since overlay sensing installs the mask this path consults).
# See OpticalFlowTracker._update_regional.
#
# Rationale (from the design review + bounded probes): at the crosshair the
# 51x51 ROI is ~half-contaminated by the X-crossing, so the OVERLAY_MAX_POINT_
# FRACTION=0.5 veto rejects frames that still hold ~15 coherent clean survivors.
# The regional path draws clean scene support from an enclosing neighborhood,
# fits a local motion model to clean-ONLY survivors, and transports the exact
# selected coordinate by it — replacing the clean-FRACTION veto with evidence-
# quality gates (absolute clean count + inliers + consensus + residual +
# locality/parallax). It does NOT change the default path or its guard.
EXPERIMENTAL_REGIONAL_MOTION = True   # default ON (2026-07-08); disable with --no-regional-motion

# Support-region geometry. Grow from MIN until the clean floor is met, STOP at
# the first sufficient radius (locality-first — never jump to MAX for feature
# count), SHRINK to the near band on parallax. MIN=40: an 81px box yields the
# clean floor for an ordinary point. MAX=120: P1 measured motion-surface
# coherence out to ~240px radius on this planar aerial scene; 120 is half that
# (a safe cap that still clears the dilated crosshair disc -
# OVERLAY_CROSSHAIR_RADIUS_PX 45 + OVERLAY_SEED_DILATE_PX 10 = 55px - for a
# center selection while limiting parallax exposure to well inside the measured
# coherent range). STEP=20 balances adaptivity against per-frame recompute cost.
REGIONAL_MIN_RADIUS_PX = 40
REGIONAL_MAX_RADIUS_PX = 120
REGIONAL_RADIUS_STEP_PX = 20
REGIONAL_GFTT_MAX_CORNERS = 120     # cap corners detected in the (larger) region

# Evidence-quality gates (replace the clean-FRACTION veto INSIDE this path only).
# MIN_CLEAN_POINTS=10: a 4-DOF similarity needs >=3 for RANSAC; 10 gives >2x
# redundancy for outlier rejection and a stable estimate (crosshair yields ~15).
# MIN_INLIERS=8: after RANSAC, >=8 inliers on a 4-DOF model stays 2x over-
# determined. MIN_INLIER_RATIO=0.6: a coherent rigid surface yields a clear
# inlier majority at a 1px threshold; a lower ratio signals mixed-depth /
# independent motion -> reject. MAX_RESIDUAL_PX=0.5: RANSAC threshold is 1.0px;
# requiring median inlier residual <=0.5px (P1 planar residual ~0.13px) enforces
# a tight fit well below the LK FB threshold.
REGIONAL_MIN_CLEAN_POINTS = 10
REGIONAL_MIN_INLIERS = 8
REGIONAL_MIN_INLIER_RATIO = 0.6
REGIONAL_MAX_RESIDUAL_PX = 0.5

# Motion-aware RANSAC/coherence threshold. Pyramidal-LK sub-pixel error grows
# ~linearly with displacement, so a FIXED 1px inlier threshold wrongly rejects
# coherent FAST motion (many correct points scatter beyond 1px at ~14px/frame).
# The inlier threshold is therefore max(LK_RANSAC_REPROJ_THRESHOLD,
# FRAC * median_support_motion): a floor of 1px at slow motion, widening with
# speed. 0.15 (~LK error is empirically 10-15% of displacement) still separates
# genuinely different motion surfaces (e.g. 5 vs 2 px = 60% apart >> 15%), which
# the scale-invariant band-agreement gate catches regardless.
REGIONAL_RANSAC_MOTION_FRAC = 0.15

# Locality / parallax gate. Split clean support into a near band (dist <
# radius/2) and a far band; fit each and transport the SELECTED point by each;
# if the two transported positions disagree by more than this, the support spans
# incompatible motion surfaces (parallax) -> shrink to the near band, or PREDICT
# if the near band is below the clean floor. AGREEMENT=1.0px: P1 measured planar
# near/far agreement of 0.073px; P2 measured 3.0px under genuine parallax; 1.0px
# (= the LK FB and RANSAC threshold) cleanly separates the two.
REGIONAL_BAND_AGREEMENT_PX = 1.0
