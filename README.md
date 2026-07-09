# Selected-Object Video Tracker

A Python + OpenCV program that tracks a **user-selected object** in a video, in
**real time on a CPU** (no GPU). The user supplies a video and a pixel on the first
frame; the program follows the object at that pixel, drawing its motion and a
surrounding tracking box on a simple UI, detects **loss** of the target, and
**reacquires** it when it returns to the frame.

This project implements the **Asio Technologies** assignment *"Tracking and
Reacquisition in Real-Time Video Processing."* See
[Assignment Mapping](#assignment-mapping) for a requirement-by-requirement status,
and [`SUMMARY.md`](SUMMARY.md) for the written summary (six questions).

> **Note on the package name.** The Python package is `ground_target_tracking/`;
> the name is retained for import stability. The tracker itself is general — it
> follows any clear user-selected object.

---

## Requirements & versions

Developed and tested with:

- **Python 3.9.6** (invoked as `python3`)
- **opencv-python 4.13.0**
- **numpy 2.0.2**

```bash
python3 -m pip install -r ground_target_tracking/requirements.txt
```

`requirements.txt` specifies minimum versions (`opencv-python>=4.13`, `numpy>=2.0`),
so a fresh install may resolve to newer releases than the tested versions above.

**Hardware / development environment:** MacBook Air 15-inch, Apple Silicon
(`arm64`), macOS 26.5.1 — **CPU-only, no GPU** and no special hardware required.

---

## How to run

**In VS Code:** `File → Open Folder` on this repo and use the integrated terminal
(rooted at the project). Run from the **project root** (the folder containing
`ground_target_tracking/`):

```bash
# Pick a local video (menu / file picker), then CLICK the target pixel
python3 -m ground_target_tracking.main

# Explicit local video (of_kalman is the default method)
python3 -m ground_target_tracking.main --video videos/v8.mp4 --method of_kalman

# A network stream also works (http/https/rtsp/rtmp), handed straight to OpenCV
python3 -m ground_target_tracking.main --video https://example.com/clip.mp4

# Manual point entry — assignment convention [i],[j] = (row, col):
python3 -m ground_target_tracking.main --video videos/v8.mp4 --point-rc 532,970
#   ...or as pixel coordinates x,y (equivalent to the line above):
python3 -m ground_target_tracking.main --video videos/v8.mp4 --point 970,532

# Reacquisition is ON by default; disable it for a run with --no-reacq
python3 -m ground_target_tracking.main --video videos/v8.mp4 --no-reacq

# Headless run with per-run logging (logs/run_NNN/: config, CSV, output.mp4, stats)
python3 -m ground_target_tracking.main --video videos/v8.mp4 --no-display --point 970,532 --save
```

In the selection window: **click** the target pixel → **Enter/Space** to confirm →
**r** to re-pick → **q/Esc** to quit. The clip then plays with the marker, moving
ROI box, and trajectory overlaid; the HUD shows the tracking state and FPS.

The delivered default profile is `--method of_kalman` with reacquisition, the
burned-in-HUD/OSD ignore mask, and regional local-motion estimation all **on**;
each has an opt-out flag. Key flags: `--method {of,of_kalman,fixed}` (default
`of_kalman`) · `--point x,y` **or** `--point-rc i,j` (row,col) · `--init-frame N` ·
`--patch-size N` · `--no-display` · `--save` · `--no-reacq` · `--no-overlay-mask` ·
`--no-regional-motion` · diagnostics `--show-preprocess`, `--show-keypoints`,
`--show-measurement`. (`--reacq` / `--overlay-mask` remain as compatibility
no-ops now that both are on by default.)

**Expected behavior by state (HUD):**
- `TRACKING` — confident lock; box follows the object.
- `LOW_CONFIDENCE` — still measuring, but degraded (appearance mismatch, frame
  edge, or a re-seeded tracker on probation after reacquisition).
- `PREDICT` — brief Kalman coast through a measurement dropout.
- `LOST` — tracking is no longer trustworthy; the estimate freezes and the box
  turns red. By default the program then searches for the object and re-locks
  when it returns (`REACQ_INIT` events are printed).
- `FEED FROZEN` — the **input feed itself** is frozen/dead (repeated identical
  frames), distinct from `LOST`: the last position is held ("FEED FROZEN —
  holding last position"), evidence and reacquisition search are suspended, and
  on the feed resuming the tracker re-validates from `LOW_CONFIDENCE`.

Run the test suite (stdlib `unittest`, no third-party test dependencies):

```bash
python3 -m unittest discover tests
```

**Grader note:** with the official sample at `videos/v8.mp4` present the full suite
passes; on a fresh clone **without** that file the 8 sample-video acceptance tests
`skipUnless` the file exists — they *skip*, they do not fail — so the rest of the
suite still passes green. Exact counts are in
[Validation & Evidence](#validation--evidence).

---

## How it works

```
Video → point selection → ROI extraction → preprocessing (gray/CLAHE/blur)
      → Lucas–Kanade optical flow (corner seeding, forward–backward filtering)
      → Kalman filtering & short-term prediction
      → TrackingSession control layer: confidence scoring + state machine
        (TRACKING / LOW_CONFIDENCE / PREDICT / LOST)
      → while LOST (reacq enabled): ORB feature matching against an immutable
        init-time reference, interleaved with a bounded template scan;
        candidate verification → re-seed → probation → TRACKING
      → visualization (marker, trajectory, ROI box, HUD)
```

- **Loss detection (M8):** combined confidence = tracker quality (flow survivors +
  forward–backward error) × patch similarity (NCC vs a conservatively adaptive
  reference) × edge penalty, with a jump veto. Sustained bad evidence → `LOST`;
  the estimate freezes instead of pretending. The adaptive reference updates only
  through a strict gate keyed to **positional trust**, so legitimate appearance
  change is followed while degraded content can never poison the reference.
- **Dead-feed detection (`FEED_FROZEN`):** a separate, orthogonal check on the raw
  input feed — a block-max frame-difference detector (max over 16×16 blocks at
  480×270, HUD-masked) flags a **frozen/repeated** feed (`< T_static` for 3
  consecutive frames) as a *distinct* condition, not a false `LOST`/`TRACKING`
  claim on a non-changing image. While frozen the point is held and all evidence
  advancement (scoring, streaks, reference, reacquisition search) is suspended; on
  the feed resuming for 2 frames the Kalman is zeroed/re-seeded at the held point
  and the state returns to `LOW_CONFIDENCE` for revalidation. `T_static = 1.5` was
  calibrated (validated safe interval (0.71, 1.91)) on the official sample, the
  external demo clip, and synthetic ≥30 fps fixtures.
- **Reacquisition (M9):** an **immutable** identity reference (ORB descriptors
  near the selected point + a grayscale context template) is built once at
  selection time. While LOST, feature matching proposes candidates
  (rotation/scale tolerant); when features have nothing to offer, a **bounded**
  incremental template scan covers the frame across scales (never the full-frame
  sweep). Every candidate must pass identity verification against the immutable
  reference, persist across evaluations, and survive a probation period before
  the state returns to `TRACKING`.
- **Last-confident SIFT route:** while tracking is trustworthy the session
  keeps one cheap frame snapshot; at loss it builds a SIFT reference from the
  clean context around the tracked point (HUD pixels excluded per keypoint). To
  stay inside the real-time budget the work is **stripe-sliced**: the reference
  builds one full-resolution stripe per recovery tick, and the query then sweeps
  one stripe per tick (~12–18 ms each at 0.8 detect scale) instead of a
  110–350 ms whole-frame match — rotation- and scale-invariant, so it recovers
  targets that return from the **opposite side of an orbit** (~150–180° viewpoint
  change), where templates and ORB are measurably blind. When the persistence
  gate is first ready to accept, a single current-frame localization fixes the
  re-seed point. Identity rests on RANSAC geometry (inlier floor calibrated
  against a full absence audit) plus persistence and probation; on probation
  success the working appearance reference is re-anchored to the validated view.
  (SIFT ships in mainline opencv-python ≥ 4.4 — no extra dependency.)
- All tunable parameters live in **`ground_target_tracking/config.py`** — the
  single source of truth; nothing is hardcoded elsewhere.

### Reacquisition status (honest)

Reacquisition is **enabled by default** (`REACQ_DECISION_ENABLED = True`); disable
it per run with `--no-reacq`. The enable decision is backed by the 9-pick
official-sample acceptance grid (loss → SIFT last-confident reacquisition →
sustained re-lock, with zero acceptances during the target's absence window), a
synthetic 1080p exit/return benchmark that reacquires to ground truth, and a
measured false-lock floor (junk candidates 57–113 px off-target, all rejected).
Known limitation: a **near-copy decoy** (an object nearly pixel-identical to the
target) can defeat the identity gates — the acceptance thresholds refuse plainly
wrong objects, but cannot separate a near-perfect copy.

---

## Repository structure

```
.
├── README.md            # this file (GitHub-rendered project page)
├── SUMMARY.md           # written summary — the assignment's six questions
├── ARCHITECTURE.md      # stage-by-stage pipeline description
├── docs/                # system block diagram + tracking state-machine (Mermaid)
├── demos/               # short screen-recording demos (reference + external)
├── videos/              # place input videos here locally (gitignored, not committed)
├── tests/               # unit + benchmark suites (python3 -m unittest discover tests)
└── ground_target_tracking/
    ├── config.py        # ALL tunable parameters (single source of truth)
    ├── main.py          # CLI + frame loop + visualization
    ├── session.py       # confidence + state machine + reacquisition control (M8/M9)
    ├── reacquisition.py # immutable reference, matching, verification, scan scheduler
    ├── trackers.py      # Tracker interface: fixed / optical-flow / Kalman
    ├── preprocessing.py # per-method preprocessing pipelines
    ├── utils.py         # data types, video I/O, ROI, drawing, overlay masks
    ├── evaluation.py    # PerfStats (FPS / ms-per-frame / runtime)
    ├── experiment.py    # RunLogger (per-run logs/run_NNN/)
    └── requirements.txt
```

Input videos (including the official sample) live in the gitignored `videos/`
folder and are **not committed** — place your own clip there or pass one with
`--video`. The only committed media are the short screen recordings in `demos/`.

---

## Known Limitations

The tracker is deliberately **conservative** — it holds honestly rather than
guessing. At a glance:

- Works best on **large, textured, distinctive** targets.
- **Low-texture animals** may not provide enough keypoints for reliable reacquisition.
- **Small / distant / fast** targets can fail at the initial tracking stage.
- **Large viewpoint or scale changes** after the target returns can prevent reacquisition.
- **Unstable / handheld camera** motion makes tracking harder.
- `FEED_FROZEN` can **false-trigger on a live but visually static scene**.
- The system **prefers `LOST`** over falsely locking onto the wrong object.

In more detail:

- **Video input:** accepts a local file path **or** an http/https/rtsp/rtmp URL
  (handed to OpenCV's `VideoCapture`); URL/stream playback depends on OpenCV's
  ffmpeg backend, the network, and the codec, so **local files are the primary
  tested path**.
- **Fixed ROI size** — no scale/rotation adaptation of the box; large scale
  changes degrade tracking and displace the box. An extreme-scale return can
  drift: a candidate localization reproduced a target's *pre-loss drifted* locus
  rather than its true on-body point, and an independent oracle could not prove an
  on-body re-lock, so a proposed larger-scale acceptance envelope was **rejected**
  rather than redefining ground truth around the observed fit.
- **Low-texture targets** are weaker on every axis: optical flow falls back to a
  seed grid, and reacquisition must rely on the (bounded) template scan because
  ORB finds too few keypoints.
- **Near-copy decoys** can defeat reacquisition identity checks (see above).
- **`FEED_FROZEN` vs a motionless live scene.** A dead/repeated feed is detected
  by frame differencing; a genuinely **motionless** live scene (nothing in frame
  moving, only sensor noise) is indistinguishable from a frozen feed by that
  method and will also be held. This is inherent to frame-difference freeze
  detection and harmless — holding the point is correct when nothing, including
  the target, is moving.
- **Corruption-frame discount (tested, rejected).** An adaptive optical-flow
  discount to make corrupt frames inert was implemented and evaluated, then
  **removed from the shipped default**: on the sample video it discounted
  *sustained live motion* during the target's exit (the rolling norm froze low),
  delaying `LOST`/reacquisition. The shipped pipeline keeps `FEED_FROZEN`
  detection only; the existing one-frame **jump veto** (`MAX_JUMP_PX`) remains the
  backstop against corrupt-frame measurement jumps.
- **Selections centered on burned-in HUD graphics** (e.g. the crosshair of the
  official sample video) carry no target appearance at all. The coordinate is
  still accepted — tracking uses the surrounding clean scene and HUD pixels
  stay excluded from all evidence — and the program warns and continues; it will
  refuse to fabricate a re-lock (measured on the official sample: best template
  response anywhere in the frame was 0.42 against the 0.60 acceptance gate).
- **Validation basis:** the official sample video + controlled synthetic
  scenes + one external demo clip; broad arbitrary-video robustness is
  demonstrated on a small demo set, not a large corpus.
- **Real-time caveat (measured, display excluded).** Reported as separate
  quantities, never conflated: (a) **processing throughput** — full decode+step
  per frame at native 1920×1080; on the shipped default (reacquisition on) the
  official sample runs at **≈4.7 ms/frame mean** with a single 150 ms
  accept-time re-localization spike (once per reacquisition) and the search phase
  ≈24 ms — every rolling 1-second window stays ≥30 processed frames; the
  `FEED_FROZEN` phase costs ≈0.3 ms/frame. (b) **Source frame rate** is reported
  per video. (c) Displayed playback is deliberately paced to the source period
  (informational). The ≥30 fps input-rate requirement is evidenced on the sample
  video (59.82 fps source) and synthetic ≥30 fps fixtures.

---

## Validation & Evidence

**Automated tests** (stdlib `unittest`, no third-party deps): **279 tests**. The
suite is fully self-contained except for the 8 official-sample acceptance tests,
which `skipUnless` `videos/v8.mp4` is present:

- with the sample present: **279 pass** (and `V8_FULL_GRID=1 python3 -m unittest
  tests.test_v8_acceptance` runs the full 9-pick acceptance grid green);
- on a fresh clone without it: **271 pass, 8 skip, 0 fail** — grader-runnable as-is.

The synthetic reacquisition benchmark asserts the complete loss → search → single
`REACQ_INIT` → probation → `TRACKING` contract against analytic ground truth, with
no video file.

**Evidence by source (honest):**

- **Official sample (v8) — strong PASS.** Validated track → lose → reacquire loop;
  9-pick acceptance grid; measured ≈4.7 ms/frame processing at 1920×1080 (well above
  30 fps). This is the primary demo (`demos/official_reference_demo.mp4`).
- **External human clip — PASS-with-caveat (functional).** On an unseen 1080p clip
  the tracker loses the subject as she leaves frame and **reacquires the same person
  on return** (SIFT last-confident, probation-confirmed), reproducibly, at real-time
  processing speed. Caveat: the source is 29.13 fps VFR, so it is **functional
  external evidence, not strict ≥30 fps input-rate proof**, and a `FEED_FROZEN`
  false-fire occurs on the empty static room. Shown in
  `demos/external_reacquisition_demo.mp4`.
- **Additional unseen clips — limitation evidence.** A dog clip reacquired only a
  transient in-frame dropout (not an off-frame return at a different scale/viewpoint);
  low-texture animals (e.g. a light-furred squirrel, a small distant surfer) produced
  too few keypoints to reacquire. These delineate the honest boundary above.

**Real-time note:** the ≈4.7 ms/frame figure is *measured*, not asserted by an
automated gate — an automated FPS gate is listed under [Future Work](#future-work).

See [`docs/system_block_diagram.md`](docs/system_block_diagram.md) for the pipeline
and [`docs/tracking_state_machine.md`](docs/tracking_state_machine.md) for the HUD
states.

### Engineering validation notes

**Runtime / performance (measured, CPU-only).** End-to-end per-frame processing was
timed on the development machine:

- Official **V8** reference video (1920×1080): **≈4.7 ms/frame**.
- External **Katyatest** clip (1080p): **≈17.1 ms/frame**.

At 30 FPS the real-time budget is **≈33.3 ms/frame** (1000 ÷ 30), so both figures sit
comfortably under it — i.e. faster than the required real-time threshold. The system is
**CPU-only** and needs no GPU or special hardware. Only **end-to-end per-frame** runtime
was measured; individual stages (optical flow, Kalman, SIFT/reacquisition, HUD, etc.)
were **not separately profiled**, so no per-stage timings are claimed.

**Accuracy / error tolerance.** Quantitative pixel-error validation exists only for the
**synthetic** tests, where ground-truth target positions are known analytically; there,
reacquisition recovery is validated within a **5-pixel tolerance**. This ±5 px bound is
specific to those synthetic cases and is **not** claimed as a universal error bound for
arbitrary videos.

**Real-video evaluation.** For real footage (the official reference video and external
clips), frame-level ground-truth annotations were **not available**. Real-video
evaluation therefore rests on **visual inspection, run logs, HUD state transitions, and
observed reacquisition success/failure** rather than on measured pixel error.

**Determinism / repeatability.** The full test suite and the V8 validation were run more
than once, with **consistent passing results** across runs.

**Input-path caveat.** **Local video files are the primary tested input path.** Direct
URL/stream passthrough is supported for **http/https/rtsp/rtmp** inputs where the
installed OpenCV/ffmpeg build, network, and codec allow it. Browser/player links (e.g.
YouTube or private Google Drive URLs) are **not guaranteed**.

## Future Work

The main directions for future improvement are:

- a **larger, more diverse external validation set**;
- a **hybrid color/HSV cue** for low-texture but color-distinct targets;
- **scale/viewpoint-robust reacquisition** (multi-scale / affine-tolerant matching);
- **better static-scene vs frozen-feed discrimination** (fewer `FEED_FROZEN` false-fires);
- **stronger automated FPS gates** in the test suite;
- an optional **segmentation / detector cue** for hard targets.

---

## Assignment Mapping

Legend:  
✅ IMPLEMENTED = requirement is implemented and validated.  
🟡 PARTIAL = implemented or tested, but with documented limitations or limited validation scope.  
❌ MISSING / PENDING = not implemented or not completed.

| # | Requirement | Current implementation | Status |
|---|-------------|------------------------|--------|
| A1 | Input: video link + pixel on first frame (mouse or manual `[i],[j]`) | Video by local **path or** http/rtsp **URL** ✅ (`--video`, menu/picker); pixel by mouse ✅, `--point x,y` ✅, manual `[i],[j]` row/col ✅ (`--point-rc`). URL playback depends on OpenCV's backend/network (local is the primary tested path) | ✅ IMPLEMENTED |
| A2 | Track the object selected by the pixel | Lucas–Kanade optical flow (+ optional Kalman) follows the pixel/patch | ✅ IMPLEMENTED |
| A3 | Display pixel + motion + tracking box; simple UI | Marker + trajectory + moving ROI box + state HUD; mouse & manual selection | ✅ IMPLEMENTED |
| A4 | Handle loss + reacquire on return | Loss detection ✅ (M8) + dead-feed `FEED_FROZEN` detection ✅. Reacquisition ✅ **on by default** (SIFT last-confident + feature + bounded template routes; `--no-reacq` to disable) — validated on the 9-pick sample grid + synthetic exit/return. Documented limitation: near-copy decoys | ✅ IMPLEMENTED |
| A5 | Real-time ≥30 fps @1920×1080, no GPU | Shipped default (reacquisition on) processing @1080p: **≈4.7 ms/frame mean** on the sample video, search phase ≈24 ms, one 150 ms accept-time re-localization per reacquisition, `FEED_FROZEN` phase ≈0.3 ms; every rolling 1 s window ≥30 processed frames. ≥30 fps input evidenced on the 59.82 fps sample + synthetic ≥30 fps fixtures | ✅ IMPLEMENTED |
| A6 | Works on any clear object, no restriction | General mechanism (LK on any textured patch); strongest on large / textured / viewpoint-stable targets. Validated on the sample + synthetic scenes + an external human clip; low-texture animals are a documented limitation | 🟡 PARTIAL |
| A7 | Tested on separate videos | Tested on unseen clips: one external human clip reacquires (PASS-with-caveat); additional dog / low-texture clips are limitation evidence — see [Validation & Evidence](#validation--evidence) | 🟡 PARTIAL |
| A8 | Python; external libraries allowed | Python 3.9.6 + OpenCV 4.13.0 + NumPy 2.0.2 | ✅ IMPLEMENTED |
| A9 | Public GitHub with all files | Published on **GitHub** with the clean committed project files | ✅ IMPLEMENTED |
| A10 | README (GitHub) with app details + versions | This root README | ✅ IMPLEMENTED |
| A11 | Short demo video | Two committed demo screen recordings: [`demos/official_reference_demo.mp4`](demos/official_reference_demo.mp4) (reference v8) and [`demos/external_reacquisition_demo.mp4`](demos/external_reacquisition_demo.mp4) (external unseen human clip) | ✅ IMPLEMENTED |
| A12 | Written summary (6 questions) | [`SUMMARY.md`](SUMMARY.md) contains the final written answers to all six required questions | ✅ IMPLEMENTED |
