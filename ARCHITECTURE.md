# Architecture

Stage-by-stage description of the **actual** pipeline for the Asio real-time
pixel/object tracking assignment. Each stage lists its responsibility, input,
output, the module/class that implements it, and its status:

✅ Implemented = implemented in the submitted system.  
🟡 Partial = implemented/tested with documented limitations or limited validation scope.  
🔭 Planned = future work, not part of the submitted implementation.  
⛔ Retired = older design path intentionally removed or not used in the submitted default.

The system is a per-frame **data path** commanded by a **control layer** (the
tracking session). Confidence, lost-state and reacquisition (on by default) are not
stages the data flows through — they decide *which* path runs each frame and
whether its output is trusted:

```
CONTROL LAYER (session.py — TrackingSession)
  Tracking Confidence + State Machine:  TRACKING / LOW_CONFIDENCE / PREDICT / LOST
  commits or freezes the estimate · owns the adaptive reference patch
  [Reacquisition — implemented (M9), ON by default; runs while LOST, --no-reacq to disable]
        │  commands / trusts / freezes
        ▼
DATA PATH (per frame)
  Video Input → Target Selection → ROI Extraction → Preprocessing
      → Motion Tracking (Lucas–Kanade) → Motion Filtering & Short-Term
      Prediction (Kalman) → Visualization
                                                     Evaluation — perf metrics + tests/synthetic/V8 validation
```

Cross-cutting: **Configuration** (`config.py`), **Experiment Logging**
(`experiment.py`), **Performance Metrics** (`evaluation.py`).

---

## Pipeline stages

| # | Stage | Responsibility | Input | Output | Module · Class/Fn | Status |
|---|-------|----------------|-------|--------|-------------------|--------|
| 1 | **Video Input** | Open + validate a source; iterate frames | video path or URL | BGR frames + `VideoMeta` | `utils` · `open_video`, `read_frames`, `list_videos`, `pick_video_via_dialog` | ✅ (local + `http`/`https`/`rtsp`/`rtmp` via OpenCV; local is the primary tested path) |
| 2 | **Target Selection** | Choose the pixel to track on the first frame | init frame; mouse or `--point x,y` | `Point2D` | `main` · `select_point_interactive`, `resolve_target_point` | ✅ |
| 3 | **ROI Extraction** | Border-safe tracking box around the pixel | frame + `Point2D` | `Patch` (image, bbox, size) | `utils` · `clamp_roi`, `extract_patch` | ✅ |
| 4 | **Preprocessing** | Per-method frame/patch prep (gray / CLAHE / blur) | BGR frame/patch | processed gray | `preprocessing` · `get_pipeline` (`of` / `orb` / `template`) | ✅ |
| 5 | **Feature Extraction** | ORB/AKAZE keypoints+descriptors (diagnostic now; reserved for reacquisition) | patch (later: full frame) | keypoints, descriptors | `trackers` · `build_detector`, `detect_features`, `feature_report` | ✅ (as diagnostic) |
| 6 | **Motion Tracking** | Follow the point frame-to-frame (Lucas–Kanade optical flow) | frames + init point | per-frame `TrackResult` (moving point) | `trackers` · `OpticalFlowTracker` | ✅ |
| 7 | **Motion Filtering & Short-Term Prediction** | Smooth jitter; coast a few frames on prediction (constant-velocity Kalman) | `TrackResult` | filtered `TrackResult` | `trackers` · `KalmanWrapper` | ✅ |
| 8 | **Tracking Confidence** | Score reliability: tracker confidence (survivors + FB error, coast decay) × patch similarity (scale-tolerant NCC vs adaptive reference) × edge penalty; jump veto | frame + `TrackResult` | combined confidence + signals | `session` · `TrackingSession` (+ `trackers` per-tracker confidence) | ✅ (M8) |
| 9 | **Lost-State Handling** | Explicit state machine with hysteresis: `TRACKING / LOW_CONFIDENCE / PREDICT / LOST`; freeze estimate on LOST; never claim TRACKING for a lost target | confidence + signals | `SessionResult` (state, committed point) | `session` · `TrackingSession`, `TrackState` | ✅ (M8) — first half of **A4** |
| 10 | **Reacquisition** | Recover the target when it re-enters: ORB feature match on the full frame (+ template fallback), decision layer confirms identity, session re-seeds while LOST | frame + immutable reference | recovered point; re-seed flow | `reacquisition` · `Reacquirer`, `HypothesisTracker`; `session` integration | ✅ implemented and **on by default** (`--no-reacq` to disable); validated |
| 11 | **Visualization** | Draw marker, moving ROI box, trajectory, HUD; simple UI | frame + state | annotated frame / window | `utils` · `draw_point`, `draw_roi`, `draw_trajectory`, `draw_hud`; `main` loop | ✅ |
| 12 | **Evaluation** | Measure generality/robustness on separate videos; compare methods | run logs | metrics / tables | `evaluation` | 🟡 (perf metrics ✅; automated tests + synthetic + V8 validation ✅; broad accuracy/comparison across arbitrary videos limited) |

---

## Stage notes

- **Video Input.** `open_video` accepts **local file paths** and direct `http://`,
  `https://`, `rtsp://`, and `rtmp://` URLs (handed to OpenCV). URL playback depends
  on the installed OpenCV/ffmpeg backend, network, and codec; local files are the
  primary tested path. Selection falls back through: `--video` → single clip in
  `videos/` → numbered menu → native file picker.
- **Target Selection.** Mouse click on the first frame (mapped back from the
  downscaled display to full-resolution coordinates) or headless `--point x,y`
  (`--init-frame` chooses the frame). `--point-rc i,j` is implemented and maps the
  assignment's row/col `[i,j]` index convention to internal `x,y`.
- **ROI Extraction.** Square patch (default 51 px, `--patch-size`), clipped safely at
  borders. This is the "tracking box" the assignment asks to display; its size is an
  implementer choice.
- **Preprocessing.** "Preprocessing is part of the tracking method": optical flow
  gets grayscale (+ optional blur/CLAHE); ORB/AKAZE get grayscale + CLAHE. No
  HSV/color segmentation (self-imposed).
- **Feature Extraction.** Implemented and visualized, but currently **diagnostic**:
  ORB yields ~0 keypoints on a 51-px patch (its 31-px edge margin exceeds the patch),
  which is exactly why **Motion Tracking uses optical-flow corners, not ORB**.
  ORB/AKAZE descriptors of the initial patch are the basis for **Reacquisition**,
  which must run detection on the **full frame**.
- **Motion Tracking (Lucas–Kanade).** Seeds `goodFeaturesToTrack` corners in the ROI
  (grid-seed fallback for low-texture patches), tracks them with pyramidal LK, keeps
  only forward-backward-consistent survivors, and moves the point by their **median
  translation**; re-seeds when survivors deplete. Reliable measurements advance the
  committed estimate; unreliable ones do not (the display coasts).
- **Motion Filtering (Kalman).** A constant-velocity `KalmanFilter(4,2)` decorator:
  predict → correct with the measurement when reliable, else coast on prediction for
  up to `KALMAN_MAX_PREDICT_FRAMES`. **This is smoothing + very short gap-bridging —
  it is not loss detection and not reacquisition.**
- **Tracking Confidence + Lost-State (M8, implemented).** `TrackingSession`
  (`session.py`) drives whichever tracker `make_tracker` built and owns the state.
  Confidence = tracker confidence (survivor count + FB error; decayed while the
  Kalman coasts) × patch similarity (NCC of the current ROI against a
  **conservatively adaptive** reference patch — re-snapshotted only on strong
  TRACKING frames through a strict gate, so junk can never become the reference)
  × an edge penalty near the frame border, with a one-frame jump veto. States:
  `TRACKING / LOW_CONFIDENCE / PREDICT / LOST` with consecutive-frame hysteresis.
  **On LOST the estimate freezes** at the last credible point (M9 reacquisition,
  on by default, can re-seed it while LOST) and
  the system stops claiming success (red box + LOST banner). Border proximity is
  treated as risk (LOW_CONFIDENCE), not loss evidence — an object grazing the
  frame edge survives as LOW_CONFIDENCE; LOST requires the loss *evidence*
  (tracker confidence × aligned similarity) to stay collapsed for a sustained
  streak. Validated on the official sample video: the edge-graze-and-return
  episode reads LOW_CONFIDENCE, and LOST fires only once the track is genuinely
  stuck on non-target content. Unit tests: `tests/test_session.py`.
- **Reacquisition (implemented M9, on by default).** The second half of **A4**:
  search while LOST and re-seed the tracker when the object returns (ORB feature
  proposer + decision layer + probation). Built + validated (M9-d); on by default,
  `--no-reacq` disables it per run (`--reacq` is retained only as a legacy
  compatibility no-op). Documented limitations: near-copy decoys and the
  template-fallback real-time gap.
- **Visualization.** Marker + moving ROI + trajectory polyline + HUD (method, points,
  status, live FPS). Deliberately minimal, per the assignment.
- **Evaluation.** `PerfStats` (FPS, ms/frame, runtime) runs every session; validation
  comes from the automated test suite, synthetic ground-truth scenes, the V8 sample
  checks, external demo evidence, and runtime measurements. Broad accuracy/robustness
  comparison across arbitrary videos remains limited (assignment A6/A7 scope).

---

## Method selection

`make_tracker(method)` builds one of:

- **`fixed`** — holds the initial pixel constant. Pre-tracking baseline / regression
  guard; not intended behavior.
- **`of`** — Lucas–Kanade optical-flow tracking (Motion Tracking).
- **`of_kalman`** — `of` wrapped by the Kalman filter (Motion Tracking + Motion
  Filtering).

All three share the `Tracker` interface (`init` / `update -> TrackResult`) and are
driven through the same `TrackingSession` control layer (M8), which adds
confidence + state on top of any tracker without changing the frame loop.

---

## Real-time posture

The assignment requires **≥30 fps at 1920×1080 on CPU**. Measured processing
throughput on a synthetic 1080p scene is **~283 fps (~3.5 ms/frame)** for both `of`
and `of_kalman` — comfortably within budget. Caveat: this measures the tracker step
only (not decode/display) and does not stress worst-case re-seeding; the tracker
also computes over the **full frame** each step (cost scales with frame size, not
ROI), so a windowed search is the natural scalability improvement if larger inputs
or heavier per-frame work are introduced.
