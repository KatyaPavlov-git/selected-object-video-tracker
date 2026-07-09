"""ground_target_tracking — classical-CV ground-target tracking pipeline.

Built incrementally. Implemented so far: Milestones 1–7 (video loading + smart
selection, ground-point selection, border-safe ROI/patch, preprocessing pipelines,
ORB/AKAZE feature extraction, feature visualization, Lucas-Kanade optical-flow
tracking, and Kalman smoothing/short-term prediction) plus experiment-logging and
performance-metrics scaffolding. Methods: `of` (M6), `of_kalman` (M7, optical flow +
Kalman), `fixed` (constant-point baseline). Loss detection / reacquisition (M8/M9)
are not yet implemented — Kalman only coasts for a few frames, it does not reacquire.
"""
