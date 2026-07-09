# Written Summary — Tracking and Reacquisition in Real-Time Video Processing

## 1. Steps taken to complete the project

I started by mapping the assignment requirements into a clear tracking pipeline: video input, target-pixel selection on the first frame, creation of a region of interest around the selected pixel, object tracking, and visual display of the tracking result.

I then implemented the input and selection flow. The program can load a video, allow the user to select the target by mouse click, or receive manual pixel coordinates using the assignment’s [i, j] convention. Around the selected pixel, the system creates a configurable tracking window.

After that, I implemented the main tracking flow. The system estimates the target position across frames and displays the selected point, tracking region, motion trajectory, and current tracking state on the video.

The next stage was improving reliability. I added confidence-based tracking states and prediction logic so that the system can distinguish between confident tracking, uncertain tracking, prediction, and loss. This prevents the tracker from presenting an unreliable position as a successful track.

I then added reacquisition logic for cases where the target is lost and later returns to the frame. The goal was not only to recover the target, but to verify it conservatively before returning to a normal tracking state.

Finally, I validated the system on the official reference video, synthetic test cases, and separate external videos. This helped me confirm the cases where the system works well and identify limitations such as low-texture targets, large appearance changes, and scenes that can look like a frozen video feed.

## 2. Guiding principles during development

The main guiding principle was to build a reliable selected-object tracker rather than a solution that only works on the reference video. I used the reference video to understand the types of challenges the system should handle, such as target disappearance, camera motion, temporary loss, and return to the frame. However, I tried to avoid overfitting the implementation to that specific video or to a single target appearance.

Another important principle was conservative tracking. When the evidence is weak, the system should not continue presenting an uncertain location as a successful track. It is better to move into LOW_CONFIDENCE, PREDICT, or LOST than to lock onto the wrong object. This principle guided the confidence checks, loss handling, and reacquisition logic.

I also wanted the system state to be clear to the user. The display should not only show a box, but also communicate whether the system is tracking confidently, predicting, lost, or detecting a frozen feed. For this reason, the HUD and visual overlays became part of the reliability of the system, not just cosmetic additions.

Finally, I kept the solution practical and evidence-based. I wanted the system to run without special hardware, remain CPU-only, and make claims only when they were supported by tests or observed behavior. I used the reference video, synthetic tests, external videos, logs, and failure cases to decide whether a behavior was reliable, limited, or should be described as future work.

## 3. Algorithms researched and conclusions

I researched several tracking and recovery approaches and combined the ones that best matched the assignment constraints.

For frame-to-frame tracking, I used Lucas–Kanade optical flow. This approach is suitable for following local motion between consecutive frames and works well when the selected target has enough visual texture and does not change appearance too abruptly. The conclusion was that optical flow is a good base tracker for continuous motion, but it should not be trusted blindly when confidence becomes low.

To make the motion estimate more stable, I used a Kalman filter. The Kalman filter helps smooth the target position and provides short-term prediction when the direct measurement becomes unreliable. However, I treated prediction as a temporary support mechanism rather than a replacement for real visual evidence.

For reacquisition after target loss, I used feature matching based on the last confident appearance of the target. The goal was to find the same object again when it returns to the frame. The conclusion was that feature-based reacquisition can work well for textured and distinctive targets, but it is limited when the target has little texture, changes scale or viewpoint significantly, or returns after a long visual change.

I also used confidence-based state logic to decide when the tracker should remain in TRACKING, move to LOW_CONFIDENCE or PREDICT, or declare LOST. This became an important part of the solution because the system must know when not to trust its own estimate.

I considered color-based cues such as HSV or color histograms as an additional direction. The conclusion was that they may help with low-texture but color-distinct targets, but adding them before submission would introduce new risks, especially false matches with similarly colored background regions. Therefore, I documented this direction as future work rather than adding it without enough validation.

## 4. Most significant challenge and how I overcame it

The most significant challenge was handling target loss and reacquisition without creating false confidence. Tracking the object while it remains clearly visible is only one part of the problem. The harder part is deciding what the system should do when the target leaves the frame, becomes unreliable, or returns with a slightly different appearance.

At first, a tracker can appear successful simply because it continues drawing a box, but that box may no longer represent the selected object. To address this, I focused on separating confident tracking from uncertain tracking. I added confidence-based states such as LOW_CONFIDENCE, PREDICT, and LOST, so the system can explicitly communicate when the current estimate is no longer reliable.

I also added a conservative reacquisition process. When the target is lost, the system searches for evidence that the same target has returned, but it does not immediately return to normal tracking. It first verifies the candidate before switching back to TRACKING. This helped reduce the risk of locking onto the wrong object.

A related challenge was avoiding overfitting to the reference video. I used the reference video to understand the expected difficulties, such as target disappearance, camera motion, and return to the frame, but I did not want the solution to rely on assumptions that only fit that specific video. To address this, I tested the tracker on synthetic cases and separate external videos, and used both successful and failed cases to refine the logic and understand where the system generalizes.

Another part of overcoming the challenge was improving the visual feedback. The HUD, tracking box, target marker, trajectory, and loss/frozen-feed labels helped make the system behavior easier to inspect and debug. Instead of only looking at the final output, I could understand why the tracker was confident, uncertain, lost, or recovering.

Together, these steps made the final system more honest and robust: it can recover in supported scenarios, but it also clearly reports uncertainty or loss when the evidence is not strong enough.

## 5. Situations where the software is limited or fails

The system is not intended to be a universal tracker for every possible object and scene. It works best when the selected target is reasonably large in the frame, visually distinctive, and contains enough texture for local tracking and reacquisition. It is more limited when the visual evidence around the selected pixel is weak or changes significantly over time.

One important limitation is low-texture targets. When the selected area contains very few stable visual features, the tracker has less information to follow and the reacquisition stage has fewer reliable matches to use. This was visible in external tests with low-texture animals, such as the dog and squirrel clips: even when the object was visible to a human viewer, the selected patch did not always provide enough feature information for robust reacquisition.

The system is also limited with very small, distant, or fast-moving targets. If the target occupies only a small number of pixels, or moves quickly relative to the frame rate, the tracking window may not contain enough stable information from frame to frame. This makes both optical-flow tracking and later reacquisition less reliable.

Another limitation is large appearance change after the target is lost. The reacquisition logic was designed to handle moderate changes in position, scale, and appearance, such as cases where the target leaves the frame and later returns. However, reacquisition is still based on comparing the returning object to the last confident appearance of the selected target. If the target returns with a very large change in scale, viewpoint, pose, lighting, or visible texture, the evidence may not be strong enough to safely confirm that it is the same object. In those cases, the system may remain in LOST instead of risking a false reacquisition. This was an intentional trade-off: I preferred a conservative missed reacquisition over incorrectly locking onto a different object.

Camera motion is another difficult case. The system can handle moderate camera motion, but unstable or abrupt camera movement makes it harder to separate target motion from global scene motion. In such cases, the tracker may enter LOW_CONFIDENCE, PREDICT, or LOST rather than risk reporting an incorrect target position.

The FEED_FROZEN detector also has a limitation. It is useful for detecting frames that appear unchanged, but a live scene can sometimes be genuinely static. In those cases, the system may temporarily report FEED_FROZEN even though the camera feed is not actually frozen. I treated this as a known limitation and documented it instead of claiming perfect frozen-feed detection.

Overall, the system deliberately prefers conservative failure over false success. In difficult cases, it may report LOW_CONFIDENCE or LOST instead of continuing to draw a confident box on the wrong object. This means the software does not solve every tracking scenario, but its limitations are visible and explainable rather than hidden.

## 6. My personal experience

During this project, I learned how important it is to turn an open-ended video-processing and object-tracking task into a clear and testable pipeline. At the beginning, the main challenge for me was not only implementing tracking, but understanding exactly what the system should prove: selecting a target from the first frame, following it over time, handling uncertainty, and explaining its behavior clearly to the user.

One of the most important things I learned was that a tracker can look correct visually while still being unreliable. A box on the screen is not enough if the system is no longer tracking the selected object. This changed the way I approached the project: instead of only trying to make the output look successful, I focused more on confidence, failure cases, and deciding when the system should stop trusting itself.

I also learned a lot about myself during this process. This was not a topic I had worked with deeply before, and there were moments of confusion and frustration. However, I learned that I can enter an unfamiliar technical area, stay with the uncertainty, break the problem into smaller parts, and gradually turn it into a working and validated solution.

Testing the system on videos outside the reference case was also an important part of the experience. Some tests confirmed that the tracker can recover well in supported scenarios, while others exposed real limitations. This was sometimes frustrating, but it helped me understand the system more deeply and document its behavior honestly.

Overall, the project improved both my technical confidence and my development process. I became better at debugging a video-processing and object-tracking pipeline using evidence rather than assumptions, and I learned the importance of balancing ambition with reliability: I preferred to build a conservative system that clearly reports uncertainty over a system that appears more successful but may lock onto the wrong object.
