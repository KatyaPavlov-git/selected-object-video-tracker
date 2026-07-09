"""v8 acceptance — HUD-covered selections on the official sample video.

Contract (2026-07-07 nearby-selection investigation, user-approved; the
mandatory verified contract applies to the SIX explicitly measured
selections — additional nearby points would be exploratory coverage, not a
proven guarantee):

  Per selection, one deterministic full-video run of the real pipeline in
  the Run-A profile (overlay mask + regional motion + reacquisition):
  1. Tracking degrades to LOST (read window asserted loosely — loss timing
     is inherent M8 margin behavior and deliberately unconstrained beyond
     sanity bounds).
  2. EXACTLY ONE episode-1 `REACQ_INIT`, cue `sift-lc`, within reads
     140-320 (the house re-enters the frame at ~read 136-140, verified
     visually) — and ZERO acceptances of any episode inside the manually
     delimited absence window (reads 104-135): no false lock while the
     target is out of frame.
  3. Human-verified GT checkpoints: early-return window (reads 270/300/330,
     tests/v8_gt_checkpoints_early.json) for ALL selections; the original
     late-window checkpoints (tests/v8_gt_checkpoints.json, reads 390-470)
     additionally for the two COMMITTED selections. Fixed 40px tolerance.
  4. PROBATION_OK, TRACKING re-entry (>=5-frame run) and >=45 consecutive
     reads never LOST/PREDICT after re-entry.

Committed default: the acceptance coordinate (968,529) plus one FRACTIONAL
mouse-click-class coordinate (967.2,531.6) — interactive clicks map through
display scaling to fractional coordinates, a class the old integer-pinned
test could never exercise. The remaining four measured selections run under
V8_FULL_GRID=1.

The official sample video is assignment material and is NOT committed
(gitignored); the test skips when it is absent.

Run from the repository root:
    python3 -m unittest tests.test_v8_acceptance -v
    V8_FULL_GRID=1 python3 -m unittest tests.test_v8_acceptance -v
"""
import json
import os
import sys
import time
import types
import unittest

import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import config as prod_config
from ground_target_tracking import session, trackers, utils
from ground_target_tracking.session import TrackState

V8 = os.path.join("videos", "v8.mp4")
FIX_EARLY = os.path.join(os.path.dirname(__file__), "v8_gt_checkpoints_early.json")
FIX_LATE = os.path.join(os.path.dirname(__file__), "v8_gt_checkpoints.json")

COMMITTED = [(968.0, 529.0), (967.2, 531.6)]
GRID_EXTRA = [(967.0, 532.0), (973.0, 524.0), (976.0, 522.0), (966.6, 532.4),
              # Live interactive clicks recorded during the 2026-07-08
              # visual-verification sessions (user-blessed evidence):
              (969.6, 531.6), (969.6, 532.8), (968.4, 529.2)]
PICKS = COMMITTED + (GRID_EXTRA if os.environ.get("V8_FULL_GRID") == "1" else [])

ABSENCE = (104, 135)          # house out of frame (delimited + verified visually)
INIT_WINDOW = (140, 320)      # episode-1 accept window (post re-entry)


def run_pipeline(point_xy):
    """One deterministic full-video run of the real pipeline in the Run-A
    profile, isolated in a config snapshot so the production module is never
    mutated."""
    cfg = types.SimpleNamespace(**{k: getattr(prod_config, k)
                                   for k in dir(prod_config) if k.isupper()})
    cfg.OVERLAY_MASK_ENABLED = True
    cfg.EXPERIMENTAL_REGIONAL_MOTION = True
    cap, meta = utils.open_video(V8)
    frame0 = utils.get_frame(cap, 0)
    tracker = trackers.make_tracker("of_kalman", cfg)
    sess = session.TrackingSession(tracker, cfg, enable_reacq=True)
    sess.init(frame0, utils.Point2D(*point_xy))
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows = []
    read = 0
    t0 = time.perf_counter()
    for _, frame in utils.read_frames(cap):
        read += 1
        out = sess.step(frame)
        rows.append((read, out.state, float(out.point.x), float(out.point.y)))
    wall = time.perf_counter() - t0
    cap.release()
    return rows, list(sess._reacq_events), wall


def init_read_of(rows, point):
    """Recover the read of a REACQ_INIT from the committed-point trace (the
    reseed commits the accepted point verbatim on its frame)."""
    ix, iy = point
    matches = [r for r, s, x, y in rows
               if abs(x - ix) < 1.0 and abs(y - iy) < 1.0]
    return min(matches) if matches else None


@unittest.skipUnless(os.path.isfile(V8),
                     "official sample video not present (gitignored)")
class TestV8Acceptance(unittest.TestCase):
    runs = None

    @classmethod
    def setUpClass(cls):
        with open(FIX_EARLY) as f:
            cls.fx_early = json.load(f)
        with open(FIX_LATE) as f:
            cls.fx_late = json.load(f)
        cls.runs = {}
        for pick in PICKS:
            rows, events, wall = run_pipeline(pick)
            cls.runs[pick] = {"rows": rows, "events": events, "wall": wall}

    def per_pick(self):
        for pick in PICKS:
            with self.subTest(pick=pick):
                yield pick, self.runs[pick]

    def test_1_loss_occurs_within_sanity_bounds(self):
        for pick, run in self.per_pick():
            first_lost = next((r for r, s, x, y in run["rows"]
                               if s is TrackState.LOST), None)
            self.assertIsNotNone(first_lost, f"{pick}: never LOST")
            self.assertTrue(60 <= first_lost <= 300,
                            f"{pick}: LOST at read {first_lost}")

    def test_2_one_ep1_reacq_init_in_window_none_in_absence(self):
        for pick, run in self.per_pick():
            inits = [e for e in run["events"] if e["event"] == "REACQ_INIT"]
            ep1 = [e for e in inits if e.get("episode") == 1]
            self.assertEqual(len(ep1), 1,
                             f"{pick}: expected exactly one episode-1 "
                             f"REACQ_INIT, got {inits}")
            self.assertEqual(ep1[0]["cue"], "sift-lc", f"{pick}")
            for e in inits:                     # every accept, any episode
                r = init_read_of(run["rows"], e["point"])
                self.assertIsNotNone(r, f"{pick}: untraceable init {e}")
                self.assertFalse(ABSENCE[0] <= r <= ABSENCE[1],
                                 f"{pick}: accept at read {r} inside the "
                                 f"absence window — false lock")
            r1 = init_read_of(run["rows"], ep1[0]["point"])
            self.assertTrue(INIT_WINDOW[0] <= r1 <= INIT_WINDOW[1],
                            f"{pick}: episode-1 REACQ_INIT at read {r1}, "
                            f"outside {INIT_WINDOW}")

    def _check_checkpoints(self, pick, run, fixture, min_checked):
        # Per-checkpoint tolerances (target-relative rule, 2026-07-08:
        # clamp(1.25 x measured target diagonal, 15, 40)); the fixture-level
        # tolerance_px remains as the fallback for reads without an entry.
        default_tol = float(fixture["tolerance_px"])
        tol_map = fixture.get("tolerances", {})
        by_read = {r: (x, y) for r, s, x, y in run["rows"]}
        ep1 = [e for e in run["events"] if e["event"] == "REACQ_INIT"
               and e.get("episode") == 1][0]
        init_read = init_read_of(run["rows"], ep1["point"])
        checked = 0
        for read_s, (gx, gy) in fixture["checkpoints"].items():
            read = int(read_s)
            if read <= init_read:
                continue
            tol = float(tol_map.get(read_s, default_tol))
            x, y = by_read[read]
            d = float(np.hypot(x - gx, y - gy))
            self.assertLessEqual(d, tol,
                                 f"{pick} read {read}: committed "
                                 f"({x:.0f},{y:.0f}) is {d:.0f}px from GT "
                                 f"({gx},{gy}), tol {tol:.1f}px")
            checked += 1
        self.assertGreaterEqual(checked, min_checked, f"{pick}")

    def test_3_landing_checkpoints_within_tolerance(self):
        for pick, run in self.per_pick():
            self._check_checkpoints(pick, run, self.fx_early, min_checked=2)
            if pick in COMMITTED:
                self._check_checkpoints(pick, run, self.fx_late,
                                        min_checked=2)

    def test_4_probation_and_sustained_relock(self):
        for pick, run in self.per_pick():
            oks = [e for e in run["events"] if e["event"] == "PROBATION_OK"]
            self.assertGreaterEqual(len(oks), 1, f"{pick}: no PROBATION_OK")
            ep1 = [e for e in run["events"] if e["event"] == "REACQ_INIT"
                   and e.get("episode") == 1][0]
            init_read = init_read_of(run["rows"], ep1["point"])
            after = [(r, s) for r, s, x, y in run["rows"] if r > init_read]
            ok_read = next(r for r, s in after if s is TrackState.TRACKING)
            run_len = best = 0
            for r, s in after:
                if r < ok_read:
                    continue
                run_len = run_len + 1 if s is TrackState.TRACKING else 0
                best = max(best, run_len)
            self.assertGreaterEqual(best, 5,
                                    f"{pick}: TRACKING re-entry run too short")
            span = 0
            for r, s in after:
                if r < ok_read:
                    continue
                if s in (TrackState.LOST, TrackState.PREDICT):
                    break
                span += 1
            self.assertGreaterEqual(span, 45,
                                    f"{pick}: post-reacquisition lock only "
                                    f"{span} reads")

    def test_5_no_tracking_during_absence_or_search(self):
        """O1 regression guard (adopted 2026-07-08): the tracker must never
        certify TRACKING between target exit (read 104) and the episode-1
        stable re-lock. The pre-O1 recovery gate counted similarity-None
        (unevaluable, reference informative) as confirming, which let 9-11
        reads of coherent ambient desert texture reach full TRACKING while
        the target was absent (reads ~113-123, every measured pick)."""
        for pick, run in self.per_pick():
            ep1 = [e for e in run["events"] if e["event"] == "REACQ_INIT"
                   and e.get("episode") == 1][0]
            init_read = init_read_of(run["rows"], ep1["point"])
            stable = next(r for r, s, x, y in run["rows"]
                          if r > init_read and s is TrackState.TRACKING)
            bad = [r for r, s, x, y in run["rows"]
                   if ABSENCE[0] <= r < stable and s is TrackState.TRACKING]
            self.assertEqual(bad, [],
                             f"{pick}: TRACKING during absence/search at "
                             f"reads {bad[:5]}")

    def _anchor_speed(self, reads_sorted, anchors, c):
        """House per-read speed near checkpoint c, derived ONLY from the
        user-confirmed anchors (conservative: max of the adjacent-segment
        speeds). No trajectory under test contributes to its own allowance."""
        i = reads_sorted.index(c)
        speeds = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(reads_sorted):
                cc = reads_sorted[j]
                (ax, ay), (bx, by) = anchors[c], anchors[cc]
                speeds.append(float(np.hypot(ax - bx, ay - by))
                              / abs(c - cc))
        return max(speeds)

    def test_5b_accepted_reacquisition_near_anchor(self):
        """459 self-healing regression guard (Section-C oracle, user-adopted
        2026-07-08): every accepted reacquisition (ANY episode) with a
        user-confirmed checkpoint within +-15 reads must land within
        tol_c + v_c * |r - c| of that anchor. v_c comes from the confirmed
        anchors alone. Accepts with no checkpoint within 15 reads are logged
        as oracle-unavailable — logged is NEVER pass, and distant checkpoints
        are never compared. Measured origin: the read-480 episode-2
        re-acceptance of pick (973,524) landed 16 px from the read-470
        anchor — the designed self-healing path, converted here into a
        guarded property."""
        anchors, tols = {}, {}
        for fx in (self.fx_early, self.fx_late):
            for rs, xy in fx["checkpoints"].items():
                anchors[int(rs)] = (float(xy[0]), float(xy[1]))
                tols[int(rs)] = float(fx.get("tolerances", {}).get(
                    rs, fx["tolerance_px"]))
        reads_sorted = sorted(anchors)
        for pick, run in self.per_pick():
            for e in (x for x in run["events"]
                      if x["event"] == "REACQ_INIT"):
                r = init_read_of(run["rows"], e["point"])
                self.assertIsNotNone(r, f"{pick}: untraceable accept {e}")
                c = min(reads_sorted, key=lambda cc: abs(cc - r))
                if abs(r - c) > 15:
                    print(f"[oracle-unavailable] {pick} accept at read {r}: "
                          f"nearest confirmed checkpoint {c} is "
                          f"{abs(r - c)} reads away")
                    continue
                gate = tols[c] + self._anchor_speed(
                    reads_sorted, anchors, c) * abs(r - c)
                ax, ay = anchors[c]
                d = float(np.hypot(e["point"][0] - ax, e["point"][1] - ay))
                self.assertLessEqual(
                    d, gate,
                    f"{pick}: accept at read {r} is {d:.0f}px from the "
                    f"confirmed anchor at {c} (gate {gate:.1f}px)")

    def test_6_runtime_logged(self):
        for pick, run in self.per_pick():
            n = len(run["rows"])
            self.assertGreater(n, 800, f"{pick}")
            print(f"\n[v8-acceptance {pick}] {n} frames in "
                  f"{run['wall']:.1f}s ({1000.0 * run['wall'] / n:.1f} "
                  f"ms/frame avg incl. SIFT ticks)")

    def test_7_feed_frozen_tail_surfaces_no_lost_no_accepts(self):
        """v8 dead-feed tail (Commit 4a, §E): the frozen feed (reads ~685-855)
        surfaces the DISTINCT FEED_FROZEN condition, declares NO LOST after
        freeze onset (the loss machinery is suspended), and produces ZERO
        reacquisition accepts on the static image (search suspended). T-b is
        disabled in the shipped bundle; this is the T-a freeze detector."""
        for pick, run in self.per_pick():
            states = [(r, s) for r, s, x, y in run["rows"]]
            frozen = [r for r, s in states if s is TrackState.FEED_FROZEN]
            self.assertTrue(frozen, f"{pick}: FEED_FROZEN never surfaced")
            onset = min(frozen)
            self.assertTrue(680 <= onset <= 700,
                            f"{pick}: FEED_FROZEN onset at read {onset} "
                            f"(feed corruption ~678-683, freeze ~685)")
            lost_after = [r for r, s in states
                          if r >= onset and s is TrackState.LOST]
            self.assertEqual(lost_after, [],
                             f"{pick}: LOST at {lost_after[:5]} after freeze "
                             f"onset {onset}")
            for e in run["events"]:
                if e["event"] != "REACQ_INIT":
                    continue
                r = init_read_of(run["rows"], e["point"])
                if r is not None:
                    self.assertLess(r, onset,
                                    f"{pick}: reacq accept at read {r} on the "
                                    f"frozen feed (onset {onset})")
            self.assertGreater(len(frozen), 100,
                               f"{pick}: frozen span only {len(frozen)} reads")


if __name__ == "__main__":
    unittest.main()
