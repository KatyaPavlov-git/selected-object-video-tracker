"""CLI-level policies: manual [i,j] point entry (the task.pdf convention:
indices are row, col) and the selection-time reacquirability guard
(Run-A postmortem: warn when a selection carries no target appearance).

Run from the repository root:
    python3 -m unittest discover tests -v
"""
import contextlib
import io
import sys
import types
import unittest

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import main as gtt_main
from ground_target_tracking import utils
from ground_target_tracking.session import TrackingSession

try:  # discovery adds tests/ to sys.path; direct module runs use the package
    from test_overlay_mask import (StubTracker as OverlayStubTracker,
                                   letterboxed_frame, make_session_cfg, res)
except ImportError:
    from tests.test_overlay_mask import (StubTracker as OverlayStubTracker,
                                         letterboxed_frame, make_session_cfg,
                                         res)


class TestPointRC(unittest.TestCase):
    def test_rc_maps_row_col_to_xy(self):
        p = gtt_main.point_from_rc(120.0, 45.0)   # [i]=row -> y, [j]=col -> x
        self.assertEqual((p.x, p.y), (45.0, 120.0))

    def test_point_rc_cli_sets_point(self):
        args = gtt_main.parse_args(["--point-rc", "120,45"])
        self.assertEqual((args.point.x, args.point.y), (45.0, 120.0))

    def test_point_xy_unchanged(self):
        args = gtt_main.parse_args(["--point", "45,120"])
        self.assertEqual((args.point.x, args.point.y), (45.0, 120.0))

    def test_point_and_point_rc_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gtt_main.parse_args(["--point", "1,2", "--point-rc", "2,1"])

    def test_malformed_rc_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gtt_main.parse_args(["--point-rc", "120;45"])


def _stub_sess(dead_zone=False, reacq="ref", has_desc=True, has_ctx=True):
    """Minimal session stand-in for the tier helper: only the fields it reads."""
    if reacq is None:
        rq = None
    elif reacq == "no-ref":
        rq = types.SimpleNamespace(reference=None)
    else:
        rq = types.SimpleNamespace(reference=types.SimpleNamespace(
            has_descriptors=has_desc, has_context=has_ctx))
    return types.SimpleNamespace(init_dead_zone=dead_zone, _reacq=rq)


class _StubReacqBothCaps:
    """Injected reacquirer whose reference CLAIMS both capabilities — the
    Run-A trap: capability flags alone look fine while the selection itself
    is a dead zone."""

    def __init__(self):
        self.reference = types.SimpleNamespace(has_descriptors=True,
                                               has_context=True)

    def build_reference(self, frame_bgr, point, overlay_mask=None):
        pass


class TestReacqSelectionTier(unittest.TestCase):
    def test_dead_zone_is_tier1_even_with_capabilities(self):
        tier, msg = gtt_main.reacq_selection_tier(
            _stub_sess(dead_zone=True, has_desc=True, has_ctx=True))
        self.assertEqual(tier, 1)
        self.assertIn("WARNING", msg)
        # Warn + continue contract: the message must say the selection IS
        # accepted and that HUD pixels stay out of evidence.
        self.assertIn("accepted", msg)
        self.assertIn("excluded from all evidence", msg)

    def test_no_reference_is_tier1(self):
        self.assertEqual(gtt_main.reacq_selection_tier(
            _stub_sess(reacq="no-ref"))[0], 1)
        self.assertEqual(gtt_main.reacq_selection_tier(
            _stub_sess(reacq=None))[0], 1)

    def test_no_capability_is_tier1(self):
        tier, _ = gtt_main.reacq_selection_tier(
            _stub_sess(has_desc=False, has_ctx=False))
        self.assertEqual(tier, 1)

    def test_template_only_is_tier2(self):
        tier, msg = gtt_main.reacq_selection_tier(
            _stub_sess(has_desc=False, has_ctx=True))
        self.assertEqual(tier, 2)
        self.assertIn("template", msg)

    def test_both_capabilities_is_silent(self):
        tier, msg = gtt_main.reacq_selection_tier(_stub_sess())
        self.assertEqual(tier, 0)
        self.assertEqual(msg, "")

    def _crosshair_session(self):
        """Real session initialized at the overlay fixture's crosshair center
        (the exact Run-A selection class): the dilated disc dominates the ROI,
        so the session declares a dead zone even though the injected reference
        claims both capabilities."""
        cfg = make_session_cfg()
        frame = letterboxed_frame(textured=True)
        cross_p = utils.Point2D(300.0, 200.0)
        sess = TrackingSession(OverlayStubTracker([res(cross_p.x, cross_p.y)]),
                               cfg, enable_reacq=True,
                               reacquirer=_StubReacqBothCaps())
        sess.init(frame, cross_p)
        return sess, frame, cross_p

    def test_crosshair_point_is_accepted_and_session_operates(self):
        # The selection is ALWAYS accepted: init succeeds, the state machine
        # runs, and HUD pixels stay out of evidence (the tracker got the
        # ignore mask; the ROI similarity is neutral, never fabricated).
        sess, frame, cross_p = self._crosshair_session()
        self.assertTrue(sess.init_dead_zone, "fixture precondition")
        self.assertEqual((sess.point.x, sess.point.y), (cross_p.x, cross_p.y))
        self.assertIsNotNone(sess._overlay, "overlay mask must be installed")
        out = sess.step(frame)
        self.assertIsNotNone(out.state)
        self.assertIsNone(out.signals["similarity"],
                          "a fully HUD-covered ROI must score neutral (None), "
                          "never a fabricated similarity")

    def test_real_session_crosshair_dead_zone_is_tier1(self):
        sess, _, _ = self._crosshair_session()
        tier, _ = gtt_main.reacq_selection_tier(sess)
        self.assertEqual(tier, 1)

    def test_announce_warns_and_continues_without_repick(self):
        # Warn + continue: the announcement prints the Tier-1 contract and
        # NEVER re-prompts — the picker must not be reachable from it.
        sess, _, _ = self._crosshair_session()
        real_picker = gtt_main.select_point_interactive

        def _forbidden(*a, **k):
            raise AssertionError("guard attempted a re-pick")

        gtt_main.select_point_interactive = _forbidden
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tier = gtt_main.announce_reacq_selection(sess)
        finally:
            gtt_main.select_point_interactive = real_picker
        self.assertEqual(tier, 1)
        self.assertIn("accepted", buf.getvalue())

    def test_announce_silent_on_capable_selection(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tier = gtt_main.announce_reacq_selection(_stub_sess())
        self.assertEqual(tier, 0)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
