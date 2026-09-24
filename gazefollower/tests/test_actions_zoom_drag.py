"""The two new modes must refuse more than they allow.

ZOOM and DRAG exist because in both of them a wink means something other than
"click here". Every test below is about something NOT happening.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.interaction import actions as A  # noqa: E402

MODES = list(A.UiMode)


def _router():
    return A.ActionRouter()


def _ask(router, action, mode, *, paused=False, now=10.0, text=None):
    request = A.ActionRequest(action, A.Source.WINK, now, at=(0.5, 0.5), text=text)
    return router.decide(request, paused=paused, ui_mode=mode, now_s=now)


class WinkIsNeverASelectionInTheNewModesTests(unittest.TestCase):
    def test_a_wink_in_zoom_is_refused_before_anything_reads_it(self) -> None:
        got = _router().screen_gesture(
            issued_at_s=10.0, now_s=10.0, ui_mode=A.UiMode.ZOOM, paused=False
        )
        self.assertIs(got, A.Refusal.WRONG_MODE)

    def test_a_wink_in_drag_is_refused_before_anything_reads_it(self) -> None:
        got = _router().screen_gesture(
            issued_at_s=10.0, now_s=10.0, ui_mode=A.UiMode.DRAG, paused=False
        )
        self.assertIs(got, A.Refusal.WRONG_MODE)

    def test_staleness_still_wins_over_the_mode(self) -> None:
        # An old event is not an intention about now, whatever mode it names.
        got = _router().screen_gesture(
            issued_at_s=0.0, now_s=10.0, ui_mode=A.UiMode.ZOOM, paused=False
        )
        self.assertIs(got, A.Refusal.STALE)

    def test_an_ordinary_click_is_refused_in_both_new_modes(self) -> None:
        for mode in (A.UiMode.ZOOM, A.UiMode.DRAG):
            for action in sorted(A.CLICKS):
                with self.subTest(mode=mode, action=action):
                    decision = _ask(_router(), action, mode)
                    self.assertFalse(decision.accepted)
                    self.assertIs(decision.reason, A.Refusal.WRONG_MODE)


class ZoomClickBelongsOnlyToZoomTests(unittest.TestCase):
    def test_it_is_accepted_in_zoom(self) -> None:
        self.assertTrue(_ask(_router(), A.Action.ZOOM_CLICK, A.UiMode.ZOOM).accepted)

    def test_it_is_refused_in_every_other_mode(self) -> None:
        for mode in MODES:
            if mode is A.UiMode.ZOOM:
                continue
            with self.subTest(mode):
                decision = _ask(_router(), A.Action.ZOOM_CLICK, mode)
                self.assertFalse(decision.accepted)
                self.assertIs(decision.reason, A.Refusal.WRONG_MODE)

    def test_it_is_not_a_member_of_the_ordinary_click_set(self) -> None:
        # If it were, every rule written for CLICKS would quietly cover it --
        # including the one that lets a wink produce them in CURSOR mode.
        self.assertNotIn(A.Action.ZOOM_CLICK, A.CLICKS)

    def test_it_is_refused_while_paused(self) -> None:
        decision = _ask(_router(), A.Action.ZOOM_CLICK, A.UiMode.ZOOM, paused=True)
        self.assertFalse(decision.accepted)
        self.assertIs(decision.reason, A.Refusal.PAUSED)


class DragActionsBelongOnlyToDragTests(unittest.TestCase):
    def test_each_one_is_accepted_in_drag(self) -> None:
        for action in sorted(A.DRAGS):
            with self.subTest(action):
                self.assertTrue(_ask(_router(), action, A.UiMode.DRAG).accepted)

    def test_each_one_is_refused_everywhere_else(self) -> None:
        for action in sorted(A.DRAGS):
            for mode in MODES:
                if mode is A.UiMode.DRAG:
                    continue
                with self.subTest(action=action, mode=mode):
                    decision = _ask(_router(), action, mode)
                    self.assertFalse(decision.accepted)
                    self.assertIs(decision.reason, A.Refusal.WRONG_MODE)

    def test_the_set_holds_exactly_the_three_steps(self) -> None:
        self.assertEqual(
            A.DRAGS,
            frozenset({A.Action.DRAG_START, A.Action.DRAG_DROP, A.Action.DRAG_CANCEL}),
        )

    def test_a_drag_step_is_refused_while_paused(self) -> None:
        for action in sorted(A.DRAGS):
            with self.subTest(action):
                decision = _ask(_router(), action, A.UiMode.DRAG, paused=True)
                self.assertFalse(decision.accepted)
                self.assertIs(decision.reason, A.Refusal.PAUSED)


class TheWayBackSurvivesEveryModeTests(unittest.TestCase):
    def test_resume_is_allowed_in_every_mode_including_the_new_ones(self) -> None:
        for mode in MODES:
            with self.subTest(mode):
                self.assertTrue(_ask(_router(), A.Action.RESUME, mode, paused=True).accepted)

    def test_pause_is_allowed_in_every_mode(self) -> None:
        for mode in MODES:
            with self.subTest(mode):
                self.assertTrue(_ask(_router(), A.Action.PAUSE, mode).accepted)


class NothingWasWidenedByAccidentTests(unittest.TestCase):
    def test_scrolling_did_not_become_reachable_from_the_new_modes(self) -> None:
        for mode in (A.UiMode.ZOOM, A.UiMode.DRAG):
            for action in sorted(A.SCROLLS):
                with self.subTest(mode=mode, action=action):
                    self.assertFalse(_ask(_router(), action, mode).accepted)

    def test_typing_did_not_become_reachable_from_the_new_modes(self) -> None:
        for mode in (A.UiMode.ZOOM, A.UiMode.DRAG):
            decision = _ask(_router(), A.Action.BACKSPACE, mode)
            self.assertFalse(decision.accepted)

    def test_every_action_is_classified_somewhere(self) -> None:
        # ``_belongs`` ends in ``return False``: an action nobody classified
        # fails closed AND silently, so the set membership is asserted here.
        classified = A.CLICKS | A.TYPING | A.SCROLLS | A.COMMANDS | A.DRAGS
        classified |= {A.Action.PAUSE, A.Action.RESUME, A.Action.ZOOM_CLICK}
        self.assertEqual(set(A.Action) - classified, set())

    def test_every_action_is_reachable_from_at_least_one_mode(self) -> None:
        for action in list(A.Action):
            text = "x" if action is A.Action.TYPE_TEXT else None
            reachable = [m for m in MODES if _ask(_router(), action, m, text=text).accepted]
            self.assertTrue(reachable, f"{action} belongs to no mode at all")


class CounterCoverageTests(unittest.TestCase):
    def test_every_mode_that_refuses_a_wink_has_a_counter(self) -> None:
        from gazelink_core.app.telemetry import _new_tally
        from gazelink_core.interaction.controller import WINK_REFUSED_IN

        router = _router()
        tally = _new_tally()
        for mode in MODES:
            refusal = router.screen_gesture(
                issued_at_s=10.0, now_s=10.0, ui_mode=mode, paused=False
            )
            if refusal is A.Refusal.WRONG_MODE:
                with self.subTest(mode):
                    self.assertIn(mode, WINK_REFUSED_IN, f"{mode} refuses winks and counts none")
                    self.assertIn(WINK_REFUSED_IN[mode], tally)


if __name__ == "__main__":
    unittest.main()
