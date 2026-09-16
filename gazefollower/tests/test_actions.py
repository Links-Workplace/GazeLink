"""The router: four reasons to say no, and the one exception to the first.

Every one of these is a safety property rather than a return value.  A router
that accepted everything would pass a test that only checked the happy path,
and a router that refused everything would pass every test that only checked a
refusal -- so each rule is asserted in BOTH directions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_actions as A  # noqa: E402


def _req(action, *, source=A.Source.WINK, at=(0.5, 0.5), issued=100.0, text=None):
    return A.ActionRequest(action, source, issued, at=at, text=text)


class VocabularyTests(unittest.TestCase):
    """The names mirror the product's own, so porting is a rename."""

    def test_the_interaction_types_the_product_already_names_are_spelled_the_same(self) -> None:
        """Written out rather than imported. The two trees are kept apart on
        purpose -- gazefollower_capture says so in its first lines -- so this
        asserts the SPELLING matches src/gazelink/domain.py without creating
        the dependency that separation exists to avoid. If the product renames
        one of these, this test is where it is noticed."""

        for name in (
            "LEFT_CLICK",
            "RIGHT_CLICK",
            "DOUBLE_CLICK",
            "PAUSE",
            "RESUME",
        ):
            self.assertEqual(str(getattr(A.Action, name)), name)
        for name in ("WINK", "DWELL", "UI"):
            self.assertEqual(str(getattr(A.Source, name)), name)

    def test_a_typed_request_must_carry_text_and_others_must_not(self) -> None:
        with self.assertRaises(ValueError):
            _req(A.Action.TYPE_TEXT)
        with self.assertRaises(ValueError):
            _req(A.Action.ENTER, text="x")
        self.assertEqual(_req(A.Action.TYPE_TEXT, text="א").text, "א")


class PausedTests(unittest.TestCase):
    def test_everything_is_refused_while_paused(self) -> None:
        router = A.ActionRouter()
        for action in (
            A.Action.LEFT_CLICK,
            A.Action.RIGHT_CLICK,
            A.Action.DOUBLE_CLICK,
            A.Action.BACK,
            A.Action.SWITCH_WINDOW,
            A.Action.BACKSPACE,
            A.Action.PAUSE,
        ):
            decision = router.decide(
                _req(action), paused=True, ui_mode=A.UiMode.CURSOR, now_s=100.0
            )
            self.assertFalse(decision.accepted, f"{action} got through while paused")
            self.assertIs(decision.reason, A.Refusal.PAUSED)

    def test_resume_is_the_one_thing_that_gets_through(self) -> None:
        """Without this there is no way back to control without a keyboard,
        which is the first priority in CLAUDE.md and not a nicety."""

        router = A.ActionRouter()
        decision = router.decide(
            _req(A.Action.RESUME, source=A.Source.DWELL),
            paused=True,
            ui_mode=A.UiMode.MENU,
            now_s=100.0,
        )
        self.assertTrue(decision.accepted, "a paused session had no way back in")

    def test_the_same_actions_are_accepted_when_not_paused(self) -> None:
        """The other half: a router that refused everything would pass the
        first test on its own."""

        router = A.ActionRouter()
        decision = router.decide(
            _req(A.Action.LEFT_CLICK), paused=False, ui_mode=A.UiMode.CURSOR, now_s=100.0
        )
        self.assertTrue(decision.accepted)


class StaleTests(unittest.TestCase):
    def test_an_event_older_than_the_limit_is_refused(self) -> None:
        router = A.ActionRouter(A.RouterLimits(max_age_ms=100.0))
        decision = router.decide(
            _req(A.Action.LEFT_CLICK, issued=100.0),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=100.5,
        )
        self.assertFalse(decision.accepted)
        self.assertIs(decision.reason, A.Refusal.STALE)

    def test_a_stale_resume_is_refused_as_well(self) -> None:
        """The exception is to the PAUSE rule, not to staleness. A queue
        drained late must not be able to resume control the person stopped."""

        router = A.ActionRouter(A.RouterLimits(max_age_ms=100.0))
        decision = router.decide(
            _req(A.Action.RESUME, issued=100.0),
            paused=True,
            ui_mode=A.UiMode.MENU,
            now_s=105.0,
        )
        self.assertFalse(decision.accepted)
        self.assertIs(decision.reason, A.Refusal.STALE)

    def test_a_fresh_event_is_not_refused(self) -> None:
        router = A.ActionRouter(A.RouterLimits(max_age_ms=100.0))
        decision = router.decide(
            _req(A.Action.LEFT_CLICK, issued=100.0),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=100.05,
        )
        self.assertTrue(decision.accepted)


class DuplicateTests(unittest.TestCase):
    def test_the_same_action_from_the_same_source_too_soon_is_refused(self) -> None:
        router = A.ActionRouter(A.RouterLimits(min_gap_ms=200.0))
        first = router.decide(
            _req(A.Action.LEFT_CLICK, issued=10.0),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=10.0,
        )
        second = router.decide(
            _req(A.Action.LEFT_CLICK, issued=10.1),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=10.1,
        )
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertIs(second.reason, A.Refusal.DUPLICATE)

    def test_a_different_action_is_not_a_duplicate(self) -> None:
        router = A.ActionRouter(A.RouterLimits(min_gap_ms=200.0))
        router.decide(
            _req(A.Action.LEFT_CLICK, issued=10.0),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=10.0,
        )
        other = router.decide(
            _req(A.Action.RIGHT_CLICK, issued=10.1),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=10.1,
        )
        self.assertTrue(other.accepted, "two different intentions were read as one repeat")

    def test_a_change_of_mode_clears_what_was_asked_recently(self) -> None:
        router = A.ActionRouter(A.RouterLimits(min_gap_ms=200.0))
        router.decide(
            _req(A.Action.LEFT_CLICK, issued=10.0),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=10.0,
        )
        router.reset()
        again = router.decide(
            _req(A.Action.LEFT_CLICK, issued=10.05),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=10.05,
        )
        self.assertTrue(again.accepted, "a deliberate return to clicking read as an echo")


class WrongModeTests(unittest.TestCase):
    def test_a_wink_does_not_click_while_scrolling_or_typing_or_choosing(self) -> None:
        router = A.ActionRouter()
        for mode in (A.UiMode.SCROLL, A.UiMode.KEYBOARD, A.UiMode.MENU):
            decision = router.decide(
                _req(A.Action.DOUBLE_CLICK),
                paused=False,
                ui_mode=mode,
                now_s=100.0,
            )
            self.assertFalse(decision.accepted, f"a wink clicked in {mode}")
            self.assertIs(decision.reason, A.Refusal.WRONG_MODE)

    def test_the_same_wink_clicks_in_cursor_mode(self) -> None:
        router = A.ActionRouter()
        decision = router.decide(
            _req(A.Action.DOUBLE_CLICK), paused=False, ui_mode=A.UiMode.CURSOR, now_s=100.0
        )
        self.assertTrue(decision.accepted)

    def test_typing_only_happens_at_the_keyboard(self) -> None:
        router = A.ActionRouter()
        outside = router.decide(
            _req(A.Action.TYPE_TEXT, source=A.Source.SCAN, text="א"),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=100.0,
        )
        inside = router.decide(
            _req(A.Action.TYPE_TEXT, source=A.Source.SCAN, text="א"),
            paused=False,
            ui_mode=A.UiMode.KEYBOARD,
            now_s=100.0,
        )
        self.assertFalse(outside.accepted)
        self.assertIs(outside.reason, A.Refusal.WRONG_MODE)
        self.assertTrue(inside.accepted)

    def test_the_named_commands_work_from_wherever_the_menu_was_opened(self) -> None:
        """The menu can be opened from any mode, so refusing its own commands
        by mode would make it unable to do the things it exists for."""

        router = A.ActionRouter()
        for i, mode in enumerate(A.UiMode):
            now = 100.0 + i
            decision = router.decide(
                _req(A.Action.BACK, source=A.Source.DWELL, issued=now),
                paused=False,
                ui_mode=mode,
                now_s=now,
            )
            self.assertTrue(decision.accepted, f"BACK was refused in {mode}")


class CountingTests(unittest.TestCase):
    def test_every_refusal_is_counted_under_its_own_reason(self) -> None:
        """"Nothing happened" needs four different fixes, so it has to come
        back as four different numbers."""

        router = A.ActionRouter(A.RouterLimits(max_age_ms=50.0, min_gap_ms=500.0))
        router.decide(
            _req(A.Action.LEFT_CLICK), paused=True, ui_mode=A.UiMode.CURSOR, now_s=100.0
        )
        router.decide(
            _req(A.Action.LEFT_CLICK, issued=99.0),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=100.0,
        )
        router.decide(
            _req(A.Action.LEFT_CLICK), paused=False, ui_mode=A.UiMode.SCROLL, now_s=100.0
        )
        router.decide(
            _req(A.Action.LEFT_CLICK), paused=False, ui_mode=A.UiMode.CURSOR, now_s=100.0
        )
        router.decide(
            _req(A.Action.LEFT_CLICK, issued=100.1),
            paused=False,
            ui_mode=A.UiMode.CURSOR,
            now_s=100.1,
        )
        self.assertEqual(router.refused[A.Refusal.PAUSED], 1)
        self.assertEqual(router.refused[A.Refusal.STALE], 1)
        self.assertEqual(router.refused[A.Refusal.WRONG_MODE], 1)
        self.assertEqual(router.refused[A.Refusal.DUPLICATE], 1)
        self.assertEqual(router.accepted[A.Action.LEFT_CLICK], 1)

    def test_a_refused_request_never_counts_as_accepted(self) -> None:
        router = A.ActionRouter()
        router.decide(
            _req(A.Action.LEFT_CLICK), paused=True, ui_mode=A.UiMode.CURSOR, now_s=100.0
        )
        self.assertEqual(router.accepted, {})


class SeparationTests(unittest.TestCase):
    def test_the_router_cannot_emit_anything(self) -> None:
        """It decides. Anything here that could reach Windows would mean the
        decision and the doing had merged again."""

        source = Path(__import__("gf_actions").__file__).read_text(
            encoding="utf-8"
        )
        # Checked as CODE rather than as words: the module docstring says
        # "no ctypes" in prose, and a bare substring search would fail on the
        # very sentence that promises the property.
        for forbidden in ("import ctypes", "ctypes.", "SendInput(", "MOUSEEVENTF", "KEYEVENTF"):
            self.assertNotIn(forbidden, source, f"{forbidden} appeared in the router")


if __name__ == "__main__":
    unittest.main()
