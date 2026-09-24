"""Drag: the button must come up, from every direction it can be taken away.

The dangerous failure of this whole project is a mouse button left down. Every
test here is about that, and several of them assert that a refusal did NOT
turn into a release in the wrong place -- refusing a drop by letting go is the
same defect wearing better clothes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.interaction import drag as DG  # noqa: E402
from gazelink_core.platform import click as CK  # noqa: E402


class Sender:
    def __init__(self):
        self.flags = []
        self.fail_on = None

    def __call__(self, flag):
        if self.fail_on is not None and flag == self.fail_on:
            raise OSError("SendInput refused")
        self.flags.append(flag)


def adapter(**kw):
    sender = Sender()
    clock = {"t": 0.0}

    def now():
        clock["t"] += 1.0
        return clock["t"]

    adapter = CK.ClickAdapter(
        enabled=True, sender=sender, clock=now, sleep=lambda s: None, **kw
    )
    return adapter, sender


DOWN, UP = CK.BUTTONS["left"]


class TheSequenceTests(unittest.TestCase):
    def test_a_pick_with_no_point_is_refused_and_nothing_is_held(self) -> None:
        seq = DG.DragSequence()
        seq.begin()
        self.assertFalse(seq.pick(None))
        self.assertFalse(seq.held)

    def test_a_drop_with_no_point_keeps_carrying_rather_than_letting_go(self) -> None:
        # Refusing into a release would drop the item where nobody chose.
        seq = DG.DragSequence()
        seq.begin()
        seq.pick((0.5, 0.5))
        self.assertFalse(seq.drop(None))
        self.assertTrue(seq.held)
        self.assertIs(seq.step, DG.Step.CARRYING)

    def test_a_full_sequence_ends_holding_nothing(self) -> None:
        seq = DG.DragSequence()
        seq.begin()
        self.assertTrue(seq.pick((0.4, 0.5)))
        self.assertTrue(seq.drop((0.6, 0.5)))
        self.assertFalse(seq.held)
        self.assertEqual(seq.tally["dropped"], 1)

    def test_a_drop_before_a_pick_is_refused(self) -> None:
        seq = DG.DragSequence()
        seq.begin()
        self.assertFalse(seq.drop((0.5, 0.5)))

    def test_a_second_pick_while_carrying_is_refused(self) -> None:
        seq = DG.DragSequence()
        seq.begin()
        seq.pick((0.4, 0.5))
        self.assertFalse(seq.pick((0.7, 0.5)))

    def test_one_instruction_is_offered_at_a_time(self) -> None:
        seq = DG.DragSequence()
        seq.begin()
        self.assertEqual(seq.say, "בחר פריט")
        seq.pick((0.4, 0.5))
        self.assertEqual(seq.say, "בחר מקום להנחה")
        seq.drop((0.6, 0.5))
        self.assertEqual(seq.say, "בוצע")

    def test_the_way_out_is_offered_at_every_step_that_holds_anything(self) -> None:
        for step in (DG.Step.IDLE, DG.Step.PICKING, DG.Step.CARRYING):
            self.assertIn("cancel", DG.OFFERS[step], step)


class EveryExitReleasesTests(unittest.TestCase):
    def _carrying(self):
        seq = DG.DragSequence()
        seq.begin()
        seq.pick((0.5, 0.5))
        return seq

    def test_cancel_releases(self) -> None:
        seq = self._carrying()
        self.assertTrue(seq.cancel())
        self.assertFalse(seq.held)

    def test_a_pause_releases(self) -> None:
        seq = self._carrying()
        self.assertTrue(seq.force_release(DG.Ended.PAUSED))
        self.assertFalse(seq.held)
        self.assertIs(seq.ended, DG.Ended.PAUSED)

    def test_a_lost_face_releases(self) -> None:
        seq = self._carrying()
        self.assertTrue(seq.force_release(DG.Ended.TRACKING_LOST))
        self.assertFalse(seq.held)

    def test_shutdown_releases(self) -> None:
        seq = self._carrying()
        self.assertTrue(seq.force_release(DG.Ended.STOPPING))
        self.assertFalse(seq.held)

    def test_forcing_twice_is_harmless(self) -> None:
        seq = self._carrying()
        self.assertTrue(seq.force_release(DG.Ended.PAUSED))
        self.assertFalse(seq.force_release(DG.Ended.STOPPING))
        self.assertEqual(seq.tally["forced"], 1)

    def test_a_forced_release_is_counted_apart_from_a_deliberate_one(self) -> None:
        deliberate = self._carrying()
        deliberate.drop((0.6, 0.5))
        forced = self._carrying()
        forced.force_release(DG.Ended.PAUSED)
        self.assertEqual((deliberate.tally["dropped"], deliberate.tally["forced"]), (1, 0))
        self.assertEqual((forced.tally["dropped"], forced.tally["forced"]), (0, 1))


class TheAdapterHoldsOnPurposeTests(unittest.TestCase):
    def test_a_press_needs_arming_and_a_release_never_does(self) -> None:
        click, sender = adapter()
        self.assertFalse(click.press(armed=False))
        self.assertEqual(sender.flags, [])
        self.assertTrue(click.press(armed=True))
        self.assertEqual(sender.flags, [DOWN])
        self.assertTrue(click.drag_release())
        self.assertEqual(sender.flags, [DOWN, UP])

    def test_a_click_during_a_drag_is_refused_not_silently_converted(self) -> None:
        # Before this distinction existed, _press_and_release read the held
        # button as stuck, released it, and clicked -- dropping the item.
        click, sender = adapter()
        click.press(armed=True)
        self.assertFalse(click.click(armed=True))
        self.assertEqual(sender.flags, [DOWN])
        self.assertTrue(click.dragging)
        self.assertEqual(click.refused_while_dragging, 1)

    def test_a_double_click_during_a_drag_is_refused_too(self) -> None:
        click, sender = adapter()
        click.press(armed=True)
        self.assertFalse(click.double_click(armed=True))
        self.assertEqual(sender.flags, [DOWN])

    def test_a_right_click_during_a_drag_is_refused_too(self) -> None:
        click, sender = adapter()
        click.press(armed=True)
        self.assertFalse(click.right_click(armed=True))
        self.assertEqual(sender.flags, [DOWN])

    def test_a_second_press_is_refused(self) -> None:
        click, sender = adapter()
        click.press(armed=True)
        self.assertFalse(click.press(armed=True))
        self.assertEqual(sender.flags, [DOWN])

    def test_releasing_twice_is_harmless(self) -> None:
        click, sender = adapter()
        click.press(armed=True)
        self.assertTrue(click.drag_release())
        self.assertFalse(click.drag_release())
        self.assertEqual(sender.flags, [DOWN, UP])

    def test_shutdown_release_clears_the_drag_flag_as_well(self) -> None:
        click, sender = adapter()
        click.press(armed=True)
        click.release()
        self.assertFalse(click.dragging)
        self.assertEqual(sender.flags, [DOWN, UP])

    def test_a_failed_release_keeps_naming_the_button(self) -> None:
        click, sender = adapter()
        click.press(armed=True)
        sender.fail_on = UP
        self.assertFalse(click.drag_release())
        self.assertEqual(click.stuck_button, "left")

    def test_a_failed_press_holds_nothing(self) -> None:
        click, sender = adapter()
        sender.fail_on = DOWN
        self.assertFalse(click.press(armed=True))
        self.assertFalse(click.dragging)

    def test_the_gap_guard_sees_the_drag(self) -> None:
        # Without this, a click immediately after a drop slipped under the
        # adapter's own minimum gap as though nothing had happened.
        click, _ = adapter()
        click.press(armed=True)
        self.assertIsNotNone(click._last_s)

    def test_a_disabled_adapter_sends_nothing_but_still_tracks_state(self) -> None:
        sender = Sender()
        click = CK.ClickAdapter(enabled=False, sender=sender, clock=lambda: 1.0)
        self.assertTrue(click.press(armed=True))
        self.assertTrue(click.dragging)
        self.assertEqual(sender.flags, [])
        self.assertTrue(click.drag_release())
        self.assertFalse(click.dragging)

    def test_the_summary_reports_the_carry(self) -> None:
        click, _ = adapter()
        click.press(armed=True)
        self.assertEqual(click.summary()["dragging"], True)
        self.assertEqual(click.summary()["drags"], 1)


class TheRouterIsNotOnTheReleasePathTests(unittest.TestCase):
    def test_the_release_helper_does_not_consult_a_permission(self) -> None:
        import inspect

        source = inspect.getsource(CK.ClickAdapter.drag_release)
        self.assertNotIn("armed", source)

    def test_only_the_start_of_a_drag_is_a_routed_action(self) -> None:
        from gazelink_core.interaction import actions as A

        self.assertIn(A.Action.DRAG_START, A.DRAGS)
        self.assertIn(A.Action.DRAG_DROP, A.DRAGS)
        self.assertIn(A.Action.DRAG_CANCEL, A.DRAGS)


if __name__ == "__main__":
    unittest.main()
