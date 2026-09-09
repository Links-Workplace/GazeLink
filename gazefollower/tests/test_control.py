"""The three control modes. Pure logic: no camera, no Qt, no OS input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_control as K  # noqa: E402
import gf_gesture as GEST  # noqa: E402


def arm(machine: K.ControlMachine) -> K.Transition:
    """Walk the menu to MOVE_AND_SELECT the way a user would."""

    last = machine.update(GEST.Event.NONE)
    while machine.highlight is not K.Mode.MOVE_AND_SELECT:
        last = machine.update(GEST.Event.CYCLE)
    return machine.update(GEST.Event.CONFIRM) if last else machine.update(GEST.Event.CONFIRM)


class StartingStateTests(unittest.TestCase):
    def test_a_session_begins_paused(self) -> None:
        """A session that begins armed can click before anyone agreed to it."""

        machine = K.ControlMachine()
        self.assertIs(machine.mode, K.Mode.PAUSED)
        self.assertFalse(machine.mode.cursor_enabled)
        self.assertFalse(machine.mode.selection_armed)

    def test_the_highlight_starts_on_the_option_that_does_nothing(self) -> None:
        """An accidental long close must not be able to arm selection."""

        machine = K.ControlMachine()
        self.assertIs(machine.highlight, K.Mode.PAUSED)
        transition = machine.update(GEST.Event.CONFIRM)
        self.assertIs(transition.mode, K.Mode.PAUSED)
        self.assertFalse(transition.selection_armed)

    def test_the_cycle_reaches_every_mode_and_comes_back(self) -> None:
        machine = K.ControlMachine()
        seen = [machine.highlight]
        for _ in range(len(K.MODE_ORDER)):
            seen.append(machine.update(GEST.Event.CYCLE).highlight)
        self.assertEqual(set(seen), set(K.MODE_ORDER))
        self.assertIs(seen[0], seen[-1])


class ModeCapabilityTests(unittest.TestCase):
    def test_only_the_armed_mode_permits_a_click(self) -> None:
        armed = [m for m in K.MODE_ORDER if m.selection_armed]
        self.assertEqual(armed, [K.Mode.MOVE_AND_SELECT])

    def test_move_only_moves_but_cannot_click(self) -> None:
        self.assertTrue(K.Mode.MOVE_ONLY.cursor_enabled)
        self.assertFalse(K.Mode.MOVE_ONLY.selection_armed)

    def test_paused_does_neither(self) -> None:
        self.assertFalse(K.Mode.PAUSED.cursor_enabled)
        self.assertFalse(K.Mode.PAUSED.selection_armed)


class CancellationTests(unittest.TestCase):
    """A part-way selection that survives an interruption is a click nobody
    asked for, delivered late, at a gaze position measured before it."""

    def test_losing_the_face_pauses_and_cancels(self) -> None:
        machine = K.ControlMachine()
        arm(machine)
        self.assertTrue(machine.mode.selection_armed)
        transition = machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertIs(transition.mode, K.Mode.PAUSED)
        self.assertTrue(transition.cancel_selection)
        self.assertIs(transition.reason, K.Reason.TRACKING_LOST)

    def test_a_profile_change_pauses_and_cancels(self) -> None:
        machine = K.ControlMachine()
        arm(machine)
        transition = machine.interrupt(K.Reason.PROFILE_CHANGED)
        self.assertIs(transition.mode, K.Mode.PAUSED)
        self.assertTrue(transition.cancel_selection)

    def test_entering_calibration_pauses_and_cancels(self) -> None:
        machine = K.ControlMachine()
        arm(machine)
        transition = machine.interrupt(K.Reason.CALIBRATION_ENTERED)
        self.assertIs(transition.mode, K.Mode.PAUSED)
        self.assertTrue(transition.cancel_selection)

    def test_leaving_the_armed_mode_cancels_what_was_filling(self) -> None:
        machine = K.ControlMachine()
        arm(machine)
        while machine.highlight is not K.Mode.MOVE_ONLY:
            machine.update(GEST.Event.CYCLE)
        transition = machine.update(GEST.Event.CONFIRM)
        self.assertIs(transition.mode, K.Mode.MOVE_ONLY)
        self.assertTrue(transition.cancel_selection)


class StaleEventTests(unittest.TestCase):
    def test_tracking_coming_back_does_not_replay_an_event(self) -> None:
        """The close that produced this event began while the face was gone.

        Asserted on the HIGHLIGHT, not on the mode. An interrupt already puts
        both back to PAUSED, so a stale CONFIRM commits PAUSED and looks
        harmless; the damage a stale event actually does is move the highlight,
        so that the user's next deliberate confirm lands on a mode they never
        chose. A first version of this test checked the mode and passed with
        the guard deleted.
        """

        machine = K.ControlMachine()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        transition = machine.update(GEST.Event.CYCLE)
        self.assertIs(
            transition.highlight,
            K.Mode.PAUSED,
            "a stale cycle moved the highlight, so the next real confirm arms "
            "a mode the user never selected",
        )
        self.assertIs(machine.update(GEST.Event.CONFIRM).mode, K.Mode.PAUSED)

    def test_a_deliberate_gesture_after_recovery_still_works(self) -> None:
        """The guard must not cost the user their only input channel."""

        machine = K.ControlMachine()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)  # one clean frame
        machine.update(GEST.Event.CYCLE)
        transition = machine.update(GEST.Event.CONFIRM)
        self.assertIs(transition.mode, K.Mode.MOVE_ONLY)
        self.assertTrue(transition.changed)

    def test_a_loss_while_paused_still_clears_the_guard_once(self) -> None:
        machine = K.ControlMachine()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)
        machine.update(GEST.Event.CYCLE)
        self.assertIs(machine.update(GEST.Event.CONFIRM).mode, K.Mode.MOVE_ONLY)


class GestureIndependenceTests(unittest.TestCase):
    def test_no_gaze_position_is_needed_to_change_mode(self) -> None:
        """Mode changes must work when the gaze position is worst.

        ``update`` takes an eyelid event and a tracking flag and nothing else;
        a signature that accepted a gaze point could come to depend on one.
        """

        import inspect  # noqa: PLC0415

        params = set(inspect.signature(K.ControlMachine.update).parameters) - {"self"}
        self.assertEqual(params, {"event", "tracking_ok"})


class ToggleMachineTests(unittest.TestCase):
    """Two states, one gesture.

    The three-mode menu asked the user to remember short-close, short-close,
    long-close and to track a highlight before they could click anything. The
    guards it carried are kept; the choreography is not.
    """

    def test_it_starts_paused_like_the_menu_did(self) -> None:
        machine = K.ToggleMachine()
        self.assertIs(machine.mode, K.Mode.PAUSED)
        self.assertFalse(machine.mode.selection_armed)
        self.assertEqual(machine.label, "PAUSED")

    def test_one_long_close_is_the_whole_sequence(self) -> None:
        machine = K.ToggleMachine()
        transition = machine.update(GEST.Event.CONFIRM)
        self.assertTrue(transition.selection_armed)
        self.assertTrue(transition.changed)
        self.assertEqual(machine.label, "ACTIVE - wink to click")

    def test_a_second_long_close_pauses_again(self) -> None:
        machine = K.ToggleMachine()
        machine.update(GEST.Event.CONFIRM)
        transition = machine.update(GEST.Event.CONFIRM)
        self.assertIs(transition.mode, K.Mode.PAUSED)
        self.assertTrue(transition.cancel_selection)

    def test_a_short_close_does_nothing_at_all(self) -> None:
        """The fewer things the gesture can do, the fewer ways it goes wrong."""

        machine = K.ToggleMachine()
        for _ in range(5):
            transition = machine.update(GEST.Event.CYCLE)
            self.assertIs(transition.mode, K.Mode.PAUSED)
            self.assertFalse(transition.changed)

    def test_short_closes_cannot_arm_it_while_active_either(self) -> None:
        machine = K.ToggleMachine()
        machine.update(GEST.Event.CONFIRM)
        for _ in range(5):
            self.assertFalse(machine.update(GEST.Event.CYCLE).changed)
        self.assertTrue(machine.mode.selection_armed)

    def test_losing_the_face_pauses_and_cancels(self) -> None:
        machine = K.ToggleMachine()
        machine.update(GEST.Event.CONFIRM)
        transition = machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertIs(transition.mode, K.Mode.PAUSED)
        self.assertTrue(transition.cancel_selection)
        self.assertIs(transition.reason, K.Reason.TRACKING_LOST)

    def test_recovery_does_not_replay_a_stale_confirm(self) -> None:
        """The close that made this event began while the face was gone."""

        machine = K.ToggleMachine()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertIs(machine.update(GEST.Event.CONFIRM).mode, K.Mode.PAUSED)

    def test_a_deliberate_close_after_recovery_still_works(self) -> None:
        machine = K.ToggleMachine()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)
        self.assertTrue(machine.update(GEST.Event.CONFIRM).selection_armed)

    def test_the_label_is_the_whole_of_what_the_screen_must_say(self) -> None:
        machine = K.ToggleMachine()
        self.assertNotIn("highlight", machine.label.lower())
        machine.update(GEST.Event.CONFIRM)
        self.assertIn("wink", machine.label.lower())

    def test_it_refuses_to_start_in_a_state_it_does_not_have(self) -> None:
        with self.assertRaises(ValueError):
            K.ToggleMachine(start=K.Mode.MOVE_ONLY)


class SuspendAndResumeTests(unittest.TestCase):
    """Looking away is not a decision.

    A pause nobody asked for, that then has to be undone by hand, costs the
    person their control every time they glance at the keyboard -- and on a
    desktop, where the only way back in is a gesture that may not be
    registering, it can cost them the session. Observed live: the view printed
    "starting ACTIVE" and the screen read PAUSED, because the very first frame
    arrived before the camera did.
    """

    def _active(self) -> K.ToggleMachine:
        return K.ToggleMachine(start=K.ToggleMachine.ACTIVE)

    def test_losing_the_face_suspends_rather_than_pausing(self) -> None:
        machine = self._active()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertIs(machine.mode, K.Mode.PAUSED)
        self.assertTrue(machine.suspended)

    def test_the_face_coming_back_restores_what_was_interrupted(self) -> None:
        machine = self._active()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        transition = machine.update(GEST.Event.NONE)
        self.assertIs(transition.mode, K.ToggleMachine.ACTIVE)
        self.assertIs(transition.reason, K.Reason.TRACKING_RETURNED)
        self.assertFalse(machine.suspended)

    def test_a_pause_the_person_asked_for_is_never_undone_for_them(self) -> None:
        machine = self._active()
        machine.update(GEST.Event.CONFIRM)  # deliberate pause
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)
        self.assertIs(machine.mode, K.Mode.PAUSED, "it re-armed itself after a deliberate pause")

    def test_a_deliberate_choice_while_suspended_outranks_the_restore(self) -> None:
        machine = self._active()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)  # resumes to ACTIVE
        machine.update(GEST.Event.CONFIRM)  # and the person pauses it
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)
        self.assertIs(machine.mode, K.Mode.PAUSED)

    def test_a_profile_change_is_final_and_not_a_suspension(self) -> None:
        """After one the gaze may be mapped through geometry that is gone."""

        machine = self._active()
        machine.interrupt(K.Reason.PROFILE_CHANGED)
        self.assertFalse(machine.suspended)
        machine.update(GEST.Event.NONE)
        self.assertIs(machine.mode, K.Mode.PAUSED)

    def test_entering_calibration_is_final_too(self) -> None:
        machine = self._active()
        machine.interrupt(K.Reason.CALIBRATION_ENTERED)
        machine.update(GEST.Event.NONE)
        self.assertIs(machine.mode, K.Mode.PAUSED)

    def test_a_long_loss_still_resumes_once(self) -> None:
        machine = self._active()
        for _ in range(50):
            machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)
        self.assertIs(machine.mode, K.ToggleMachine.ACTIVE)

    def test_a_suspension_says_so_instead_of_reading_as_a_pause(self) -> None:
        """ "PAUSED" when nobody paused anything sends the person looking for
        a way back in, and there is nothing to find."""

        machine = self._active()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertIn("WAITING", machine.label)
        self.assertNotEqual(machine.label, "PAUSED")

    def test_a_session_that_began_paused_does_not_wake_up_armed(self) -> None:
        machine = K.ToggleMachine()
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE)
        self.assertIs(machine.mode, K.Mode.PAUSED)
        self.assertFalse(machine.mode.selection_armed)
