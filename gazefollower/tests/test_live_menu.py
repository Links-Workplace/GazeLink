"""The menu and the keyboard inside the live loop, driven with fakes.

Every piece here was already tested on its own.  What is tested here is the
ASSEMBLY, because that is where this project keeps losing sessions.

The one that matters most is the way back from a pause.  A pause with no way
back is a session lost, and on a desktop there is nowhere to draw a panel and
no keyboard the person can reach.  So: paused, long close, walk the menu,
choose RESUME, active -- with nothing but eyelids.  And its other half, which
is the safety property: on the same route, no other tile emits anything.

No camera, no window, no real OS input: every send, every notch and every
keystroke is recorded by a fake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Sibling test helpers: discovery adds this directory, a standalone run does not.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_click as CK  # noqa: E402
import gf_common as C  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_keyboard as KB  # noqa: E402
import gf_keys as KEYS  # noqa: E402
import gf_live as L  # noqa: E402
import gf_menu as MENU  # noqa: E402
from test_live_loop import ON_SCREEN, _last_mode_line, _profile, _run  # noqa: E402

DEAD = ON_SCREEN
LONG_CLOSE = [(0.0, GEST.Event.CONFIRM)]


def _centres(page: str, *, paused: bool = False) -> dict[str, tuple[float, float]]:
    items = MENU._pages(paused=paused)[page]
    return {b.key: b.centre for b in MENU.page_buttons(items)}


MAIN = _centres("main")
NAV = _centres("nav")


def _gaze(*legs):
    out = []
    for point, reads in legs:
        out.extend([point] * reads)
    return out


class OpeningTests(unittest.TestCase):
    def test_a_long_close_opens_the_menu(self) -> None:
        """The events were already being computed on the camera thread every
        frame and thrown away. Opening the menu with them adds no detector."""

        display, _sends, _runner = _run(
            self,
            points=[DEAD] * 400,
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            max_seconds=0.6,
        )
        self.assertTrue(display.boards, "the long close did nothing")
        self.assertIn("scroll", display.boards[-1])

    def test_nothing_opens_it_without_the_gesture(self) -> None:
        display, _sends, _runner = _run(self, points=[DEAD] * 200, max_seconds=0.3)
        self.assertEqual(display.boards, [])

    def test_a_second_long_close_does_not_shut_it(self) -> None:
        """The gesture has ONE meaning, and that is not a simplification.

        Section 57 measured, twice and independently, that lifting the chin
        shrinks the projected eye area until both eyes read as SHUT with the
        eyes wide open -- median 0.442 and 0.397 of baseline against a gate at
        0.55. The top row of tiles sits at y 0.04-0.24, so looking at a tile
        raises the chin and the detector produces a long close nobody made,
        about 800 ms later.

        As a toggle that closed the menu roughly a second after it opened,
        every time. Reported as "it disappears before I can choose".
        """

        display, _sends, _runner = _run(
            self,
            points=[DEAD] * 600,
            gestures=[(0.0, GEST.Event.CONFIRM), (0.0, GEST.Event.CONFIRM)],
            gesture_after=20,
            max_seconds=0.8,
        )
        self.assertTrue(display.boards, "it never opened")
        self.assertEqual(
            display.order[-1], "board", "a second long close shut the menu"
        )

    def test_a_tile_still_closes_it(self) -> None:
        """The other half: a menu nothing could close would be its own trap."""

        display, _sends, _runner = _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["keyboard"], 300)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=1.0,
        )
        self.assertTrue(display.boards, "it never opened")
        self.assertEqual(display.order[-1], "scan", "choosing a tile left the menu up")

    def test_the_menu_can_be_turned_off(self) -> None:
        display, _sends, _runner = _run(
            self,
            points=[DEAD] * 400,
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_enabled=False,
            max_seconds=0.6,
        )
        self.assertEqual(display.boards, [])

    def test_the_eyes_toggle_and_the_menu_are_refused_together(self) -> None:
        """One gesture cannot mean two things. With both on, every attempt to
        open the menu would also flip PAUSED/ACTIVE."""

        with self.assertRaises(SystemExit) as caught:
            L.run_live(
                _profile(),
                move_cursor=True,
                confirmed=True,
                click_by="wink",
                toggle_by="eyes",
                menu_enabled=True,
            )
        self.assertIn("--no-menu", str(caught.exception))


class ResumeWithoutHandsTests(unittest.TestCase):
    """The property this whole section turns on.

    CLAUDE.md puts the ability to regain control above everything else. A
    pause tile with no resume beside it inverts that.
    """

    # Paused, so the pause tile is the RESUME tile.
    SYSTEM = _centres("system", paused=True)

    def _walk_to_pause(self, **kw):
        """Long close, MORE, MORE, then the pause tile. Nothing but eyelids.

        The gaze returns to the dead centre between tiles because that is what
        re-arms the next page -- a page that went live under a gaze already
        resting on it would choose a tile nobody picked.
        """

        return _run(
            self,
            start_active=False,
            points=_gaze(
                (DEAD, 30),
                (MAIN["more-1"], 40),
                (DEAD, 16),
                (NAV["more-2"], 40),
                (DEAD, 16),
                (self.SYSTEM["pause"], 200),
            ),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=2.0,
            **kw,
        )

    def test_a_paused_session_can_be_resumed_with_the_eyes_alone(self) -> None:
        display, _sends, _runner = self._walk_to_pause()
        self.assertTrue(display.boards, "the menu never opened while paused")
        pages = [set(names) for names in display.boards]
        self.assertTrue(
            any("pause" in names for names in pages),
            f"the walk never reached the pause tile: {pages[-1] if pages else None}",
        )
        self.assertIn(
            "ACTIVE",
            _last_mode_line(display),
            "the session was still paused after choosing RESUME",
        )

    def test_no_other_tile_on_that_route_emits_anything(self) -> None:
        """The other half. Everything but RESUME is refused while paused, so
        walking to it must not have clicked, typed or scrolled on the way."""

        _display, sends, runner = self._walk_to_pause()
        self.assertEqual(sends, [], "something reached Windows while paused")
        self.assertEqual(runner.keystrokes, [], "a keystroke escaped while paused")
        self.assertEqual(runner.wheels, [], "the wheel turned while paused")

    def test_a_command_tile_sends_nothing_while_paused(self) -> None:
        """Chosen directly rather than as a by-product: BACK is a real
        keystroke and PAUSED is the only thing stopping it."""

        _display, _sends, runner = _run(
            self,
            start_active=False,
            points=_gaze(
                (DEAD, 30),
                (MAIN["more-1"], 40),
                (DEAD, 16),
                (NAV["back"], 200),
            ),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=1.6,
        )
        self.assertEqual(runner.keystrokes, [], "BACK was sent from a paused session")

    def test_the_same_command_does_send_when_active(self) -> None:
        """A route that emitted nothing ever would pass the test above."""

        _display, _sends, runner = _run(
            self,
            start_active=True,
            points=_gaze(
                (DEAD, 30),
                (MAIN["more-1"], 40),
                (DEAD, 16),
                (NAV["back"], 200),
            ),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=1.6,
        )
        self.assertTrue(runner.keystrokes, "BACK never reached the key adapter at all")
        self.assertIn(
            KEYS.VK_MENU,
            [vk for vk, _scan, _flags in runner.keystrokes],
            "BACK was sent without its modifier",
        )


# Two deliberate winks at the scanning keyboard: one to open a group, one to
# take a key. Spaced in TIME, because that is the only way they can arrive --
# the detector wants the eye to open and a cooldown to pass between them, and
# the keyboard refuses a cell that has not been on screen long enough to be a
# reaction to. Fired in the same instant they are one closure's jitter, which
# is the thing that must NOT type a letter (asserted in test_keyboard).
_TWO_WINKS = {
    "winks": [(0.0, DEAD), (0.0, DEAD)],
    "wink_after": 120,
    "wink_gap_s": 0.5,
}


class KeyboardModeTests(unittest.TestCase):
    def _into_keyboard(self, **kw):
        return _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["keyboard"], 400)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            **kw,
        )

    def test_choosing_the_keyboard_puts_the_scanning_strip_on_screen(self) -> None:
        display, _sends, _runner = self._into_keyboard(max_seconds=1.0)
        self.assertTrue(display.scans, "the keyboard never opened")
        self.assertIn("פעולות", display.scans[-1], "the actions group is missing")

    def test_two_winks_type_one_character_and_send_no_click(self) -> None:
        """The first wink opens a group; the second takes a key. Neither
        clicks -- that is ``UiMode.KEYBOARD`` in the router, not a hope."""

        _display, sends, runner = self._into_keyboard(
            **_TWO_WINKS, max_seconds=2.0
        )
        self.assertNotIn(CK.MOUSEEVENTF_LEFTDOWN, sends, "a wink clicked at the keyboard")
        self.assertTrue(runner.keystrokes, "two winks typed nothing")
        typed = [scan for vk, scan, _flags in runner.keystrokes if vk == 0]
        self.assertIn(ord(KB.HEBREW[0]), typed, "the character typed was not the highlighted one")

    def test_the_keystrokes_the_loop_sends_are_the_fake_ones(self) -> None:
        """The wheel taught this: the harness patched the click sender and not
        the wheel sender, and a real notch escaped the suite. The keyboard is
        the third entry point and it is patched from its first day."""

        _display, _sends, runner = self._into_keyboard(
            **_TWO_WINKS, max_seconds=2.0
        )
        self.assertTrue(runner.keystrokes)
        for vk, scan, flags in runner.keystrokes:
            self.assertTrue(
                vk != 0 or flags & KEYS.KEYEVENTF_UNICODE,
                f"the recorder saw something that is not a keystroke: {(vk, scan, flags)}",
            )

    def test_the_loop_hands_the_keyboard_the_winks_own_moment(self) -> None:
        """The stamp has been dropped on this path once before (section 60,
        finding 6) and nothing downstream can tell that it was: a wink judged
        by the drain time silently takes whichever cell the sweep had reached.
        Read from the source, the way the pointer-button rule in test_keys is.
        """

        from pathlib import Path  # noqa: PLC0415

        # The wink loop moved into the interaction controller (ARCH-01 stage F).
        source = (
            Path(__file__).resolve().parent.parent
            / "gazelink_core" / "interaction" / "controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "keyboard.select(now, when_s=when)",
            source,
            "the scanning keyboard is being judged by the drain time again",
        )

    def test_typing_stops_when_the_window_underneath_changes(self) -> None:
        """The overlay never steals focus, which only says WE do not take it.
        A notification taking focus would have had the text continue into it."""

        seen = {"n": 0}

        def wandering() -> int:
            # The FIRST call is the lock, when the keyboard opens. Every call
            # after it is a check before a keystroke, and by then the window
            # has changed -- which is exactly a notification stealing focus
            # between opening the keyboard and typing into it.
            seen["n"] += 1
            return 4242 if seen["n"] < 2 else 9999

        _display, _sends, runner = self._into_keyboard(
            **_TWO_WINKS,
            max_seconds=2.0,
            foreground=wandering,
        )
        self.assertEqual(
            runner.keystrokes, [], "text was typed into a window nobody chose"
        )


class WhichEyeTests(unittest.TestCase):
    """The eye picks the button (operator, 24.9.2026).

    RIGHT wink: always a right click, nothing to choose. LEFT wink: a left click,
    single by default; the menu's first tile toggles it to double. The menu no
    longer has a click-type page at all.
    """

    def test_a_right_wink_is_a_right_click_with_nothing_chosen(self) -> None:
        _display, sends, _runner = _run(
            self, winks=[(0.0, DEAD, GEST.Eye.RIGHT)], max_seconds=0.4
        )
        self.assertEqual(
            sends.count(CK.MOUSEEVENTF_RIGHTDOWN), 1, "the right wink did not right click"
        )
        self.assertEqual(sends.count(CK.MOUSEEVENTF_RIGHTUP), 1, "the right button was left down")
        self.assertNotIn(CK.MOUSEEVENTF_LEFTDOWN, sends, "a right wink left clicked")

    def test_a_left_wink_is_a_single_left_click_by_default(self) -> None:
        _display, sends, _runner = _run(
            self, winks=[(0.0, DEAD, GEST.Eye.LEFT)], max_seconds=0.4
        )
        self.assertEqual(
            sends.count(CK.MOUSEEVENTF_LEFTDOWN), 1, "the default stopped being single"
        )
        self.assertEqual(sends.count(CK.MOUSEEVENTF_LEFTUP), 1)
        self.assertNotIn(CK.MOUSEEVENTF_RIGHTDOWN, sends)

    def _toggle_then_wink(self, eye: GEST.Eye) -> list[int]:
        _display, sends, _runner = _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["double-toggle"], 40), (DEAD, 300)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            winks=[(0.0, DEAD, eye)],
            wink_after=200,
            max_seconds=1.8,
        )
        return sends

    def test_the_menu_toggle_makes_a_left_wink_a_double_click(self) -> None:
        sends = self._toggle_then_wink(GEST.Eye.LEFT)
        self.assertEqual(
            sends.count(CK.MOUSEEVENTF_LEFTDOWN), 2, "the toggle did not make it double"
        )
        self.assertEqual(sends.count(CK.MOUSEEVENTF_LEFTDOWN), sends.count(CK.MOUSEEVENTF_LEFTUP))

    def test_the_toggle_leaves_the_right_wink_alone(self) -> None:
        sends = self._toggle_then_wink(GEST.Eye.RIGHT)
        self.assertEqual(sends.count(CK.MOUSEEVENTF_RIGHTDOWN), 1)
        self.assertNotIn(CK.MOUSEEVENTF_LEFTDOWN, sends)

    def test_there_is_no_click_type_page_any_more(self) -> None:
        self.assertNotIn("click", MENU._pages(paused=False))
        self.assertIn("double-toggle", MAIN)


class WinkCancelReachesTheDetectorTests(unittest.TestCase):
    """The fake runner can only show the QUEUE being dropped.

    The other half -- a closure already accumulating when the mode changes --
    lives in the detector, on the camera thread, and is asserted here against
    the real ``LiveRunner``.
    """

    def _runner(self) -> L.LiveRunner:
        rig = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
        return L.LiveRunner(None, None, rig, None, head_builder=lambda _f: None)

    def _frame(self, runner: L.LiveRunner) -> None:
        runner.on_frame(
            SimpleNamespace(status=True, left_eye_openness=100.0, right_eye_openness=100.0),
            SimpleNamespace(status=False),
        )

    def test_a_cancel_reaches_the_detector_on_the_next_frame(self) -> None:
        """Applied on the CAMERA thread, before the detector is updated.

        The eyes are open in this fixture, so the reopen latch that the cancel
        sets is consumed by that same frame -- which is the point of it -- and
        what it leaves behind is the cooldown. That is the observable
        difference between "the cancel arrived" and "it never did".
        """

        runner = self._runner()
        runner.cancel_wink()
        self.assertTrue(runner._cancel_wink_requested)
        self._frame(runner)
        self.assertFalse(runner._cancel_wink_requested, "the request was never applied")
        self.assertIsNotNone(
            runner.wink_detector._cooldown_until_s,
            "the detector was never told to throw away what was in flight",
        )

    def test_it_reports_how_many_had_already_fired(self) -> None:
        runner = self._runner()
        runner.wink_events.append((1.0, (0.5, 0.5)))
        runner.wink_events.append((1.1, (0.5, 0.5)))
        self.assertEqual(runner.cancel_wink(), 2)
        self.assertEqual(runner.wink_events, [])

    def test_a_frame_with_no_cancel_leaves_the_detector_alone(self) -> None:
        """The other half: a detector that always looked cancelled would pass
        the test above without the wiring existing at all."""

        runner = self._runner()
        self._frame(runner)
        self.assertIsNone(runner.wink_detector._cooldown_until_s)
        self.assertFalse(runner.wink_detector._await_reopen)


class ModeEdgeTests(unittest.TestCase):
    def test_a_wink_while_the_menu_is_open_clicks_nothing(self) -> None:
        """The tiles are chosen by DWELL. A wink that also clicked would fire
        two mechanisms from one gesture, which this project has done once."""

        _display, sends, runner = _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["keyboard"], 400)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            winks=[(0.0, DEAD)],
            wink_after=26,
            max_seconds=1.2,
        )
        self.assertTrue(runner.cancels, "no mode change happened, so this proves nothing")
        self.assertNotIn(
            CK.MOUSEEVENTF_LEFTDOWN, sends, "a wink clicked while the menu was open"
        )

    def test_every_change_of_mode_drops_what_was_in_flight(self) -> None:
        """The wiring, asserted where it can be seen: the loop calls the
        runner's cancel on each change, and the runner drops the queue."""

        _display, _sends, runner = _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["keyboard"], 400)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=1.2,
        )
        # Into the menu, and then into the keyboard.
        self.assertGreaterEqual(runner.cancels, 2, "a mode change did not clear the queue")


class PausedEmitsNothingAtAllTests(unittest.TestCase):
    """Choosing a MODE in the menu does not go through the router -- a mode is
    internal state, not an action -- so anything a mode change DOES has to be
    safe on its own.  Entering scroll mode puts the pointer back on the
    content, and ``jump_to`` deliberately ignores ``cursor.paused``.
    """

    def test_choosing_scroll_while_paused_does_not_move_the_real_pointer(self) -> None:
        _display, _sends, runner = _run(
            self,
            start_active=False,
            points=_gaze((DEAD, 30), (MAIN["scroll"], 300)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=1.2,
        )
        moved = [m for m in runner.pointer_moves if m != (0, 0)]
        self.assertEqual(moved, [], "the pointer moved during a paused session")

    def test_the_same_choice_does_move_it_when_active(self) -> None:
        """The other half: a pointer that never moved would pass the above."""

        _display, _sends, runner = _run(
            self,
            start_active=True,
            points=_gaze((DEAD, 30), (MAIN["scroll"], 300)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            max_seconds=1.2,
        )
        moved = [m for m in runner.pointer_moves if m != (0, 0)]
        self.assertTrue(moved, "the pointer never went back to the content")


class StaleWinkTests(unittest.TestCase):
    """A wink carries the moment it FIRED, not the moment the loop reached it.

    Stamped with ``now`` instead, every wink was eternally fresh and the
    router's staleness rule could never fire on the one path that most needs
    it -- the one that goes through a queue.
    """

    def test_a_wink_that_arrived_too_late_does_not_click(self) -> None:
        _display, sends, _runner = _run(
            self, winks=[(0.0, DEAD)], wink_age_s=5.0, max_seconds=0.4
        )
        self.assertEqual(sends, [], "a five-second-old wink clicked")

    def test_a_fresh_one_still_does(self) -> None:
        _display, sends, _runner = _run(self, winks=[(0.0, DEAD)], max_seconds=0.4)
        self.assertIn(CK.MOUSEEVENTF_LEFTDOWN, sends)

    def test_a_stale_wink_does_not_move_the_scanning_keyboard_either(self) -> None:
        """Refused only in the router, this would already have moved the scan
        into a group nobody chose: ``select`` changes state as it is called."""

        display, _sends, runner = _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["keyboard"], 400)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            winks=[(0.0, DEAD), (0.0, DEAD)],
            wink_after=120,
            wink_age_s=5.0,
            max_seconds=1.4,
        )
        self.assertTrue(display.scans, "the keyboard never opened")
        self.assertEqual(runner.keystrokes, [], "a stale wink typed a character")
        self.assertEqual(
            display.scans[-1],
            [g.label for g in KB.layout("hebrew")],
            "a stale wink descended into a group",
        )


class KeyboardTargetLossTests(unittest.TestCase):
    """The recovery this got wrong the first time.

    Offering "look at the target and wink" inside keyboard mode was a promise
    the mode cannot keep: the pointer is frozen and no click is sent, so a
    wink could only re-lock whatever had STOLEN the focus.  It leaves the mode
    instead, where a click can actually focus the window the person wants.
    """

    def _wandering(self):
        seen = {"n": 0}

        def front() -> int:
            # The FIRST call is the lock, when the keyboard opens. Every call
            # after it is a check before a keystroke, and by then the window
            # has changed.
            seen["n"] += 1
            return 4242 if seen["n"] < 2 else 9999

        return front

    def _run_it(self):
        return _run(
            self,
            points=_gaze((DEAD, 30), (MAIN["keyboard"], 400)),
            gestures=list(LONG_CLOSE),
            gesture_after=20,
            menu_dwell_ms=40.0,
            **_TWO_WINKS,
            max_seconds=2.0,
            foreground=self._wandering(),
        )

    def test_a_lost_target_closes_the_keyboard(self) -> None:
        display, _sends, runner = self._run_it()
        self.assertTrue(display.scans, "the keyboard never opened")
        self.assertEqual(runner.keystrokes, [], "text went into the window that stole focus")
        self.assertEqual(
            display.order[-1], "live", "it stayed in a mode it could not recover from"
        )

    def test_the_person_is_told_why_it_closed(self) -> None:
        display, _sends, _runner = self._run_it()
        self.assertTrue(
            any("changed" in line for line in display.huds[-1]),
            f"nothing on screen said why the keyboard closed: {display.huds[-1]}",
        )


class MidFrameCancelTests(unittest.TestCase):
    """The race the flag alone did not close.

    A mode change can call ``cancel_wink`` after the camera thread has read
    the cancel flag for this frame and before that frame appends its wink. The
    queue is cleared, and then the old frame puts a wink from the mode the
    person has just left straight back into it.

    Reproduced deterministically by cancelling from inside the rig's
    ``cm_to_norm``, which the frame calls (for the library's raw estimate)
    after detecting the wink and before publishing it. It used to be the head
    builder; since ARCH-01 stage E the head pose is taken by the tracking
    adapter BEFORE the frame is processed, so that hook no longer sits inside
    the window.
    """

    # The library names its eyes after the IMAGE, so its "left" is the
    # person's RIGHT. A right wink is therefore a small left_eye_openness.
    OPEN = (200.0, 200.0)
    WINK = (20.0, 200.0)

    def _runner(self, hook=None, on_frame_number: int = 0) -> L.LiveRunner:
        clock = {"t": 100.0}
        calls = {"n": 0}

        def tick() -> float:
            return clock["t"]

        class HookedRig(C.RigGeometry):
            def cm_to_norm(self, x_cm: float, y_cm: float):  # noqa: ANN201
                # Called once per frame, BETWEEN the wink being detected and
                # the wink being published -- exactly the window the race
                # lives in. Fired on one chosen frame only: cancelling on
                # every frame would keep the detector permanently reset, so no
                # wink would ever reach the publishing step.
                calls["n"] += 1
                if hook is not None and calls["n"] == on_frame_number:
                    hook()
                return super().cm_to_norm(x_cm, y_cm)

        rig = HookedRig(60.0, 63.6, 120.0, 33.75, 5120, 1440)
        runner = L.LiveRunner(None, None, rig, None, clock=tick, head_builder=lambda _f: None)
        runner._clock_state = clock
        return runner

    def _feed(self, runner: L.LiveRunner, frames, step: float = 0.05) -> None:
        for left, right in frames:
            runner.on_frame(
                SimpleNamespace(
                    status=True, left_eye_openness=left, right_eye_openness=right
                ),
                # A gaze result with no features: nothing is predicted, but
                # the library's raw estimate is converted every frame, which
                # is the hook's call site.
                SimpleNamespace(status=True, features=None, raw_gaze_coordinates=(1.0, 1.0)),
            )
            runner._clock_state["t"] += step

    def test_a_wink_is_published_when_nothing_cancels(self) -> None:
        """The fixture first: without this the test below proves nothing."""

        runner = self._runner()
        self._feed(runner, [self.OPEN, self.WINK, self.WINK, self.WINK])
        self.assertEqual(len(runner.wink_events), 1, "the fixture never produced a wink")

    def test_a_cancel_arriving_mid_frame_drops_that_frame_s_wink(self) -> None:
        holder = {}
        # Frame 3 is the one the wink fires on: frame 2 starts the hold and
        # frame 3 is 50 ms later, past the 35 ms the rule asks for.
        runner = self._runner(hook=lambda: holder["r"].cancel_wink(), on_frame_number=3)
        holder["r"] = runner
        self._feed(runner, [self.OPEN, self.WINK, self.WINK, self.WINK])
        self.assertEqual(
            runner.wink_events, [], "a wink from the abandoned mode was published anyway"
        )
        self.assertTrue(
            runner.winks_dropped_mid_frame, "nothing was actually dropped mid-frame"
        )


if __name__ == "__main__":
    unittest.main()
