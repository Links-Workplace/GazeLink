"""Scroll mode inside the live loop: the transitions, not the arithmetic.

The arithmetic is in ``test_scroll.py``.  What is here is what can only go
wrong once the pieces are wired together, and every one of them would be
invisible to a unit test of either piece alone:

* a wink while scrolling must not click, because ``selection_armed`` is still
  true -- the mode never left ACTIVE;
* entering scroll mode must leave the pointer on the CONTENT, not wherever the
  gaze travelled to ask for it;
* a wink already in the queue when the mode changes must not arrive at the
  click path afterwards;
* ``--start-scrolling`` must survive the blindness every session opens with.

The way IN changed in section 60.  There is no SCROLL tile any more -- the
operator's verdict on it was that it was hard to reach, which is where
``--start-scrolling`` came from -- so scroll mode is entered from the menu a
long eye-close opens.  These tests drive that route, because it is the route
the person now has.

No camera, no model, no real OS input: every send is recorded by a fake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_click as CK  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_menu as MENU  # noqa: E402
import gf_screen_check as SC  # noqa: E402
import gf_scroll as SCR  # noqa: E402
from test_live_loop import DESKTOP, MONITOR, ON_SCREEN, _run  # noqa: E402

UP_BAND = SCR.scroll_zones()[0].centre
DOWN_BAND = SCR.scroll_zones()[1].centre
# The middle of the screen: content, and also the menu's dead zone, which is
# what the gaze has to be seen in before any tile becomes selectable.
CONTENT = ON_SCREEN

_MAIN = {b.key: b.centre for b in MENU.page_buttons(MENU._pages(paused=False)["main"])}
MENU_SCROLL = _MAIN["scroll"]
MENU_KEYBOARD = _MAIN["keyboard"]

# One long close, delivered once the gaze has settled on content.
LONG_CLOSE = [(0.0, GEST.Event.CONFIRM)]


def _gaze(*legs):
    """A gaze path: (point, how many reads to stay there), laid end to end."""

    out = []
    for point, reads in legs:
        out.extend([point] * reads)
    return out


def _into_scroll(test, *, after=20, dwell=60.0, then=(), **kw):
    """Open the menu with a long close and choose the scroll tile.

    ``after`` reads of content first, so the pointer has followed the gaze to
    somewhere that counts as content before anything else happens -- that is
    the position entering scroll mode has to put it back to.
    """

    # Off the tile again once it has been chosen. The scroll tile of the menu
    # sits in the TOP row, which overlaps the top scroll band -- leaving the
    # gaze there would scroll up before the test had asked for anything.
    legs = [(CONTENT, after + 6), (MENU_SCROLL, 40), (CONTENT, 20), *then]
    return _run(
        test,
        points=_gaze(*legs),
        gestures=list(LONG_CLOSE),
        gesture_after=after,
        menu_dwell_ms=dwell,
        **kw,
    )


class EnteringScrollModeTests(unittest.TestCase):
    def test_a_long_close_and_a_tile_get_into_scroll_mode(self) -> None:
        display, _sends, _runner = _into_scroll(self, max_seconds=0.8)
        self.assertTrue(display.boards, "the menu never opened")
        self.assertTrue(
            any("SCROLL UP" in names for names in display.zones),
            "the scroll tile was chosen and scroll mode never started",
        )

    def test_the_pointer_is_left_on_the_content_not_on_the_menu(self) -> None:
        """The trap this design turns on, and it has not gone away.

        A wheel event goes to whatever is under the pointer. To CHOOSE the
        scroll tile the gaze travels to the tile, and if the pointer followed
        it there the wheel would scroll the menu's corner of the screen and
        the page would never move.

        Two things stop it now and both are asserted: the pointer is frozen
        for the whole time the menu is open, so it never travels at all, and
        entering scroll mode puts it back on the remembered content position
        explicitly.
        """

        _display, _sends, runner = _into_scroll(self, max_seconds=0.8)
        content_px = SC.to_desktop_pixels(CONTENT, MONITOR, DESKTOP)
        tile_px = SC.to_desktop_pixels(MENU_SCROLL, MONITOR, DESKTOP)
        self.assertNotEqual(content_px, tile_px, "the fixture cannot tell the two apart")
        moved = [m for m in runner.pointer_moves if m != (0, 0)]
        self.assertTrue(moved, "the pointer never moved at all")
        self.assertNotIn(
            tile_px, moved, "the pointer followed the gaze onto the menu tile"
        )
        self.assertEqual(
            moved[-1],
            content_px,
            "the pointer was not put back on the content, so the wheel would go elsewhere",
        )

    def test_the_bands_are_not_drawn_before_scroll_mode_starts(self) -> None:
        """There is nothing on screen until the person asks for it."""

        display, _sends, _runner = _run(self, max_seconds=0.2)
        self.assertTrue(display.zones, "nothing was drawn at all")
        self.assertNotIn("SCROLL UP", display.zones[-1], "the bands were shown unasked")
        self.assertEqual(display.boards, [], "the menu opened with no gesture")


class WinksDoNotClickWhileScrollingTests(unittest.TestCase):
    """The safety property. The mode is still ACTIVE while scrolling, so
    ``selection_armed`` is still true and nothing else would stop the click."""

    def test_a_wink_during_scroll_mode_sends_no_click(self) -> None:
        _display, sends, _runner = _into_scroll(
            self,
            then=((UP_BAND, 400),),
            winks=[(0.0, CONTENT)],
            wink_after=260,
            scroll_arm_ms=40.0,
            max_seconds=1.0,
        )
        self.assertNotIn(
            CK.MOUSEEVENTF_LEFTDOWN, sends, "a wink clicked in the middle of a scroll"
        )

    def test_the_same_wink_outside_scroll_mode_does_click(self) -> None:
        """The other half: blocking everything would also pass the first."""

        _display, sends, _runner = _run(self, winks=[(0.0, CONTENT)], max_seconds=0.4)
        self.assertIn(CK.MOUSEEVENTF_LEFTDOWN, sends, "the ordinary click path stopped working")


class LeavingScrollModeTests(unittest.TestCase):
    def test_a_lost_face_stops_the_wheel_without_ending_the_mode(self) -> None:
        """The wheel stops; the mode does not.

        Cancelling the mode on a lost face was a trap: ``state_face_ok`` is
        false for the first frames of every session, before the camera has
        delivered anything. With no tile to ask again with, that left the
        person in a mode they could not re-enter -- 0 of 187 frames drew the
        bands. So absence stops the repeat and nothing else, and the way back
        out is the menu, which a long close opens whatever else is happening.
        """

        display, _sends, runner = _into_scroll(
            self,
            then=((UP_BAND, 600),),
            blind_after=300,
            scroll_arm_ms=40.0,
            scroll_repeat_ms=40.0,
            max_seconds=1.2,
        )
        self.assertTrue(
            any("SCROLL UP" in names for names in display.zones),
            "scroll mode never started, so this proves nothing",
        )
        self.assertIn(
            "SCROLL UP",
            display.zones[-1],
            "the mode was cancelled by a lost face, leaving no way back in",
        )
        del runner


class NoRealInputEscapesTests(unittest.TestCase):
    """Found by checking rather than by assuming.

    The harness patched the click entry point and not the WHEEL one, so the
    suite sent a real notch of +120 to Windows. ``gf_click`` opens with "no
    test in this project emits a real click", and a second entry point that
    nobody patched made that untrue the moment scrolling existed. There is now
    a third -- the keyboard -- and it was patched from the first day.
    """

    def test_the_wheel_the_loop_sends_is_the_fake_one(self) -> None:
        """The adapter calls exactly one sender, so a notch arriving at the
        recorder is proof the real one was not the one that got it."""

        _display, _sends, runner = _into_scroll(
            self,
            then=((UP_BAND, 400),),
            scroll_arm_ms=40.0,
            max_seconds=1.0,
        )
        self.assertTrue(runner.wheels, "nothing scrolled, so this proves nothing")
        self.assertTrue(
            all(abs(w) % 120 == 0 for w in runner.wheels),
            f"the recorder saw something that is not a wheel notch: {runner.wheels}",
        )

    def test_looking_up_scrolls_up_and_down_scrolls_down(self) -> None:
        """Positive mouseData is the wheel turned away from the person, which
        Windows reads as scrolling UP. Getting the sign backwards would be
        invisible in every other test here."""

        _d, _s, up = _into_scroll(
            self, then=((UP_BAND, 400),), scroll_arm_ms=40.0, max_seconds=1.0
        )
        _d2, _s2, down = _into_scroll(
            self, then=((DOWN_BAND, 400),), scroll_arm_ms=40.0, max_seconds=1.0
        )
        self.assertTrue(up.wheels and down.wheels, "one of the bands never scrolled")
        self.assertTrue(all(w > 0 for w in up.wheels), f"the top band scrolled {up.wheels}")
        self.assertTrue(all(w < 0 for w in down.wheels), f"the bottom band scrolled {down.wheels}")


class StartScrollingTests(unittest.TestCase):
    """--start-scrolling opens straight into the bands, past the menu.

    ``state_face_ok`` is false for the first frames of EVERY session, before
    the camera has delivered anything -- that is the "WAITING FOR YOUR FACE"
    line every log opens with. Cancelling scroll mode on a lost face therefore
    switched it off a moment after it started: measured, 0 of 187 frames drew
    the bands. Reported as "I cannot see the scroll bands at all".
    """

    def test_the_bands_survive_the_blindness_every_session_starts_with(self) -> None:
        display, _sends, runner = _run(
            self,
            points=[UP_BAND] * 600,
            blind_reads=60,
            start_scrolling=True,
            scroll_arm_ms=40.0,
            scroll_repeat_ms=60.0,
            max_seconds=1.0,
        )
        drew = [names for names in display.zones if "SCROLL UP" in names]
        self.assertTrue(drew, "the bands were never drawn at all")
        self.assertEqual(len(drew), len(display.zones), "the bands came and went")
        self.assertTrue(runner.wheels, "nothing scrolled once the face arrived")

    def test_there_is_no_tile_anywhere_in_this_mode(self) -> None:
        """It sat in the middle, where the gaze rests to STOP."""

        display, _sends, _runner = _run(
            self, points=[CONTENT] * 300, start_scrolling=True, max_seconds=0.4
        )
        self.assertTrue(display.zones)
        self.assertNotIn(SCR.TILE, display.zones[-1])
        self.assertIn("SCROLL UP", display.zones[-1], "resting in the middle ended scrolling")


if __name__ == "__main__":
    unittest.main()
