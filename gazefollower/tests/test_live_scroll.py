"""Scroll mode inside the live loop: the transitions, not the arithmetic.

The arithmetic is in ``test_scroll.py``.  What is here is the four things that
can only go wrong once the pieces are wired together, and every one of them
would be invisible to a unit test of either piece alone:

* a wink while scrolling must not click, because ``selection_armed`` is still
  true -- the mode never left ACTIVE;
* entering scroll mode must put the pointer back on the CONTENT, not leave it
  on the tile the gaze had to travel to;
* one rest on the tile is one change of mode, not a flip-flop for as long as
  the gaze stays there;
* a wink already in the queue when scroll mode changes must not arrive at the
  click path afterwards.

No camera, no model, no real OS input: every send is recorded by a fake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_click as CK  # noqa: E402
import gf_screen_check as SC  # noqa: E402
import gf_scroll as SCR  # noqa: E402
from test_live_loop import DESKTOP, MONITOR, ON_SCREEN, _run  # noqa: E402

TILE = SCR.scroll_tile().centre
UP_BAND = SCR.scroll_zones()[0].centre
DOWN_BAND = SCR.scroll_zones()[1].centre
CONTENT = ON_SCREEN


def _gaze(*legs):
    """A gaze path: (point, how many reads to stay there), laid end to end."""

    out = []
    for point, reads in legs:
        out.extend([point] * reads)
    return out


class EnteringScrollModeTests(unittest.TestCase):
    def test_the_pointer_goes_back_to_the_content_not_the_tile(self) -> None:
        """The trap this whole design turns on.

        To look at the tile, the gaze travels to the tile -- and the pointer
        follows it there, because following the gaze is what it does. Freezing
        it where it happens to be would freeze it ON the tile, and a wheel
        event goes to whatever is under the pointer, so the page would never
        move. What has to be remembered is where the pointer was while the
        gaze was still on CONTENT.
        """

        display, _sends, runner = _run(
            self,
            points=_gaze((CONTENT, 40), (TILE, 400)),
            scroll_toggle_ms=60.0,
            max_seconds=0.6,
        )
        self.assertTrue(
            any("SCROLL UP" in names for names in display.zones),
            "scroll mode never started, so this proves nothing",
        )
        content_px = SC.to_desktop_pixels(CONTENT, MONITOR, DESKTOP)
        tile_px = SC.to_desktop_pixels(TILE, MONITOR, DESKTOP)
        self.assertNotEqual(content_px, tile_px, "the fixture cannot tell the two apart")
        moved = [m for m in runner.pointer_moves if m != (0, 0)]
        self.assertEqual(
            moved[-1],
            content_px,
            "the pointer was left on the tile, so the wheel would scroll the wrong thing",
        )

    def test_resting_on_the_tile_changes_the_mode_once(self) -> None:
        """Not in and straight back out while the gaze is still on it."""

        display, _sends, _runner = _run(
            self,
            points=_gaze((CONTENT, 20), (TILE, 600)),
            scroll_toggle_ms=60.0,
            max_seconds=0.8,
        )
        with_bands = [names for names in display.zones if "SCROLL UP" in names]
        self.assertTrue(with_bands, "it never entered scroll mode")
        # Once in, it stays in: the gaze never leaves the tile, and the dwell
        # engine latches until the point is seen somewhere else.
        self.assertIn("SCROLL UP", display.zones[-1], "it flipped back out on its own")

    def test_the_tile_is_on_screen_before_scrolling_starts(self) -> None:
        """There has to be a visible way in."""

        display, _sends, _runner = _run(self, max_seconds=0.2)
        self.assertTrue(display.zones, "nothing was drawn at all")
        self.assertIn(SCR.TILE, display.zones[-1])
        self.assertNotIn("SCROLL UP", display.zones[-1], "the bands were shown outside scroll mode")


class WinksDoNotClickWhileScrollingTests(unittest.TestCase):
    """The safety property. The mode is still ACTIVE while scrolling, so
    ``selection_armed`` is still true and nothing else would stop the click."""

    def test_a_wink_during_scroll_mode_sends_no_click(self) -> None:
        _display, sends, _runner = _run(
            self,
            points=_gaze((CONTENT, 20), (TILE, 200), (UP_BAND, 400)),
            winks=[(0.0, CONTENT)],
            wink_after=120,
            scroll_toggle_ms=60.0,
            scroll_arm_ms=40.0,
            max_seconds=0.8,
        )
        self.assertNotIn(
            CK.MOUSEEVENTF_LEFTDOWN, sends, "a wink double clicked in the middle of a scroll"
        )

    def test_the_same_wink_outside_scroll_mode_does_click(self) -> None:
        """The other half: blocking everything would also pass the first."""

        _display, sends, _runner = _run(self, winks=[(0.0, CONTENT)], max_seconds=0.4)
        self.assertIn(CK.MOUSEEVENTF_LEFTDOWN, sends, "the ordinary click path stopped working")


class LeavingScrollModeTests(unittest.TestCase):
    def test_a_lost_face_cancels_scroll_mode_and_does_not_resume_it(self) -> None:
        """A pause or a lost face ends it: input that REPEATS has to be asked
        for again, deliberately. Section 48 restores the pointer, not this."""

        display, _sends, _runner = _run(
            self,
            points=_gaze((CONTENT, 20), (TILE, 120), (CONTENT, 400)),
            blind_after=200,
            scroll_toggle_ms=60.0,
            max_seconds=0.8,
        )
        self.assertTrue(
            any("SCROLL UP" in names for names in display.zones),
            "scroll mode never started, so losing it proves nothing",
        )
        self.assertNotIn(
            "SCROLL UP",
            display.zones[-1],
            "scroll mode survived the face going away",
        )


class NoRealInputEscapesTests(unittest.TestCase):
    """Found by checking rather than by assuming.

    The harness patched the click entry point and not the WHEEL one, so the
    suite sent a real notch of +120 to Windows. ``gf_click`` opens with "no
    test in this project emits a real click", and a second entry point that
    nobody patched made that untrue the moment scrolling existed.
    """

    def test_the_wheel_the_loop_sends_is_the_fake_one(self) -> None:
        """The adapter calls exactly one sender, so a notch arriving at the
        recorder is proof the real one was not the one that got it."""

        _display, _sends, runner = _run(
            self,
            points=_gaze((CONTENT, 20), (TILE, 200), (UP_BAND, 400)),
            scroll_toggle_ms=60.0,
            scroll_arm_ms=40.0,
            max_seconds=0.8,
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

        _d, _s, up = _run(
            self,
            points=_gaze((CONTENT, 20), (TILE, 200), (UP_BAND, 400)),
            scroll_toggle_ms=60.0,
            scroll_arm_ms=40.0,
            max_seconds=0.8,
        )
        _d2, _s2, down = _run(
            self,
            points=_gaze((CONTENT, 20), (TILE, 200), (DOWN_BAND, 400)),
            scroll_toggle_ms=60.0,
            scroll_arm_ms=40.0,
            max_seconds=0.8,
        )
        self.assertTrue(up.wheels and down.wheels, "one of the bands never scrolled")
        self.assertTrue(all(w > 0 for w in up.wheels), f"the top band scrolled {up.wheels}")
        self.assertTrue(all(w < 0 for w in down.wheels), f"the bottom band scrolled {down.wheels}")


class StartScrollingTests(unittest.TestCase):
    """--start-scrolling has no tile, so nothing may switch it off.

    ``state_face_ok`` is false for the first frames of EVERY session, before
    the camera has delivered anything -- that is the "WAITING FOR YOUR FACE"
    line every log opens with. Cancelling scroll mode on a lost face therefore
    switched it off a moment after it started, and with no tile there was no
    way back: measured, 0 of 187 frames drew the bands. Reported as "I cannot
    see the scroll bands at all".
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

    def test_there_is_no_tile_in_this_mode(self) -> None:
        """It sat in the middle, where the gaze rests to STOP."""

        display, _sends, _runner = _run(
            self, points=[TILE] * 300, start_scrolling=True, max_seconds=0.4
        )
        self.assertTrue(display.zones)
        self.assertNotIn(SCR.TILE, display.zones[-1])
        self.assertIn("SCROLL UP", display.zones[-1], "resting where the tile was ended scrolling")


if __name__ == "__main__":
    unittest.main()
