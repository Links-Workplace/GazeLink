"""The click-through overlay. A fake user32; no window is ever created."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_overlay as OV  # noqa: E402


class _User32:
    """Records the calls and can be told which of them fail."""

    def __init__(self, *, style: int = 0, layered_ok: int = 1, raises: bool = False) -> None:
        self.style = style
        self.layered_ok = layered_ok
        self.raises = raises
        self.set_style: int | None = None
        self.key: int | None = None
        self.async_state = 0

    def GetWindowLongW(self, hwnd: int, index: int) -> int:  # noqa: N802
        if self.raises:
            raise OSError("refused")
        return self.style

    def SetWindowLongW(self, hwnd: int, index: int, value: int) -> int:  # noqa: N802
        self.set_style = value
        return value

    def SetLayeredWindowAttributes(  # noqa: N802
        self, hwnd: int, key: int, alpha: int, flags: int
    ) -> int:
        self.key = key
        return self.layered_ok

    def GetAsyncKeyState(self, key: int) -> int:  # noqa: N802
        return self.async_state


class ClickThroughTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real = OV.is_windows
        OV.is_windows = lambda: True
        self.addCleanup(lambda: setattr(OV, "is_windows", self._real))

    def test_both_styles_are_applied(self) -> None:
        """Layered makes it see-through; transparent makes the mouse miss it."""

        api = _User32()
        state = OV.make_click_through(1, user32=api)
        self.assertTrue(state.usable_over_other_windows)
        self.assertTrue(api.set_style & OV.WS_EX_LAYERED)
        self.assertTrue(api.set_style & OV.WS_EX_TRANSPARENT)

    def test_existing_styles_are_kept_rather_than_replaced(self) -> None:
        api = _User32(style=0x1234)
        OV.make_click_through(1, user32=api)
        self.assertEqual(api.set_style & 0x1234, 0x1234)

    def test_a_window_that_swallows_clicks_is_never_reported_as_usable(self) -> None:
        """Invisible but still clickable is worse than opaque: the clicks
        disappear with nothing on screen to show for it."""

        api = _User32(layered_ok=0)
        state = OV.make_click_through(1, user32=api)
        self.assertFalse(state.see_through)
        self.assertFalse(state.usable_over_other_windows)
        self.assertIn("opaque", state.reason)

    def test_a_refusal_is_reported_and_never_raised(self) -> None:
        state = OV.make_click_through(1, user32=_User32(raises=True))
        self.assertFalse(state.usable_over_other_windows)
        self.assertIn("refused", state.reason)

    def test_the_colour_key_matches_what_the_overlay_paints(self) -> None:
        api = _User32()
        OV.make_click_through(1, user32=api)
        r, g, b = OV.TRANSPARENT_KEY
        self.assertEqual(api.key, r | (g << 8) | (b << 16))


class NonWindowsTests(unittest.TestCase):
    def test_it_says_so_instead_of_claiming_a_transparent_window(self) -> None:
        real = OV.is_windows
        OV.is_windows = lambda: False
        self.addCleanup(lambda: setattr(OV, "is_windows", real))
        state = OV.make_click_through(1, user32=_User32())
        self.assertFalse(state.usable_over_other_windows)
        self.assertIn("not Windows", state.reason)

    def test_the_stop_never_reports_a_press_it_cannot_read(self) -> None:
        real = OV.is_windows
        OV.is_windows = lambda: False
        self.addCleanup(lambda: setattr(OV, "is_windows", real))
        self.assertFalse(OV.escape_is_down(_User32()))


class EmergencyStopTests(unittest.TestCase):
    """A click-through window never takes focus, so pygame stops seeing keys
    the moment the overlay starts working. A stop that only functions while
    the dangerous mode is off is not a stop."""

    def setUp(self) -> None:
        real = OV.is_windows
        OV.is_windows = lambda: True
        self.addCleanup(lambda: setattr(OV, "is_windows", real))

    def test_a_held_key_is_down(self) -> None:
        api = _User32()
        api.async_state = 0x8000
        self.assertTrue(OV.escape_is_down(api))

    def test_a_key_pressed_since_last_asked_is_not_a_key_held_now(self) -> None:
        """The low bit fires once and then goes quiet, so reading it would
        make the stop depend on when it was last polled."""

        api = _User32()
        api.async_state = 0x0001
        self.assertFalse(OV.escape_is_down(api))

    def test_nothing_pressed_is_not_a_stop(self) -> None:
        self.assertFalse(OV.escape_is_down(_User32()))

    def test_a_failure_to_read_the_keyboard_is_not_a_stop(self) -> None:
        """Stopping on an unrelated error would end a session for no reason."""

        class Broken:
            def GetAsyncKeyState(self, key: int) -> int:  # noqa: N802
                raise OSError("no keyboard")

        self.assertFalse(OV.escape_is_down(Broken()))


class KeyReadingTests(unittest.TestCase):
    """Anything that must work while the overlay is up has to ask the keyboard
    directly: a click-through window never takes focus, so pygame's event
    queue stays empty however hard a key is pressed."""

    def setUp(self) -> None:
        real = OV.is_windows
        OV.is_windows = lambda: True
        self.addCleanup(lambda: setattr(OV, "is_windows", real))

    def test_space_and_escape_are_read_from_the_same_place(self) -> None:
        api = _User32()
        api.async_state = 0x8000
        self.assertTrue(OV.key_is_down(OV.VK_SPACE, api))
        self.assertTrue(OV.escape_is_down(api))

    def test_a_key_that_is_not_held_is_not_down(self) -> None:
        self.assertFalse(OV.key_is_down(OV.VK_SPACE, _User32()))

    def test_the_pressed_since_last_asked_bit_is_not_held_now(self) -> None:
        """Reading it would make the answer depend on the caller's frame rate."""

        api = _User32()
        api.async_state = 0x0001
        self.assertFalse(OV.key_is_down(OV.VK_SPACE, api))
