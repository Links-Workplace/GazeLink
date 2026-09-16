"""A window you can see through and click through, and a stop that always works.

The practice screen and the live view both open a FULLSCREEN pygame window, so
a real click landed on that window and never on the application underneath.
That is fine for a practice target drawn by the window itself, and it is not
desktop control however real the click is -- which is exactly the confusion
this module exists to end.

Two things are needed to click on someone else's window:

``see_through``   the overlay must not be a grey sheet over the desktop. One
                  colour is declared transparent, the window is painted in it,
                  and only what is drawn on top is visible.
``click_through`` the overlay must not be the thing that gets clicked. Windows
                  decides that per window with ``WS_EX_TRANSPARENT``: hit
                  testing falls through to whatever is behind.

And one consequence that has to be handled rather than discovered: a
click-through window never takes focus, so keyboard events never reach it and
``pygame``'s Esc stops arriving. The emergency stop cannot depend on the thing
it is meant to interrupt, so it is read from the keyboard state directly,
which works whoever has focus.

Windows only. On anything else the calls are skipped and the caller is told,
rather than a transparent window being claimed and not delivered.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import Any

# Chosen because nothing in this overlay draws in it: the colour key makes
# every pixel of this exact value invisible, so a dot that happened to be this
# colour would be a hole in itself.
TRANSPARENT_KEY = (255, 0, 255)

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
LWA_COLORKEY = 0x00000001
VK_ESCAPE = 0x1B
VK_SPACE = 0x20


@dataclass(frozen=True)
class OverlayState:
    """What was actually achieved, rather than what was requested."""

    see_through: bool
    click_through: bool
    reason: str

    @property
    def usable_over_other_windows(self) -> bool:
        """Both, or the overlay is a sheet over the desktop and not an overlay."""

        return self.see_through and self.click_through


def is_windows() -> bool:
    return sys.platform == "win32"


def make_click_through(hwnd: int, *, user32: Any = None) -> OverlayState:
    """Make one window transparent to sight and to the mouse.

    Returns what succeeded. A partial result is reported as such and never as
    success: a window that is invisible but still swallows clicks is worse
    than an opaque one, because the clicks disappear with nothing to show for
    it.
    """

    if not is_windows():
        return OverlayState(False, False, f"not Windows ({sys.platform})")
    api = user32 or ctypes.windll.user32
    try:
        style = api.GetWindowLongW(hwnd, GWL_EXSTYLE)
        api.SetWindowLongW(
            hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
        )
        key = TRANSPARENT_KEY[0] | (TRANSPARENT_KEY[1] << 8) | (TRANSPARENT_KEY[2] << 16)
        ok = api.SetLayeredWindowAttributes(hwnd, key, 255, LWA_COLORKEY)
    except Exception as exc:  # noqa: BLE001 - report, never crash the session
        return OverlayState(False, False, f"the window styles were refused: {exc}")
    if not ok:
        return OverlayState(False, True, "the colour key was refused, so the overlay is opaque")
    return OverlayState(True, True, "layered, colour-keyed and transparent to the mouse")


def key_is_down(vk: int, user32: Any = None) -> bool:
    """Is this key held right now, whoever has keyboard focus?

    A click-through window never takes focus, so pygame's event queue is empty
    however hard a key is pressed. Anything that must work while the overlay
    is up has to ask the keyboard directly.
    """

    if not is_windows():
        return False
    api = user32 or ctypes.windll.user32
    try:
        # The high bit is "down now". The low bit is "pressed since last
        # asked", which fires once and then goes quiet, so a caller polling it
        # would see a key press or not depending on its own frame rate.
        return bool(api.GetAsyncKeyState(vk) & 0x8000)
    except Exception:  # noqa: BLE001
        return False


def escape_is_down(user32: Any = None) -> bool:
    """Is Esc held right now, whoever has keyboard focus?

    A click-through window never takes focus, so pygame's event queue never
    sees a key press and the ordinary Esc stops working the moment the overlay
    starts working. An emergency stop that only functions while the dangerous
    mode is off is not an emergency stop, so this asks the keyboard directly.
    """

    if not is_windows():
        return False
    api = user32 or ctypes.windll.user32
    try:
        # The high bit is "down now", as opposed to the low bit, which is
        # "pressed since last asked" and would fire once and then go quiet.
        return bool(api.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
    except Exception:  # noqa: BLE001
        return False
