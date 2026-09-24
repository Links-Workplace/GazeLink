"""The only file in this project that can press a mouse button or turn its wheel.

Kept apart from ``gf_cursor`` deliberately, and the separation is enforced by
a test that reads that file and fails if click words appear in it.  Moving a
pointer is undoable by moving it back; a click is not, and selection at rest
still produced activations as recently as this week.  So the code that can
press a button is one small file with one function that talks to Windows.

What makes a click safe here is not one gate but the pairing of down and up.
A down that is not followed by an up leaves the machine with a button held:
every subsequent gaze movement becomes a drag, over whatever the user is
looking at, and there is no way to release it by looking.  That is the worst
failure this module can produce and it is worse than not clicking at all --
so the up is guaranteed by ``finally``, re-attempted on release, and a button
still believed held is reported rather than forgotten.

There are two buttons now.  Which one is held is recorded rather than assumed,
because the release has to send the matching up: a right button released with
the left flag is a right button still held, and it would look exactly like a
successful release from here.

``enabled=False`` is a full simulation: it counts and records every click and
calls nothing.  The two paths differ in one line, so what the tests exercise
is what runs for real.  No test in this project emits a real click.

The wheel lives here too, in ``ScrollAdapter``, for the same reason the click
does: it is OS input, and this project keeps every path that can produce OS
input in one audited file rather than letting a second one appear quietly.  It
has none of the down/up risk -- a wheel notch is a single event with nothing
left held -- but it has a risk the click does not: it REPEATS.  So the rate
limit is the safety property here, and it is enforced in this file rather than
trusted to the caller.
"""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass
from typing import Any

from gazelink_core.platform import real_input as RI

# --- The Windows call, isolated ---------------------------------------------

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
# What Windows counts as one detent of the wheel. Everything is a multiple of
# it; a smaller number is a partial notch and applications may ignore it.
WHEEL_DELTA = 120
INPUT_MOUSE = 0

# Down and up, paired by name. The pairing is the safety property of this
# module, so it is written down once here rather than being re-derived at
# each call site -- a right button released with the LEFT up flag is a right
# button still held, and nothing on screen would say so.
BUTTONS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
}
_RELEASE_FLAGS = frozenset(up for _down, up in BUTTONS.values())


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("mi", _MouseInput),)


class _Input(ctypes.Structure):
    _fields_ = (("type", ctypes.c_ulong), ("union", _InputUnion))


def _send(flag: int) -> None:
    """One mouse event at the pointer's current position.

    No coordinates: this module never moves the pointer.  Where the click
    lands is decided by whoever moved it, which keeps the two decisions in two
    files and makes a stray click impossible to blame on this one.
    """

    # A button-up is a release and always goes out; everything else needs an
    # entry point that armed real input (gazelink_core.platform.real_input).
    RI.require(f"mouse flag {flag:#x}", release=flag in _RELEASE_FLAGS)
    event = _Input(INPUT_MOUSE, _InputUnion(_MouseInput(0, 0, 0, flag, 0, None)))
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
    if sent != 1:
        raise OSError(f"SendInput sent {sent} of 1 events")


def _send_wheel(delta: int) -> None:
    """Turn the wheel by ``delta``: positive scrolls up, negative down.

    Same shape as ``_send`` and the same silence about position: no
    coordinates, so where it lands is decided by whoever moved the pointer.

    ``mouseData`` is declared ``c_ulong``, so a negative delta is masked to its
    two's-complement pattern EXPLICITLY here rather than left to ctypes to
    coerce. The coercion happens to produce the right bits today; writing it
    down means a scroll-down cannot quietly become a scroll-up if that changes.
    """

    RI.require("mouse wheel")
    data = delta & 0xFFFFFFFF
    event = _Input(INPUT_MOUSE, _InputUnion(_MouseInput(0, 0, data, MOUSEEVENTF_WHEEL, 0, None)))
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
    if sent != 1:
        raise OSError(f"SendInput sent {sent} of 1 events")


def system_double_click_ms(user32: Any = None) -> int:
    """What Windows itself counts as a double click, in ms.

    Read rather than assumed: it is a user setting, the default is 500 ms, and
    a pair that misses it by 20 ms opens nothing while looking identical to
    the person who made it.
    """

    try:
        api = user32 or ctypes.windll.user32
        value = int(api.GetDoubleClickTime())
    except Exception:  # noqa: BLE001 - a missing setting is not a failed click
        return 500
    return value if value > 0 else 500


@dataclass
class ClickLimits:
    """Bounds a click must satisfy, whatever the caller believes."""

    # Two clicks closer together than this, at DIFFERENT places, are one
    # gesture misfiring twice rather than two intentions.
    min_gap_s: float = 0.4
    # A second click this soon after the first, at the SAME pixel, is a double
    # click and is what the operator asked for: it is what opens most things.
    # Deliberately allowed past ``min_gap_s`` -- the guard exists to catch a
    # gesture firing twice by accident, and an accident does not land on the
    # same pixel while the pointer is being held still for exactly this.
    double_window_s: float = 0.6
    # How far apart two clicks may be and still count as the same place.
    same_place_px: int = 8
    # The gap inside a double click WE send. Long enough that Windows reads
    # two clicks rather than one slow press, short enough to be nowhere near
    # the 500 ms window -- so it cannot fail on a slow frame or a busy machine.
    double_gap_s: float = 0.06

    def __post_init__(self) -> None:
        if self.min_gap_s < 0.0:
            raise ValueError("min_gap_s must not be negative")
        if self.double_window_s <= 0.0:
            raise ValueError("double_window_s must be positive")
        if self.same_place_px < 0:
            raise ValueError("same_place_px must not be negative")
        if not 0.0 < self.double_gap_s < self.double_window_s:
            raise ValueError("double_gap_s must be positive and inside the double window")


class ClickAdapter:
    """Emits clicks, or pretends to. One click per explicit request."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        limits: ClickLimits | None = None,
        sender: Any = None,
        clock: Any = None,
        sleep: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limits = limits or ClickLimits()
        self._send = sender or _send
        import time  # noqa: PLC0415

        self._now = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.clicks = 0
        self.doubles = 0
        # Counted apart from ``clicks`` on purpose. A right click opens a
        # context menu rather than activating anything, so "how many times did
        # the system commit to something" and "how many times did it offer a
        # choice" are different questions and one number cannot answer both.
        self.right_clicks = 0
        self.last_double_gap_ms: float | None = None
        self.refused_not_armed = 0
        self.refused_too_soon = 0
        self.refused_button_held = 0
        self.failed = 0
        # WHICH button is still held, not merely that one is. Written as a
        # boolean this said "a button is stuck" and left the reader to guess
        # which, and the two are undone by different flags.
        self.stuck_button: str | None = None
        self._held: str | None = None
        # Held ON PURPOSE, which is a different thing from ``_held``. Without
        # this distinction ``_press_and_release`` reads an intentional drag as
        # a stuck button and releases it mid-gesture, because "a button is
        # down" was the whole vocabulary.
        self._dragging: bool = False
        self.drags = 0
        self.refused_while_dragging = 0
        self._last_s: float | None = None
        self._last_at: tuple[int, int] | None = None

    def __enter__(self) -> ClickAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    @property
    def button_stuck(self) -> bool:
        """Is any button still believed held? ``stuck_button`` says which."""

        return self.stuck_button is not None

    def release(self) -> None:
        """Make sure no button is left held, whatever happened.

        Runs on normal exit and on exception alike, and suppresses its own
        failure: a shutdown caused by an error must not have that error
        replaced by this one. ``stuck_button`` keeps naming the button if the
        up could not be delivered, so the report says so instead of the state
        being lost.

        It releases whichever button is held, not the left one. Written for a
        single button this would leave a held RIGHT button held for ever --
        the same failure this module opens by calling the worst thing it can
        produce, arrived at by adding a second button and not its release.
        """

        if self._held is not None:
            with contextlib.suppress(Exception):
                if self.enabled:
                    self._send(BUTTONS[self._held][1])
                self._held = None
                self.stuck_button = None
                self._dragging = False

    def click(self, *, armed: bool, at: tuple[int, int] | None = None) -> bool:
        """Emit one left click. Returns whether it happened.

        ``armed`` is the caller's control mode, passed in rather than read
        from anywhere: a click module that can look up its own permission is
        a click module that can be wrong about it alone.

        ``at`` is where the pointer is, used only to tell a deliberate double
        click from a gesture firing twice. Nothing here moves the pointer.
        """

        if not armed:
            self.refused_not_armed += 1
            return False
        now = self._now()
        if self._last_s is not None and (now - self._last_s) < self.limits.min_gap_s:
            gap = now - self._last_s
            if self._is_double(now, at, gap):
                # Allowed through on purpose: a second click at the same pixel
                # inside the double window is the gesture that opens things,
                # and refusing it here is what made a double click impossible.
                self.doubles += 1
                self.last_double_gap_ms = gap * 1000.0
            else:
                self.refused_too_soon += 1
                return False
        if not self._press_and_release("left"):
            return False
        self.clicks += 1
        self._last_s = now
        self._last_at = at
        return True

    def double_click(self, *, armed: bool, at: tuple[int, int] | None = None) -> bool:
        """Emit a double click as ONE action. Returns whether it happened.

        Two clicks the person makes separately have to clear a wink's hold, a
        reopening and a cooldown between them, and then still land inside the
        system's double-click window -- measured at 500 ms here. That is a
        timing test the person can fail through no fault of their own, and
        failing it silently produces two ordinary clicks that open nothing.

        Sent from here instead, the gap is ours to choose: short enough that
        Windows always reads a double, long enough that it is two clicks and
        not one long press. What the person has to do is unchanged, which is
        the point -- one wink.
        """

        if not armed:
            self.refused_not_armed += 1
            return False
        now = self._now()
        if self._last_s is not None and (now - self._last_s) < self.limits.min_gap_s:
            self.refused_too_soon += 1
            return False
        if not self._press_and_release():
            return False
        self._sleep(self.limits.double_gap_s)
        if not self._press_and_release():
            # The first half landed and the second did not, so the click that
            # reached Windows was a single. Counted as one, because that is
            # what happened, and the failure is already counted too.
            self.clicks += 1
            self._last_s = now
            self._last_at = at
            return False
        self.clicks += 1
        self.doubles += 1
        self.last_double_gap_ms = self.limits.double_gap_s * 1000.0
        self._last_s = now
        self._last_at = at
        return True

    def right_click(self, *, armed: bool, at: tuple[int, int] | None = None) -> bool:
        """Emit one right click. Returns whether it happened.

        Same shape as ``click`` and the same two gates, with one rule left
        out: there is no double. A right click opens a context menu, a second
        one closes it and opens it again, and there is nothing a pair of them
        does that one does not -- so the same-pixel exception that lets a
        double through does not apply here, and the ordinary gap guard stands
        unweakened.
        """

        if not armed:
            self.refused_not_armed += 1
            return False
        now = self._now()
        if self._last_s is not None and (now - self._last_s) < self.limits.min_gap_s:
            self.refused_too_soon += 1
            return False
        if not self._press_and_release("right"):
            return False
        self.right_clicks += 1
        self._last_s = now
        self._last_at = at
        return True

    def press(self, button: str = "left", *, armed: bool) -> bool:
        """Hold a button DOWN and leave it down. The start of a drag.

        Separate from ``click`` because the release is not part of it, and
        because everything that follows must know the button is down on
        purpose. Refused while anything is already held: two drags at once
        cannot both be released by a record that names one button.
        """

        if self._held is not None:
            self.refused_while_dragging += 1
            return False
        if not armed:
            self.refused_not_armed += 1
            return False
        down, _up = BUTTONS[button]
        try:
            if self.enabled:
                self._send(down)
        except Exception:  # noqa: BLE001 - a failed press is not a press
            self.failed += 1
            return False
        self._held = button
        self._dragging = True
        self.drags += 1
        # The gap guard must see the drag, or a click straight after the drop
        # slips under ``min_gap_s`` as though nothing had happened.
        self._last_s = self._now()
        return True

    @property
    def dragging(self) -> bool:
        return self._dragging

    def drag_release(self) -> bool:
        """Let go, deliberately. Never gated on a permission.

        A release is not a new action: it undoes one. ``SendInput``'s own
        interlock already lets button-up through while everything else is
        refused, and this path must behave the same -- a pause, a lost face or
        a shutdown arriving mid-drag has to be able to let go.

        Kept apart from ``release``, which belongs to shutdown: sharing one
        method would make a failed drop and a failed cleanup the same number.
        """

        if self._held is None:
            self._dragging = False
            return False
        button = self._held
        try:
            if self.enabled:
                self._send(BUTTONS[button][1])
        except Exception:  # noqa: BLE001
            self.stuck_button = button
            self.failed += 1
            return False
        self._held = None
        self._dragging = False
        self.stuck_button = None
        self._last_s = self._now()
        return True

    def _press_and_release(self, button: str = "left") -> bool:
        """One down/up pair, with the up guaranteed. False if either failed."""

        if self._dragging:
            # An intentional drag is in progress. Releasing it here to make
            # room for a click would drop whatever is being carried somewhere
            # nobody chose. Refused and counted instead.
            self.refused_while_dragging += 1
            return False
        if self._held is not None:
            # A button is still down from a previous request, and pressing
            # anything now would overwrite the record of WHICH one -- after
            # which even release() could not let go of it, because it would no
            # longer know. Retried first, and refused if the retry fails: a
            # click that does not happen is recoverable, a button held for the
            # rest of the session is not.
            self.release()
            if self._held is not None:
                self.refused_button_held += 1
                return False
        down, up = BUTTONS[button]
        try:
            if self.enabled:
                self._send(down)
            self._held = button
        except Exception:  # noqa: BLE001 - a failed press is not a press
            self.failed += 1
            self._held = None
            return False
        try:
            if self.enabled:
                self._send(up)
            self._held = None
        except Exception:  # noqa: BLE001
            # The press landed and the release did not. Recorded, and retried
            # by release(); until then the button really is held.
            self.stuck_button = button
            self.failed += 1
            return False
        return True

    def _is_double(self, now: float, at: tuple[int, int] | None, gap: float) -> bool:
        """Is this the second half of a deliberate double click?

        Both halves must be true: soon enough, and at the same place. Time
        alone would let a gesture that misfired twice while the gaze moved
        count as a double, which is how a stray pair opens something.
        """

        if gap > self.limits.double_window_s:
            return False
        if at is None or self._last_at is None:
            return False
        reach = self.limits.same_place_px
        return abs(at[0] - self._last_at[0]) <= reach and abs(at[1] - self._last_at[1]) <= reach

    def summary(self) -> dict[str, Any]:
        system_ms = system_double_click_ms()
        return {
            "enabled": self.enabled,
            "clicks": self.clicks,
            "doubles": self.doubles,
            "last_double_gap_ms": self.last_double_gap_ms,
            # Whether Windows itself would have read the last pair as a double
            # click. Ours can be a double by our rule and still land outside
            # the system's window, in which case nothing opens and the person
            # has no way to tell why.
            "system_double_click_ms": system_ms,
            "last_double_within_system_window": (
                None if self.last_double_gap_ms is None else self.last_double_gap_ms <= system_ms
            ),
            "right_clicks": self.right_clicks,
            "refused_not_armed": self.refused_not_armed,
            "refused_too_soon": self.refused_too_soon,
            "refused_button_held": self.refused_button_held,
            # Drag is reported only once there has been one, the way
            # ``ActionRouter.summary`` reports only refusals that happened. A
            # session that never dragged says nothing about dragging, and the
            # line stays what every earlier report printed.
            **(
                {
                    "drags": self.drags,
                    "refused_while_dragging": self.refused_while_dragging,
                    "dragging": self._dragging,
                }
                if (self.drags or self.refused_while_dragging or self._dragging)
                else {}
            ),
            "failed": self.failed,
            "button_stuck": self.button_stuck,
            "stuck_button": self.stuck_button,
        }


# SystemParametersInfo, SPI_GETMOUSEWHEELROUTING.
SPI_GETMOUSEWHEELROUTING = 0x201C
_WHEEL_ROUTING = {
    0: "the FOCUSED window",
    1: "hybrid",
    2: "the window under the POINTER",
}


def wheel_routing(user32: Any = None) -> tuple[int, str]:
    """Where Windows sends a wheel event, read rather than assumed.

    It is a user setting -- "scroll inactive windows when I hover over them"
    -- and it decides whether freezing the pointer over the target is what
    makes a gaze scroll work or merely harmless. Measured on this rig on
    10.9.2026: 2, the window under the pointer. That is why the caller parks
    the pointer on the content before the first notch.

    If it ever reads 0, the pointer no longer decides and the target window
    would have to be FOCUSED instead -- which nothing in this project does,
    and which would be a new kind of OS side effect. Reported rather than
    handled, because a guess about which one is in force is worse than a
    number on the screen.
    """

    try:
        api = user32 or ctypes.windll.user32
        value = ctypes.c_uint(0)
        if not api.SystemParametersInfoW(SPI_GETMOUSEWHEELROUTING, 0, ctypes.byref(value), 0):
            return (-1, "unknown")
        return (value.value, _WHEEL_ROUTING.get(value.value, "unknown"))
    except Exception:  # noqa: BLE001 - an unreadable setting is not a failed scroll
        return (-1, "unknown")


@dataclass
class ScrollLimits:
    """Bounds a scroll must satisfy, whatever the caller believes.

    The click's guard exists to stop one gesture firing twice. This one exists
    because scrolling REPEATS by design, so "how fast, at most" is the whole of
    its safety: a caller stuck in a loop, or a zone the gaze is resting in by
    accident, must not be able to run the page away.
    """

    # No more than one notch this often, whoever asks. Well under the repeat
    # rate the caller uses, so it never bites in normal use and only catches a
    # caller that has lost control of its own clock.
    min_gap_s: float = 0.05
    # A single request can never be more than this many notches, so a bad
    # number arriving from a config or a flag cannot jump a page a screenful
    # at a time.
    max_notches: int = 3

    def __post_init__(self) -> None:
        if self.min_gap_s < 0.0:
            raise ValueError("min_gap_s must not be negative")
        if self.max_notches < 1:
            raise ValueError("max_notches must be at least 1")


class ScrollAdapter:
    """Turns the mouse wheel, or pretends to. One turn per explicit request.

    Where it scrolls is decided by whoever moved the pointer -- this file sends
    no coordinates. Which of the two things that means is now measured rather
    than assumed: ``wheel_routing`` read 2 on this rig, "the window under the
    POINTER", so parking the pointer on the content is not a precaution, it is
    the thing that makes a gaze scroll work at all. The value is a user
    setting, so it is read again and reported in ``summary`` every session.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        limits: ScrollLimits | None = None,
        sender: Any = None,
        clock: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limits = limits or ScrollLimits()
        self._send_wheel = sender or _send_wheel
        import time  # noqa: PLC0415

        self._now = clock or time.monotonic
        self.notches_up = 0
        self.notches_down = 0
        self.refused_not_armed = 0
        self.refused_too_soon = 0
        self.failed = 0
        self._last_s: float | None = None

    def __enter__(self) -> ScrollAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        # Nothing to release: a notch leaves nothing held. Present so this
        # adapter is used exactly like the click one, and a later reader does
        # not have to check which of the two needs cleaning up.
        return None

    def ready(self, *, armed: bool = True) -> bool:
        """Would a notch be accepted right now? Read-only; counts nothing."""

        if not armed:
            return False
        if self._last_s is None:
            return True
        return (self._now() - self._last_s) >= self.limits.min_gap_s

    def scroll(self, notches: int, *, armed: bool) -> bool:
        """Send ``notches`` detents: positive up, negative down.

        Returns whether it happened. ``armed`` is the caller's control mode,
        passed in rather than read from anywhere, for the same reason
        ``click`` takes it.
        """

        if not armed:
            self.refused_not_armed += 1
            return False
        if notches == 0:
            return False
        now = self._now()
        if self._last_s is not None and (now - self._last_s) < self.limits.min_gap_s:
            self.refused_too_soon += 1
            return False
        capped = max(-self.limits.max_notches, min(self.limits.max_notches, int(notches)))
        try:
            if self.enabled:
                self._send_wheel(capped * WHEEL_DELTA)
        except Exception:  # noqa: BLE001 - a failed turn is not a turn
            self.failed += 1
            return False
        if capped > 0:
            self.notches_up += capped
        else:
            self.notches_down += -capped
        self._last_s = now
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "notches_up": self.notches_up,
            "notches_down": self.notches_down,
            "refused_not_armed": self.refused_not_armed,
            "refused_too_soon": self.refused_too_soon,
            "failed": self.failed,
            # Reported every session because it is a user setting: if it reads
            # 0, the pointer no longer decides where a notch lands and this
            # whole design needs revisiting.
            "wheel_goes_to": wheel_routing()[1],
            "note": "this adapter cannot press a button; there is no down or up in it",
        }
