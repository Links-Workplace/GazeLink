"""The only file in this project that can emit a mouse click.

Kept apart from ``gf_cursor`` deliberately, and the separation is enforced by
a test that reads that file and fails if click words appear in it.  Moving a
pointer is undoable by moving it back; a click is not, and selection at rest
still produced activations as recently as this week.  So the code that can
press a button is one small file with one function that talks to Windows.

What makes a click safe here is not one gate but the pairing of down and up.
A down that is not followed by an up leaves the machine with the left button
held: every subsequent gaze movement becomes a drag, over whatever the user is
looking at, and there is no way to release it by looking.  That is the worst
failure this module can produce and it is worse than not clicking at all --
so the up is guaranteed by ``finally``, re-attempted on release, and a button
still believed held is reported rather than forgotten.

``enabled=False`` is a full simulation: it counts and records every click and
calls nothing.  The two paths differ in one line, so what the tests exercise
is what runs for real.  No test in this project emits a real click.
"""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass
from typing import Any

# --- The Windows call, isolated ---------------------------------------------

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
INPUT_MOUSE = 0


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

    event = _Input(INPUT_MOUSE, _InputUnion(_MouseInput(0, 0, 0, flag, 0, None)))
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
    """Emits left clicks, or pretends to. One click per explicit request."""

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
        self.last_double_gap_ms: float | None = None
        self.refused_not_armed = 0
        self.refused_too_soon = 0
        self.failed = 0
        self.button_stuck = False
        self._down = False
        self._last_s: float | None = None
        self._last_at: tuple[int, int] | None = None

    def __enter__(self) -> ClickAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def release(self) -> None:
        """Make sure no button is left held, whatever happened.

        Runs on normal exit and on exception alike, and suppresses its own
        failure: a shutdown caused by an error must not have that error
        replaced by this one. ``button_stuck`` stays True if the up could not
        be delivered, so the report says so instead of the state being lost.
        """

        if self._down:
            with contextlib.suppress(Exception):
                if self.enabled:
                    self._send(MOUSEEVENTF_LEFTUP)
                self._down = False
                self.button_stuck = False

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
        try:
            if self.enabled:
                self._send(MOUSEEVENTF_LEFTDOWN)
            self._down = True
        except Exception:  # noqa: BLE001 - a failed press is not a press
            self.failed += 1
            self._down = False
            return False
        try:
            if self.enabled:
                self._send(MOUSEEVENTF_LEFTUP)
            self._down = False
        except Exception:  # noqa: BLE001
            # The press landed and the release did not. Recorded, and retried
            # by release(); until then the button really is held.
            self.button_stuck = True
            self.failed += 1
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

    def _press_and_release(self) -> bool:
        """One down/up pair, with the up guaranteed. False if either failed."""

        try:
            if self.enabled:
                self._send(MOUSEEVENTF_LEFTDOWN)
            self._down = True
        except Exception:  # noqa: BLE001 - a failed press is not a press
            self.failed += 1
            self._down = False
            return False
        try:
            if self.enabled:
                self._send(MOUSEEVENTF_LEFTUP)
            self._down = False
        except Exception:  # noqa: BLE001
            self.button_stuck = True
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
            "refused_not_armed": self.refused_not_armed,
            "refused_too_soon": self.refused_too_soon,
            "failed": self.failed,
            "button_stuck": self.button_stuck,
        }
