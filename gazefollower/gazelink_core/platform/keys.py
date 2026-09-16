"""The only file in this project that can press a key.

Written to the same discipline as ``gf_click``, and for the same reason: a
keystroke is OS input, it goes to whatever window happens to be in front, and
some of them cannot be taken back.  So every path that can produce one lives
in one small audited file, and a test reads this file and fails if any pointer
button word appears in it -- the two must not grow into each other.

``enabled=False`` is a full simulation: it counts and records everything and
calls nothing.  The two paths differ in one line, so what the tests exercise
is what runs for real.  No test in this project emits a real keystroke.

What makes typing safe here is two properties, not one.

**Nothing is left held.**  The failure this module can produce that is worse
than typing nothing is a stuck modifier: Alt held down turns every later
keystroke -- including the ones the person makes with their hands -- into a
command, and there is no way to release it by looking.  So every press has a
release guaranteed by ``finally``, re-attempted on ``release()``, and a key
still believed down is reported rather than forgotten.

**The text goes where it was aimed.**  This one was nearly missed.  The
overlay never steals focus, which guarantees only that WE do not take it -- it
says nothing about the original window keeping it.  A notification, an
installer, a background application stealing focus, and the typing would have
continued silently into something else.  So the target is LOCKED when the
keyboard opens: the foreground window is recorded, compared before every
emission, and typing STOPS the moment it differs.  Coming back is explicit --
the person looks at what they want and chooses it again, which is also what
puts the focus back where they meant it.  Nothing resumes on its own.
"""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass
from typing import Any

from gazelink_core.platform import real_input as RI

# --- The Windows call, isolated ---------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_MENU = 0x12  # Alt
VK_ESCAPE = 0x1B
VK_LEFT = 0x25
VK_RIGHT = 0x27

# Keys Windows expects with the extended flag. Left out, a browser's Back
# reads the wrong arrow on some layouts and simply does nothing.
EXTENDED = frozenset({VK_LEFT, VK_RIGHT})


class _KeyInput(ctypes.Structure):
    _fields_ = (
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


# The union has to be the size of the BIGGEST member of the real Win32
# INPUT, not of the one member this file uses. ``SendInput`` is given
# ``sizeof(INPUT)`` and refuses anything that is not exactly right: measured
# here, a union holding only the keyboard member produced a 32-byte INPUT
# against the 40 that 64-bit Windows requires, so every real keystroke would
# have been rejected -- while every test passed, because the tests replace the
# sender. ``gf_click`` gets 40 by accident: the pointer member IS the largest
# one, so a union holding only it happens to be the right size.
#
# Padding rather than the other member, so this file still cannot name a
# pointer button. The size is asserted against ``gf_click``'s structure in the
# tests, because that is the one that has been proven against Windows.
_INPUT_UNION_BYTES = 32


class _InputUnion(ctypes.Union):
    _fields_ = (("ki", _KeyInput), ("_pad", ctypes.c_byte * _INPUT_UNION_BYTES))


class _Input(ctypes.Structure):
    _fields_ = (("type", ctypes.c_ulong), ("union", _InputUnion))


def _send_key(vk: int, scan: int, flags: int) -> None:
    """One keyboard event, exactly as asked.

    No decision of its own: which key, and whether it is a press or a
    release, is settled by the caller. That keeps "what to send" and "how to
    send it" in two places and makes a stray keystroke impossible to blame on
    this one.
    """

    # A key-up is a release and always goes out (a held Alt must be undoable
    # during shutdown); a key-down needs an armed entry point.
    RI.require(f"key {vk:#x}", release=bool(flags & KEYEVENTF_KEYUP))
    event = _Input(INPUT_KEYBOARD, _InputUnion(_KeyInput(vk, scan, flags, 0, None)))
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
    if sent != 1:
        raise OSError(f"SendInput sent {sent} of 1 events")


def foreground_window(user32: Any = None) -> int:
    """Which window has the keyboard right now. Read only; changes nothing.

    There was no call to this anywhere in the repository before typing
    existed, and there did not need to be -- nothing could type. It exists
    now because "the text went into the right window" is a claim that has to
    be checked rather than argued for.
    """

    try:
        api = user32 or ctypes.windll.user32
        return int(api.GetForegroundWindow())
    except Exception:  # noqa: BLE001 - an unreadable target is reported, not raised
        return 0


def window_title(hwnd: int, user32: Any = None) -> str:
    """The target's name, so the person can see what they are typing into.

    A locked target the person cannot identify is not something they agreed
    to. Truncated, because a title can be a whole sentence and this goes on
    one line over their work.
    """

    if not hwnd:
        return "(no window)"
    try:
        api = user32 or ctypes.windll.user32
        buffer = ctypes.create_unicode_buffer(256)
        api.GetWindowTextW(hwnd, buffer, 256)
        text = buffer.value.strip()
    except Exception:  # noqa: BLE001
        return "(unreadable)"
    if not text:
        return "(untitled)"
    return text if len(text) <= 48 else text[:45] + "..."


@dataclass(frozen=True)
class KeyLimits:
    """Bounds a keystroke must satisfy, whatever the caller believes."""

    # The gap inside a chord: the modifier goes down, this long passes, then
    # the key. Short enough to be imperceptible, long enough that the
    # application sees them as a combination and not as two separate keys.
    chord_gap_s: float = 0.02
    # One request can never type more than this many characters. A scanning
    # keyboard sends one at a time, so anything longer is a bug in the caller
    # or a string that arrived from somewhere it should not have.
    max_text_len: int = 64

    def __post_init__(self) -> None:
        if self.chord_gap_s < 0.0:
            raise ValueError("chord_gap_s must not be negative")
        if self.max_text_len < 1:
            raise ValueError("max_text_len must be at least 1")


class KeyAdapter:
    """Emits keystrokes, or pretends to. One keystroke per explicit request.

    The target lock is the safety property here, in the way the down/up
    pairing is the click's. Both are enforced in this file rather than
    trusted to the caller.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        limits: KeyLimits | None = None,
        sender: Any = None,
        sleep: Any = None,
        foreground: Any = None,
        title: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limits = limits or KeyLimits()
        self._send = sender or _send_key
        self._foreground = foreground or foreground_window
        self._title = title or window_title
        import time  # noqa: PLC0415

        self._sleep = sleep or time.sleep
        self.chars_typed = 0
        self.commands_sent = 0
        self.failed = 0
        self.keys_stuck = False
        self.refused_target_changed = 0
        self.refused_no_target = 0
        self.refused_keys_held = 0
        # Every key believed to be DOWN right now, in the order it went down,
        # so release() can undo them in the reverse order they were made.
        self._held: list[int] = []
        self._target: int | None = None
        self._target_name: str = ""
        # Latched the moment the target differs. Cleared only by locking a
        # target again on purpose -- never by the old window coming back,
        # because the person cannot see that it did.
        self.target_lost = False

    def __enter__(self) -> KeyAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    # -- the target lock ----------------------------------------------------

    @property
    def target(self) -> int | None:
        return self._target

    @property
    def target_name(self) -> str:
        return self._target_name

    def lock_target(self) -> int:
        """Record the window the typing is meant for. Returns its handle.

        Called when the keyboard opens, and again whenever the person chooses
        a target after one was lost. Both are deliberate acts, which is the
        whole point: nothing here re-locks on its own.
        """

        self._target = self._foreground()
        self._target_name = self._title(self._target)
        self.target_lost = False
        return self._target

    def clear_target(self) -> None:
        """Forget the target. Nothing may be typed until one is locked again."""

        self._target = None
        self._target_name = ""

    def target_ok(self) -> bool:
        """Is the window that had the keyboard when we locked still in front?

        False also when nothing was ever locked: typing with no recorded
        target is typing into whatever happens to be there, which is the
        failure this exists to prevent, not a lesser version of it.
        """

        if self._target is None:
            return False
        if self.target_lost:
            return False
        return self._foreground() == self._target

    def _may_emit(self) -> bool:
        if self._held:
            # A key is still down from a previous request. Emitting anything
            # on top of it compounds the failure -- a character typed under a
            # held Alt is a command -- so the release is retried first and the
            # request is refused if it cannot be delivered.
            self.release()
            if self._held:
                self.refused_keys_held += 1
                return False
        if self._target is None:
            self.refused_no_target += 1
            return False
        if self.target_lost or self._foreground() != self._target:
            # Latched: one comparison that fails ends the session's typing.
            # Recovering on the next frame because the window came back would
            # mean the person never learns that anything happened.
            self.target_lost = True
            self.refused_target_changed += 1
            return False
        return True

    # -- emission -----------------------------------------------------------

    def release(self) -> None:
        """Make sure no key is left held, whatever happened.

        Runs on normal exit and on exception alike, and suppresses its own
        failure: a shutdown caused by an error must not have that error
        replaced by this one. ``keys_stuck`` stays True if an up could not be
        delivered, so the report says so instead of the state being lost.

        Deliberately NOT gated on the target: a held Alt is held whoever is in
        front, and refusing to release it because focus moved would leave the
        person's real keyboard unusable.
        """

        while self._held:
            vk = self._held[-1]
            try:
                if self.enabled:
                    self._send(vk, 0, self._flags(vk) | KEYEVENTF_KEYUP)
            except Exception:  # noqa: BLE001 - a shutdown must not raise a second error
                # It could not be delivered. Stop rather than spin, and leave
                # the key on the list so the report says a key is still held
                # instead of the state being quietly lost.
                self.keys_stuck = True
                break
            self._held.pop()
        if not self._held:
            self.keys_stuck = False

    @staticmethod
    def _flags(vk: int) -> int:
        return KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0

    def _down(self, vk: int) -> bool:
        try:
            if self.enabled:
                self._send(vk, 0, self._flags(vk))
            self._held.append(vk)
        except Exception:  # noqa: BLE001 - a failed press is not a press
            self.failed += 1
            return False
        return True

    def _up(self, vk: int) -> bool:
        try:
            if self.enabled:
                self._send(vk, 0, self._flags(vk) | KEYEVENTF_KEYUP)
        except Exception:  # noqa: BLE001
            # The press landed and the release did not. The key stays ON the
            # held list on purpose, so release() retries it; removing it here
            # would mean the one thing that can undo it no longer knows.
            self.keys_stuck = True
            self.failed += 1
            return False
        with contextlib.suppress(ValueError):
            self._held.remove(vk)
        return True

    def tap(self, vk: int, *, armed: bool) -> bool:
        """One key down and up. Returns whether it happened.

        ``armed`` is the caller's control mode, passed in rather than read
        from anywhere, for the same reason ``ClickAdapter.click`` takes it.
        """

        if not armed or not self._may_emit():
            return False
        if not self._down(vk):
            return False
        try:
            return self._up(vk)
        finally:
            # Belt and braces: if _up raised past its own handler the key is
            # still recorded as held, and this is the last chance to undo it
            # before the caller moves on to the next request.
            if vk in self._held:
                self.release()

    def chord(self, modifier: int, vk: int, *, armed: bool) -> bool:
        """A modifier held across one key. Returns whether it happened.

        The release of the modifier is guaranteed by ``finally``, including
        when the inner key fails: this is the stuck-Alt case, and it is the
        worst thing this module can do.
        """

        if not armed or not self._may_emit():
            return False
        if not self._down(modifier):
            return False
        inner = False
        try:
            self._sleep(self.limits.chord_gap_s)
            if self._down(vk):
                inner = self._up(vk)
        finally:
            if not self._up(modifier):
                # Retried at once rather than left for shutdown. Until this
                # succeeds the modifier really is down, and the loop would go
                # on accepting requests underneath it.
                self.release()
        # Held keys make this a failure whatever the inner key did: reporting
        # success with Alt still down is how a stuck modifier goes unnoticed
        # until the person tries to use their own keyboard.
        return inner and not self._held

    def type_text(self, text: str, *, armed: bool) -> bool:
        """Type characters into the locked window. Returns whether it happened.

        Sent as UNICODE rather than as virtual keys: a virtual key is
        interpreted through the active keyboard LAYOUT, so an aleph typed that
        way arrives as whatever letter sits on that key in whatever layout the
        person's machine happens to have selected. Unicode carries the
        character itself, so Hebrew arrives as Hebrew with no layout switch
        and nothing for the person to set up.
        """

        if not armed or not self._may_emit():
            return False
        if not text:
            return False
        if len(text) > self.limits.max_text_len:
            raise ValueError(
                f"{len(text)} characters is more than this adapter will send at once "
                f"({self.limits.max_text_len}); a scanning keyboard sends one"
            )
        sent = 0
        # UTF-16 code units, not characters: anything outside the basic plane
        # is two units and both have to go, or half a character arrives.
        units = list(text.encode("utf-16-le"))
        pairs = [units[i] | (units[i + 1] << 8) for i in range(0, len(units), 2)]
        for unit in pairs:
            try:
                if self.enabled:
                    self._send(0, unit, KEYEVENTF_UNICODE)
                    self._send(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)
            except Exception:  # noqa: BLE001 - a failed character is not typed
                self.failed += 1
                return False
            sent += 1
        self.chars_typed += len(text)
        return sent > 0

    # -- the named commands -------------------------------------------------

    def back(self, *, armed: bool) -> bool:
        return self._command(lambda: self.chord(VK_MENU, VK_LEFT, armed=armed))

    def forward(self, *, armed: bool) -> bool:
        return self._command(lambda: self.chord(VK_MENU, VK_RIGHT, armed=armed))

    def switch_window(self, *, armed: bool) -> bool:
        """Alt+Tab.

        Whether Windows honours this from SendInput while a topmost overlay is
        up has NOT been verified on this rig; the session report says what was
        attempted so a live run can answer it.
        """

        return self._command(lambda: self.chord(VK_MENU, VK_TAB, armed=armed))

    def escape(self, *, armed: bool) -> bool:
        return self._command(lambda: self.tap(VK_ESCAPE, armed=armed))

    def enter(self, *, armed: bool) -> bool:
        return self._command(lambda: self.tap(VK_RETURN, armed=armed))

    def backspace(self, *, armed: bool) -> bool:
        return self._command(lambda: self.tap(VK_BACK, armed=armed))

    def _command(self, run: Any) -> bool:
        ok = bool(run())
        if ok:
            self.commands_sent += 1
        return ok

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "chars_typed": self.chars_typed,
            "commands_sent": self.commands_sent,
            "target": self._target_name or "(none locked)",
            "target_lost": self.target_lost,
            "refused_target_changed": self.refused_target_changed,
            "refused_no_target": self.refused_no_target,
            "refused_keys_held": self.refused_keys_held,
            "failed": self.failed,
            "keys_stuck": self.keys_stuck,
            "note": "this adapter cannot press a pointer button; there is no button in it",
        }
