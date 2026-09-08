"""Intentional eye-close gesture: a way to act without hands.

Self-contained on purpose.  The product has a gesture state machine of its own
(`src/gazelink/recovery_gesture.py`) and this deliberately does not import it:
the two trees are kept separate, and that one is written against a different
eye signal.  What is borrowed is the *design* -- short close cycles, long close
confirms, absence aborts -- not the code and not its numbers.

Every threshold here was measured on this rig with `gf_gesture_probe.py`
rather than guessed, because the guessed ones did not survive contact:

    natural blinking   3 closures, longest   297 ms
    deliberate holds   4 closures, shortest 1172 ms, median 2156 ms

The inherited 1500 ms confirm would have fired on none of those holds.  The
inherited "no bridging" would have seen 27 fragments instead of 4 holds.

Two rules carry the safety of this, and both are tested:

* **Absence is never bridged.**  Bridging forgives an openness estimate that
  flickers mid-hold.  It must never forgive a lost face, or looking away --
  or standing up and leaving -- would keep accumulating toward a confirm with
  nobody at the desk.
* **The harmless option is selected first.**  A gesture built on an
  imperfect signal will sometimes fire when it should not, so the thing it
  does by default has to be the thing that costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Event(StrEnum):
    NONE = "none"
    CYCLE = "cycle"  # a deliberate close, released before the confirm time
    CONFIRM = "confirm"  # a deliberate close held long enough to mean it


@dataclass(frozen=True)
class GestureConfig:
    """Durations in milliseconds. Defaults are measured, not assumed.

    ``natural_blink_max_ms`` sits above the longest natural blink observed
    (297 ms) with margin; anything shorter is ignored entirely.
    ``confirm_ms`` sits below the shortest deliberate hold observed (1172 ms)
    and well above the blink ceiling, so it can be reached on purpose and not
    by accident.
    """

    bridge_ms: float = 100.0
    natural_blink_max_ms: float = 400.0
    confirm_ms: float = 800.0
    cooldown_ms: float = 800.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.bridge_ms <= 500.0:
            raise ValueError("bridge_ms must be between 0 and 500")
        if self.confirm_ms <= self.natural_blink_max_ms:
            raise ValueError(
                f"confirm_ms ({self.confirm_ms}) must exceed natural_blink_max_ms "
                f"({self.natural_blink_max_ms}), or ordinary blinking would confirm"
            )
        if self.cooldown_ms < 0.0:
            raise ValueError("cooldown_ms must not be negative")


class EyeCloseDetector:
    """Turns per-frame (face present, eyes shut) into deliberate events.

    Fed from the same signal the probe measured, so the thresholds mean here
    exactly what they meant there.  ``update`` is called once per frame with
    monotonic seconds and returns at most one event.
    """

    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self._start_s: float | None = None
        self._last_shut_s: float | None = None
        # Latched after a confirm and cleared only by actually seeing the eyes
        # open again. A purely time-based cooldown re-arms while they are
        # still shut, so one long hold fires repeatedly -- measured: a
        # five-second hold produced three confirms before this existed.
        self._await_reopen = False
        self._cooldown_until_s: float | None = None

    def reset(self) -> None:
        self._start_s = None
        self._last_shut_s = None

    @property
    def holding_ms(self) -> float:
        """How long the current hold has run, for drawing progress."""

        if self._start_s is None or self._last_shut_s is None:
            return 0.0
        return (self._last_shut_s - self._start_s) * 1000.0

    def update(self, now_s: float, *, face_present: bool, eyes_shut: bool) -> Event:
        cfg = self.config

        # Absence ends the hold at once and is never bridged. The reopen latch
        # deliberately SURVIVES absence: if the face goes away after a confirm
        # and comes back with the eyes still shut, that is not evidence of a
        # fresh intention, and treating it as one would let a confirm repeat.
        if not face_present:
            self.reset()
            return Event.NONE

        if eyes_shut:
            if self._await_reopen:
                return Event.NONE
            if self._cooldown_until_s is not None and now_s < self._cooldown_until_s:
                return Event.NONE
            self._cooldown_until_s = None
            if self._start_s is None:
                self._start_s = now_s
            self._last_shut_s = now_s
            if self.holding_ms >= cfg.confirm_ms:
                # Fired while the eyes are still shut: the person needs to know
                # it took effect before they open them, or they cannot tell a
                # confirm from a miss.
                self._await_reopen = True
                self.reset()
                return Event.CONFIRM
            return Event.NONE

        # Eyes open, face present.
        if self._await_reopen:
            self._await_reopen = False
            self._cooldown_until_s = now_s + cfg.cooldown_ms / 1000.0
            self.reset()
            return Event.NONE
        if self._start_s is None or self._last_shut_s is None:
            return Event.NONE
        gap_ms = (now_s - self._last_shut_s) * 1000.0
        if gap_ms <= cfg.bridge_ms:
            # A flicker in the openness estimate, not a real reopening.
            return Event.NONE
        held_ms = self.holding_ms
        self.reset()
        if held_ms > cfg.natural_blink_max_ms:
            self._cooldown_until_s = now_s + cfg.cooldown_ms / 1000.0
            return Event.CYCLE
        return Event.NONE


@dataclass
class MenuOption:
    key: str
    label: str


class RecoveryMenu:
    """A short list where the first entry costs nothing.

    Opened by a cycle rather than shown always, so it cannot be confirmed
    without a deliberate act first.  It closes itself after
    ``timeout_s`` of no gesture, so an accidental opening does not sit on
    screen waiting to be confirmed by the next stray blink.
    """

    def __init__(self, options: list[MenuOption], *, timeout_s: float = 6.0) -> None:
        if not options:
            raise ValueError("a menu needs at least one option")
        self.options = options
        self.timeout_s = timeout_s
        self.open = False
        self.index = 0
        self._last_activity_s: float | None = None

    @property
    def selected(self) -> MenuOption:
        return self.options[self.index]

    def tick(self, now_s: float) -> None:
        """Close an idle menu, so nothing waits around to be confirmed."""

        idle = self._last_activity_s is not None and now_s - self._last_activity_s > self.timeout_s
        if self.open and idle:
            self.close()

    def close(self) -> None:
        self.open = False
        self.index = 0
        self._last_activity_s = None

    def handle(self, event: Event, now_s: float) -> MenuOption | None:
        """Apply one gesture event; returns the option if one was taken."""

        if event is Event.NONE:
            self.tick(now_s)
            return None
        self._last_activity_s = now_s
        if not self.open:
            # The first deliberate close only opens the menu. A confirm that
            # arrives with nothing on screen must not select anything.
            self.open = True
            self.index = 0
            return None
        if event is Event.CYCLE:
            self.index = (self.index + 1) % len(self.options)
            return None
        chosen = self.selected
        self.close()
        return chosen
