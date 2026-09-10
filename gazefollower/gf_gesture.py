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

import math
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


@dataclass(frozen=True)
class WinkConfig:
    """What separates a deliberate one-eyed close from an ordinary blink.

    Measured on a 100 s run, 6338 frames at 63 fps, while the operator winked
    repeatedly with the right eye:

    * the right eye reached **0.003** of its own baseline, so the closure is
      unmistakable in the signal;
    * the LEFT eye was under the gate on **658** frames against the right's
      331 -- it closes too, every time. Requiring the left to be open was
      therefore requiring something the person cannot do, and worse, both eyes
      under the gate read as a blink and armed a 600 ms veto that killed the
      wink that followed;
    * only **112** frames looked like a clean wink, 1.77 s in total across the
      whole run, so a single wink lasts on the order of 150 ms. A 500 ms hold
      could never be reached.

    So the test is ASYMMETRY, not closure: in a blink both eyes go together,
    in a wink one goes far further than the other. Duration then only has to
    exclude a momentary flicker, not carry the decision on its own.
    """

    # The right eye must be at least this closed, against its own baseline.
    shut_ratio: float = 0.45
    # ...and the left must be at least this many times more open than it.
    # A blink drives both to a similar depth and fails here; a wink does not.
    asymmetry: float = 2.5
    # Long enough to exclude a single noisy frame, short enough to be reached
    # by a real wink: about nine frames at 63 fps.
    # The detector runs on the CAMERA thread, at about 30 fps, so this is
    # counted in frames whether it is written in them or not. At 140 ms it
    # asked for 4.2 unbroken frames, and three measured sessions matched 3, 4
    # and 9 display-loop samples -- fewer still in camera frames, because the
    # display samples the same frame more than once. The winks were deep
    # enough (right to 0.14-0.17) and asymmetric enough (left 4-6x that), and
    # simply too SHORT. At 70 ms it asks for about two frames.
    #
    # Duration is not what keeps a blink out: the asymmetry rule does that, and
    # a blink drives both eyes together however long it lasts. This only has to
    # exclude a single-frame flicker, and two frames does.
    # 50 ms, not 70: the first matching frame only STARTS the clock, so the
    # hold is reached on the frame after ``hold_ms`` has elapsed. At 30 fps
    # that makes the real requirement ceil(hold/33) + 1 frames -- 70 ms asks
    # for four, and the measured winks were three. 50 ms asks for three.
    hold_ms: float = 50.0
    # A gap shorter than this does not end a wink. The eye-close detector has
    # had this since it was written -- "a flicker in the openness estimate, not
    # a real reopening" -- and the wink detector did not, so one noisy frame
    # restarted the hold from zero. Measured over a two-minute session: 103
    # frames matched the wink rule and only TWO winks fired, because the runs
    # kept being broken and started again.
    bridge_ms: float = 40.0
    # Short on purpose. A wink already cannot fire twice from one closure --
    # ``_await_reopen`` requires the eye to open again first -- so this is only
    # a guard against jitter around the reopening, and it does not have to be
    # long. At 700 ms it WAS long: the soonest a second wink could fire was
    # about reopen + 700 + 140 = 900 ms after the first, and Windows counts a
    # double click only within 500 ms. Two deliberate winks could therefore
    # never open anything, which is exactly what was reported.
    cooldown_ms: float = 120.0

    def __post_init__(self) -> None:
        if not 0.0 < self.shut_ratio < 1.0:
            raise ValueError("shut_ratio must be between 0 and 1")
        if self.asymmetry <= 1.0:
            raise ValueError("asymmetry must be greater than 1, or a blink qualifies")
        for name in ("hold_ms", "cooldown_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not 0.0 <= self.bridge_ms < self.hold_ms:
            raise ValueError(
                "bridge_ms must be at least 0 and under hold_ms, or a wink could be "
                "bridged for longer than it has to be held and never end"
            )


class RightWinkDetector:
    """Right eye closing much further than the left, held on purpose.

    Judged on each eye's ratio to its own baseline rather than on a shut/open
    decision, because the operator's left eye narrows whenever the right one
    closes. A rule that needed the left OPEN rejected every real wink; a rule
    that compares the two depths does not.

    Fires while the eye is still shut, like the confirm, so the person learns
    it took effect before they open it.
    """

    def __init__(self, config: WinkConfig | None = None) -> None:
        self.config = config or WinkConfig()
        self._start_s: float | None = None
        self._await_reopen = False
        self._cooldown_until_s: float | None = None
        self._hold_ms = 0.0
        self._last_match_s: float | None = None

    def reset(self) -> None:
        self._start_s = None
        self._last_match_s = None

    @property
    def holding_ms(self) -> float:
        return self._hold_ms

    def looks_like_a_wink(self, left_ratio: float, right_ratio: float) -> bool:
        """The whole test, exposed so it can be measured without the timing.

        Both ratios are the PERSON's eyes, not the image's. Callers translate
        with :func:`eyes_as_the_person_has_them` before they get here.
        """

        cfg = self.config
        if not (math.isfinite(left_ratio) and math.isfinite(right_ratio)):
            return False
        if right_ratio >= cfg.shut_ratio:
            return False
        # Guard the division: a right eye at exactly zero is as asymmetric as
        # it gets, provided the left is not there with it.
        if right_ratio <= 0.0:
            return left_ratio > cfg.shut_ratio
        return left_ratio >= right_ratio * cfg.asymmetry

    def update(
        self,
        now_s: float,
        *,
        face_present: bool,
        left_ratio: float,
        right_ratio: float,
    ) -> bool:
        """One frame. True exactly once per deliberate wink."""

        cfg = self.config
        if not face_present:
            self.reset()
            self._await_reopen = False
            self._hold_ms = 0.0
            return False
        if not self.looks_like_a_wink(left_ratio, right_ratio):
            bridged = (
                self._start_s is not None
                and self._last_match_s is not None
                and (now_s - self._last_match_s) * 1000.0 <= cfg.bridge_ms
                and not self._await_reopen
            )
            if bridged:
                # A flicker in the estimate, not the eye opening. The hold
                # keeps running; without this a single noisy frame sent it
                # back to zero and the wink never reached its own threshold.
                return False
            self._start_s = None
            self._hold_ms = 0.0
            if self._await_reopen:
                self._await_reopen = False
                self._cooldown_until_s = now_s + cfg.cooldown_ms / 1000.0
            return False
        self._last_match_s = now_s
        if self._await_reopen:
            return False
        if self._cooldown_until_s is not None and now_s < self._cooldown_until_s:
            return False
        self._cooldown_until_s = None
        if self._start_s is None:
            self._start_s = now_s
            return False
        self._hold_ms = (now_s - self._start_s) * 1000.0
        if self._hold_ms >= cfg.hold_ms:
            self._await_reopen = True
            self.reset()
            return True
        return False


def eyes_as_the_person_has_them(library_left: float, library_right: float) -> tuple[float, float]:
    """Translate the library's eye labels into the person's own left and right.

    The camera faces the person, so the eye on the LEFT of the image is the
    person's RIGHT eye. The library names its fields after the image -- its
    ``left_eye_openness`` is built from MediaPipe landmarks 33/133, the
    image-left eye -- and this project then wrote a right-wink detector that
    watched the field named "right", which is the person's LEFT eye.

    Confirmed twice over: the operator winked their right eye throughout a
    100 s run and the library's LEFT field was under the gate on 658 frames
    against the right's 331, and the operator said so directly.

    Returns ``(person_left, person_right)``. One place, so a mirror bug cannot
    be introduced in a second one.
    """

    return library_right, library_left


@dataclass(frozen=True)
class OpennessGateConfig:
    """How far an eye must close, relative to how open it usually is."""

    # Measured live on 9.9 over 627 frames: left openness ran a median of
    # 191.5 and a minimum of 27.0, right 160.0 and 64.5. The absolute
    # threshold is 10.0, so NOTHING reached it and no eyelid gesture could
    # fire. As a fraction of each eye's own baseline those minima are 0.14 and
    # 0.40, which is a signal -- just not one an absolute number can read.
    shut_fraction: float = 0.55
    # Above this, the eye is open enough for the gaze estimate to be worth
    # trusting. BELOW it and above ``shut_fraction`` is the half-closed band:
    # the blink gate still passes, so a prediction is produced and used, and it
    # is made from an eye that is disappearing behind its own lid. That is what
    # makes the pointer wander the moment a wink begins -- long before the
    # closure is deep enough for anything to notice.
    steady_fraction: float = 0.80
    # Only open samples move the baseline, so a long closure cannot drag it
    # down until a shut eye counts as open.
    adapt: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.shut_fraction < 1.0:
            raise ValueError("shut_fraction must be between 0 and 1")
        if not 0.0 < self.adapt <= 1.0:
            raise ValueError("adapt must be in (0, 1]")
        if not self.shut_fraction < self.steady_fraction <= 1.0:
            raise ValueError("steady_fraction must sit above shut_fraction and at most 1")


class OpennessGate:
    """Is this eye shut, judged against how open it has been?

    Eye openness here is polygon AREA in px^2, which scales with how close the
    person sits and how large the image is. A fixed threshold therefore means
    different things on different days -- measured across five sessions, the
    same person's median ran 116 to 184 and the closure minimum ran 0.5 to 74.
    A ratio against the eye's own recent baseline does not have that problem.

    One gate per eye. Nothing here decides what a gesture is; it only answers
    "shut or open" so the detectors above can.
    """

    def __init__(self, config: OpennessGateConfig | None = None) -> None:
        self.config = config or OpennessGateConfig()
        self.baseline: float | None = None

    @property
    def ratio(self) -> float | None:
        return None if self.baseline is None else self._ratio

    def is_open(self, value: float) -> bool:
        if not math.isfinite(value) or value <= 0.0:
            self._ratio = 0.0
            return False
        if self.baseline is None:
            # The first sample is taken as an open eye. A session that began
            # with the eyes already shut would seed low, so the baseline is
            # allowed to rise freely below.
            self.baseline = value
        self._ratio = value / self.baseline if self.baseline else 0.0
        if value >= self.baseline:
            # Rising always updates immediately: sitting closer, or opening
            # wider, must not read as an eye that never opens.
            self.baseline = value
            return True
        if self._ratio >= self.config.shut_fraction:
            a = self.config.adapt
            self.baseline = self.baseline * (1.0 - a) + value * a
            return True
        return False
