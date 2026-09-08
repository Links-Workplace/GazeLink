"""A deliberate eye-close, read without any calibration at all.

This exists because freezing gaze is only half a safety feature.  When the
display changes, :mod:`gazelink.display_watch` withholds every gaze point --
correctly, because the calibration no longer describes the screen.  But a
person who cannot use their hands then has no way to ask for a recalibration,
and a freeze with no exit is a trap rather than a protection.

The signal has to be one that survives the freeze, which rules out gaze:
gaze is exactly what stopped being trustworthy.  What remains is the eyelid,
and the useful fact about it is that eyelid geometry is scale-invariant and
never passes through the calibration model, so it means the same thing on a
screen we have never calibrated for.

**What the signal actually is.**  Not eye openness.  The obvious design reads
``EyeObservation.openness`` and waits for it to approach zero, and that does
not work: :func:`gazelink.features.extract_eye_features` returns an *invalid*
result once the eyelid gap reaches its geometry epsilon, and returns early
when the iris landmarks are missing, which is the normal case for a shut eye.
So ``left_eye`` and ``right_eye`` are ``None`` precisely during the gesture.
The detectable event is therefore the *conjunction*: the face is still being
tracked while both eyes have stopped yielding valid geometry, sustained over
time.  A face that vanishes entirely is not a close -- somebody left, or the
camera was blocked -- and is treated as an abort rather than a hold.

**Two gestures, one channel.**  A short hold means "move to the next option"
and a long hold means "choose this one", which is enough to drive a menu of a
few large targets with no pointing of any kind.  Anything shorter than
``natural_blink_max_ms`` is a blink and is ignored, which is the difference
between a safety feature and a machine that reacts every time you blink.

**Thresholds are configuration, not constants**: they come from
:class:`~gazelink.config.GestureTimingConfig`, the one place this project keeps
tunable timings.  They have **not** been validated against real users or real
video; the defaults are reasoned, not measured, and the durations a person can
comfortably hold are exactly the sort of thing that has to be observed rather
than assumed.

All time is milliseconds from a monotonic clock.  Nothing here imports Qt or
touches a camera, so every transition below is tested deterministically.

M3-03 owns blink, wink and dwell.  This is deliberately the *shared* mechanism
rather than a second parallel detector: when M3-03 lands it should extend this
component, not reimplement it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from gazelink.config import GestureTimingConfig
from gazelink.domain import ContractValidationError, TrackingState, VisionObservation

__all__ = [
    "EyeCloseDetector",
    "advance_recovery",
    "RecoveryMenu",
    "RecoveryOption",
    "RecoveryGesture",
    "eyes_unreadable_with_face_present",
    "face_is_present",
]


class RecoveryGesture(StrEnum):
    """What a completed close was asking for."""

    #: A short deliberate close: move the highlight to the next option.
    CYCLE = "CYCLE"
    #: A long deliberate close: act on the highlighted option.
    CONFIRM = "CONFIRM"


def face_is_present(observation: VisionObservation) -> bool:
    """Is there exactly one face to read a gesture from?

    The single definition, deliberately: this used to be spelled one way here
    and another way at the call site, and the two disagreed about
    ``MULTIPLE_FACES``.  A second person walking into frame mid-hold then read
    as "eyes opened", which fired a gesture nobody asked for -- from the only
    input channel a frozen session still has.

    ``MULTIPLE_FACES`` counts as absent because with more than one face there
    is no way to say whose eyes closed, and guessing would let a passer-by
    trigger a recalibration.
    """

    if not isinstance(observation, VisionObservation):
        raise ContractValidationError("observation must be a VisionObservation")
    return observation.tracking_state not in (
        TrackingState.LOST,
        TrackingState.MULTIPLE_FACES,
    )


def eyes_unreadable_with_face_present(observation: VisionObservation) -> bool:
    """Is this frame the signature of a deliberately closed pair of eyes?

    Reads the *raw* observation on purpose.  The confidence policy rejects
    these frames -- closed eyes are low confidence by construction -- so a
    detector built on the accepted observation would go blind exactly when it
    is needed.  Callers must pass ``tick.observation``, never
    ``tick.accepted_observation``.

    ``MULTIPLE_FACES`` counts as absent: with more than one face in frame there
    is no way to say whose eyes closed, and guessing would let a passer-by
    trigger a recalibration.
    """

    return (
        face_is_present(observation)
        and observation.left_eye is None
        and observation.right_eye is None
    )


@dataclass(frozen=True, slots=True)
class _Progress:
    """How far the current hold has gone, for drawing a countdown."""

    held_ms: float
    confirm_at_ms: int

    @property
    def fraction(self) -> float:
        if self.confirm_at_ms <= 0:
            return 1.0
        return min(1.0, max(0.0, self.held_ms / self.confirm_at_ms))


class EyeCloseDetector:
    """Turns a stream of frames into at most one gesture at a time.

    Deliberate properties, each of which is a way this could otherwise hurt
    someone:

    * A blink shorter than ``natural_blink_max_ms`` yields nothing.  Reacting
      to natural blinks would make the machine unusable rather than helpful.
    * ``CONFIRM`` fires **while the eyes are still shut**, not on release, so
      the user gets the countdown they were promised and does not have to guess
      when to open.  It fires once per close.
    * Once ``CONFIRM`` has fired, releasing does not additionally fire
      ``CYCLE``: one close is one instruction.
    * A face lost mid-hold aborts rather than completing.  Somebody who turned
      away has not asked for anything.
    * A cooldown after every accepted gesture stops one long close from being
      read as a burst of commands.
    """

    def __init__(self, timing: GestureTimingConfig | None = None) -> None:
        if timing is not None and not isinstance(timing, GestureTimingConfig):
            raise ContractValidationError("timing must be a GestureTimingConfig")
        self._timing = timing or GestureTimingConfig()
        self._closed_since_ms: float | None = None
        self._confirmed_this_close = False
        self._ready_at_ms: float | None = None

    @property
    def timing(self) -> GestureTimingConfig:
        return self._timing

    @property
    def is_holding(self) -> bool:
        """Are the eyes currently shut in a way that could become a gesture?"""

        return self._closed_since_ms is not None

    def progress(self, now_monotonic_ms: float) -> float:
        """0.0 to 1.0 towards ``CONFIRM``, for a visible countdown.

        A user holding their eyes shut cannot see a progress bar, so this is
        for whatever the interface does with it -- a tone, or a state the
        person sees the instant they open.  It is 0.0 when nothing is held.
        """

        if self._closed_since_ms is None:
            return 0.0
        held = now_monotonic_ms - self._closed_since_ms
        return _Progress(held, self._timing.recovery_confirm_ms).fraction

    def reset(self) -> None:
        """Forget any hold in progress, without arming a cooldown."""

        self._closed_since_ms = None
        self._confirmed_this_close = False

    def update(
        self,
        *,
        eyes_closed: bool,
        now_monotonic_ms: float,
        face_present: bool = True,
    ) -> RecoveryGesture | None:
        """Advance the machine by one frame and report a completed gesture.

        ``eyes_closed`` is the conjunction described in
        :func:`eyes_unreadable_with_face_present`.

        ``face_present`` is separate from it on purpose.  Both a released hold
        and a lost face arrive as "not closed", but they mean opposite things:
        releasing is the user finishing an instruction, while losing the face
        is the user turning away or being occluded.  Collapsing the two made
        turning your head mid-hold emit ``CYCLE`` -- an action nobody asked
        for, from the one input channel a frozen session still has.
        """

        now = _finite(now_monotonic_ms, "now_monotonic_ms")

        if not face_present:
            # Abort in silence. Turning away is not consent.
            self._closed_since_ms = None
            self._confirmed_this_close = False
            return None

        if not eyes_closed:
            # A genuine release: the user opened their eyes.
            started = self._closed_since_ms
            confirmed = self._confirmed_this_close
            self._closed_since_ms = None
            self._confirmed_this_close = False
            if started is None or confirmed:
                # Nothing was held, or CONFIRM already spoke for this close.
                return None
            held = now - started
            if held < self._timing.natural_blink_max_ms:
                # A blink. Silence here is the whole point.
                return None
            return self._accept(RecoveryGesture.CYCLE, now)

        if self._closed_since_ms is None:
            if self._ready_at_ms is not None and now < self._ready_at_ms:
                # Still cooling down; do not start a hold whose duration would
                # be measured from inside the previous gesture's tail.
                return None
            self._closed_since_ms = now
            self._confirmed_this_close = False
            return None

        if self._confirmed_this_close:
            return None
        if now - self._closed_since_ms >= self._timing.recovery_confirm_ms:
            self._confirmed_this_close = True
            return self._accept(RecoveryGesture.CONFIRM, now)
        return None

    def _accept(self, gesture: RecoveryGesture, now: float) -> RecoveryGesture:
        self._ready_at_ms = now + self._timing.cooldown_ms
        return gesture


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractValidationError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class RecoveryOption:
    """One thing a frozen screen will let the user ask for.

    ``key`` is what the caller switches on; ``label`` is what the person reads.
    """

    key: str
    label: str

    def __post_init__(self) -> None:
        for name, value in (("key", self.key), ("label", self.label)):
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"{name} must be a non-empty string")


class RecoveryMenu:
    """A few large choices, driven entirely by eyelid gestures.

    Deliberately tiny.  Gaze is untrustworthy at the moment this appears --
    that is why it appeared -- so nothing here may depend on pointing at
    anything.  A short close moves the highlight, a long close takes the
    highlighted option, and the option count is kept to two or three so that
    cycling to the one you want is a couple of gestures rather than a chore.

    The highlight starts on the *least* destructive option so that a stray
    long close cannot do something drastic.
    """

    def __init__(self, options: Sequence[RecoveryOption]) -> None:
        chosen = tuple(options)
        if len(chosen) < 2:
            raise ContractValidationError("a recovery menu needs at least two options")
        for option in chosen:
            if not isinstance(option, RecoveryOption):
                raise ContractValidationError("options must be RecoveryOption values")
        if len({option.key for option in chosen}) != len(chosen):
            raise ContractValidationError("recovery option keys must be unique")
        self._options = chosen
        self._index = 0

    @property
    def options(self) -> tuple[RecoveryOption, ...]:
        return self._options

    @property
    def highlighted(self) -> RecoveryOption:
        return self._options[self._index]

    @property
    def index(self) -> int:
        return self._index

    def cycle(self) -> RecoveryOption:
        """Move to the next option, wrapping around."""

        self._index = (self._index + 1) % len(self._options)
        return self.highlighted

    def apply(self, gesture: RecoveryGesture) -> RecoveryOption | None:
        """Feed a completed gesture in; get a chosen option out, or ``None``."""

        if not isinstance(gesture, RecoveryGesture):
            raise ContractValidationError("gesture must be a RecoveryGesture")
        if gesture is RecoveryGesture.CYCLE:
            self.cycle()
            return None
        return self.highlighted

    def describe(self) -> str:
        """The menu as one line of text, marking the highlighted option."""

        return "   ".join(
            f"[ {option.label} ]" if index == self._index else f"  {option.label}  "
            for index, option in enumerate(self._options)
        )


def advance_recovery(
    detector: EyeCloseDetector,
    observation: VisionObservation,
    *,
    now_monotonic_ms: float,
) -> RecoveryGesture | None:
    """Feed one frame to the detector, deriving both inputs the same way.

    This exists so the two questions -- "are the eyes shut?" and "is a face
    there at all?" -- cannot be answered by different rules.  When the caller
    assembled them itself they disagreed about ``MULTIPLE_FACES``, and a
    passer-by entering frame mid-hold completed somebody else's gesture.
    """

    if not isinstance(detector, EyeCloseDetector):
        raise ContractValidationError("detector must be an EyeCloseDetector")
    return detector.update(
        eyes_closed=eyes_unreadable_with_face_present(observation),
        face_present=face_is_present(observation),
        now_monotonic_ms=now_monotonic_ms,
    )
