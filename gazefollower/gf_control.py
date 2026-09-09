"""Three control modes, and the rules for moving between them.

The cursor adapter already refuses to move on lost tracking, and the dwell
engine already fires at most once per entry.  Neither of them answers the
question this module exists for: *is the user asking for control at all right
now?*  Without that, a person who looks away from their work to think has no
way to say so, and a click mechanism is armed the whole time they are reading.

    PAUSED            nothing moves and nothing clicks
    MOVE_ONLY         the pointer follows the gaze; clicks are off
    MOVE_AND_SELECT   the pointer follows and the click mechanism is armed

Starting in PAUSED is not a default, it is the safety property: a session that
begins armed can emit a click before the user has agreed to anything.

Transitions use the same idiom as the recovery menu, for the same reason it
was chosen there -- it works with the eyelids alone and needs no accurate gaze
position, which matters most exactly when the gaze is NOT accurate.  A short
deliberate close moves the highlight, a long one commits it.  The highlight
starts on PAUSED so that an accidental long close cannot arm selection.

Cancellation is the other half.  Pausing, losing the face, changing profile or
entering calibration all discard a selection that was part-way through, and a
part-way selection that survives one of those is a click nobody asked for,
delivered later, at a gaze position measured before the interruption.

Pure logic on a monotonic clock: no camera, no Qt, no OS input.  What the
caller does with ``Transition.cancel_selection`` is what makes it true, so
there is a test asserting the live path actually applies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import gf_gesture as GEST


class Mode(StrEnum):
    PAUSED = "paused"
    MOVE_ONLY = "move-only"
    MOVE_AND_SELECT = "move-and-select"

    @property
    def cursor_enabled(self) -> bool:
        return self is not Mode.PAUSED

    @property
    def selection_armed(self) -> bool:
        return self is Mode.MOVE_AND_SELECT

    @property
    def description(self) -> str:
        return {
            Mode.PAUSED: "paused - nothing moves, nothing clicks",
            Mode.MOVE_ONLY: "pointer follows your gaze - clicks are OFF",
            Mode.MOVE_AND_SELECT: "pointer follows and selection is ARMED",
        }[self]


# Ordered least-capable first: the cycle walks this list, so the option a
# stray confirm lands on is the one that does nothing.
MODE_ORDER = (Mode.PAUSED, Mode.MOVE_ONLY, Mode.MOVE_AND_SELECT)


class Reason(StrEnum):
    """Why a transition or a cancellation happened. Shown, and logged."""

    STARTED = "started"
    USER_CONFIRMED = "user confirmed"
    TRACKING_LOST = "tracking lost"
    TRACKING_RETURNED = "tracking returned"
    PROFILE_CHANGED = "profile changed"
    CALIBRATION_ENTERED = "calibration entered"


@dataclass(frozen=True)
class Transition:
    """What the caller must do about this frame."""

    mode: Mode
    highlight: Mode
    changed: bool
    cancel_selection: bool
    reason: Reason | None = None

    @property
    def cursor_enabled(self) -> bool:
        return self.mode.cursor_enabled

    @property
    def selection_armed(self) -> bool:
        return self.mode.selection_armed


class ControlMachine:
    """The mode, the highlight, and every rule that clears a pending click."""

    def __init__(self, *, start: Mode = Mode.PAUSED) -> None:
        self._mode = start
        self._highlight = start
        # An event produced from a close that began before the face came back
        # must not be honoured. The detector is reset on loss, but a caller
        # that forgets would otherwise deliver a stale CONFIRM on the first
        # frame of recovery -- so recovery is not trusted until one clean
        # frame has passed here as well.
        self._settled = True

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def highlight(self) -> Mode:
        return self._highlight

    def _to(self, mode: Mode, reason: Reason) -> Transition:
        was_armed = self._mode.selection_armed
        changed = mode is not self._mode
        self._mode = mode
        self._highlight = mode
        # Leaving an armed mode discards whatever was filling. Cancelling on
        # every transition rather than only on the armed ones costs nothing
        # and removes a case to get wrong later.
        return Transition(mode, mode, changed, cancel_selection=was_armed or changed, reason=reason)

    def interrupt(self, reason: Reason) -> Transition:
        """Tracking loss, a profile change, or entering calibration.

        All three drop to PAUSED rather than to MOVE_ONLY: after any of them
        the gaze may be mapped through geometry that no longer applies, and
        the user has to say so again.
        """

        self._settled = False
        return self._to(Mode.PAUSED, reason)

    def update(self, event: GEST.Event, *, tracking_ok: bool = True) -> Transition:
        """One frame. ``event`` comes from the eyelid detector."""

        if not tracking_ok:
            return self.interrupt(Reason.TRACKING_LOST)
        if not self._settled:
            # First frame back. A quiet frame clears the guard; an event on
            # this frame is discarded, because it cannot have been started
            # deliberately after recovery.
            if event is GEST.Event.NONE:
                self._settled = True
            return Transition(self._mode, self._highlight, False, cancel_selection=False)
        if event is GEST.Event.CYCLE:
            index = MODE_ORDER.index(self._highlight)
            self._highlight = MODE_ORDER[(index + 1) % len(MODE_ORDER)]
            return Transition(self._mode, self._highlight, False, cancel_selection=False)
        if event is GEST.Event.CONFIRM:
            return self._to(self._highlight, Reason.USER_CONFIRMED)
        return Transition(self._mode, self._highlight, False, cancel_selection=False)


class ToggleMachine:
    """Two states and one gesture: the simplest thing that is still safe.

    The three-mode menu asked the user to remember a sequence -- short close,
    short close, long close -- and to track a highlight, before they could
    click anything. That is a lot to hold in mind for an action a hand does
    without thinking, and the operator asked for it to go.

    So: PAUSED and ACTIVE, and one long close of both eyes moves between them.
    Short closes are ignored entirely. Nothing else changes -- the same
    starting state, the same cancellations, the same guard against a stale
    event -- because those are what make the thing safe, and none of them cost
    the user anything to have.

    ``MOVE_AND_SELECT`` is reused as the active state rather than inventing a
    third name, so ``cursor_enabled`` and ``selection_armed`` keep meaning
    exactly what they mean everywhere else.
    """

    ACTIVE = Mode.MOVE_AND_SELECT

    def __init__(self, *, start: Mode = Mode.PAUSED) -> None:
        if start not in (Mode.PAUSED, self.ACTIVE):
            raise ValueError(f"{start} is not one of the two states")
        self._mode = start
        self._settled = True
        # What to go back to when the face returns, if the pause was not asked
        # for. ``None`` means there is nothing owed.
        self._resume_to: Mode | None = None

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def suspended(self) -> bool:
        """Paused because the face went, with a mode still owed back."""

        return self._resume_to is not None

    @property
    def highlight(self) -> Mode:
        """What a long close would choose now. Always the other state."""

        return Mode.PAUSED if self._mode is self.ACTIVE else self.ACTIVE

    @property
    def label(self) -> str:
        """The whole of what the screen has to say about the mode.

        A suspension says so rather than reading as a pause: "PAUSED" when the
        person did not pause anything sends them looking for the way back in,
        and there is nothing to find.
        """

        if self._resume_to is not None:
            return "WAITING FOR YOUR FACE - resumes on its own"
        return "PAUSED" if self._mode is Mode.PAUSED else "ACTIVE - wink to click"

    def _to(self, mode: Mode, reason: Reason) -> Transition:
        was_armed = self._mode.selection_armed
        changed = mode is not self._mode
        self._mode = mode
        return Transition(
            mode, self.highlight, changed, cancel_selection=was_armed or changed, reason=reason
        )

    def interrupt(self, reason: Reason) -> Transition:
        """Drop to paused. Losing the face SUSPENDS; anything else is final.

        The difference matters because looking away is not a decision. A pause
        the person did not ask for, that then has to be undone by hand, means
        every glance at the keyboard costs them their control -- and on a
        desktop, where the only way back in is a gesture that may not be
        registering, it can cost them the session.

        What is NOT resumed automatically is a pause the person asked for, or
        one caused by a profile change or a calibration: after those the gaze
        may be mapped through geometry that no longer applies.
        """

        if reason is Reason.TRACKING_LOST:
            # Only the FIRST loss records what to go back to. A loss lasts many
            # frames, and re-recording on each one would overwrite ACTIVE with
            # the PAUSED it had just been put into -- so a face gone for more
            # than a single frame would never come back armed.
            if self._resume_to is None:
                self._resume_to = self._mode
        else:
            self._resume_to = None
        self._settled = False
        return self._to(Mode.PAUSED, reason)

    def update(self, event: GEST.Event, *, tracking_ok: bool = True) -> Transition:
        if not tracking_ok:
            return self.interrupt(Reason.TRACKING_LOST)
        if not self._settled:
            if event is GEST.Event.NONE:
                self._settled = True
                if self._resume_to is not None and self._resume_to is not self._mode:
                    # The face is back and this frame is clean. Restoring what
                    # was interrupted, not arming something new.
                    resumed, self._resume_to = self._resume_to, None
                    return self._to(resumed, Reason.TRACKING_RETURNED)
                self._resume_to = None
            return Transition(self._mode, self.highlight, False, cancel_selection=False)
        if event is GEST.Event.CONFIRM:
            # A deliberate choice outranks anything waiting to be restored.
            self._resume_to = None
            return self._to(
                Mode.PAUSED if self._mode is self.ACTIVE else self.ACTIVE,
                Reason.USER_CONFIRMED,
            )
        # A short close is not an instruction here. Ignoring it rather than
        # giving it a meaning is deliberate: the fewer things the gesture can
        # do, the fewer ways it can do the wrong one.
        return Transition(self._mode, self.highlight, False, cancel_selection=False)
