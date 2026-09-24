"""Carrying something across the screen, as three separate askings.

Pure state machine. No camera, no pygame, no OS input: it says what should
happen and ``ActionExecutor`` is the only thing that can make it happen.

Why three steps and not one gesture
-----------------------------------

Drag is the most dangerous action this system has, because its failure is a
button left down: every later look becomes a selection, and the person cannot
let go. So it is never one indivisible thing that could fail halfway. Pick,
carry, drop -- each asked for separately, each refusable, and the carry state
visible the whole time.

The rule that matters most
--------------------------

**Letting go is not an action.** Starting a drag needs a permission; ending one
never does. A pause, a lost face, an Esc or a shutdown arriving mid-drag must
release the button, and every one of those is a state in which the router
refuses everything. So the drop path goes straight to the adapter, exactly as
``platform.real_input.require`` already lets a button-up through while refusing
a button-down.

``DragSequence`` therefore separates two vocabularies: ``drop`` is what the
person asked for, and ``force_release`` is what safety did to them. Both end
with the button up; only the first is a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Step(StrEnum):
    IDLE = "idle"  # nothing carried; the sequence has not begun
    PICKING = "picking"  # choosing what to carry
    CARRYING = "carrying"  # the button is DOWN and something is being carried
    DONE = "done"  # dropped where the person chose


class Ended(StrEnum):
    """Why a carry stopped, when it was not a deliberate drop."""

    CANCELLED = "the person cancelled"
    PAUSED = "control was paused"
    TRACKING_LOST = "the face was lost"
    STOPPING = "the session is shutting down"
    REFUSED = "the drop was refused"


# What the screen says at each step. One short line, never two: the whole
# point of this shape is that the person is told the next thing only.
SAY = {
    Step.IDLE: "בחר פריט",
    Step.PICKING: "בחר פריט",
    Step.CARRYING: "בחר מקום להנחה",
    Step.DONE: "בוצע",
}
# Which targets a step offers. The last one is always the way out, and it is
# always in the navigation slot, so cancel never moves.
OFFERS = {
    Step.IDLE: ("pick", "cancel"),
    Step.PICKING: ("pick", "cancel"),
    Step.CARRYING: ("drop", "cancel"),
    Step.DONE: ("finish",),
}


@dataclass
class DragSequence:
    """Pick, carry, drop. Nothing here emits; it only decides.

    ``held`` is the single source of truth for "a button is down because of
    this sequence". Every exit path clears it, and a caller that cannot see it
    cleared has not released anything.
    """

    step: Step = Step.IDLE
    picked_at: tuple[float, float] | None = None
    dropped_at: tuple[float, float] | None = None
    ended: Ended | None = None
    # Counted, because a drag that was cancelled by the person and a drag that
    # safety took away are different reports needing different fixes.
    tally: dict[str, int] = field(
        default_factory=lambda: {
            "started": 0, "dropped": 0, "cancelled": 0, "forced": 0, "refused": 0
        }
    )

    @property
    def held(self) -> bool:
        """Is a button down because of this sequence?"""

        return self.step is Step.CARRYING

    @property
    def say(self) -> str:
        return SAY[self.step]

    @property
    def offers(self) -> tuple[str, ...]:
        return OFFERS[self.step]

    def begin(self) -> None:
        """Enter the sequence. Nothing is held yet and nothing is sent."""

        self.step = Step.PICKING
        self.picked_at = None
        self.dropped_at = None
        self.ended = None

    def pick(self, at: tuple[float, float] | None) -> bool:
        """Ask to take what is under ``at``. True when the button should go DOWN.

        Refused without a point: a pick with nowhere to mean is not a pick,
        and pressing anyway would grab whatever the pointer was last left on.
        """

        if self.step is not Step.PICKING or at is None:
            self.tally["refused"] += 1
            return False
        self.picked_at = at
        self.step = Step.CARRYING
        self.tally["started"] += 1
        return True

    def drop(self, at: tuple[float, float] | None) -> bool:
        """Ask to let go here. True when the button should come UP.

        A drop with no point is refused and the carry CONTINUES -- refusing
        into a release would drop the item somewhere nobody chose, which is
        the same failure as never releasing, wearing better clothes.
        """

        if self.step is not Step.CARRYING or at is None:
            self.tally["refused"] += 1
            return False
        self.dropped_at = at
        self.step = Step.DONE
        self.ended = None
        self.tally["dropped"] += 1
        return True

    def cancel(self) -> bool:
        """The person asked to stop. True when a button must come up."""

        was_held = self.held
        self.step = Step.IDLE
        self.picked_at = None
        self.dropped_at = None
        self.ended = Ended.CANCELLED if was_held else None
        self.tally["cancelled"] += 1
        return was_held

    def force_release(self, why: Ended) -> bool:
        """Safety took the drag away. True when a button must come up.

        Idempotent on purpose: it is called from a pause, from a tracking
        loss, from a mode change and from shutdown, and several of those can
        arrive for the same event.
        """

        was_held = self.held
        self.step = Step.IDLE
        self.picked_at = None
        self.dropped_at = None
        if was_held:
            self.ended = why
            self.tally["forced"] += 1
        return was_held

    def finish(self) -> None:
        """Acknowledge a completed drop and leave the sequence."""

        if self.step is Step.DONE:
            self.step = Step.IDLE

    def summary(self) -> dict[str, object]:
        return {
            **self.tally,
            "ended": str(self.ended) if self.ended else "",
            "note": "held is the only truth about the button; every exit clears it",
        }
