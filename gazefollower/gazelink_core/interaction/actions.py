"""What the system can be asked to do, and the one place that says no.

Pure: no camera, no pygame, no ctypes, no OS input.  Everything here is a
decision about a request, and the decision is returned rather than acted on --
the adapters that can actually press a button stay the only things that press
one.  That is the same separation ``gf_click`` opens with, applied one level
up: until now every refusal lived inside whichever adapter happened to be
holding the request, so "the wink was seen and dropped" had a different
counter, and a different rule, in each path it could be dropped from.

The vocabulary MIRRORS ``InteractionType``/``InteractionSource`` in
src/gazelink/domain.py and deliberately does not import them.  The two trees
are kept apart -- ``gazefollower_capture`` says so in its first lines, and
``gf_dwell.Activation`` and ``gf_gesture`` already mirror rather than import.
The names are identical so that porting this into the product is a rename and
not a rewrite.

Four reasons a request is refused, each with its own counter, because they
need opposite fixes and "nothing happened" cannot tell them apart:

``PAUSED``      the person has stopped the system.  RESUME is the one
                exception, and it has to be: a pause with no way back is a
                session lost, and CLAUDE.md puts the ability to regain
                control above everything else here.
``STALE``       the request describes a moment that has passed.  A gesture
                that fired before a mode change, a queue drained late, a
                frame that took too long -- none of them are an intention
                about the screen as it is now.
``DUPLICATE``   the same action from the same source, again, too soon.  One
                gesture misfiring twice is not two intentions.
``WRONG_MODE``  the action does not belong to the mode the person is in.  A
                wink while scrolling is a request about the wheel, not a
                click on whatever the page has moved under the pointer; a
                wink at the keyboard chooses a key.

``UiMode`` is what the person is doing, and it is ORTHOGONAL to ``gf_control``
by design.  ``Mode`` stays two-state and ``cursor_enabled``/``selection_armed``
keep meaning exactly what they mean everywhere else -- the same reasoning that
kept scroll mode out of the ToggleMachine, now that there are four of these
rather than a boolean called ``scrolling``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Action(StrEnum):
    """Every outcome the system can be asked for.

    The first four names are ``InteractionType``'s own.  The rest are what a
    person needs to browse and type and which the product has no name for
    yet; when it grows one, these are the names it should grow.
    """

    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    SCROLL_STOP = "SCROLL_STOP"
    BACK = "BACK"
    FORWARD = "FORWARD"
    SWITCH_WINDOW = "SWITCH_WINDOW"
    ESCAPE = "ESCAPE"
    TYPE_TEXT = "TYPE_TEXT"
    BACKSPACE = "BACKSPACE"
    ENTER = "ENTER"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


class Source(StrEnum):
    """Who asked. Mirrors ``InteractionSource``; SCAN is new to the keyboard."""

    WINK = "WINK"
    DWELL = "DWELL"
    UI = "UI"
    SCAN = "SCAN"
    KEY = "KEY"


class UiMode(StrEnum):
    """What the person is doing right now.

    Replaces the ``scrolling`` boolean in the live loop.  A boolean could hold
    two of these; there are four, and a second boolean beside the first is how
    a state machine stops being one.
    """

    CURSOR = "CURSOR"
    SCROLL = "SCROLL"
    MENU = "MENU"
    KEYBOARD = "KEYBOARD"


class Refusal(StrEnum):
    NONE = "none"
    PAUSED = "paused"
    STALE = "stale"
    DUPLICATE = "duplicate"
    WRONG_MODE = "wrong mode"


# Which actions reach the OS as a mouse button. Named once, because three
# separate rules below ask the same question and a fourth will be added.
CLICKS = frozenset({Action.LEFT_CLICK, Action.RIGHT_CLICK, Action.DOUBLE_CLICK})
TYPING = frozenset({Action.TYPE_TEXT, Action.BACKSPACE, Action.ENTER})
SCROLLS = frozenset({Action.SCROLL_UP, Action.SCROLL_DOWN, Action.SCROLL_STOP})
# Reachable from the menu whatever the person was doing when they opened it:
# the menu is how they got here, so refusing them by mode would make the menu
# unable to do the only things it exists for.
COMMANDS = frozenset(
    {Action.BACK, Action.FORWARD, Action.SWITCH_WINDOW, Action.ESCAPE}
)


@dataclass(frozen=True)
class ActionRequest:
    """One asking. Frozen, and it carries a COPY of where the gaze was.

    ``at`` is normalised screen coordinates or None.  It is a copy on purpose:
    a request sits in a queue until the loop drains it, and a request that
    read the CURRENT gaze when it was finally looked at would act on a place
    the person had already left.  ``gf_live`` learned this with winks.

    ``issued_at_s`` is monotonic seconds, per CLAUDE.md 8.
    """

    action: Action
    source: Source
    issued_at_s: float
    at: tuple[float, float] | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if self.action is Action.TYPE_TEXT and not self.text:
            raise ValueError("TYPE_TEXT with no text is not a request for anything")
        if self.action is not Action.TYPE_TEXT and self.text is not None:
            raise ValueError(f"{self.action} does not carry text")

    def age_ms(self, now_s: float) -> float:
        return (now_s - self.issued_at_s) * 1000.0


@dataclass(frozen=True)
class Decision:
    """Yes or no, and why. Never the doing of it."""

    accepted: bool
    reason: Refusal
    request: ActionRequest

    @property
    def action(self) -> Action:
        return self.request.action


@dataclass(frozen=True)
class RouterLimits:
    """Bounds every request must satisfy, whoever believes otherwise."""

    # Older than this and the request describes a screen that has moved on.
    # Generous next to a 30 fps camera frame (33 ms) and far under the time
    # any mode change takes, which is what it is really there to catch.
    max_age_ms: float = 400.0
    # The same action from the same source, again, sooner than this. Under
    # the click adapter's own 400 ms gap on purpose: this catches a request
    # duplicated in the routing, and the adapter still applies its own rule
    # to what gets through.
    min_gap_ms: float = 250.0

    def __post_init__(self) -> None:
        if self.max_age_ms <= 0.0:
            raise ValueError("max_age_ms must be positive")
        if self.min_gap_ms < 0.0:
            raise ValueError("min_gap_ms must not be negative")


@dataclass
class ActionRouter:
    """One place that decides, and counts what it decided.

    It does not execute. The caller takes an accepted decision to whichever
    adapter can carry it out, and that adapter applies its own rules on top --
    two gates, like the cursor's, and neither substitutes for the other.
    """

    limits: RouterLimits = field(default_factory=RouterLimits)

    def __post_init__(self) -> None:
        self.refused: dict[Refusal, int] = {r: 0 for r in Refusal if r is not Refusal.NONE}
        self.accepted: dict[Action, int] = {}
        self._last_s: dict[tuple[Action, Source], float] = {}

    def reset(self) -> None:
        """Forget what was asked recently, not what was counted.

        Called on a mode change and on a pause: the duplicate rule exists to
        catch one gesture firing twice, and after a deliberate change of mode
        the next request is a new intention rather than an echo of the last.
        """

        self._last_s.clear()

    def decide(
        self,
        request: ActionRequest,
        *,
        paused: bool,
        ui_mode: UiMode,
        now_s: float,
    ) -> Decision:
        """Should this happen? Returns the answer and the reason for it.

        ``paused`` is the caller's control mode, passed in rather than read
        from anywhere -- the same reason ``ClickAdapter.click`` takes
        ``armed``: a module that can look up its own permission is a module
        that can be wrong about it alone.
        """

        action = request.action
        # Staleness first, and it applies to RESUME as well. An old event is
        # not an intention about now, whatever it asks for, and letting the
        # exception skip this would make a queue drained late able to resume
        # control the person had deliberately stopped.
        if request.age_ms(now_s) > self.limits.max_age_ms:
            return self._no(request, Refusal.STALE)
        if paused and action is not Action.RESUME:
            # Everything else, without exception: while paused, nothing this
            # system does may reach the OS. The menu still OPENS -- that is
            # the gesture layer, not this one -- so there is a way back.
            return self._no(request, Refusal.PAUSED)
        last = self._last_s.get((action, request.source))
        if last is not None and (now_s - last) * 1000.0 < self.limits.min_gap_ms:
            return self._no(request, Refusal.DUPLICATE)
        if not self._belongs(action, ui_mode):
            return self._no(request, Refusal.WRONG_MODE)
        self._last_s[(action, request.source)] = now_s
        self.accepted[action] = self.accepted.get(action, 0) + 1
        return Decision(True, Refusal.NONE, request)

    def screen_gesture(
        self, *, issued_at_s: float, now_s: float, ui_mode: UiMode, paused: bool
    ) -> Refusal:
        """May a SELECTION gesture (a wink) be looked at at all? Counts nothing.

        Checked before anything stateful reads the gesture -- the scanning
        keyboard's ``select`` moves its scan as it is called, so a wink refused
        after it would already have changed what the person sees. The rules
        are this router's own, in the order the live loop has always applied
        them; ``decide`` still re-checks the resulting request when it is
        executed.
        """

        if (now_s - issued_at_s) * 1000.0 > self.limits.max_age_ms:
            return Refusal.STALE
        if ui_mode in (UiMode.MENU, UiMode.SCROLL):
            # The tiles are chosen by dwell and the eyes drive the wheel: a
            # wink there is not a selection, and letting it click would fire two
            # mechanisms from one gesture.
            return Refusal.WRONG_MODE
        if paused:
            return Refusal.PAUSED
        return Refusal.NONE

    @staticmethod
    def _belongs(action: Action, ui_mode: UiMode) -> bool:
        """Does this action belong to the mode the person is in?

        PAUSE and RESUME belong to every mode: the way out and the way back
        must not depend on where the person happens to be standing.
        """

        if action in (Action.PAUSE, Action.RESUME):
            return True
        if action in COMMANDS:
            # Reached from the menu, which can be opened from anywhere.
            return True
        if action in CLICKS:
            # Not while the eyes are driving the wheel, choosing a key, or
            # choosing a tile. In each of those the same wink means something
            # else, and letting it also click is how one gesture fires two
            # mechanisms -- which this project has already done once.
            return ui_mode is UiMode.CURSOR
        if action in SCROLLS:
            return ui_mode is UiMode.SCROLL
        if action in TYPING:
            return ui_mode is UiMode.KEYBOARD
        return False

    def _no(self, request: ActionRequest, reason: Refusal) -> Decision:
        self.refused[reason] += 1
        return Decision(False, reason, request)

    def summary(self) -> dict[str, object]:
        return {
            "accepted": {str(a): n for a, n in sorted(self.accepted.items())},
            "refused": {str(r): n for r, n in self.refused.items() if n},
            "note": "this router decides; it cannot emit anything",
        }
