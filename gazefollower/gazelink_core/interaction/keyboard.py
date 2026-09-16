"""A scanning keyboard: the highlight moves on its own, a wink takes it.

Pure state machine -- no camera, no pygame, no OS input, no keystrokes.  What
it produces is a ``Key``; ``gf_keys`` is the only thing that can send one.

Why scanning rather than a keyboard laid out on screen
------------------------------------------------------

Because it needs no spatial accuracy at all, and accuracy is this rig's actual
limit.  A drawn keyboard would need thirty targets inside a band 0.40 of the
screen wide against a worst horizontal bias of 0.056 -- ``gf_dwell`` already
documents THREE columns in that band as expected to miss.  A scanning
keyboard asks the gaze for nothing: the highlight moves by itself and the
person supplies one bit, at a moment of their choosing, with an eyelid.

The cost is time, and it is paid twice -- once to choose the group, once to
choose the letter within it -- which is why the letters are grouped rather
than swept one at a time.  Five groups of about six is two waits of at most
five and six steps instead of one wait of thirty.

Safety
------

* **A wink here chooses a key and does not click.**  Enforced in
  ``ActionRouter`` by ``UiMode.KEYBOARD``, not by hoping the caller
  remembers -- the same rule that stops a wink clicking while scrolling.
* **The sweep stops itself.**  After ``max_sweeps`` passes with nothing
  chosen the highlight parks and waits.  A highlight that blinks round a
  layout for ever is a screen the person cannot rest their eyes on, and it
  also means every stray wink lands on whatever the timer happened to reach.
* **Nothing here knows about the window being typed into.**  That belongs to
  ``gf_keys``, which locks it and refuses when it changes.

The rate is an opening guess
----------------------------

``scan_ms`` has never been measured on this rig or with this person, and
nothing in the repository constrains it.  It is a flag for the same reason
``--wink-hold-ms`` and ``--scroll-arm-ms`` are: the report prints what
happened and the number moves when a measurement says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from gazelink_core.interaction import actions as A


class Level(StrEnum):
    GROUPS = "groups"  # sweeping the groups
    KEYS = "keys"  # sweeping the keys inside one chosen group
    PARKED = "parked"  # stopped itself after too many quiet sweeps


class Control(StrEnum):
    """A key that changes the keyboard rather than producing anything."""

    CANCEL = "cancel"  # back to the groups, choosing nothing
    LAYOUT = "layout"  # the next layout
    CLOSE = "close"  # leave the keyboard


@dataclass(frozen=True)
class Key:
    """One thing the highlight can stop on."""

    label: str
    text: str | None = None
    action: A.Action | None = None
    control: Control | None = None

    def __post_init__(self) -> None:
        given = [x for x in (self.text, self.action, self.control) if x is not None]
        if len(given) != 1:
            raise ValueError(f"key {self.label!r} must do exactly one thing, not {len(given)}")


@dataclass(frozen=True)
class Group:
    label: str
    keys: list[Key]

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError(f"group {self.label!r} has no keys")


def _letters(chars: str, per_group: int) -> list[Group]:
    """Split a run of characters into groups, labelled by their own extremes.

    The label is what the person reads while the group is highlighted, so it
    says which letters are inside rather than a number that means nothing.
    """

    groups = []
    for start in range(0, len(chars), per_group):
        chunk = chars[start : start + per_group]
        keys = [Key(c, text=c) for c in chunk]
        groups.append(Group(f"{chunk[0]}-{chunk[-1]}", keys))
    return groups


# The final forms are included as ordinary letters: a person typing Hebrew
# needs them, and nothing here is clever enough to substitute one at the end
# of a word -- guessing wrong would silently change what they wrote.
HEBREW = "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"
ENGLISH = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
SYMBOLS = ".,?!-:;'\"()@/#%&+="

# Present in every layout, always last, always in the same order. Consistent
# so it can be learned: wherever the person is, the last group is the one that
# does something other than type a letter.
def _actions() -> Group:
    return Group(
        "פעולות",
        [
            Key("רווח", text=" "),
            Key("מחיקה", action=A.Action.BACKSPACE),
            Key("Enter", action=A.Action.ENTER),
            # One tile, not two. "Cancel" and "back to the groups" are the
            # same act -- choosing nothing and starting again -- and giving
            # them separate keys would mean the person has to work out which
            # of two identical outcomes they want while a highlight moves.
            Key("ביטול", control=Control.CANCEL),
            Key("פריסה", control=Control.LAYOUT),
            Key("סגירה", control=Control.CLOSE),
        ],
    )


LAYOUT_ORDER = ("hebrew", "english", "digits", "symbols")
LAYOUT_LABEL = {
    "hebrew": "עברית",
    "english": "English",
    "digits": "מספרים",
    "symbols": "סימנים",
}


def layout(name: str) -> list[Group]:
    """The groups of one layout, ending with the actions group."""

    if name == "hebrew":
        groups = _letters(HEBREW, 6)
    elif name == "english":
        groups = _letters(ENGLISH, 6)
    elif name == "digits":
        groups = _letters(DIGITS, 5)
    elif name == "symbols":
        groups = _letters(SYMBOLS, 6)
    else:
        raise KeyError(f"no layout called {name!r}")
    return [*groups, _actions()]


@dataclass(frozen=True)
class ScanConfig:
    """How fast the highlight moves, and when it gives up.

    ``scan_ms`` is an OPENING GUESS. It has to be slower than the time it
    takes to see the highlight move, decide, and complete a wink -- a wink on
    this rig is about 150 ms of closure on top of the seeing and deciding --
    and faster than the point where typing a word stops being worth it.
    """

    scan_ms: float = 1200.0
    # Full passes with nothing chosen before the highlight parks itself.
    max_sweeps: int = 3
    # How long the eyelid had already been moving when the wink was STAMPED.
    #
    # Reported live: "I choose letters and it types other ones."  A wink is
    # not stamped when the person decides -- it is stamped on the camera frame
    # in which the closure finally crossed ``RightWinkConfig.hold_ms``, which
    # is at least that long after the lid started to move, plus however long
    # the camera and the landmark model took to deliver the frame.  Resolving
    # the wink against the highlight that was live at the STAMP therefore
    # reads the cell the sweep had already moved on to.
    #
    # Only the detector's own hold is known here, so that is the default the
    # caller passes in; the camera's share has not been measured on this rig
    # and is left at zero rather than guessed.  ``--wink-lag-ms`` moves it
    # when a measurement says so.
    wink_lag_ms: float = 0.0
    # How long the FIRST cell of a new screen must have been up before a wink
    # may take it.  A new screen is the descent into a group, the return to
    # the groups after a key, a layout change, and the keyboard opening --
    # every point where what is in front of the person changes wholesale.
    #
    # What it stops: one closure producing a second wink out of the jitter
    # around the reopening.  The detector's floor there is its cooldown
    # (120 ms) plus its hold (35 ms) on top of a wink measured at about
    # 150 ms, so the soonest an unwanted second one can arrive is on the
    # order of 300 ms.  Without this guard that second wink descends into a
    # group nobody read, or takes whatever sits first inside one.
    #
    # Deliberately NOT applied to an ordinary step of the sweep.  There the
    # floor is ``wink_lag_ms`` and it is already exact: a wink stamped less
    # than the lag after a step resolves to the cell before it, because that
    # is the one the person was watching.  Charging every cell 350 ms on top
    # of that would refuse a fast deliberate wink and cost them a whole pass.
    #
    # NOT measured on this person -- a flag for the same reason ``scan_ms``
    # is, and the report counts what it refused.
    settle_ms: float = 350.0

    def __post_init__(self) -> None:
        if self.scan_ms <= 0.0:
            raise ValueError("scan_ms must be positive")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be at least 1")
        if self.wink_lag_ms < 0.0:
            raise ValueError("wink_lag_ms must not be negative")
        if self.settle_ms < 0.0:
            raise ValueError("settle_ms must not be negative")
        if self.wink_lag_ms >= self.scan_ms:
            # The machine keeps ONE step of history, so a lag longer than a
            # cell would resolve to a cell it no longer remembers and quietly
            # take the wrong one anyway -- the failure this is here to fix.
            raise ValueError("wink_lag_ms must be shorter than scan_ms")
        if self.settle_ms >= self.scan_ms:
            # A settle as long as the cell itself means no cell is ever
            # takeable and the keyboard silently types nothing at all.
            raise ValueError("settle_ms must be shorter than scan_ms")


@dataclass
class ScanningKeyboard:
    """The highlight, what it is over, and what a wink does about it."""

    config: ScanConfig = field(default_factory=ScanConfig)
    layout_name: str = "hebrew"

    def __post_init__(self) -> None:
        if self.layout_name not in LAYOUT_ORDER:
            raise ValueError(f"{self.layout_name!r} is not one of {LAYOUT_ORDER}")
        self.level = Level.GROUPS
        self.index = 0
        self.group_index: int | None = None
        self._groups = layout(self.layout_name)
        self._last_step_s: float | None = None
        # The cell that was live immediately BEFORE the current one, and the
        # moment it became live, so a wink stamped before the last step can be
        # resolved against what the person was actually looking at. None at
        # every level change: there is no "before" on the other side of one.
        self._previous: tuple[int, float] | None = None
        self._sweeps = 0
        # What has been SENT, for the echo on screen. Appended only when the
        # caller confirms the keystroke actually left, so the display cannot
        # show text that never arrived anywhere.
        self.typed = ""
        self.selections = 0
        self.parked_times = 0
        # Both exist so a live run can say whether the timing rules are doing
        # anything, rather than the numbers staying a guess. "It typed the
        # wrong letter" and "it typed nothing" are different failures.
        self.winks_resolved_back = 0
        self.winks_too_soon = 0
        # When the last refusal happened, so the screen can SAY so. A wink
        # that does nothing and shows nothing is "I winked and nothing
        # happened", which CLAUDE.md 4.5 puts on the wrong side of the line.
        self.refused_at_s: float | None = None

    # -- what the screen reads ---------------------------------------------

    @property
    def groups(self) -> list[Group]:
        return list(self._groups)

    @property
    def keys(self) -> list[Key]:
        """The keys of the group being swept, or nothing at the group level."""

        if self.group_index is None:
            return []
        return list(self._groups[self.group_index].keys)

    @property
    def highlighted(self) -> Key | Group | None:
        """Exactly what a wink would take right now."""

        if self.level is Level.PARKED:
            return None
        if self.level is Level.GROUPS:
            return self._groups[self.index] if self._groups else None
        keys = self.keys
        return keys[self.index] if keys else None

    @property
    def parked(self) -> bool:
        return self.level is Level.PARKED

    # -- the machine --------------------------------------------------------

    def reset(self, *, keep_text: bool = True) -> None:
        """Back to the top of the groups, nothing chosen, sweep count zero."""

        self.level = Level.GROUPS
        self.index = 0
        self.group_index = None
        self._last_step_s = None
        self._previous = None
        self._sweeps = 0
        if not keep_text:
            self.typed = ""

    def switch_layout(self) -> str:
        """The next layout, wrapping. Returns its name."""

        nxt = LAYOUT_ORDER[(LAYOUT_ORDER.index(self.layout_name) + 1) % len(LAYOUT_ORDER)]
        self.layout_name = nxt
        self._groups = layout(nxt)
        self.reset()
        return nxt

    def update(self, now_s: float) -> None:
        """One frame. Moves the highlight on when its time is up.

        A parked keyboard does nothing here: it is waiting for a wink, and a
        wink is the only thing that starts it moving again.
        """

        if self.level is Level.PARKED:
            return
        if self._last_step_s is None:
            self._last_step_s = now_s
            return
        if (now_s - self._last_step_s) * 1000.0 < self.config.scan_ms:
            return
        count = len(self._groups) if self.level is Level.GROUPS else len(self.keys)
        if count <= 0:
            return
        # Recorded BEFORE the step, and only once the step is certain to
        # happen: a wink stamped a moment ago was a reaction to this cell and
        # not to the one about to replace it.
        self._previous = (self.index, self._last_step_s)
        self._last_step_s = now_s
        self.index += 1
        if self.index < count:
            return
        # A full pass with nothing taken.
        self.index = 0
        self._sweeps += 1
        if self._sweeps >= self.config.max_sweeps:
            # Parked rather than blinking round for ever: a screen that never
            # stops moving cannot be rested on, and every stray wink would
            # land on whatever the timer had reached.
            self.level = Level.PARKED
            self.parked_times += 1

    def resolve(self, when_s: float) -> tuple[int, float] | None:
        """Which cell was live when a wink happened, and since when.

        ``when_s`` is the moment the WINK was stamped, not the moment the loop
        got round to it -- the two differ by a queue and by a display frame,
        and at the group level that difference is a group nobody chose.

        None means no cell may be taken: nothing has been shown yet, or the
        one that was has not been on screen long enough to be a reaction to.
        """

        at = when_s - self.config.wink_lag_ms / 1000.0
        cell = self.index
        since = self._last_step_s
        # No previous cell means nothing has stepped since the last level
        # change, so what is on screen is a whole screen the person has not
        # read yet. That is the only place the settle is charged.
        guard = self.config.settle_ms if self._previous is None else 0.0
        if self._previous is not None and since is not None and at < since:
            # Stamped before the last step: the person was looking at the cell
            # the sweep has already moved off.
            cell, since = self._previous
        if since is None or (at - since) * 1000.0 < guard:
            return None
        return cell, since

    def select(self, now_s: float, when_s: float | None = None) -> Key | None:
        """A wink. Returns a key only when one was actually chosen.

        ``when_s`` is when the wink HAPPENED; ``now_s`` is when this is being
        called. Passing only the second is the old behaviour and is kept for
        callers that have no stamp -- but the live loop has one, and using it
        is the difference between taking the cell the person was watching and
        taking whichever one the timer had reached by the time the queue was
        drained.

        At the group level a wink descends into that group and produces
        nothing -- the same shape as ``RecoveryMenu``, where the first
        deliberate close only opens the list. A wink that arrives with nothing
        on screen must not select anything.
        """

        if self.level is Level.PARKED:
            # The way back in. It chooses nothing: the highlight was parked
            # wherever the last sweep left it, and taking that would be taking
            # a key the person never watched arrive.
            self.reset()
            self._last_step_s = now_s
            return None
        resolved = self.resolve(now_s if when_s is None else when_s)
        if resolved is None:
            # Too soon to be a choice. Nothing is taken AND nothing is
            # disturbed: the sweep count is left alone and the highlight keeps
            # its own clock, so a stray wink costs the person no more than the
            # rest of the cell they are already waiting through.
            self.winks_too_soon += 1
            self.refused_at_s = now_s
            return None
        cell, _since = resolved
        if cell != self.index:
            self.winks_resolved_back += 1
        self._sweeps = 0
        self._last_step_s = now_s
        self._previous = None
        if self.level is Level.GROUPS:
            if not self._groups:
                return None
            self.group_index = cell
            self.level = Level.KEYS
            self.index = 0
            return None
        keys = self.keys
        if not keys:
            self.reset()
            return None
        if cell >= len(keys):
            # Cannot happen while the level is unchanged, but a resolved cell
            # is an index from a moment ago and an out-of-range one must not
            # reach into the list.
            self.reset()
            return None
        key = keys[cell]
        self.selections += 1
        if key.control is Control.CANCEL:
            self.reset()
            return key
        if key.control is Control.LAYOUT:
            self.switch_layout()
            return key
        if key.control is Control.CLOSE:
            self.reset()
            return key
        # A produced key: back to the groups so the next letter starts from
        # the top, rather than continuing inside whichever group it came from.
        self.reset()
        return key

    def refused_recently(self, now_s: float, within_s: float = 1.2) -> bool:
        """Was a wink refused as too soon, recently enough to still say so?"""

        if self.refused_at_s is None:
            return False
        return 0.0 <= now_s - self.refused_at_s <= within_s

    def on_sent(self, key: Key) -> None:
        """The caller confirming a keystroke actually left the machine.

        Separate from ``select`` on purpose: a keystroke can be refused after
        it is chosen -- the target window changed, the mode is paused -- and
        an echo that showed it anyway would tell the person their text is
        somewhere it is not.
        """

        if key.text is not None:
            self.typed += key.text
        elif key.action is A.Action.BACKSPACE:
            self.typed = self.typed[:-1]
        elif key.action is A.Action.ENTER:
            self.typed = ""

    def summary(self) -> dict[str, object]:
        return {
            "layout": self.layout_name,
            "scan_ms": self.config.scan_ms,
            "settle_ms": self.config.settle_ms,
            "wink_lag_ms": self.config.wink_lag_ms,
            "selections": self.selections,
            "winks taken as the previous cell": self.winks_resolved_back,
            "winks refused as too soon": self.winks_too_soon,
            "times it parked itself": self.parked_times,
            "characters echoed": len(self.typed),
        }
