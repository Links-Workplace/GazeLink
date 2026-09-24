"""The one bar: three actions, the way to the rest, and a pause that never moves.

Pure state machine and geometry over ``desk_layout`` -- no camera, no pygame,
no OS input. ``update`` returns a ``Pick`` describing what the person chose;
the controller decides what to do about it and the router decides whether it is
allowed.

Why a persistent bar needs a rule the menu did not
--------------------------------------------------

``MenuModel`` is safe partly because it only exists after a deliberate
gesture, and it re-arms its seed rule every time it is shown. A bar that is
always on screen has no such moment: ``DwellEngine`` clears its latch as soon
as the point is seen anywhere else (its own docstring says absence is not
evidence of having left, but dead space IS), so nothing would stop the bar
opening again 900 ms after every glance at the bottom of the screen.

So the bar carries its own arming rule: **after it closes it cannot be used
again until the gaze has been seen on CONTENT** -- that is, outside every
target this module owns. Glancing down does nothing; coming back and resting
does. The ring is drawn while it fills, so a person who did not mean it can
look away and watch it stop.

What is deliberately NOT here
-----------------------------

The ordinary click. Looking at a thing and confirming is not a trip through
this bar, and the "click" button is the CHOSEN KIND of click rather than a
step on the way to one. Putting every click through a menu was the thing that
made the first design unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gazelink_core.interaction import actions as A
from gazelink_core.interaction import desk_layout as L
from gazelink_core.interaction import dwell as D


class View(StrEnum):
    COMPACT = "compact"  # the bar, three actions and the way to the rest
    EXPANDED = "expanded"  # the six secondary actions
    MINIMISED = "minimised"  # drawn as a sliver; the TARGET is unchanged


class Effect(StrEnum):
    """What choosing something does, apart from anything it sends."""

    CLICK_KIND = "click kind"  # what a confirm will do from now on
    MODE = "mode"  # leave for a UiMode
    VIEW = "view"  # change what the bar is showing
    COMMAND = "command"  # ask for an Action
    NOTHING = "nothing"


# What a LEFT wink does. The eye picks the button: a RIGHT wink is always a
# right click, so "right" is no longer a kind to choose. The "click" tile is a
# toggle between single (default) and double -- the operator's decision on
# 24.9.2026, the same one the menu carries (``menu.DOUBLE_TOGGLE_LABEL``).
CLICK_KINDS = ("single", "double")
CLICK_ACTION = {
    "single": A.Action.LEFT_CLICK,
    "double": A.Action.DOUBLE_CLICK,
    "right": A.Action.RIGHT_CLICK,
}
CLICK_LABEL = {"single": "לחיצה כפולה: כבוי", "double": "לחיצה כפולה: פעיל"}


@dataclass(frozen=True)
class Item:
    key: str
    effect: Effect
    mode: A.UiMode | None = None
    view: View | None = None
    click_kind: str | None = None
    action: A.Action | None = None

    @property
    def label(self) -> str:
        return L.LABELS.get(self.key, self.key)


@dataclass(frozen=True)
class Pick:
    item: Item

    @property
    def key(self) -> str:
        return self.item.key

    @property
    def effect(self) -> Effect:
        return self.item.effect


BAR_ITEMS = (
    Item("click", Effect.CLICK_KIND, click_kind="single"),
    Item("scroll", Effect.MODE, mode=A.UiMode.SCROLL),
    Item("keyboard", Effect.MODE, mode=A.UiMode.KEYBOARD),
    Item("more", Effect.VIEW, view=View.EXPANDED),
)
MENU_ITEMS = (
    Item("zoom", Effect.MODE, mode=A.UiMode.ZOOM),
    Item("drag", Effect.MODE, mode=A.UiMode.DRAG),
    Item("settings", Effect.VIEW, view=View.COMPACT),
    Item("close", Effect.VIEW, view=View.COMPACT),
)
PAUSE_ITEM = Item(L.PAUSE_KEY, Effect.COMMAND, action=A.Action.PAUSE)
RESUME_ITEM = Item("resume", Effect.COMMAND, action=A.Action.RESUME)


@dataclass
class ControlBar:
    """What is on screen, what the gaze is doing about it, and what may fire."""

    dwell_ms: float = 900.0
    click_kind: str = "single"
    view: View = View.COMPACT

    def __post_init__(self) -> None:
        if self.click_kind not in CLICK_KINDS:
            raise ValueError(f"{self.click_kind!r} is not one of {CLICK_KINDS}")
        self.paused = False
        # Armed at the start: a session opens with the person already looking
        # at their screen, not at a target they did not choose.
        self._armed = True
        self.seen_in: dict[str, int] = {}
        self.refused_unarmed = 0
        self._engine: D.DwellEngine | None = None
        self._items: tuple[Item, ...] = ()
        self._build()

    # -- what the screen reads ------------------------------------------------

    @property
    def items(self) -> list[Item]:
        return list(self._items)

    @property
    def buttons(self) -> list[D.Button]:
        return list(self._engine.buttons) if self._engine is not None else []

    @property
    def armed(self) -> bool:
        """May anything be chosen right now?"""

        return self._armed

    @property
    def hovered(self) -> str | None:
        return None if self._engine is None else self._engine.hovered

    @property
    def progress(self) -> float:
        """0..1 for the ring. Zero while unarmed: nothing is filling."""

        if self._engine is None or not self._armed:
            return 0.0
        return self._engine.progress

    @property
    def pause_item(self) -> Item:
        """The pause target, which is the resume target, in one place."""

        return RESUME_ITEM if self.paused else PAUSE_ITEM

    def label_for(self, item: Item) -> str:
        if item.key == "click":
            return CLICK_LABEL[self.click_kind]
        return item.label

    # -- state ----------------------------------------------------------------

    def _build(self) -> None:
        if self.view is View.EXPANDED:
            self._items = MENU_ITEMS
            targets = L.menu([i.key for i in MENU_ITEMS])
        else:
            self._items = BAR_ITEMS
            targets = L.bar([i.key for i in BAR_ITEMS])
        # The pause target is part of every view, at the same rectangle, and
        # it is in the SAME engine so it cannot be chosen while something else
        # is filling.
        targets = [*targets, L.PAUSE]
        self._engine = D.DwellEngine(targets, D.DwellConfig(dwell_ms=self.dwell_ms))
        for key in [*(i.key for i in self._items), L.PAUSE_KEY, "content"]:
            self.seen_in.setdefault(key, 0)

    def set_view(self, view: View) -> None:
        if view is self.view:
            return
        self.view = view
        self._build()
        # Re-armed on every view change, for the reason the bar arms at all:
        # the gaze is resting on the thing that was just chosen, and the thing
        # that replaces it is a different thing.
        self._armed = False

    def set_paused(self, paused: bool) -> None:
        """Keep the pause target telling the truth about a mode that changed."""

        self.paused = bool(paused)

    def disarm(self) -> None:
        """Nothing may be chosen until the gaze has been seen on content."""

        self._armed = False
        if self._engine is not None:
            self._engine.reset()

    # -- one frame ------------------------------------------------------------

    def update(
        self, now_s: float, point: tuple[float, float] | None, *, fresh: bool = True
    ) -> Pick | None:
        """One frame. Returns a choice only on the frame it is made."""

        engine = self._engine
        if engine is None:
            return None
        usable = point if fresh else None
        if usable is None:
            engine.update(now_s, None, fresh=False)
            return None
        over = engine.find(usable)
        self.seen_in["content" if over is None else over.key] += 1
        if not self._armed:
            if over is None:
                # Seen on content: from here the person is driving.
                self._armed = True
            else:
                self.refused_unarmed += 1
            engine.update(now_s, None, fresh=False)
            return None
        activation = engine.update(now_s, usable, fresh=True)
        if activation is None:
            return None
        if activation.button == L.PAUSE_KEY:
            return Pick(self.pause_item)
        item = next(i for i in self._items if i.key == activation.button)
        return self._apply(item)

    def _apply(self, item: Item) -> Pick:
        if item.effect is Effect.CLICK_KIND:
            # A toggle, not a fixed value: each pick flips single <-> double.
            self.click_kind = "single" if self.click_kind == "double" else "double"
            # A kind is a setting, not a trip: the bar stays where it is and
            # the next confirm uses the new kind.
            self.set_view(View.COMPACT)
        elif item.effect is Effect.VIEW and item.view is not None:
            self.set_view(item.view)
        else:
            # Everything else leaves the bar behind, and leaving disarms it so
            # coming back is a decision rather than a glance.
            self.set_view(View.COMPACT)
            self.disarm()
        return Pick(item)

    def summary(self) -> dict[str, object]:
        return {
            "click kind": self.click_kind,
            "targets the gaze reached": {k: n for k, n in self.seen_in.items() if n},
            "frames refused before the gaze returned to content": self.refused_unarmed,
        }


def layout_warnings() -> list[str]:
    """Everything this bar's geometry is expected to get wrong, said up front."""

    out: list[str] = []
    for buttons in (L.bar(), L.menu(), [L.PAUSE]):
        for line in L.warnings(buttons):
            if line not in out:
                out.append(line)
    return out
