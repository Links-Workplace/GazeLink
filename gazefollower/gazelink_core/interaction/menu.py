"""The menu a long eye-close opens: four tiles a page, and a way back.

Pure state machine and geometry -- no camera, no pygame, no OS input, and
nothing here emits anything.  ``update`` returns a ``Choice`` describing what
the person picked; the live loop decides what to do about it, and the router
decides whether it is allowed.  Choosing a tile is an internal activation and
never reaches ``SendInput``.

Why a menu at all
-----------------

The SCROLL tile was the only way to reach a second capability, and the
operator's verdict on it was direct: *"the tile is hard to reach, which is why
I asked to remove it for scrolling"* -- which is where ``--start-scrolling``
came from.  A single tile sitting in the validated band was already at the
edge of what the gaze can put itself on; four capabilities could not each have
one.  So the way in is a GESTURE, which needs no accuracy at all, and the
tiles only have to be reachable once the person has already committed.

Why four tiles, and why this shape
----------------------------------

Horizontally the model is trusted only inside ``gf_dwell.BAND`` (0.30..0.70)
and the worst measured bias is 0.056 of screen width.  Two columns inside the
band give a half-width of 0.085 and a neighbour 0.145 away -- a miss is
possible at the worst positions and activating the WRONG tile is not, which is
the trade ``gf_dwell`` states in its own docstring.  Three columns are
documented there as expected to miss, so there are two.

Vertically the bias reaches 0.387 of screen height, which is why ``gf_dwell``
refuses to build a grid.  Two rows are used here anyway, and the reason is the
same one ``gf_scroll`` gives for its bands and which a live run has since
confirmed: rows at the two extremes with a very large dead zone between them
fail by MISSING, never by reaching the other row -- the neighbour is 0.62
away against a worst bias of 0.387.  The geometry is deliberately the scroll
bands' own (0.04 margin, 0.20 tall), because those are the only vertical
targets on this rig that have been reached by a real person.

That makes this an experiment with a measurement attached, exactly as the
bands were.  ``seen_in`` counts every frame the gaze spent on each tile, so
"I looked at it and nothing happened" and "the tile never saw me" -- the same
sentence from the person -- come back as two different numbers.

Two safety rules, each with a test
----------------------------------

* **Nothing is selectable until the gaze has been seen in the dead zone.**  A
  long close ENDS with the eyes opening, and the gaze lands wherever it lands;
  a menu that appeared under it would start filling a tile immediately, and
  the person never asked for that tile.  ``DwellEngine`` already clears its
  latch only in dead space; this is the same idea applied to arriving.
* **The menu opens while paused, and pausing is not the end of the road.**
  The pause tile is the resume tile -- same place, same gesture -- and the
  gesture layer that opens this is not gated on ``cursor_enabled``, so it
  still works when everything else has stopped.  Nothing chosen here reaches
  the OS while paused: ``ActionRouter`` refuses it all, RESUME excepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gazelink_core.interaction import actions as A
from gazelink_core.interaction import dwell as D

# The horizontal band the model is trusted in, and the vertical geometry the
# scroll bands proved reachable on this rig. Not re-derived: the same numbers.
BAND = D.BAND
ROW_MARGIN = 0.04
ROW_HEIGHT = 0.20
# The dead space between the two columns, inside the band. What is left over
# after it is split between them, so widening this narrows the tiles.
COLUMN_GAP = 0.06


class Effect(StrEnum):
    """What choosing a tile does, apart from anything it sends.

    Kept separate from ``Action`` because most tiles do not ask the OS for
    anything at all: they change the page, the mode, or the kind of click the
    wink will make.
    """

    PAGE = "page"  # go to another page of the menu; it stays open
    MODE = "mode"  # leave the menu for a UiMode
    CLICK_TYPE = "click type"  # what one wink will do from now on
    COMMAND = "command"  # ask for an Action and close
    CLOSE = "close"  # just close


# What one wink does. The default stays DOUBLE: that is what was measured and
# used live, and CLAUDE.md 8 forbids replacing a working implementation.
CLICK_TYPES = ("single", "double", "right")
CLICK_ACTION = {
    "single": A.Action.LEFT_CLICK,
    "double": A.Action.DOUBLE_CLICK,
    "right": A.Action.RIGHT_CLICK,
}
CLICK_LABEL = {"single": "לחיצה יחידה", "double": "לחיצה כפולה", "right": "לחיצה ימנית"}


@dataclass(frozen=True)
class Item:
    """One tile: what it is called, and what picking it does."""

    key: str
    label: str
    effect: Effect
    # Exactly one of these is meaningful, decided by ``effect``.
    page: str | None = None
    mode: A.UiMode | None = None
    click_type: str | None = None
    action: A.Action | None = None


@dataclass(frozen=True)
class Choice:
    """What the person picked, described rather than done."""

    item: Item

    @property
    def key(self) -> str:
        return self.item.key

    @property
    def effect(self) -> Effect:
        return self.item.effect

    @property
    def action(self) -> A.Action | None:
        return self.item.action


# The last tile of every page is always MORE, and it always leads one page on.
# Consistent so it can be learned: wherever the person is, the bottom outside
# tile takes them somewhere new and never commits to anything.
def _pages(*, paused: bool) -> dict[str, list[Item]]:
    """The pages, built fresh because two tiles depend on the current state.

    The pause tile is the resume tile. One place, one gesture, both directions
    -- a pause whose undo lives somewhere else is a pause the person can fall
    out of the system through.
    """

    return {
        "main": [
            Item("click-type", "סוג לחיצה", Effect.PAGE, page="click"),
            Item("scroll", "גלילה", Effect.MODE, mode=A.UiMode.SCROLL),
            Item("keyboard", "מקלדת", Effect.MODE, mode=A.UiMode.KEYBOARD),
            Item("more-1", "עוד", Effect.PAGE, page="nav"),
        ],
        "nav": [
            Item("back", "חזור", Effect.COMMAND, action=A.Action.BACK),
            Item("forward", "קדימה", Effect.COMMAND, action=A.Action.FORWARD),
            Item("switch", "החלף חלון", Effect.COMMAND, action=A.Action.SWITCH_WINDOW),
            Item("more-2", "עוד", Effect.PAGE, page="system"),
        ],
        "system": [
            Item("escape", "Escape", Effect.COMMAND, action=A.Action.ESCAPE),
            Item(
                "pause",
                "המשך שליטה" if paused else "השהיה",
                Effect.COMMAND,
                action=A.Action.RESUME if paused else A.Action.PAUSE,
            ),
            Item("close", "סגור תפריט", Effect.CLOSE),
            Item("more-3", "עוד", Effect.PAGE, page="main"),
        ],
        "click": [
            Item("single", CLICK_LABEL["single"], Effect.CLICK_TYPE, click_type="single"),
            Item("double", CLICK_LABEL["double"], Effect.CLICK_TYPE, click_type="double"),
            Item("right", CLICK_LABEL["right"], Effect.CLICK_TYPE, click_type="right"),
            Item("click-back", "חזרה", Effect.PAGE, page="main"),
        ],
    }


def tile_rects(
    *,
    band: tuple[float, float] = BAND,
    gap: float = COLUMN_GAP,
    margin: float = ROW_MARGIN,
    height: float = ROW_HEIGHT,
) -> list[tuple[float, float, float, float]]:
    """The four rectangles, in the order the items are given.

    Right to left, because the labels are Hebrew: the first tile is the one a
    Hebrew reader's eye reaches first, at the TOP RIGHT, and ``עוד`` ends up
    at the bottom left where a reader finishes. Putting item one at the top
    left would put the first choice where this language looks last.
    """

    lo, hi = band
    width = (hi - lo - gap) / 2.0
    if width <= 0.0:
        raise ValueError(f"a gap of {gap} leaves no room for two columns in {band}")
    right = (hi - width, hi)
    left = (lo, lo + width)
    top = (margin, margin + height)
    bottom = (1.0 - margin - height, 1.0 - margin)
    if bottom[0] <= top[1]:
        raise ValueError("the rows would meet, leaving no dead zone between them")
    return [
        (right[0], top[0], right[1], top[1]),
        (left[0], top[0], left[1], top[1]),
        (right[0], bottom[0], right[1], bottom[1]),
        (left[0], bottom[0], left[1], bottom[1]),
    ]


def page_buttons(items: list[Item], **geometry: float) -> list[D.Button]:
    """Dwell targets for one page, keyed by the item they belong to."""

    if len(items) != 4:
        raise ValueError(f"a page has exactly four tiles, not {len(items)}")
    rects = tile_rects(**geometry)  # type: ignore[arg-type]
    return [D.Button(item.key, *rect) for item, rect in zip(items, rects, strict=True)]


@dataclass
class MenuModel:
    """Open or shut, which page, and what the gaze is doing about it.

    Its own ``DwellEngine`` over its own targets, rebuilt on every page
    change, so the way through the menu can never be selected by anything else
    and nothing else can be selected by it -- the separation
    ``gf_click_practice`` uses for its MODE tile and ``gf_live`` used for the
    scroll tile.
    """

    dwell_ms: float = 900.0
    click_type: str = "double"

    def __post_init__(self) -> None:
        if self.click_type not in CLICK_TYPES:
            raise ValueError(f"{self.click_type!r} is not one of {CLICK_TYPES}")
        self.open = False
        self.page = "main"
        self.paused = False
        self._engine: D.DwellEngine | None = None
        self._items: list[Item] = []
        # Nothing may be picked until the gaze has been seen OUTSIDE every
        # tile at least once since the menu appeared. A long close ends with
        # the eyes opening somewhere unknown, and a tile already under that
        # gaze would begin filling with nobody having chosen it.
        self._seeded = False
        self.seen_in: dict[str, int] = {}
        self.refused_unseeded = 0

    # -- what the screen reads ---------------------------------------------

    @property
    def items(self) -> list[Item]:
        return list(self._items)

    @property
    def buttons(self) -> list[D.Button]:
        return list(self._engine.buttons) if self._engine is not None else []

    @property
    def seeded(self) -> bool:
        """Has the gaze been seen in the dead zone since the menu opened?"""

        return self._seeded

    @property
    def hovered(self) -> str | None:
        return None if self._engine is None else self._engine.hovered

    @property
    def progress(self) -> float:
        """0..1 for the ring. Zero while unseeded, because nothing is filling."""

        if self._engine is None or not self._seeded:
            return 0.0
        return self._engine.progress

    # -- opening and closing ------------------------------------------------

    def toggle(self, *, paused: bool = False) -> bool:
        """The long close: open it, or shut it again. Returns whether it is open."""

        if self.open:
            self.close()
        else:
            self.show(paused=paused)
        return self.open

    def show(self, *, paused: bool = False, page: str = "main") -> None:
        self.open = True
        self.paused = paused
        self._seeded = False
        self._go(page)

    def close(self) -> None:
        self.open = False
        self.page = "main"
        self._items = []
        self._engine = None
        self._seeded = False

    def set_paused(self, paused: bool) -> None:
        """Keep the pause tile telling the truth while the menu is up.

        The mode can change under an open menu -- a lost face suspends it --
        and a tile that still says "pause" when everything is already paused
        offers the person the one thing that will not help them.
        """

        if paused == self.paused:
            return
        self.paused = paused
        if self.open:
            self._go(self.page)

    def _go(self, page: str) -> None:
        pages = _pages(paused=self.paused)
        if page not in pages:
            raise KeyError(f"no menu page called {page!r}")
        self.page = page
        self._items = pages[page]
        self._engine = D.DwellEngine(
            page_buttons(self._items), D.DwellConfig(dwell_ms=self.dwell_ms)
        )
        for item in self._items:
            self.seen_in.setdefault(item.key, 0)
        self.seen_in.setdefault("dead zone", 0)
        self.seen_in.setdefault("no point", 0)

    # -- the machine --------------------------------------------------------

    def update(
        self, now_s: float, point: tuple[float, float] | None, *, fresh: bool = True
    ) -> Choice | None:
        """One frame. Returns a choice only on the frame it is made."""

        if not self.open or self._engine is None:
            return None
        usable = point if fresh else None
        if usable is None:
            self.seen_in["no point"] += 1
            self._engine.update(now_s, None, fresh=False)
            return None
        over = self._engine.find(usable)
        self.seen_in["dead zone" if over is None else over.key] += 1
        if not self._seeded:
            if over is None:
                # Seen where nothing can be chosen: from here on the person is
                # driving, and the tiles are live.
                self._seeded = True
            else:
                self.refused_unseeded += 1
            # Fed as absent either way, so no progress accumulates before the
            # gaze has demonstrably arrived under the person's own control.
            self._engine.update(now_s, None, fresh=False)
            return None
        activation = self._engine.update(now_s, usable, fresh=True)
        if activation is None:
            return None
        item = next(i for i in self._items if i.key == activation.button)
        return self._apply(item)

    def _apply(self, item: Item) -> Choice:
        """Carry out the part of a choice that belongs to the menu itself."""

        if item.effect is Effect.PAGE:
            # The only outcome that does NOT close: MORE and the sub-menu's
            # way back are navigation, and closing on them would make the
            # deeper pages unreachable.
            self._go(item.page or "main")
            # Re-seed on every page change, for the reason the menu seeds at
            # all: the gaze is resting on the tile that was just chosen, and
            # the tile that replaces it is a different tile.
            self._seeded = False
            return Choice(item)
        if item.effect is Effect.CLICK_TYPE and item.click_type:
            self.click_type = item.click_type
        # Everything else ends the menu. The operator asked for exactly this:
        # pick a thing and get on with it, rather than pick and then have to
        # close what you picked from.
        self.close()
        return Choice(item)

    def summary(self) -> dict[str, object]:
        looked = sum(self.seen_in.values())
        return {
            "click type": self.click_type,
            "tiles the gaze reached": {k: n for k, n in self.seen_in.items() if n},
            "frames with the menu open": looked,
            "frames refused before the gaze was seen in the dead zone": self.refused_unseeded,
        }


def layout_warnings() -> list[str]:
    """What this layout is expected to get wrong, said before it is used.

    Reported rather than inferred from a bad result afterwards, the way
    ``gf_dwell.layout_warnings`` is.

    Checked ROW BY ROW, which matters. ``gf_dwell.layout_warnings`` was
    written for full-height columns: it sorts by x0 and treats every adjacent
    pair as horizontal neighbours. Handed all four tiles at once it pairs a
    tile with the one BELOW it, computes a negative gap, and reports a
    neighbour risk that does not exist -- the two are half a screen apart
    vertically. Each row has exactly two tiles side by side, which is the
    shape that function actually describes.

    The vertical MISS warning is EXPECTED and is not a reason not to ship: the
    scroll bands carry the identical one at the identical geometry, and a real
    person reached them. What would be a reason is a NEIGHBOUR warning, and
    there is none: the columns are 0.145 apart against a bias of 0.056, and
    the rows 0.62 apart against 0.387.
    """

    items = _pages(paused=False)["main"]
    buttons = page_buttons(items)
    rows = [buttons[0:2], buttons[2:4]]
    out: list[str] = []
    for row in rows:
        for line in D.layout_warnings(row, bias_x=0.056, bias_y=0.387):
            if line not in out:
                out.append(line)
    return out
