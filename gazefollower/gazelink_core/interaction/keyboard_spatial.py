"""A keyboard chosen by looking at it: eight keys a page, four fixed actions.

Pure state machine -- no camera, no pygame, no OS input, no keystrokes. What it
produces is a ``Choice``; ``platform.keys`` is the only thing that can send one.

Why this and not the scanning keyboard
---------------------------------------

``interaction.keyboard`` asks the gaze for nothing: a highlight moves by itself
and a wink takes it. That was the right answer when three columns inside the
band were documented as expected to miss. The 2026-09-22 run changed the
premise -- nine targets across the band, 17 of 18, zero wrong -- so keys can be
LOOKED at, and no wink is required for anything (CLAUDE.md 4.2 asks for exactly
that: never force a wink).

The cost is paging, and it is paid openly. Four columns give a reach of 0.069
against a measured bias of 0.052, so eight keys is what fits at a size the gaze
can actually land on. Squeezing in a full alphabet would buy fewer pages with
targets under the bias, which is the trade ``desk_layout`` refuses to make.

**The page key names the letters it leads to.** "More" on its own asks the
person to remember a layout they cannot see; the preview means they never have
to.

Switching language is paging
----------------------------

There is no fifth target for "change language". The pages simply continue:
after the last Hebrew page comes the first English one, and the preview shows
that it does. One mechanism, one thing to learn, and the fixed four stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gazelink_core.interaction import actions as A
from gazelink_core.interaction import desk_layout as L
from gazelink_core.interaction import dwell as D

# Final forms are ordinary letters here. A person typing Hebrew needs them, and
# nothing in this module is clever enough to substitute one at the end of a
# word -- guessing wrong would silently change what they wrote.
HEBREW = "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"
ENGLISH = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
SYMBOLS = ".,?!-:;'\"()@/"

PER_PAGE = L.KEYBOARD_COLUMNS * len(L.KEYBOARD_LETTER_TOPS)

LAYOUTS = (
    ("hebrew", HEBREW),
    ("english", ENGLISH),
    ("digits", DIGITS),
    ("symbols", SYMBOLS),
)
LAYOUT_LABEL = {
    "hebrew": "עברית", "english": "English", "digits": "מספרים", "symbols": "סימנים",
}


class Control(StrEnum):
    NEXT_PAGE = "next-page"
    CLOSE = "close"


@dataclass(frozen=True)
class Choice:
    """One key. Exactly one of the three fields means anything."""

    key: str
    text: str | None = None
    action: A.Action | None = None
    control: Control | None = None

    def __post_init__(self) -> None:
        given = [x for x in (self.text, self.action, self.control) if x is not None]
        if len(given) != 1:
            raise ValueError(f"key {self.key!r} must do exactly one thing, not {len(given)}")


def _pages() -> list[tuple[str, list[str]]]:
    """Every page in order, each tagged with the layout it belongs to."""

    out: list[tuple[str, list[str]]] = []
    for name, chars in LAYOUTS:
        for start in range(0, len(chars), PER_PAGE):
            out.append((name, list(chars[start : start + PER_PAGE])))
    return out


PAGES = _pages()

ACTIONS = {
    "space": Choice("space", text=" "),
    "backspace": Choice("backspace", action=A.Action.BACKSPACE),
    "next-page": Choice("next-page", control=Control.NEXT_PAGE),
    "close": Choice("close", control=Control.CLOSE),
}


@dataclass
class SpatialKeyboard:
    """Which page is up, what the gaze is doing about it, and what was typed."""

    dwell_ms: float = 900.0
    page_index: int = 0

    def __post_init__(self) -> None:
        self.typed = ""
        # Nothing selectable until the gaze has been seen off every key since
        # the keyboard opened, or since the page changed: the gaze is resting
        # on the key that was just taken, and a different key is there now.
        self._armed = False
        self.seen_in: dict[str, int] = {}
        self.refused_unarmed = 0
        self.pages_turned = 0
        self._engine: D.DwellEngine | None = None
        self._build()

    # -- what the screen reads ------------------------------------------------

    @property
    def layout_name(self) -> str:
        return PAGES[self.page_index][0]

    @property
    def layout_label(self) -> str:
        return LAYOUT_LABEL[self.layout_name]

    @property
    def letters(self) -> list[str]:
        return list(PAGES[self.page_index][1])

    @property
    def next_letters(self) -> list[str]:
        return list(PAGES[(self.page_index + 1) % len(PAGES)][1])

    @property
    def next_preview(self) -> str:
        """What the page key leads to, so "more" never has to be trusted."""

        return " ".join(self.next_letters[:4])

    @property
    def buttons(self) -> list[D.Button]:
        return list(self._engine.buttons) if self._engine is not None else []

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def hovered(self) -> str | None:
        return None if self._engine is None else self._engine.hovered

    @property
    def progress(self) -> float:
        if self._engine is None or not self._armed:
            return 0.0
        return self._engine.progress

    # -- state ----------------------------------------------------------------

    def _build(self) -> None:
        self._engine = D.DwellEngine(
            L.keyboard(self.letters), D.DwellConfig(dwell_ms=self.dwell_ms)
        )
        for key in [*self.letters, *L.KEYBOARD_ACTIONS]:
            self.seen_in.setdefault(key, 0)
        self.seen_in.setdefault("content", 0)
        self._armed = False

    def reset(self, *, keep_text: bool = True) -> None:
        """Back to the first page. Called on open and on a mode change."""

        self.page_index = 0
        if not keep_text:
            self.typed = ""
        self._build()

    def turn_page(self) -> None:
        self.page_index = (self.page_index + 1) % len(PAGES)
        self.pages_turned += 1
        self._build()

    # -- one frame ------------------------------------------------------------

    def update(
        self, now_s: float, point: tuple[float, float] | None, *, fresh: bool = True
    ) -> Choice | None:
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
                self._armed = True
            else:
                self.refused_unarmed += 1
            engine.update(now_s, None, fresh=False)
            return None
        activation = engine.update(now_s, usable, fresh=True)
        if activation is None:
            return None
        return self._resolve(activation.button)

    def _resolve(self, key: str) -> Choice:
        if key in ACTIONS:
            choice = ACTIONS[key]
            if choice.control is Control.NEXT_PAGE:
                # Paging changes the keyboard and nothing else. Nothing is
                # sent, so the caller has nothing to route.
                self.turn_page()
            return choice
        return Choice(key, text=key)

    def on_sent(self, choice: Choice) -> None:
        """Echo a key only once it has actually left for the window.

        Called by the caller AFTER the adapter reported success, so the line
        the person reads is what was delivered rather than what was asked for.
        """

        if choice.text is not None:
            self.typed += choice.text
        elif choice.action is A.Action.BACKSPACE:
            self.typed = self.typed[:-1]

    def summary(self) -> dict[str, object]:
        return {
            "layout": self.layout_name,
            "pages turned": self.pages_turned,
            "characters echoed": len(self.typed),
            "keys the gaze reached": {k: n for k, n in self.seen_in.items() if n},
            "frames refused before the gaze returned to content": self.refused_unarmed,
        }


def layout_warnings() -> list[str]:
    return L.warnings(L.keyboard(list(HEBREW[:PER_PAGE])))
