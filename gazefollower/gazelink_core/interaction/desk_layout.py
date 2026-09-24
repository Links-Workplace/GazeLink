"""Where every desk control sits, and why it is allowed to sit there.

Pure geometry over ``dwell.Button`` -- no camera, no pygame, no OS input, no
drawing. The renderer is handed rectangles and labels, exactly as
``menu.tile_rects`` already hands them over.

The rule every layout here obeys
--------------------------------

``dwell.layout_warnings`` names two different failures with two different
thresholds. A target is MISSED once the bias exceeds its half-width; the
NEIGHBOUR is only reached once the bias exceeds half-width PLUS the gap.
Missing costs a second look. Activating the neighbour presses the wrong thing.

So the binding constraint is the GAP, not the size:

    half_width + gap  >  worst measured bias

Measured on 2026-09-22 with the active profile and the active model, nine
targets across the band (``results/resolution/dwell_main.json``): 17 of 18
correct first time, zero wrong selections, one non-selection, zero unintended
activations in sixty seconds at rest. The offsets in that run reached 0.052 of
screen width, against a half-width of 0.039 and a gap of 0.055 -- which is
exactly why it produced a miss and no wrong selection.

``BIAS`` below is that number. The 0.056/0.387 pair in ``dwell``'s docstring
predates the linear correction layer adopted in M2-CAL-01 and is not valid for
the active model; the vertical figure in that run was 0.006.

**One sitting, eighteen trials, one day.** A confirmation run on another day is
a blocking condition before these numbers are trusted further (TASKS M4-UI-01).
"""

from __future__ import annotations

from dataclasses import dataclass

from gazelink_core.interaction import dwell as D

# The horizontal band the model is trusted in. Not re-derived: dwell's own.
BAND = D.BAND

# The worst offset measured in the 2026-09-22 run, as a fraction of screen
# width. Every gap here is checked against it.
BIAS = 0.052


class LayoutTooTight(ValueError):
    """A layout whose neighbour is within reach of the measured bias."""


@dataclass(frozen=True)
class Rect:
    """A region that is drawn but never selected (a panel, a text line)."""

    x0: float
    y0: float
    x1: float
    y1: float


def _span(count: int, width: float, gap: float, band: tuple[float, float]) -> list[float]:
    """Left edges for ``count`` targets centred in the band, right-hand first.

    Right to left, because the labels are Hebrew: the first item is the one a
    Hebrew reader's eye reaches first.
    """

    if count < 1:
        raise ValueError("a row needs at least one target")
    lo, hi = band
    total = count * width + (count - 1) * gap
    if total > hi - lo + 1e-9:
        raise LayoutTooTight(
            f"{count} targets of {width} with gaps of {gap} need {total:.3f}, "
            f"but the band is only {hi - lo:.3f} wide"
        )
    reach = width / 2.0 + gap
    if count > 1 and reach <= BIAS:
        raise LayoutTooTight(
            f"from a target of {width} the neighbour starts {reach:.3f} from its centre, "
            f"at or under the measured bias {BIAS:.3f}: a biased point could activate it"
        )
    start = lo + (hi - lo - total) / 2.0
    return [start + i * (width + gap) for i in range(count)][::-1]


def row(
    keys: list[str],
    *,
    width: float,
    gap: float,
    top: float,
    height: float,
    band: tuple[float, float] = BAND,
) -> list[D.Button]:
    """One row of targets inside the band, first key rightmost."""

    lefts = _span(len(keys), width, gap, band)
    return [D.Button(k, x, top, x + width, top + height) for k, x in zip(keys, lefts, strict=True)]


def grid(
    keys: list[str],
    *,
    columns: int,
    width: float,
    gap: float,
    tops: list[float],
    height: float,
    band: tuple[float, float] = BAND,
) -> list[D.Button]:
    """Row-major, each row laid out right to left."""

    if len(keys) > columns * len(tops):
        raise ValueError(f"{len(keys)} keys do not fit in {columns}x{len(tops)}")
    lefts = _span(columns, width, gap, band)
    out = []
    for i, key in enumerate(keys):
        top = tops[i // columns]
        out.append(D.Button(key, lefts[i % columns], top, lefts[i % columns] + width, top + height))
    return out


# --- the fixed furniture ------------------------------------------------------

# Pause is NOT in the bar. It is its own isolated target in the same place on
# every screen, so it can never be reached by mistake in place of an action and
# is never somewhere new. CLAUDE.md 4.1 puts regaining control above everything
# else here, and a pause whose undo moves is a pause a person falls through.
PAUSE_KEY = "pause"
PAUSE = D.Button(PAUSE_KEY, 0.600, 0.025, 0.700, 0.125)

# The bottom row: four slots, the safe geometry (reach 0.065 against 0.052).
BAR_WIDTH = 0.070
BAR_GAP = 0.030
BAR_TOP = 0.760
BAR_HEIGHT = 0.150

# The LEFTMOST slot of the bottom row is always navigation and never an
# action: "more" at the root, "back" everywhere else. One place to learn.
NAV_SLOT = -1

BAR_KEYS = ["click", "scroll", "keyboard", "more"]
MENU_KEYS = ["zoom", "right-click", "double-click", "drag", "settings", "close"]

LABELS = {
    "click": "לחיצה",
    "scroll": "גלילה",
    "keyboard": "מקלדת",
    "more": "עוד",
    "back": "חזרה",
    "cancel": "ביטול",
    "pause": "השהיה",
    "resume": "המשך שליטה",
    "zoom": "דייק",
    "right-click": "לחיצה ימנית",
    "double-click": "לחיצה כפולה",
    "drag": "גרירה",
    "settings": "הגדרות",
    "close": "סגור תפריט",
    "confirm": "אשר",
    "zoom-in": "הגדל עוד",
    "overview": "כל המסך",
    "up": "למעלה",
    "down": "למטה",
    "finish": "סיום",
    "space": "רווח",
    "backspace": "מחיקה",
    "next-page": "עמוד הבא",
    "pick": "בחר",
    "drop": "הנח כאן",
}


def bar(keys: list[str] | None = None) -> list[D.Button]:
    """The one bar: three actions and the way to the rest."""

    return row(
        list(keys or BAR_KEYS),
        width=BAR_WIDTH, gap=BAR_GAP, top=BAR_TOP, height=BAR_HEIGHT,
    )


def bottom_row(keys: list[str]) -> list[D.Button]:
    """Any screen's bottom row, on the bar's own grid so nothing shifts."""

    if len(keys) > len(BAR_KEYS):
        raise ValueError(f"the bottom row holds at most {len(BAR_KEYS)} slots")
    lefts = _span(len(BAR_KEYS), BAR_WIDTH, BAR_GAP, BAND)
    chosen = lefts[: len(keys) - 1] + [lefts[NAV_SLOT]] if len(keys) > 1 else [lefts[NAV_SLOT]]
    return [
        D.Button(k, x, BAR_TOP, x + BAR_WIDTH, BAR_TOP + BAR_HEIGHT)
        for k, x in zip(keys, chosen, strict=True)
    ]


def menu(keys: list[str] | None = None) -> list[D.Button]:
    """The secondary actions: three across, two down."""

    return grid(
        list(keys or MENU_KEYS),
        columns=3, width=0.09667, gap=0.05167, tops=[0.200, 0.459], height=0.160,
    )


KEYBOARD_COLUMNS = 4
KEYBOARD_WIDTH = 0.0775
KEYBOARD_GAP = 0.030
KEYBOARD_LETTER_TOPS = [0.200, 0.444]
KEYBOARD_ACTION_TOP = 0.696
KEYBOARD_HEIGHT = 0.150
KEYBOARD_ACTIONS = ["space", "backspace", "next-page", "close"]


def keyboard(letters: list[str]) -> list[D.Button]:
    """Eight letters over two rows, then the four fixed actions.

    Eight and not more: the keys stay at a size the gaze can land on. Paging is
    the cost, and it is paid openly -- the page key names the letters it leads
    to, so "more" alone never has to be trusted.
    """

    if len(letters) > KEYBOARD_COLUMNS * len(KEYBOARD_LETTER_TOPS):
        raise ValueError(f"a page holds at most {KEYBOARD_COLUMNS * len(KEYBOARD_LETTER_TOPS)}")
    out = grid(
        list(letters),
        columns=KEYBOARD_COLUMNS, width=KEYBOARD_WIDTH, gap=KEYBOARD_GAP,
        tops=KEYBOARD_LETTER_TOPS, height=KEYBOARD_HEIGHT,
    )
    out += row(
        list(KEYBOARD_ACTIONS),
        width=KEYBOARD_WIDTH, gap=KEYBOARD_GAP,
        top=KEYBOARD_ACTION_TOP, height=KEYBOARD_HEIGHT,
    )
    return out


SCROLL_UP_TOP = 0.178
SCROLL_DOWN_TOP = 0.563
SCROLL_BAND_HEIGHT = 0.178


def scroll() -> list[D.Button]:
    """Up, down, and the way out. Nothing else on this screen.

    The two bands span the whole band width: there is no second target beside
    them, so the horizontal bias cannot reach a neighbour at all. The dead
    middle is what stops the wheel, and it is 0.207 tall.
    """

    lo, hi = BAND
    return [
        D.Button("up", lo, SCROLL_UP_TOP, hi, SCROLL_UP_TOP + SCROLL_BAND_HEIGHT),
        D.Button("down", lo, SCROLL_DOWN_TOP, hi, SCROLL_DOWN_TOP + SCROLL_BAND_HEIGHT),
        *bottom_row(["finish"]),
    ]


# The magnifier's frozen view. Drawn, never selected: the point inside it is
# chosen by the gaze against the image, not against a target grid.
ZOOM_PANEL = Rect(0.300, 0.193, 0.700, 0.593)
# How much of the screen the first magnification shows. 0.10 of the width in a
# panel 0.40 wide is 4x; the panel cannot be wider than the band, so this is
# the only lever there is.
ZOOM_SPAN = 0.10
ZOOM_SPAN_DEEP = 0.05
ZOOM_KEYS = ["confirm", "zoom-in", "overview", "back"]


def zoom_controls() -> list[D.Button]:
    return bottom_row(list(ZOOM_KEYS))


def warnings(buttons: list[D.Button]) -> list[str]:
    """What this layout is expected to get wrong, said before it is used.

    Reported against the bias actually measured, not against the pre-correction
    figures in ``dwell``'s docstring. Rows are checked separately, because
    ``dwell.layout_warnings`` pairs by x and would otherwise call a target and
    the one below it neighbours.
    """

    by_top: dict[float, list[D.Button]] = {}
    for button in buttons:
        by_top.setdefault(round(button.y0, 4), []).append(button)
    out: list[str] = []
    for top in sorted(by_top):
        for line in D.layout_warnings(by_top[top], bias_x=BIAS, bias_y=BIAS):
            if line not in out:
                out.append(line)
    return out
