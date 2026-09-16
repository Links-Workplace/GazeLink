"""Where practice prompts go on screen. Pure: no pygame (ARCH-01 stage D).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# --- practice prompt placement (pure; no pygame) -----------------------------
# "top": the original placement, unchanged -- lines centred on the screen,
# starting PROMPT_TOP_PX down, PROMPT_PITCH_PX apart.
# "sides": the same block drawn twice, once in each margin OUTSIDE the working
# band, vertically centred. Two identical copies so reading the task does not
# pull the gaze towards one side more than the other.
PROMPT_ANCHORS = ("top", "sides")
PROMPT_TOP_PX = 40
PROMPT_PITCH_PX = 70
PROMPT_SIDE_BAND = (0.30, 0.70)  # mirrors gf_dwell.BAND; not imported, as elsewhere here
PROMPT_SIDE_LINE_GAP_PX = 14


def prompt_layout(
    sizes: Sequence[tuple[int, int]],
    anchor: str,
    screen_w: int,
    screen_h: int,
) -> list[tuple[int, int, int, int]]:
    """Where each prompt line is drawn, as (x, y, w, h) in window pixels.

    ``sizes`` are the rendered (w, h) of each line. For "sides" the result
    holds the left copy's lines followed by the right copy's.
    """

    if anchor not in PROMPT_ANCHORS:
        raise ValueError(f"prompt_anchor must be one of {PROMPT_ANCHORS}, got {anchor!r}")
    if anchor == "top":
        return [
            (screen_w // 2 - w // 2, PROMPT_TOP_PX + i * PROMPT_PITCH_PX, w, h)
            for i, (w, h) in enumerate(sizes)
        ]
    if not sizes:
        return []
    pitch = max(h for _w, h in sizes) + PROMPT_SIDE_LINE_GAP_PX
    top = screen_h // 2 - (pitch * len(sizes)) // 2
    lo, hi = PROMPT_SIDE_BAND
    out: list[tuple[int, int, int, int]] = []
    margin = lo * screen_w
    widest = max(w for w, _h in sizes)
    if widest > margin or pitch * len(sizes) > screen_h:
        # Refused rather than clipped: text spilling into the band would put
        # the reading gaze back among the targets, which is what "sides" is
        # for avoiding. Only the 5120 px rig has been checked to fit.
        raise ValueError(
            f"prompt does not fit outside the band: widest line {widest}px, side margin "
            f"{margin:.0f}px, block {pitch * len(sizes)}px of {screen_h}px"
        )
    for centre_x in (lo * screen_w / 2.0, (1.0 + hi) * screen_w / 2.0):
        for i, (w, h) in enumerate(sizes):
            out.append((int(round(centre_x - w / 2.0)), top + i * pitch, w, h))
    return out


def prompt_button_gaps(
    rects: Sequence[tuple[int, int, int, int]],
    buttons: Sequence[Any],
    screen_w: int,
    screen_h: int,
) -> list[tuple[str, float, float]]:
    """For every (prompt line, button) pair: (key, gap_x, gap_y) as screen fractions.

    A gap is 0 on an axis where the two overlap; both 0 means the text is
    drawn on the button. Compared against the measured bias by the caller.
    """

    out = []
    for x, y, w, h in rects:
        x0, x1 = x / screen_w, (x + w) / screen_w
        y0, y1 = y / screen_h, (y + h) / screen_h
        for b in buttons:
            gap_x = max(0.0, b.x0 - x1, x0 - b.x1)
            gap_y = max(0.0, b.y0 - y1, y0 - b.y1)
            out.append((b.key, gap_x, gap_y))
    return out
