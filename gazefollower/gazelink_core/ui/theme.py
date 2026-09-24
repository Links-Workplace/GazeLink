"""The desk interface's visual language, as tokens and nothing else.

No drawing, no pygame, no state. A renderer reads these; nothing here reads a
renderer. Kept separate from the existing screens on purpose: every protocol
window in this project was laid out against its own inline colours, and
CLAUDE.md 8 says not to silently restyle something that works.

Why these values
----------------

Charcoal ground, gold for "this is what you are about to choose", blue for the
gaze itself. The three are never the ONLY carrier of a state: every state that
matters also changes shape (border weight, a ring, a dashed edge) and carries a
word, because TECHNICAL_SPEC 12 forbids relying on colour alone.

Contrast was chosen against the ground each colour is actually used on, not in
the abstract: ``TEXT`` on ``SURFACE`` and ``INK_ON_ACCENT`` on ``ACCENT`` are
the two pairs that carry running text.
"""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Palette:
    """Every colour the desk screens may use. Nothing else is allowed."""

    # Grounds
    BACKDROP: RGB = (14, 15, 17)
    SURFACE: RGB = (20, 22, 26)
    SURFACE_RAISED: RGB = (35, 38, 44)
    # Lines and edges
    EDGE: RGB = (76, 82, 92)
    EDGE_QUIET: RGB = (42, 46, 54)
    # Text
    TEXT: RGB = (244, 243, 240)
    TEXT_QUIET: RGB = (168, 170, 177)
    # The accent: "about to be chosen", and the chosen state
    ACCENT: RGB = (212, 162, 76)
    ACCENT_BRIGHT: RGB = (240, 204, 126)
    INK_ON_ACCENT: RGB = (20, 22, 26)
    # The gaze itself, never used for anything else
    GAZE: RGB = (47, 111, 208)
    GAZE_TRACK: RGB = (201, 217, 236)
    # Outcomes
    DONE: RGB = (79, 174, 124)
    ALARM: RGB = (224, 112, 95)


@dataclass(frozen=True)
class Metrics:
    """Sizes in normalised screen units, or in points where they are type.

    Normalised because the rig is a 5120x1440 panel and a second machine will
    not share its pixels; the layouts in ``interaction.desk_layout`` are in the
    same units for the same reason.
    """

    # Borders: the quiet state and the "about to be chosen" state differ by
    # WEIGHT as well as colour, so the difference survives a colour deficit.
    BORDER_PX: int = 3
    BORDER_ACTIVE_PX: int = 3
    # The dwell ring drawn on whatever is filling.
    RING_WIDTH_PX: int = 8
    # Type sizes, in points at the reference panel height.
    TEXT_PT: int = 21
    LABEL_PT: int = 22
    LETTER_PT: int = 52
    SAY_PT: int = 25
    # A short animation is a courtesy; a moving target is a defect. Nothing
    # that can be selected may move, so this applies to fades only.
    FADE_MS: float = 120.0


@dataclass(frozen=True)
class Theme:
    palette: Palette = Palette()
    metrics: Metrics = Metrics()

    def edge_for(self, *, active: bool) -> RGB:
        return self.palette.ACCENT if active else self.palette.EDGE

    def fill_for(self, *, active: bool) -> RGB:
        return self.palette.ACCENT_BRIGHT if active else self.palette.SURFACE

    def ink_for(self, *, active: bool) -> RGB:
        return self.palette.INK_ON_ACCENT if active else self.palette.TEXT


DESK = Theme()

# The word shown on whatever is currently selected. Present because colour
# alone may not carry a state (TECHNICAL_SPEC 12): the badge is the text half
# of that rule and the border weight is the shape half.
CHOSEN_BADGE = "נבחר"
