"""One mapping between a normalized point and where a glyph is drawn.

The project had two different normalized-to-pixel mappings that never agreed.
Scoring used ``normalized_to_pixel`` -- ``round(n * (W - 1))`` -- while every
window drew with ``round(n * (W - glyph_width))`` and then moved the glyph's
TOP-LEFT corner to that position.  So the glyph's *centre*, which is what a
person actually looks at, landed at ``n * (W - g) + g/2``.

The gap between the two is ``g/2 - n * (g - 1)``: zero at the middle of the
screen and ``+-g/2`` at the edges.  With the 44pt marker used by the test and
calibration screens (about 64 logical px wide, 80 tall) that is roughly +-23 px
horizontally and +-34 px vertically over the region test targets actually
occupy -- a systematic error of 20-34% against a 100 px button, largest exactly
where accuracy is hardest to achieve and most in doubt.

This module removes the second mapping rather than reconciling the two.
``centered_top_left`` does not restate the formula: it calls
``normalized_to_pixel`` and subtracts half the glyph, so the identity

    drawn_center(centered_top_left(p, g, ...), ...) == normalized_to_pixel(p, g)

holds by construction, in integers, with no tolerance.  A drawing surface and
an error metric can then no longer drift apart, because there is only one
formula left to change.

Nothing here imports Qt, so all of it is testable without a display.
"""

from __future__ import annotations

from gazelink.domain import ContractValidationError, GazePoint, PixelPoint, ScreenGeometry
from gazelink.gaze_engine import normalized_to_pixel


def _glyph_extent(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name} must be an integer number of pixels")
    if value < 0:
        raise ContractValidationError(f"{field_name} must be >= 0")
    return value


def centered_top_left(
    point: GazePoint,
    geometry: ScreenGeometry,
    *,
    glyph_width_px: int,
    glyph_height_px: int,
) -> tuple[int, int]:
    """Where to move a glyph so its centre sits on ``point``'s scored pixel.

    Returns a plain ``tuple[int, int]``, deliberately, rather than a
    ``PixelPoint``: a centred glyph at ``n = 0`` has a negative top-left, and
    ``PixelPoint`` rejects negatives.  That value is correct -- the glyph should
    hang half off the edge, because nudging it back inside is precisely the
    inset error this module exists to remove.  A widget position may be
    negative; a measured screen coordinate may not, so they are different
    types.
    """

    if not isinstance(point, GazePoint):
        raise ContractValidationError("point must be a GazePoint")
    width = _glyph_extent(glyph_width_px, "glyph_width_px")
    height = _glyph_extent(glyph_height_px, "glyph_height_px")
    scored = normalized_to_pixel(point, geometry)
    return (scored.x_px - (width // 2), scored.y_px - (height // 2))


def drawn_center(
    top_left: tuple[int, int],
    *,
    glyph_width_px: int,
    glyph_height_px: int,
) -> PixelPoint:
    """The pixel a glyph placed at ``top_left`` is actually centred on.

    The inverse of :func:`centered_top_left`, and the function a test or a
    recorded run uses to state *where the target really was* rather than where
    the code believed it put it.  Both use ``// 2`` so an odd glyph size round
    trips exactly instead of drifting by one pixel.
    """

    if not isinstance(top_left, tuple) or len(top_left) != 2:
        raise ContractValidationError("top_left must be a two-element tuple")
    left, top = top_left
    for name, value in (("top_left[0]", left), ("top_left[1]", top)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(f"{name} must be an integer")
    width = _glyph_extent(glyph_width_px, "glyph_width_px")
    height = _glyph_extent(glyph_height_px, "glyph_height_px")
    return PixelPoint(left + (width // 2), top + (height // 2))


def legacy_inset_center(
    point: GazePoint,
    surface_width_px: int,
    surface_height_px: int,
    *,
    glyph_width_px: int,
    glyph_height_px: int,
) -> tuple[int, int]:
    """The centre the OLD drawing formula produced. Kept only to test against.

    Not used in production.  It exists so a test can pin how far the previous
    mapping diverged from the scored pixel, which makes a silent revert fail
    loudly instead of quietly reintroducing an edge-weighted bias.
    """

    if not isinstance(point, GazePoint):
        raise ContractValidationError("point must be a GazePoint")
    width = _glyph_extent(glyph_width_px, "glyph_width_px")
    height = _glyph_extent(glyph_height_px, "glyph_height_px")
    clamped_x = min(1.0, max(0.0, point.x))
    clamped_y = min(1.0, max(0.0, point.y))
    left = round(clamped_x * max(0, surface_width_px - width))
    top = round(clamped_y * max(0, surface_height_px - height))
    return (left + (width // 2), top + (height // 2))
