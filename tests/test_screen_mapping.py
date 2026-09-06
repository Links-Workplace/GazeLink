"""The drawn glyph centre and the scored pixel must be the same pixel.

No Qt, no camera, no display: these run everywhere, every time. The tolerance
is exactly zero, because the identity is integer arithmetic by construction --
anything looser would be hiding a real disagreement rather than measuring one.
"""

from __future__ import annotations

import pytest

from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry
from gazelink.gaze_engine import normalized_to_pixel
from gazelink.screen_mapping import centered_top_left, drawn_center, legacy_inset_center

pytestmark = pytest.mark.unit

# The real display, plus two ordinary ones so the property is not an accident
# of one aspect ratio. dpi_scale varies and must not enter the arithmetic.
_GEOMETRIES = [
    ScreenGeometry("ultrawide", 4096, 1152, 1.25),
    ScreenGeometry("laptop", 1920, 1080, 1.0),
    ScreenGeometry("small", 800, 600, 1.0),
]

# Centre, the four corners, the edge midpoints, and the bounds test targets
# actually occupy (x in [0.12, 0.88], y in [0.08, 0.92]).
_COORDS = [0.0, 0.08, 0.12, 0.25, 0.5, 0.75, 0.88, 0.92, 1.0]

# Odd sizes are included on purpose: they catch a `// 2` that truncates on one
# side of the round trip but not the other.
_GLYPHS = [(1, 1), (2, 2), (63, 63), (64, 80), (96, 96)]


def _points() -> list[GazePoint]:
    return [GazePoint(x, y) for x in _COORDS for y in _COORDS]


@pytest.mark.parametrize("geometry", _GEOMETRIES, ids=lambda g: g.screen_id)
@pytest.mark.parametrize("glyph", _GLYPHS, ids=lambda g: f"{g[0]}x{g[1]}")
def test_drawn_center_is_exactly_the_scored_pixel(
    geometry: ScreenGeometry, glyph: tuple[int, int]
) -> None:
    width, height = glyph
    for point in _points():
        top_left = centered_top_left(point, geometry, glyph_width_px=width, glyph_height_px=height)
        centre = drawn_center(top_left, glyph_width_px=width, glyph_height_px=height)
        assert centre == normalized_to_pixel(point, geometry), (
            f"{point} on {geometry.screen_id} with a {width}x{height} glyph"
        )


def test_a_centered_glyph_may_hang_off_the_left_edge() -> None:
    """Pushing it back inside is the bug, not the fix.

    A target at n=0 belongs at pixel 0. Its glyph must therefore straddle the
    edge. Clamping the top-left to zero would move the drawn centre inward by
    half a glyph -- the exact error this module removes -- so a negative
    coordinate here is the correct answer and must not raise.
    """

    top_left = centered_top_left(
        GazePoint(0.0, 0.0), _GEOMETRIES[0], glyph_width_px=64, glyph_height_px=80
    )

    assert top_left == (-32, -40)
    assert drawn_center(top_left, glyph_width_px=64, glyph_height_px=80).x_px == 0


def test_dpi_scale_does_not_enter_the_mapping() -> None:
    """Two screens differing only in dpi_scale must map identically.

    Widget coordinates are Qt logical pixels and so is the scored ruler, so
    introducing devicePixelRatio here would create a second unit and a new
    class of disagreement.
    """

    logical = ScreenGeometry("a", 4096, 1152, 1.0)
    scaled = ScreenGeometry("b", 4096, 1152, 1.25)

    for point in _points():
        assert centered_top_left(
            point, logical, glyph_width_px=64, glyph_height_px=80
        ) == centered_top_left(point, scaled, glyph_width_px=64, glyph_height_px=80)


@pytest.mark.parametrize("coordinate", [0.12, 0.88])
def test_the_legacy_inset_mapping_was_badly_wrong_at_the_edges(coordinate: float) -> None:
    """Pin the size of the fixed bug so a silent revert fails loudly.

    If someone restores the old ``n * (W - glyph)`` drawing formula, this test
    fails rather than the error quietly reappearing where it is hardest to
    notice -- at the screen edges, where accuracy is already worst.
    """

    geometry = _GEOMETRIES[0]
    point = GazePoint(coordinate, 0.5)

    scored = normalized_to_pixel(point, geometry)
    legacy_x, _ = legacy_inset_center(
        point,
        geometry.width_px,
        geometry.height_px,
        glyph_width_px=64,
        glyph_height_px=80,
    )

    assert abs(legacy_x - scored.x_px) >= 20


def test_the_two_mappings_agree_in_the_middle_which_is_why_this_hid() -> None:
    """At the screen centre the old formula was right, so nothing looked wrong."""

    geometry = _GEOMETRIES[0]
    point = GazePoint(0.5, 0.5)

    legacy = legacy_inset_center(
        point, geometry.width_px, geometry.height_px, glyph_width_px=64, glyph_height_px=80
    )
    scored = normalized_to_pixel(point, geometry)

    assert abs(legacy[0] - scored.x_px) <= 1


@pytest.mark.parametrize("bad", [-1, 1.5, True, "64"])
def test_a_nonsensical_glyph_size_is_refused(bad: object) -> None:
    with pytest.raises(ContractValidationError):
        centered_top_left(
            GazePoint(0.5, 0.5),
            _GEOMETRIES[0],
            glyph_width_px=bad,  # type: ignore[arg-type]
            glyph_height_px=64,
        )


def test_an_off_screen_target_still_maps_through_the_scored_ruler() -> None:
    """``normalized_to_pixel`` clamps for drawing; the identity must survive it."""

    geometry = _GEOMETRIES[0]
    point = GazePoint(1.4, -0.3)

    top_left = centered_top_left(point, geometry, glyph_width_px=64, glyph_height_px=80)
    centre = drawn_center(top_left, glyph_width_px=64, glyph_height_px=80)

    assert centre == normalized_to_pixel(point, geometry)
