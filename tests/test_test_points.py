"""Deterministic held-out test-target generation; no Qt, camera, or display."""

from __future__ import annotations

import math

import pytest

from gazelink.calibration import targets_for_screen_geometry
from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry
from gazelink.eyegestures_engine import calibration_grid_points
from gazelink.test_points import (
    DEFAULT_TEST_SEED,
    MIN_DISTANCE_BETWEEN_TEST_POINTS_PX,
    MIN_DISTANCE_FROM_CALIBRATION_PX,
    TEST_AREA_X_MAX,
    TEST_AREA_X_MIN,
    TEST_AREA_Y_MAX,
    TEST_AREA_Y_MIN,
    TEST_POINT_COUNT,
    calibration_bounding_box,
    generate_test_targets,
    is_extrapolated,
)

pytestmark = pytest.mark.unit


def _reference_geometry() -> ScreenGeometry:
    """The operator's actual display; placement must succeed on this one."""

    return ScreenGeometry("LS49C95xU", 4096, 1152, 1.25)


def _distance_px(left: GazePoint, right: GazePoint, geometry: ScreenGeometry) -> float:
    return math.hypot(
        (left.x - right.x) * (geometry.width_px - 1),
        (left.y - right.y) * (geometry.height_px - 1),
    )


def test_same_seed_reproduces_identical_targets() -> None:
    geometry = _reference_geometry()
    first = generate_test_targets(geometry, seed=DEFAULT_TEST_SEED)
    second = generate_test_targets(geometry, seed=DEFAULT_TEST_SEED)
    assert first == second


def test_different_seed_produces_different_targets() -> None:
    geometry = _reference_geometry()
    assert generate_test_targets(geometry, seed=1) != generate_test_targets(geometry, seed=2)


def test_generation_is_immune_to_global_random_state() -> None:
    """A private Random means unrelated draws elsewhere cannot shift the points."""

    import random

    geometry = _reference_geometry()
    random.seed(1)
    first = generate_test_targets(geometry)
    random.seed(999)
    [random.random() for _ in range(50)]
    assert generate_test_targets(geometry) == first


def test_full_count_is_placeable_on_the_reference_display() -> None:
    """10 points must fit the real 4096x1152 screen under the real thresholds.

    If this ever fails the fix is to lower a separation threshold and say so,
    never to quietly measure fewer points.
    """

    targets = generate_test_targets(_reference_geometry())
    assert len(targets) == TEST_POINT_COUNT


def test_every_target_lies_inside_the_working_area() -> None:
    for target in generate_test_targets(_reference_geometry()):
        assert TEST_AREA_X_MIN <= target.screen_position.x <= TEST_AREA_X_MAX
        assert TEST_AREA_Y_MIN <= target.screen_position.y <= TEST_AREA_Y_MAX


def test_every_target_is_held_out_from_every_calibration_target() -> None:
    """This is the property the whole measurement rests on."""

    geometry = _reference_geometry()
    calibration = [
        GazePoint(target.screen_position.x, target.screen_position.y)
        for target in targets_for_screen_geometry(geometry)
    ]
    for target in generate_test_targets(geometry):
        for calibration_point in calibration:
            distance = _distance_px(target.screen_position, calibration_point, geometry)
            assert distance >= MIN_DISTANCE_FROM_CALIBRATION_PX


def test_targets_are_separated_from_each_other() -> None:
    geometry = _reference_geometry()
    targets = generate_test_targets(geometry)
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            distance = _distance_px(target.screen_position, other.screen_position, geometry)
            assert distance >= MIN_DISTANCE_BETWEEN_TEST_POINTS_PX


def test_target_names_are_unique_and_ordered() -> None:
    targets = generate_test_targets(_reference_geometry())
    assert [target.name for target in targets] == [f"TEST_{index}" for index in range(len(targets))]


def test_impossible_area_raises_instead_of_returning_fewer_points() -> None:
    """A weaker measurement must never masquerade as a completed one."""

    geometry = _reference_geometry()
    with pytest.raises(ContractValidationError) as error:
        generate_test_targets(geometry, area=(0.49, 0.51, 0.49, 0.51), count=8)
    assert "do not silently measure fewer points" in str(error.value)


def test_rejects_invalid_arguments() -> None:
    geometry = _reference_geometry()
    with pytest.raises(ContractValidationError):
        generate_test_targets(geometry, count=0)
    with pytest.raises(ContractValidationError):
        generate_test_targets(geometry, area=(0.8, 0.2, 0.1, 0.9))
    with pytest.raises(ContractValidationError):
        generate_test_targets(geometry, min_distance_from_calibration_px=-1.0)
    with pytest.raises(ContractValidationError):
        generate_test_targets("not a geometry")  # type: ignore[arg-type]


def test_bounding_box_tracks_the_real_calibration_grid() -> None:
    """Read from the grid itself, so it cannot drift from what training used."""

    geometry = _reference_geometry()
    targets = targets_for_screen_geometry(geometry)
    x_min, x_max, y_min, y_max = calibration_bounding_box(geometry)
    assert x_min == min(target.screen_position.x for target in targets)
    assert x_max == max(target.screen_position.x for target in targets)
    assert y_min == min(target.screen_position.y for target in targets)
    assert y_max == max(target.screen_position.y for target in targets)


def test_extrapolation_flag_matches_the_calibration_box() -> None:
    geometry = _reference_geometry()
    x_min, x_max, y_min, y_max = calibration_bounding_box(geometry)
    inside = GazePoint((x_min + x_max) / 2, (y_min + y_max) / 2)
    outside_x = GazePoint(max(0.0, x_min - 0.02), (y_min + y_max) / 2)
    outside_y = GazePoint((x_min + x_max) / 2, max(0.0, y_min - 0.02))
    assert not is_extrapolated(inside, geometry)
    assert is_extrapolated(outside_x, geometry)
    assert is_extrapolated(outside_y, geometry)


# --- held out from WHICH calibration set? -----------------------------------


def _nearest_px(points: list[GazePoint], point: GazePoint, geometry: ScreenGeometry) -> float:
    return min(
        math.hypot(
            (point.x - other.x) * (geometry.width_px - 1),
            (point.y - other.y) * (geometry.height_px - 1),
        )
        for other in points
    )


def test_the_default_target_set_is_not_held_out_from_the_eyegestures_grid() -> None:
    """Pins the defect, so the fix cannot be quietly dropped.

    ``generate_test_targets`` defaults to avoiding GAZELINK's own 9 calibration
    points. EyeGestures calibrates on a different 36-point grid, so on that
    engine the default set is not held out at all -- some targets sit right on
    top of points the library was trained on. This asserts the overlap is real,
    so that if someone measures an external engine with the default set, the
    reason the number is not a generalisation measurement is written down here.
    """

    geometry = ScreenGeometry("ultrawide", 4096, 1152, 1.25)
    grid = list(calibration_grid_points())

    targets = generate_test_targets(geometry)
    too_close = [
        target for target in targets if _nearest_px(grid, target.screen_position, geometry) < 250.0
    ]

    assert len(too_close) >= 1


def test_avoid_points_produces_a_set_held_out_from_the_engines_own_grid() -> None:
    """The measurement must be able to claim generalisation, and mean it."""

    geometry = ScreenGeometry("ultrawide", 4096, 1152, 1.25)
    grid = list(calibration_grid_points())
    native = [
        GazePoint(target.screen_position.x, target.screen_position.y)
        for target in targets_for_screen_geometry(geometry)
    ]

    targets = generate_test_targets(geometry, avoid_points=grid + native)

    assert len(targets) == 10
    for target in targets:
        assert _nearest_px(grid, target.screen_position, geometry) >= 250.0
        assert _nearest_px(native, target.screen_position, geometry) >= 250.0


def test_omitting_avoid_points_keeps_the_previous_behaviour_exactly() -> None:
    """Backward compatibility is what lets the native path stay untouched."""

    geometry = ScreenGeometry("ultrawide", 4096, 1152, 1.25)
    native = [
        GazePoint(target.screen_position.x, target.screen_position.y)
        for target in targets_for_screen_geometry(geometry)
    ]

    assert generate_test_targets(geometry) == generate_test_targets(geometry, avoid_points=native)


def test_avoid_points_rejects_the_wrong_point_type() -> None:
    """``targets_for_screen_geometry`` yields NormalizedPoint, not GazePoint.

    Silently accepting either would make it easy to pass the wrong coordinate
    space and never find out.
    """

    geometry = ScreenGeometry("ultrawide", 4096, 1152, 1.25)
    wrong = [target.screen_position for target in targets_for_screen_geometry(geometry)]

    with pytest.raises(ContractValidationError):
        generate_test_targets(geometry, avoid_points=wrong)  # type: ignore[arg-type]
