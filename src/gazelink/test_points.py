"""Randomized held-out test targets, deliberately disjoint from calibration.

The 9 calibration targets in :func:`gazelink.calibration.targets_for_screen_geometry`
are the points a model is *trained* on.  Measuring that same model on those
same points answers "did it memorize?", not "did it learn".  The five fixed
targets in :func:`gazelink.live_validation.default_live_validation_targets`
are better but still partly overlap the calibration grid -- ``CENTER`` is
literally calibration target 4, and ``LEFT_CENTER``/``RIGHT_CENTER`` sit on
the calibration mid-row.

This module produces targets that are held out by construction: every point
is a minimum pixel distance away from every calibration target and from every
other test point, drawn from a seeded generator so the same seed reproduces
the same points across runs and across days.  Nothing here trains, maps, or
corrects gaze; it only decides *where to look*.
"""

from __future__ import annotations

import math
import random

from gazelink.calibration import targets_for_screen_geometry
from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry
from gazelink.live_validation import LiveValidationTarget

# --- Tunable working area ------------------------------------------------
#
# Normalized [0.0, 1.0] screen coordinates, top-left origin.  Distances are
# expressed in PIXELS, not normalized units, because this product's reference
# display is 4096x1152: 0.05 normalized is 205px horizontally but only 58px
# vertically, so a single normalized threshold would mean two very different
# separations on the two axes.
#
# On the reference ultrawide display the calibration grid spans x 0.15-0.85
# and y 0.05-0.95.  TEST_AREA_X_* deliberately reaches OUTSIDE that span:
# those points measure extrapolation rather than interpolation, which is a
# different question and is reported separately by analyze.py.  TEST_AREA_Y_*
# stays inside the calibrated span on purpose, so only one axis varies.
TEST_AREA_X_MIN = 0.12
TEST_AREA_X_MAX = 0.88
TEST_AREA_Y_MIN = 0.08
TEST_AREA_Y_MAX = 0.92

TEST_POINT_COUNT = 10
DEFAULT_TEST_SEED = 20260903

MIN_DISTANCE_FROM_CALIBRATION_PX = 250.0
MIN_DISTANCE_BETWEEN_TEST_POINTS_PX = 200.0
MAX_PLACEMENT_ATTEMPTS = 5000


def _pixel_distance(left: GazePoint, right: GazePoint, geometry: ScreenGeometry) -> float:
    """Screen-pixel separation, matching live_validation._pixel_error's scaling."""

    return math.hypot(
        (left.x - right.x) * (geometry.width_px - 1),
        (left.y - right.y) * (geometry.height_px - 1),
    )


def calibration_bounding_box(
    geometry: ScreenGeometry,
) -> tuple[float, float, float, float]:
    """Return ``(x_min, x_max, y_min, y_max)`` of the real calibration grid.

    Read from :func:`targets_for_screen_geometry` rather than restated as
    constants, so it cannot drift from the grid actually used for training.
    A test point outside this box is an extrapolation, and is reported as
    such instead of being silently averaged into one interpolation number.
    """

    targets = targets_for_screen_geometry(geometry)
    xs = [target.screen_position.x for target in targets]
    ys = [target.screen_position.y for target in targets]
    return (min(xs), max(xs), min(ys), max(ys))


def is_extrapolated(point: GazePoint, geometry: ScreenGeometry) -> bool:
    """Whether ``point`` falls outside the calibration grid's bounding box."""

    x_min, x_max, y_min, y_max = calibration_bounding_box(geometry)
    return not (x_min <= point.x <= x_max and y_min <= point.y <= y_max)


def generate_test_targets(
    geometry: ScreenGeometry,
    *,
    seed: int = DEFAULT_TEST_SEED,
    count: int = TEST_POINT_COUNT,
    area: tuple[float, float, float, float] = (
        TEST_AREA_X_MIN,
        TEST_AREA_X_MAX,
        TEST_AREA_Y_MIN,
        TEST_AREA_Y_MAX,
    ),
    min_distance_from_calibration_px: float = MIN_DISTANCE_FROM_CALIBRATION_PX,
    min_distance_between_points_px: float = MIN_DISTANCE_BETWEEN_TEST_POINTS_PX,
) -> tuple[LiveValidationTarget, ...]:
    """Return ``count`` reproducible held-out targets, or raise.

    Uses a private :class:`random.Random` seeded with ``seed`` -- never the
    global ``random`` module -- so the same seed yields byte-identical points
    no matter what else in the process has drawn random numbers.

    Raises :class:`ContractValidationError` if it cannot place ``count``
    points under the separation constraints.  It deliberately does NOT return
    a shorter tuple: fewer test points is a weaker measurement that would look
    identical in the output, and silently weakening the measurement is exactly
    the failure this module exists to prevent.
    """

    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be a ScreenGeometry")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ContractValidationError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractValidationError("seed must be an integer")
    x_min, x_max, y_min, y_max = area
    if not 0.0 <= x_min < x_max <= 1.0 or not 0.0 <= y_min < y_max <= 1.0:
        raise ContractValidationError("area must be an ordered box within [0.0, 1.0]")
    for name, value in (
        ("min_distance_from_calibration_px", min_distance_from_calibration_px),
        ("min_distance_between_points_px", min_distance_between_points_px),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
            raise ContractValidationError(f"{name} must be a non-negative number")

    calibration = tuple(
        GazePoint(target.screen_position.x, target.screen_position.y)
        for target in targets_for_screen_geometry(geometry)
    )
    generator = random.Random(seed)
    chosen: list[GazePoint] = []
    for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
        if len(chosen) == count:
            break
        candidate = GazePoint(
            generator.uniform(x_min, x_max),
            generator.uniform(y_min, y_max),
        )
        if any(
            _pixel_distance(candidate, target, geometry) < min_distance_from_calibration_px
            for target in calibration
        ):
            continue
        if any(
            _pixel_distance(candidate, other, geometry) < min_distance_between_points_px
            for other in chosen
        ):
            continue
        chosen.append(candidate)
    if len(chosen) != count:
        raise ContractValidationError(
            f"could only place {len(chosen)} of {count} test points in "
            f"{MAX_PLACEMENT_ATTEMPTS} attempts; widen the working area or lower "
            f"the separation thresholds -- do not silently measure fewer points"
        )
    return tuple(LiveValidationTarget(f"TEST_{index}", point) for index, point in enumerate(chosen))
