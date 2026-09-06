"""Pure decisions extracted from the held-out test window.

These are the parts of the measured screen that can be wrong without anything
looking wrong: a stale marker left on screen, or a window that is not the
surface the errors are scored against. The window class itself needs Qt and a
camera, so the decisions live at module level and are asserted here.
"""

from __future__ import annotations

import pytest

from gazelink.domain import GazePoint, ReasonCode, ScreenGeometry
from gazelink.gaze_engine import GazeEstimationResult
from gazelink.live_validation import (
    LiveValidationPhase,
    LiveValidationTarget,
    LiveValidationView,
)
from gazelink.test_window import overlay_result_for_tick, screen_matches_geometry

pytestmark = pytest.mark.unit

_GEOMETRY = ScreenGeometry("ultrawide", 4096, 1152, 1.25)


class _Tick:
    """Stands in for a RuntimeTick; only its presence matters here."""


def _view(target: LiveValidationTarget | None) -> LiveValidationView:
    """A real view, not a stand-in: the function reads a documented field."""

    return LiveValidationView(
        target=target,
        target_number=1,
        total_targets=10,
        phase=LiveValidationPhase.COLLECTING,
        accepted_samples=0,
        required_samples=5,
        feedback="",
        measurements=(),
    )


def _result() -> GazeEstimationResult:
    return GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE,))


def _target() -> LiveValidationTarget:
    return LiveValidationTarget("TEST_0", GazePoint(0.5, 0.5))


def test_a_dropped_frame_hides_the_marker() -> None:
    """The regression this function exists for.

    ``_on_tick`` used to return on a dropped tick before touching the overlay,
    so the previous frame's prediction stayed on screen and read as current.
    """

    assert overlay_result_for_tick(None, _result(), _view(_target())) is None


def test_no_active_target_hides_the_marker() -> None:
    assert overlay_result_for_tick(_Tick(), _result(), _view(None)) is None


def test_a_missing_result_hides_the_marker() -> None:
    assert overlay_result_for_tick(_Tick(), None, _view(_target())) is None


def test_a_live_frame_with_a_target_draws_that_result() -> None:
    result = _result()

    assert overlay_result_for_tick(_Tick(), result, _view(_target())) is result


def test_the_surface_must_be_the_ruler() -> None:
    assert screen_matches_geometry(4096, 1152, _GEOMETRY) is True


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (4095, 1152),
        (4096, 1151),
        (5120, 1440),  # the same panel in PHYSICAL pixels: still a mismatch
        (1920, 1080),
        (0, 0),
    ],
)
def test_any_size_mismatch_refuses_rather_than_warns(width: int, height: int) -> None:
    """Off-by-one included: a wrong ruler is wrong at every magnitude.

    5120x1440 is the same display in device pixels. Accepting it would silently
    score every error with a 1.25x stretch, which is exactly the class of bug a
    single shared unit is meant to make impossible.
    """

    assert screen_matches_geometry(width, height, _GEOMETRY) is False
