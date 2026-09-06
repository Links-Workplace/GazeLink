"""Deterministic prediction-overlay state; no Qt, camera, or display required."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gazelink.domain import ContractValidationError, GazePoint, GazeSample, PixelPoint, ReasonCode
from gazelink.gaze_engine import GazeEstimationResult
from gazelink.prediction_overlay import (
    PredictionOverlayState,
    compute_prediction_overlay_state,
)

pytestmark = pytest.mark.unit


def _accepted(point: GazePoint) -> GazeEstimationResult:
    sample = GazeSample(
        source_frame_id=1,
        sampled_at_monotonic_ms=1.0,
        raw_normalized=point,
        corrected_normalized=point,
        filtered_normalized=point,
        screen_position=PixelPoint(0, 0),
        screen_id="primary",
        confidence=0.5,
        valid_for_control=False,
    )
    return GazeEstimationResult(sample, ())


def test_no_tick_result_is_hidden_with_no_stale_point() -> None:
    state = compute_prediction_overlay_state(None)
    assert state == PredictionOverlayState(
        visible=False, point=None, reason_text="no tracking this frame"
    )


def test_accepted_prediction_is_visible_with_its_raw_point() -> None:
    point = GazePoint(0.42, 0.61)
    state = compute_prediction_overlay_state(_accepted(point))
    assert state.visible
    assert state.point == point
    assert state.reason_text == ""


def test_rejected_estimation_is_hidden_and_reports_every_reason() -> None:
    result = GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE, ReasonCode.LEFT_EYE_OCCLUDED))
    state = compute_prediction_overlay_state(result)
    assert not state.visible
    assert state.point is None
    assert "LOW_CONFIDENCE" in state.reason_text
    assert "LEFT_EYE_OCCLUDED" in state.reason_text


def test_geometry_mismatch_is_a_readable_failure_not_a_blank_marker() -> None:
    """A wrong-screen overlay model must fail loudly, not silently look idle."""

    result = GazeEstimationResult(None, (ReasonCode.CALIBRATION_INVALID,))
    state = compute_prediction_overlay_state(result)
    assert not state.visible
    assert "CALIBRATION_INVALID" in state.reason_text


def test_state_never_pairs_hidden_with_a_leftover_point() -> None:
    """This is the property the whole overlay exists to guarantee."""

    with pytest.raises(ContractValidationError):
        PredictionOverlayState(visible=False, point=GazePoint(0.5, 0.5), reason_text="")


def test_state_never_claims_visible_without_a_point() -> None:
    with pytest.raises(ContractValidationError):
        PredictionOverlayState(visible=True, point=None, reason_text="")


def test_rejects_a_result_of_the_wrong_type() -> None:
    with pytest.raises(ContractValidationError):
        compute_prediction_overlay_state("not a result")  # type: ignore[arg-type]


def test_consecutive_ticks_never_hold_a_prior_points_visibility() -> None:
    """Simulate one tick with a prediction followed by one tick with none.

    A caller re-deriving state fresh every tick (as PredictionOverlay.update
    does) can never let target1's point leak into the following hidden tick.
    """

    first = compute_prediction_overlay_state(_accepted(GazePoint(0.2, 0.3)))
    second = compute_prediction_overlay_state(None)
    assert first.visible and first.point is not None
    assert not second.visible and second.point is None


# --- an off-screen prediction must not look like an edge prediction ---------


def test_an_off_screen_prediction_is_still_shown_with_its_true_value() -> None:
    """It has to be drawn somewhere, but not as if it landed at the edge.

    The recorded value stays unclamped; this is only about the reader being
    able to tell the difference on screen.
    """

    state = compute_prediction_overlay_state(_accepted(GazePoint(1.4, -0.3)))

    assert state.visible is True
    # The state carries the RAW value; clamping happens only when placing it.
    assert (state.point.x, state.point.y) == (1.4, -0.3)  # type: ignore[union-attr]


def test_the_drawn_point_is_the_raw_prediction_not_a_filtered_one() -> None:
    """What is displayed must be what is measured, or the run proves nothing."""

    accepted = _accepted(GazePoint(0.25, 0.75))
    assert accepted.sample is not None
    filtered = replace(accepted.sample, filtered_normalized=GazePoint(0.9, 0.9))

    state = compute_prediction_overlay_state(GazeEstimationResult(filtered, ()))

    assert state.point == GazePoint(0.25, 0.75)
