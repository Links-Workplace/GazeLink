"""The shared prediction port must not change what the native engine does.

These tests pin the one property that makes the port safe to introduce: with
``--engine native`` the result is exactly what the untouched
``GazeEstimator`` produced, for accepted and rejected input alike.
"""

from __future__ import annotations

import pytest

from gazelink.domain import (
    EyeFeatures,
    FramePacket,
    HeadPose,
    NormalizedPoint,
    PixelFormat,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_engine import CalibrationModel, GazeEstimator
from gazelink.gaze_model import GazeModelKind, RegressionModel
from gazelink.gaze_predictor import (
    ENGINE_CHOICES,
    EYEGESTURES_ENGINE,
    NATIVE_ENGINE,
    NativeGazePredictor,
)

pytestmark = pytest.mark.unit

_GEOMETRY = ScreenGeometry("primary", 1001, 1001, 1.0)


def _constant_model(x: float, y: float) -> CalibrationModel:
    """A model that predicts ``(x, y)`` for any input; see tests/test_analyze.py."""

    regression = RegressionModel(
        GazeModelKind.LINEAR, (x,) + (0.0,) * 9, (y,) + (0.0,) * 9, ridge_lambda=0.0
    )
    return CalibrationModel.create(
        regression=regression,
        screen_geometry=_GEOMETRY,
        camera_id="camera-0",
        calibration_id="test-calibration",
        trained_at_utc="2026-09-03T00:00:00+00:00",
    )


def _frame() -> FramePacket:
    return FramePacket(
        frame_id=1,
        captured_at_monotonic_ms=1_000.0,
        width=2,
        height=2,
        pixel_format=PixelFormat.RGB24,
        image=bytes(2 * 2 * 3),
    )


def _observation(
    *, state: TrackingState = TrackingState.TRACKED, at_ms: float = 1_000.0
) -> VisionObservation:
    """Mirrors the builder in tests/test_gaze_engine.py so both suites agree
    on what a usable observation looks like."""

    return VisionObservation(
        frame_id=11,
        observed_at_monotonic_ms=at_ms,
        tracking_state=state,
        face_box=None,
        left_eye=EyeFeatures(None, 0.5, 0.5, NormalizedPoint(0.5, 0.5)),
        right_eye=EyeFeatures(None, 0.55, 0.5, NormalizedPoint(0.5, 0.5)),
        head_pose=HeadPose(0.0, 0.0, 0.0),
        overall_confidence=0.5,
    )


def test_engine_choices_are_the_two_documented_names() -> None:
    assert ENGINE_CHOICES == (NATIVE_ENGINE, EYEGESTURES_ENGINE)
    assert NATIVE_ENGINE == "native"
    assert EYEGESTURES_ENGINE == "eyegestures"


def test_native_predictor_returns_exactly_what_the_estimator_returned() -> None:
    estimator = GazeEstimator(_constant_model(0.5, 0.25), live_screen_geometry=_GEOMETRY)
    predictor = NativeGazePredictor(estimator)
    observation = _observation()

    direct = estimator.estimate(observation, now_monotonic_ms=1_000.0)
    through_port = predictor.predict(
        frame=_frame(), observation=observation, now_monotonic_ms=1_000.0
    )

    assert direct.sample is not None
    assert through_port.sample is not None
    assert through_port.sample.raw_normalized == direct.sample.raw_normalized
    assert through_port.sample.screen_position == direct.sample.screen_position
    assert through_port.sample.valid_for_control == direct.sample.valid_for_control
    assert through_port.reason_codes == direct.reason_codes


def test_native_predictor_passes_rejections_through_unchanged() -> None:
    estimator = GazeEstimator(_constant_model(0.5, 0.5), live_screen_geometry=_GEOMETRY)
    predictor = NativeGazePredictor(estimator)

    result = predictor.predict(
        frame=_frame(),
        observation=_observation(state=TrackingState.LOST),
        now_monotonic_ms=1_000.0,
    )

    assert result.sample is None
    assert result.reason_codes  # a rejection always carries a reason
    assert ReasonCode.LOW_CONFIDENCE in result.reason_codes


def test_native_predictor_ignores_the_frame_it_is_handed() -> None:
    """The native path consumed the frame upstream; the port must not care."""

    estimator = GazeEstimator(_constant_model(0.5, 0.25), live_screen_geometry=_GEOMETRY)
    predictor = NativeGazePredictor(estimator)
    observation = _observation()

    broken_frame = FramePacket(
        frame_id=7,
        captured_at_monotonic_ms=1.0,
        width=4,
        height=4,
        pixel_format=PixelFormat.GRAY8,
        image=None,
    )
    result = predictor.predict(
        frame=broken_frame, observation=observation, now_monotonic_ms=1_000.0
    )

    assert result.sample is not None
    assert result.sample.raw_normalized.x == pytest.approx(0.5)


def test_native_predictor_exposes_identity_and_geometry() -> None:
    model = _constant_model(0.5, 0.5)
    estimator = GazeEstimator(model, live_screen_geometry=_GEOMETRY)
    predictor = NativeGazePredictor(estimator)

    assert predictor.engine_name == NATIVE_ENGINE
    assert predictor.screen_geometry == _GEOMETRY
    assert predictor.model_id == model.model_id
    assert predictor.estimator is estimator
    predictor.close()  # must be safe and a no-op
