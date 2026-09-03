"""Deterministic tests for the EyeGestures adapter: no camera, no Qt, no GPL library.

The external library is replaced by an injected double shaped like the slice of
its API the adapter uses. That keeps the whole suite runnable without the
optional ``eyegestures`` extra installed, and -- more importantly -- lets the
safety properties be asserted directly, including the ones that only appear
when the library misbehaves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gazelink.domain import (
    ContractValidationError,
    FramePacket,
    GazePoint,
    PixelFormat,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.eyegestures_engine import (
    EyeGesturesGazePredictor,
    build_calibration_map,
    frame_to_rgb_array,
)
from gazelink.gaze_engine import normalized_to_pixel

pytestmark = pytest.mark.unit

_GEOMETRY = ScreenGeometry("primary", 1000, 1000, 1.0)


# --- doubles ---------------------------------------------------------------


class _FakeGazeEvent:
    def __init__(self, point: Any) -> None:
        self.point = point


class _FakeCalibrationEvent:
    def __init__(self, point: Any, acceptance_radius: float = 100.0) -> None:
        self.point = point
        self.acceptance_radius = acceptance_radius


class _FakeGestures:
    """Scripted stand-in for ``EyeGestures_v2``; records what it was asked."""

    def __init__(self, script: list[tuple[Any, Any]] | None = None) -> None:
        self._script = list(script or [])
        self.step_calls = 0
        self.calibration_flags: list[bool] = []
        self.calibration_map: Any = None
        self.classical_impact: int | None = None
        self.fixation: float | None = None

    def uploadCalibrationMap(  # noqa: N802 - mirrors the EyeGestures API
        self, points: Any, context: str = "main"
    ) -> None:
        self.calibration_map = points

    def setClassicalImpact(self, impact: int) -> None:  # noqa: N802
        self.classical_impact = impact

    def setFixation(self, fixation: float) -> None:  # noqa: N802
        self.fixation = fixation

    def step(
        self, frame: Any, calibration: bool, width: int, height: int, context: str = "main"
    ) -> tuple[Any, Any]:
        self.step_calls += 1
        self.calibration_flags.append(calibration)
        if self._script:
            return self._script.pop(0)
        return (None, None)


def _frame(
    *,
    frame_id: int = 1,
    width: int = 4,
    height: int = 2,
    pixel_format: PixelFormat = PixelFormat.RGB24,
    image: bytes | None = None,
) -> FramePacket:
    payload = image if image is not None else bytes(range(width * height * 3))
    return FramePacket(
        frame_id=frame_id,
        captured_at_monotonic_ms=1_000.0,
        width=width,
        height=height,
        pixel_format=pixel_format,
        image=payload,
    )


def _observation(
    *,
    state: TrackingState = TrackingState.TRACKED,
    confidence: float = 0.75,
    reason_codes: tuple[ReasonCode, ...] = (),
) -> VisionObservation:
    """A minimal observation: this adapter reads only state/reasons/confidence."""

    return VisionObservation(
        frame_id=1,
        observed_at_monotonic_ms=1_000.0,
        tracking_state=state,
        face_box=None,
        left_eye=None,
        right_eye=None,
        head_pose=None,
        overall_confidence=confidence,
        reason_codes=reason_codes,
    )


def _calibrated_predictor(gestures: _FakeGestures, *, points: int = 2) -> EyeGesturesGazePredictor:
    """Drive the adapter through ``points`` completed calibration targets."""

    predictor = EyeGesturesGazePredictor(
        screen_geometry=_GEOMETRY, gestures=gestures, calibration_points=points
    )
    return predictor


# --- frame conversion -------------------------------------------------------


def test_rgb_frame_is_converted_unchanged() -> None:
    frame = _frame(pixel_format=PixelFormat.RGB24)
    array = frame_to_rgb_array(frame)
    assert array is not None
    assert array.shape == (2, 4, 3)
    assert array[0, 0].tolist() == [0, 1, 2]


def test_bgr_frame_channels_are_swapped_to_rgb() -> None:
    frame = _frame(pixel_format=PixelFormat.BGR24)
    array = frame_to_rgb_array(frame)
    assert array is not None
    # Source pixel (B, G, R) == (0, 1, 2) must arrive as (R, G, B) == (2, 1, 0).
    assert array[0, 0].tolist() == [2, 1, 0]
    assert array.flags["C_CONTIGUOUS"]  # the reversed view must not leak out


def test_frame_without_image_bytes_is_rejected() -> None:
    frame = FramePacket(
        frame_id=1,
        captured_at_monotonic_ms=1.0,
        width=4,
        height=2,
        pixel_format=PixelFormat.RGB24,
        image=None,
    )
    assert frame_to_rgb_array(frame) is None


def test_unsupported_pixel_format_is_rejected() -> None:
    frame = _frame(pixel_format=PixelFormat.GRAY8, image=bytes(4 * 2))
    assert frame_to_rgb_array(frame) is None


def test_buffer_that_disagrees_with_geometry_is_rejected() -> None:
    """A short buffer must not be reshaped into a partly-valid image."""

    frame = _frame(image=bytes(5))
    assert frame_to_rgb_array(frame) is None


def test_frame_must_be_a_frame_packet() -> None:
    with pytest.raises(ContractValidationError):
        frame_to_rgb_array("not a frame")  # type: ignore[arg-type]


# --- calibration map --------------------------------------------------------


def test_calibration_map_is_deterministic_for_a_seed() -> None:
    first = build_calibration_map(seed=7)
    second = build_calibration_map(seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, build_calibration_map(seed=8))
    assert first.shape == (36, 2)
    assert first.min() >= 0.0 and first.max() <= 1.0


def test_constructor_configures_the_library_once() -> None:
    gestures = _FakeGestures()
    EyeGesturesGazePredictor(screen_geometry=_GEOMETRY, gestures=gestures)
    assert gestures.calibration_map is not None
    assert gestures.classical_impact == 2
    assert gestures.fixation == 1.0


# --- THE safety property: an unfitted engine must never emit a point --------


def test_uncalibrated_engine_never_emits_its_zero_zero_placeholder() -> None:
    """EyeGestures returns (0, 0) before it is fitted -- the screen's top-left
    corner, a perfectly legitimate-looking coordinate. It must not get out."""

    gestures = _FakeGestures([(_FakeGazeEvent((0.0, 0.0)), _FakeCalibrationEvent((10.0, 10.0)))])
    predictor = _calibrated_predictor(gestures, points=3)

    result = predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=1_000.0)

    assert result.sample is None
    assert ReasonCode.CALIBRATION_INVALID in result.reason_codes
    assert not predictor.is_calibrated


def _drive_calibration(gestures: _FakeGestures, predictor: EyeGesturesGazePredictor) -> None:
    """Consume the scripted calibration frames so the predictor is READY."""

    for index in range(2):
        predictor.predict(
            frame=_frame(), observation=_observation(), now_monotonic_ms=float(index + 1)
        )


def test_low_confidence_observation_still_feeds_the_engine_but_emits_nothing() -> None:
    """The engine is fed every frame so its own calibration is not starved,
    while our policy still decides whether any point may be emitted.

    Gating the feed as well was measured to stretch a calibration pass to
    roughly five minutes, because only ~18% of frames pass our policy.
    """

    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (_FakeGazeEvent((500.0, 250.0)), _FakeCalibrationEvent((20.0, 20.0))),
        ]
    )
    predictor = _calibrated_predictor(gestures, points=1)
    _drive_calibration(gestures, predictor)
    assert predictor.is_calibrated
    calls_before = gestures.step_calls

    result = predictor.predict(
        frame=_frame(),
        observation=_observation(
            state=TrackingState.LOW_CONFIDENCE, reason_codes=(ReasonCode.LOW_CONFIDENCE,)
        ),
        now_monotonic_ms=1_000.0,
    )

    assert gestures.step_calls == calls_before + 1  # the engine WAS fed
    assert result.sample is None  # but nothing was emitted
    assert ReasonCode.LOW_CONFIDENCE in result.reason_codes


def test_tracking_loss_reports_our_own_reason_codes() -> None:
    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (_FakeGazeEvent((500.0, 250.0)), _FakeCalibrationEvent((20.0, 20.0))),
        ]
    )
    predictor = _calibrated_predictor(gestures, points=1)
    _drive_calibration(gestures, predictor)

    result = predictor.predict(
        frame=_frame(),
        observation=_observation(
            state=TrackingState.LOST, reason_codes=(ReasonCode.FACE_NOT_FOUND,)
        ),
        now_monotonic_ms=1_000.0,
    )

    assert result.sample is None
    assert result.reason_codes == (ReasonCode.FACE_NOT_FOUND,)


def test_calibration_advances_even_on_frames_our_policy_rejects() -> None:
    """The regression that made the screen look hung: calibration must not
    depend on our own confidence policy accepting the frame."""

    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (None, _FakeCalibrationEvent((30.0, 30.0))),
        ]
    )
    predictor = _calibrated_predictor(gestures, points=5)
    rejected = _observation(state=TrackingState.LOST, reason_codes=(ReasonCode.FACE_NOT_FOUND,))

    for index in range(3):
        predictor.predict(frame=_frame(), observation=rejected, now_monotonic_ms=float(index + 1))

    assert gestures.step_calls == 3
    assert predictor.calibration_view().completed_points == 2


def test_a_missing_observation_still_feeds_calibration_and_emits_nothing() -> None:
    gestures = _FakeGestures([(_FakeGazeEvent((10.0, 10.0)), _FakeCalibrationEvent((10.0, 10.0)))])
    predictor = _calibrated_predictor(gestures, points=5)

    result = predictor.predict(frame=_frame(), observation=None, now_monotonic_ms=1.0)

    assert gestures.step_calls == 1
    assert result.sample is None
    assert predictor.calibration_view().target_normalized is not None


def test_unconvertible_frame_is_an_error_not_a_guess() -> None:
    gestures = _FakeGestures()
    predictor = _calibrated_predictor(gestures)

    result = predictor.predict(
        frame=_frame(image=bytes(3)),  # buffer disagrees with declared geometry
        observation=_observation(),
        now_monotonic_ms=1_000.0,
    )

    assert result.sample is None
    assert result.reason_codes == (ReasonCode.ERROR,)
    assert gestures.step_calls == 0


def test_no_face_from_the_library_is_reported_not_invented() -> None:
    gestures = _FakeGestures([(None, None)])
    predictor = _calibrated_predictor(gestures)

    result = predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=1_000.0)

    assert result.sample is None
    assert result.reason_codes == (ReasonCode.FACE_NOT_FOUND,)


class _ExplodingGestures(_FakeGestures):
    """EyeGestures 3.2.4 raises instead of returning empty output when
    MediaPipe finds no face -- reproduced here as the real library does it."""

    def step(
        self, frame: Any, calibration: bool, width: int, height: int, context: str = "main"
    ) -> tuple[Any, Any]:
        self.step_calls += 1
        raise TypeError("'NoneType' object is not subscriptable")


def test_a_crash_inside_the_external_library_is_contained_as_a_lost_frame() -> None:
    """Verified upstream defect: their face.py dereferences None on any frame
    without a face -- the normal state before the user sits down. It must not
    escape the adapter and take the window down."""

    gestures = _ExplodingGestures()
    predictor = _calibrated_predictor(gestures)

    result = predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=1_000.0)

    assert gestures.step_calls == 1  # we really did call into the library
    assert result.sample is None
    assert result.reason_codes == (ReasonCode.FACE_NOT_FOUND,)


def test_the_adapter_survives_repeated_library_crashes() -> None:
    """A crash on every frame must degrade to 'no gaze', not accumulate state."""

    gestures = _ExplodingGestures()
    predictor = _calibrated_predictor(gestures)
    observation = _observation()

    for index in range(5):
        result = predictor.predict(
            frame=_frame(), observation=observation, now_monotonic_ms=float(index + 1)
        )
        assert result.sample is None
    assert predictor.is_calibrated is False
    assert gestures.step_calls == 5


# --- calibration progress ---------------------------------------------------


def test_first_target_sighting_completes_nothing_and_moves_advance_the_count() -> None:
    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),  # first sighting
            (None, _FakeCalibrationEvent((10.0, 10.0))),  # unchanged -> no progress
            (None, _FakeCalibrationEvent((20.0, 20.0))),  # moved -> 1 completed
        ]
    )
    predictor = _calibrated_predictor(gestures, points=5)
    observation = _observation()

    predictor.predict(frame=_frame(), observation=observation, now_monotonic_ms=1.0)
    assert predictor.calibration_view().completed_points == 0
    predictor.predict(frame=_frame(), observation=observation, now_monotonic_ms=2.0)
    assert predictor.calibration_view().completed_points == 0
    predictor.predict(frame=_frame(), observation=observation, now_monotonic_ms=3.0)
    assert predictor.calibration_view().completed_points == 1


def test_calibration_view_reports_target_and_progress_while_active() -> None:
    gestures = _FakeGestures(
        [(None, _FakeCalibrationEvent((250.0, 500.0), acceptance_radius=42.0))]
    )
    predictor = _calibrated_predictor(gestures, points=4)
    predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=1.0)

    view = predictor.calibration_view()
    assert view.active is True
    assert view.target_normalized == GazePoint(0.25, 0.5)
    assert view.acceptance_radius_px == 42.0
    assert view.progress_text == "0/4"


def test_calibration_flag_passed_to_the_library_flips_only_after_completion() -> None:
    """The flag is chosen before the call, so the frame that *completes*
    calibration still requests it; the next frame is the first that does not."""

    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),  # first sighting
            (None, _FakeCalibrationEvent((20.0, 20.0))),  # 1 completed
            (None, _FakeCalibrationEvent((30.0, 30.0))),  # 2 completed -> done
            (None, _FakeCalibrationEvent((30.0, 30.0))),
        ]
    )
    predictor = _calibrated_predictor(gestures, points=2)
    observation = _observation()
    for index in range(4):
        predictor.predict(
            frame=_frame(), observation=observation, now_monotonic_ms=float(index + 1)
        )

    assert predictor.is_calibrated
    assert gestures.calibration_flags == [True, True, True, False]
    assert predictor.calibration_view().active is False
    assert predictor.calibration_view().target_normalized is None


# --- the happy path ---------------------------------------------------------


def _drive_to_calibrated(gestures: _FakeGestures) -> EyeGesturesGazePredictor:
    predictor = _calibrated_predictor(gestures, points=1)
    observation = _observation()
    predictor.predict(frame=_frame(), observation=observation, now_monotonic_ms=1.0)
    predictor.predict(frame=_frame(), observation=observation, now_monotonic_ms=2.0)
    assert predictor.is_calibrated
    return predictor


def test_calibrated_engine_converts_pixels_to_the_shared_sample_type() -> None:
    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (_FakeGazeEvent((500.0, 250.0)), _FakeCalibrationEvent((20.0, 20.0))),
        ]
    )
    predictor = _drive_to_calibrated(gestures)

    result = predictor.predict(
        frame=_frame(frame_id=9),
        observation=_observation(confidence=0.75),
        now_monotonic_ms=3_000.0,
    )

    sample = result.sample
    assert sample is not None
    expected = GazePoint(0.5, 0.25)
    assert sample.raw_normalized == expected
    # This engine applies neither local correction nor its own filtering.
    assert sample.corrected_normalized == expected
    assert sample.filtered_normalized == expected
    assert sample.screen_position == normalized_to_pixel(expected, _GEOMETRY)
    assert sample.source_frame_id == 9
    assert sample.sampled_at_monotonic_ms == 3_000.0
    assert sample.screen_id == "primary"
    # Confidence is ours, measured by our vision stage -- not invented.
    assert sample.confidence == 0.75
    assert result.reason_codes == ()


def test_no_engine_output_is_ever_authorized_for_control() -> None:
    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (_FakeGazeEvent((500.0, 250.0)), _FakeCalibrationEvent((20.0, 20.0))),
        ]
    )
    predictor = _drive_to_calibrated(gestures)
    result = predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=3.0)
    assert result.sample is not None
    assert result.sample.valid_for_control is False


def test_point_outside_the_screen_is_flagged_like_the_native_engine() -> None:
    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (_FakeGazeEvent((1500.0, -30.0)), _FakeCalibrationEvent((20.0, 20.0))),
        ]
    )
    predictor = _drive_to_calibrated(gestures)

    result = predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=3.0)

    assert result.sample is not None
    assert ReasonCode.OUT_OF_RANGE in result.reason_codes
    assert ReasonCode.CLAMPED_TO_SCREEN in result.reason_codes


def test_non_finite_point_from_the_library_is_rejected() -> None:
    gestures = _FakeGestures(
        [
            (None, _FakeCalibrationEvent((10.0, 10.0))),
            (None, _FakeCalibrationEvent((20.0, 20.0))),
            (_FakeGazeEvent((float("nan"), 5.0)), _FakeCalibrationEvent((20.0, 20.0))),
        ]
    )
    predictor = _drive_to_calibrated(gestures)

    result = predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=3.0)

    assert result.sample is None
    assert result.reason_codes == (ReasonCode.ERROR,)


# --- lifecycle --------------------------------------------------------------


def test_closed_predictor_refuses_to_predict() -> None:
    predictor = _calibrated_predictor(_FakeGestures())
    predictor.close()
    predictor.close()  # repeated close must stay safe
    with pytest.raises(RuntimeError):
        predictor.predict(frame=_frame(), observation=_observation(), now_monotonic_ms=1.0)


def test_geometry_and_engine_name_are_exposed() -> None:
    predictor = _calibrated_predictor(_FakeGestures())
    assert predictor.engine_name == "eyegestures"
    assert predictor.screen_geometry == _GEOMETRY


def test_calibration_points_must_be_a_positive_integer() -> None:
    with pytest.raises(ContractValidationError):
        EyeGesturesGazePredictor(
            screen_geometry=_GEOMETRY, gestures=_FakeGestures(), calibration_points=0
        )
