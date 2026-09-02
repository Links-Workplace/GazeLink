"""Tests for framework-independent domain contracts."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from gazelink.domain import (
    CalibrationStatus,
    CameraStatus,
    ContractValidationError,
    ControlState,
    EyeFeatures,
    FramePacket,
    GazePoint,
    GazeSample,
    HeadPose,
    InteractionEvent,
    InteractionSource,
    InteractionType,
    NormalizedBox,
    NormalizedPoint,
    PixelFormat,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    SystemState,
    SystemStatus,
    TrackingState,
    VisionObservation,
)


def test_domain_contracts_round_trip_through_json_without_frame_image() -> None:
    frame = FramePacket(3, 101.5, 1280, 720, PixelFormat.BGR24, image=b"not-persisted")
    observation = VisionObservation(
        frame_id=3,
        observed_at_monotonic_ms=106.0,
        tracking_state=TrackingState.TRACKED,
        face_box=NormalizedBox(0.2, 0.1, 0.4, 0.7),
        left_eye=EyeFeatures(NormalizedPoint(0.4, 0.3), 0.82, 0.91, NormalizedPoint(0.45, 0.5)),
        right_eye=EyeFeatures(NormalizedPoint(0.6, 0.3), 0.80, 0.93, NormalizedPoint(0.55, 0.5)),
        head_pose=HeadPose(4.5, -2.0, 0.25),
        overall_confidence=0.90,
    )
    sample = GazeSample(
        source_frame_id=3,
        sampled_at_monotonic_ms=107.0,
        raw_normalized=GazePoint(1.03, 0.55),
        corrected_normalized=GazePoint(1.01, 0.55),
        filtered_normalized=GazePoint(1.0, 0.55),
        screen_position=PixelPoint(5119, 792),
        screen_id="primary",
        confidence=0.89,
        valid_for_control=False,
        reason_codes=(ReasonCode.CLAMPED_TO_SCREEN,),
    )
    event = InteractionEvent(
        event_id=9,
        event_type=InteractionType.PAUSE,
        occurred_at_monotonic_ms=109.0,
        source=InteractionSource.SAFETY,
        position=None,
        confidence=1.0,
        requires_active_control=False,
    )
    status = SystemStatus(
        system_state=SystemState.PAUSED,
        camera_status=CameraStatus.ACTIVE,
        tracking_state=TrackingState.TRACKED,
        calibration_status=CalibrationStatus.VALID,
        control_state=ControlState.PAUSED,
        paused_reason=ReasonCode.PAUSED,
        active_profile_id="ada",
        fps=30.0,
        pipeline_latency_ms=22.3,
    )

    assert FramePacket.from_dict(json.loads(json.dumps(frame.to_dict()))) == FramePacket(
        3, 101.5, 1280, 720, PixelFormat.BGR24
    )
    assert VisionObservation.from_dict(json.loads(json.dumps(observation.to_dict()))) == observation
    assert GazeSample.from_dict(json.loads(json.dumps(sample.to_dict()))) == sample
    assert InteractionEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event
    assert SystemStatus.from_dict(json.loads(json.dumps(status.to_dict()))) == status
    assert "image" not in frame.to_dict()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FramePacket(True, 1.0, 1, 1, PixelFormat.RGB24), "frame_id"),
        (lambda: NormalizedPoint(1.01, 0.5), "within"),
        (lambda: GazePoint(float("nan"), 0.5), "finite"),
        (lambda: PixelPoint(True, 0), "x_px"),
        (lambda: NormalizedBox(0.8, 0.1, 0.3, 0.1), "fit"),
        (lambda: ScreenGeometry("", 1920, 1080, 1.0), "screen_id"),
        (
            lambda: VisionObservation(
                frame_id=1,
                observed_at_monotonic_ms=1.0,
                tracking_state=TrackingState.LOST,
                face_box=None,
                left_eye=None,
                right_eye=None,
                head_pose=None,
                overall_confidence=0.0,
                reason_codes=[ReasonCode.FACE_NOT_FOUND],  # type: ignore[arg-type]
            ),
            "immutable tuple",
        ),
        (
            lambda: EyeFeatures(NormalizedPoint(0.4, 0.3), 0.8, 0.9, iris_in_eye=(0.5, 0.5)),  # type: ignore[arg-type]
            "iris_in_eye",
        ),
    ],
)
def test_invalid_domain_values_are_rejected_with_clear_errors(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(ContractValidationError, match=message):
        factory()
