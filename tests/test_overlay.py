"""Offscreen tests for the provider-neutral M1 debug overlay."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from gazelink.domain import (
    EyeFeatures,
    FramePacket,
    HeadPose,
    NormalizedBox,
    NormalizedPoint,
    PixelFormat,
    ReasonCode,
    TrackingState,
    VisionObservation,
)
from gazelink.overlay import (
    CircleCommand,
    DebugOverlayRenderer,
    OverlayCommand,
    OverlayStatus,
    OverlayViewModel,
    RectangleCommand,
    TextCommand,
)


def _shape_commands(view: OverlayViewModel) -> list[OverlayCommand]:
    """Return every drawn landmark shape; text diagnostics are never landmarks."""

    return [
        command
        for command in view.commands
        if isinstance(command, (RectangleCommand, CircleCommand))
    ]


def _texts(view: OverlayViewModel) -> list[str]:
    return [command.text for command in view.commands if isinstance(command, TextCommand)]


def _frame(frame_id: int = 7) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        captured_at_monotonic_ms=100.0,
        width=640,
        height=480,
        pixel_format=PixelFormat.BGR24,
        image=b"synthetic-frame-bytes",
    )


def _observation(
    *,
    frame_id: int = 7,
    state: TrackingState = TrackingState.TRACKED,
    reasons: tuple[ReasonCode, ...] = (),
) -> VisionObservation:
    return VisionObservation(
        frame_id=frame_id,
        observed_at_monotonic_ms=100.0,
        tracking_state=state,
        face_box=NormalizedBox(0.1, 0.2, 0.5, 0.4),
        left_eye=EyeFeatures(NormalizedPoint(0.25, 0.45), openness=0.7, confidence=0.9),
        right_eye=EyeFeatures(NormalizedPoint(0.55, 0.45), openness=0.6, confidence=0.9),
        head_pose=HeadPose(yaw_deg=3.0, pitch_deg=-2.0, roll_deg=1.0),
        overall_confidence=0.88,
        reason_codes=reasons,
    )


def test_tracked_overlay_is_frame_aligned_and_includes_required_diagnostics() -> None:
    view = DebugOverlayRenderer().render(_frame(), _observation(), fps=29.8, latency_ms=18.2)

    assert view.status is OverlayStatus.TRACKED
    face = next(command for command in view.commands if isinstance(command, RectangleCommand))
    assert face == RectangleCommand(64.0, 96.0, 320.0, 192.0, "#18a558", "face")
    irises = [command for command in view.commands if isinstance(command, CircleCommand)]
    assert [(iris.center_x_px, iris.center_y_px, iris.label) for iris in irises] == [
        (160.0, 216.0, "left_iris"),
        (352.0, 216.0, "right_iris"),
    ]
    labels = [command.text for command in view.commands if isinstance(command, TextCommand)]
    assert "Confidence: 0.88" in labels
    assert "FPS: 29.8" in labels
    assert "Latency: 18.2 ms" in labels
    assert "Left openness: 0.70" in labels
    assert "Right openness: 0.60" in labels
    assert "Head pose: yaw=3.0°, pitch=-2.0°, roll=1.0°" in labels


@pytest.mark.parametrize(
    ("state", "reasons", "expected_status"),
    [
        (TrackingState.LOST, (ReasonCode.FACE_NOT_FOUND,), OverlayStatus.LOST),
        (TrackingState.MULTIPLE_FACES, (ReasonCode.MULTIPLE_FACES,), OverlayStatus.MULTIPLE_FACES),
        (TrackingState.LOST, (ReasonCode.ERROR,), OverlayStatus.ERROR),
    ],
)
def test_lost_multiple_and_error_states_never_draw_stale_landmarks(
    state: TrackingState,
    reasons: tuple[ReasonCode, ...],
    expected_status: OverlayStatus,
) -> None:
    view = DebugOverlayRenderer().render(
        _frame(), _observation(state=state, reasons=reasons), fps=20.0, latency_ms=4.0
    )

    assert view.status is expected_status
    assert not _shape_commands(view)
    assert any("Reasons:" in text for text in _texts(view))


def test_frame_mismatch_suppresses_landmarks_and_marks_current_frame() -> None:
    view = DebugOverlayRenderer().render(
        _frame(frame_id=8), _observation(frame_id=7), fps=30.0, latency_ms=8.0
    )

    assert view.frame_id == 8
    assert view.status is OverlayStatus.FRAME_MISMATCH
    assert not _shape_commands(view)
    assert any("Observation frame 7 ignored for current frame 8" in text for text in _texts(view))


def test_extra_policy_reason_codes_are_shown_without_duplicating_domain_reasons() -> None:
    view = DebugOverlayRenderer().render(
        _frame(),
        _observation(state=TrackingState.LOST, reasons=(ReasonCode.FACE_NOT_FOUND,)),
        fps=10.0,
        latency_ms=5.0,
        extra_reason_codes=("HEAD_POSE_MISSING", "FACE_NOT_FOUND"),
    )

    detail = next(text for text in _texts(view) if text.startswith("Reasons:"))
    assert detail == "Reasons: FACE_NOT_FOUND, HEAD_POSE_MISSING"


def test_extra_reason_codes_reject_anything_that_could_carry_a_payload() -> None:
    renderer = DebugOverlayRenderer()

    with pytest.raises(ValueError, match="extra_reason_codes"):
        renderer.render(
            _frame(),
            _observation(),
            fps=None,
            latency_ms=None,
            extra_reason_codes=("x" * 65,),
        )


def test_overlay_is_transient_and_has_no_persistence_or_frame_buffer_field() -> None:
    frame = _frame()
    view = DebugOverlayRenderer().render(frame, _observation(), fps=None, latency_ms=None)
    source = Path("src/gazelink/overlay.py").read_text(encoding="utf-8")

    assert not hasattr(view, "image")
    # Scan every field of every emitted command: a binary payload must never be
    # reachable from a view model, and no rendered text may embed frame bytes.
    values = [
        getattr(command, field.name) for command in view.commands for field in fields(command)
    ]
    assert values, "the tracked overlay must emit commands for this assertion to mean anything"
    assert not any(isinstance(value, (bytes, bytearray, memoryview)) for value in values)
    assert frame.image is not None
    decoded = frame.image.decode("utf-8")
    assert not any(isinstance(value, str) and decoded in value for value in values)
    assert "Path(" not in source
    assert ".write(" not in source
    assert ".save(" not in source


def test_invalid_measurements_are_rejected() -> None:
    renderer = DebugOverlayRenderer()

    with pytest.raises(ValueError, match="fps"):
        renderer.render(_frame(), _observation(), fps=-1.0, latency_ms=0.0)
    with pytest.raises(ValueError, match="latency_ms"):
        renderer.render(_frame(), _observation(), fps=1.0, latency_ms=float("nan"))
