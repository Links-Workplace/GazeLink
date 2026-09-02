"""Provider-neutral debug-overlay view models for the M1 vision pipeline.

This module describes transient drawing commands only. It intentionally does
not create a window, retain a ``FramePacket.image``, or save frames/screenshots.
The future desktop UI may translate the commands to Qt, while automated tests
can inspect them without a display server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from gazelink.domain import FramePacket, ReasonCode, TrackingState, VisionObservation


class OverlayStatus(StrEnum):
    """The current debug state, chosen without reusing an older observation."""

    TRACKED = "TRACKED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOST = "LOST"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    ERROR = "ERROR"
    FRAME_MISMATCH = "FRAME_MISMATCH"


@dataclass(frozen=True, slots=True)
class RectangleCommand:
    """An outline rectangle in pixels relative to the current frame."""

    x_px: float
    y_px: float
    width_px: float
    height_px: float
    color: str
    label: str


@dataclass(frozen=True, slots=True)
class CircleCommand:
    """An outline circle in pixels relative to the current frame."""

    center_x_px: float
    center_y_px: float
    radius_px: float
    color: str
    label: str


@dataclass(frozen=True, slots=True)
class TextCommand:
    """A diagnostic label; it never contains frame bytes or raw landmarks."""

    text: str
    x_px: float
    y_px: float
    color: str


OverlayCommand: TypeAlias = RectangleCommand | CircleCommand | TextCommand


@dataclass(frozen=True, slots=True)
class OverlayViewModel:
    """A one-frame debug view model with no image buffer or persistence hook."""

    frame_id: int
    frame_width_px: int
    frame_height_px: int
    status: OverlayStatus
    commands: tuple[OverlayCommand, ...]


class DebugOverlayRenderer:
    """Build a fresh overlay for exactly one frame/observation pair."""

    def render(
        self,
        frame: FramePacket,
        observation: VisionObservation,
        *,
        fps: float | None,
        latency_ms: float | None,
        extra_reason_codes: tuple[str, ...] = (),
    ) -> OverlayViewModel:
        """Return commands for ``frame``; mismatches and unsafe states draw no landmarks.

        ``extra_reason_codes`` carries policy-level reasons that are not part of
        the public :class:`~gazelink.domain.ReasonCode` enum, so a runtime can
        explain a rejection without widening the domain contract.
        """

        _validate_measurement(fps, "fps")
        _validate_measurement(latency_ms, "latency_ms")
        _validate_reason_codes(extra_reason_codes)
        if frame.frame_id != observation.frame_id:
            return OverlayViewModel(
                frame_id=frame.frame_id,
                frame_width_px=frame.width,
                frame_height_px=frame.height,
                status=OverlayStatus.FRAME_MISMATCH,
                commands=_diagnostic_commands(
                    frame,
                    status=OverlayStatus.FRAME_MISMATCH,
                    confidence=None,
                    fps=fps,
                    latency_ms=latency_ms,
                    detail=(
                        f"Observation frame {observation.frame_id} ignored for current "
                        f"frame {frame.frame_id}"
                    ),
                ),
            )

        status = _status_for(observation)
        commands: list[OverlayCommand] = list(
            _diagnostic_commands(
                frame,
                status=status,
                confidence=observation.overall_confidence,
                fps=fps,
                latency_ms=latency_ms,
                detail=_reason_detail(observation.reason_codes, extra_reason_codes),
            )
        )
        if status in {OverlayStatus.TRACKED, OverlayStatus.LOW_CONFIDENCE}:
            commands.extend(_landmark_commands(frame, observation, status=status))
        return OverlayViewModel(
            frame_id=frame.frame_id,
            frame_width_px=frame.width,
            frame_height_px=frame.height,
            status=status,
            commands=tuple(commands),
        )


def _validate_measurement(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_reason_codes(codes: tuple[str, ...]) -> None:
    """Reject anything but short diagnostic identifiers, so no payload can leak here."""

    if not isinstance(codes, tuple):
        raise ValueError("extra_reason_codes must be an immutable tuple")
    for code in codes:
        if not isinstance(code, str) or not code.strip() or len(code) > 64:
            raise ValueError("extra_reason_codes must be short non-empty identifiers")


def _status_for(observation: VisionObservation) -> OverlayStatus:
    if ReasonCode.ERROR in observation.reason_codes:
        return OverlayStatus.ERROR
    match observation.tracking_state:
        case TrackingState.TRACKED:
            return OverlayStatus.TRACKED
        case TrackingState.LOW_CONFIDENCE:
            return OverlayStatus.LOW_CONFIDENCE
        case TrackingState.LOST:
            return OverlayStatus.LOST
        case TrackingState.MULTIPLE_FACES:
            return OverlayStatus.MULTIPLE_FACES


def _diagnostic_commands(
    frame: FramePacket,
    *,
    status: OverlayStatus,
    confidence: float | None,
    fps: float | None,
    latency_ms: float | None,
    detail: str | None,
) -> tuple[TextCommand, ...]:
    color = _status_color(status)
    labels = [
        f"State: {status.value}",
        "Confidence: unavailable" if confidence is None else f"Confidence: {confidence:.2f}",
        "FPS: unavailable" if fps is None else f"FPS: {fps:.1f}",
        "Latency: unavailable" if latency_ms is None else f"Latency: {latency_ms:.1f} ms",
    ]
    if detail:
        labels.append(detail)
    return tuple(
        TextCommand(text=label, x_px=8.0, y_px=float(24 + index * 22), color=color)
        for index, label in enumerate(labels)
    )


def _landmark_commands(
    frame: FramePacket,
    observation: VisionObservation,
    *,
    status: OverlayStatus,
) -> tuple[OverlayCommand, ...]:
    color = _status_color(status)
    commands: list[OverlayCommand] = []
    if observation.face_box is not None:
        face = observation.face_box
        commands.append(
            RectangleCommand(
                x_px=face.x * frame.width,
                y_px=face.y * frame.height,
                width_px=face.width * frame.width,
                height_px=face.height * frame.height,
                color=color,
                label="face",
            )
        )
    iris_radius = max(3.0, min(frame.width, frame.height) * 0.015)
    for side, eye in (("left", observation.left_eye), ("right", observation.right_eye)):
        if eye is None:
            commands.append(
                TextCommand(
                    text=f"{side.title()} eye: unavailable",
                    x_px=8.0,
                    y_px=float(140 + len(commands) * 22),
                    color=color,
                )
            )
            continue
        commands.append(
            TextCommand(
                text=f"{side.title()} openness: {eye.openness:.2f}",
                x_px=8.0,
                y_px=float(140 + len(commands) * 22),
                color=color,
            )
        )
        if eye.iris_center is not None:
            commands.append(
                CircleCommand(
                    center_x_px=eye.iris_center.x * frame.width,
                    center_y_px=eye.iris_center.y * frame.height,
                    radius_px=iris_radius,
                    color=color,
                    label=f"{side}_iris",
                )
            )
    pose = observation.head_pose
    pose_text = (
        "Head pose: unavailable"
        if pose is None
        else (
            f"Head pose: yaw={pose.yaw_deg:.1f}°, "
            f"pitch={pose.pitch_deg:.1f}°, roll={pose.roll_deg:.1f}°"
        )
    )
    commands.append(TextCommand(text=pose_text, x_px=8.0, y_px=118.0, color=color))
    return tuple(commands)


def _reason_detail(
    reason_codes: tuple[ReasonCode, ...],
    extra_reason_codes: tuple[str, ...] = (),
) -> str | None:
    names = [reason.value for reason in reason_codes]
    names.extend(code for code in extra_reason_codes if code not in names)
    if not names:
        return None
    return "Reasons: " + ", ".join(names)


def _status_color(status: OverlayStatus) -> str:
    if status is OverlayStatus.TRACKED:
        return "#18a558"
    if status is OverlayStatus.LOW_CONFIDENCE:
        return "#e6a700"
    return "#d83a3a"
