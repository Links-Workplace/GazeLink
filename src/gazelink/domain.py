"""Framework-independent contracts shared by GAZELINK pipeline stages.

The contracts deliberately describe values rather than camera, UI, or operating
system objects.  All timestamps in this module are milliseconds from a
monotonic clock; wall-clock time does not belong in control decisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, TypeAlias

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ContractValidationError(ValueError):
    """Raised when a domain value cannot safely satisfy its contract."""


class PixelFormat(StrEnum):
    """Supported in-memory frame pixel formats."""

    BGR24 = "BGR24"
    RGB24 = "RGB24"
    GRAY8 = "GRAY8"


class TrackingState(StrEnum):
    TRACKED = "TRACKED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOST = "LOST"
    MULTIPLE_FACES = "MULTIPLE_FACES"


class ReasonCode(StrEnum):
    """Machine-readable explanations for degraded or rejected data."""

    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LEFT_EYE_OCCLUDED = "LEFT_EYE_OCCLUDED"
    RIGHT_EYE_OCCLUDED = "RIGHT_EYE_OCCLUDED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STALE_SAMPLE = "STALE_SAMPLE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    CLAMPED_TO_SCREEN = "CLAMPED_TO_SCREEN"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CALIBRATION_STALE = "CALIBRATION_STALE"
    DISPLAY_CHANGED = "DISPLAY_CHANGED"
    CORRECTION_INSUFFICIENT = "CORRECTION_INSUFFICIENT"
    CORRECTION_UNSTABLE = "CORRECTION_UNSTABLE"
    CORRECTION_CONFLICT = "CORRECTION_CONFLICT"
    CORRECTION_INCOMPATIBLE = "CORRECTION_INCOMPATIBLE"
    CORRECTION_NO_IMPROVEMENT = "CORRECTION_NO_IMPROVEMENT"
    PAUSED = "PAUSED"
    CONTROL_DISABLED = "CONTROL_DISABLED"
    SAFETY_BLOCKED = "SAFETY_BLOCKED"
    ERROR = "ERROR"


class InteractionType(StrEnum):
    MOVE = "MOVE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DWELL_SELECT = "DWELL_SELECT"
    SCROLL = "SCROLL"
    DRAG_START = "DRAG_START"
    DRAG_END = "DRAG_END"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


class InteractionSource(StrEnum):
    BLINK = "BLINK"
    WINK = "WINK"
    DWELL = "DWELL"
    UI = "UI"
    SAFETY = "SAFETY"
    GAZE = "GAZE"


class CameraStatus(StrEnum):
    UNAVAILABLE = "UNAVAILABLE"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class CalibrationStatus(StrEnum):
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATING = "CALIBRATING"
    VALID = "VALID"
    INVALID = "INVALID"
    STALE = "STALE"


class ControlState(StrEnum):
    DISABLED = "DISABLED"
    READY = "READY"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"


class SystemState(StrEnum):
    STARTING = "STARTING"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    TRACKING_NOT_READY = "TRACKING_NOT_READY"
    READY_FOR_CALIBRATION = "READY_FOR_CALIBRATION"
    CALIBRATING = "CALIBRATING"
    READY = "READY"
    CONTROL_ACTIVE = "CONTROL_ACTIVE"
    PAUSED = "PAUSED"
    TRACKING_LOST = "TRACKING_LOST"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class ScreenOrientation(StrEnum):
    LANDSCAPE = "LANDSCAPE"
    PORTRAIT = "PORTRAIT"


def _finite_number(value: object, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError(f"{field_name} must be finite")
    if minimum is not None and number < minimum:
        raise ContractValidationError(f"{field_name} must be >= {minimum}")
    return number


def _positive_int(value: object, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name} must be an integer, not {value!r}")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        raise ContractValidationError(f"{field_name} must be >= {lower_bound}")
    return value


def _confidence(value: object, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ContractValidationError(f"{field_name} must be within [0.0, 1.0]")
    return number


def _optional_ratio(value: object, field_name: str) -> float | None:
    """Validate an optional normalized scalar without inventing a value."""

    if value is None:
        return None
    return _confidence(value, field_name)


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _reason_codes(value: tuple[ReasonCode, ...]) -> tuple[ReasonCode, ...]:
    if not isinstance(value, tuple):
        raise ContractValidationError("reason_codes must be an immutable tuple")
    if any(not isinstance(reason, ReasonCode) for reason in value):
        raise ContractValidationError("reason_codes must contain only ReasonCode values")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return value


def _required(data: Mapping[str, Any], key: str) -> Any:
    """Return a required serialized value so constructor validation owns errors."""

    if key not in data:
        return None
    return data[key]


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """A frame-normalized point using top-left origin and [0.0, 1.0] bounds."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _confidence(self.x, "x"))
        object.__setattr__(self, "y", _confidence(self.y, "y"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, value: object) -> NormalizedPoint:
        data = _mapping(value, "normalized point")
        return cls(x=_required(data, "x"), y=_required(data, "y"))


@dataclass(frozen=True, slots=True)
class GazePoint:
    """A normalized gaze prediction; values may be outside [0.0, 1.0]."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "x"))
        object.__setattr__(self, "y", _finite_number(self.y, "y"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, value: object) -> GazePoint:
        data = _mapping(value, "gaze point")
        return cls(x=_required(data, "x"), y=_required(data, "y"))


@dataclass(frozen=True, slots=True)
class PixelPoint:
    """A screen pixel point, with top-left origin and non-negative coordinates."""

    x_px: int
    y_px: int

    def __post_init__(self) -> None:
        _positive_int(self.x_px, "x_px", allow_zero=True)
        _positive_int(self.y_px, "y_px", allow_zero=True)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"x_px": self.x_px, "y_px": self.y_px}

    @classmethod
    def from_dict(cls, value: object) -> PixelPoint:
        data = _mapping(value, "pixel point")
        return cls(x_px=_required(data, "x_px"), y_px=_required(data, "y_px"))


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """A normalized frame box using top-left origin and positive dimensions."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        x = _confidence(self.x, "face_box.x")
        y = _confidence(self.y, "face_box.y")
        width = _finite_number(self.width, "face_box.width", minimum=0.0)
        height = _finite_number(self.height, "face_box.height", minimum=0.0)
        if width == 0.0 or height == 0.0 or x + width > 1.0 or y + height > 1.0:
            raise ContractValidationError(
                "face_box must fit within [0.0, 1.0] with positive dimensions"
            )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, value: object) -> NormalizedBox:
        data = _mapping(value, "normalized box")
        return cls(
            x=_required(data, "x"),
            y=_required(data, "y"),
            width=_required(data, "width"),
            height=_required(data, "height"),
        )


@dataclass(frozen=True, slots=True)
class EyeFeatures:
    """Per-eye measurements; openness and confidence are normalized ratios.

    Two different iris coordinates are carried on purpose, because they answer
    different questions and must never be substituted for one another:

    ``iris_center``
        Frame-normalized, top-left origin. This is where the iris appears in
        the captured image, so it is what a debug overlay draws.
    ``iris_in_eye``
        Eye-relative: x runs from the outer to the inner corner. y is the
        perpendicular offset from the eye-corner axis, divided by eye width
        and shifted so the corner line is 0.5; positive y points downward.
        The stable corner axis is used instead of moving eyelids so vertical
        gaze is not cancelled by lid motion.

    ``openness`` is the eyelid gap divided by the eye width, so it is also
    independent of frame resolution and of distance from the camera.
    """

    iris_center: NormalizedPoint | None
    openness: float
    confidence: float
    iris_in_eye: NormalizedPoint | None = None
    iris_in_lids_y: float | None = None

    def __post_init__(self) -> None:
        for attribute in ("iris_center", "iris_in_eye"):
            value = getattr(self, attribute)
            if value is not None and not isinstance(value, NormalizedPoint):
                raise ContractValidationError(f"{attribute} must be a NormalizedPoint or None")
        object.__setattr__(self, "openness", _confidence(self.openness, "openness"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        object.__setattr__(
            self, "iris_in_lids_y", _optional_ratio(self.iris_in_lids_y, "iris_in_lids_y")
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "iris_center": None if self.iris_center is None else self.iris_center.to_dict(),
            "iris_in_eye": None if self.iris_in_eye is None else self.iris_in_eye.to_dict(),
            "openness": self.openness,
            "confidence": self.confidence,
            "iris_in_lids_y": self.iris_in_lids_y,
        }

    @classmethod
    def from_dict(cls, value: object) -> EyeFeatures:
        data = _mapping(value, "eye features")
        iris_center = data.get("iris_center")
        iris_in_eye = data.get("iris_in_eye")
        return cls(
            iris_center=None if iris_center is None else NormalizedPoint.from_dict(iris_center),
            iris_in_eye=None if iris_in_eye is None else NormalizedPoint.from_dict(iris_in_eye),
            openness=_required(data, "openness"),
            confidence=_required(data, "confidence"),
            iris_in_lids_y=data.get("iris_in_lids_y"),
        )


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Head orientation in degrees: yaw, pitch, then roll."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    def __post_init__(self) -> None:
        for attribute in ("yaw_deg", "pitch_deg", "roll_deg"):
            object.__setattr__(self, attribute, _finite_number(getattr(self, attribute), attribute))

    def to_dict(self) -> dict[str, JSONValue]:
        return {"yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg}

    @classmethod
    def from_dict(cls, value: object) -> HeadPose:
        data = _mapping(value, "head pose")
        return cls(
            yaw_deg=_required(data, "yaw_deg"),
            pitch_deg=_required(data, "pitch_deg"),
            roll_deg=_required(data, "roll_deg"),
        )


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """Single-screen geometry and scaling used by calibration and gaze output."""

    screen_id: str
    width_px: int
    height_px: int
    dpi_scale: float
    orientation: ScreenOrientation = ScreenOrientation.LANDSCAPE

    def __post_init__(self) -> None:
        _nonempty_string(self.screen_id, "screen_id")
        _positive_int(self.width_px, "width_px")
        _positive_int(self.height_px, "height_px")
        object.__setattr__(
            self, "dpi_scale", _finite_number(self.dpi_scale, "dpi_scale", minimum=0.01)
        )
        if not isinstance(self.orientation, ScreenOrientation):
            raise ContractValidationError("orientation must be a ScreenOrientation")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "screen_id": self.screen_id,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "dpi_scale": self.dpi_scale,
            "orientation": self.orientation.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> ScreenGeometry:
        data = _mapping(value, "screen geometry")
        return cls(
            screen_id=_required(data, "screen_id"),
            width_px=_required(data, "width_px"),
            height_px=_required(data, "height_px"),
            dpi_scale=_required(data, "dpi_scale"),
            orientation=ScreenOrientation(_required(data, "orientation")),
        )


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A non-persistent frame envelope; ``image`` is excluded from serialization."""

    frame_id: int
    captured_at_monotonic_ms: float
    width: int
    height: int
    pixel_format: PixelFormat
    image: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_int(self.frame_id, "frame_id", allow_zero=True)
        object.__setattr__(
            self,
            "captured_at_monotonic_ms",
            _finite_number(self.captured_at_monotonic_ms, "captured_at_monotonic_ms", minimum=0.0),
        )
        _positive_int(self.width, "width")
        _positive_int(self.height, "height")
        if not isinstance(self.pixel_format, PixelFormat):
            raise ContractValidationError("pixel_format must be a PixelFormat")
        if self.image is not None and not isinstance(self.image, bytes):
            raise ContractValidationError("image must be immutable bytes or None")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "frame_id": self.frame_id,
            "captured_at_monotonic_ms": self.captured_at_monotonic_ms,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> FramePacket:
        data = _mapping(value, "frame packet")
        return cls(
            frame_id=_required(data, "frame_id"),
            captured_at_monotonic_ms=_required(data, "captured_at_monotonic_ms"),
            width=_required(data, "width"),
            height=_required(data, "height"),
            pixel_format=PixelFormat(_required(data, "pixel_format")),
        )


@dataclass(frozen=True, slots=True)
class VisionObservation:
    """Output of vision processing, without model-specific landmark objects."""

    frame_id: int
    observed_at_monotonic_ms: float
    tracking_state: TrackingState
    face_box: NormalizedBox | None
    left_eye: EyeFeatures | None
    right_eye: EyeFeatures | None
    head_pose: HeadPose | None
    overall_confidence: float
    reason_codes: tuple[ReasonCode, ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.frame_id, "frame_id", allow_zero=True)
        object.__setattr__(
            self,
            "observed_at_monotonic_ms",
            _finite_number(self.observed_at_monotonic_ms, "observed_at_monotonic_ms", minimum=0.0),
        )
        if not isinstance(self.tracking_state, TrackingState):
            raise ContractValidationError("tracking_state must be a TrackingState")
        for attribute, expected in (
            ("face_box", NormalizedBox),
            ("left_eye", EyeFeatures),
            ("right_eye", EyeFeatures),
            ("head_pose", HeadPose),
        ):
            value = getattr(self, attribute)
            if value is not None and not isinstance(value, expected):
                raise ContractValidationError(f"{attribute} must be a {expected.__name__} or None")
        object.__setattr__(
            self, "overall_confidence", _confidence(self.overall_confidence, "overall_confidence")
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "frame_id": self.frame_id,
            "observed_at_monotonic_ms": self.observed_at_monotonic_ms,
            "tracking_state": self.tracking_state.value,
            "face_box": None if self.face_box is None else self.face_box.to_dict(),
            "left_eye": None if self.left_eye is None else self.left_eye.to_dict(),
            "right_eye": None if self.right_eye is None else self.right_eye.to_dict(),
            "head_pose": None if self.head_pose is None else self.head_pose.to_dict(),
            "overall_confidence": self.overall_confidence,
            "reason_codes": [reason.value for reason in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: object) -> VisionObservation:
        data = _mapping(value, "vision observation")
        return cls(
            frame_id=_required(data, "frame_id"),
            observed_at_monotonic_ms=_required(data, "observed_at_monotonic_ms"),
            tracking_state=TrackingState(_required(data, "tracking_state")),
            face_box=None
            if data.get("face_box") is None
            else NormalizedBox.from_dict(data["face_box"]),
            left_eye=None
            if data.get("left_eye") is None
            else EyeFeatures.from_dict(data["left_eye"]),
            right_eye=None
            if data.get("right_eye") is None
            else EyeFeatures.from_dict(data["right_eye"]),
            head_pose=None
            if data.get("head_pose") is None
            else HeadPose.from_dict(data["head_pose"]),
            overall_confidence=_required(data, "overall_confidence"),
            reason_codes=tuple(ReasonCode(item) for item in data.get("reason_codes", [])),
        )


@dataclass(frozen=True, slots=True)
class GazeSample:
    """Base, locally corrected, and filtered gaze coordinates.

    ``raw_normalized`` is always the immutable base-model prediction.
    ``corrected_normalized`` is the optional local-correction output before
    temporal filtering. ``filtered_normalized`` is reserved for M2-04's
    smoothing output. Keeping all three explicit prevents diagnostics from
    silently attributing correction improvements to smoothing (or vice versa).
    """

    source_frame_id: int
    sampled_at_monotonic_ms: float
    raw_normalized: GazePoint
    corrected_normalized: GazePoint
    filtered_normalized: GazePoint
    screen_position: PixelPoint
    screen_id: str
    confidence: float
    valid_for_control: bool
    reason_codes: tuple[ReasonCode, ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.source_frame_id, "source_frame_id", allow_zero=True)
        object.__setattr__(
            self,
            "sampled_at_monotonic_ms",
            _finite_number(self.sampled_at_monotonic_ms, "sampled_at_monotonic_ms", minimum=0.0),
        )
        if not isinstance(self.raw_normalized, GazePoint):
            raise ContractValidationError("raw_normalized must be a GazePoint")
        if not isinstance(self.corrected_normalized, GazePoint):
            raise ContractValidationError("corrected_normalized must be a GazePoint")
        if not isinstance(self.filtered_normalized, GazePoint):
            raise ContractValidationError("filtered_normalized must be a GazePoint")
        if not isinstance(self.screen_position, PixelPoint):
            raise ContractValidationError("screen_position must be a PixelPoint")
        _nonempty_string(self.screen_id, "screen_id")
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        if not isinstance(self.valid_for_control, bool):
            raise ContractValidationError("valid_for_control must be a bool")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "source_frame_id": self.source_frame_id,
            "sampled_at_monotonic_ms": self.sampled_at_monotonic_ms,
            "raw_normalized": self.raw_normalized.to_dict(),
            "corrected_normalized": self.corrected_normalized.to_dict(),
            "filtered_normalized": self.filtered_normalized.to_dict(),
            "screen_position": self.screen_position.to_dict(),
            "screen_id": self.screen_id,
            "confidence": self.confidence,
            "valid_for_control": self.valid_for_control,
            "reason_codes": [reason.value for reason in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: object) -> GazeSample:
        data = _mapping(value, "gaze sample")
        raw = GazePoint.from_dict(_required(data, "raw_normalized"))
        corrected_value = data.get("corrected_normalized")
        return cls(
            source_frame_id=_required(data, "source_frame_id"),
            sampled_at_monotonic_ms=_required(data, "sampled_at_monotonic_ms"),
            raw_normalized=raw,
            corrected_normalized=(
                raw if corrected_value is None else GazePoint.from_dict(corrected_value)
            ),
            filtered_normalized=GazePoint.from_dict(_required(data, "filtered_normalized")),
            screen_position=PixelPoint.from_dict(_required(data, "screen_position")),
            screen_id=_required(data, "screen_id"),
            confidence=_required(data, "confidence"),
            valid_for_control=_required(data, "valid_for_control"),
            reason_codes=tuple(ReasonCode(item) for item in data.get("reason_codes", [])),
        )


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """A single event emitted by the interaction layer."""

    event_id: int
    event_type: InteractionType
    occurred_at_monotonic_ms: float
    source: InteractionSource
    position: PixelPoint | None
    confidence: float
    requires_active_control: bool

    def __post_init__(self) -> None:
        _positive_int(self.event_id, "event_id", allow_zero=True)
        object.__setattr__(
            self,
            "occurred_at_monotonic_ms",
            _finite_number(self.occurred_at_monotonic_ms, "occurred_at_monotonic_ms", minimum=0.0),
        )
        if not isinstance(self.event_type, InteractionType):
            raise ContractValidationError("event_type must be an InteractionType")
        if not isinstance(self.source, InteractionSource):
            raise ContractValidationError("source must be an InteractionSource")
        if self.position is not None and not isinstance(self.position, PixelPoint):
            raise ContractValidationError("position must be a PixelPoint or None")
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        if not isinstance(self.requires_active_control, bool):
            raise ContractValidationError("requires_active_control must be a bool")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at_monotonic_ms": self.occurred_at_monotonic_ms,
            "source": self.source.value,
            "position": None if self.position is None else self.position.to_dict(),
            "confidence": self.confidence,
            "requires_active_control": self.requires_active_control,
        }

    @classmethod
    def from_dict(cls, value: object) -> InteractionEvent:
        data = _mapping(value, "interaction event")
        position = data.get("position")
        return cls(
            event_id=_required(data, "event_id"),
            event_type=InteractionType(_required(data, "event_type")),
            occurred_at_monotonic_ms=_required(data, "occurred_at_monotonic_ms"),
            source=InteractionSource(_required(data, "source")),
            position=None if position is None else PixelPoint.from_dict(position),
            confidence=_required(data, "confidence"),
            requires_active_control=_required(data, "requires_active_control"),
        )


@dataclass(frozen=True, slots=True)
class SystemStatus:
    """Serializable snapshot of pipeline health; no biometric detail is included."""

    CONTRACT_VERSION: ClassVar[int] = 1

    system_state: SystemState
    camera_status: CameraStatus
    tracking_state: TrackingState
    calibration_status: CalibrationStatus
    control_state: ControlState
    paused_reason: ReasonCode | None
    active_profile_id: str | None
    fps: float | None
    pipeline_latency_ms: float | None
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for attribute, expected in (
            ("system_state", SystemState),
            ("camera_status", CameraStatus),
            ("tracking_state", TrackingState),
            ("calibration_status", CalibrationStatus),
            ("control_state", ControlState),
        ):
            if not isinstance(getattr(self, attribute), expected):
                raise ContractValidationError(f"{attribute} must be a {expected.__name__}")
        if self.paused_reason is not None and not isinstance(self.paused_reason, ReasonCode):
            raise ContractValidationError("paused_reason must be a ReasonCode or None")
        if self.active_profile_id is not None:
            _nonempty_string(self.active_profile_id, "active_profile_id")
        for attribute in ("fps", "pipeline_latency_ms"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(self, attribute, _finite_number(value, attribute, minimum=0.0))
        _positive_int(self.contract_version, "contract_version")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "system_state": self.system_state.value,
            "camera_status": self.camera_status.value,
            "tracking_state": self.tracking_state.value,
            "calibration_status": self.calibration_status.value,
            "control_state": self.control_state.value,
            "paused_reason": None if self.paused_reason is None else self.paused_reason.value,
            "active_profile_id": self.active_profile_id,
            "fps": self.fps,
            "pipeline_latency_ms": self.pipeline_latency_ms,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> SystemStatus:
        data = _mapping(value, "system status")
        paused_reason = data.get("paused_reason")
        return cls(
            system_state=SystemState(_required(data, "system_state")),
            camera_status=CameraStatus(_required(data, "camera_status")),
            tracking_state=TrackingState(_required(data, "tracking_state")),
            calibration_status=CalibrationStatus(_required(data, "calibration_status")),
            control_state=ControlState(_required(data, "control_state")),
            paused_reason=None if paused_reason is None else ReasonCode(paused_reason),
            active_profile_id=data.get("active_profile_id"),
            fps=data.get("fps"),
            pipeline_latency_ms=data.get("pipeline_latency_ms"),
            contract_version=data.get("contract_version", cls.CONTRACT_VERSION),
        )
