"""Pure, versioned feature extraction for gaze mapping.

The vector order and units are a persisted contract:

1. left iris x, eye-relative and increasing toward screen-right
2. left iris y, perpendicular to the eye-corner axis, positive downward
3. right iris x, sign-corrected to increase toward screen-right
4. right iris y, perpendicular to the eye-corner axis, positive downward
5. left iris position within the eyelid opening, positive downward
6. right iris position within the eyelid opening, positive downward
7. head yaw, degrees
8. head pitch, degrees
9. head roll, degrees

``EyeFeatures.iris_in_eye.x`` runs outer-to-inner for each anatomical eye.
Those anatomical axes point in opposite screen-horizontal directions, so the
right-eye value is transformed as ``1 - x``. Confidence is deliberately absent:
the current live adapter pins it to a constant and it carries no model signal.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from gazelink.calibration import FEATURE_SCHEMA_VERSION, CalibrationSample
from gazelink.domain import ContractValidationError, JSONValue, VisionObservation

FEATURE_NAMES = (
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "left_iris_lid_y",
    "right_iris_lid_y",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
)


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class GazeFeatureVector:
    """Immutable nine-element vector stamped with the persisted schema."""

    values: tuple[float, ...]
    schema_version: int = FEATURE_SCHEMA_VERSION
    size: ClassVar[int] = len(FEATURE_NAMES)

    def __post_init__(self) -> None:
        if len(self.values) != self.size:
            raise ContractValidationError(f"feature vector must contain {self.size} values")
        object.__setattr__(
            self,
            "values",
            tuple(_finite(value, f"features[{index}]") for index, value in enumerate(self.values)),
        )
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ContractValidationError("schema_version must be an integer")
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported feature schema {self.schema_version}; "
                f"expected {FEATURE_SCHEMA_VERSION}"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {"schema_version": self.schema_version, "values": list(self.values)}

    @classmethod
    def from_dict(cls, value: object) -> GazeFeatureVector:
        if not isinstance(value, Mapping):
            raise ContractValidationError("feature vector must be an object")
        raw_values = value.get("values")
        if not isinstance(raw_values, list):
            raise ContractValidationError("feature vector values must be a list")
        return cls(
            values=tuple(raw_values),
            schema_version=value.get("schema_version"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class FeatureWindowStability:
    """Worst-eye gaze spread and head-pose spread within one capture window."""

    eye_p95: float
    head_pose_p95_deg: float


def measure_feature_window_stability(
    features: Sequence[GazeFeatureVector],
) -> FeatureWindowStability:
    if not features:
        raise ContractValidationError("feature stability requires at least one vector")
    medians = tuple(
        statistics.median(vector.values[index] for vector in features)
        for index in range(GazeFeatureVector.size)
    )
    eye_radii = [
        max(
            math.hypot(vector.values[0] - medians[0], vector.values[1] - medians[1]),
            math.hypot(vector.values[2] - medians[2], vector.values[3] - medians[3]),
        )
        for vector in features
    ]
    head_radii = [
        math.sqrt(
            (vector.values[6] - medians[6]) ** 2
            + (vector.values[7] - medians[7]) ** 2
            + (vector.values[8] - medians[8]) ** 2
        )
        for vector in features
    ]
    return FeatureWindowStability(
        eye_p95=_percentile(eye_radii, 95.0),
        head_pose_p95_deg=_percentile(head_radii, 95.0),
    )


def select_stable_feature_window(
    features: Sequence[GazeFeatureVector],
    *,
    window_size: int,
    max_eye_p95: float,
    max_head_pose_p95_deg: float,
) -> tuple[int, int, FeatureWindowStability] | None:
    """Select the best passing contiguous window; never splice distant frames."""

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ContractValidationError("window_size must be a positive integer")
    max_eye_p95 = _finite(max_eye_p95, "max_eye_p95")
    max_head_pose_p95_deg = _finite(max_head_pose_p95_deg, "max_head_pose_p95_deg")
    if max_eye_p95 <= 0.0:
        raise ContractValidationError("max_eye_p95 must be > 0")
    if max_head_pose_p95_deg <= 0.0:
        raise ContractValidationError("max_head_pose_p95_deg must be > 0")
    if len(features) < window_size:
        return None
    best: tuple[int, int, FeatureWindowStability] | None = None
    best_score = math.inf
    for start in range(len(features) - window_size + 1):
        end = start + window_size
        stability = measure_feature_window_stability(features[start:end])
        if stability.eye_p95 > max_eye_p95 or stability.head_pose_p95_deg > max_head_pose_p95_deg:
            continue
        score = (stability.eye_p95 / max_eye_p95) + (
            stability.head_pose_p95_deg / max_head_pose_p95_deg
        )
        if score < best_score:
            best = (start, end, stability)
            best_score = score
    return best


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def from_calibration_sample(sample: CalibrationSample) -> GazeFeatureVector | None:
    """Return usable features from an accepted sample, else ``None``."""

    if not isinstance(sample, CalibrationSample):
        raise ContractValidationError("sample must be a CalibrationSample")
    if not sample.accepted:
        return None
    left = sample.left_iris_in_eye
    right = sample.right_iris_in_eye
    required = (
        left,
        right,
        sample.head_yaw_deg,
        sample.head_pitch_deg,
        sample.head_roll_deg,
    )
    if any(value is None for value in required):
        return None
    assert left is not None and right is not None
    left_lid_y = left.y if sample.left_iris_in_lids_y is None else sample.left_iris_in_lids_y
    right_lid_y = right.y if sample.right_iris_in_lids_y is None else sample.right_iris_in_lids_y
    assert sample.head_yaw_deg is not None
    assert sample.head_pitch_deg is not None
    assert sample.head_roll_deg is not None
    return _build(
        left.x,
        left.y,
        right.x,
        right.y,
        left_lid_y,
        right_lid_y,
        sample.head_yaw_deg,
        sample.head_pitch_deg,
        sample.head_roll_deg,
    )


def from_observation(observation: VisionObservation) -> GazeFeatureVector | None:
    """Extract features only when both eyes, both irises, and head pose exist."""

    if not isinstance(observation, VisionObservation):
        raise ContractValidationError("observation must be a VisionObservation")
    left = observation.left_eye
    right = observation.right_eye
    pose = observation.head_pose
    if (
        left is None
        or right is None
        or pose is None
        or left.iris_in_eye is None
        or right.iris_in_eye is None
    ):
        return None
    return _build(
        left.iris_in_eye.x,
        left.iris_in_eye.y,
        right.iris_in_eye.x,
        right.iris_in_eye.y,
        left.iris_in_eye.y if left.iris_in_lids_y is None else left.iris_in_lids_y,
        right.iris_in_eye.y if right.iris_in_lids_y is None else right.iris_in_lids_y,
        pose.yaw_deg,
        pose.pitch_deg,
        pose.roll_deg,
    )


def _build(
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
    left_lid_y: float,
    right_lid_y: float,
    yaw: float,
    pitch: float,
    roll: float,
) -> GazeFeatureVector:
    return GazeFeatureVector(
        (
            left_x,
            left_y,
            1.0 - right_x,
            right_y,
            left_lid_y,
            right_lid_y,
            yaw,
            pitch,
            roll,
        )
    )
