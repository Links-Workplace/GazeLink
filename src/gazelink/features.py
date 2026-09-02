"""Pure, model-agnostic geometry used to derive eye features.

All input landmarks are normalized frame coordinates with a top-left origin and
values in ``[0.0, 1.0]``. ``EyeLandmarks`` names corners from the user's point
of view; their ordering does not affect the calculated ratio.

Two iris coordinates are produced, and they are not interchangeable.
``EyeFeatures.iris_center`` stays in frame coordinates, so a debug overlay can
draw it in the right place. ``EyeFeatures.iris_in_eye`` is normalized against
the stable eye-corner axis: x runs outer-to-inner corner; y is the perpendicular
iris offset from that axis, scaled by eye width and shifted so the corner line
is 0.5. Eyelids are deliberately not the y reference because they move with
vertical gaze and can cancel the iris motion the model needs. Mixing the two
silently places the iris at the centre of the frame, so both are named
explicitly rather than inferred.

``HeadPose`` is passed through unchanged and uses degrees (yaw, pitch, roll);
adapters that derive it from a transform matrix own that model-specific
conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    HeadPose,
    NormalizedPoint,
    ReasonCode,
)

_GEOMETRY_EPSILON = 1e-9


def _confidence(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractValidationError(f"{field_name} must be within [0.0, 1.0]")
    return result


def _distance(first: NormalizedPoint, second: NormalizedPoint) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


@dataclass(frozen=True, slots=True)
class EyeLandmarks:
    """Minimal normalized landmarks required for model-independent eye geometry."""

    outer_corner: NormalizedPoint
    inner_corner: NormalizedPoint
    upper_lid: NormalizedPoint
    lower_lid: NormalizedPoint
    iris_frame_points: tuple[NormalizedPoint, ...]
    detector_confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.iris_frame_points, tuple) or any(
            not isinstance(point, NormalizedPoint) for point in self.iris_frame_points
        ):
            raise ContractValidationError("iris_frame_points must be an immutable tuple of points")
        object.__setattr__(
            self,
            "detector_confidence",
            _confidence(self.detector_confidence, "detector_confidence"),
        )


@dataclass(frozen=True, slots=True)
class EyeExtractionResult:
    """Geometry result where validity is explicit and never inferred from a value."""

    eye: EyeFeatures | None
    valid: bool
    reason_codes: tuple[ReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ContractValidationError("valid must be a bool")
        if self.valid != (self.eye is not None):
            raise ContractValidationError(
                "valid eye extraction must have features; invalid extraction must not"
            )
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(reason, ReasonCode) for reason in self.reason_codes
        ):
            raise ContractValidationError(
                "reason_codes must be an immutable tuple of ReasonCode values"
            )


@dataclass(frozen=True, slots=True)
class FeatureExtractionResult:
    """Both eye results plus the model-independent head-pose value supplied by an adapter."""

    left_eye: EyeExtractionResult
    right_eye: EyeExtractionResult
    head_pose: HeadPose | None
    reason_codes: tuple[ReasonCode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.left_eye, EyeExtractionResult):
            raise ContractValidationError("left_eye must be an EyeExtractionResult")
        if not isinstance(self.right_eye, EyeExtractionResult):
            raise ContractValidationError("right_eye must be an EyeExtractionResult")
        if self.head_pose is not None and not isinstance(self.head_pose, HeadPose):
            raise ContractValidationError("head_pose must be a HeadPose or None")
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(reason, ReasonCode) for reason in self.reason_codes
        ):
            raise ContractValidationError(
                "reason_codes must be an immutable tuple of ReasonCode values"
            )


def _invalid(reason: ReasonCode) -> EyeExtractionResult:
    return EyeExtractionResult(eye=None, valid=False, reason_codes=(reason,))


def extract_eye_features(
    landmarks: EyeLandmarks | None,
    *,
    occlusion_reason: ReasonCode,
) -> EyeExtractionResult:
    """Calculate scale-invariant openness plus both iris coordinates.

    The result is invalid rather than clamped when landmarks are missing,
    degenerate, or place the iris outside the eyelid/corner bounds.  That keeps
    invalid geometry from becoming an apparently useful gaze signal.
    """

    if landmarks is None or not landmarks.iris_frame_points:
        return _invalid(occlusion_reason)

    corner_width = _distance(landmarks.outer_corner, landmarks.inner_corner)
    lid_gap = _distance(landmarks.upper_lid, landmarks.lower_lid)
    if corner_width <= _GEOMETRY_EPSILON or lid_gap <= _GEOMETRY_EPSILON:
        return _invalid(ReasonCode.LOW_CONFIDENCE)
    openness = lid_gap / corner_width
    if openness > 1.0:
        return _invalid(ReasonCode.LOW_CONFIDENCE)

    iris_x = sum(point.x for point in landmarks.iris_frame_points) / len(
        landmarks.iris_frame_points
    )
    iris_y = sum(point.y for point in landmarks.iris_frame_points) / len(
        landmarks.iris_frame_points
    )
    horizontal_x = (landmarks.inner_corner.x - landmarks.outer_corner.x) / corner_width
    horizontal_y = (landmarks.inner_corner.y - landmarks.outer_corner.y) / corner_width
    normal_x = -horizontal_y
    normal_y = horizontal_x

    def project_horizontal(point: NormalizedPoint) -> float:
        return ((point.x - landmarks.outer_corner.x) * horizontal_x) + (
            (point.y - landmarks.outer_corner.y) * horizontal_y
        )

    def project_normal(point: NormalizedPoint) -> float:
        return ((point.x - landmarks.outer_corner.x) * normal_x) + (
            (point.y - landmarks.outer_corner.y) * normal_y
        )

    iris_relative_x = (
        ((iris_x - landmarks.outer_corner.x) * horizontal_x)
        + ((iris_y - landmarks.outer_corner.y) * horizontal_y)
    ) / corner_width
    upper_normal = project_normal(landmarks.upper_lid)
    lower_normal = project_normal(landmarks.lower_lid)
    iris_normal = ((iris_x - landmarks.outer_corner.x) * normal_x) + (
        (iris_y - landmarks.outer_corner.y) * normal_y
    )
    normal_gap = lower_normal - upper_normal
    if abs(normal_gap) <= _GEOMETRY_EPSILON:
        return _invalid(ReasonCode.LOW_CONFIDENCE)
    # The normal flips between anatomical eyes because their outer->inner
    # axes point in opposite image directions. Orient it using the observed
    # upper/lower lids so positive y always means downward, then normalize by
    # stable corner width rather than moving lid gap. This preserves vertical
    # iris motion even when the eyelids follow the iris.
    downward_sign = 1.0 if normal_gap > 0.0 else -1.0
    iris_relative_y = 0.5 + (iris_normal * downward_sign / corner_width)
    iris_in_lids_y = (iris_normal - upper_normal) * downward_sign / abs(normal_gap)

    # The horizontal projections confirm that the named corners form a segment.
    if abs(project_horizontal(landmarks.inner_corner) - corner_width) > _GEOMETRY_EPSILON:
        return _invalid(ReasonCode.LOW_CONFIDENCE)
    if not (
        0.0 <= iris_relative_x <= 1.0
        and 0.0 <= iris_relative_y <= 1.0
        and 0.0 <= iris_in_lids_y <= 1.0
    ):
        return _invalid(ReasonCode.OUT_OF_RANGE)

    return EyeExtractionResult(
        eye=EyeFeatures(
            iris_center=NormalizedPoint(iris_x, iris_y),
            iris_in_eye=NormalizedPoint(iris_relative_x, iris_relative_y),
            openness=openness,
            confidence=landmarks.detector_confidence,
            iris_in_lids_y=iris_in_lids_y,
        ),
        valid=True,
    )


def extract_features(
    left_landmarks: EyeLandmarks | None,
    right_landmarks: EyeLandmarks | None,
    *,
    head_pose: HeadPose | None,
) -> FeatureExtractionResult:
    """Extract both eyes without binding the domain layer to a landmark provider."""

    left_eye = extract_eye_features(
        left_landmarks,
        occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED,
    )
    right_eye = extract_eye_features(
        right_landmarks,
        occlusion_reason=ReasonCode.RIGHT_EYE_OCCLUDED,
    )
    reason_codes = left_eye.reason_codes + right_eye.reason_codes
    return FeatureExtractionResult(
        left_eye=left_eye,
        right_eye=right_eye,
        head_pose=head_pose,
        reason_codes=reason_codes,
    )
