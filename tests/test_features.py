"""Deterministic tests for model-independent eye geometry."""

from __future__ import annotations

import pytest

from gazelink.domain import HeadPose, NormalizedPoint, ReasonCode
from gazelink.features import EyeLandmarks, extract_eye_features, extract_features


def _eye(*, offset_x: float = 0.0, scale: float = 1.0) -> EyeLandmarks:
    center_y = 0.20
    return EyeLandmarks(
        outer_corner=NormalizedPoint(offset_x + (0.10 * scale), center_y),
        inner_corner=NormalizedPoint(offset_x + (0.30 * scale), center_y),
        upper_lid=NormalizedPoint(offset_x + (0.20 * scale), center_y - (0.03 * scale)),
        lower_lid=NormalizedPoint(offset_x + (0.20 * scale), center_y + (0.03 * scale)),
        iris_frame_points=(
            NormalizedPoint(offset_x + (0.19 * scale), center_y),
            NormalizedPoint(offset_x + (0.21 * scale), center_y),
        ),
        detector_confidence=0.9,
    )


def test_eye_openness_and_iris_position_are_scale_invariant() -> None:
    regular = extract_eye_features(_eye(), occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED)
    scaled = extract_eye_features(
        _eye(offset_x=0.30, scale=0.5), occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED
    )

    assert regular.valid and scaled.valid
    assert regular.eye is not None and scaled.eye is not None
    assert regular.eye.openness == pytest.approx(scaled.eye.openness)
    # The eye-relative coordinate is what must survive a resolution change.
    assert regular.eye.iris_in_eye is not None and scaled.eye.iris_in_eye is not None
    assert regular.eye.iris_in_eye.x == pytest.approx(0.5)
    assert regular.eye.iris_in_eye.y == pytest.approx(0.5)
    assert scaled.eye.iris_in_eye.x == pytest.approx(0.5)
    assert scaled.eye.iris_in_eye.y == pytest.approx(0.5)
    assert regular.eye.iris_in_lids_y == pytest.approx(0.5)
    assert scaled.eye.iris_in_lids_y == pytest.approx(0.5)
    # The frame coordinate must NOT be scale invariant: it tracks where the eye
    # actually sits in the image, which is what a debug overlay draws.
    assert regular.eye.iris_center is not None and scaled.eye.iris_center is not None
    assert regular.eye.iris_center.x == pytest.approx(0.20)
    assert scaled.eye.iris_center.x == pytest.approx(0.40)
    assert regular.eye.iris_center.x != pytest.approx(scaled.eye.iris_center.x)


@pytest.mark.parametrize(("iris_y", "expected_lid_y"), [(0.16, 3.0 / 14.0), (0.24, 11.0 / 14.0)])
def test_iris_position_relative_to_lids_is_preserved(iris_y: float, expected_lid_y: float) -> None:
    landmarks = EyeLandmarks(
        outer_corner=NormalizedPoint(0.10, 0.20),
        inner_corner=NormalizedPoint(0.30, 0.20),
        upper_lid=NormalizedPoint(0.20, 0.13),
        lower_lid=NormalizedPoint(0.20, 0.27),
        iris_frame_points=(NormalizedPoint(0.20, iris_y),),
        detector_confidence=0.9,
    )
    result = extract_eye_features(landmarks, occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED)
    assert result.valid and result.eye is not None
    assert result.eye.iris_in_lids_y == pytest.approx(expected_lid_y)


def test_frame_and_eye_relative_iris_coordinates_are_not_interchangeable() -> None:
    """An off-centre eye must not report the same value in both conventions."""

    result = extract_eye_features(
        _eye(offset_x=0.55), occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED
    )

    assert result.eye is not None
    assert result.eye.iris_center is not None and result.eye.iris_in_eye is not None
    assert result.eye.iris_center.x == pytest.approx(0.75)
    assert result.eye.iris_in_eye.x == pytest.approx(0.5)


@pytest.mark.parametrize(("iris_y", "expected_y"), [(0.16, 0.30), (0.24, 0.70)])
def test_vertical_iris_motion_survives_eyelids_following_the_iris(
    iris_y: float, expected_y: float
) -> None:
    """Vertical gaze must not cancel when the eyelids move with the iris.

    An eyelid-relative ratio would report 0.5 for both cases because the iris
    remains centered between the moving lids. The corner-axis feature keeps
    the corners as the stable reference and therefore preserves the motion.
    """

    landmarks = EyeLandmarks(
        outer_corner=NormalizedPoint(0.10, 0.20),
        inner_corner=NormalizedPoint(0.30, 0.20),
        upper_lid=NormalizedPoint(0.20, iris_y - 0.03),
        lower_lid=NormalizedPoint(0.20, iris_y + 0.03),
        iris_frame_points=(NormalizedPoint(0.20, iris_y),),
        detector_confidence=0.9,
    )

    result = extract_eye_features(landmarks, occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED)

    assert result.valid
    assert result.eye is not None and result.eye.iris_in_eye is not None
    assert result.eye.iris_in_eye.y == pytest.approx(expected_y)


def test_occluded_or_degenerate_eye_is_invalid_without_feature_values() -> None:
    missing = extract_eye_features(None, occlusion_reason=ReasonCode.RIGHT_EYE_OCCLUDED)
    degenerate = extract_eye_features(
        EyeLandmarks(
            outer_corner=NormalizedPoint(0.2, 0.2),
            inner_corner=NormalizedPoint(0.2, 0.2),
            upper_lid=NormalizedPoint(0.2, 0.2),
            lower_lid=NormalizedPoint(0.2, 0.3),
            iris_frame_points=(NormalizedPoint(0.2, 0.25),),
        ),
        occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED,
    )

    assert missing.eye is None and not missing.valid
    assert missing.reason_codes == (ReasonCode.RIGHT_EYE_OCCLUDED,)
    assert degenerate.eye is None and not degenerate.valid
    assert degenerate.reason_codes == (ReasonCode.LOW_CONFIDENCE,)


def test_implausible_openness_is_invalid_instead_of_raising() -> None:
    implausible = EyeLandmarks(
        outer_corner=NormalizedPoint(0.1, 0.5),
        inner_corner=NormalizedPoint(0.3, 0.5),
        upper_lid=NormalizedPoint(0.2, 0.1),
        lower_lid=NormalizedPoint(0.2, 0.9),
        iris_frame_points=(NormalizedPoint(0.2, 0.5),),
    )

    result = extract_eye_features(implausible, occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED)

    assert not result.valid
    assert result.eye is None
    assert result.reason_codes == (ReasonCode.LOW_CONFIDENCE,)


def test_iris_outside_eye_is_not_clamped_into_a_valid_value() -> None:
    landmarks = _eye()
    outside = EyeLandmarks(
        outer_corner=landmarks.outer_corner,
        inner_corner=landmarks.inner_corner,
        upper_lid=landmarks.upper_lid,
        lower_lid=landmarks.lower_lid,
        iris_frame_points=(NormalizedPoint(0.05, 0.20),),
    )

    result = extract_eye_features(outside, occlusion_reason=ReasonCode.LEFT_EYE_OCCLUDED)

    assert not result.valid
    assert result.eye is None
    assert result.reason_codes == (ReasonCode.OUT_OF_RANGE,)


def test_feature_result_preserves_explicit_head_pose_and_eye_reasons() -> None:
    result = extract_features(
        _eye(),
        None,
        head_pose=HeadPose(yaw_deg=2.0, pitch_deg=-1.0, roll_deg=0.0),
    )

    assert result.left_eye.valid
    assert not result.right_eye.valid
    assert result.head_pose == HeadPose(2.0, -1.0, 0.0)
    assert result.reason_codes == (ReasonCode.RIGHT_EYE_OCCLUDED,)
