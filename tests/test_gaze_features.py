"""Deterministic tests for the persisted M2 gaze feature schema."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gazelink.calibration import FEATURE_SCHEMA_VERSION, CalibrationSample
from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    HeadPose,
    NormalizedPoint,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_features import (
    FEATURE_NAMES,
    GazeFeatureVector,
    from_calibration_sample,
    from_observation,
    select_stable_feature_window,
)


def _sample(*, accepted: bool = True, missing: bool = False) -> CalibrationSample:
    return CalibrationSample(
        source_frame_id=7,
        observed_at_monotonic_ms=100.0,
        target_index=4,
        left_iris_in_eye=None if missing else NormalizedPoint(0.25, 0.4),
        right_iris_in_eye=None if missing else NormalizedPoint(0.75, 0.6),
        left_openness=None if missing else 0.5,
        right_openness=None if missing else 0.7,
        left_iris_in_lids_y=None if missing else 0.3,
        right_iris_in_lids_y=None if missing else 0.7,
        head_yaw_deg=None if missing else 3.0,
        head_pitch_deg=None if missing else -2.0,
        head_roll_deg=None if missing else 1.0,
        confidence=0.5,
        accepted=accepted,
        reason="accepted" if accepted else "features_unavailable",
    )


def _observation() -> VisionObservation:
    return VisionObservation(
        frame_id=7,
        observed_at_monotonic_ms=100.0,
        tracking_state=TrackingState.TRACKED,
        face_box=None,
        left_eye=EyeFeatures(None, 0.5, 0.5, NormalizedPoint(0.25, 0.4), 0.3),
        right_eye=EyeFeatures(None, 0.7, 0.5, NormalizedPoint(0.75, 0.6), 0.7),
        head_pose=HeadPose(3.0, -2.0, 1.0),
        overall_confidence=0.5,
    )


def test_feature_order_and_right_eye_handedness_are_pinned() -> None:
    features = from_calibration_sample(_sample())
    assert features is not None
    assert FEATURE_NAMES == (
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
    assert features.values == (0.25, 0.4, 0.25, 0.6, 0.3, 0.7, 3.0, -2.0, 1.0)


def test_legacy_sample_without_lid_ratio_uses_corner_axis_as_a_safe_fallback() -> None:
    features = from_calibration_sample(
        replace(_sample(), left_iris_in_lids_y=None, right_iris_in_lids_y=None)
    )
    assert features is not None
    assert features.values[4:6] == (0.4, 0.6)


def test_live_and_calibration_feature_paths_are_identical() -> None:
    assert from_observation(_observation()) == from_calibration_sample(_sample())


def test_rejected_or_missing_sample_is_not_trainable() -> None:
    assert from_calibration_sample(_sample(accepted=False, missing=True)) is None
    assert from_calibration_sample(_sample(missing=True)) is None


def test_missing_live_feature_is_not_predictable() -> None:
    observation = _observation()
    missing = VisionObservation(
        frame_id=observation.frame_id,
        observed_at_monotonic_ms=observation.observed_at_monotonic_ms,
        tracking_state=observation.tracking_state,
        face_box=None,
        left_eye=observation.left_eye,
        right_eye=None,
        head_pose=observation.head_pose,
        overall_confidence=observation.overall_confidence,
    )
    assert from_observation(missing) is None


def test_feature_vector_round_trip_and_schema_mismatch_rejection() -> None:
    vector = from_observation(_observation())
    assert vector is not None
    assert GazeFeatureVector.from_dict(vector.to_dict()) == vector
    with pytest.raises(ContractValidationError, match="unsupported feature schema"):
        GazeFeatureVector(vector.values, schema_version=FEATURE_SCHEMA_VERSION + 1)


def _vector(*, eye: float = 0.5, yaw: float = 0.0) -> GazeFeatureVector:
    return GazeFeatureVector((eye, eye, eye, eye, 0.5, 0.5, yaw, 0.0, 0.0))


def test_stable_window_selector_skips_a_noisy_prefix_without_splicing_frames() -> None:
    vectors = (
        _vector(eye=0.1),
        _vector(eye=0.9),
        _vector(eye=0.500),
        _vector(eye=0.501),
        _vector(eye=0.499),
    )

    selected = select_stable_feature_window(
        vectors,
        window_size=3,
        max_eye_p95=0.01,
        max_head_pose_p95_deg=1.0,
    )

    assert selected is not None
    assert selected[:2] == (2, 5)


def test_stable_window_selector_rejects_eye_or_head_motion() -> None:
    assert (
        select_stable_feature_window(
            (_vector(eye=0.1), _vector(eye=0.5), _vector(eye=0.9)),
            window_size=3,
            max_eye_p95=0.08,
            max_head_pose_p95_deg=5.0,
        )
        is None
    )
    assert (
        select_stable_feature_window(
            (_vector(yaw=-10.0), _vector(yaw=0.0), _vector(yaw=10.0)),
            window_size=3,
            max_eye_p95=0.08,
            max_head_pose_p95_deg=5.0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_size", 0),
        ("window_size", True),
        ("max_eye_p95", 0.0),
        ("max_eye_p95", float("inf")),
        ("max_head_pose_p95_deg", 0.0),
    ],
)
def test_stable_window_selector_rejects_invalid_thresholds(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "window_size": 2,
        "max_eye_p95": 0.08,
        "max_head_pose_p95_deg": 5.0,
    }
    kwargs[field] = value
    with pytest.raises(ContractValidationError, match=field):
        select_stable_feature_window((_vector(), _vector()), **kwargs)  # type: ignore[arg-type]
