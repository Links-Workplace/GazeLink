"""Behavioral and safety tests for M2 gaze training, estimation, and storage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gazelink.calibration import (
    FEATURE_SCHEMA_VERSION,
    CalibrationSample,
    CalibrationSessionResult,
    default_nine_point_targets,
)
from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    GazePoint,
    HeadPose,
    NormalizedPoint,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_engine import (
    CalibrationEngine,
    CalibrationModel,
    CalibrationStore,
    GazeEstimator,
    normalized_to_pixel,
)
from gazelink.gaze_model import GazeModelKind, GazeModelProfile, RegressionModel


def _geometry(*, screen_id: str = "primary") -> ScreenGeometry:
    return ScreenGeometry(screen_id, 1920, 1080, 1.0)


def _result() -> CalibrationSessionResult:
    targets = default_nine_point_targets(edge_inset=0.1)
    samples: list[CalibrationSample] = []
    frame_id = 0
    for target in targets:
        x = target.screen_position.x
        y = target.screen_position.y
        for offset in (-0.004, 0.0, 0.004):
            samples.append(
                CalibrationSample(
                    source_frame_id=frame_id,
                    observed_at_monotonic_ms=100.0 + frame_id,
                    target_index=target.index,
                    left_iris_in_eye=NormalizedPoint(x + offset, y - offset),
                    right_iris_in_eye=NormalizedPoint(1.0 - x + offset, y + offset),
                    left_openness=0.5,
                    right_openness=0.55,
                    head_yaw_deg=(x - 0.5) * 4,
                    head_pitch_deg=(y - 0.5) * 3,
                    head_roll_deg=offset,
                    confidence=0.5,
                    accepted=True,
                    reason="accepted",
                )
            )
            frame_id += 1
    samples.append(
        CalibrationSample(
            source_frame_id=frame_id,
            observed_at_monotonic_ms=500.0,
            target_index=0,
            left_iris_in_eye=None,
            right_iris_in_eye=None,
            left_openness=None,
            right_openness=None,
            head_yaw_deg=None,
            head_pitch_deg=None,
            head_roll_deg=None,
            confidence=0.0,
            accepted=False,
            reason="tracking_not_ready",
        )
    )
    return CalibrationSessionResult(
        target_order=tuple(range(9)),
        targets=targets,
        sample_counts=(3,) * 9,
        samples=tuple(samples),
        camera_id="camera-0",
        screen_geometry=_geometry(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        min_samples_per_target=3,
        started_at_monotonic_ms=100.0,
        completed_at_monotonic_ms=600.0,
    )


def _observation(*, at_ms: float = 1_000.0) -> VisionObservation:
    return VisionObservation(
        frame_id=11,
        observed_at_monotonic_ms=at_ms,
        tracking_state=TrackingState.TRACKED,
        face_box=None,
        left_eye=EyeFeatures(None, 0.5, 0.5, NormalizedPoint(0.5, 0.5)),
        right_eye=EyeFeatures(None, 0.55, 0.5, NormalizedPoint(0.5, 0.5)),
        head_pose=HeadPose(0.0, 0.0, 0.0),
        overall_confidence=0.5,
    )


def _constant_model(x: float, y: float) -> CalibrationModel:
    coefficients_x = (x,) + (0.0,) * 9
    coefficients_y = (y,) + (0.0,) * 9
    return CalibrationModel.create(
        regression=RegressionModel(
            GazeModelKind.LINEAR, coefficients_x, coefficients_y, ridge_lambda=0.0
        ),
        screen_geometry=_geometry(),
        camera_id="camera-0",
        calibration_id="test-calibration",
        trained_at_utc="2026-09-02T00:00:00+00:00",
    )


def test_training_uses_real_target_coordinates_and_returns_a_measured_model() -> None:
    outcome = CalibrationEngine().train(_result())
    assert outcome.model.screen_geometry == _geometry()
    assert outcome.model.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert len(outcome.comparison.baseline.per_target_error_normalized) == 9
    assert len(outcome.comparison.advanced.per_target_error_normalized) == 9
    assert any(
        model.regression.kind is GazeModelKind.LINEAR
        and model.regression.profile is GazeModelProfile.AXIS_IRIS
        for model in outcome.candidate_models
    )
    assert any(
        model.regression.kind is GazeModelKind.LINEAR
        and model.regression.profile is GazeModelProfile.AXIS_IRIS_LIDS
        for model in outcome.candidate_models
    )
    assert outcome.promotable
    assert outcome.quality_reasons == ()


def test_training_rejects_a_dataset_from_the_previous_feature_meaning() -> None:
    old_schema_result = replace(_result(), feature_schema_version=FEATURE_SCHEMA_VERSION - 1)

    with pytest.raises(ContractValidationError, match="feature schema is incompatible"):
        CalibrationEngine().train(old_schema_result)


def test_model_identity_changes_when_the_calibration_session_changes() -> None:
    engine = CalibrationEngine()
    first = engine.train(_result()).model
    second_result = replace(_result(), completed_at_monotonic_ms=601.0)
    second = engine.train(second_result).model

    assert first.calibration_id != second.calibration_id
    assert first.model_id != second.model_id


def test_live_estimation_keeps_raw_corrected_and_filtered_explicit() -> None:
    estimator = GazeEstimator(_constant_model(0.4, 0.6), live_screen_geometry=_geometry())
    result = estimator.estimate(_observation(), now_monotonic_ms=1_010.0)
    assert result.reason_codes == ()
    assert result.sample is not None
    assert result.sample.raw_normalized == GazePoint(0.4, 0.6)
    assert result.sample.corrected_normalized == result.sample.raw_normalized
    assert result.sample.filtered_normalized == result.sample.raw_normalized
    assert result.sample.valid_for_control is False


def test_stale_future_or_missing_input_never_fabricates_a_coordinate() -> None:
    estimator = GazeEstimator(_constant_model(0.4, 0.6), live_screen_geometry=_geometry())
    stale = estimator.estimate(_observation(at_ms=1_000.0), now_monotonic_ms=1_300.0)
    future = estimator.estimate(_observation(at_ms=1_100.0), now_monotonic_ms=1_000.0)
    missing = estimator.estimate(replace(_observation(), left_eye=None), now_monotonic_ms=1_010.0)
    assert stale.sample is None and stale.reason_codes == (ReasonCode.CALIBRATION_STALE,)
    assert future.sample is None and future.reason_codes == (ReasonCode.CALIBRATION_STALE,)
    assert missing.sample is None and missing.reason_codes == (ReasonCode.CALIBRATION_INVALID,)


def test_mismatched_screen_rejects_model_without_prediction() -> None:
    estimator = GazeEstimator(
        _constant_model(0.4, 0.6), live_screen_geometry=_geometry(screen_id="other")
    )
    result = estimator.estimate(_observation(), now_monotonic_ms=1_010.0)
    assert result.sample is None
    assert result.reason_codes == (ReasonCode.CALIBRATION_INVALID,)


def test_out_of_range_prediction_preserves_raw_but_clamps_pixel_output() -> None:
    estimator = GazeEstimator(_constant_model(1.2, -0.2), live_screen_geometry=_geometry())
    result = estimator.estimate(_observation(), now_monotonic_ms=1_010.0)
    assert result.sample is not None
    assert result.sample.raw_normalized == GazePoint(1.2, -0.2)
    assert result.sample.screen_position == normalized_to_pixel(GazePoint(1.2, -0.2), _geometry())
    assert result.reason_codes == (ReasonCode.OUT_OF_RANGE, ReasonCode.CLAMPED_TO_SCREEN)


def test_normalized_pixel_conversion_uses_screen_extents_without_dpi_guessing() -> None:
    assert normalized_to_pixel(GazePoint(0.0, 0.0), _geometry()).to_dict() == {
        "x_px": 0,
        "y_px": 0,
    }
    assert normalized_to_pixel(GazePoint(1.0, 1.0), _geometry()).to_dict() == {
        "x_px": 1919,
        "y_px": 1079,
    }


def test_model_and_dataset_persistence_round_trip_and_recover_from_corruption(
    tmp_path: Path,
) -> None:
    outcome = CalibrationEngine().train(_result())
    store = CalibrationStore(tmp_path)
    dataset_path = store.save_dataset(_result())
    model_path = store.save_model(outcome.model)
    assert dataset_path.exists() and model_path.exists()
    assert store.load_latest_model(_geometry()) == outcome.model
    assert store.load_latest_model(_geometry(screen_id="other")) is None
    (tmp_path / "latest_model.json").write_text("{broken", encoding="utf-8")
    assert store.load_latest_model(_geometry()) is None


def test_dataset_can_be_resolved_from_the_candidate_calibration_identity(tmp_path: Path) -> None:
    result = _result()
    outcome = CalibrationEngine().train(result)
    store = CalibrationStore(tmp_path)
    store.save_dataset(result)

    assert store.load_dataset_for_calibration(outcome.model.calibration_id) == result
    assert store.load_dataset_for_calibration("missing-calibration") is None


def test_latest_model_quarantine_is_recoverable_and_disables_loading(tmp_path: Path) -> None:
    outcome = CalibrationEngine().train(_result())
    store = CalibrationStore(tmp_path)
    store.save_model(outcome.model)

    quarantined = store.quarantine_latest_model()

    assert quarantined is not None and quarantined.exists()
    assert "rejected_latest_model_" in quarantined.name
    assert not (tmp_path / "latest_model.json").exists()
    assert store.load_latest_model(_geometry()) is None


def test_pending_model_never_replaces_latest_until_explicit_promotion(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    old = _constant_model(0.1, 0.1)
    candidate = _constant_model(0.7, 0.7)
    store.save_model(old)

    pending = store.save_pending_model(candidate)

    assert pending.candidate_path.exists()
    assert store.load_latest_model(_geometry()) == old
    assert store.load_pending_model(_geometry()) == pending
    promoted = store.promote_pending_model(_geometry())
    assert promoted is not None and promoted.exists()
    assert store.load_latest_model(_geometry()) == candidate
    assert store.load_pending_model(_geometry()) is None


def test_live_comparison_can_promote_a_non_default_pending_candidate(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    linear = _constant_model(0.2, 0.2)
    polynomial = _constant_model(0.8, 0.8)

    pending = store.save_pending_models((linear, polynomial), selected_model_id=polynomial.model_id)

    loaded = store.load_pending_model(_geometry())
    assert loaded == pending
    assert loaded is not None and len(loaded.candidate_models) == 2
    promoted = store.promote_pending_model(_geometry(), model_id=linear.model_id)
    assert promoted is not None
    assert store.load_latest_model(_geometry()) == linear


def test_tampered_model_identity_is_rejected() -> None:
    model = _constant_model(0.4, 0.6)
    payload = model.to_dict()
    payload["model_id"] = "tampered"
    with pytest.raises(ContractValidationError, match="model_id"):
        CalibrationModel.from_dict(json.loads(json.dumps(payload)))
