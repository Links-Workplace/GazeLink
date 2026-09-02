"""Deterministic safety properties for local supervised gaze correction."""

from __future__ import annotations

from pathlib import Path

import pytest

from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    GazePoint,
    GazeSample,
    HeadPose,
    NormalizedPoint,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_correction import (
    CorrectionSample,
    CorrectionSettings,
    CorrectionStore,
    LocalCorrectionEngine,
    accept_candidate,
    capture_correction,
    pixel_to_normalized,
    validate_candidate,
)
from gazelink.gaze_engine import CalibrationModel, GazeEstimator
from gazelink.gaze_features import GazeFeatureVector
from gazelink.gaze_model import GazeModelKind, RegressionModel


def _geometry(*, screen_id: str = "primary") -> ScreenGeometry:
    return ScreenGeometry(screen_id, 1001, 501, 1.0)


def _model(*, x: float = 0.6, y: float = 0.5) -> CalibrationModel:
    return CalibrationModel.create(
        regression=RegressionModel(
            GazeModelKind.LINEAR,
            (x,) + (0.0,) * 9,
            (y,) + (0.0,) * 9,
            0.0,
        ),
        screen_geometry=_geometry(),
        camera_id="camera-0",
        calibration_id="test-calibration",
        trained_at_utc="2026-09-02T00:00:00+00:00",
    )


def _estimator() -> GazeEstimator:
    return GazeEstimator(_model(), live_screen_geometry=_geometry())


def _observation(frame_id: int, *, at_ms: float, iris_offset: float = 0.0) -> VisionObservation:
    return VisionObservation(
        frame_id=frame_id,
        observed_at_monotonic_ms=at_ms,
        tracking_state=TrackingState.TRACKED,
        face_box=None,
        left_eye=EyeFeatures(None, 0.5 + iris_offset, 0.5, NormalizedPoint(0.4 + iris_offset, 0.5)),
        right_eye=EyeFeatures(
            None, 0.55 + iris_offset, 0.5, NormalizedPoint(0.6 - iris_offset, 0.5)
        ),
        head_pose=HeadPose(iris_offset, -iris_offset, 0.0),
        overall_confidence=0.5,
    )


def _window(*, start_frame: int = 1, base_ms: float = 700.0) -> tuple[VisionObservation, ...]:
    return tuple(
        _observation(
            start_frame + index,
            at_ms=base_ms + (index * 50.0),
            iris_offset=(index - 2) * 0.001,
        )
        for index in range(5)
    )


def _candidate(*, raw_x: float = 0.6, true_x: float = 0.8) -> CorrectionSample:
    return CorrectionSample.create(
        features=GazeFeatureVector((0.4, 0.5, 0.4, 0.5, 0.5, 0.55, 0.0, 0.0, 0.0)),
        base_raw_prediction=GazePoint(raw_x, 0.5),
        true_target=GazePoint(true_x, 0.5),
        screen_geometry=_geometry(),
        base_model_id=_model().model_id,
        capture_sample_count=5,
        source_frame_ids=(1, 2, 3, 4, 5),
        feature_spread=(0.0,) * 9,
        captured_at_utc="2026-09-02T00:00:00+00:00",
    )


def _sample(raw: GazePoint) -> GazeSample:
    return GazeSample(
        source_frame_id=1,
        sampled_at_monotonic_ms=1_000.0,
        raw_normalized=raw,
        corrected_normalized=raw,
        filtered_normalized=raw,
        screen_position=PixelPoint(600, 250),
        screen_id="primary",
        confidence=0.5,
        valid_for_control=False,
    )


def test_capture_uses_only_recent_tracked_feature_complete_observations() -> None:
    lost = VisionObservation(
        frame_id=99,
        observed_at_monotonic_ms=900.0,
        tracking_state=TrackingState.LOST,
        face_box=None,
        left_eye=None,
        right_eye=None,
        head_pose=None,
        overall_confidence=0.0,
        reason_codes=(ReasonCode.FACE_NOT_FOUND,),
    )
    result = capture_correction(
        (*_window(), lost),
        true_target=GazePoint(0.8, 0.5),
        estimator=_estimator(),
        now_monotonic_ms=1_000.0,
        captured_at_utc="2026-09-02T00:00:00+00:00",
    )
    assert result.reason_codes == ()
    assert result.candidate is not None
    assert result.candidate.capture_sample_count == 5
    assert 99 not in result.candidate.source_frame_ids


def test_short_or_unstable_capture_is_rejected_without_candidate() -> None:
    short = capture_correction(
        _window()[:4],
        true_target=GazePoint(0.8, 0.5),
        estimator=_estimator(),
        now_monotonic_ms=1_000.0,
    )
    unstable_window = (*_window()[:4], _observation(10, at_ms=950.0, iris_offset=0.2))
    unstable = capture_correction(
        unstable_window,
        true_target=GazePoint(0.8, 0.5),
        estimator=_estimator(),
        now_monotonic_ms=1_000.0,
    )
    assert short.candidate is None
    assert short.reason_codes == (ReasonCode.CORRECTION_INSUFFICIENT,)
    assert unstable.candidate is None
    assert unstable.reason_codes == (ReasonCode.CORRECTION_UNSTABLE,)


def test_pixel_ground_truth_converts_exactly_to_normalized_space() -> None:
    assert pixel_to_normalized(PixelPoint(1000, 500), _geometry()) == GazePoint(1.0, 1.0)
    assert pixel_to_normalized(PixelPoint(500, 250), _geometry()) == GazePoint(0.5, 0.5)
    with pytest.raises(ContractValidationError, match="within screen bounds"):
        pixel_to_normalized(PixelPoint(1001, 0), _geometry())


def test_one_correction_reduces_local_error_but_has_zero_distant_effect() -> None:
    engine = LocalCorrectionEngine(_model().model_id, _geometry()).with_correction(_candidate())
    local = engine.correct(GazePoint(0.6, 0.5))
    distant = engine.correct(GazePoint(0.9, 0.5))
    assert abs(local.corrected.x - 0.8) < abs(0.6 - 0.8)
    assert distant.corrected == GazePoint(0.9, 0.5)
    assert distant.contributing_corrections == 0


def test_influence_is_exactly_zero_at_the_configured_radius_boundary() -> None:
    settings = CorrectionSettings(influence_radius=0.2)
    engine = LocalCorrectionEngine(
        _model().model_id, _geometry(), (_candidate(),), settings=settings
    )
    assert engine.correct(GazePoint(0.8, 0.5)).corrected == GazePoint(0.8, 0.5)


def test_repeated_consistent_corrections_strengthen_gradually_and_stay_bounded() -> None:
    settings = CorrectionSettings(max_total_offset=0.12)
    empty = LocalCorrectionEngine(_model().model_id, _geometry(), settings=settings)
    once = empty.with_correction(_candidate()).correct(GazePoint(0.6, 0.5)).corrected
    repeated_engine = empty
    for _ in range(10):
        repeated_engine = repeated_engine.with_correction(_candidate())
    repeated = repeated_engine.correct(GazePoint(0.6, 0.5)).corrected
    assert 0.6 < once.x < repeated.x
    assert repeated.x <= 0.72


def test_conflicting_nearby_corrections_fail_conservatively() -> None:
    positive = _candidate(raw_x=0.6, true_x=0.8)
    negative = _candidate(raw_x=0.6, true_x=0.4)
    engine = LocalCorrectionEngine(_model().model_id, _geometry(), (positive, negative))
    result = engine.correct(GazePoint(0.6, 0.5))
    assert result.corrected == GazePoint(0.6, 0.5)
    assert result.reason_codes == (ReasonCode.CORRECTION_CONFLICT,)


def test_candidate_is_inert_until_held_out_validation_accepts_it() -> None:
    estimator = _estimator()
    capture = capture_correction(
        _window(),
        true_target=GazePoint(0.8, 0.5),
        estimator=estimator,
        now_monotonic_ms=1_000.0,
        captured_at_utc="2026-09-02T00:00:00+00:00",
    )
    assert capture.candidate is not None
    empty = LocalCorrectionEngine(estimator.model_id, _geometry())
    assert empty.correct(GazePoint(0.6, 0.5)).corrected == GazePoint(0.6, 0.5)
    validation = validate_candidate(
        capture.candidate,
        _window(start_frame=20, base_ms=1_100.0),
        validation_target=GazePoint(0.8, 0.5),
        estimator=estimator,
        active_engine=empty,
        now_monotonic_ms=1_400.0,
    )
    assert validation.accepted
    assert validation.after_error_normalized < validation.before_error_normalized
    accepted = accept_candidate(validation, empty)
    assert accepted.correct(GazePoint(0.6, 0.5)).corrected.x > 0.6


def test_validation_rejects_reused_capture_frames() -> None:
    estimator = _estimator()
    candidate = capture_correction(
        _window(),
        true_target=GazePoint(0.8, 0.5),
        estimator=estimator,
        now_monotonic_ms=1_000.0,
    ).candidate
    assert candidate is not None
    validation = validate_candidate(
        candidate,
        _window(),
        validation_target=GazePoint(0.8, 0.5),
        estimator=estimator,
        active_engine=LocalCorrectionEngine(estimator.model_id, _geometry()),
        now_monotonic_ms=1_000.0,
    )
    assert not validation.accepted
    assert validation.reason_codes == (ReasonCode.CORRECTION_INSUFFICIENT,)


def test_undo_clear_disable_and_reset_reproduce_base_behavior() -> None:
    base = LocalCorrectionEngine(_model().model_id, _geometry())
    corrected = base.with_correction(_candidate())
    raw_sample = _sample(GazePoint(0.6, 0.5))
    assert corrected.apply_to_sample(raw_sample).corrected_normalized.x > 0.6
    assert corrected.disable().apply_to_sample(raw_sample) == raw_sample
    assert corrected.undo_last().apply_to_sample(raw_sample) == raw_sample
    assert corrected.clear().apply_to_sample(raw_sample) == raw_sample


def test_persistence_round_trip_and_incompatible_or_corrupt_files_reset_to_base(
    tmp_path: Path,
) -> None:
    engine = LocalCorrectionEngine(_model().model_id, _geometry()).with_correction(_candidate())
    store = CorrectionStore(tmp_path)
    path = store.save(engine)
    assert path.exists()
    assert store.load(base_model_id=_model().model_id, screen_geometry=_geometry()) == engine
    incompatible = store.load(base_model_id="different", screen_geometry=_geometry())
    assert incompatible.corrections == ()
    (tmp_path / "latest_corrections.json").write_text("broken", encoding="utf-8")
    recovered = store.load(base_model_id=_model().model_id, screen_geometry=_geometry())
    assert recovered.corrections == ()


def test_model_mismatch_rejects_stale_correction() -> None:
    other_model = _model(x=0.5)
    engine = LocalCorrectionEngine(other_model.model_id, _geometry())
    with pytest.raises(ContractValidationError, match="incompatible"):
        engine.with_correction(_candidate())
