"""Held-out test dataset schema and recorder; no Qt, camera, or display."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gazelink.calibration import CalibrationSample
from gazelink.calibration_ui import CalibrationTimingSettings
from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    GazePoint,
    GazeSample,
    HeadPose,
    NormalizedBox,
    NormalizedPoint,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_engine import GazeEstimationResult
from gazelink.gaze_features import from_calibration_sample
from gazelink.live_validation import LiveValidationController, LiveValidationTarget
from gazelink.test_dataset import (
    TestPointResult,
    TestSample,
    TestSampleRecorder,
    write_test_dataset,
)

pytestmark = pytest.mark.unit


def _geometry() -> ScreenGeometry:
    return ScreenGeometry("LS49C95xU", 4096, 1152, 1.25)


def _sample_fields() -> dict[str, object]:
    return {
        "source_frame_id": 7,
        "observed_at_monotonic_ms": 123.5,
        "left_iris_in_eye": NormalizedPoint(0.40, 0.30),
        "right_iris_in_eye": NormalizedPoint(0.60, 0.30),
        "left_openness": 0.70,
        "right_openness": 0.70,
        "left_iris_in_lids_y": 0.50,
        "right_iris_in_lids_y": 0.50,
        "head_yaw_deg": 1.0,
        "head_pitch_deg": -20.0,
        "head_roll_deg": 2.0,
        "confidence": 0.5,
        "accepted": True,
        "reason": "accepted",
    }


def _observation(frame_id: int) -> VisionObservation:
    eye = EyeFeatures(
        iris_center=NormalizedPoint(0.5, 0.5),
        openness=0.7,
        confidence=0.5,
        iris_in_eye=NormalizedPoint(0.45, 0.35),
        iris_in_lids_y=0.5,
    )
    return VisionObservation(
        frame_id=frame_id,
        observed_at_monotonic_ms=float(frame_id),
        tracking_state=TrackingState.TRACKED,
        face_box=NormalizedBox(0.2, 0.2, 0.4, 0.4),
        left_eye=eye,
        right_eye=eye,
        head_pose=HeadPose(yaw_deg=1.0, pitch_deg=-20.0, roll_deg=2.0),
        overall_confidence=0.5,
    )


def _estimation(frame_id: int) -> GazeEstimationResult:
    point = GazePoint(0.5, 0.5)
    return GazeEstimationResult(
        GazeSample(
            source_frame_id=frame_id,
            sampled_at_monotonic_ms=float(frame_id),
            raw_normalized=point,
            corrected_normalized=point,
            filtered_normalized=point,
            screen_position=PixelPoint(0, 0),
            screen_id="LS49C95xU",
            confidence=0.5,
            valid_for_control=False,
        ),
        (),
    )


def _controller(target_count: int = 2) -> LiveValidationController:
    return LiveValidationController(
        _geometry(),
        targets=tuple(
            LiveValidationTarget(f"TEST_{index}", GazePoint(0.3 + 0.1 * index, 0.5))
            for index in range(target_count)
        ),
        timing=CalibrationTimingSettings(
            stabilization_ms=0.0,
            capture_window_ms=0.0,
            min_sample_interval_ms=0.0,
        ),
    )


# --- schema parity -------------------------------------------------------


def test_sample_dict_is_identical_to_a_calibration_sample() -> None:
    """analyze.py reads both files with one code path; the schemas must match."""

    fields = _sample_fields()
    calibration = CalibrationSample(target_index=3, **fields)  # type: ignore[arg-type]
    test_sample = TestSample(target_index=3, **fields)  # type: ignore[arg-type]
    assert calibration.to_dict() == test_sample.to_dict()


def test_target_index_may_exceed_the_nine_calibration_targets() -> None:
    sample = TestSample(target_index=11, **_sample_fields())  # type: ignore[arg-type]
    assert sample.target_index == 11


def test_sample_round_trips_through_json() -> None:
    sample = TestSample(target_index=9, **_sample_fields())  # type: ignore[arg-type]
    assert TestSample.from_dict(json.loads(json.dumps(sample.to_dict()))) == sample


def test_features_match_the_calibration_extractor_exactly() -> None:
    """Feature math is reused, never reimplemented for test points."""

    fields = _sample_fields()
    calibration = CalibrationSample(target_index=0, **fields)  # type: ignore[arg-type]
    test_sample = TestSample(target_index=9, **fields)  # type: ignore[arg-type]
    assert test_sample.features() == from_calibration_sample(calibration)


def test_rejected_sample_yields_no_features() -> None:
    fields = {**_sample_fields(), "accepted": False, "reason": "low_confidence"}
    assert TestSample(target_index=0, **fields).features() is None  # type: ignore[arg-type]


def test_sample_rejects_a_negative_target_index() -> None:
    with pytest.raises(ContractValidationError):
        TestSample(target_index=-1, **_sample_fields())  # type: ignore[arg-type]


def test_sample_still_enforces_the_calibration_field_contract() -> None:
    """Validation is delegated, so a bad confidence must still be caught."""

    with pytest.raises(ContractValidationError):
        TestSample(target_index=0, **{**_sample_fields(), "confidence": 5.0})  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError):
        TestSample(target_index=0, **{**_sample_fields(), "reason": "invented_reason"})  # type: ignore[arg-type]


# --- result -------------------------------------------------------------


def _result(samples: tuple[TestSample, ...], target_count: int = 2) -> TestPointResult:
    return TestPointResult(
        targets=tuple(
            LiveValidationTarget(f"TEST_{index}", GazePoint(0.3 + 0.1 * index, 0.5))
            for index in range(target_count)
        ),
        sample_counts=(len(samples), 0)[:target_count],
        samples=samples,
        camera_id="camera-0",
        screen_geometry=_geometry(),
        seed=20260903,
        overlay_model_id=None,
    )


def test_result_round_trips_through_json() -> None:
    sample = TestSample(target_index=0, **_sample_fields())  # type: ignore[arg-type]
    result = _result((sample,))
    assert TestPointResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result


def test_result_rejects_a_sample_pointing_at_an_undeclared_target() -> None:
    sample = TestSample(target_index=5, **_sample_fields())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError):
        _result((sample,))


def test_written_dataset_lands_in_its_own_directory(tmp_path: Path) -> None:
    """Test data must never be mistaken for, or mixed with, calibration data."""

    sample = TestSample(target_index=0, **_sample_fields())  # type: ignore[arg-type]
    paths = write_test_dataset(
        _result((sample,)),
        directory=tmp_path / "test_points",
        generated_at=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert paths.json_path.name.startswith("test_dataset_")
    assert paths.json_path.parent.name == "test_points"
    reloaded = TestPointResult.from_dict(json.loads(paths.json_path.read_text(encoding="utf-8")))
    assert reloaded.samples[0] == sample
    assert "No OS input was emitted." in paths.text_path.read_text(encoding="utf-8")
    assert not list((tmp_path / "test_points").glob("*.tmp"))


# --- recorder ------------------------------------------------------------


def _drive(
    controller: LiveValidationController,
    recorder: TestSampleRecorder,
    frames: range,
) -> None:
    for frame_id in frames:
        view = controller.ingest(_estimation(frame_id), now_monotonic_ms=float(frame_id))
        recorder.observe(view, _observation(frame_id))


def test_recorder_commits_exactly_what_the_controller_measured() -> None:
    """The wrapper's inference is checked against the controller's own count."""

    controller = _controller(target_count=1)
    recorder = TestSampleRecorder()
    _drive(controller, recorder, range(1, 20))
    view = controller.view()
    assert view.complete
    assert recorder.sample_counts == (view.measurements[0].sample_count,)
    assert len(recorder.samples) == view.measurements[0].sample_count
    assert {sample.target_index for sample in recorder.samples} == {0}


def test_recorder_attributes_samples_to_the_right_target() -> None:
    controller = _controller(target_count=2)
    recorder = TestSampleRecorder()
    _drive(controller, recorder, range(1, 40))
    view = controller.view()
    assert view.complete
    assert len(recorder.sample_counts) == 2
    for index, measurement in enumerate(view.measurements):
        rows = [sample for sample in recorder.samples if sample.target_index == index]
        assert len(rows) == measurement.sample_count


def test_recorder_discards_pending_rows_when_a_target_resets() -> None:
    """Withheld tracking resets the target; its partial rows must not survive."""

    controller = _controller(target_count=1)
    recorder = TestSampleRecorder()
    for frame_id in range(1, 4):
        view = controller.ingest(_estimation(frame_id), now_monotonic_ms=float(frame_id))
        recorder.observe(view, _observation(frame_id))
    assert recorder.samples == ()
    withheld = GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE,))
    view = controller.ingest(withheld, now_monotonic_ms=10.0)
    recorder.observe(view, None)
    assert view.accepted_samples == 0
    _drive(controller, recorder, range(11, 30))
    completed = controller.view()
    assert completed.complete
    # Only the samples from the successful retry are on disk.
    assert len(recorder.samples) == completed.measurements[0].sample_count


def test_recorder_drops_everything_on_restart() -> None:
    """Rows from two different passes must never blend into one file."""

    controller = _controller(target_count=1)
    recorder = TestSampleRecorder()
    _drive(controller, recorder, range(1, 20))
    assert recorder.samples
    recorder.observe(controller.restart(), None)
    assert recorder.samples == ()
    assert recorder.sample_counts == ()


def test_recorder_rejects_a_non_view() -> None:
    with pytest.raises(ContractValidationError):
        TestSampleRecorder().observe("not a view", None)  # type: ignore[arg-type]
