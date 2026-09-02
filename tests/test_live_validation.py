"""Deterministic fresh-observation validation tests; no Qt/camera required."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gazelink.calibration_ui import CalibrationTimingSettings
from gazelink.domain import GazePoint, GazeSample, PixelPoint, ReasonCode, ScreenGeometry
from gazelink.gaze_engine import GazeEstimationResult
from gazelink.live_validation import (
    LiveValidationCandidate,
    LiveValidationComparisonController,
    LiveValidationController,
    LiveValidationPhase,
    LiveValidationTarget,
    default_live_validation_targets,
    format_live_validation_comparison_summary,
    format_live_validation_summary,
    write_live_validation_comparison_report,
    write_live_validation_report,
)


def _geometry() -> ScreenGeometry:
    return ScreenGeometry("primary", 1920, 1080, 1.0)


def _result(frame_id: int, point: GazePoint) -> GazeEstimationResult:
    sample = GazeSample(
        source_frame_id=frame_id,
        sampled_at_monotonic_ms=float(frame_id),
        raw_normalized=point,
        corrected_normalized=point,
        filtered_normalized=point,
        screen_position=PixelPoint(0, 0),
        screen_id="primary",
        confidence=0.5,
        valid_for_control=False,
    )
    return GazeEstimationResult(sample, ())


def _controller() -> LiveValidationController:
    return LiveValidationController(
        _geometry(),
        targets=(LiveValidationTarget("CENTER", GazePoint(0.5, 0.5)),),
        timing=CalibrationTimingSettings(
            stabilization_ms=0.0,
            capture_window_ms=0.0,
            min_sample_interval_ms=0.0,
        ),
    )


def _comparison_controller() -> LiveValidationComparisonController:
    return LiveValidationComparisonController(
        _geometry(),
        candidates=(
            LiveValidationCandidate("linear", "LINEAR"),
            LiveValidationCandidate("polynomial", "POLYNOMIAL_RIDGE"),
        ),
        targets=(LiveValidationTarget("CENTER", GazePoint(0.5, 0.5)),),
        timing=CalibrationTimingSettings(
            stabilization_ms=0.0,
            capture_window_ms=0.0,
            min_sample_interval_ms=0.0,
        ),
    )


def test_fresh_validation_reports_live_error_without_declaring_product_pass() -> None:
    controller = _controller()
    for frame_id in range(5):
        view = controller.ingest(
            _result(frame_id, GazePoint(0.7, 0.5)), now_monotonic_ms=float(frame_id)
        )

    assert view.complete
    assert view.phase is LiveValidationPhase.COMPLETE
    assert len(view.measurements) == 1
    measurement = view.measurements[0]
    assert measurement.predicted_normalized == GazePoint(0.7, 0.5)
    assert measurement.sample_count == 5
    assert measurement.median_error_px == pytest.approx(383.8)
    assert measurement.p95_error_px == pytest.approx(383.8)


def test_withheld_tracking_resets_a_partial_window_instead_of_reusing_it() -> None:
    controller = _controller()
    controller.ingest(_result(1, GazePoint(0.5, 0.5)), now_monotonic_ms=1.0)
    view = controller.ingest(
        GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE,)), now_monotonic_ms=2.0
    )

    assert not view.complete
    assert view.accepted_samples == 0
    assert "withheld" in view.feedback


def test_default_validation_includes_centre_and_unseen_upper_positions() -> None:
    positions = {
        target.name: target.screen_position for target in default_live_validation_targets()
    }

    assert positions["CENTER"] == GazePoint(0.5, 0.5)
    assert positions["UP_CENTER"] == GazePoint(0.5, 0.18)
    assert positions["UP_RIGHT"] == GazePoint(0.82, 0.18)


def test_completed_validation_writes_shareable_aggregate_only_reports(tmp_path: Path) -> None:
    controller = _controller()
    for frame_id in range(5):
        view = controller.ingest(
            _result(frame_id, GazePoint(0.5, 0.5)), now_monotonic_ms=float(frame_id)
        )

    paths = write_live_validation_report(
        view,
        model_id="model-1",
        geometry=_geometry(),
        directory=tmp_path,
        generated_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    assert paths.text_path.exists() and paths.json_path.exists()
    text = paths.text_path.read_text(encoding="utf-8")
    payload = paths.json_path.read_text(encoding="utf-8")
    assert "CENTER: target=(0.50, 0.50) predicted=(0.50, 0.50)" in text
    assert "No OS input was emitted." in text
    assert '"model_id": "model-1"' in payload
    assert "landmark" not in text.lower()
    assert "iris" not in payload.lower()
    assert "Fresh validation measurements" in format_live_validation_summary(view)


def test_live_comparison_recommends_only_a_model_that_dominates_every_target(
    tmp_path: Path,
) -> None:
    controller = _comparison_controller()
    for frame_id in range(5):
        view = controller.ingest(
            {
                "linear": _result(frame_id, GazePoint(0.5, 0.5)),
                "polynomial": _result(frame_id, GazePoint(0.7, 0.5)),
            },
            now_monotonic_ms=float(frame_id),
        )

    assert view.complete
    assert view.recommended_model_id == "linear"
    assert "Recommendation: LINEAR" in format_live_validation_comparison_summary(view)
    paths = write_live_validation_comparison_report(
        view,
        geometry=_geometry(),
        directory=tmp_path,
        generated_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    payload = paths.json_path.read_text(encoding="utf-8")
    assert '"recommended_model_id": "linear"' in payload
    assert '"label": "LINEAR"' in payload
    assert "iris" not in payload.lower()
