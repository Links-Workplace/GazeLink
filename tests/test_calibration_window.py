"""Tests for the calibration session log: pure formatting plus file writing.

Only these two functions are tested -- the rest of ``calibration_window.py``
is the Qt window itself, which requires a display server and a live camera,
consistent with the existing precedent for ``debug_window.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gazelink.calibration import (
    FEATURE_SCHEMA_VERSION,
    CalibrationSample,
    CalibrationSessionResult,
    default_nine_point_targets,
)
from gazelink.calibration_window import (
    CalibrationLogEvent,
    format_calibration_log,
    format_training_summary,
    train_and_persist_calibration,
    write_calibration_log,
)
from gazelink.domain import NormalizedPoint, ScreenGeometry
from gazelink.gaze_engine import CalibrationEngine, CalibrationStore


def _geometry() -> ScreenGeometry:
    return ScreenGeometry(screen_id="primary", width_px=1920, height_px=1080, dpi_scale=1.25)


def _dataset_sample(*, accepted: bool = True, target_index: int = 0) -> CalibrationSample:
    return CalibrationSample(
        source_frame_id=1,
        observed_at_monotonic_ms=100.0,
        target_index=target_index,
        left_iris_in_eye=NormalizedPoint(0.42, 0.53) if accepted else None,
        right_iris_in_eye=NormalizedPoint(0.64, 0.68) if accepted else None,
        left_openness=0.55 if accepted else None,
        right_openness=0.65 if accepted else None,
        head_yaw_deg=-0.5 if accepted else None,
        head_pitch_deg=-40.0 if accepted else None,
        head_roll_deg=8.2 if accepted else None,
        confidence=0.5,
        accepted=accepted,
        reason="accepted" if accepted else "tracking_not_ready",
    )


def _pose_sample(
    *,
    accepted: bool,
    reason: str,
    yaw: float | None = None,
    pitch: float | None = None,
    roll: float | None = None,
    target_index: int = 0,
) -> CalibrationSample:
    """Build a sample with an explicit reason and head pose for Metrics tests.

    Unlike ``_dataset_sample`` (which is fixed to "accepted"/"tracking_not_ready"
    and a single hard-coded pose), this lets a test choose any closed-set
    reason and any head pose independently of ``accepted``, so the Metrics
    section's reason-count and head-pose-spread rendering can be pinned
    against hand-computed expected values.
    """

    return CalibrationSample(
        source_frame_id=1,
        observed_at_monotonic_ms=100.0,
        target_index=target_index,
        left_iris_in_eye=NormalizedPoint(0.5, 0.5) if accepted else None,
        right_iris_in_eye=NormalizedPoint(0.5, 0.5) if accepted else None,
        left_openness=0.5 if accepted else None,
        right_openness=0.5 if accepted else None,
        head_yaw_deg=yaw,
        head_pitch_deg=pitch,
        head_roll_deg=roll,
        confidence=0.5,
        accepted=accepted,
        reason=reason,
    )


def _result(*, samples: tuple[CalibrationSample, ...] = ()) -> CalibrationSessionResult:
    return CalibrationSessionResult(
        target_order=tuple(range(9)),
        targets=default_nine_point_targets(),
        sample_counts=(2,) * 9,
        samples=samples,
        camera_id="camera-0",
        screen_geometry=_geometry(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        min_samples_per_target=2,
        started_at_monotonic_ms=1_000.0,
        completed_at_monotonic_ms=1_500.0,
    )


def test_log_reports_accepted_and_rejected_event_counts() -> None:
    events = [
        CalibrationLogEvent(1, True, "TRACKED", 0.5, "Good sample captured."),
        CalibrationLogEvent(
            1, False, "LOST", 0.0, "Tracking is not ready; keep your face visible."
        ),
        CalibrationLogEvent(1, False, "TRACKED", 0.3, "Tracking quality is too low."),
    ]

    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COLLECTING",
        events=events,
        result=None,
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "3 total, 1 accepted, 2 rejected" in content
    assert "target=1 accepted=True" in content
    assert 'feedback="Good sample captured."' in content


def test_log_includes_no_final_result_section_when_session_did_not_complete() -> None:
    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="CANCELLED",
        events=(),
        result=None,
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "Outcome: CANCELLED" in content
    assert "No final result (session did not reach COMPLETE)" in content
    assert "sample_counts" not in content


def test_log_includes_full_result_when_session_completed() -> None:
    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "Outcome: COMPLETE" in content
    assert "sample_counts: (2, 2, 2, 2, 2, 2, 2, 2, 2)" in content
    assert f"feature_schema_version: {FEATURE_SCHEMA_VERSION}" in content
    assert "started_at_monotonic_ms: 1000.0" in content
    assert "completed_at_monotonic_ms: 1500.0" in content


def test_log_contains_no_frame_or_image_data() -> None:
    """A privacy smoke test: no raw frame/image byte payload -- e.g. a
    ``bytes`` repr, or a base64/hex-looking blob -- should ever end up in a
    persisted calibration log. Scalar eye/iris/head-pose *ratios* are the
    explicit purpose of the dataset section below and are not forbidden here
    -- they are the same class of aggregate diagnostic already shown on the
    M1 debug overlay, not a raw biometric sample or image."""

    events = [CalibrationLogEvent(1, True, "TRACKED", 0.5, "Good sample captured.")]
    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=events,
        result=_result(samples=(_dataset_sample(), _dataset_sample(accepted=False))),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    for forbidden in ("landmark", "face_box", "b'", "base64"):
        assert forbidden not in content.lower()


def test_log_dataset_section_lists_every_sample_with_real_feature_values() -> None:
    accepted = _dataset_sample(accepted=True, target_index=2)
    rejected = _dataset_sample(accepted=False, target_index=2)
    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=(accepted, rejected)),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "2 samples, 1 accepted, 1 rejected" in content
    assert "target=2 accepted=True reason=accepted" in content
    assert "left_iris_in_eye=(0.420, 0.530)" in content
    assert "right_openness=0.650" in content
    assert "head_pose=(yaw=-0.5, pitch=-40.0, roll=8.2)" in content
    assert "target=2 accepted=False reason=tracking_not_ready" in content
    assert "left_iris_in_eye=unavailable" in content
    assert "head_pose=unavailable" in content


def test_log_dataset_section_absent_when_session_has_no_samples() -> None:
    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=()),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "0 samples, 0 accepted, 0 rejected" in content


def test_write_calibration_log_creates_a_readable_file_in_the_given_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "calibration_logs"

    path = write_calibration_log("session content\n", directory=directory)

    assert path.parent == directory
    assert path.name.startswith("calibration_session_")
    assert path.read_text(encoding="utf-8") == "session content\n"


def test_metrics_section_reports_reason_counts_and_head_pose_spread_for_mixed_dataset() -> None:
    """Pins the Metrics section's reason-count and head-pose-spread rendering
    against hand-computed numbers -- not merely that some string came back.

    yaws=[0.0, 10.0, -6.0] -> median 0.0, max deviation |10.0 - 0.0| = 10.0
    pitches=[0.0, -4.0, 8.0] -> median 0.0, max deviation |8.0 - 0.0| = 8.0
    rolls=[0.0, 2.0, -2.0] -> median 0.0, max deviation |2.0 - 0.0| = 2.0
    """

    samples = (
        _pose_sample(accepted=True, reason="accepted", yaw=0.0, pitch=0.0, roll=0.0),
        _pose_sample(accepted=False, reason="low_confidence"),
        _pose_sample(accepted=True, reason="accepted", yaw=10.0, pitch=-4.0, roll=2.0),
        _pose_sample(accepted=False, reason="eye_not_visible"),
        _pose_sample(accepted=True, reason="accepted", yaw=-6.0, pitch=8.0, roll=-2.0),
        _pose_sample(accepted=False, reason="eye_not_visible"),
    )

    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=samples),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    metrics_section = content[content.index("--- Metrics ---") : content.index("--- Dataset")]
    assert "reason_count[accepted]: 3" in metrics_section
    assert "reason_count[eye_not_visible]: 2" in metrics_section
    assert "reason_count[low_confidence]: 1" in metrics_section
    assert "median_yaw_deg: 0.0" in metrics_section
    assert "max_yaw_deviation_deg: 10.0" in metrics_section
    assert "median_pitch_deg: 0.0" in metrics_section
    assert "max_pitch_deviation_deg: 8.0" in metrics_section
    assert "median_roll_deg: 0.0" in metrics_section
    assert "max_roll_deviation_deg: 2.0" in metrics_section


def test_metrics_section_reason_counts_are_sorted_by_name_regardless_of_ingest_order() -> None:
    """The rendered reason-count ordering must be deterministic (sorted by
    reason name), not whatever order samples happened to be ingested in --
    otherwise the same session could render two different logs."""

    ingest_order_a = (
        _pose_sample(accepted=False, reason="tracking_not_ready"),
        _pose_sample(accepted=True, reason="accepted", yaw=1.0, pitch=1.0, roll=1.0),
        _pose_sample(accepted=False, reason="low_confidence"),
    )
    ingest_order_b = (
        _pose_sample(accepted=False, reason="low_confidence"),
        _pose_sample(accepted=False, reason="tracking_not_ready"),
        _pose_sample(accepted=True, reason="accepted", yaw=1.0, pitch=1.0, roll=1.0),
    )

    content_a = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=ingest_order_a),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    content_b = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=ingest_order_b),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    expected_markers = [
        "reason_count[accepted]:",
        "reason_count[low_confidence]:",
        "reason_count[tracking_not_ready]:",
    ]
    for content in (content_a, content_b):
        positions = [content.index(marker) for marker in expected_markers]
        assert positions == sorted(positions)


def test_metrics_section_renders_explicit_placeholders_not_fabricated_zeros_when_empty() -> None:
    """A COMPLETE result with no samples must say "none"/"unavailable" rather
    than silently omitting the section or -- worse -- printing a fabricated
    ``0.0`` that would misreport "the head never moved"."""

    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=()),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert "reason_count: none" in content
    assert "head_pose_spread: unavailable" in content
    assert "median_yaw_deg: 0.0" not in content
    assert "max_yaw_deviation_deg: 0.0" not in content


def test_metrics_section_is_ordered_between_final_result_and_dataset_sections() -> None:
    content = format_calibration_log(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=2,
        outcome="COMPLETE",
        events=(),
        result=_result(samples=(_dataset_sample(),)),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    final_result_index = content.index("--- Final result ---")
    metrics_index = content.index("--- Metrics ---")
    dataset_index = content.index("--- Dataset")
    assert final_result_index < metrics_index < dataset_index


def test_uninformative_result_is_saved_but_not_promoted_as_latest(tmp_path: Path) -> None:
    samples = tuple(_dataset_sample(target_index=index) for index in range(9))
    persisted = train_and_persist_calibration(
        _result(samples=samples), store=CalibrationStore(tmp_path)
    )
    assert persisted.dataset_path.exists()
    assert persisted.pending_model is None
    assert not (tmp_path / "latest_model.json").exists()
    assert not persisted.training.promotable
    assert persisted.training.quality_reasons
    summary = format_training_summary(persisted.training)
    assert "selected=" in summary
    assert "promotable=False" in summary
    assert "linear_x=" in summary
    assert "linear_y=" in summary
    assert "advanced_x=" in summary
    assert "advanced_y=" in summary
    assert "center_median=" in summary
    assert "baseline_median=" in summary
    assert "advanced_p95=" in summary


def test_rejected_new_attempt_preserves_an_existing_active_model(tmp_path: Path) -> None:
    result = _result(samples=tuple(_dataset_sample(target_index=index) for index in range(9)))
    store = CalibrationStore(tmp_path)
    existing = CalibrationEngine().train(result).model
    store.save_model(existing)
    before = (tmp_path / "latest_model.json").read_bytes()

    persisted = train_and_persist_calibration(result, store=store)

    assert persisted.pending_model is None
    assert (tmp_path / "latest_model.json").read_bytes() == before
