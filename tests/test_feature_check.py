"""Deterministic tests for the five-direction pre-calibration diagnostic."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from gazelink.domain import (
    EyeFeatures,
    HeadPose,
    NormalizedPoint,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.feature_check import (
    DirectionVerdict,
    FeatureCheckController,
    FeatureCheckDirection,
    FeatureCheckPhase,
    FeatureCheckSample,
    FeatureCheckSettings,
    VerticalSignalSource,
    analyze_feature_check,
)
from gazelink.feature_check_window import format_feature_check_report, write_feature_check_report
from gazelink.gaze_features import GazeFeatureVector


class FakeClock:
    def __init__(self) -> None:
        self.now_ms = 1_000.0

    def __call__(self) -> float:
        return self.now_ms


def _observation(frame_id: int, at_ms: float) -> VisionObservation:
    return VisionObservation(
        frame_id=frame_id,
        observed_at_monotonic_ms=at_ms,
        tracking_state=TrackingState.TRACKED,
        face_box=None,
        left_eye=EyeFeatures(None, 0.5, 0.5, NormalizedPoint(0.5, 0.5)),
        right_eye=EyeFeatures(None, 0.5, 0.5, NormalizedPoint(0.5, 0.5)),
        head_pose=HeadPose(0.0, 0.0, 0.0),
        overall_confidence=0.5,
    )


def _diagnostic_samples(*, invert_right: bool = False) -> tuple[FeatureCheckSample, ...]:
    positions = {
        FeatureCheckDirection.CENTER: (0.50, 0.50),
        FeatureCheckDirection.LEFT: (0.40, 0.50),
        FeatureCheckDirection.RIGHT: (0.40 if invert_right else 0.60, 0.50),
        FeatureCheckDirection.UP: (0.50, 0.40),
        FeatureCheckDirection.DOWN: (0.50, 0.60),
    }
    samples: list[FeatureCheckSample] = []
    frame_id = 0
    for direction, (horizontal, vertical) in positions.items():
        for index in range(10):
            noise = -0.002 if index % 2 == 0 else 0.002
            samples.append(
                FeatureCheckSample(
                    direction=direction,
                    frame_id=frame_id,
                    observed_at_monotonic_ms=1_000.0 + frame_id * 80.0,
                    features=GazeFeatureVector(
                        (
                            horizontal + noise,
                            vertical - noise,
                            horizontal - noise,
                            vertical + noise,
                            0.5,
                            0.5,
                            0.0,
                            0.0,
                            0.0,
                        )
                    ),
                )
            )
            frame_id += 1
    return tuple(samples)


def test_controller_waits_collects_a_window_and_resets_after_tracking_loss() -> None:
    clock = FakeClock()
    controller = FeatureCheckController(
        clock_ms=clock,
        settings=FeatureCheckSettings(
            stabilization_ms=100.0,
            collection_ms=200.0,
            min_sample_interval_ms=20.0,
            min_samples=3,
        ),
    )
    assert controller.ingest(_observation(1, clock.now_ms)).phase is FeatureCheckPhase.STABILIZING
    clock.now_ms = 1_100.0
    controller.ingest(_observation(2, clock.now_ms))
    clock.now_ms = 1_120.0
    controller.ingest(_observation(3, clock.now_ms))
    assert controller.view().accepted_samples == 2

    clock.now_ms = 1_130.0
    lost = controller.tracking_lost()
    assert lost.phase is FeatureCheckPhase.STABILIZING
    assert lost.accepted_samples == 0

    clock.now_ms = 1_230.0
    controller.ingest(_observation(4, clock.now_ms))
    clock.now_ms = 1_250.0
    controller.ingest(_observation(5, clock.now_ms))
    clock.now_ms = 1_270.0
    controller.ingest(_observation(6, clock.now_ms))
    clock.now_ms = 1_430.0
    advanced = controller.ingest(_observation(7, clock.now_ms))
    assert advanced.target_number == 2
    assert advanced.target is not None
    assert advanced.target.direction is FeatureCheckDirection.LEFT
    assert advanced.phase is FeatureCheckPhase.STABILIZING


def test_analysis_passes_clear_stable_motion_in_all_four_directions() -> None:
    result = analyze_feature_check(_diagnostic_samples())
    assert result.overall_pass
    assert len(result.stable_directions) == 5
    assert all(check.verdict is DirectionVerdict.PASS for check in result.direction_checks)


def test_analysis_identifies_an_inverted_right_direction() -> None:
    result = analyze_feature_check(_diagnostic_samples(invert_right=True))
    right = next(
        check for check in result.direction_checks if check.direction is FeatureCheckDirection.RIGHT
    )
    assert right.verdict is DirectionVerdict.INVERTED
    assert not result.overall_pass


def test_axis_specific_threshold_accepts_measured_vertical_signal_only() -> None:
    samples = list(_diagnostic_samples())
    axis_values = {
        FeatureCheckDirection.LEFT: (0, 2, 0.475),
        FeatureCheckDirection.RIGHT: (0, 2, 0.525),
        FeatureCheckDirection.UP: (1, 3, 0.475),
        FeatureCheckDirection.DOWN: (1, 3, 0.525),
    }
    for index, sample in enumerate(samples):
        if sample.direction is FeatureCheckDirection.CENTER:
            continue
        first_axis, second_axis, value = axis_values[sample.direction]
        values = list(sample.features.values)
        noise = -0.002 if index % 2 == 0 else 0.002
        values[first_axis] = value + noise
        values[second_axis] = value - noise
        samples[index] = replace(sample, features=GazeFeatureVector(tuple(values)))

    result = analyze_feature_check(tuple(samples))
    by_direction = {check.direction: check.verdict for check in result.direction_checks}

    assert by_direction[FeatureCheckDirection.LEFT] is DirectionVerdict.NO_SEPARATION
    assert by_direction[FeatureCheckDirection.RIGHT] is DirectionVerdict.NO_SEPARATION
    assert by_direction[FeatureCheckDirection.UP] is DirectionVerdict.PASS
    assert by_direction[FeatureCheckDirection.DOWN] is DirectionVerdict.PASS


def test_vertical_gate_accepts_weaker_eye_only_when_binocular_signal_is_sufficient() -> None:
    samples = list(_diagnostic_samples())
    for index, sample in enumerate(samples):
        if sample.direction is not FeatureCheckDirection.UP:
            continue
        values = list(sample.features.values)
        noise = -0.001 if index % 2 == 0 else 0.001
        values[1] = 0.487 + noise
        values[3] = 0.4794 - noise
        samples[index] = replace(sample, features=GazeFeatureVector(tuple(values)))

    result = analyze_feature_check(tuple(samples))
    up = next(
        check for check in result.direction_checks if check.direction is FeatureCheckDirection.UP
    )

    assert abs(up.left_eye_delta + 0.013) < 1e-12
    assert abs(up.right_eye_delta + 0.0206) < 1e-12
    assert up.verdict is DirectionVerdict.PASS


def test_vertical_gate_rejects_weak_combined_signal_even_when_eyes_agree() -> None:
    samples = list(_diagnostic_samples())
    for index, sample in enumerate(samples):
        if sample.direction is not FeatureCheckDirection.DOWN:
            continue
        values = list(sample.features.values)
        noise = -0.001 if index % 2 == 0 else 0.001
        values[1] = 0.509 + noise
        values[3] = 0.512 - noise
        samples[index] = replace(sample, features=GazeFeatureVector(tuple(values)))

    result = analyze_feature_check(tuple(samples))
    down = next(
        check for check in result.direction_checks if check.direction is FeatureCheckDirection.DOWN
    )

    assert down.verdict is DirectionVerdict.NO_SEPARATION


def test_vertical_check_uses_lid_relative_signal_when_corner_axis_has_no_separation() -> None:
    samples = list(_diagnostic_samples())
    for index, sample in enumerate(samples):
        if sample.direction is not FeatureCheckDirection.UP:
            continue
        values = list(sample.features.values)
        noise = -0.001 if index % 2 == 0 else 0.001
        values[1] = 0.5 + noise
        values[3] = 0.5 - noise
        values[4] = 0.46 + noise
        values[5] = 0.46 - noise
        samples[index] = replace(sample, features=GazeFeatureVector(tuple(values)))

    result = analyze_feature_check(tuple(samples))
    up = next(
        check for check in result.direction_checks if check.direction is FeatureCheckDirection.UP
    )

    assert up.verdict is DirectionVerdict.PASS
    assert up.vertical_signal_source is VerticalSignalSource.LID_RELATIVE


def test_controller_retries_same_direction_when_no_stable_window_exists() -> None:
    clock = FakeClock()
    controller = FeatureCheckController(
        clock_ms=clock,
        settings=FeatureCheckSettings(
            stabilization_ms=0.0,
            collection_ms=0.0,
            min_sample_interval_ms=0.0,
            min_samples=3,
        ),
    )
    for frame_id, value in enumerate((0.10, 0.50, 0.90), start=1):
        observation = _observation(frame_id, clock.now_ms)
        eye = EyeFeatures(None, 0.5, 0.5, NormalizedPoint(value, value))
        view = controller.ingest(replace(observation, left_eye=eye, right_eye=eye))

    assert view.phase is FeatureCheckPhase.STABILIZING
    assert view.target_number == 1
    assert view.accepted_samples == 0
    assert "No stable contiguous window" in view.feedback


def test_opposite_eye_noise_cannot_cancel_into_a_false_stable_result() -> None:
    samples = list(_diagnostic_samples())
    for index, sample in enumerate(samples):
        if sample.direction is not FeatureCheckDirection.CENTER:
            continue
        values = list(sample.features.values)
        shift = 0.10 if index % 2 == 0 else -0.10
        values[0] += shift
        values[2] -= shift
        samples[index] = replace(sample, features=GazeFeatureVector(tuple(values)))

    result = analyze_feature_check(tuple(samples))

    assert FeatureCheckDirection.CENTER not in result.stable_directions
    assert not result.overall_pass


def test_report_persists_only_scalar_diagnostics_and_human_summary(tmp_path: Path) -> None:
    result = analyze_feature_check(_diagnostic_samples())
    geometry = ScreenGeometry("primary", 1920, 1080, 1.0)
    generated = datetime(2026, 9, 2, tzinfo=UTC)
    artifacts = write_feature_check_report(
        result,
        camera_id="camera-0",
        screen_geometry=geometry,
        directory=tmp_path,
        generated_at=generated,
    )
    assert artifacts.json_path.exists() and artifacts.text_path.exists()
    payload = artifacts.json_path.read_text(encoding="utf-8")
    assert '"feature_names"' in payload
    assert '"min_horizontal_delta": 0.03' in payload
    assert '"min_vertical_combined_delta": 0.015' in payload
    assert '"min_vertical_per_eye_delta": 0.01' in payload
    assert "frame_bytes" not in payload
    summary = format_feature_check_report(
        result,
        camera_id="camera-0",
        screen_geometry=geometry,
        generated_at=generated,
    )
    assert "Overall diagnostic pass: True" in summary
    assert "horizontal_delta=0.03000 vertical_combined_delta=0.01500" in summary
    assert "vertical_per_eye_delta=0.01000" in summary
    assert "RIGHT: verdict=PASS" in summary
    assert "head_pose_p95_deg=" in summary
