"""Deterministic M2-04 smoothing and jitter-metric tests."""

from gazelink.domain import GazePoint, ScreenGeometry
from gazelink.gaze_filter import (
    GazeFilterKind,
    GazeStabilityFilter,
    measure_stationary_jitter,
)


def test_each_candidate_reduces_stationary_jitter_on_same_sequence() -> None:
    geometry = ScreenGeometry("primary", 1920, 1080, 1.0)
    source = [
        GazePoint(0.5 + (0.02 if index % 2 else -0.02), 0.5 + (0.01 if index % 3 else -0.01))
        for index in range(60)
    ]
    raw = measure_stationary_jitter(source, geometry)
    assert raw is not None

    for kind in GazeFilterKind:
        gaze_filter = GazeStabilityFilter(kind)
        filtered = [gaze_filter.update(point, index * 33.0) for index, point in enumerate(source)]
        metrics = measure_stationary_jitter(filtered[10:], geometry)
        assert metrics is not None
        assert metrics.p95_frame_jump_px < raw.p95_frame_jump_px


def test_reset_does_not_replay_stale_position_after_tracking_loss() -> None:
    gaze_filter = GazeStabilityFilter(GazeFilterKind.EMA)
    gaze_filter.update(GazePoint(0.1, 0.1), 100.0)
    gaze_filter.update(GazePoint(0.2, 0.2), 133.0)
    gaze_filter.reset()
    recovered = gaze_filter.update(GazePoint(0.8, 0.8), 200.0)
    assert recovered == GazePoint(0.8, 0.8)
