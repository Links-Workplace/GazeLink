"""Scenario tests for the joined M1 runtime, using fakes and a fake clock."""

from __future__ import annotations

import pytest

from gazelink.camera import CameraError, FixtureCameraSource
from gazelink.confidence import (
    ConfidencePolicySettings,
    TrackingRecoverySettings,
)
from gazelink.config import ConfidenceThresholds
from gazelink.domain import (
    EyeFeatures,
    FramePacket,
    HeadPose,
    NormalizedBox,
    NormalizedPoint,
    PixelFormat,
    ReasonCode,
    TrackingState,
    VisionObservation,
)
from gazelink.metrics import InMemoryMetrics
from gazelink.overlay import CircleCommand, OverlayStatus, RectangleCommand, TextCommand
from gazelink.runtime import RuntimeTick, VisionRuntime

FRAME_INTERVAL_MS = 20.0


class FakeClock:
    """A monotonic millisecond clock the test advances explicitly."""

    def __init__(self, start_ms: float = 1_000.0) -> None:
        self.now_ms = start_ms

    def __call__(self) -> float:
        return self.now_ms

    def advance(self, milliseconds: float) -> float:
        self.now_ms += milliseconds
        return self.now_ms


class ScriptedEngine:
    """A VisionEngine that replays one prepared observation per frame."""

    def __init__(self, states: list[TrackingState]) -> None:
        self._states = list(states)
        self.closed = 0
        self.observed_frame_ids: list[int] = []

    def observe(self, frame: FramePacket) -> VisionObservation:
        self.observed_frame_ids.append(frame.frame_id)
        state = self._states.pop(0) if self._states else TrackingState.TRACKED
        return _observation(frame, state)

    def close(self) -> None:
        self.closed += 1


def _observation(frame: FramePacket, state: TrackingState) -> VisionObservation:
    tracked = state is TrackingState.TRACKED
    return VisionObservation(
        frame_id=frame.frame_id,
        observed_at_monotonic_ms=frame.captured_at_monotonic_ms,
        tracking_state=state,
        face_box=NormalizedBox(0.2, 0.2, 0.4, 0.4) if tracked else None,
        left_eye=EyeFeatures(NormalizedPoint(0.35, 0.4), 0.5, 0.5, NormalizedPoint(0.5, 0.5))
        if tracked
        else None,
        right_eye=EyeFeatures(NormalizedPoint(0.55, 0.4), 0.5, 0.5, NormalizedPoint(0.5, 0.5))
        if tracked
        else None,
        head_pose=HeadPose(1.0, -1.0, 0.5) if tracked else None,
        overall_confidence=0.5 if tracked else 0.0,
        reason_codes=() if tracked else (ReasonCode.FACE_NOT_FOUND,),
    )


def _packets(count: int, *, start_ms: float = 1_000.0) -> list[FramePacket]:
    return [
        FramePacket(
            frame_id=index,
            captured_at_monotonic_ms=start_ms + (index * FRAME_INTERVAL_MS),
            width=64,
            height=48,
            pixel_format=PixelFormat.BGR24,
            image=b"\x00" * (64 * 48 * 3),
        )
        for index in range(count)
    ]


def _settings(
    *, lost_debounce_ms: float = 150.0, recovery_stable_ms: float = 100.0
) -> TrackingRecoverySettings:
    return TrackingRecoverySettings(
        confidence=ConfidencePolicySettings(thresholds=ConfidenceThresholds()),
        lost_debounce_ms=lost_debounce_ms,
        recovery_stable_ms=recovery_stable_ms,
    )


def _build(
    states: list[TrackingState],
    *,
    frames: int | None = None,
    clock: FakeClock | None = None,
    settings: TrackingRecoverySettings | None = None,
) -> tuple[VisionRuntime, FixtureCameraSource, ScriptedEngine, FakeClock, InMemoryMetrics]:
    clock = clock or FakeClock()
    source = FixtureCameraSource(_packets(frames if frames is not None else len(states)))
    engine = ScriptedEngine(states)
    metrics = InMemoryMetrics()
    runtime = VisionRuntime(
        source=source,
        engine=engine,
        recovery_settings=settings or _settings(),
        metrics=metrics,
        clock_ms=clock,
    )
    return runtime, source, engine, clock, metrics


def _tick_at(runtime: VisionRuntime, clock: FakeClock, frame_index: int) -> RuntimeTick:
    """Advance the clock to just after the frame's capture time, then tick."""

    clock.now_ms = 1_000.0 + (frame_index * FRAME_INTERVAL_MS) + 1.0
    tick = runtime.tick()
    assert tick is not None
    return tick


def _shapes(tick: RuntimeTick) -> list[object]:
    return [
        command
        for command in tick.view.commands
        if isinstance(command, (RectangleCommand, CircleCommand))
    ]


def _texts(tick: RuntimeTick) -> list[str]:
    return [command.text for command in tick.view.commands if isinstance(command, TextCommand)]


def test_happy_path_draws_current_frame_landmarks_and_reports_fps_and_latency() -> None:
    runtime, _source, engine, clock, _metrics = _build([TrackingState.TRACKED] * 12)
    runtime.start()
    try:
        for index in range(12):
            tick = _tick_at(runtime, clock, index)
    finally:
        runtime.close()

    assert tick.tracking_state is TrackingState.TRACKED
    assert tick.view.status is OverlayStatus.TRACKED
    assert tick.view.frame_id == tick.frame.frame_id == 11
    assert tick.accepted_observation is not None
    assert tick.accepted_observation.frame_id == 11
    assert engine.observed_frame_ids == list(range(12))
    assert _shapes(tick), "an accepted observation must draw its landmarks"
    assert tick.latency_ms == pytest.approx(1.0)
    assert tick.fps == pytest.approx(1_000.0 / FRAME_INTERVAL_MS)


def test_no_face_reports_lost_and_draws_no_geometry() -> None:
    runtime, _source, _engine, clock, _metrics = _build([TrackingState.LOST] * 3)
    runtime.start()
    try:
        ticks = [_tick_at(runtime, clock, index) for index in range(3)]
    finally:
        runtime.close()

    assert all(tick.tracking_state is TrackingState.LOST for tick in ticks)
    assert all(tick.view.status is OverlayStatus.LOST for tick in ticks)
    assert not any(_shapes(tick) for tick in ticks)
    assert any("FACE_NOT_FOUND" in text for text in _texts(ticks[-1]))


def test_a_rejected_frame_never_reuses_the_previous_frames_landmarks() -> None:
    """The regression this pins: a lost frame must not repaint the last good face."""

    runtime, _source, _engine, clock, _metrics = _build(
        [TrackingState.TRACKED, TrackingState.TRACKED, TrackingState.LOST, TrackingState.LOST],
        settings=_settings(lost_debounce_ms=150.0, recovery_stable_ms=0.0),
    )
    runtime.start()
    try:
        good = _tick_at(runtime, clock, 0)
        _tick_at(runtime, clock, 1)
        debouncing = runtime.tick()  # first rejected frame opens the debounce
        clock.now_ms += 200.0  # and this one outlasts it
        lost = runtime.tick()
    finally:
        runtime.close()

    assert good.view.frame_id == 0
    assert _shapes(good), "an accepted frame draws its own landmarks"

    # While debouncing, the policy deliberately still reports TRACKED so a
    # single dropped frame does not flap the UI -- but the geometry of the last
    # good frame must not be repainted to fill the gap.
    assert debouncing is not None
    assert debouncing.view.frame_id == 2
    assert debouncing.tracking_state is TrackingState.TRACKED
    assert debouncing.accepted_observation is None
    assert not _shapes(debouncing)

    # Once the debounce elapses the loss is reported for real.
    assert lost is not None
    assert lost.view.frame_id == 3
    assert lost.tracking_state is TrackingState.LOST
    assert lost.accepted_observation is None
    assert not _shapes(lost)


def test_recovery_withholds_landmarks_until_the_stability_window_completes() -> None:
    clock = FakeClock()
    runtime, _source, _engine, _clock, _metrics = _build(
        [TrackingState.LOST] + [TrackingState.TRACKED] * 5,
        clock=clock,
        settings=_settings(recovery_stable_ms=60.0),
    )
    runtime.start()
    try:
        first = _tick_at(runtime, clock, 0)
        during = [_tick_at(runtime, clock, index) for index in (1, 2, 3)]
        after = _tick_at(runtime, clock, 4)
    finally:
        runtime.close()

    assert first.tracking_state is TrackingState.LOST
    # The stability window opens at the first fresh frame and lasts 60ms, while
    # frames arrive every 20ms.  Every frame inside that window is provisional
    # and must not draw a single landmark, however good it looks on its own.
    assert [tick.tracking_state for tick in during] == [TrackingState.LOW_CONFIDENCE] * 3
    assert not any(_shapes(tick) for tick in during)
    # Only once the window has fully elapsed may geometry appear again.
    assert after.tracking_state is TrackingState.TRACKED
    assert _shapes(after)


def test_multiple_faces_is_surfaced_without_picking_one_of_them() -> None:
    runtime, _source, _engine, clock, _metrics = _build([TrackingState.MULTIPLE_FACES])
    runtime.start()
    try:
        tick = _tick_at(runtime, clock, 0)
    finally:
        runtime.close()

    assert tick.tracking_state is TrackingState.LOST
    assert not _shapes(tick)


def test_stale_observation_is_rejected_when_the_clock_runs_far_ahead() -> None:
    runtime, _source, _engine, clock, _metrics = _build([TrackingState.TRACKED] * 2)
    runtime.start()
    try:
        _tick_at(runtime, clock, 0)
        clock.now_ms = 1_000.0 + 5_000.0  # far beyond max_sample_age_ms
        stale = runtime.tick()
    finally:
        runtime.close()

    assert stale is not None
    assert stale.tracking_state is TrackingState.LOST
    assert not _shapes(stale)
    assert any("STALE_SAMPLE" in text for text in _texts(stale))


def test_camera_loss_releases_the_camera_and_the_model_and_reports_recovery() -> None:
    runtime, source, engine, clock, _metrics = _build([TrackingState.TRACKED], frames=1)
    runtime.start()
    try:
        _tick_at(runtime, clock, 0)
        with pytest.raises(CameraError):
            runtime.tick()
    finally:
        runtime.close()

    assert source.close_count == 1
    assert engine.closed == 1
    assert runtime.last_error is not None
    assert runtime.last_error.recovery_action.value
    assert "camera" in runtime.last_error.code.value


def test_close_is_idempotent_and_releases_both_resources_once() -> None:
    runtime, source, engine, _clock, _metrics = _build([TrackingState.TRACKED], frames=1)
    runtime.start()

    runtime.close()
    runtime.close()

    assert source.close_count == 1
    assert engine.closed == 1


def test_shutdown_during_an_active_interaction_stops_further_work() -> None:
    runtime, _source, _engine, clock, _metrics = _build([TrackingState.TRACKED] * 3)
    runtime.start()
    _tick_at(runtime, clock, 0)

    runtime.close()

    with pytest.raises(RuntimeError, match="not running"):
        runtime.tick()


def test_runtime_records_loss_and_recovery_metrics_without_biometric_labels() -> None:
    clock = FakeClock()
    runtime, _source, _engine, _clock, metrics = _build(
        [TrackingState.LOST] + [TrackingState.TRACKED] * 5,
        clock=clock,
        settings=_settings(recovery_stable_ms=60.0),
    )
    runtime.start()
    try:
        for index in range(6):
            _tick_at(runtime, clock, index)
    finally:
        runtime.close()

    names = [sample.name for sample in metrics.snapshot()]
    assert "tracking_loss_total" in names
    assert "tracking_recovery_ms" in names
    assert "tracking_confidence" in names
    for sample in metrics.snapshot():
        for _key, value in sample.labels:
            assert value in {
                "capture",
                "vision",
                "runtime",
                "pipeline",
                TrackingState.LOST.value,
            }


def test_runtime_never_exposes_a_frame_buffer_in_its_view_models() -> None:
    runtime, _source, _engine, clock, _metrics = _build([TrackingState.TRACKED])
    runtime.start()
    try:
        tick = _tick_at(runtime, clock, 0)
    finally:
        runtime.close()

    assert not any(isinstance(text, bytes) for text in _texts(tick))
    assert tick.frame.image is not None
    assert not any(tick.frame.image.hex()[:16] in text for text in _texts(tick))
