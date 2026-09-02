"""The M1 runtime that joins capture, vision, confidence, and overlay.

This module owns the order in which the M1 stages run, but it deliberately owns
no window, thread, or timer.  A caller drives it one ``tick`` at a time, so the
same runtime can be exercised by a Qt timer, a benchmark loop, or a
deterministic test with a fake camera and a fake clock.

The safety rule this module enforces is narrow and explicit: landmarks reach the
overlay only for an observation that the tracking policy actually accepted.
When the policy is debouncing a loss or waiting out a recovery window, the
runtime blanks the geometry rather than drawing a position that nothing has
authorized.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter_ns

from gazelink.camera import CameraError, CameraSource
from gazelink.confidence import TrackingRecoveryPolicy, TrackingRecoverySettings
from gazelink.domain import CameraStatus, FramePacket, TrackingState, VisionObservation
from gazelink.errors import UserFacingError
from gazelink.metrics import InMemoryMetrics, MetricsCollector
from gazelink.overlay import DebugOverlayRenderer, OverlayViewModel
from gazelink.pipeline import LatestFramePipeline
from gazelink.vision import VisionEngine

DEFAULT_FPS_WINDOW = 30


def _monotonic_ms() -> float:
    return perf_counter_ns() / 1_000_000


@dataclass(frozen=True, slots=True)
class RuntimeTick:
    """One frame's worth of output; nothing here survives into the next tick."""

    frame: FramePacket
    observation: VisionObservation
    accepted_observation: VisionObservation | None
    view: OverlayViewModel
    tracking_state: TrackingState
    fps: float | None
    latency_ms: float | None


class VisionRuntime:
    """Drive one frame at a time through the M1 stages, in dependency order."""

    def __init__(
        self,
        *,
        source: CameraSource,
        engine: VisionEngine,
        recovery_settings: TrackingRecoverySettings,
        renderer: DebugOverlayRenderer | None = None,
        metrics: MetricsCollector | None = None,
        clock_ms: Callable[[], float] = _monotonic_ms,
        fps_window: int = DEFAULT_FPS_WINDOW,
    ) -> None:
        if fps_window < 2:
            raise ValueError("fps_window must include at least two frames")
        self._source = source
        self._engine = engine
        self._renderer = renderer or DebugOverlayRenderer()
        self._metrics = metrics or InMemoryMetrics()
        self._clock_ms = clock_ms
        self._policy = TrackingRecoveryPolicy(recovery_settings, clock=clock_ms)
        self._pipeline: LatestFramePipeline[VisionObservation] = LatestFramePipeline(
            source, engine.observe, metrics=self._metrics, clock_ms=clock_ms
        )
        self._capture_times_ms: deque[float] = deque(maxlen=fps_window)
        self._last_error: UserFacingError | None = None
        self._lost_since_ms: float | None = None
        self._closed = False

    @property
    def metrics(self) -> MetricsCollector:
        """Aggregate-only operational metrics; frames are never recorded."""

        return self._metrics

    @property
    def camera_status(self) -> CameraStatus:
        return self._source.status

    @property
    def last_error(self) -> UserFacingError | None:
        """The most recent display-ready failure, or ``None`` while healthy."""

        return self._last_error

    def start(self) -> None:
        """Open the camera; a failure leaves nothing acquired."""

        try:
            self._pipeline.start()
        except CameraError as error:
            self._last_error = error.failure
            raise

    def tick(self) -> RuntimeTick | None:
        """Capture and process exactly one frame.

        Returns ``None`` when the pipeline dropped the result as stale, which
        keeps a late observation from being presented as the current one.
        Raises :class:`~gazelink.camera.CameraError` after releasing the camera.
        """

        try:
            packet = self._pipeline.capture_once()
            observation = self._pipeline.process_latest()
        except CameraError as error:
            self._last_error = error.failure
            self.close()
            raise
        if observation is None:
            return None

        now_ms = self._clock_ms()
        self._capture_times_ms.append(packet.captured_at_monotonic_ms)
        update = self._policy.update(observation, now_monotonic_ms=now_ms)
        self._record_tracking_metrics(update.tracking_state, update.recovered, now_ms)

        latency_ms = max(0.0, now_ms - packet.captured_at_monotonic_ms)
        fps = self._current_fps()
        self._metrics.record_confidence(observation.overall_confidence, component="vision")
        if fps is not None:
            self._metrics.record_fps(fps, component="runtime")

        view = self._renderer.render(
            packet,
            _display_observation(observation, update.accepted_observation, update.tracking_state),
            fps=fps,
            latency_ms=latency_ms,
            extra_reason_codes=tuple(
                str(reason.value) for reason in update.confidence.reason_codes
            ),
        )
        return RuntimeTick(
            frame=packet,
            observation=_display_observation(
                observation, update.accepted_observation, update.tracking_state
            ),
            accepted_observation=update.accepted_observation,
            view=view,
            tracking_state=update.tracking_state,
            fps=fps,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        """Release the camera and the model exactly once, on every exit path."""

        if self._closed:
            return
        self._closed = True
        try:
            self._pipeline.close()
        finally:
            self._engine.close()
            self._capture_times_ms.clear()
            self._policy.reset()

    def _current_fps(self) -> float | None:
        if len(self._capture_times_ms) < 2:
            return None
        span_ms = self._capture_times_ms[-1] - self._capture_times_ms[0]
        if span_ms <= 0:
            return None
        return (len(self._capture_times_ms) - 1) * 1_000.0 / span_ms

    def _record_tracking_metrics(
        self, tracking_state: TrackingState, recovered: bool, now_ms: float
    ) -> None:
        if tracking_state is TrackingState.LOST:
            if self._lost_since_ms is None:
                self._lost_since_ms = now_ms
                self._metrics.record_tracking_loss(reason_code=TrackingState.LOST.value)
            return
        if recovered and self._lost_since_ms is not None:
            self._metrics.record_tracking_recovery(
                max(0.0, now_ms - self._lost_since_ms), reason_code=TrackingState.LOST.value
            )
        if tracking_state is TrackingState.TRACKED:
            self._lost_since_ms = None


def _display_observation(
    observation: VisionObservation,
    accepted: VisionObservation | None,
    tracking_state: TrackingState,
) -> VisionObservation:
    """Return what the overlay may draw for this frame.

    An accepted observation is shown as-is.  Anything else keeps the current
    frame's identity and reasons but loses its geometry, so a rejected frame can
    never put a face box or an iris on screen.
    """

    if accepted is not None:
        return accepted
    return replace(
        observation,
        tracking_state=tracking_state,
        face_box=None,
        left_eye=None,
        right_eye=None,
        head_pose=None,
    )
