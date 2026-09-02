"""A bounded, caller-driven runtime pipeline with latest-frame backpressure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Generic, TypeVar

from gazelink.camera import CameraSource
from gazelink.domain import FramePacket
from gazelink.metrics import InMemoryMetrics, MetricsCollector

ResultT = TypeVar("ResultT")


def _monotonic_ms() -> float:
    return perf_counter_ns() / 1_000_000


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Non-biometric pipeline health state, suitable for diagnostics."""

    is_running: bool
    is_closed: bool
    pending_frame_id: int | None
    captured_frames: int
    processed_frames: int
    dropped_frames: int
    stale_results: int


class LatestFramePipeline(Generic[ResultT]):
    """Keep only the newest frame while a caller schedules capture and inference.

    The pipeline intentionally owns no thread, task, or listener. A UI/runtime
    scheduler calls ``capture_once`` and ``process_latest``; this avoids hidden
    workers and makes shutdown deterministic. The one-slot pending buffer is
    protected by a lock so capture and inference may be scheduled separately.
    """

    def __init__(
        self,
        source: CameraSource,
        process_frame: Callable[[FramePacket], ResultT],
        *,
        metrics: MetricsCollector | None = None,
        clock_ms: Callable[[], float] = _monotonic_ms,
    ) -> None:
        self._source = source
        self._process_frame = process_frame
        self._metrics = metrics or InMemoryMetrics()
        self._clock_ms = clock_ms
        self._lock = Lock()
        self._latest: FramePacket | None = None
        self._last_submitted_frame_id: int | None = None
        self._is_running = False
        self._is_closed = False
        self._captured_frames = 0
        self._processed_frames = 0
        self._dropped_frames = 0
        self._stale_results = 0
        self._last_capture_at_ms: float | None = None

    @property
    def metrics(self) -> MetricsCollector:
        """Expose aggregate-only metrics; frames are never included."""

        return self._metrics

    def start(self) -> None:
        """Open capture and enable caller-driven work exactly once."""

        with self._lock:
            if self._is_closed:
                raise RuntimeError("pipeline is closed")
            if self._is_running:
                return
        self._source.open()
        with self._lock:
            self._is_running = True

    def capture_once(self) -> FramePacket:
        """Read one frame and replace any not-yet-processed older frame."""

        self._require_running()
        try:
            packet = self._source.read()
        except Exception:
            self.close()
            raise
        self.submit(packet)
        return packet

    def submit(self, packet: FramePacket) -> None:
        """Submit a captured packet, retaining only the newest pending packet."""

        with self._lock:
            if not self._is_running or self._is_closed:
                raise RuntimeError("pipeline is not running")
            previous_id = self._last_submitted_frame_id
            if previous_id is not None and packet.frame_id <= previous_id:
                raise ValueError("frame IDs must strictly increase within a pipeline session")
            if self._latest is not None:
                self._dropped_frames += 1
                self._metrics.counter("dropped_frames_total", labels={"stage": "capture"})
            self._latest = packet
            self._last_submitted_frame_id = packet.frame_id
            self._captured_frames += 1
            self._metrics.counter("captured_frames_total", labels={"stage": "capture"})
            if self._last_capture_at_ms is not None:
                elapsed_ms = packet.captured_at_monotonic_ms - self._last_capture_at_ms
                if elapsed_ms > 0:
                    self._metrics.record_fps(1_000 / elapsed_ms, component="capture")
            self._last_capture_at_ms = packet.captured_at_monotonic_ms

    def process_latest(self) -> ResultT | None:
        """Process the newest available packet and suppress a stale completion."""

        self._require_running()
        with self._lock:
            packet, self._latest = self._latest, None
        if packet is None:
            return None
        started_at_ms = self._clock_ms()
        try:
            result = self._process_frame(packet)
        except Exception:
            self.close()
            raise
        completed_at_ms = self._clock_ms()
        latency_ms = max(0.0, completed_at_ms - packet.captured_at_monotonic_ms)
        processing_ms = max(0.0, completed_at_ms - started_at_ms)
        with self._lock:
            newer_packet_waiting = (
                self._latest is not None and self._latest.frame_id > packet.frame_id
            )
            if self._is_closed or newer_packet_waiting:
                self._stale_results += 1
                self._metrics.counter("stale_results_total", labels={"stage": "vision"})
                return None
            self._processed_frames += 1
        self._metrics.record_latency_ms(latency_ms, component="pipeline")
        self._metrics.record_latency_ms(processing_ms, component="vision")
        return result

    def close(self) -> None:
        """Discard pending data and release the camera; repeated calls are harmless."""

        with self._lock:
            if self._is_closed:
                return
            self._is_running = False
            self._is_closed = True
            self._latest = None
        self._source.close()

    def snapshot(self) -> PipelineSnapshot:
        """Return bounded-buffer counters without exposing frame contents."""

        with self._lock:
            return PipelineSnapshot(
                is_running=self._is_running,
                is_closed=self._is_closed,
                pending_frame_id=None if self._latest is None else self._latest.frame_id,
                captured_frames=self._captured_frames,
                processed_frames=self._processed_frames,
                dropped_frames=self._dropped_frames,
                stale_results=self._stale_results,
            )

    def _require_running(self) -> None:
        with self._lock:
            if not self._is_running or self._is_closed:
                raise RuntimeError("pipeline is not running")
