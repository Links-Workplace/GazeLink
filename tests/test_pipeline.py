"""Backpressure and shutdown tests using only in-memory synthetic frames."""

from __future__ import annotations

import pytest

from gazelink.camera import CameraError, FixtureCameraSource
from gazelink.domain import FramePacket, PixelFormat
from gazelink.metrics import InMemoryMetrics
from gazelink.pipeline import LatestFramePipeline


def _packet(frame_id: int, captured_at_ms: float | None = None) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        captured_at_monotonic_ms=float(frame_id) if captured_at_ms is None else captured_at_ms,
        width=1,
        height=1,
        pixel_format=PixelFormat.GRAY8,
        image=b"x",
    )


def test_pipeline_retains_latest_frame_and_bounds_pending_memory() -> None:
    source = FixtureCameraSource()
    metrics = InMemoryMetrics()
    pipeline = LatestFramePipeline(source, lambda packet: packet.frame_id, metrics=metrics)
    pipeline.start()

    for frame_id in range(1_000):
        pipeline.submit(_packet(frame_id))

    assert pipeline.snapshot().pending_frame_id == 999
    assert pipeline.process_latest() == 999
    snapshot = pipeline.snapshot()
    assert snapshot.captured_frames == 1_000
    assert snapshot.dropped_frames == 999
    assert snapshot.processed_frames == 1
    assert snapshot.pending_frame_id is None
    assert sum(sample.name == "dropped_frames_total" for sample in metrics.snapshot()) == 999


def test_pipeline_drops_completed_result_when_newer_frame_arrives_during_inference() -> None:
    source = FixtureCameraSource()
    pipeline: LatestFramePipeline[int]

    def process(packet: FramePacket) -> int:
        if packet.frame_id == 0:
            pipeline.submit(_packet(1, captured_at_ms=2.0))
        return packet.frame_id

    pipeline = LatestFramePipeline(source, process, clock_ms=lambda: 5.0)
    pipeline.start()
    pipeline.submit(_packet(0, captured_at_ms=1.0))

    assert pipeline.process_latest() is None
    assert pipeline.snapshot().stale_results == 1
    assert pipeline.snapshot().pending_frame_id == 1
    assert pipeline.process_latest() == 1


def test_capture_once_uses_source_packet_and_records_capture_and_pipeline_latency() -> None:
    source = FixtureCameraSource([_packet(0, captured_at_ms=10.0)])
    metrics = InMemoryMetrics()
    pipeline = LatestFramePipeline(
        source,
        lambda packet: packet.frame_id,
        metrics=metrics,
        clock_ms=lambda: 25.0,
    )
    pipeline.start()

    assert pipeline.capture_once().frame_id == 0
    assert pipeline.process_latest() == 0
    assert [sample.name for sample in metrics.snapshot()] == [
        "captured_frames_total",
        "pipeline_latency_ms",
        "pipeline_latency_ms",
    ]


def test_capture_error_closes_pipeline_and_source_without_a_worker() -> None:
    source = FixtureCameraSource()
    pipeline = LatestFramePipeline(source, lambda packet: packet.frame_id)
    pipeline.start()

    with pytest.raises(CameraError):
        pipeline.capture_once()

    assert pipeline.snapshot().is_closed is True
    assert pipeline.snapshot().is_running is False
    assert source.status.value == "DISCONNECTED"
    # The failing capture already released the source. A later close() is
    # idempotent, so the camera is released exactly once on the error path.
    assert source.close_count == 1
    pipeline.close()
    assert source.close_count == 1


def test_close_is_idempotent_discards_pending_frame_and_blocks_new_work() -> None:
    source = FixtureCameraSource()
    pipeline = LatestFramePipeline(source, lambda packet: packet.frame_id)
    pipeline.start()
    pipeline.submit(_packet(0))

    pipeline.close()
    pipeline.close()

    assert pipeline.snapshot().pending_frame_id is None
    assert source.close_count == 1
    with pytest.raises(RuntimeError, match="not running"):
        pipeline.submit(_packet(1))
    with pytest.raises(RuntimeError, match="not running"):
        pipeline.process_latest()


def test_processor_error_closes_camera_and_pipeline() -> None:
    source = FixtureCameraSource([_packet(0)])

    def fail(_: FramePacket) -> int:
        raise RuntimeError("synthetic vision failure")

    pipeline = LatestFramePipeline(source, fail)
    pipeline.start()
    pipeline.capture_once()

    with pytest.raises(RuntimeError, match="synthetic vision failure"):
        pipeline.process_latest()

    assert pipeline.snapshot().is_closed is True
    assert source.close_count == 1
