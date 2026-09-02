from __future__ import annotations

import pytest

from gazelink.metrics import InMemoryMetrics, MetricKind


def test_in_memory_collector_records_foundation_pipeline_metrics() -> None:
    metrics = InMemoryMetrics()

    metrics.record_fps(29.7, component="capture")
    metrics.record_latency_ms(18.2, component="vision")
    metrics.record_confidence(0.91, component="tracking")
    metrics.record_tracking_loss(reason_code="no_face")
    metrics.record_tracking_recovery(430.0, reason_code="face_found")

    samples = metrics.snapshot()
    assert [sample.name for sample in samples] == [
        "capture_fps",
        "pipeline_latency_ms",
        "tracking_confidence",
        "tracking_loss_total",
        "tracking_recovery_ms",
    ]
    assert samples[0].kind is MetricKind.GAUGE
    assert samples[3].kind is MetricKind.COUNTER
    assert samples[4].labels == (("reason_code", "face_found"),)


def test_metric_labels_are_immutable_snapshots() -> None:
    metrics = InMemoryMetrics()
    labels = {"component": "capture"}

    metrics.gauge("capture_fps", 30.0, labels=labels)
    labels["component"] = "mutated"

    assert metrics.snapshot()[0].labels == (("component", "capture"),)


def test_metrics_reject_invalid_names_and_nonfinite_values() -> None:
    metrics = InMemoryMetrics()

    with pytest.raises(ValueError, match="metric names"):
        metrics.counter("bad metric")
    with pytest.raises(ValueError, match="finite"):
        metrics.histogram("pipeline_latency_ms", float("nan"))
