"""Replaceable, in-process metrics contracts.

Metrics describe operational health only.  Labels are copied and frozen when a
sample is recorded so callers cannot mutate historical measurements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Protocol, TypeAlias

LabelValue: TypeAlias = str | int | float | bool
MetricLabels: TypeAlias = Mapping[str, LabelValue] | None
FrozenLabels: TypeAlias = tuple[tuple[str, str], ...]


class MetricKind(StrEnum):
    """Supported aggregation semantics for locally collected metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One immutable locally collected metric sample."""

    name: str
    kind: MetricKind
    value: float
    labels: FrozenLabels = ()


class MetricsCollector(Protocol):
    """Metrics port that can be replaced by production diagnostics later."""

    def counter(self, name: str, value: float = 1.0, *, labels: MetricLabels = None) -> None: ...

    def gauge(self, name: str, value: float, *, labels: MetricLabels = None) -> None: ...

    def histogram(self, name: str, value: float, *, labels: MetricLabels = None) -> None: ...

    def record_fps(self, fps: float, *, component: str) -> None: ...

    def record_latency_ms(self, latency_ms: float, *, component: str) -> None: ...

    def record_confidence(self, confidence: float, *, component: str) -> None: ...

    def record_tracking_loss(self, *, reason_code: str) -> None: ...

    def record_tracking_recovery(self, recovery_ms: float, *, reason_code: str) -> None: ...


def _freeze_labels(labels: MetricLabels) -> FrozenLabels:
    if labels is None:
        return ()
    frozen: list[tuple[str, str]] = []
    for key, value in labels.items():
        if not key or not key.replace("_", "").isalnum():
            raise ValueError("metric label names must be non-empty alphanumeric snake_case")
        if isinstance(value, bool):
            text = str(value).lower()
        elif isinstance(value, (str, int, float)):
            text = str(value)
        else:
            raise TypeError("metric label values must be scalar")
        frozen.append((key, text))
    return tuple(sorted(frozen))


def _validate_sample(name: str, value: float) -> float:
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("metric names must be non-empty alphanumeric snake_case")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("metric values must be finite")
    return numeric_value


class InMemoryMetrics:
    """Thread-safe metrics collector for tests and the local Foundation runtime."""

    def __init__(self) -> None:
        self._samples: list[MetricSample] = []
        self._lock = Lock()

    def counter(self, name: str, value: float = 1.0, *, labels: MetricLabels = None) -> None:
        self._record(name, MetricKind.COUNTER, value, labels)

    def gauge(self, name: str, value: float, *, labels: MetricLabels = None) -> None:
        self._record(name, MetricKind.GAUGE, value, labels)

    def histogram(self, name: str, value: float, *, labels: MetricLabels = None) -> None:
        self._record(name, MetricKind.HISTOGRAM, value, labels)

    def record_fps(self, fps: float, *, component: str) -> None:
        self.gauge("capture_fps", fps, labels={"component": component})

    def record_latency_ms(self, latency_ms: float, *, component: str) -> None:
        self.histogram("pipeline_latency_ms", latency_ms, labels={"component": component})

    def record_confidence(self, confidence: float, *, component: str) -> None:
        self.histogram("tracking_confidence", confidence, labels={"component": component})

    def record_tracking_loss(self, *, reason_code: str) -> None:
        self.counter("tracking_loss_total", labels={"reason_code": reason_code})

    def record_tracking_recovery(self, recovery_ms: float, *, reason_code: str) -> None:
        self.histogram("tracking_recovery_ms", recovery_ms, labels={"reason_code": reason_code})

    def snapshot(self) -> tuple[MetricSample, ...]:
        """Return an immutable point-in-time view of collected samples."""

        with self._lock:
            return tuple(self._samples)

    def _record(self, name: str, kind: MetricKind, value: float, labels: MetricLabels) -> None:
        sample = MetricSample(
            name=name,
            kind=kind,
            value=_validate_sample(name, value),
            labels=_freeze_labels(labels),
        )
        with self._lock:
            self._samples.append(sample)
