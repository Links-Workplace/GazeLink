"""Deterministic M2 gaze smoothing and stationary-jitter measurements."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry


class GazeFilterKind(StrEnum):
    EMA = "EMA"
    ONE_EURO = "ONE_EURO"
    KALMAN = "KALMAN"


@dataclass(frozen=True, slots=True)
class GazeFilterSettings:
    """Centralized, unvalidated tuning placeholders for M2-04."""

    ema_alpha: float = 0.18
    one_euro_min_cutoff_hz: float = 0.8
    one_euro_beta: float = 0.04
    one_euro_derivative_cutoff_hz: float = 1.0
    kalman_process_noise: float = 0.0005
    kalman_measurement_noise: float = 0.01


_DEFAULT_FILTER_SETTINGS = GazeFilterSettings()


class _LowPass:
    def __init__(self) -> None:
        self.value: float | None = None

    def update(self, value: float, alpha: float) -> float:
        self.value = value if self.value is None else alpha * value + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class GazeStabilityFilter:
    """One resettable 2-D filter with no camera or UI dependency."""

    def __init__(
        self,
        kind: GazeFilterKind = GazeFilterKind.ONE_EURO,
        settings: GazeFilterSettings = _DEFAULT_FILTER_SETTINGS,
    ) -> None:
        if not isinstance(kind, GazeFilterKind):
            raise ContractValidationError("kind must be a GazeFilterKind")
        self.kind = kind
        self.settings = settings
        self._x = _LowPass()
        self._y = _LowPass()
        self._dx = _LowPass()
        self._dy = _LowPass()
        self._last_point: GazePoint | None = None
        self._last_ms: float | None = None
        self._kalman_x: list[float | None] = [None, None]
        self._kalman_p = [1.0, 1.0]

    def reset(self) -> None:
        for low_pass in (self._x, self._y, self._dx, self._dy):
            low_pass.reset()
        self._last_point = None
        self._last_ms = None
        self._kalman_x = [None, None]
        self._kalman_p = [1.0, 1.0]

    def update(self, point: GazePoint, timestamp_ms: float) -> GazePoint:
        if not isinstance(point, GazePoint):
            raise ContractValidationError("point must be a GazePoint")
        now = float(timestamp_ms)
        if not math.isfinite(now) or now < 0.0:
            raise ContractValidationError("timestamp_ms must be finite and >= 0")
        if self._last_ms is None or now <= self._last_ms:
            self.reset()
            self._last_point = point
            self._last_ms = now
            if self.kind is GazeFilterKind.KALMAN:
                self._kalman_x = [point.x, point.y]
            else:
                self._x.value, self._y.value = point.x, point.y
            return point
        dt_s = max(0.001, (now - self._last_ms) / 1000.0)
        if self.kind is GazeFilterKind.EMA:
            x = self._x.update(point.x, self.settings.ema_alpha)
            y = self._y.update(point.y, self.settings.ema_alpha)
        elif self.kind is GazeFilterKind.ONE_EURO:
            assert self._last_point is not None
            derivative_alpha = _alpha(self.settings.one_euro_derivative_cutoff_hz, dt_s)
            dx = self._dx.update((point.x - self._last_point.x) / dt_s, derivative_alpha)
            dy = self._dy.update((point.y - self._last_point.y) / dt_s, derivative_alpha)
            x = self._x.update(
                point.x,
                _alpha(
                    self.settings.one_euro_min_cutoff_hz + self.settings.one_euro_beta * abs(dx),
                    dt_s,
                ),
            )
            y = self._y.update(
                point.y,
                _alpha(
                    self.settings.one_euro_min_cutoff_hz + self.settings.one_euro_beta * abs(dy),
                    dt_s,
                ),
            )
        else:
            values: list[float] = []
            for axis, measurement in enumerate((point.x, point.y)):
                estimate = self._kalman_x[axis]
                if estimate is None:
                    estimate = measurement
                covariance = self._kalman_p[axis] + self.settings.kalman_process_noise
                gain = covariance / (covariance + self.settings.kalman_measurement_noise)
                estimate = estimate + gain * (measurement - estimate)
                self._kalman_x[axis] = estimate
                self._kalman_p[axis] = (1.0 - gain) * covariance
                values.append(estimate)
            x, y = values
        self._last_point = point
        self._last_ms = now
        return GazePoint(x, y)


@dataclass(frozen=True, slots=True)
class JitterMetrics:
    sample_count: int
    median_radius_px: float
    p95_radius_px: float
    p95_frame_jump_px: float


def measure_stationary_jitter(
    points: list[GazePoint], geometry: ScreenGeometry
) -> JitterMetrics | None:
    if len(points) < 3:
        return None
    center_x = statistics.median(point.x for point in points)
    center_y = statistics.median(point.y for point in points)
    width = max(1, geometry.width_px - 1)
    height = max(1, geometry.height_px - 1)
    radii = [
        math.hypot((point.x - center_x) * width, (point.y - center_y) * height) for point in points
    ]
    jumps = [
        math.hypot((current.x - previous.x) * width, (current.y - previous.y) * height)
        for previous, current in zip(points, points[1:], strict=False)
    ]
    return JitterMetrics(
        sample_count=len(points),
        median_radius_px=float(statistics.median(radii)),
        p95_radius_px=_percentile(radii, 95.0),
        p95_frame_jump_px=_percentile(jumps, 95.0),
    )


class RollingJitterMonitor:
    def __init__(self, geometry: ScreenGeometry, *, window_ms: float = 2000.0) -> None:
        self._geometry = geometry
        self._window_ms = window_ms
        self._samples: deque[tuple[float, GazePoint, GazePoint]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def add(self, timestamp_ms: float, raw: GazePoint, filtered: GazePoint) -> None:
        self._samples.append((timestamp_ms, raw, filtered))
        cutoff = timestamp_ms - self._window_ms
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def metrics(self) -> tuple[JitterMetrics | None, JitterMetrics | None]:
        return (
            measure_stationary_jitter([sample[1] for sample in self._samples], self._geometry),
            measure_stationary_jitter([sample[2] for sample in self._samples], self._geometry),
        )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
