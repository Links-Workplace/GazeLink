"""Time-aware, resettable filters for calibrated 2-D gaze points.

The public coordinates are normalised screen fractions.  Filtering happens in
device pixels so horizontal and vertical motion use the same physical unit and
the One Euro ``beta`` value means Hz per (pixel/second).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class FilterKind(StrEnum):
    OFF = "off"
    EMA = "ema"
    ONE_EURO = "one-euro"
    KALMAN = "kalman"


@dataclass(frozen=True)
class FilterSettings:
    """Central tuning settings, with all time values and units documented."""

    width_px: int
    height_px: int
    kind: FilterKind = FilterKind.ONE_EURO
    reset_gap_ms: float = 250.0
    ema_cutoff_hz: float = 2.0
    one_euro_min_cutoff_hz: float = 1.2
    one_euro_beta_hz_per_px_s: float = 0.0005
    one_euro_derivative_cutoff_hz: float = 1.0
    kalman_acceleration_noise_px2_s4: float = 400.0
    kalman_measurement_noise_px2: float = 900.0

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("screen dimensions must be positive")
        if not isinstance(self.kind, FilterKind):
            raise ValueError("kind must be a FilterKind")
        positive = {
            "reset_gap_ms": self.reset_gap_ms,
            "ema_cutoff_hz": self.ema_cutoff_hz,
            "one_euro_min_cutoff_hz": self.one_euro_min_cutoff_hz,
            "one_euro_derivative_cutoff_hz": self.one_euro_derivative_cutoff_hz,
            "kalman_acceleration_noise_px2_s4": self.kalman_acceleration_noise_px2_s4,
            "kalman_measurement_noise_px2": self.kalman_measurement_noise_px2,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        beta = self.one_euro_beta_hz_per_px_s
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("one_euro_beta_hz_per_px_s must be finite and >= 0")


# Named One Euro parameter sets, as (min_cutoff_hz, beta_hz_per_px_s).
#
# "responsive" is what gf_filter_benchmark selected on round2/TUNE under a
# 100ms synthetic-step guard.  "high-stability" is the heavy setting the
# operator confirmed by eye in round9: visibly steadier during fixation, but
# its cutoff never adapts to speed, so a large gaze move takes about a second
# to catch up.  Neither is asserted here to be the right default; that is
# decided from measurements, not from this table.
ONE_EURO_PRESETS: dict[str, tuple[float, float]] = {
    "responsive": (1.2, 0.0005),
    "high-stability": (0.4, 0.0),
}


class _LowPass:
    def __init__(self) -> None:
        self.value: float | None = None

    def update(self, value: float, alpha: float) -> float:
        self.value = value if self.value is None else alpha * value + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau_s = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau_s / dt_s)


class _KalmanAxis:
    """Constant-velocity Kalman state for one screen axis, in pixel units."""

    def __init__(self) -> None:
        self.position: float | None = None
        self.velocity = 0.0
        self.p00 = 1.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1000.0

    def reset(self) -> None:
        self.__init__()

    def update(
        self, measurement: float, dt_s: float, acceleration_noise: float, measurement_noise: float
    ) -> float:
        if self.position is None:
            self.position = measurement
            return measurement

        self.position += dt_s * self.velocity
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        p00 = (
            self.p00
            + dt_s * (self.p01 + self.p10)
            + dt2 * self.p11
            + acceleration_noise * dt4 / 4.0
        )
        p01 = self.p01 + dt_s * self.p11 + acceleration_noise * dt3 / 2.0
        p10 = self.p10 + dt_s * self.p11 + acceleration_noise * dt3 / 2.0
        p11 = self.p11 + acceleration_noise * dt2

        innovation_variance = p00 + measurement_noise
        k0 = p00 / innovation_variance
        k1 = p10 / innovation_variance
        residual = measurement - self.position
        self.position += k0 * residual
        self.velocity += k1 * residual
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p10 = p10 - k1 * p00
        self.p11 = p11 - k1 * p01
        return self.position


class GazePointFilter:
    """Filter one calibrated stream; invalid tracking must call ``reset``."""

    def __init__(self, settings: FilterSettings) -> None:
        self.settings = settings
        self._x = _LowPass()
        self._y = _LowPass()
        self._dx = _LowPass()
        self._dy = _LowPass()
        self._kalman_x = _KalmanAxis()
        self._kalman_y = _KalmanAxis()
        self._last_px: tuple[float, float] | None = None
        self._last_s: float | None = None

    def reset(self) -> None:
        for low_pass in (self._x, self._y, self._dx, self._dy):
            low_pass.reset()
        self._kalman_x.reset()
        self._kalman_y.reset()
        self._last_px = None
        self._last_s = None

    def update(self, point_norm: tuple[float, float], timestamp_s: float) -> tuple[float, float]:
        x_norm, y_norm = float(point_norm[0]), float(point_norm[1])
        now_s = float(timestamp_s)
        if not all(math.isfinite(value) for value in (x_norm, y_norm, now_s)):
            self.reset()
            raise ValueError("point and timestamp must be finite")

        point_px = (x_norm * self.settings.width_px, y_norm * self.settings.height_px)
        if self.settings.kind is FilterKind.OFF:
            self._last_px, self._last_s = point_px, now_s
            return (x_norm, y_norm)

        first = self._last_s is None
        if not first:
            assert self._last_s is not None
            dt_s = now_s - self._last_s
            if dt_s <= 0.0 or dt_s * 1000.0 > self.settings.reset_gap_ms:
                self.reset()
                first = True
        if first:
            self._seed(point_px, now_s)
            return (x_norm, y_norm)

        assert self._last_s is not None and self._last_px is not None
        dt_s = now_s - self._last_s
        if self.settings.kind is FilterKind.EMA:
            alpha = _alpha(self.settings.ema_cutoff_hz, dt_s)
            x_px = self._x.update(point_px[0], alpha)
            y_px = self._y.update(point_px[1], alpha)
        elif self.settings.kind is FilterKind.ONE_EURO:
            derivative_alpha = _alpha(self.settings.one_euro_derivative_cutoff_hz, dt_s)
            dx = self._dx.update((point_px[0] - self._last_px[0]) / dt_s, derivative_alpha)
            dy = self._dy.update((point_px[1] - self._last_px[1]) / dt_s, derivative_alpha)
            x_cutoff = (
                self.settings.one_euro_min_cutoff_hz
                + self.settings.one_euro_beta_hz_per_px_s * abs(dx)
            )
            y_cutoff = (
                self.settings.one_euro_min_cutoff_hz
                + self.settings.one_euro_beta_hz_per_px_s * abs(dy)
            )
            x_px = self._x.update(point_px[0], _alpha(x_cutoff, dt_s))
            y_px = self._y.update(point_px[1], _alpha(y_cutoff, dt_s))
        else:
            x_px = self._kalman_x.update(
                point_px[0],
                dt_s,
                self.settings.kalman_acceleration_noise_px2_s4,
                self.settings.kalman_measurement_noise_px2,
            )
            y_px = self._kalman_y.update(
                point_px[1],
                dt_s,
                self.settings.kalman_acceleration_noise_px2_s4,
                self.settings.kalman_measurement_noise_px2,
            )

        self._last_px, self._last_s = point_px, now_s
        return (x_px / self.settings.width_px, y_px / self.settings.height_px)

    def _seed(self, point_px: tuple[float, float], timestamp_s: float) -> None:
        self._x.value, self._y.value = point_px
        self._kalman_x.position, self._kalman_y.position = point_px
        self._last_px, self._last_s = point_px, timestamp_s
