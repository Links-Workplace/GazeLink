"""Pure five-direction diagnostic for validating gaze features before calibration.

This module deliberately does not train or load a gaze model.  It answers the
earlier question: do vetted eye features move consistently when the user looks
left, right, up, and down?  Only scalar feature vectors are retained; no frame,
image, or landmark array is stored.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum

from gazelink.confidence import Clock
from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    JSONValue,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_features import (
    FEATURE_NAMES,
    GazeFeatureVector,
    from_observation,
    measure_feature_window_stability,
    select_stable_feature_window,
)


class FeatureCheckDirection(StrEnum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


class FeatureCheckPhase(StrEnum):
    STABILIZING = "STABILIZING"
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"


class DirectionVerdict(StrEnum):
    PASS = "PASS"
    NO_SEPARATION = "NO_SEPARATION"
    INVERTED = "INVERTED"
    EYES_DISAGREE = "EYES_DISAGREE"


@dataclass(frozen=True, slots=True)
class FeatureCheckSettings:
    """Unvalidated diagnostic thresholds, centralized for later live tuning."""

    stabilization_ms: float = 1_000.0
    collection_ms: float = 1_500.0
    min_sample_interval_ms: float = 80.0
    min_samples: int = 10
    min_horizontal_delta: float = 0.03
    min_vertical_combined_delta: float = 0.015
    min_vertical_per_eye_delta: float = 0.01
    max_stationary_p95: float = 0.08
    max_head_pose_p95_deg: float = 5.0

    def __post_init__(self) -> None:
        for name in ("stabilization_ms", "collection_ms", "min_sample_interval_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractValidationError(f"{name} must be a finite number")
            if not math.isfinite(float(value)) or value < 0:
                raise ContractValidationError(f"{name} must be finite and >= 0")
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < 1
        ):
            raise ContractValidationError("min_samples must be >= 1")
        for name in (
            "min_horizontal_delta",
            "min_vertical_combined_delta",
            "min_vertical_per_eye_delta",
            "max_stationary_p95",
            "max_head_pose_p95_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractValidationError(f"{name} must be a positive finite number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ContractValidationError(f"{name} must be > 0")


DEFAULT_FEATURE_CHECK_SETTINGS = FeatureCheckSettings()


@dataclass(frozen=True, slots=True)
class FeatureCheckTarget:
    direction: FeatureCheckDirection
    position: GazePoint


FEATURE_CHECK_TARGETS = (
    FeatureCheckTarget(FeatureCheckDirection.CENTER, GazePoint(0.50, 0.50)),
    FeatureCheckTarget(FeatureCheckDirection.LEFT, GazePoint(0.25, 0.50)),
    FeatureCheckTarget(FeatureCheckDirection.RIGHT, GazePoint(0.75, 0.50)),
    FeatureCheckTarget(FeatureCheckDirection.UP, GazePoint(0.50, 0.25)),
    FeatureCheckTarget(FeatureCheckDirection.DOWN, GazePoint(0.50, 0.75)),
)


@dataclass(frozen=True, slots=True)
class FeatureCheckSample:
    direction: FeatureCheckDirection
    frame_id: int
    observed_at_monotonic_ms: float
    features: GazeFeatureVector

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "direction": self.direction.value,
            "frame_id": self.frame_id,
            "observed_at_monotonic_ms": self.observed_at_monotonic_ms,
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FeatureCheckView:
    phase: FeatureCheckPhase
    target: FeatureCheckTarget | None
    target_number: int
    total_targets: int
    accepted_samples: int
    required_samples: int
    remaining_ms: float
    feedback: str


@dataclass(frozen=True, slots=True)
class FeatureTargetMetrics:
    direction: FeatureCheckDirection
    sample_count: int
    feature_medians: tuple[float, ...]
    gaze_horizontal_median: float
    gaze_vertical_median: float
    stationary_p95: float
    head_pose_p95_deg: float

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "direction": self.direction.value,
            "sample_count": self.sample_count,
            "feature_medians": {
                name: value for name, value in zip(FEATURE_NAMES, self.feature_medians, strict=True)
            },
            "gaze_horizontal_median": self.gaze_horizontal_median,
            "gaze_vertical_median": self.gaze_vertical_median,
            "stationary_p95": self.stationary_p95,
            "head_pose_p95_deg": self.head_pose_p95_deg,
        }


@dataclass(frozen=True, slots=True)
class DirectionCheck:
    direction: FeatureCheckDirection
    combined_delta: float
    left_eye_delta: float
    right_eye_delta: float
    verdict: DirectionVerdict

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "direction": self.direction.value,
            "combined_delta": self.combined_delta,
            "left_eye_delta": self.left_eye_delta,
            "right_eye_delta": self.right_eye_delta,
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True, slots=True)
class FeatureCheckResult:
    samples: tuple[FeatureCheckSample, ...]
    metrics: tuple[FeatureTargetMetrics, ...]
    direction_checks: tuple[DirectionCheck, ...]
    stable_directions: tuple[FeatureCheckDirection, ...]
    overall_pass: bool
    settings: FeatureCheckSettings

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "samples": [sample.to_dict() for sample in self.samples],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "direction_checks": [check.to_dict() for check in self.direction_checks],
            "stable_directions": [direction.value for direction in self.stable_directions],
            "overall_pass": self.overall_pass,
            "settings": {
                "stabilization_ms": self.settings.stabilization_ms,
                "collection_ms": self.settings.collection_ms,
                "min_sample_interval_ms": self.settings.min_sample_interval_ms,
                "min_samples": self.settings.min_samples,
                "min_horizontal_delta": self.settings.min_horizontal_delta,
                "min_vertical_combined_delta": self.settings.min_vertical_combined_delta,
                "min_vertical_per_eye_delta": self.settings.min_vertical_per_eye_delta,
                "max_stationary_p95": self.settings.max_stationary_p95,
                "max_head_pose_p95_deg": self.settings.max_head_pose_p95_deg,
            },
        }


class FeatureCheckController:
    """Collect five independent windows without training a gaze model."""

    def __init__(
        self,
        *,
        clock_ms: Clock,
        settings: FeatureCheckSettings = DEFAULT_FEATURE_CHECK_SETTINGS,
    ) -> None:
        self._clock_ms = clock_ms
        self._settings = settings
        self._phase = FeatureCheckPhase.STABILIZING
        self._target_index = 0
        self._phase_started_ms = clock_ms()
        self._last_frame_id: int | None = None
        self._last_sample_ms: float | None = None
        self._current_samples: list[FeatureCheckSample] = []
        self._completed_samples: list[FeatureCheckSample] = []
        self._feedback = "Look at CENTER and hold your gaze still."

    @property
    def phase(self) -> FeatureCheckPhase:
        return self._phase

    def ingest(self, observation: VisionObservation) -> FeatureCheckView:
        if self._phase not in {FeatureCheckPhase.STABILIZING, FeatureCheckPhase.COLLECTING}:
            return self.view()
        if observation.tracking_state is not TrackingState.TRACKED:
            return self.tracking_lost()
        if observation.frame_id == self._last_frame_id:
            return self.view()
        self._last_frame_id = observation.frame_id
        now_ms = self._clock_ms()
        if self._phase is FeatureCheckPhase.STABILIZING:
            if now_ms - self._phase_started_ms < self._settings.stabilization_ms:
                self._feedback = "Hold still; collection will start automatically."
                return self.view()
            self._phase = FeatureCheckPhase.COLLECTING
            self._phase_started_ms = now_ms
            self._last_sample_ms = None
        if (
            self._last_sample_ms is not None
            and now_ms - self._last_sample_ms < self._settings.min_sample_interval_ms
        ):
            return self.view()
        features = from_observation(observation)
        if features is None:
            self._feedback = "Both eyes and head pose must remain visible."
            return self.view()
        self._current_samples.append(
            FeatureCheckSample(
                direction=self._current_target.direction,
                frame_id=observation.frame_id,
                observed_at_monotonic_ms=observation.observed_at_monotonic_ms,
                features=features,
            )
        )
        self._last_sample_ms = now_ms
        elapsed_ms = now_ms - self._phase_started_ms
        if (
            elapsed_ms < self._settings.collection_ms
            or len(self._current_samples) < self._settings.min_samples
        ):
            self._feedback = "Collecting eye features; keep looking only at the target."
            return self.view()
        selection = select_stable_feature_window(
            [sample.features for sample in self._current_samples],
            window_size=self._settings.min_samples,
            max_eye_p95=self._settings.max_stationary_p95,
            max_head_pose_p95_deg=self._settings.max_head_pose_p95_deg,
        )
        if selection is None:
            self._current_samples.clear()
            self._reset_current_target(now_ms)
            self._feedback = (
                "No stable contiguous window was found. Hold the same target and retry."
            )
            return self.view()
        start, end, _stability = selection
        self._completed_samples.extend(self._current_samples[start:end])
        self._current_samples.clear()
        self._target_index += 1
        if self._target_index == len(FEATURE_CHECK_TARGETS):
            self._phase = FeatureCheckPhase.COMPLETE
            self._feedback = "Feature check complete."
            return self.view()
        self._reset_current_target(now_ms)
        self._feedback = f"Move your gaze to {self._current_target.direction.value} and hold still."
        return self.view()

    def tracking_lost(self) -> FeatureCheckView:
        """Discard a partial window so reacquisition never mixes two face poses."""

        if self._phase in {FeatureCheckPhase.STABILIZING, FeatureCheckPhase.COLLECTING}:
            self._current_samples.clear()
            self._reset_current_target(self._clock_ms())
            self._feedback = "Tracking lost. Reacquire your face, then hold on the same target."
        return self.view()

    def restart(self) -> FeatureCheckView:
        self._phase = FeatureCheckPhase.STABILIZING
        self._target_index = 0
        self._completed_samples.clear()
        self._current_samples.clear()
        self._last_frame_id = None
        self._reset_current_target(self._clock_ms())
        self._feedback = "Feature check restarted. Look at CENTER and hold still."
        return self.view()

    def cancel(self) -> FeatureCheckView:
        self._phase = FeatureCheckPhase.CANCELLED
        self._feedback = "Feature check cancelled."
        return self.view()

    def result(self) -> FeatureCheckResult:
        if self._phase is not FeatureCheckPhase.COMPLETE:
            raise ContractValidationError("feature-check result is available only after COMPLETE")
        return analyze_feature_check(tuple(self._completed_samples), settings=self._settings)

    def view(self) -> FeatureCheckView:
        if self._phase in {FeatureCheckPhase.COMPLETE, FeatureCheckPhase.CANCELLED}:
            return FeatureCheckView(
                phase=self._phase,
                target=None,
                target_number=len(FEATURE_CHECK_TARGETS),
                total_targets=len(FEATURE_CHECK_TARGETS),
                accepted_samples=len(self._completed_samples),
                required_samples=len(FEATURE_CHECK_TARGETS) * self._settings.min_samples,
                remaining_ms=0.0,
                feedback=self._feedback,
            )
        now_ms = self._clock_ms()
        duration = (
            self._settings.stabilization_ms
            if self._phase is FeatureCheckPhase.STABILIZING
            else self._settings.collection_ms
        )
        return FeatureCheckView(
            phase=self._phase,
            target=self._current_target,
            target_number=self._target_index + 1,
            total_targets=len(FEATURE_CHECK_TARGETS),
            accepted_samples=len(self._current_samples),
            required_samples=self._settings.min_samples,
            remaining_ms=max(0.0, duration - (now_ms - self._phase_started_ms)),
            feedback=self._feedback,
        )

    @property
    def _current_target(self) -> FeatureCheckTarget:
        return FEATURE_CHECK_TARGETS[self._target_index]

    def _reset_current_target(self, now_ms: float) -> None:
        self._phase = FeatureCheckPhase.STABILIZING
        self._phase_started_ms = now_ms
        self._last_sample_ms = None


def analyze_feature_check(
    samples: tuple[FeatureCheckSample, ...],
    *,
    settings: FeatureCheckSettings = DEFAULT_FEATURE_CHECK_SETTINGS,
) -> FeatureCheckResult:
    grouped = {
        direction: tuple(sample for sample in samples if sample.direction is direction)
        for direction in FeatureCheckDirection
    }
    if any(len(values) < settings.min_samples for values in grouped.values()):
        raise ContractValidationError("every feature-check direction requires enough samples")
    metrics = tuple(
        _target_metrics(direction, grouped[direction]) for direction in FeatureCheckDirection
    )
    by_direction = {metric.direction: metric for metric in metrics}
    center = by_direction[FeatureCheckDirection.CENTER]
    checks = tuple(
        _direction_check(direction, by_direction[direction], center, settings)
        for direction in (
            FeatureCheckDirection.LEFT,
            FeatureCheckDirection.RIGHT,
            FeatureCheckDirection.UP,
            FeatureCheckDirection.DOWN,
        )
    )
    stable = tuple(
        metric.direction
        for metric in metrics
        if metric.stationary_p95 <= settings.max_stationary_p95
        and metric.head_pose_p95_deg <= settings.max_head_pose_p95_deg
    )
    return FeatureCheckResult(
        samples=samples,
        metrics=metrics,
        direction_checks=checks,
        stable_directions=stable,
        overall_pass=(
            len(stable) == len(FeatureCheckDirection)
            and all(check.verdict is DirectionVerdict.PASS for check in checks)
        ),
        settings=settings,
    )


def _target_metrics(
    direction: FeatureCheckDirection, samples: tuple[FeatureCheckSample, ...]
) -> FeatureTargetMetrics:
    medians = tuple(
        statistics.median(sample.features.values[index] for sample in samples)
        for index in range(GazeFeatureVector.size)
    )
    horizontal_values = [
        (sample.features.values[0] + sample.features.values[2]) / 2.0 for sample in samples
    ]
    vertical_values = [
        (sample.features.values[1] + sample.features.values[3]) / 2.0 for sample in samples
    ]
    horizontal_median = statistics.median(horizontal_values)
    vertical_median = statistics.median(vertical_values)
    stability = measure_feature_window_stability([sample.features for sample in samples])
    return FeatureTargetMetrics(
        direction=direction,
        sample_count=len(samples),
        feature_medians=medians,
        gaze_horizontal_median=horizontal_median,
        gaze_vertical_median=vertical_median,
        stationary_p95=stability.eye_p95,
        head_pose_p95_deg=stability.head_pose_p95_deg,
    )


def _direction_check(
    direction: FeatureCheckDirection,
    metric: FeatureTargetMetrics,
    center: FeatureTargetMetrics,
    settings: FeatureCheckSettings,
) -> DirectionCheck:
    horizontal = direction in {FeatureCheckDirection.LEFT, FeatureCheckDirection.RIGHT}
    feature_index_left = 0 if horizontal else 1
    feature_index_right = 2 if horizontal else 3
    combined = (
        metric.gaze_horizontal_median - center.gaze_horizontal_median
        if horizontal
        else metric.gaze_vertical_median - center.gaze_vertical_median
    )
    left = metric.feature_medians[feature_index_left] - center.feature_medians[feature_index_left]
    right = (
        metric.feature_medians[feature_index_right] - center.feature_medians[feature_index_right]
    )
    expected_sign = (
        -1.0 if direction in {FeatureCheckDirection.LEFT, FeatureCheckDirection.UP} else 1.0
    )
    if horizontal:
        insufficient_separation = (
            abs(left) < settings.min_horizontal_delta or abs(right) < settings.min_horizontal_delta
        )
    else:
        insufficient_separation = (
            abs(combined) < settings.min_vertical_combined_delta
            or min(abs(left), abs(right)) < settings.min_vertical_per_eye_delta
        )
    if insufficient_separation:
        verdict = DirectionVerdict.NO_SEPARATION
    elif math.copysign(1.0, left) != math.copysign(1.0, right):
        verdict = DirectionVerdict.EYES_DISAGREE
    elif math.copysign(1.0, combined) != expected_sign:
        verdict = DirectionVerdict.INVERTED
    else:
        verdict = DirectionVerdict.PASS
    return DirectionCheck(direction, combined, left, right, verdict)
