"""Pure confidence gating and debounced tracking recovery policy.

Times are monotonically increasing milliseconds.  The tracking policy only
returns the observation passed to the current update; it never re-emits a cached
sample while input is stale, missing, or recovering.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from gazelink.config import ConfidenceThresholds
from gazelink.domain import ContractValidationError, ReasonCode, TrackingState, VisionObservation


class ConfidenceReasonCode(StrEnum):
    """Policy-specific reasons not represented by the public domain event enum."""

    HEAD_POSE_MISSING = "HEAD_POSE_MISSING"
    HEAD_POSE_OUT_OF_RANGE = "HEAD_POSE_OUT_OF_RANGE"


PolicyReasonCode = ReasonCode | ConfidenceReasonCode
Clock = Callable[[], float]


def monotonic_ms() -> float:
    """Default injectable clock expressed in monotonic milliseconds."""

    return time.monotonic() * 1000.0


def _milliseconds(value: object, field_name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number of milliseconds")
    result = float(value)
    minimum = 0.0 if allow_zero else math.nextafter(0.0, math.inf)
    if not math.isfinite(result) or result < minimum:
        raise ContractValidationError(f"{field_name} must be {'>= 0' if allow_zero else '> 0'}")
    return result


def _positive_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ContractValidationError(f"{field_name} must be > 0")
    return result


def _unique(reasons: tuple[PolicyReasonCode, ...]) -> tuple[PolicyReasonCode, ...]:
    return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True, slots=True)
class HeadPoseLimits:
    """Maximum absolute head angles in degrees, using HeadPose yaw/pitch/roll order."""

    max_abs_yaw_deg: float = 30.0
    max_abs_pitch_deg: float = 25.0
    max_abs_roll_deg: float = 25.0

    def __post_init__(self) -> None:
        for attribute in ("max_abs_yaw_deg", "max_abs_pitch_deg", "max_abs_roll_deg"):
            object.__setattr__(
                self, attribute, _positive_number(getattr(self, attribute), attribute)
            )


@dataclass(frozen=True, slots=True)
class ConfidencePolicySettings:
    """Named policy limits; thresholds are owned by ``ProfileConfig``."""

    thresholds: ConfidenceThresholds
    max_sample_age_ms: float = 250.0
    head_pose_limits: HeadPoseLimits = HeadPoseLimits()

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, ConfidenceThresholds):
            raise ContractValidationError("thresholds must be ConfidenceThresholds")
        object.__setattr__(
            self,
            "max_sample_age_ms",
            _milliseconds(self.max_sample_age_ms, "max_sample_age_ms", allow_zero=True),
        )
        if not isinstance(self.head_pose_limits, HeadPoseLimits):
            raise ContractValidationError("head_pose_limits must be HeadPoseLimits")


@dataclass(frozen=True, slots=True)
class ConfidenceDecision:
    """Eligibility is separate from raw confidence and carries every rejection reason."""

    display_eligible: bool
    calibration_eligible: bool
    control_eligible: bool
    fresh: bool
    reason_codes: tuple[PolicyReasonCode, ...]


class ConfidencePolicy:
    """Evaluate current VisionObservation data without changing it or caching it."""

    def __init__(self, settings: ConfidencePolicySettings, *, clock: Clock = monotonic_ms) -> None:
        if not isinstance(settings, ConfidencePolicySettings):
            raise ContractValidationError("settings must be ConfidencePolicySettings")
        self._settings = settings
        self._clock = clock

    def evaluate(
        self,
        observation: VisionObservation | None,
        *,
        now_monotonic_ms: float | None = None,
    ) -> ConfidenceDecision:
        """Gate one observation; a missing observation is immediately ineligible."""

        now = self._clock() if now_monotonic_ms is None else now_monotonic_ms
        now = _milliseconds(now, "now_monotonic_ms", allow_zero=True)
        if observation is None:
            return ConfidenceDecision(False, False, False, False, (ReasonCode.FACE_NOT_FOUND,))

        age_ms = now - observation.observed_at_monotonic_ms
        fresh = 0.0 <= age_ms <= self._settings.max_sample_age_ms
        reasons: list[PolicyReasonCode] = list(observation.reason_codes)
        if not fresh:
            reasons.append(ReasonCode.STALE_SAMPLE)
        if observation.tracking_state is TrackingState.LOST:
            reasons.append(ReasonCode.FACE_NOT_FOUND)
        elif observation.tracking_state is TrackingState.MULTIPLE_FACES:
            reasons.append(ReasonCode.MULTIPLE_FACES)
        elif observation.tracking_state is TrackingState.LOW_CONFIDENCE:
            reasons.append(ReasonCode.LOW_CONFIDENCE)
        if observation.left_eye is None:
            reasons.append(ReasonCode.LEFT_EYE_OCCLUDED)
        if observation.right_eye is None:
            reasons.append(ReasonCode.RIGHT_EYE_OCCLUDED)
        if observation.head_pose is None:
            reasons.append(ConfidenceReasonCode.HEAD_POSE_MISSING)
        elif (
            abs(observation.head_pose.yaw_deg) > self._settings.head_pose_limits.max_abs_yaw_deg
            or abs(observation.head_pose.pitch_deg)
            > self._settings.head_pose_limits.max_abs_pitch_deg
            or abs(observation.head_pose.roll_deg)
            > self._settings.head_pose_limits.max_abs_roll_deg
        ):
            reasons.append(ConfidenceReasonCode.HEAD_POSE_OUT_OF_RANGE)

        hard_invalid = bool(reasons)
        if observation.overall_confidence < self._settings.thresholds.display:
            reasons.append(ReasonCode.LOW_CONFIDENCE)
        display_eligible = (
            fresh
            and not hard_invalid
            and (observation.overall_confidence >= self._settings.thresholds.display)
        )
        calibration_eligible = display_eligible and (
            observation.overall_confidence >= self._settings.thresholds.calibration
        )
        control_eligible = display_eligible and (
            observation.overall_confidence >= self._settings.thresholds.control
        )
        return ConfidenceDecision(
            display_eligible=display_eligible,
            calibration_eligible=calibration_eligible,
            control_eligible=control_eligible,
            fresh=fresh,
            reason_codes=_unique(tuple(reasons)),
        )


@dataclass(frozen=True, slots=True)
class TrackingRecoverySettings:
    """Debounce loss and require a stable run of fresh display-valid samples to recover."""

    confidence: ConfidencePolicySettings
    lost_debounce_ms: float = 150.0
    recovery_stable_ms: float = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, ConfidencePolicySettings):
            raise ContractValidationError("confidence must be ConfidencePolicySettings")
        object.__setattr__(
            self,
            "lost_debounce_ms",
            _milliseconds(self.lost_debounce_ms, "lost_debounce_ms", allow_zero=True),
        )
        object.__setattr__(
            self,
            "recovery_stable_ms",
            _milliseconds(self.recovery_stable_ms, "recovery_stable_ms", allow_zero=True),
        )


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    """Current-state result; ``accepted_observation`` is never a previous sample."""

    tracking_state: TrackingState
    accepted_observation: VisionObservation | None
    confidence: ConfidenceDecision
    recovered: bool


class TrackingRecoveryPolicy:
    """Stateful debounce/recovery wrapper driven by an injected monotonic clock."""

    def __init__(self, settings: TrackingRecoverySettings, *, clock: Clock = monotonic_ms) -> None:
        if not isinstance(settings, TrackingRecoverySettings):
            raise ContractValidationError("settings must be TrackingRecoverySettings")
        self._confidence_policy = ConfidencePolicy(settings.confidence, clock=clock)
        self._settings = settings
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Forget all timing state, including any observation that could be stale."""

        self._tracking_state = TrackingState.LOST
        self._loss_started_at_ms: float | None = None
        self._recovery_started_at_ms: float | None = None

    def update(
        self,
        observation: VisionObservation | None,
        *,
        now_monotonic_ms: float | None = None,
    ) -> TrackingUpdate:
        """Consume exactly one fresh observation and never replay prior input."""

        now = self._clock() if now_monotonic_ms is None else now_monotonic_ms
        now = _milliseconds(now, "now_monotonic_ms", allow_zero=True)
        decision = self._confidence_policy.evaluate(observation, now_monotonic_ms=now)
        if decision.display_eligible and observation is not None:
            self._loss_started_at_ms = None
            if self._tracking_state is TrackingState.TRACKED:
                return TrackingUpdate(TrackingState.TRACKED, observation, decision, recovered=False)
            if self._recovery_started_at_ms is None:
                self._recovery_started_at_ms = now
            if now - self._recovery_started_at_ms >= self._settings.recovery_stable_ms:
                self._tracking_state = TrackingState.TRACKED
                self._recovery_started_at_ms = None
                return TrackingUpdate(TrackingState.TRACKED, observation, decision, recovered=True)
            self._tracking_state = TrackingState.LOW_CONFIDENCE
            return TrackingUpdate(TrackingState.LOW_CONFIDENCE, None, decision, recovered=False)

        self._recovery_started_at_ms = None
        if self._tracking_state is TrackingState.TRACKED:
            if self._loss_started_at_ms is None:
                self._loss_started_at_ms = now
            if now - self._loss_started_at_ms < self._settings.lost_debounce_ms:
                return TrackingUpdate(TrackingState.TRACKED, None, decision, recovered=False)
        self._tracking_state = TrackingState.LOST
        self._loss_started_at_ms = None
        return TrackingUpdate(TrackingState.LOST, None, decision, recovered=False)
