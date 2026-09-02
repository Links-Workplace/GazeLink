"""Tests for confidence gates and debounced, non-replaying tracking recovery."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gazelink.confidence import (
    ConfidencePolicy,
    ConfidencePolicySettings,
    ConfidenceReasonCode,
    HeadPoseLimits,
    TrackingRecoveryPolicy,
    TrackingRecoverySettings,
)
from gazelink.config import ConfidenceThresholds, ConfigurationError
from gazelink.domain import (
    EyeFeatures,
    HeadPose,
    NormalizedBox,
    NormalizedPoint,
    ReasonCode,
    TrackingState,
    VisionObservation,
)

_NEUTRAL_POSE = HeadPose(0.0, 0.0, 0.0)


@dataclass
class FakeClock:
    value_ms: float = 0.0

    def __call__(self) -> float:
        return self.value_ms


def _observation(
    *,
    observed_at_ms: float,
    confidence: float = 0.9,
    tracking_state: TrackingState = TrackingState.TRACKED,
    left: bool = True,
    right: bool = True,
    pose: HeadPose | None = _NEUTRAL_POSE,
) -> VisionObservation:
    eye = EyeFeatures(NormalizedPoint(0.5, 0.5), openness=0.3, confidence=0.9)
    return VisionObservation(
        frame_id=int(observed_at_ms),
        observed_at_monotonic_ms=observed_at_ms,
        tracking_state=tracking_state,
        face_box=NormalizedBox(0.2, 0.2, 0.4, 0.5),
        left_eye=eye if left else None,
        right_eye=eye if right else None,
        head_pose=pose,
        overall_confidence=confidence,
    )


def _settings() -> ConfidencePolicySettings:
    return ConfidencePolicySettings(
        thresholds=ConfidenceThresholds(display=0.5, calibration=0.7, control=0.85),
        max_sample_age_ms=100.0,
        head_pose_limits=HeadPoseLimits(20.0, 15.0, 10.0),
    )


def test_confidence_thresholds_and_low_confidence_are_gated_separately() -> None:
    policy = ConfidencePolicy(_settings())

    middle = policy.evaluate(
        _observation(observed_at_ms=100.0, confidence=0.75), now_monotonic_ms=150.0
    )
    low = policy.evaluate(
        _observation(observed_at_ms=100.0, confidence=0.49), now_monotonic_ms=150.0
    )

    assert middle.display_eligible and middle.calibration_eligible
    assert not middle.control_eligible
    assert not low.display_eligible
    assert ReasonCode.LOW_CONFIDENCE in low.reason_codes


def test_confidence_threshold_ordering_is_validated_before_policy_creation() -> None:
    with pytest.raises(ConfigurationError, match="display"):
        ConfidenceThresholds(display=0.8, calibration=0.7, control=0.9)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"left": False}, ReasonCode.LEFT_EYE_OCCLUDED),
        ({"right": False}, ReasonCode.RIGHT_EYE_OCCLUDED),
        ({"pose": None}, ConfidenceReasonCode.HEAD_POSE_MISSING),
        ({"pose": HeadPose(21.0, 0.0, 0.0)}, ConfidenceReasonCode.HEAD_POSE_OUT_OF_RANGE),
    ],
)
def test_missing_eyes_and_invalid_pose_are_never_eligible(
    kwargs: dict[str, object], reason: object
) -> None:
    decision = ConfidencePolicy(_settings()).evaluate(
        _observation(observed_at_ms=100.0, **kwargs),  # type: ignore[arg-type]
        now_monotonic_ms=150.0,
    )

    assert not decision.display_eligible
    assert not decision.calibration_eligible
    assert not decision.control_eligible
    assert reason in decision.reason_codes


def test_stale_or_future_observation_is_rejected() -> None:
    policy = ConfidencePolicy(_settings())

    stale = policy.evaluate(_observation(observed_at_ms=0.0), now_monotonic_ms=101.0)
    future = policy.evaluate(_observation(observed_at_ms=200.0), now_monotonic_ms=150.0)

    assert not stale.fresh and ReasonCode.STALE_SAMPLE in stale.reason_codes
    assert not future.fresh and ReasonCode.STALE_SAMPLE in future.reason_codes


def test_tracking_loss_is_debounced_and_recovery_requires_stable_fresh_samples() -> None:
    clock = FakeClock()
    tracker = TrackingRecoveryPolicy(
        TrackingRecoverySettings(_settings(), lost_debounce_ms=50.0, recovery_stable_ms=40.0),
        clock=clock,
    )

    clock.value_ms = 0.0
    assert tracker.update(_observation(observed_at_ms=0.0)).accepted_observation is None
    clock.value_ms = 40.0
    recovered = tracker.update(_observation(observed_at_ms=40.0))
    assert recovered.tracking_state is TrackingState.TRACKED
    assert recovered.recovered
    assert recovered.accepted_observation is not None

    clock.value_ms = 50.0
    noisy = tracker.update(_observation(observed_at_ms=50.0, left=False))
    assert noisy.tracking_state is TrackingState.TRACKED
    assert noisy.accepted_observation is None
    clock.value_ms = 101.0
    lost = tracker.update(None)
    assert lost.tracking_state is TrackingState.LOST
    assert lost.accepted_observation is None

    clock.value_ms = 110.0
    assert tracker.update(_observation(observed_at_ms=110.0)).accepted_observation is None
    clock.value_ms = 130.0
    assert tracker.update(_observation(observed_at_ms=130.0)).accepted_observation is None
    clock.value_ms = 150.0
    stable = tracker.update(_observation(observed_at_ms=150.0))
    assert stable.tracking_state is TrackingState.TRACKED
    assert stable.recovered
    assert stable.accepted_observation is not None


def test_reset_discards_recovery_progress_and_never_reuses_a_previous_sample() -> None:
    clock = FakeClock(0.0)
    tracker = TrackingRecoveryPolicy(
        TrackingRecoverySettings(_settings(), recovery_stable_ms=20.0),
        clock=clock,
    )
    first = _observation(observed_at_ms=0.0)
    assert tracker.update(first).accepted_observation is None

    tracker.reset()
    clock.value_ms = 20.0
    after_reset = tracker.update(_observation(observed_at_ms=20.0))

    assert after_reset.tracking_state is TrackingState.LOW_CONFIDENCE
    assert after_reset.accepted_observation is None


def test_noisy_recovery_sequence_restarts_the_stability_window() -> None:
    clock = FakeClock(0.0)
    tracker = TrackingRecoveryPolicy(
        TrackingRecoverySettings(_settings(), recovery_stable_ms=30.0),
        clock=clock,
    )

    assert tracker.update(_observation(observed_at_ms=0.0)).accepted_observation is None
    clock.value_ms = 20.0
    assert tracker.update(None).tracking_state is TrackingState.LOST
    clock.value_ms = 30.0
    assert tracker.update(_observation(observed_at_ms=30.0)).accepted_observation is None
    clock.value_ms = 50.0
    assert tracker.update(_observation(observed_at_ms=50.0)).accepted_observation is None
    clock.value_ms = 60.0
    recovered = tracker.update(_observation(observed_at_ms=60.0))

    assert recovered.tracking_state is TrackingState.TRACKED
    assert recovered.recovered
