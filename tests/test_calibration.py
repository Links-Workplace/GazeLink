"""Tests for the 9-point calibration session state machine.

Covers every state transition and, per DoD, every *invalid* transition too --
a test should fail if a partial session could ever be exported as a valid
result, or if an action allowed only while ``COLLECTING`` silently succeeds
from another state.
"""

from __future__ import annotations

import pytest

from gazelink.calibration import (
    DEFAULT_MIN_SAMPLES_PER_TARGET,
    FEATURE_SCHEMA_VERSION,
    NUM_CALIBRATION_TARGETS,
    CalibrationReasonCode,
    CalibrationSample,
    CalibrationSession,
    CalibrationSessionResult,
    CalibrationSessionState,
    CalibrationTarget,
    default_nine_point_targets,
    targets_for_screen_geometry,
)
from gazelink.domain import ContractValidationError, NormalizedPoint, ScreenGeometry


class FakeClock:
    def __init__(self, start_ms: float = 1_000.0) -> None:
        self.now_ms = start_ms

    def __call__(self) -> float:
        return self.now_ms

    def advance(self, ms: float) -> float:
        self.now_ms += ms
        return self.now_ms


def _geometry() -> ScreenGeometry:
    return ScreenGeometry(screen_id="primary", width_px=1920, height_px=1080, dpi_scale=1.0)


def _session(
    *, clock: FakeClock | None = None, min_samples_per_target: int = 2, **kwargs: object
) -> CalibrationSession:
    return CalibrationSession(
        camera_id="camera-0",
        screen_geometry=_geometry(),
        min_samples_per_target=min_samples_per_target,
        clock_ms=clock or FakeClock(),
        **kwargs,  # type: ignore[arg-type]
    )


def _sample(
    session: CalibrationSession,
    *,
    accepted: bool = True,
    reason: str = "accepted",
    frame_id: int = 1,
    target_index: int | None = None,
) -> CalibrationSample:
    """Build a real CalibrationSample for the session's current target.

    ``target_index`` can be supplied explicitly for tests that build a sample
    against a session that has already left COLLECTING (reading
    ``current_target`` there raises on its own, which would attribute a
    later failure to the wrong call).
    """

    return CalibrationSample(
        source_frame_id=frame_id,
        observed_at_monotonic_ms=100.0,
        target_index=session.current_target.index if target_index is None else target_index,
        left_iris_in_eye=NormalizedPoint(0.5, 0.5),
        right_iris_in_eye=NormalizedPoint(0.5, 0.5),
        left_openness=0.6,
        right_openness=0.6,
        head_yaw_deg=1.0,
        head_pitch_deg=-1.0,
        head_roll_deg=0.5,
        confidence=0.5,
        accepted=accepted,
        reason=reason,
    )


def _complete(session: CalibrationSession) -> None:
    """Drive a fresh session straight through to COMPLETE."""

    frame_id = 1
    for _ in range(NUM_CALIBRATION_TARGETS):
        for _ in range(session.min_samples_per_target):
            session.record_sample(_sample(session, frame_id=frame_id))
            frame_id += 1
        session.advance()


# --- CalibrationTarget / default_nine_point_targets -------------------------


def test_default_nine_point_targets_are_a_well_formed_grid() -> None:
    targets = default_nine_point_targets()

    assert len(targets) == NUM_CALIBRATION_TARGETS
    assert {target.index for target in targets} == set(range(NUM_CALIBRATION_TARGETS))
    assert all(0.0 <= target.screen_position.x <= 1.0 for target in targets)
    assert all(0.0 <= target.screen_position.y <= 1.0 for target in targets)
    # Index 4 is the row-major center of a 3x3 grid.
    assert targets[4].screen_position == NormalizedPoint(0.5, 0.5)


def test_ultrawide_grid_uses_a_more_comfortable_horizontal_inset() -> None:
    targets = targets_for_screen_geometry(ScreenGeometry("wide", 5120, 1440, 1.0))
    assert targets[0].screen_position == NormalizedPoint(0.15, 0.05)
    assert targets[4].screen_position == NormalizedPoint(0.5, 0.5)
    assert targets[8].screen_position == NormalizedPoint(0.85, 0.95)


def test_target_edge_inset_keeps_corners_off_the_true_screen_edge() -> None:
    targets = default_nine_point_targets(edge_inset=0.1)

    corner = targets[0]
    assert corner.screen_position.x == pytest.approx(0.1)
    assert corner.screen_position.y == pytest.approx(0.1)


def test_invalid_edge_inset_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="edge_inset"):
        default_nine_point_targets(edge_inset=0.5)


def test_calibration_target_round_trips_through_serialization() -> None:
    target = CalibrationTarget(index=3, screen_position=NormalizedPoint(0.2, 0.8))

    assert CalibrationTarget.from_dict(target.to_dict()) == target


def test_calibration_target_index_out_of_range_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="index"):
        CalibrationTarget(index=9, screen_position=NormalizedPoint(0.5, 0.5))


# --- Construction -------------------------------------------------------


def test_session_starts_collecting_at_the_first_target_in_order() -> None:
    session = _session()

    assert session.state is CalibrationSessionState.COLLECTING
    assert session.current_target.index == 0
    assert session.current_target_sample_count == 0


def test_custom_target_order_is_honored() -> None:
    order = (8, 0, 4, 1, 2, 3, 5, 6, 7)
    session = _session(target_order=order)

    assert session.current_target.index == 8


@pytest.mark.parametrize(
    "bad_order",
    [
        (0, 1, 2, 3, 4, 5, 6, 7, 7),  # duplicate
        (0, 1, 2, 3, 4, 5, 6, 7, 9),  # out of range
        (0, 1, 2, 3, 4, 5, 6, 7),  # too short
    ],
)
def test_invalid_target_order_is_rejected(bad_order: tuple[int, ...]) -> None:
    with pytest.raises(ContractValidationError, match="target_order"):
        _session(target_order=bad_order)


def test_empty_camera_id_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="camera_id"):
        CalibrationSession(camera_id="  ", screen_geometry=_geometry())


def test_non_positive_min_samples_per_target_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="min_samples_per_target"):
        _session(min_samples_per_target=0)


def test_wrong_number_of_targets_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="9 targets"):
        _session(targets=default_nine_point_targets()[:8])


# --- record_sample / advance ---------------------------------------------


def test_recording_a_valid_sample_increments_only_the_current_target() -> None:
    session = _session(min_samples_per_target=3)

    session.record_sample(_sample(session))
    session.record_sample(_sample(session))

    assert session.current_target_sample_count == 2


def test_recording_an_invalid_sample_does_not_increment_the_count() -> None:
    session = _session()

    session.record_sample(_sample(session, accepted=False, reason="low_confidence"))

    assert session.current_target_sample_count == 0


def test_advance_before_the_minimum_is_reached_is_rejected() -> None:
    session = _session(min_samples_per_target=3)
    session.record_sample(_sample(session))

    with pytest.raises(ContractValidationError, match="required samples"):
        session.advance()

    # The forbidden transition must not have moved state or position.
    assert session.state is CalibrationSessionState.COLLECTING
    assert session.current_target.index == 0


def test_advance_after_the_minimum_moves_to_the_next_target() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session))

    session.advance()

    assert session.state is CalibrationSessionState.COLLECTING
    assert session.current_target.index == 1
    assert session.current_target_sample_count == 0


def test_advancing_past_the_last_target_completes_the_session() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)

    assert session.state is CalibrationSessionState.COMPLETE


def test_cannot_record_a_sample_after_the_session_is_complete() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)
    sample = _sample(session, target_index=0)  # built before the raises block on purpose

    with pytest.raises(ContractValidationError, match="COMPLETE"):
        session.record_sample(sample)


def test_cannot_advance_a_completed_session() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)

    with pytest.raises(ContractValidationError, match="COMPLETE"):
        session.advance()


def test_reading_current_target_after_completion_is_rejected() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)

    with pytest.raises(ContractValidationError, match="COMPLETE"):
        _ = session.current_target


# --- cancel / fail ---------------------------------------------------------


def test_cancel_from_collecting_marks_the_session_cancelled() -> None:
    session = _session()

    session.cancel()

    assert session.state is CalibrationSessionState.CANCELLED


def test_cannot_cancel_a_session_that_is_already_cancelled() -> None:
    session = _session()
    session.cancel()

    with pytest.raises(ContractValidationError, match="CANCELLED"):
        session.cancel()


def test_cannot_cancel_a_completed_session() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)

    with pytest.raises(ContractValidationError, match="COMPLETE"):
        session.cancel()


def test_fail_from_collecting_marks_the_session_failed() -> None:
    session = _session()

    session.fail()

    assert session.state is CalibrationSessionState.FAILED


def test_cannot_fail_a_cancelled_session() -> None:
    session = _session()
    session.cancel()

    with pytest.raises(ContractValidationError, match="CANCELLED"):
        session.fail()


# --- restart ----------------------------------------------------------------


def test_restart_from_mid_collection_discards_progress() -> None:
    session = _session(min_samples_per_target=3)
    session.record_sample(_sample(session))
    session.record_sample(_sample(session))

    session.restart()

    assert session.state is CalibrationSessionState.COLLECTING
    assert session.current_target.index == 0
    assert session.current_target_sample_count == 0


def test_restart_from_a_completed_session_allows_collecting_again() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)

    session.restart()

    assert session.state is CalibrationSessionState.COLLECTING
    with pytest.raises(ContractValidationError):
        session.result()


def test_restart_from_a_cancelled_session_allows_collecting_again() -> None:
    session = _session()
    session.cancel()

    session.restart()

    assert session.state is CalibrationSessionState.COLLECTING


# --- result() / "a partial session is never marked valid" -----------------


@pytest.mark.parametrize(
    "make_non_complete",
    [
        lambda session: None,  # still COLLECTING
        lambda session: session.cancel(),
        lambda session: session.fail(),
    ],
)
def test_result_is_unavailable_for_anything_but_a_complete_session(
    make_non_complete: object,
) -> None:
    session = _session()
    make_non_complete(session)  # type: ignore[operator]

    with pytest.raises(ContractValidationError, match="COMPLETE"):
        session.result()


def test_result_is_available_and_correct_once_complete() -> None:
    clock = FakeClock(start_ms=1_000.0)
    session = _session(clock=clock, min_samples_per_target=1)
    clock.advance(500.0)
    _complete(session)

    result = session.result()

    assert result.camera_id == "camera-0"
    assert result.screen_geometry == _geometry()
    assert result.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert result.min_samples_per_target == 1
    assert result.sample_counts == (1,) * NUM_CALIBRATION_TARGETS
    assert result.target_order == tuple(range(NUM_CALIBRATION_TARGETS))
    assert result.started_at_monotonic_ms == 1_000.0
    assert result.completed_at_monotonic_ms == 1_500.0


def test_calibration_session_result_round_trips_through_serialization() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)
    result = session.result()

    assert CalibrationSessionResult.from_dict(result.to_dict()) == result


def test_result_completed_before_started_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="completed_at_monotonic_ms"):
        CalibrationSessionResult(
            target_order=tuple(range(NUM_CALIBRATION_TARGETS)),
            targets=default_nine_point_targets(),
            sample_counts=(1,) * NUM_CALIBRATION_TARGETS,
            samples=(),
            camera_id="camera-0",
            screen_geometry=_geometry(),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            min_samples_per_target=DEFAULT_MIN_SAMPLES_PER_TARGET,
            started_at_monotonic_ms=2_000.0,
            completed_at_monotonic_ms=1_000.0,
        )


# --- CalibrationSample: the real dataset (M2-02 Stage A) -------------------


def test_calibration_sample_round_trips_through_serialization_with_real_values() -> None:
    sample = CalibrationSample(
        source_frame_id=42,
        observed_at_monotonic_ms=1_234.5,
        target_index=3,
        left_iris_in_eye=NormalizedPoint(0.4, 0.6),
        right_iris_in_eye=NormalizedPoint(0.55, 0.45),
        left_openness=0.62,
        right_openness=0.58,
        head_yaw_deg=12.3,
        head_pitch_deg=-4.5,
        head_roll_deg=1.1,
        confidence=0.5,
        accepted=True,
        reason="accepted",
    )

    assert CalibrationSample.from_dict(sample.to_dict()) == sample


def test_calibration_sample_permits_none_features_for_a_rejected_frame() -> None:
    """A LOST/no-face observation has no eye or head-pose features at all --
    the contract must accept that rather than forcing fabricated zeros."""

    sample = CalibrationSample(
        source_frame_id=1,
        observed_at_monotonic_ms=0.0,
        target_index=0,
        left_iris_in_eye=None,
        right_iris_in_eye=None,
        left_openness=None,
        right_openness=None,
        head_yaw_deg=None,
        head_pitch_deg=None,
        head_roll_deg=None,
        confidence=0.0,
        accepted=False,
        reason="tracking_not_ready",
    )

    assert CalibrationSample.from_dict(sample.to_dict()) == sample


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("confidence", 1.5),
        ("confidence", -0.1),
        ("left_openness", 2.0),
        ("head_yaw_deg", float("nan")),
        ("target_index", 9),
        ("source_frame_id", -1),
        ("reason", "not_a_real_reason"),
        ("reason", ""),
        ("reason", 123),
    ],
)
def test_calibration_sample_rejects_invalid_field_values(field: str, bad_value: object) -> None:
    kwargs: dict[str, object] = dict(
        source_frame_id=1,
        observed_at_monotonic_ms=0.0,
        target_index=0,
        left_iris_in_eye=None,
        right_iris_in_eye=None,
        left_openness=None,
        right_openness=None,
        head_yaw_deg=None,
        head_pitch_deg=None,
        head_roll_deg=None,
        confidence=0.5,
        accepted=True,
        reason="accepted",
    )
    kwargs[field] = bad_value

    with pytest.raises(ContractValidationError):
        CalibrationSample(**kwargs)  # type: ignore[arg-type]


def _raw_sample(**overrides: object) -> CalibrationSample:
    """Build a CalibrationSample directly, not tied to any session's current
    target -- for tests that only care about ``reason`` validation or the
    result-level derived metrics (``reason_counts``, ``head_pose_spread``),
    where a session's target-matching rules would just be noise."""

    kwargs: dict[str, object] = dict(
        source_frame_id=1,
        observed_at_monotonic_ms=0.0,
        target_index=0,
        left_iris_in_eye=None,
        right_iris_in_eye=None,
        left_openness=None,
        right_openness=None,
        head_yaw_deg=None,
        head_pitch_deg=None,
        head_roll_deg=None,
        confidence=0.5,
        accepted=True,
        reason="accepted",
    )
    kwargs.update(overrides)
    return CalibrationSample(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("reason_code", list(CalibrationReasonCode))
def test_every_closed_set_reason_code_is_accepted(reason_code: CalibrationReasonCode) -> None:
    sample = _raw_sample(reason=reason_code.value)

    assert sample.reason == reason_code.value


def test_passing_the_enum_member_normalizes_to_its_plain_string_value() -> None:
    """The whole point of normalizing in __post_init__ is byte-identical
    serialization: a caller that passes the enum member must still get a
    plain ``str`` back out of ``sample.reason`` and ``to_dict()``, so an
    existing session log's ``f"reason={sample.reason}"`` rendering, and any
    already-written serialized dataset, never change shape because of which
    form a caller happened to pass in."""

    sample = _raw_sample(reason=CalibrationReasonCode.LOW_CONFIDENCE)

    assert sample.reason == "low_confidence"
    assert type(sample.reason) is str
    assert sample.to_dict()["reason"] == "low_confidence"


def test_rejected_samples_are_stored_but_never_counted() -> None:
    session = _session(min_samples_per_target=2)

    session.record_sample(_sample(session, accepted=False, reason="low_confidence"))

    assert session.current_target_sample_count == 0


def test_record_sample_for_the_wrong_target_is_rejected() -> None:
    session = _session(min_samples_per_target=2)
    wrong_target_sample = _sample(session, target_index=5)  # current target is 0

    with pytest.raises(ContractValidationError, match="target_index"):
        session.record_sample(wrong_target_sample)


def test_result_samples_contains_both_accepted_and_rejected_entries_in_order() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session, accepted=False, reason="low_confidence", frame_id=1))
    session.record_sample(_sample(session, accepted=True, reason="accepted", frame_id=2))
    session.advance()
    for target in range(1, NUM_CALIBRATION_TARGETS):
        session.record_sample(_sample(session, frame_id=100 + target))
        session.advance()

    result = session.result()

    assert len(result.samples) == NUM_CALIBRATION_TARGETS + 1  # 8 extra + 1 rejected
    first_target_samples = [s for s in result.samples if s.target_index == 0]
    assert [s.accepted for s in first_target_samples] == [False, True]
    assert [s.source_frame_id for s in first_target_samples] == [1, 2]


def test_retry_current_target_discards_stored_samples_for_that_target_only() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session, frame_id=1))

    session.retry_current_target()

    assert session.current_target_sample_count == 0
    session.record_sample(_sample(session, frame_id=2))
    session.advance()
    for target in range(1, NUM_CALIBRATION_TARGETS):
        session.record_sample(_sample(session, frame_id=100 + target))
        session.advance()

    result = session.result()

    target_0_samples = [s for s in result.samples if s.target_index == 0]
    # Only the post-retry sample survives: retry must have discarded frame_id=1.
    assert [s.source_frame_id for s in target_0_samples] == [2]


# --- current_target_accepted_samples / accepted_samples (M2-02 Stage B) ----


def test_current_target_accepted_samples_is_empty_on_a_fresh_session() -> None:
    session = _session()

    assert session.current_target_accepted_samples == ()


def test_current_target_accepted_samples_reflects_only_the_current_target() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session, accepted=False, reason="low_confidence", frame_id=1))
    session.record_sample(_sample(session, accepted=True, reason="accepted", frame_id=2))

    assert [s.source_frame_id for s in session.current_target_accepted_samples] == [2]

    session.advance()  # moves to target 1

    # The new target must not see target 0's accepted samples.
    assert session.current_target_accepted_samples == ()


def test_retry_current_target_clears_current_target_accepted_samples() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session, frame_id=1))
    assert len(session.current_target_accepted_samples) == 1

    session.retry_current_target()

    assert not session.current_target_accepted_samples


def test_current_target_accepted_samples_raises_once_session_leaves_collecting() -> None:
    session = _session(min_samples_per_target=1)
    _complete(session)

    with pytest.raises(ContractValidationError, match="COMPLETE"):
        _ = session.current_target_accepted_samples


def test_accepted_samples_accumulates_across_targets_and_excludes_rejected() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session, accepted=False, reason="low_confidence", frame_id=1))
    session.record_sample(_sample(session, accepted=True, reason="accepted", frame_id=2))
    session.advance()
    session.record_sample(_sample(session, accepted=True, reason="accepted", frame_id=3))

    accepted = session.accepted_samples

    assert [s.source_frame_id for s in accepted] == [2, 3]
    assert all(s.accepted for s in accepted)


def test_restart_resets_accepted_samples_to_empty() -> None:
    session = _session(min_samples_per_target=1)
    session.record_sample(_sample(session))
    assert session.accepted_samples != ()

    session.restart()

    assert session.accepted_samples == ()


def test_accepted_samples_is_readable_on_a_complete_session_without_raising() -> None:
    """Unlike current_target_accepted_samples, this accessor must stay
    readable after COMPLETE -- a descriptive metric like head-pose spread
    needs to be computable on a finished session, not just mid-collection."""

    session = _session(min_samples_per_target=1)
    _complete(session)

    accepted = session.accepted_samples

    assert len(accepted) == NUM_CALIBRATION_TARGETS
    assert all(s.accepted for s in accepted)


# --- CalibrationSessionResult.reason_counts / head_pose_spread -------------


def _result_with_samples(samples: tuple[CalibrationSample, ...]) -> CalibrationSessionResult:
    return CalibrationSessionResult(
        target_order=tuple(range(NUM_CALIBRATION_TARGETS)),
        targets=default_nine_point_targets(),
        sample_counts=(1,) * NUM_CALIBRATION_TARGETS,
        samples=samples,
        camera_id="camera-0",
        screen_geometry=_geometry(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        min_samples_per_target=DEFAULT_MIN_SAMPLES_PER_TARGET,
        started_at_monotonic_ms=1_000.0,
        completed_at_monotonic_ms=1_500.0,
    )


def test_reason_counts_tallies_every_reason_present_and_omits_absent_ones() -> None:
    samples = (
        _raw_sample(source_frame_id=1, accepted=True, reason="accepted"),
        _raw_sample(source_frame_id=2, accepted=True, reason="accepted"),
        _raw_sample(source_frame_id=3, accepted=False, reason="low_confidence"),
        _raw_sample(source_frame_id=4, accepted=False, reason="eye_not_visible"),
        _raw_sample(source_frame_id=5, accepted=False, reason="eye_not_visible"),
    )
    result = _result_with_samples(samples)

    # Exactly the reasons present, each counted correctly; no zero entries
    # for the five CalibrationReasonCode values that never occurred.
    assert result.reason_counts() == {
        "accepted": 2,
        "low_confidence": 1,
        "eye_not_visible": 2,
    }


def test_head_pose_spread_excludes_rejected_samples_from_the_medians() -> None:
    """A rejected sample's extreme head pose must never widen the reported
    spread -- head_pose_spread is diagnostic over the *accepted* dataset
    only, and letting a correctly-rejected outlier leak into it would
    misreport how stable the user's head was during real calibration."""

    accepted_poses = [(-2.0, 1.0, 0.0), (0.0, -1.0, 0.5), (2.0, 3.0, -0.5)]
    accepted_samples = tuple(
        _raw_sample(
            source_frame_id=i,
            accepted=True,
            reason="accepted",
            head_yaw_deg=yaw,
            head_pitch_deg=pitch,
            head_roll_deg=roll,
        )
        for i, (yaw, pitch, roll) in enumerate(accepted_poses)
    )
    wildly_off_rejected = _raw_sample(
        source_frame_id=99,
        accepted=False,
        reason="head_pose_out_of_range",
        head_yaw_deg=89.0,
        head_pitch_deg=-89.0,
        head_roll_deg=45.0,
    )
    result = _result_with_samples((*accepted_samples, wildly_off_rejected))

    spread = result.head_pose_spread()

    assert set(spread) == {
        "median_yaw_deg",
        "median_pitch_deg",
        "median_roll_deg",
        "max_yaw_deviation_deg",
        "max_pitch_deviation_deg",
        "max_roll_deviation_deg",
    }
    # Medians/deviations computed only from the 3 accepted poses above --
    # if the rejected sample leaked in, these values would be very different.
    assert spread["median_yaw_deg"] == pytest.approx(0.0)
    assert spread["median_pitch_deg"] == pytest.approx(1.0)
    assert spread["median_roll_deg"] == pytest.approx(0.0)
    assert spread["max_yaw_deviation_deg"] == pytest.approx(2.0)
    assert spread["max_pitch_deviation_deg"] == pytest.approx(2.0)
    assert spread["max_roll_deviation_deg"] == pytest.approx(0.5)


def test_head_pose_spread_is_empty_when_no_accepted_sample_has_pose_data() -> None:
    samples = (
        _raw_sample(source_frame_id=1, accepted=True, reason="accepted"),  # no pose fields
        _raw_sample(source_frame_id=2, accepted=False, reason="low_confidence"),
    )
    result = _result_with_samples(samples)

    # Must be an empty dict, not zeros -- zeros would falsely claim a
    # perfectly still head when there is no pose data to judge from at all.
    assert result.head_pose_spread() == {}


def test_head_pose_spread_is_empty_for_a_result_with_no_accepted_samples() -> None:
    samples = (
        _raw_sample(
            source_frame_id=1,
            accepted=False,
            reason="low_confidence",
            head_yaw_deg=5.0,
            head_pitch_deg=5.0,
            head_roll_deg=5.0,
        ),
    )
    result = _result_with_samples(samples)

    assert result.head_pose_spread() == {}
