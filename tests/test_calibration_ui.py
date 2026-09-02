"""Deterministic tests for the Qt-free guided-calibration controller."""

from __future__ import annotations

import pytest

from gazelink.calibration import CalibrationSession, CalibrationSessionState
from gazelink.calibration_ui import (
    CalibrationCollectionPhase,
    CalibrationQualitySettings,
    CalibrationTimingSettings,
    CalibrationUiController,
)
from gazelink.confidence import HeadPoseLimits
from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    HeadPose,
    NormalizedBox,
    NormalizedPoint,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.vision import STRUCTURAL_CONFIDENCE

# `_observation`'s default `observed_at_monotonic_ms` and `FakeClock`'s
# starting value are pinned to the SAME constant on purpose. The controller's
# stale-sample gate computes `clock_ms() - observation.observed_at_monotonic_ms`;
# if the clock and the observation defaults ever drifted apart (e.g. a clock
# that starts at 1_000.0 while observations default to ~1.0, as an earlier
# version of this fixture did), every "happy path" test below would start
# failing the NEW stale gate for the wrong reason -- not because the
# controller is broken, but because the fixture accidentally manufactured a
# ~999ms-old sample. Pin both to this one constant so that class of drift
# cannot recur unnoticed; a test that specifically wants an old sample must
# pass an explicit, older `observed_at_monotonic_ms`.
_CLOCK_START_MS = 1_000.0

# Module-level singleton so `_observation`'s default iris position isn't a
# function call evaluated in the argument-default list (ruff B008); frozen
# NormalizedPoint is safe to share across every call that doesn't override it.
_DEFAULT_IRIS_IN_EYE = NormalizedPoint(0.5, 0.5)


class FakeClock:
    def __init__(self) -> None:
        self.now_ms = _CLOCK_START_MS

    def __call__(self) -> float:
        return self.now_ms


def _controller(
    *,
    order: tuple[int, ...] | None = None,
    quality: CalibrationQualitySettings | None = None,
    timing: CalibrationTimingSettings | None = None,
    min_samples_per_target: int = 2,
) -> CalibrationUiController:
    clock = FakeClock()
    return CalibrationUiController(
        CalibrationSession(
            camera_id="camera-0",
            screen_geometry=ScreenGeometry("primary", 1920, 1080, 1.0),
            target_order=order,
            min_samples_per_target=min_samples_per_target,
            clock_ms=clock,
        ),
        clock_ms=clock,
        quality=quality if quality is not None else CalibrationQualitySettings(),
        timing=timing if timing is not None else CalibrationTimingSettings(0.0, 0.0, 0.0),
    )


def _observation(
    frame_id: int,
    *,
    state: TrackingState = TrackingState.TRACKED,
    confidence: float = STRUCTURAL_CONFIDENCE,
    observed_at_monotonic_ms: float = _CLOCK_START_MS,
    left_iris_in_eye: NormalizedPoint | None = _DEFAULT_IRIS_IN_EYE,
    right_iris_in_eye: NormalizedPoint | None = _DEFAULT_IRIS_IN_EYE,
    left_openness: float = 0.5,
    right_openness: float = 0.5,
    left_eye_confidence: float = 0.5,
    right_eye_confidence: float = 0.5,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> VisionObservation:
    """Build a fake observation whose default confidence matches the real
    adapter's fixed ``STRUCTURAL_CONFIDENCE`` (0.50), not an arbitrary value.

    A prior version of this fixture defaulted to 0.9 -- a confidence the real
    ``FaceLandmarkerAdapter`` can never produce -- which let every "happy
    path" test here pass while the real UI silently rejected every sample in
    production. Keep this pinned to the real constant so this class of drift
    cannot happen again unnoticed.

    ``observed_at_monotonic_ms`` defaults to ``_CLOCK_START_MS`` -- the same
    constant ``FakeClock`` starts at -- so a default-built observation is
    exactly 0ms old and never spuriously trips the stale-sample gate; a test
    exercising that gate passes an explicitly older value instead.
    """
    tracked = state is TrackingState.TRACKED
    return VisionObservation(
        frame_id=frame_id,
        observed_at_monotonic_ms=observed_at_monotonic_ms,
        tracking_state=state,
        face_box=NormalizedBox(0.2, 0.2, 0.4, 0.4) if tracked else None,
        left_eye=EyeFeatures(
            NormalizedPoint(0.35, 0.4), left_openness, left_eye_confidence, left_iris_in_eye
        )
        if tracked
        else None,
        right_eye=EyeFeatures(
            NormalizedPoint(0.55, 0.4), right_openness, right_eye_confidence, right_iris_in_eye
        )
        if tracked
        else None,
        head_pose=HeadPose(yaw_deg, pitch_deg, roll_deg) if tracked else None,
        overall_confidence=confidence,
    )


def test_target_waits_for_stabilization_and_samples_across_capture_window() -> None:
    clock = FakeClock()
    controller = CalibrationUiController(
        CalibrationSession(
            camera_id="camera-0",
            screen_geometry=ScreenGeometry("primary", 1920, 1080, 1.0),
            min_samples_per_target=2,
            clock_ms=clock,
        ),
        clock_ms=clock,
        timing=CalibrationTimingSettings(
            stabilization_ms=100.0,
            capture_window_ms=200.0,
            min_sample_interval_ms=50.0,
        ),
    )

    first = controller.ingest(_observation(1, observed_at_monotonic_ms=clock.now_ms))
    assert first.phase is CalibrationCollectionPhase.STABILIZING
    assert first.accepted_samples == 0
    assert not first.sample_evaluated

    clock.now_ms = 1_100.0
    collecting = controller.ingest(_observation(2, observed_at_monotonic_ms=clock.now_ms))
    assert collecting.phase is CalibrationCollectionPhase.COLLECTING
    assert collecting.accepted_samples == 1
    assert not collecting.sample_evaluated

    clock.now_ms = 1_120.0
    too_close = controller.ingest(_observation(3, observed_at_monotonic_ms=clock.now_ms))
    assert not too_close.sample_accepted
    assert not too_close.sample_evaluated
    assert too_close.accepted_samples == 1

    clock.now_ms = 1_150.0
    enough_samples_too_early = controller.ingest(
        _observation(4, observed_at_monotonic_ms=clock.now_ms)
    )
    assert enough_samples_too_early.target_number == 1
    assert enough_samples_too_early.accepted_samples == 2

    clock.now_ms = 1_300.0
    advanced = controller.ingest(_observation(5, observed_at_monotonic_ms=clock.now_ms))
    assert advanced.target_number == 2
    assert advanced.phase is CalibrationCollectionPhase.STABILIZING


def test_good_unique_observations_fill_a_target_and_auto_advance() -> None:
    controller = _controller()

    first = controller.ingest(_observation(1))
    second = controller.ingest(_observation(2))

    assert first.accepted_samples == 1
    assert second.target_number == 2
    assert second.accepted_samples == 0
    assert "Target complete" in second.feedback


def test_low_confidence_never_counts_and_explains_why() -> None:
    controller = _controller()

    view = controller.ingest(_observation(1, confidence=STRUCTURAL_CONFIDENCE - 0.1))

    assert view.accepted_samples == 0
    assert "too low" in view.feedback


def test_the_real_adapters_fixed_confidence_is_sufficient_to_calibrate() -> None:
    """Pins the actual bug this session found: gating on the M2 `.calibration`
    threshold (0.70) instead of `.display` (0.50) made every real sample fail
    unconditionally, since `vision.STRUCTURAL_CONFIDENCE` is always exactly
    0.50. A default observation here must be accepted.
    """

    controller = _controller()

    view = controller.ingest(_observation(1))

    assert view.accepted_samples == 1


def test_tracked_observation_with_blanked_geometry_is_rejected_not_accepted() -> None:
    """Pins a real bug found by reading a user's shared session log: 4 of 5
    "accepted" samples for one target had every feature field `None`.

    ``VisionRuntime``'s loss-debounce window can report
    ``tracking_state=TRACKED`` with ``left_eye``/``right_eye``/``head_pose``
    all blanked to ``None`` -- while leaving ``overall_confidence`` untouched,
    since it comes from a still-good raw observation the confidence *policy*
    (not the vision adapter) rejected for an unrelated reason (e.g. a
    momentary head-pose excursion). Before this fix, such a tick passed both
    of `ingest`'s existing gates (tracking_state, confidence) and was
    recorded as ``accepted=True`` with nothing usable inside it.
    """

    controller = _controller()
    # Fresh stamp (matches `_CLOCK_START_MS`/`FakeClock`'s starting value) is
    # required here: a stale stamp would make this observation get rejected
    # by the newer stale-sample gate instead of the features-unavailable gate
    # this test exists to pin, silently destroying its regression value.
    blanked = VisionObservation(
        frame_id=1,
        observed_at_monotonic_ms=_CLOCK_START_MS,
        tracking_state=TrackingState.TRACKED,
        face_box=None,
        left_eye=None,
        right_eye=None,
        head_pose=None,
        overall_confidence=STRUCTURAL_CONFIDENCE,
    )

    view = controller.ingest(blanked)

    assert view.sample_accepted is False
    assert view.accepted_samples == 0
    assert "not ready" in view.feedback

    # Drive the session to completion, with the blanked tick already recorded
    # as the first sample, so we can inspect what actually got stored.
    _complete_via_ingest(controller)
    stored = next(s for s in controller.result().samples if s.source_frame_id == 1)
    assert stored.accepted is False
    assert stored.reason == "features_unavailable"


def test_lost_tracking_never_counts_and_explains_recovery() -> None:
    controller = _controller()

    view = controller.ingest(_observation(1, state=TrackingState.LOST, confidence=0.0))

    assert view.accepted_samples == 0
    assert "face visible" in view.feedback


def test_same_frame_cannot_be_counted_twice() -> None:
    controller = _controller()
    observation = _observation(1)

    controller.ingest(observation)
    view = controller.ingest(observation)

    assert view.accepted_samples == 1
    assert "new camera frame" in view.feedback


def test_configured_target_order_is_reflected_in_progress() -> None:
    controller = _controller(order=(8, 0, 4, 1, 2, 3, 5, 6, 7))

    view = controller.view()

    assert view.target is not None
    assert view.target.index == 8
    assert view.target_number == 1


def test_retry_discards_only_the_current_targets_samples() -> None:
    controller = _controller()
    controller.ingest(_observation(1))

    view = controller.retry()

    assert view.target_number == 1
    assert view.accepted_samples == 0
    assert "reset" in view.feedback


def test_restart_discards_all_prior_samples_and_returns_to_first_target() -> None:
    controller = _controller()
    controller.ingest(_observation(1))
    controller.ingest(_observation(2))
    assert controller.view().target_number == 2

    view = controller.restart()

    assert view.target_number == 1
    assert view.accepted_samples == 0


def test_cancel_prevents_result_export_and_future_ingestion() -> None:
    controller = _controller()
    controller.ingest(_observation(1))

    view = controller.cancel()
    after = controller.ingest(_observation(2))

    assert view.cancelled is True
    assert after.cancelled is True
    with pytest.raises(ContractValidationError, match="COMPLETE"):
        controller.result()


def test_complete_session_requires_quality_samples_for_every_target() -> None:
    controller = _controller()
    frame_id = 1
    while controller.session.state is CalibrationSessionState.COLLECTING:
        controller.ingest(_observation(frame_id))
        frame_id += 1

    view = controller.view()

    assert view.complete is True
    assert controller.result().sample_counts == (2,) * 9
    # Pins the exact regression a user reported live: the view previously
    # hardcoded accepted_samples=0 once the session left COLLECTING, so a
    # completed session showed "Calibration complete" together with "0/2"
    # even though every sample had genuinely been captured.
    assert view.accepted_samples == 18  # 9 targets * 2 samples each
    assert view.required_samples == 18


def test_view_reflects_real_totals_repeatedly_after_completion() -> None:
    """Reading the view multiple times after completion must not drift or
    raise -- `result()` is read-only, so repeated UI refreshes on the
    "complete" screen must keep showing the same real totals."""

    controller = _controller()
    frame_id = 1
    while controller.session.state is CalibrationSessionState.COLLECTING:
        controller.ingest(_observation(frame_id))
        frame_id += 1

    first_view = controller.view()
    second_view = controller.view()

    assert first_view.accepted_samples == second_view.accepted_samples == 18


def test_invalid_controller_arguments_are_rejected() -> None:
    with pytest.raises(ContractValidationError, match="session"):
        CalibrationUiController(object())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError, match="calibration_confidence"):
        CalibrationUiController(_controller().session, calibration_confidence=1.1)


def test_sample_accepted_is_true_only_when_a_stable_window_is_committed() -> None:
    """Provisional candidates are not reported as final acceptance."""

    controller = _controller()

    good = controller.ingest(_observation(1))
    low_confidence = controller.ingest(_observation(2, confidence=STRUCTURAL_CONFIDENCE - 0.1))
    not_tracked = controller.ingest(_observation(3, state=TrackingState.LOST, confidence=0.0))
    duplicate = controller.ingest(_observation(3, state=TrackingState.LOST, confidence=0.0))

    assert good.sample_accepted is False
    assert good.sample_evaluated is False
    assert low_confidence.sample_accepted is False
    assert not_tracked.sample_accepted is False
    assert duplicate.sample_accepted is False


def test_view_retry_cancel_and_restart_never_report_a_sample_as_accepted() -> None:
    """Only `ingest` evaluates an observation; every other call must report
    `sample_accepted=False` even immediately after a real sample was counted."""

    controller = _controller()
    controller.ingest(_observation(1))

    assert controller.view().sample_accepted is False
    assert controller.retry().sample_accepted is False
    assert controller.restart().sample_accepted is False
    assert controller.cancel().sample_accepted is False


def _complete_via_ingest(
    controller: CalibrationUiController, *, first_observation: object = None
) -> None:
    """Drive a controller's session to COMPLETE using only `ingest`.

    ``first_observation``, when given, is ingested before the generic filler
    observations -- used to plant one specific (e.g. rejected) sample as the
    first entry for target 0 while still reaching a real, exportable result.
    """

    frame_id = 1
    if first_observation is not None:
        controller.ingest(first_observation)  # type: ignore[arg-type]
        frame_id += 1
    while controller.session.state is CalibrationSessionState.COLLECTING:
        controller.ingest(_observation(frame_id))
        frame_id += 1


def test_ingest_stores_real_iris_openness_and_head_pose_on_the_sample() -> None:
    """M2-02 Stage A: `ingest` must build a real dataset entry, not just a
    count -- this is the exact gap the user's shared session log surfaced."""

    controller = _controller()
    _complete_via_ingest(controller)

    sample = next(s for s in controller.result().samples if s.accepted)

    assert sample.left_iris_in_eye == NormalizedPoint(0.5, 0.5)
    assert sample.right_iris_in_eye == NormalizedPoint(0.5, 0.5)
    assert sample.left_openness == pytest.approx(0.5)
    assert sample.right_openness == pytest.approx(0.5)
    assert sample.head_yaw_deg == pytest.approx(0.0)
    assert sample.head_pitch_deg == pytest.approx(0.0)
    assert sample.head_roll_deg == pytest.approx(0.0)
    assert sample.confidence == pytest.approx(STRUCTURAL_CONFIDENCE)


def test_a_rejected_lost_observation_is_stored_with_no_fabricated_features() -> None:
    controller = _controller()
    lost = _observation(1, state=TrackingState.LOST, confidence=0.0)
    _complete_via_ingest(controller, first_observation=lost)

    rejected = next(s for s in controller.result().samples if not s.accepted)

    assert rejected.left_iris_in_eye is None
    assert rejected.right_iris_in_eye is None
    assert rejected.head_yaw_deg is None
    assert rejected.reason == "tracking_not_ready"


# --- M2-02 Stage B: ingest-time quality gates -------------------------------


def test_stale_sample_is_rejected_with_actionable_feedback_and_reason() -> None:
    """An observation older than `max_sample_age_ms` must be rejected -- a
    stale sample's geometry no longer reflects what the user is doing right
    now, and training on it would poison the dataset with an unrelated
    glance."""

    controller = _controller()
    stale = _observation(1, observed_at_monotonic_ms=_CLOCK_START_MS - 501.0)

    view = controller.ingest(stale)

    assert view.sample_accepted is False
    assert view.accepted_samples == 0
    assert "too late" in view.feedback

    _complete_via_ingest(controller)
    stored = next(s for s in controller.result().samples if s.source_frame_id == 1)
    assert stored.accepted is False
    assert stored.reason == "stale_sample"


def test_sample_exactly_at_the_max_age_boundary_is_still_accepted() -> None:
    """Pins the `>` (not `>=`) boundary on the stale-sample gate: an age
    exactly equal to `max_sample_age_ms` must still count as fresh."""

    quality = CalibrationQualitySettings(max_sample_age_ms=500.0)
    controller = _controller(quality=quality)
    boundary = _observation(1, observed_at_monotonic_ms=_CLOCK_START_MS - 500.0)

    view = controller.ingest(boundary)

    assert view.sample_accepted is False
    assert view.sample_evaluated is False
    assert view.accepted_samples == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_openness": 0.05},
        {"right_openness": 0.05},
        {"left_iris_in_eye": None},
        {"right_iris_in_eye": None},
    ],
)
def test_eye_not_visible_rejects_low_openness_or_missing_iris_independently(
    kwargs: dict[str, object],
) -> None:
    """Each real per-eye signal -- openness below the floor, or a missing
    iris position despite a present `EyeFeatures` object -- must reject on
    its own, for either eye. This is exactly the case a bare `is None` check
    on the whole eye object can never catch, since `EyeFeatures` itself is
    present in every one of these cases; only a field inside it is bad."""

    controller = _controller()

    view = controller.ingest(_observation(1, **kwargs))  # type: ignore[arg-type]

    assert view.sample_accepted is False
    assert view.accepted_samples == 0
    assert "eyes open and visible" in view.feedback

    _complete_via_ingest(controller)
    stored = next(s for s in controller.result().samples if s.source_frame_id == 1)
    assert stored.reason == "eye_not_visible"


def test_low_per_eye_confidence_alone_never_rejects_a_sample() -> None:
    """Pins the "do not threshold a structural constant" regression:
    `EyeFeatures.confidence` mirrors `vision.STRUCTURAL_CONFIDENCE`, a fixed
    0.50 the real adapter can never exceed -- it is not a real per-eye
    signal yet. Gating on it would silently reject every real sample, the
    same class of bug
    `test_the_real_adapters_fixed_confidence_is_sufficient_to_calibrate`
    pins for `overall_confidence`. Good openness and a present iris must be
    accepted regardless of how low per-eye `confidence` reads."""

    controller = _controller()

    first = controller.ingest(_observation(1, left_eye_confidence=0.0, right_eye_confidence=0.0))
    view = controller.ingest(_observation(2, left_eye_confidence=0.0, right_eye_confidence=0.0))

    assert first.sample_evaluated is False
    assert view.sample_accepted is True
    assert len(controller.session.accepted_samples) == 2


def test_absolute_head_pose_uses_calibrations_generous_limits_not_live_controls() -> None:
    """A pose that WOULD be rejected by `confidence.HeadPoseLimits()`'s
    live-control defaults (yaw 30 deg) must still be accepted here, and a
    pose beyond calibration's own, more generous limit (yaw 55 deg) must be
    rejected -- proving the generous, calibration-specific limits are
    actually wired in, not the live-control ones."""

    live_control_defaults = HeadPoseLimits()
    assert live_control_defaults.max_abs_yaw_deg < 45.0  # sanity: would fail live control

    within_controller = _controller()
    within_controller.ingest(_observation(1, yaw_deg=45.0))
    within_calibration_limits = within_controller.ingest(_observation(2, yaw_deg=45.0))
    assert within_calibration_limits.sample_accepted is True

    controller = _controller()
    beyond_calibration_limits = controller.ingest(_observation(1, yaw_deg=60.0))
    assert beyond_calibration_limits.sample_accepted is False
    assert "screen more directly" in beyond_calibration_limits.feedback

    _complete_via_ingest(controller)
    stored = next(s for s in controller.result().samples if s.source_frame_id == 1)
    assert stored.reason == "head_pose_out_of_range"


def test_noisy_first_window_is_rejected_without_poisoning_the_target() -> None:
    controller = _controller(min_samples_per_target=3)
    for frame_id, value in enumerate((0.10, 0.50, 0.90), start=1):
        view = controller.ingest(
            _observation(
                frame_id,
                left_iris_in_eye=NormalizedPoint(value, value),
                right_iris_in_eye=NormalizedPoint(value, value),
            )
        )

    assert view.target_number == 1
    assert view.phase is CalibrationCollectionPhase.STABILIZING
    assert view.accepted_samples == 0
    assert not view.sample_accepted
    assert view.sample_evaluated
    assert "too noisy" in view.feedback


def test_best_stable_contiguous_suffix_is_committed_and_noisy_prefix_rejected() -> None:
    clock = FakeClock()
    controller = CalibrationUiController(
        CalibrationSession(
            camera_id="camera-0",
            screen_geometry=ScreenGeometry("primary", 1920, 1080, 1.0),
            min_samples_per_target=3,
            clock_ms=clock,
        ),
        clock_ms=clock,
        timing=CalibrationTimingSettings(0.0, 50.0, 0.0),
    )
    points = (0.10, 0.90, 0.500, 0.501, 0.499)
    for frame_id, value in enumerate(points, start=1):
        if frame_id == len(points):
            clock.now_ms += 50.0
        view = controller.ingest(
            _observation(
                frame_id,
                observed_at_monotonic_ms=clock.now_ms,
                left_iris_in_eye=NormalizedPoint(value, value),
                right_iris_in_eye=NormalizedPoint(value, value),
            )
        )

    assert view.sample_accepted
    assert view.target_number == 2
    stored = controller.session.accepted_samples
    assert tuple(sample.source_frame_id for sample in stored) == (3, 4, 5)


def test_retry_discards_an_uncommitted_candidate_window() -> None:
    controller = _controller(min_samples_per_target=3)
    controller.ingest(_observation(1))
    assert controller.view().accepted_samples == 1

    retried = controller.retry()

    assert retried.accepted_samples == 0
    assert controller.session.accepted_samples == ()


def test_short_term_head_pose_motion_retries_the_target_window() -> None:
    """Settled poses may differ by target, but motion inside one window is noise."""

    controller = _controller()

    first = controller.ingest(_observation(1, yaw_deg=-50.0))
    second = controller.ingest(_observation(2, yaw_deg=50.0))

    assert not first.sample_evaluated
    assert not second.sample_accepted
    assert second.sample_evaluated
    assert second.phase is CalibrationCollectionPhase.STABILIZING
    assert second.accepted_samples == 0
    assert "too noisy" in second.feedback


def test_result_reason_counts_matches_exactly_what_was_ingested() -> None:
    """`CalibrationSessionResult.reason_counts()` must be a faithful tally of
    every stored sample's reason, accepted and rejected alike -- a test that
    would fail if a gate silently stopped recording its rejections."""

    controller = _controller()
    controller.ingest(_observation(1, state=TrackingState.LOST, confidence=0.0))
    controller.ingest(_observation(2, confidence=STRUCTURAL_CONFIDENCE - 0.1))
    controller.ingest(_observation(3, left_openness=0.0))
    frame_id = 4
    while controller.session.state is CalibrationSessionState.COLLECTING:
        controller.ingest(_observation(frame_id))
        frame_id += 1

    counts = controller.result().reason_counts()

    assert counts["tracking_not_ready"] == 1
    assert counts["low_confidence"] == 1
    assert counts["eye_not_visible"] == 1
    assert counts["accepted"] == 18


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_sample_age_ms", 0.0),
        ("max_sample_age_ms", -1.0),
        ("max_sample_age_ms", float("inf")),
        ("max_sample_age_ms", "500"),
        ("head_pose_limits", object()),
        ("min_eye_openness", -0.1),
        ("min_eye_openness", 1.1),
    ],
)
def test_calibration_quality_settings_rejects_invalid_field_values(
    field: str, value: object
) -> None:
    with pytest.raises(ContractValidationError, match=field):
        CalibrationQualitySettings(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_stable_eye_p95", 0.0),
        ("max_stable_eye_p95", float("inf")),
        ("max_stable_head_pose_p95_deg", 0.0),
        ("max_stable_head_pose_p95_deg", -1.0),
    ],
)
def test_calibration_timing_settings_rejects_invalid_stability_thresholds(
    field: str, value: object
) -> None:
    with pytest.raises(ContractValidationError, match=field):
        CalibrationTimingSettings(**{field: value})  # type: ignore[arg-type]


def test_controller_rejects_a_non_calibration_quality_settings_value() -> None:
    with pytest.raises(ContractValidationError, match="quality"):
        CalibrationUiController(_controller().session, quality=object())  # type: ignore[arg-type]
