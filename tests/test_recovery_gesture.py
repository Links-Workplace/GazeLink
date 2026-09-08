"""The way out of a freeze, exercised without a camera or a calibration.

Every test here is a safety property.  The detector is the only escape hatch a
user without hands has once gaze is withheld, so the failures that matter are:
reacting to a natural blink (unusable), missing a deliberate hold (trapped),
and firing twice for one gesture (unpredictable).
"""

from __future__ import annotations

import pytest

from gazelink.config import GestureTimingConfig
from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    NormalizedPoint,
    TrackingState,
    VisionObservation,
)
from gazelink.recovery_gesture import (
    EyeCloseDetector,
    RecoveryGesture,
    RecoveryMenu,
    RecoveryOption,
    advance_recovery,
    eyes_unreadable_with_face_present,
    face_is_present,
)

pytestmark = pytest.mark.unit

TIMING = GestureTimingConfig()
BLINK_MS = TIMING.natural_blink_max_ms  # 400
CONFIRM_MS = TIMING.recovery_confirm_ms  # 1500
COOLDOWN_MS = TIMING.cooldown_ms  # 800


def _hold(
    detector: EyeCloseDetector, duration_ms: float, *, start_ms: float = 0.0, step_ms: float = 33.0
) -> list[RecoveryGesture]:
    """Close the eyes for ``duration_ms``, then open, at ~30 fps.

    Returns every gesture emitted, so a test can assert on how many fired as
    well as which -- firing twice for one close is its own bug.
    """

    fired: list[RecoveryGesture] = []
    now = start_ms
    end = start_ms + duration_ms
    while now < end:
        gesture = detector.update(eyes_closed=True, now_monotonic_ms=now)
        if gesture is not None:
            fired.append(gesture)
        now += step_ms
    gesture = detector.update(eyes_closed=False, now_monotonic_ms=end)
    if gesture is not None:
        fired.append(gesture)
    return fired


def _observation(
    *,
    tracking_state: TrackingState = TrackingState.LOW_CONFIDENCE,
    left: EyeFeatures | None = None,
    right: EyeFeatures | None = None,
) -> VisionObservation:
    return VisionObservation(
        frame_id=1,
        observed_at_monotonic_ms=0.0,
        tracking_state=tracking_state,
        face_box=None,
        left_eye=left,
        right_eye=right,
        head_pose=None,
        overall_confidence=0.2,
    )


def _eye() -> EyeFeatures:
    return EyeFeatures(
        iris_center=NormalizedPoint(0.5, 0.5),
        openness=0.3,
        confidence=0.9,
    )


# --------------------------------------------------------------------------
# The signal: what a closed eye actually looks like in this pipeline
# --------------------------------------------------------------------------


def test_closed_eyes_are_a_tracked_face_with_no_valid_eye_geometry() -> None:
    """Closed eyes yield ``None`` eyes, not a small openness.

    ``extract_eye_features`` invalidates once the lid gap hits its epsilon, so
    a detector waiting for ``openness`` near zero would wait forever.
    """

    assert eyes_unreadable_with_face_present(_observation())


def test_low_confidence_does_not_disqualify_the_signal() -> None:
    """Closed eyes are low confidence by construction.

    If the detector demanded TRACKED it would go blind exactly when needed.
    """

    assert eyes_unreadable_with_face_present(
        _observation(tracking_state=TrackingState.LOW_CONFIDENCE)
    )


def test_a_lost_face_is_not_a_close() -> None:
    """Someone who turned away or was occluded has not asked for anything."""

    assert not eyes_unreadable_with_face_present(_observation(tracking_state=TrackingState.LOST))


def test_multiple_faces_is_not_a_close() -> None:
    """A passer-by must not be able to trigger a recalibration."""

    assert not eyes_unreadable_with_face_present(
        _observation(tracking_state=TrackingState.MULTIPLE_FACES)
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [(_eye(), None), (None, _eye()), (_eye(), _eye())],
    ids=["left-open", "right-open", "both-open"],
)
def test_any_readable_eye_means_the_eyes_are_not_shut(
    left: EyeFeatures | None, right: EyeFeatures | None
) -> None:
    assert not eyes_unreadable_with_face_present(_observation(left=left, right=right))


def test_signal_rejects_a_non_observation() -> None:
    with pytest.raises(ContractValidationError):
        eyes_unreadable_with_face_present(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Blink rejection: the difference between a helpful machine and an unusable one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("duration", [0.0, 50.0, 150.0, BLINK_MS - 1])
def test_a_natural_blink_produces_nothing(duration: float) -> None:
    assert _hold(EyeCloseDetector(), duration) == []


def test_a_long_run_of_blinks_never_fires() -> None:
    """Blinking is continuous and involuntary; it must stay silent forever."""

    detector = EyeCloseDetector()
    now = 0.0
    for _ in range(50):
        assert _hold(detector, 150.0, start_ms=now) == []
        now += 3_000.0


# --------------------------------------------------------------------------
# The two gestures
# --------------------------------------------------------------------------


def test_a_deliberate_short_hold_cycles() -> None:
    assert _hold(EyeCloseDetector(), BLINK_MS + 100.0) == [RecoveryGesture.CYCLE]


def test_a_long_hold_confirms_while_the_eyes_are_still_shut() -> None:
    """The user must not have to guess when to open.

    ``CONFIRM`` fires during the hold, so opening afterwards is just release.
    """

    detector = EyeCloseDetector()
    now = 0.0
    fired = None
    while now < CONFIRM_MS + 200.0:
        fired = detector.update(eyes_closed=True, now_monotonic_ms=now)
        if fired is not None:
            break
        now += 33.0

    assert fired is RecoveryGesture.CONFIRM
    assert now >= CONFIRM_MS
    assert detector.is_holding  # still shut: it fired without waiting for release


def test_a_long_hold_fires_confirm_once_and_never_also_cycles() -> None:
    """One close is one instruction."""

    assert _hold(EyeCloseDetector(), CONFIRM_MS + 500.0) == [RecoveryGesture.CONFIRM]


def test_holding_far_past_confirm_does_not_repeat() -> None:
    assert _hold(EyeCloseDetector(), CONFIRM_MS * 4) == [RecoveryGesture.CONFIRM]


# --------------------------------------------------------------------------
# Aborts and cooldown
# --------------------------------------------------------------------------


def test_losing_the_face_mid_hold_aborts_silently() -> None:
    """Turning away is not consent.

    Regression: a lost face and a released hold both arrive as "not closed",
    and collapsing them made turning your head emit ``CYCLE`` -- an unrequested
    action, from the only input channel a frozen session still has.
    """

    detector = EyeCloseDetector()
    now = 0.0
    while now < CONFIRM_MS - 200.0:
        assert detector.update(eyes_closed=True, now_monotonic_ms=now) is None
        now += 33.0

    assert detector.update(eyes_closed=False, face_present=False, now_monotonic_ms=now) is None
    assert not detector.is_holding


def test_a_face_lost_and_regained_starts_a_fresh_hold() -> None:
    """The aborted portion must not count towards the next gesture."""

    detector = EyeCloseDetector()
    detector.update(eyes_closed=True, now_monotonic_ms=0.0)
    detector.update(eyes_closed=False, face_present=False, now_monotonic_ms=CONFIRM_MS)

    # Closing again now must need the full duration from this moment on.
    assert detector.update(eyes_closed=True, now_monotonic_ms=CONFIRM_MS + 10.0) is None
    assert detector.progress(CONFIRM_MS + 10.0) == pytest.approx(0.0)


def test_reset_discards_a_hold_in_progress() -> None:
    detector = EyeCloseDetector()
    detector.update(eyes_closed=True, now_monotonic_ms=0.0)
    assert detector.is_holding

    detector.reset()
    assert not detector.is_holding
    assert detector.update(eyes_closed=False, now_monotonic_ms=CONFIRM_MS) is None


def test_cooldown_stops_one_close_becoming_a_burst() -> None:
    """A second gesture immediately after the first is not accepted."""

    detector = EyeCloseDetector()
    assert _hold(detector, BLINK_MS + 100.0) == [RecoveryGesture.CYCLE]

    # Starting again inside the cooldown must not produce a gesture.
    end = BLINK_MS + 100.0
    assert _hold(detector, BLINK_MS + 100.0, start_ms=end + 10.0) == []


def test_a_gesture_is_accepted_again_after_the_cooldown() -> None:
    detector = EyeCloseDetector()
    assert _hold(detector, BLINK_MS + 100.0) == [RecoveryGesture.CYCLE]

    later = BLINK_MS + 100.0 + COOLDOWN_MS + 50.0
    assert _hold(detector, BLINK_MS + 100.0, start_ms=later) == [RecoveryGesture.CYCLE]


# --------------------------------------------------------------------------
# Progress, and contract validation
# --------------------------------------------------------------------------


def test_progress_is_zero_when_nothing_is_held() -> None:
    assert EyeCloseDetector().progress(1_000.0) == 0.0


def test_progress_rises_to_one_and_is_clamped() -> None:
    detector = EyeCloseDetector()
    detector.update(eyes_closed=True, now_monotonic_ms=0.0)

    assert detector.progress(0.0) == pytest.approx(0.0)
    assert detector.progress(CONFIRM_MS / 2) == pytest.approx(0.5)
    assert detector.progress(CONFIRM_MS) == pytest.approx(1.0)
    assert detector.progress(CONFIRM_MS * 10) == pytest.approx(1.0)


def test_detector_rejects_bad_timing_and_bad_clock() -> None:
    with pytest.raises(ContractValidationError):
        EyeCloseDetector(timing=object())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError):
        EyeCloseDetector().update(eyes_closed=True, now_monotonic_ms="soon")  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError):
        EyeCloseDetector().update(eyes_closed=True, now_monotonic_ms=float("inf"))


def test_thresholds_come_from_configuration_not_constants() -> None:
    """A deployment must be able to retune these without editing code."""

    patient = GestureTimingConfig(
        natural_blink_max_ms=200,
        intentional_hold_min_ms=400,
        recovery_confirm_ms=3_000,
    )
    detector = EyeCloseDetector(patient)

    assert _hold(detector, 250.0) == [RecoveryGesture.CYCLE]  # would be a blink by default
    assert _hold(detector, CONFIRM_MS + 100.0, start_ms=10_000.0) == [RecoveryGesture.CYCLE]


# --------------------------------------------------------------------------
# RecoveryMenu: choosing without pointing at anything
# --------------------------------------------------------------------------


def _menu() -> RecoveryMenu:
    return RecoveryMenu(
        (
            RecoveryOption("recalibrate", "Calibrate for this screen"),
            RecoveryOption("exit", "Close GAZELINK"),
        )
    )


def test_the_highlight_starts_on_the_least_destructive_option() -> None:
    """A stray long close must not do something drastic."""

    assert _menu().highlighted.key == "recalibrate"


def test_a_short_close_moves_the_highlight_and_wraps() -> None:
    menu = _menu()
    assert menu.apply(RecoveryGesture.CYCLE) is None
    assert menu.highlighted.key == "exit"

    menu.apply(RecoveryGesture.CYCLE)
    assert menu.highlighted.key == "recalibrate"


def test_a_long_close_takes_the_highlighted_option() -> None:
    menu = _menu()
    chosen = menu.apply(RecoveryGesture.CONFIRM)
    assert chosen is not None
    assert chosen.key == "recalibrate"


def test_cycling_then_confirming_takes_the_second_option() -> None:
    """The whole point: reach any option using eyelids alone."""

    menu = _menu()
    menu.apply(RecoveryGesture.CYCLE)
    chosen = menu.apply(RecoveryGesture.CONFIRM)
    assert chosen is not None
    assert chosen.key == "exit"


def test_the_menu_marks_which_option_is_highlighted() -> None:
    menu = _menu()
    assert "[ Calibrate for this screen ]" in menu.describe()

    menu.cycle()
    assert "[ Close GAZELINK ]" in menu.describe()


def test_a_menu_needs_at_least_two_options_and_unique_keys() -> None:
    with pytest.raises(ContractValidationError):
        RecoveryMenu((RecoveryOption("only", "Only"),))
    with pytest.raises(ContractValidationError):
        RecoveryMenu((RecoveryOption("a", "A"), RecoveryOption("a", "Also A")))
    with pytest.raises(ContractValidationError):
        RecoveryMenu((object(), object()))  # type: ignore[arg-type]


def test_an_option_needs_real_text() -> None:
    with pytest.raises(ContractValidationError):
        RecoveryOption("", "Label")
    with pytest.raises(ContractValidationError):
        RecoveryOption("key", "   ")


def test_menu_rejects_something_that_is_not_a_gesture() -> None:
    with pytest.raises(ContractValidationError):
        _menu().apply("CONFIRM")  # type: ignore[arg-type]


def test_the_live_window_wires_the_gesture_to_a_real_action() -> None:
    """A detected gesture that does nothing is not recovery."""

    import inspect

    from gazelink import gaze_window

    source = inspect.getsource(gaze_window)
    assert "EyeCloseDetector()" in source
    assert "RecoveryMenu(" in source
    assert "advance_recovery(" in source
    assert "self._recovery_menu.apply(gesture)" in source
    assert "self._act_on_recovery(" in source
    # The raw observation, never the vetted one: closed eyes are rejected by
    # the confidence policy, so the accepted observation is None exactly here.
    assert "tick.observation" in source


# --------------------------------------------------------------------------
# Regressions in the wiring itself
# --------------------------------------------------------------------------


def test_a_second_face_mid_hold_cannot_fire_a_gesture() -> None:
    """A passer-by must not be able to press a button with their face.

    Goes through :func:`advance_recovery` -- the function the window actually
    calls -- because the bug was in *composing* the two checks, not in either
    of them.  An earlier version of this test assembled the call itself and so
    stayed green while the real wiring was broken.
    """

    detector = EyeCloseDetector()
    now = 0.0
    while now < CONFIRM_MS - 200.0:
        advance_recovery(detector, _observation(), now_monotonic_ms=now)
        now += 33.0

    gesture = advance_recovery(
        detector,
        _observation(tracking_state=TrackingState.MULTIPLE_FACES),
        now_monotonic_ms=now,
    )

    assert gesture is None
    assert not detector.is_holding


def test_advance_recovery_still_reports_a_real_gesture() -> None:
    """The fix must not have bought safety by never firing."""

    detector = EyeCloseDetector()
    now = 0.0
    while now < BLINK_MS + 100.0:
        assert advance_recovery(detector, _observation(), now_monotonic_ms=now) is None
        now += 33.0

    open_eyed = _observation(tracking_state=TrackingState.TRACKED, left=_eye(), right=_eye())
    assert advance_recovery(detector, open_eyed, now_monotonic_ms=now) is RecoveryGesture.CYCLE


def test_advance_recovery_rejects_a_non_detector() -> None:
    with pytest.raises(ContractValidationError):
        advance_recovery(object(), _observation(), now_monotonic_ms=0.0)  # type: ignore[arg-type]


def test_the_two_face_present_checks_are_one_definition() -> None:
    """They disagreed once; nothing should let them drift apart again."""

    for state in TrackingState:
        observation = _observation(tracking_state=state)
        if not face_is_present(observation):
            assert not eyes_unreadable_with_face_present(observation)


def test_choosing_recalibrate_actually_starts_a_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An option that announces a recalibration must perform one.

    Drives the real CLI dispatch with both screens faked, so it fails if the
    caller goes back to simply returning the exit code.  The previous version
    scanned the module source for a name that appears elsewhere in the file
    anyway, and so passed while the wiring was broken.
    """

    import gazelink.calibration_window as calibration_window
    import gazelink.gaze_window as gaze_window
    from gazelink.app import cli

    calls: list[str] = []

    def _fake_gaze_check(**_kwargs: object) -> int:
        calls.append("gaze_check")
        return gaze_window.RECALIBRATE_REQUESTED_EXIT_CODE

    def _fake_calibration(**_kwargs: object) -> int:
        calls.append("calibration")
        return 0

    monkeypatch.setattr(gaze_window, "run_gaze_check", _fake_gaze_check)
    monkeypatch.setattr(calibration_window, "run_guided_calibration", _fake_calibration)

    assert cli(["--gaze-check"]) == 0
    assert calls == ["gaze_check", "calibration"]


def test_an_ordinary_exit_does_not_start_a_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the recalibration code chains; a normal close must just close."""

    import gazelink.calibration_window as calibration_window
    import gazelink.gaze_window as gaze_window
    from gazelink.app import cli

    calls: list[str] = []

    monkeypatch.setattr(
        gaze_window, "run_gaze_check", lambda **_k: (calls.append("gaze_check"), 0)[1]
    )
    monkeypatch.setattr(
        calibration_window,
        "run_guided_calibration",
        lambda **_k: (calls.append("calibration"), 0)[1],
    )

    assert cli(["--gaze-check"]) == 0
    assert calls == ["gaze_check"]


def test_recalibration_targets_the_display_the_request_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The screen name must survive the hand-off from gaze check to calibration.

    Regression: the choice reached calibration but the *display* did not, so a
    user who had dragged the window to their second monitor got a calibration
    for the first one -- and was told it had worked.
    """

    import gazelink.calibration_window as calibration_window
    import gazelink.gaze_window as gaze_window
    from gazelink.app import cli

    seen: dict[str, object] = {}

    def _fake_gaze_check(**kwargs: object) -> int:
        callback = kwargs["on_recalibration_request"]
        assert callable(callback)
        callback("DISPLAY2")
        return gaze_window.RECALIBRATE_REQUESTED_EXIT_CODE

    def _fake_calibration(**kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(gaze_window, "run_gaze_check", _fake_gaze_check)
    monkeypatch.setattr(calibration_window, "run_guided_calibration", _fake_calibration)

    assert cli(["--gaze-check"]) == 0
    assert seen["screen_name"] == "DISPLAY2"
