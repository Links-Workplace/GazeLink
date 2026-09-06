"""The phase machine, the freeze verification, and the file a run produces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    GazeSample,
    PixelPoint,
    ScreenGeometry,
    TrackingState,
)
from gazelink.eyegestures_run import (
    ArrivalTimer,
    FreezeGate,
    MeasurementPhase,
    PhaseMachine,
    build_predictions_payload,
    model_stability,
    sample_from_gaze,
    write_predictions,
)
from gazelink.live_validation import LiveValidationTarget

pytestmark = pytest.mark.unit

_GEOMETRY = ScreenGeometry("ultrawide", 4096, 1152, 1.25)


def _gaze(point: GazePoint, *, frame_id: int = 1, confidence: float = 0.5) -> GazeSample:
    return GazeSample(
        source_frame_id=frame_id,
        sampled_at_monotonic_ms=1000.0,
        raw_normalized=point,
        corrected_normalized=point,
        filtered_normalized=point,
        screen_position=PixelPoint(0, 0),
        screen_id=_GEOMETRY.screen_id,
        confidence=confidence,
        valid_for_control=False,
        reason_codes=(),
    )


# --- the freeze gate --------------------------------------------------------


def test_the_gate_waits_while_training_threads_are_still_running() -> None:
    gate = FreezeGate(settle_floor_ms=1500.0, timeout_ms=10_000.0)
    gate.start(0.0, (1.0, 2.0))

    status = gate.update(2000.0, pending_threads=2, fingerprint=(1.0, 2.0))

    assert status.ready is False
    assert "threads still running: 2" in status.detail


def test_the_gate_waits_for_the_settle_floor_even_with_no_threads() -> None:
    """The library's own 20-frame average needs time the threads do not."""

    gate = FreezeGate(settle_floor_ms=1500.0, timeout_ms=10_000.0)
    gate.start(0.0, (1.0, 2.0))

    assert gate.update(500.0, pending_threads=0, fingerprint=(1.0, 2.0)).ready is False
    assert gate.update(1500.0, pending_threads=0, fingerprint=(1.0, 2.0)).ready is True


def test_a_quiet_unchanged_model_is_the_only_verified_outcome() -> None:
    gate = FreezeGate(settle_floor_ms=100.0, timeout_ms=10_000.0)
    gate.start(0.0, (1.0, 2.0))

    status = gate.update(200.0, pending_threads=0, fingerprint=(1.0, 2.0))

    assert (status.ready, status.verified) == (True, True)


def test_a_fit_landing_while_settling_does_not_invalidate_the_run() -> None:
    """Settling ends only when no thread is running, so a landing is expected.

    An earlier version compared against the fingerprint taken at the moment of
    freezing and rejected runs for it. That was too strict: it disqualified
    runs whose model was completely stable by the time anything was recorded.
    What must not move is the model DURING measurement, checked separately by
    model_stability.
    """

    gate = FreezeGate(settle_floor_ms=100.0, timeout_ms=10_000.0)
    gate.start(0.0, (1.0, 2.0))

    status = gate.update(200.0, pending_threads=0, fingerprint=(1.0, 99.0))

    assert status.ready is True
    assert status.verified is True


def test_the_gate_gives_up_rather_than_hanging_the_screen() -> None:
    """A screen that looks frozen is its own failure; say so and move on."""

    gate = FreezeGate(settle_floor_ms=1500.0, timeout_ms=10_000.0)
    gate.start(0.0, (1.0, 2.0))

    status = gate.update(10_000.0, pending_threads=3, fingerprint=(1.0, 2.0))

    assert (status.ready, status.verified, status.timed_out) == (True, False, True)


def test_unreadable_internals_proceed_but_are_never_called_verified() -> None:
    gate = FreezeGate(settle_floor_ms=100.0, timeout_ms=10_000.0)
    gate.start(0.0, None)

    status = gate.update(200.0, pending_threads=None, fingerprint=None)

    assert status.ready is True
    assert status.verified is False
    assert "NOT verified" in status.detail


def test_the_gate_refuses_to_report_before_it_was_started() -> None:
    with pytest.raises(ContractValidationError):
        FreezeGate().update(0.0, pending_threads=0, fingerprint=None)


# --- arrival timing ---------------------------------------------------------


def test_arrival_is_recorded_per_radius_from_when_the_target_appeared() -> None:
    timer = ArrivalTimer(_GEOMETRY, radii_px=(100.0, 400.0))
    target = GazePoint(0.5, 0.5)
    timer.target_shown(0, 1000.0)

    # About 205px away: inside 400, outside 100.
    timer.observe(0, GazePoint(0.55, 0.5), target, 1300.0)
    # Dead on, so 100 is satisfied too, but later.
    timer.observe(0, GazePoint(0.5, 0.5), target, 1800.0)

    timing = timer.results()[0]
    assert timing.arrivals_ms[400.0] == pytest.approx(300.0)
    assert timing.arrivals_ms[100.0] == pytest.approx(800.0)


def test_a_target_that_was_never_reached_records_none_not_a_big_number() -> None:
    """A large sentinel would average in as if it had been reached, late."""

    timer = ArrivalTimer(_GEOMETRY, radii_px=(100.0,))
    timer.target_shown(0, 0.0)
    timer.observe(0, GazePoint(0.9, 0.9), GazePoint(0.1, 0.1), 500.0)

    assert timer.results()[0].arrivals_ms[100.0] is None


def test_only_the_first_arrival_counts() -> None:
    timer = ArrivalTimer(_GEOMETRY, radii_px=(100.0,))
    timer.target_shown(0, 0.0)
    timer.observe(0, GazePoint(0.5, 0.5), GazePoint(0.5, 0.5), 200.0)
    timer.observe(0, GazePoint(0.5, 0.5), GazePoint(0.5, 0.5), 900.0)

    assert timer.results()[0].arrivals_ms[100.0] == pytest.approx(200.0)


# --- rows -------------------------------------------------------------------


def test_accepted_means_the_library_predicted_not_that_our_gate_agreed() -> None:
    """The whole point of the format: our gate must not silently filter.

    analyze.py hides accepted=false by default, so putting the gate verdict
    there would delete most of the library's output from its own measurement.
    """

    row = sample_from_gaze(
        target_index=0,
        sample=_gaze(GazePoint(0.5, 0.5)),
        phase=MeasurementPhase.MEASURING,
        gate_accepts=False,
        engine_calibration_flag=False,
        geometry=_GEOMETRY,
        tracking_state=TrackingState.LOW_CONFIDENCE,
        now_ms=1234.0,
        frame_id=7,
    )

    assert row is not None
    assert row.accepted is True
    assert row.gate_would_accept is False


def test_no_prediction_produces_no_row_rather_than_an_invented_coordinate() -> None:
    assert (
        sample_from_gaze(
            target_index=0,
            sample=None,
            phase=MeasurementPhase.MEASURING,
            gate_accepts=True,
            engine_calibration_flag=False,
            geometry=_GEOMETRY,
            tracking_state=TrackingState.TRACKED,
            now_ms=0.0,
            frame_id=1,
        )
        is None
    )


def test_an_off_screen_prediction_is_recorded_unclamped() -> None:
    """Clamping before measuring would shrink the error it deserves."""

    row = sample_from_gaze(
        target_index=0,
        sample=_gaze(GazePoint(1.4, -0.3)),
        phase=MeasurementPhase.MEASURING,
        gate_accepts=True,
        engine_calibration_flag=False,
        geometry=_GEOMETRY,
        tracking_state=TrackingState.TRACKED,
        now_ms=0.0,
        frame_id=1,
    )

    assert row is not None
    assert (row.predicted.x, row.predicted.y) == (1.4, -0.3)
    assert row.accepted is True
    assert row.to_dict()["predicted_x"] == 1.4


# --- the file ---------------------------------------------------------------


def _payload(*, gate_filtered: bool = False) -> dict[str, Any]:
    targets = [
        LiveValidationTarget("TEST_0", GazePoint(0.3, 0.4)),
        LiveValidationTarget("TEST_1", GazePoint(0.7, 0.6)),
    ]
    rows = [
        sample_from_gaze(
            target_index=index,
            sample=_gaze(GazePoint(0.3, 0.4), frame_id=index),
            phase=MeasurementPhase.MEASURING,
            gate_accepts=index == 0,
            engine_calibration_flag=False,
            geometry=_GEOMETRY,
            tracking_state=TrackingState.TRACKED,
            now_ms=float(index),
            frame_id=index,
        )
        for index in range(2)
    ]
    return build_predictions_payload(
        targets=targets,
        geometry=_GEOMETRY,
        samples=[row for row in rows if row is not None],
        timings=(),
        run_metadata={"engine": "eyegestures", "freeze_verified": True},
        gate_filtered=gate_filtered,
    )


def test_the_payload_carries_the_five_fields_analyze_reads() -> None:
    payload = _payload()
    sample = payload["samples"][0]

    for key in ("target_index", "predicted_x", "predicted_y", "timestamp", "accepted"):
        assert key in sample


def test_the_gated_companion_file_is_where_our_gate_does_filter() -> None:
    """Both pictures exist; neither is the silent default."""

    main = _payload()["samples"]
    gated = _payload(gate_filtered=True)["samples"]

    assert [row["accepted"] for row in main] == [True, True]
    assert [row["accepted"] for row in gated] == [True, False]


def test_the_file_records_where_the_target_was_actually_drawn() -> None:
    """Makes display and measurement agreement checkable from the file alone."""

    target = _payload()["targets"][0]

    assert target["drawn_target_center_px"] == {"x_px": 1228, "y_px": 460}


def test_the_unit_is_stated_in_the_file_not_left_as_bare_px() -> None:
    run = _payload()["run"]

    assert run["units"] == "qt_logical_px"
    assert run["physical_width_px"] == 5120


def test_writing_is_atomic_and_leaves_no_temporary_behind(tmp_path: Path) -> None:
    path = write_predictions(tmp_path / "nested" / "predictions.json", _payload())

    assert path.exists()
    assert list(tmp_path.rglob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["run"]["engine"] == "eyegestures"


# --- the phase machine ------------------------------------------------------


def _machine(**kwargs: Any) -> PhaseMachine:
    return PhaseMachine(
        freeze_gate=FreezeGate(settle_floor_ms=100.0, timeout_ms=1000.0),
        total_points=36,
        **kwargs,
    )


def _tick(
    machine: PhaseMachine,
    now_ms: float,
    *,
    is_calibrated: bool = False,
    completed: int = 0,
    threads: int | None = 0,
    fingerprint: tuple[float, ...] | None = (1.0,),
    complete: bool = False,
    on_freeze: Any = None,
) -> Any:
    return machine.update(
        now_ms,
        is_calibrated=is_calibrated,
        completed_points=completed,
        pending_threads=threads,
        fingerprint=fingerprint,
        measurement_complete=complete,
        on_freeze=on_freeze,
    )


def test_nothing_is_recorded_while_the_engine_is_still_calibrating() -> None:
    """Scoring during calibration would score the model on its training points."""

    view = _tick(_machine(), 0.0, completed=5)

    assert view.phase is MeasurementPhase.ENGINE_CALIBRATING
    assert view.recording is False
    assert "NOTHING IS BEING MEASURED" in view.banner


def test_finishing_calibration_freezes_before_it_measures() -> None:
    machine = _machine()
    frozen: list[bool] = []

    view = _tick(machine, 0.0, is_calibrated=True, on_freeze=lambda: frozen.append(True))

    assert frozen == [True]
    assert view.phase is MeasurementPhase.FREEZING
    assert view.recording is False


def test_measuring_starts_only_after_the_freeze_is_settled() -> None:
    machine = _machine()
    _tick(machine, 0.0, is_calibrated=True, on_freeze=lambda: None)

    assert _tick(machine, 50.0, is_calibrated=True).recording is False
    view = _tick(machine, 200.0, is_calibrated=True)

    assert view.phase is MeasurementPhase.MEASURING
    assert view.recording is True
    assert machine.freeze_verified is True


def test_an_unreadable_engine_measures_but_says_it_was_not_verified() -> None:
    """Hiding it would be worse: the run is useful, just not quotable."""

    machine = _machine()
    _tick(machine, 0.0, is_calibrated=True, fingerprint=None, threads=None, on_freeze=lambda: None)

    view = _tick(machine, 200.0, is_calibrated=True, fingerprint=None, threads=None)

    assert view.phase is MeasurementPhase.MEASURING
    assert machine.freeze_verified is False
    assert "FREEZE NOT VERIFIED" in view.banner


def test_a_model_that_moved_during_measurement_is_not_verified() -> None:
    """The check that actually matters: rows must describe ONE model."""

    stable, detail = model_stability((1.0, 2.0), (1.0, 99.0))

    assert stable is False
    assert "MODEL CHANGED DURING MEASUREMENT" in detail


def test_a_model_that_held_still_across_the_run_is_verified() -> None:
    stable, _ = model_stability((1.0, 2.0), (1.0, 2.0))

    assert stable is True


def test_an_unreadable_model_is_neither_verified_nor_rejected() -> None:
    """None must never be read as a yes."""

    assert model_stability(None, (1.0, 2.0))[0] is None
    assert model_stability((1.0, 2.0), None)[0] is None


def test_the_freezing_banner_never_looks_hung() -> None:
    """It reports progress and how to cancel on every tick."""

    machine = _machine()
    _tick(machine, 0.0, is_calibrated=True, on_freeze=lambda: None)

    view = _tick(machine, 50.0, is_calibrated=True, threads=2)

    assert "Esc" in view.banner
    assert "threads still running: 2" in view.banner


def test_calibrating_and_measuring_are_visually_distinct() -> None:
    machine = _machine()
    calibrating = _tick(machine, 0.0, completed=1)
    _tick(machine, 0.0, is_calibrated=True, on_freeze=lambda: None)
    measuring = _tick(machine, 200.0, is_calibrated=True)

    assert calibrating.accent != measuring.accent


def test_aborting_stops_recording_and_says_nothing_was_kept() -> None:
    machine = _machine()
    _tick(machine, 0.0, is_calibrated=True, on_freeze=lambda: None)
    _tick(machine, 200.0, is_calibrated=True)

    view = machine.abort("the window is not the size errors are scored against")

    assert view.phase is MeasurementPhase.ABORTED
    assert view.recording is False
    assert "nothing was recorded" in view.banner


def test_the_machine_never_goes_backwards() -> None:
    """A late tick must not restart calibration and silently retrain."""

    machine = _machine()
    _tick(machine, 0.0, is_calibrated=True, on_freeze=lambda: None)
    _tick(machine, 200.0, is_calibrated=True)
    _tick(machine, 300.0, is_calibrated=True, complete=True)

    view = _tick(machine, 400.0, is_calibrated=False, completed=0)

    assert view.phase is MeasurementPhase.COMPLETE
    assert view.recording is False
