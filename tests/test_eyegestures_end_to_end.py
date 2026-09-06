"""Library output -> our arithmetic -> file -> score, with no camera or display.

This is the test that would catch a break anywhere along the chain the live run
depends on. Everything is scripted: the engine is a double returning known
pixels, so the expected TEST_MEDIAN can be computed by hand and asserted
exactly, rather than eyeballed on a screen.

No GPL import, no Qt, no camera.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from gazelink.domain import (
    FramePacket,
    GazePoint,
    GazeSample,
    PixelFormat,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.eyegestures_engine import EyeGesturesGazePredictor, gate_would_accept
from gazelink.eyegestures_run import (
    ArrivalTimer,
    MeasurementPhase,
    build_predictions_payload,
    sample_from_gaze,
    write_predictions,
)
from gazelink.live_validation import LiveValidationTarget

import analyze  # isort: skip

pytestmark = pytest.mark.integration

_GEOMETRY = ScreenGeometry("ultrawide", 4096, 1152, 1.25)


class _GazeEvent:
    def __init__(self, point: tuple[float, float]) -> None:
        self.point = point


class _CalibrationEvent:
    def __init__(self, point: tuple[float, float]) -> None:
        self.point = point
        self.acceptance_radius = 500.0


class _ScriptedGestures:
    """Returns a fixed sequence of library outputs, in library pixels."""

    def __init__(self, script: list[tuple[object, object]]) -> None:
        self._script = list(script)
        self.calibration_flags: list[bool] = []

    def uploadCalibrationMap(self, points: object, context: str = "main") -> None:  # noqa: N802
        return None

    def setClassicalImpact(self, impact: int) -> None:  # noqa: N802
        return None

    def setFixation(self, fixation: float) -> None:  # noqa: N802
        return None

    def step(
        self, frame: object, calibration: bool, width: int, height: int, context: str = "main"
    ) -> tuple[object, object]:
        self.calibration_flags.append(calibration)
        return self._script.pop(0) if self._script else (None, None)


def _frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        captured_at_monotonic_ms=float(frame_id),
        width=2,
        height=2,
        pixel_format=PixelFormat.RGB24,
        image=bytes(12),
    )


def _observation(state: TrackingState = TrackingState.TRACKED) -> VisionObservation:
    return VisionObservation(
        frame_id=1,
        observed_at_monotonic_ms=0.0,
        tracking_state=state,
        face_box=None,
        left_eye=None,
        right_eye=None,
        head_pose=None,
        overall_confidence=0.5,
        reason_codes=() if state is TrackingState.TRACKED else (ReasonCode.LOW_CONFIDENCE,),
    )


def test_a_scripted_run_scores_exactly_the_error_that_was_injected(tmp_path: Path) -> None:
    """One known offset per target, so TEST_MEDIAN is arithmetic, not opinion.

    Each target is answered by a prediction displaced a known number of pixels,
    so the median the analyser prints must equal the median of those offsets.
    Anything between the library and the report -- normalisation, the file
    format, the scoring -- that quietly changes a coordinate breaks this.
    """

    targets = [
        LiveValidationTarget("TEST_0", GazePoint(0.30, 0.40)),
        LiveValidationTarget("TEST_1", GazePoint(0.50, 0.50)),
        LiveValidationTarget("TEST_2", GazePoint(0.70, 0.60)),
    ]
    offsets_px = [100.0, 200.0, 300.0]

    # One calibration advance, then one gaze frame per target. The library
    # speaks in ITS pixels (n * W), so that is what the double returns.
    script: list[tuple[object, object]] = [
        (None, _CalibrationEvent((10.0, 10.0))),
        (None, _CalibrationEvent((20.0, 20.0))),
    ]
    for target, offset in zip(targets, offsets_px, strict=True):
        script.append(
            (
                _GazeEvent(
                    (
                        target.screen_position.x * _GEOMETRY.width_px + offset,
                        target.screen_position.y * _GEOMETRY.height_px,
                    )
                ),
                _CalibrationEvent((20.0, 20.0)),
            )
        )

    gestures = _ScriptedGestures(script)
    predictor = EyeGesturesGazePredictor(
        screen_geometry=_GEOMETRY,
        gestures=gestures,
        calibration_points=1,
        require_tracked_observation=False,
    )

    # Drive the two calibration frames, then freeze before measuring anything.
    for frame_id in range(2):
        predictor.predict(
            frame=_frame(frame_id), observation=_observation(), now_monotonic_ms=float(frame_id)
        )
    assert predictor.is_calibrated
    predictor.freeze_calibration()

    timer = ArrivalTimer(_GEOMETRY, radii_px=(150.0, 400.0))
    rows = []
    for index, target in enumerate(targets):
        now_ms = 1000.0 + index * 100.0
        timer.target_shown(index, now_ms)
        observation = _observation()
        result = predictor.predict(
            frame=_frame(10 + index), observation=observation, now_monotonic_ms=now_ms
        )
        row = sample_from_gaze(
            target_index=index,
            sample=result.sample,
            phase=MeasurementPhase.MEASURING,
            gate_accepts=gate_would_accept(observation),
            engine_calibration_flag=gestures.calibration_flags[-1],
            geometry=_GEOMETRY,
            tracking_state=observation.tracking_state,
            now_ms=now_ms,
            frame_id=10 + index,
        )
        assert row is not None
        rows.append(row)
        timer.observe(index, row.predicted, target.screen_position, now_ms + 50.0)

    # Freezing must have held for every measured frame.
    assert gestures.calibration_flags[-3:] == [False, False, False]

    payload = build_predictions_payload(
        targets=targets,
        geometry=_GEOMETRY,
        samples=rows,
        timings=timer.results(),
        run_metadata={
            "engine": "eyegestures",
            "freeze_verified": True,
            "library_version": "3.2.4",
        },
    )
    path = write_predictions(tmp_path / "predictions.json", payload)
    history = tmp_path / "history.csv"

    exit_code = analyze.main(
        ["--predictions", str(path), "--note", "end-to-end harness", "--history", str(history)]
    )

    assert exit_code == 0
    history_row = list(csv.DictReader(history.read_text(encoding="utf-8").splitlines()))[-1]

    # The library was told to predict `offset` px away along x, in ITS n*W
    # convention; the score uses n*(W-1), so the recovered error is scaled by
    # (W-1)/W. Computing it rather than hard-coding keeps the test honest about
    # which convention each side uses.
    scale = (_GEOMETRY.width_px - 1) / _GEOMETRY.width_px
    expected = [offset * scale for offset in offsets_px]
    assert float(history_row["test_median"]) == pytest.approx(sorted(expected)[1], abs=0.5)
    assert history_row["verdict"] == "N/A"


def test_a_run_that_could_not_verify_the_freeze_is_marked_diagnostic(tmp_path: Path) -> None:
    """The label has to be enforced by the analyser, not just written down.

    A field in the file that the reader must notice is not protection. This is
    the check that an unverified run cannot be quoted as an accuracy result.
    """

    targets = [LiveValidationTarget("TEST_0", GazePoint(0.5, 0.5))]
    rows = [
        sample_from_gaze(
            target_index=0,
            sample=_scripted_sample(GazePoint(0.52, 0.5)),
            phase=MeasurementPhase.MEASURING,
            gate_accepts=True,
            engine_calibration_flag=False,
            geometry=_GEOMETRY,
            tracking_state=TrackingState.TRACKED,
            now_ms=0.0,
            frame_id=1,
        )
    ]
    payload = build_predictions_payload(
        targets=targets,
        geometry=_GEOMETRY,
        samples=[row for row in rows if row is not None],
        timings=(),
        run_metadata={
            "engine": "eyegestures",
            "freeze_verified": False,
            "freeze_detail": "gave up waiting after 10.0s (threads still running: 2)",
        },
    )
    path = write_predictions(tmp_path / "unverified.json", payload)
    history = tmp_path / "history.csv"

    exit_code = analyze.main(
        ["--predictions", str(path), "--note", "unverified", "--history", str(history)]
    )

    assert exit_code == 0
    row = list(csv.DictReader(history.read_text(encoding="utf-8").splitlines()))[-1]
    assert row["verdict"] == "DIAGNOSTIC"


def test_the_button_table_is_withheld_from_an_unverified_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hit rate is the easiest number to quote out of context.

    It must not exist in a report that cannot stand behind it.
    """

    targets = [LiveValidationTarget("TEST_0", GazePoint(0.5, 0.5))]
    row = sample_from_gaze(
        target_index=0,
        sample=_scripted_sample(GazePoint(0.5, 0.5)),
        phase=MeasurementPhase.MEASURING,
        gate_accepts=True,
        engine_calibration_flag=False,
        geometry=_GEOMETRY,
        tracking_state=TrackingState.TRACKED,
        now_ms=0.0,
        frame_id=1,
    )
    assert row is not None
    path = write_predictions(
        tmp_path / "unverified.json",
        build_predictions_payload(
            targets=targets,
            geometry=_GEOMETRY,
            samples=[row],
            timings=(),
            run_metadata={"engine": "eyegestures", "freeze_verified": False},
        ),
    )

    analyze.main(["--predictions", str(path), "--note", "n", "--history", str(tmp_path / "h.csv")])

    printed = capsys.readouterr().out
    assert "DIAGNOSTIC ONLY" in printed
    assert "withheld" in printed
    assert "hit rate for a square button" not in printed


def test_a_verified_run_reports_button_sizes_and_arrival_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    targets = [LiveValidationTarget("TEST_0", GazePoint(0.5, 0.5))]
    row = sample_from_gaze(
        target_index=0,
        sample=_scripted_sample(GazePoint(0.51, 0.5)),
        phase=MeasurementPhase.MEASURING,
        gate_accepts=True,
        engine_calibration_flag=False,
        geometry=_GEOMETRY,
        tracking_state=TrackingState.TRACKED,
        now_ms=0.0,
        frame_id=1,
    )
    assert row is not None
    timer = ArrivalTimer(_GEOMETRY, radii_px=(100.0,))
    timer.target_shown(0, 0.0)
    timer.observe(0, GazePoint(0.5, 0.5), GazePoint(0.5, 0.5), 420.0)

    path = write_predictions(
        tmp_path / "verified.json",
        build_predictions_payload(
            targets=targets,
            geometry=_GEOMETRY,
            samples=[row],
            timings=timer.results(),
            run_metadata={"engine": "eyegestures", "freeze_verified": True},
        ),
    )

    analyze.main(["--predictions", str(path), "--note", "n", "--history", str(tmp_path / "h.csv")])

    printed = capsys.readouterr().out
    assert "DIAGNOSTIC ONLY" not in printed
    assert "hit rate for a square button" in printed
    # ~41px off: inside a 100px button, outside nothing smaller here.
    assert "100" in printed
    assert "time to reach the target" in printed
    assert "420ms" in printed


def test_a_file_with_no_freeze_claim_keeps_its_previous_meaning(tmp_path: Path) -> None:
    """Tobii and other external files never needed a freeze.

    Marking them diagnostic would retroactively invalidate comparisons that are
    already recorded and were never suspect.
    """

    payload = {
        "screen_geometry": _GEOMETRY.to_dict(),
        "targets": [{"index": 0, "name": "TEST_0", "screen_position": {"x": 0.5, "y": 0.5}}],
        "samples": [
            {
                "target_index": 0,
                "predicted_x": 0.51,
                "predicted_y": 0.5,
                "timestamp": 0.0,
                "accepted": True,
            }
        ],
    }
    path = write_predictions(tmp_path / "legacy.json", payload)
    history = tmp_path / "history.csv"

    analyze.main(["--predictions", str(path), "--note", "legacy", "--history", str(history)])

    row = list(csv.DictReader(history.read_text(encoding="utf-8").splitlines()))[-1]
    assert row["verdict"] == "N/A"


def _scripted_sample(point: GazePoint) -> GazeSample:
    """A GazeSample carrying exactly this prediction."""

    return GazeSample(
        source_frame_id=1,
        sampled_at_monotonic_ms=0.0,
        raw_normalized=point,
        corrected_normalized=point,
        filtered_normalized=point,
        screen_position=PixelPoint(
            max(0, math.floor(point.x * (_GEOMETRY.width_px - 1))),
            max(0, math.floor(point.y * (_GEOMETRY.height_px - 1))),
        ),
        screen_id=_GEOMETRY.screen_id,
        confidence=0.5,
        valid_for_control=False,
        reason_codes=(),
    )
