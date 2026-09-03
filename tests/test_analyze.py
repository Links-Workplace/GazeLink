"""Deterministic tests for analyze.py: synthetic fixtures, no camera or Qt.

Every fixture uses a model that predicts a FIXED screen point regardless of
input features (the same "constant model" trick used in
tests/test_gaze_engine.py). Combined with targets placed at known, deliberately
chosen positions, this makes every per-sample pixel error an exact, hand-
computed number -- so the PASS/FAIL boundary and the median-vs-mean behavior
can be asserted precisely instead of approximately.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import analyze
import pytest

from gazelink.calibration import CalibrationSample, CalibrationSessionResult, CalibrationTarget
from gazelink.domain import ContractValidationError, GazePoint, NormalizedPoint, ScreenGeometry
from gazelink.gaze_engine import CalibrationModel
from gazelink.gaze_model import GazeModelKind, RegressionModel
from gazelink.live_validation import LiveValidationTarget
from gazelink.test_dataset import TestPointResult, TestSample

pytestmark = pytest.mark.unit

# width_px - 1 == height_px - 1 == 1000, so 0.001 normalized == 1px on both axes.
_GEOMETRY = ScreenGeometry("primary", 1001, 1001, 1.0)


def _constant_model(x: float, y: float) -> CalibrationModel:
    """A LINEAR/FULL model that predicts (x, y) for every input, always.

    coefficients are [bias, *9 zero-weighted features]; predict() computes
    bias*1.0 + 0*f0 + ... + 0*f8 == bias, unconditionally.
    """

    coefficients_x = (x,) + (0.0,) * 9
    coefficients_y = (y,) + (0.0,) * 9
    regression = RegressionModel(
        GazeModelKind.LINEAR, coefficients_x, coefficients_y, ridge_lambda=0.0
    )
    return CalibrationModel.create(
        regression=regression,
        screen_geometry=_GEOMETRY,
        camera_id="camera-0",
        calibration_id="test-calibration",
        trained_at_utc="2026-09-03T00:00:00+00:00",
    )


def _sample_fields(*, target_index: int, accepted: bool = True) -> dict[str, object]:
    """Arbitrary but complete feature fields; irrelevant to a constant model's output."""

    return {
        "source_frame_id": target_index,
        "observed_at_monotonic_ms": 100.0 + target_index,
        "target_index": target_index,
        "left_iris_in_eye": NormalizedPoint(0.45, 0.35),
        "right_iris_in_eye": NormalizedPoint(0.55, 0.35),
        "left_openness": 0.6,
        "right_openness": 0.6,
        "left_iris_in_lids_y": 0.5,
        "right_iris_in_lids_y": 0.5,
        "head_yaw_deg": 1.0,
        "head_pitch_deg": -5.0,
        "head_roll_deg": 0.5,
        "confidence": 0.5,
        "accepted": accepted,
        "reason": "accepted" if accepted else "low_confidence",
    }


def _calibration_result(*, target_x: float, target_y: float) -> CalibrationSessionResult:
    """All 9 calibration targets at one shared position -> one exact per-sample error."""

    targets = tuple(
        CalibrationTarget(index=i, screen_position=NormalizedPoint(target_x, target_y))
        for i in range(9)
    )
    samples = tuple(
        CalibrationSample(**_sample_fields(target_index=i))  # type: ignore[arg-type]
        for i in range(9)
    )
    return CalibrationSessionResult(
        target_order=tuple(range(9)),
        targets=targets,
        sample_counts=(1,) * 9,
        samples=samples,
        camera_id="camera-0",
        screen_geometry=_GEOMETRY,
        feature_schema_version=3,
        min_samples_per_target=1,
        started_at_monotonic_ms=0.0,
        completed_at_monotonic_ms=100.0,
    )


def _test_result(offsets: list[tuple[float, float]]) -> TestPointResult:
    """One test target and one sample per requested (x, y) position."""

    targets = tuple(
        LiveValidationTarget(f"TEST_{i}", GazePoint(x, y)) for i, (x, y) in enumerate(offsets)
    )
    samples = tuple(
        TestSample(**_sample_fields(target_index=i))  # type: ignore[arg-type]
        for i in range(len(offsets))
    )
    return TestPointResult(
        targets=targets,
        sample_counts=(1,) * len(offsets),
        samples=samples,
        camera_id="camera-0",
        screen_geometry=_GEOMETRY,
        seed=1,
        overlay_model_id=None,
    )


# --- PASS / FAIL boundary, all three combinations -------------------------


def test_verdict_passes_when_both_conditions_are_met() -> None:
    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)  # exact match -> 0px
    test = _test_result([(0.503, 0.5)] * 5)  # dx=3px -> median 3px

    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    test_errors = analyze.compute_test_errors(model, test, include_rejected=False)
    calib_median = analyze.overall_median(calib_errors)
    test_median = analyze.overall_median(test_errors)
    gap = test_median - calib_median  # type: ignore[operator]

    assert calib_median == pytest.approx(0.0)
    assert test_median == pytest.approx(3.0)
    assert gap == pytest.approx(3.0)
    verdict, reasons = analyze.compute_verdict(calib_median, test_median, gap)
    assert verdict == "PASS"
    assert reasons == []


def test_verdict_fails_on_test_median_only() -> None:
    """calib_median=90, test_median=130 -> gap=40 (<50, passes), test_median fails alone."""

    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.59, target_y=0.5)  # dx=90px
    test = _test_result([(0.63, 0.5)] * 3)  # dx=130px

    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    test_errors = analyze.compute_test_errors(model, test, include_rejected=False)
    calib_median = analyze.overall_median(calib_errors)
    test_median = analyze.overall_median(test_errors)
    gap = test_median - calib_median  # type: ignore[operator]

    assert calib_median == pytest.approx(90.0)
    assert test_median == pytest.approx(130.0)
    assert gap == pytest.approx(40.0)
    verdict, reasons = analyze.compute_verdict(calib_median, test_median, gap)
    assert verdict == "FAIL"
    assert len(reasons) == 1
    assert "test_median" in reasons[0]
    assert "gap" not in reasons[0]


def test_verdict_fails_on_gap_only() -> None:
    """calib_median=10, test_median=75 (<120, passes) -> gap=65 fails alone."""

    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.51, target_y=0.5)  # dx=10px
    test = _test_result([(0.575, 0.5)] * 3)  # dx=75px

    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    test_errors = analyze.compute_test_errors(model, test, include_rejected=False)
    calib_median = analyze.overall_median(calib_errors)
    test_median = analyze.overall_median(test_errors)
    gap = test_median - calib_median  # type: ignore[operator]

    assert calib_median == pytest.approx(10.0)
    assert test_median == pytest.approx(75.0)
    assert gap == pytest.approx(65.0)
    verdict, reasons = analyze.compute_verdict(calib_median, test_median, gap)
    assert verdict == "FAIL"
    assert len(reasons) == 1
    assert "gap" in reasons[0]
    assert "test_median" not in reasons[0]


def test_verdict_fails_on_both_conditions() -> None:
    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)  # 0px
    test = _test_result([(0.7, 0.5)] * 3)  # dx=200px

    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    test_errors = analyze.compute_test_errors(model, test, include_rejected=False)
    calib_median = analyze.overall_median(calib_errors)
    test_median = analyze.overall_median(test_errors)
    gap = test_median - calib_median  # type: ignore[operator]

    verdict, reasons = analyze.compute_verdict(calib_median, test_median, gap)
    assert verdict == "FAIL"
    assert len(reasons) == 2
    assert any("test_median" in reason for reason in reasons)
    assert any("gap" in reason for reason in reasons)


def test_pass_threshold_is_exclusive_at_the_boundary() -> None:
    """A median of exactly 120px must fail, matching the '< 120px' spec."""

    verdict, reasons = analyze.compute_verdict(0.0, 120.0, 120.0)
    assert verdict == "FAIL"
    assert reasons  # both conditions trip at exactly the threshold


# --- median, not mean -------------------------------------------------------


def test_overall_median_resists_a_single_outlier_point() -> None:
    """5 points at 40px + 1 outlier at 900px: median stays near 40, not ~272 (the mean)."""

    model = _constant_model(0.5, 0.5)
    offsets = [(0.54, 0.5)] * 5 + [(1.4, 0.5)]  # dx=40px x5, dx=900px x1
    test = _test_result(offsets)
    test_errors = analyze.compute_test_errors(model, test, include_rejected=False)
    test_median = analyze.overall_median(test_errors)

    assert test_median is not None
    assert test_median == pytest.approx(40.0)
    mean = sum(error.euclid_px for error in test_errors.errors) / len(test_errors.errors)
    assert mean == pytest.approx(183.33, abs=0.1)
    assert mean > 4 * test_median  # the mean the median must NOT report


def test_per_point_spread_mad_is_zero_for_repeated_identical_errors() -> None:
    """A constant model looking at one target repeatedly gives the same error every time."""

    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)
    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    rows = analyze.per_point_rows(calib_errors)
    assert all(row.spread_mad == pytest.approx(0.0) for row in rows)


def test_per_point_spread_mad_reflects_genuine_intra_point_noise() -> None:
    """Euclid errors [10, 10, 20, 90] at one point: median=15, MAD=median(|e-15|)=5."""

    errors = (
        analyze.SampleError(0, 10.0, 0.0, 10.0),
        analyze.SampleError(0, 10.0, 0.0, 10.0),
        analyze.SampleError(0, 20.0, 0.0, 20.0),
        analyze.SampleError(0, 90.0, 0.0, 90.0),
    )
    file_errors = analyze.FileErrors(
        label="TEST",
        geometry=_GEOMETRY,
        point_names=("TEST_0",),
        point_positions=(GazePoint(0.5, 0.5),),
        point_extrap=(False,),
        errors=errors,
        accepted_count=4,
        rejected_count=0,
        skipped_count=0,
    )
    rows = analyze.per_point_rows(file_errors)
    assert len(rows) == 1
    # euclid values [10, 10, 20, 90]; median = 15; MAD = median(|e-15|) = median([5,5,5,75]) = 5.
    assert rows[0].median_euclid == pytest.approx(15.0)
    assert rows[0].spread_mad == pytest.approx(5.0)
    assert rows[0].n == 4


# --- extrapolation split is diagnostic-only, never overrides the verdict ---


def test_extrapolated_points_are_flagged_and_never_soften_a_fail_verdict() -> None:
    """A low interp-only median must not turn an overall FAIL into a PASS."""

    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)
    # Inside the calibration bounding box (x in [0.05, 0.95] on this 1:1
    # geometry): tiny error. Outside it (x=1.3, well past 0.95): huge error.
    # An even 3-and-3 split (rather than 5-and-1) is deliberate: it lets the
    # extrapolated points actually pull the pooled median past the interp
    # figure, instead of being outvoted the way a lone outlier is.
    test = _test_result([(0.51, 0.5)] * 3 + [(1.3, 0.5)] * 3)
    test_errors = analyze.compute_test_errors(model, test, include_rejected=False)

    assert test_errors.point_extrap[:3] == (False,) * 3
    assert test_errors.point_extrap[3:] == (True,) * 3

    interp_median = analyze.overall_median(test_errors, only_extrap=False)
    extrap_median = analyze.overall_median(test_errors, only_extrap=True)
    overall = analyze.overall_median(test_errors)
    assert interp_median == pytest.approx(10.0)
    assert extrap_median == pytest.approx(800.0)
    # The pooled (real, verdict-relevant) median sits between the two, driven
    # up by the extrapolated points -- it is not just the interpolated figure.
    assert overall is not None and interp_median is not None
    assert overall > interp_median

    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    calib_median = analyze.overall_median(calib_errors)
    gap = overall - calib_median  # type: ignore[operator]
    verdict, reasons = analyze.compute_verdict(calib_median, overall, gap)
    # A low TEST_MEDIAN_INTERP_ONLY (10px) must not be able to rescue this.
    assert verdict == "FAIL"
    assert reasons


def test_calibration_points_are_never_flagged_as_extrapolated() -> None:
    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)
    calib_errors = analyze.compute_calibration_errors(model, calib, include_rejected=False)
    assert all(flag is False for flag in calib_errors.point_extrap)


# --- skipped and rejected samples -------------------------------------------


def test_a_sample_with_no_usable_features_is_counted_as_skipped_not_silently_dropped() -> None:
    model = _constant_model(0.5, 0.5)
    fields = _sample_fields(target_index=0)
    fields["head_yaw_deg"] = None  # from_calibration_sample requires all pose fields present
    incomplete = CalibrationSample(**fields)  # type: ignore[arg-type]
    targets = tuple(
        CalibrationTarget(index=i, screen_position=NormalizedPoint(0.5, 0.5)) for i in range(9)
    )
    result = CalibrationSessionResult(
        target_order=tuple(range(9)),
        targets=targets,
        sample_counts=(1,) + (0,) * 8,
        samples=(incomplete,),
        camera_id="camera-0",
        screen_geometry=_GEOMETRY,
        feature_schema_version=3,
        min_samples_per_target=1,
        started_at_monotonic_ms=0.0,
        completed_at_monotonic_ms=1.0,
    )
    errors = analyze.compute_calibration_errors(model, result, include_rejected=False)
    assert errors.skipped_count == 1
    assert errors.errors == ()


def test_rejected_samples_are_excluded_by_default() -> None:
    model = _constant_model(0.5, 0.5)
    accepted = CalibrationSample(**_sample_fields(target_index=0, accepted=True))  # type: ignore[arg-type]
    rejected = CalibrationSample(**_sample_fields(target_index=0, accepted=False))  # type: ignore[arg-type]
    targets = tuple(
        CalibrationTarget(index=i, screen_position=NormalizedPoint(0.5, 0.5)) for i in range(9)
    )
    result = CalibrationSessionResult(
        target_order=tuple(range(9)),
        targets=targets,
        sample_counts=(1,) + (0,) * 8,
        samples=(accepted, rejected),
        camera_id="camera-0",
        screen_geometry=_GEOMETRY,
        feature_schema_version=3,
        min_samples_per_target=1,
        started_at_monotonic_ms=0.0,
        completed_at_monotonic_ms=1.0,
    )
    default_errors = analyze.compute_calibration_errors(model, result, include_rejected=False)
    assert len(default_errors.errors) == 1
    assert default_errors.accepted_count == 1
    assert default_errors.rejected_count == 1

    included_errors = analyze.compute_calibration_errors(model, result, include_rejected=True)
    assert len(included_errors.errors) == 2


# --- history.csv: the one file this script writes --------------------------


def test_history_csv_is_created_with_a_header_then_appended(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    analyze.append_history(
        history_path,
        note="first run",
        model_id="model-a",
        calib_median=10.0,
        test_median=30.0,
        test_median_interp=25.0,
        test_median_extrap=None,
        gap=20.0,
        worst_point_label="TEST_0",
        verdict="PASS",
    )
    analyze.append_history(
        history_path,
        note="second run",
        model_id="model-a",
        calib_median=12.0,
        test_median=140.0,
        test_median_interp=None,
        test_median_extrap=140.0,
        gap=128.0,
        worst_point_label="TEST_3",
        verdict="FAIL",
    )
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(analyze._HISTORY_HEADER)
    assert len(rows) == 3
    assert rows[1][1] == "first run"
    assert rows[1][-1] == "PASS"
    assert rows[2][1] == "second run"
    assert rows[2][-1] == "FAIL"
    assert rows[2][5] == ""  # test_median_interp was None -> blank, not "None"


# --- full CLI: read-only except history.csv ---------------------------------


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_writes_only_history_csv_and_reports_the_correct_verdict(tmp_path: Path) -> None:
    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)
    test = _test_result([(0.503, 0.5)] * 5)

    model_path = tmp_path / "model.json"
    calib_path = tmp_path / "dataset.json"
    test_path = tmp_path / "test_dataset.json"
    history_path = tmp_path / "history.csv"
    _write_json(model_path, model.to_dict())
    _write_json(calib_path, calib.to_dict())
    _write_json(test_path, test.to_dict())

    before_mtimes = {path: path.stat().st_mtime_ns for path in (model_path, calib_path, test_path)}

    exit_code = analyze.main(
        [
            "--calib",
            str(calib_path),
            "--test",
            str(test_path),
            "--model",
            str(model_path),
            "--note",
            "smoke test",
            "--history",
            str(history_path),
        ]
    )

    assert exit_code == 0  # PASS
    assert history_path.exists()
    for path, mtime in before_mtimes.items():
        assert path.stat().st_mtime_ns == mtime, f"{path} was modified by a read-only script"
    # Nothing else was created in the directory besides history.csv.
    created = set(tmp_path.iterdir()) - {model_path, calib_path, test_path, history_path}
    assert created == set()


def test_main_returns_nonzero_exit_code_on_fail(tmp_path: Path) -> None:
    model = _constant_model(0.5, 0.5)
    calib = _calibration_result(target_x=0.5, target_y=0.5)
    test = _test_result([(0.7, 0.5)] * 3)  # dx=200px -> FAIL

    model_path = tmp_path / "model.json"
    calib_path = tmp_path / "dataset.json"
    test_path = tmp_path / "test_dataset.json"
    _write_json(model_path, model.to_dict())
    _write_json(calib_path, calib.to_dict())
    _write_json(test_path, test.to_dict())

    exit_code = analyze.main(
        [
            "--calib",
            str(calib_path),
            "--test",
            str(test_path),
            "--model",
            str(model_path),
            "--note",
            "should fail",
            "--history",
            str(tmp_path / "history.csv"),
        ]
    )
    assert exit_code == 1


# --- predictions mode: pre-computed (x, y), no model/calibration of our own -


def _predictions_result(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    accepted: list[bool] | None = None,
) -> analyze.PredictionResult:
    """One held-out target and one sample per (target_xy, predicted_xy) pair."""

    targets = tuple(
        LiveValidationTarget(f"TEST_{i}", GazePoint(*t)) for i, (t, _p) in enumerate(pairs)
    )
    flags = accepted if accepted is not None else [True] * len(pairs)
    samples = tuple(
        analyze.PredictionSample(
            target_index=i, predicted=GazePoint(*p), timestamp=float(i), accepted=flags[i]
        )
        for i, (_t, p) in enumerate(pairs)
    )
    return analyze.PredictionResult(targets=targets, screen_geometry=_GEOMETRY, samples=samples)


def test_compute_prediction_errors_uses_the_same_pixel_math_as_test_errors() -> None:
    result = _predictions_result([((0.5, 0.5), (0.503, 0.5))] * 5)  # dx=3px -> median 3px
    errors = analyze.compute_prediction_errors(result, include_rejected=False)
    assert analyze.overall_median(errors) == pytest.approx(3.0)
    assert errors.skipped_count == 0  # no feature extraction exists in this mode to skip on


def test_compute_prediction_errors_flags_extrapolated_points_using_calibration_grid() -> None:
    """Extrapolation is judged against targets_for_screen_geometry(), same as compute_test_errors --
    a predictions file cannot supply its own notion of what counts as extrapolation."""

    result = _predictions_result([((0.51, 0.5), (0.51, 0.5))] * 3 + [((1.3, 0.5), (1.3, 0.5))] * 3)
    errors = analyze.compute_prediction_errors(result, include_rejected=False)
    assert errors.point_extrap[:3] == (False,) * 3
    assert errors.point_extrap[3:] == (True,) * 3


def test_predictions_rejected_samples_are_excluded_by_default() -> None:
    result = _predictions_result(
        [((0.5, 0.5), (0.503, 0.5)), ((0.5, 0.5), (0.9, 0.5))],
        accepted=[True, False],
    )
    default_errors = analyze.compute_prediction_errors(result, include_rejected=False)
    assert len(default_errors.errors) == 1
    assert default_errors.accepted_count == 1
    assert default_errors.rejected_count == 1

    included_errors = analyze.compute_prediction_errors(result, include_rejected=True)
    assert len(included_errors.errors) == 2


def test_format_predictions_report_has_no_calib_median_gap_or_verdict() -> None:
    result = _predictions_result([((0.5, 0.5), (0.503, 0.5))] * 3)
    errors = analyze.compute_prediction_errors(result, include_rejected=False)
    report = analyze.format_predictions_report(
        predictions=errors, source_label="preds.json", geometry_warnings=[]
    )
    assert "CALIB_MEDIAN   n/a" in report
    assert "GAP            n/a" in report
    assert "TEST_MEDIAN    3 px" in report
    assert "VERDICT" not in report


def test_predictions_from_dict_rejects_out_of_range_target_index() -> None:
    payload = {
        "screen_geometry": _GEOMETRY.to_dict(),
        "targets": [{"index": 0, "name": "TEST_0", "screen_position": {"x": 0.5, "y": 0.5}}],
        "samples": [
            {
                "target_index": 1,
                "predicted_x": 0.5,
                "predicted_y": 0.5,
                "timestamp": 0.0,
                "accepted": True,
            }
        ],
    }
    with pytest.raises(ContractValidationError):
        analyze.PredictionResult.from_dict(payload)


def test_main_predictions_mode_writes_only_history_csv(tmp_path: Path) -> None:
    result = _predictions_result([((0.5, 0.5), (0.503, 0.5))] * 5)
    predictions_path = tmp_path / "predictions.json"
    history_path = tmp_path / "history.csv"
    _write_json(
        predictions_path,
        {
            "screen_geometry": result.screen_geometry.to_dict(),
            "targets": [
                {"index": i, "name": t.name, "screen_position": t.screen_position.to_dict()}
                for i, t in enumerate(result.targets)
            ],
            "samples": [
                {
                    "target_index": s.target_index,
                    "predicted_x": s.predicted.x,
                    "predicted_y": s.predicted.y,
                    "timestamp": s.timestamp,
                    "accepted": s.accepted,
                }
                for s in result.samples
            ],
        },
    )
    before_mtime = predictions_path.stat().st_mtime_ns

    exit_code = analyze.main(
        [
            "--predictions",
            str(predictions_path),
            "--note",
            "eyegestures smoke test",
            "--history",
            str(history_path),
        ]
    )

    assert exit_code == 0
    assert predictions_path.stat().st_mtime_ns == before_mtime
    created = set(tmp_path.iterdir()) - {predictions_path, history_path}
    assert created == set()

    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = list(analyze._HISTORY_HEADER)
    row = dict(zip(header, rows[1], strict=True))
    assert row["calib_median"] == ""
    assert row["gap"] == ""
    assert row["test_median"] == "3.0"
    assert row["verdict"] == "N/A"
    assert row["model_id"] == "predictions:predictions"


def test_main_rejects_predictions_combined_with_calib_test_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        analyze.main(
            [
                "--predictions",
                str(tmp_path / "predictions.json"),
                "--calib",
                str(tmp_path / "dataset.json"),
                "--note",
                "should be rejected",
            ]
        )
    assert excinfo.value.code == 2


def test_main_rejects_missing_required_args_without_predictions(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        analyze.main(["--note", "nothing given"])
    assert excinfo.value.code == 2
