#!/usr/bin/env python
"""Measurement-day analysis: is the accuracy real, or memorized?

This script reads two datasets already on disk -- a calibration dataset
(``.gazelink/calibration/dataset_*.json``, the 9 points a model trained on)
and a held-out test dataset (``.gazelink/test_points/test_dataset_*.json``,
random points a model never saw) -- runs one named model's predictions
against both, and reports whether the model's accuracy on the calibration
points generalizes to the held-out points.

It is READ-ONLY except for one thing: it appends one row to ``history.csv``
so a sequence of runs across days can be compared instead of re-derived from
memory each time.

It does not calibrate, train, correct, or promote anything. It does not
touch the calibration/mapping/gesture code at all -- it only imports the
already-trained ``CalibrationModel.regression.predict()`` and the already-
extracted per-sample features, and computes pixel distances.

Usage:
    python analyze.py --calib .gazelink/calibration/dataset_XXXX.json \\
                       --test  .gazelink/test_points/test_dataset_YYYY.json \\
                       --model .gazelink/calibration/latest_model.json \\
                       --note "baseline, no changes"

A second, independent mode accepts predictions already computed by something
other than this project's own model -- e.g. an external gaze library scored
against the same held-out points via ``export_test_targets.py``. It replaces
``--test``/``--model`` with a single pre-computed predictions file and skips
CALIB_MEDIAN, GAP, and VERDICT entirely: without a calibration file of our
own there is nothing to compare the test error against, so this mode reports
TEST_MEDIAN and the interpolation/extrapolation breakdown only.

    python analyze.py --predictions predictions.json --note "eyegestures v2"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from gazelink.calibration import CalibrationSample, CalibrationSessionResult
from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry
from gazelink.gaze_engine import CalibrationModel
from gazelink.gaze_features import GazeFeatureVector, from_calibration_sample
from gazelink.live_validation import LiveValidationTarget
from gazelink.test_dataset import TestPointResult, TestSample
from gazelink.test_points import is_extrapolated

# --- Fill these in for your setup --------------------------------------
SCREEN_WIDTH_PX = 4096
SCREEN_HEIGHT_PX = 1152
SCREEN_WIDTH_CM = 120  # fill in; 0.0 means degree conversion is reported as n/a
VIEWING_DISTANCE_CM = 50  # fill in; 0.0 means degree conversion is reported as n/a

# --- Fixed pass threshold. Do not change this to make a run pass. -------
PASS_TEST_MEDIAN_PX = 120.0
PASS_GAP_PX = 50.0

DEFAULT_HISTORY_PATH = Path("history.csv")
_HISTORY_HEADER = (
    "timestamp",
    "note",
    "model_id",
    "calib_median",
    "test_median",
    "test_median_interp",
    "test_median_extrap",
    "gap",
    "worst_point",
    "verdict",
)

Sample = CalibrationSample | TestSample


# --- Data loading (read-only) --------------------------------------------


def load_calibration(path: Path) -> CalibrationSessionResult:
    return CalibrationSessionResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_test(path: Path) -> TestPointResult:
    return TestPointResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_model(path: Path) -> CalibrationModel:
    return CalibrationModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- Predictions mode: pre-computed (x, y) from something else entirely ----
#
# No feature extraction and no CalibrationModel involved -- the file already
# carries the predicted screen point for every sample. This is how an
# external gaze library (which owns its own model end to end) is scored on
# the exact same held-out points, without adopting or reimplementing this
# project's calibration/model/mapping logic.


@dataclass(frozen=True)
class PredictionSample:
    target_index: int
    predicted: GazePoint
    timestamp: float
    accepted: bool


@dataclass(frozen=True)
class PredictionResult:
    targets: tuple[LiveValidationTarget, ...]
    screen_geometry: ScreenGeometry
    samples: tuple[PredictionSample, ...]

    @classmethod
    def from_dict(cls, value: object) -> PredictionResult:
        if not isinstance(value, dict):
            raise ContractValidationError("predictions file must be a JSON object")
        raw_targets = value.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ContractValidationError(
                "predictions file must declare a non-empty 'targets' list"
            )
        targets = tuple(
            LiveValidationTarget(str(entry["name"]), GazePoint.from_dict(entry["screen_position"]))
            for entry in raw_targets
        )
        screen_geometry = ScreenGeometry.from_dict(value.get("screen_geometry"))
        samples = []
        for entry in value.get("samples", ()):
            target_index = entry["target_index"]
            if (
                isinstance(target_index, bool)
                or not isinstance(target_index, int)
                or not 0 <= target_index < len(targets)
            ):
                raise ContractValidationError(
                    f"sample target_index {target_index!r} is out of range for "
                    f"{len(targets)} declared targets"
                )
            samples.append(
                PredictionSample(
                    target_index=target_index,
                    predicted=GazePoint(entry["predicted_x"], entry["predicted_y"]),
                    timestamp=float(entry["timestamp"]),
                    accepted=bool(entry["accepted"]),
                )
            )
        return cls(targets=targets, screen_geometry=screen_geometry, samples=tuple(samples))


def load_predictions(path: Path) -> PredictionResult:
    return PredictionResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- Per-sample error computation ----------------------------------------


@dataclass(frozen=True)
class SampleError:
    target_index: int
    dx_px: float
    dy_px: float
    euclid_px: float


def _extract_features(sample: Sample) -> GazeFeatureVector | None:
    """Route to the one real feature extractor, whichever sample type this is.

    ``from_calibration_sample`` requires an actual ``CalibrationSample``
    instance (it checks ``isinstance``), so a ``TestSample`` -- identical in
    every field but not a subclass of it -- must go through its own
    ``.features()`` method instead. That method internally builds a real
    ``CalibrationSample`` carrier and calls the SAME extractor, so both
    branches below ultimately run the exact same feature math.
    """

    if isinstance(sample, TestSample):
        return sample.features()
    return from_calibration_sample(sample)


def _features_for_analysis(sample: Sample, *, include_rejected: bool) -> GazeFeatureVector | None:
    """Return usable features, optionally for an otherwise-rejected sample.

    The extractor returns ``None`` for any sample with ``accepted=False`` by
    design -- that is the training-time quality gate, and this script must
    not reimplement or route around that math to get a second opinion on it.
    What ``--include-rejected`` does instead is view a rejected sample as if
    it had been accepted, purely so its features (when present) can be
    extracted through the SAME unmodified function used everywhere else. A
    sample rejected for missing tracking data still yields ``None`` here,
    exactly as it would during real training.
    """

    if sample.accepted or not include_rejected:
        return _extract_features(sample)
    return _extract_features(replace(sample, accepted=True))


def _pixel_error(
    predicted: GazePoint, target: GazePoint, geometry: ScreenGeometry
) -> tuple[float, float, float]:
    dx = (predicted.x - target.x) * (geometry.width_px - 1)
    dy = (predicted.y - target.y) * (geometry.height_px - 1)
    return dx, dy, math.hypot(dx, dy)


@dataclass(frozen=True)
class FileErrors:
    label: str
    geometry: ScreenGeometry
    point_names: tuple[str, ...]
    point_positions: tuple[GazePoint, ...]
    point_extrap: tuple[bool, ...]
    errors: tuple[SampleError, ...]
    accepted_count: int
    rejected_count: int
    skipped_count: int


def compute_calibration_errors(
    model: CalibrationModel,
    result: CalibrationSessionResult,
    *,
    include_rejected: bool,
) -> FileErrors:
    target_positions = {
        target.index: GazePoint(target.screen_position.x, target.screen_position.y)
        for target in result.targets
    }
    accepted = sum(1 for sample in result.samples if sample.accepted)
    rejected = len(result.samples) - accepted
    errors: list[SampleError] = []
    skipped = 0
    for sample in result.samples:
        if not sample.accepted and not include_rejected:
            continue
        features = _features_for_analysis(sample, include_rejected=include_rejected)
        if features is None:
            skipped += 1
            continue
        predicted = model.regression.predict(features)
        target = target_positions[sample.target_index]
        dx, dy, euclid = _pixel_error(predicted, target, result.screen_geometry)
        errors.append(SampleError(sample.target_index, dx, dy, euclid))
    return FileErrors(
        label="CALIBRATION",
        geometry=result.screen_geometry,
        point_names=tuple(f"target[{target.index}]" for target in result.targets),
        point_positions=tuple(
            GazePoint(result.targets[i].screen_position.x, result.targets[i].screen_position.y)
            for i in range(len(result.targets))
        ),
        point_extrap=tuple(False for _ in result.targets),  # the calibration file defines the box
        errors=tuple(errors),
        accepted_count=accepted,
        rejected_count=rejected,
        skipped_count=skipped,
    )


def compute_test_errors(
    model: CalibrationModel,
    result: TestPointResult,
    *,
    include_rejected: bool,
) -> FileErrors:
    accepted = sum(1 for sample in result.samples if sample.accepted)
    rejected = len(result.samples) - accepted
    errors: list[SampleError] = []
    skipped = 0
    for sample in result.samples:
        if not sample.accepted and not include_rejected:
            continue
        features = _features_for_analysis(sample, include_rejected=include_rejected)
        if features is None:
            skipped += 1
            continue
        predicted = model.regression.predict(features)
        target = result.targets[sample.target_index].screen_position
        dx, dy, euclid = _pixel_error(predicted, target, result.screen_geometry)
        errors.append(SampleError(sample.target_index, dx, dy, euclid))
    extrap = tuple(
        is_extrapolated(target.screen_position, result.screen_geometry) for target in result.targets
    )
    return FileErrors(
        label="TEST",
        geometry=result.screen_geometry,
        point_names=tuple(target.name for target in result.targets),
        point_positions=tuple(target.screen_position for target in result.targets),
        point_extrap=extrap,
        errors=tuple(errors),
        accepted_count=accepted,
        rejected_count=rejected,
        skipped_count=skipped,
    )


def compute_prediction_errors(
    result: PredictionResult,
    *,
    include_rejected: bool,
) -> FileErrors:
    """Same dx/dy/extrap math as compute_test_errors, sourced from a ready (x, y).

    There is no CalibrationModel to call and no feature extraction to skip
    samples on, so skipped_count is always 0 here -- every PredictionSample
    already carries a valid predicted point by construction.
    """

    accepted = sum(1 for sample in result.samples if sample.accepted)
    rejected = len(result.samples) - accepted
    errors: list[SampleError] = []
    for sample in result.samples:
        if not sample.accepted and not include_rejected:
            continue
        target = result.targets[sample.target_index].screen_position
        dx, dy, euclid = _pixel_error(sample.predicted, target, result.screen_geometry)
        errors.append(SampleError(sample.target_index, dx, dy, euclid))
    extrap = tuple(
        is_extrapolated(target.screen_position, result.screen_geometry) for target in result.targets
    )
    return FileErrors(
        label="PREDICTIONS",
        geometry=result.screen_geometry,
        point_names=tuple(target.name for target in result.targets),
        point_positions=tuple(target.screen_position for target in result.targets),
        point_extrap=extrap,
        errors=tuple(errors),
        accepted_count=accepted,
        rejected_count=rejected,
        skipped_count=0,
    )


# --- Aggregation -----------------------------------------------------------


@dataclass(frozen=True)
class PointRow:
    name: str
    median_abs_dx: float
    median_abs_dy: float
    spread_mad: float
    n: int
    extrap: bool
    median_euclid: float


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median(abs(value - center) for value in values)


def per_point_rows(file_errors: FileErrors) -> list[PointRow]:
    rows: list[PointRow] = []
    for index, name in enumerate(file_errors.point_names):
        point_errors = [error for error in file_errors.errors if error.target_index == index]
        if not point_errors:
            rows.append(PointRow(name, 0.0, 0.0, 0.0, 0, file_errors.point_extrap[index], 0.0))
            continue
        euclids = [error.euclid_px for error in point_errors]
        rows.append(
            PointRow(
                name=name,
                median_abs_dx=median(abs(error.dx_px) for error in point_errors),
                median_abs_dy=median(abs(error.dy_px) for error in point_errors),
                spread_mad=_mad(euclids),
                n=len(point_errors),
                extrap=file_errors.point_extrap[index],
                median_euclid=median(euclids),
            )
        )
    return rows


def worst_point(rows: list[PointRow]) -> PointRow | None:
    scored = [row for row in rows if row.n > 0]
    if not scored:
        return None
    return max(scored, key=lambda row: row.median_euclid)


def overall_median(file_errors: FileErrors, *, only_extrap: bool | None = None) -> float | None:
    """Median euclidean error pooled across samples.

    ``only_extrap=None`` pools everything; ``True``/``False`` restricts to
    samples whose target falls outside/inside the calibration bounding box.
    Returns ``None`` (not 0.0) when no sample qualifies, so an empty subset
    is never misreported as a perfect score.
    """

    if only_extrap is None:
        pool = list(file_errors.errors)
    else:
        pool = [
            error
            for error in file_errors.errors
            if file_errors.point_extrap[error.target_index] is only_extrap
        ]
    if not pool:
        return None
    return median(error.euclid_px for error in pool)


# --- Degree conversion (diagnostic only) -----------------------------------


def px_to_deg(error_px: float, geometry: ScreenGeometry) -> float | None:
    if SCREEN_WIDTH_CM <= 0.0 or VIEWING_DISTANCE_CM <= 0.0:
        return None
    px_per_cm = geometry.width_px / SCREEN_WIDTH_CM
    error_cm = error_px / px_per_cm
    return math.degrees(2.0 * math.atan((error_cm / 2.0) / VIEWING_DISTANCE_CM))


def _fmt_deg(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}deg"


# --- Rendering ---------------------------------------------------------


def format_point_table(rows: list[PointRow]) -> str:
    header = (
        f"{'point':<14}{'|dX| med':>10}{'|dY| med':>10}{'spread(MAD)':>13}{'n':>5}{'extrap':>8}"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.name:<14}{row.median_abs_dx:>10.1f}{row.median_abs_dy:>10.1f}"
            f"{row.spread_mad:>13.1f}{row.n:>5}{('yes' if row.extrap else ''):>8}"
        )
    return "\n".join(lines)


def _format_file_section(label: str, file_errors: FileErrors) -> list[str]:
    """Per-point table plus worst-point line for one FileErrors; shared by
    every report so calibration/test/predictions files render identically."""

    rows = per_point_rows(file_errors)
    lines = [
        f"--- {label} "
        f"({len(file_errors.errors)} used, {file_errors.skipped_count} skipped, "
        f"{file_errors.accepted_count} accepted, {file_errors.rejected_count} rejected) ---",
        format_point_table(rows),
    ]
    worst = worst_point(rows)
    if worst is not None:
        lines.append(
            f"worst point: {worst.name}  median_euclid={worst.median_euclid:.0f}px  "
            f"|dX|={worst.median_abs_dx:.0f}px  |dY|={worst.median_abs_dy:.0f}px"
            + ("  [extrapolated]" if worst.extrap else "")
        )
    else:
        lines.append("worst point: n/a (no usable samples)")
    lines.append("")
    return lines


def _format_diagnostic_breakdown(file_errors: FileErrors) -> list[str]:
    """TEST_MEDIAN_INTERP_ONLY / TEST_MEDIAN_EXTRAP_ONLY; never part of a verdict."""

    interp = overall_median(file_errors, only_extrap=False)
    extrap = overall_median(file_errors, only_extrap=True)
    interp_n = sum(
        1 for error in file_errors.errors if not file_errors.point_extrap[error.target_index]
    )
    extrap_n = sum(
        1 for error in file_errors.errors if file_errors.point_extrap[error.target_index]
    )
    total_n = interp_n + extrap_n
    return [
        "  diagnostic only, not part of the verdict:",
        "  TEST_MEDIAN_INTERP_ONLY  "
        + ("n/a" if interp is None else f"{interp:.0f} px")
        + f"   ({interp_n} of {total_n} points, extrap excluded)",
        "  TEST_MEDIAN_EXTRAP_ONLY  "
        + ("n/a" if extrap is None else f"{extrap:.0f} px")
        + f"   ({extrap_n} points)",
        "",
    ]


def format_predictions_report(
    *,
    predictions: FileErrors,
    source_label: str,
    geometry_warnings: list[str],
) -> str:
    """Predictions-mode report: TEST_MEDIAN and the extrap breakdown only.

    CALIB_MEDIAN and GAP are always n/a here -- this mode has no calibration
    file of ours to compare against, by design (see the module docstring).
    There is no VERDICT: the fixed PASS thresholds were never validated for
    a source with no calibration/test gap of its own, so this mode does not
    pass or fail anything, only measures and records it in history.csv.
    """

    lines: list[str] = []
    for warning in geometry_warnings:
        lines.append(f"WARNING: {warning}")
    if geometry_warnings:
        lines.append("")

    lines.append(f"PREDICTIONS SOURCE   {source_label}")
    lines.append("")

    lines.append("CALIB_MEDIAN   n/a (predictions mode: no calibration file of our own)")
    test_median = overall_median(predictions)
    if test_median is None:
        lines.append("TEST_MEDIAN    n/a (no usable prediction samples)")
    else:
        test_deg = _fmt_deg(px_to_deg(test_median, predictions.geometry))
        lines.append(f"TEST_MEDIAN    {test_median:.0f} px   ({test_deg})")
    lines.append("GAP            n/a (predictions mode: no calibration file of our own)")
    lines.append("")

    lines.extend(_format_diagnostic_breakdown(predictions))
    lines.extend(_format_file_section("Predictions file", predictions))
    return "\n".join(lines)


def format_report(
    *,
    model: CalibrationModel,
    calib: FileErrors,
    test: FileErrors,
    geometry_warnings: list[str],
) -> str:
    calib_median = overall_median(calib)
    test_median = overall_median(test)

    lines: list[str] = []
    for warning in geometry_warnings:
        lines.append(f"WARNING: {warning}")
    if geometry_warnings:
        lines.append("")

    lines.append(
        f"MODEL      {model.model_id[:16]}  KIND={model.regression.kind.value}  "
        f"PROFILE={model.regression.profile.value}"
    )
    lines.append("")

    if calib_median is None:
        lines.append("CALIB_MEDIAN   n/a (no usable calibration samples)")
    else:
        calib_deg = _fmt_deg(px_to_deg(calib_median, calib.geometry))
        lines.append(f"CALIB_MEDIAN   {calib_median:.0f} px   ({calib_deg})")
    if test_median is None:
        lines.append("TEST_MEDIAN    n/a (no usable test samples)")
    else:
        test_deg = _fmt_deg(px_to_deg(test_median, test.geometry))
        lines.append(f"TEST_MEDIAN    {test_median:.0f} px   ({test_deg})")

    gap = None if calib_median is None or test_median is None else test_median - calib_median
    lines.append("GAP            " + ("n/a" if gap is None else f"{gap:.0f} px"))
    lines.append("")
    lines.extend(_format_diagnostic_breakdown(test))

    verdict, reasons = compute_verdict(calib_median, test_median, gap)
    if verdict == "PASS":
        lines.append("VERDICT: PASS")
    else:
        lines.append(f"VERDICT: FAIL  ({'; '.join(reasons)})")
    lines.append("")

    for label, file_errors in (("Calibration file", calib), ("Test file", test)):
        lines.extend(_format_file_section(label, file_errors))

    return "\n".join(lines)


def compute_verdict(
    calib_median: float | None, test_median: float | None, gap: float | None
) -> tuple[str, list[str]]:
    """The two conditions fixed at the top of the file. Never changed at runtime."""

    reasons: list[str] = []
    if test_median is None:
        reasons.append("test_median unavailable (no usable test samples)")
    elif test_median >= PASS_TEST_MEDIAN_PX:
        reasons.append(f"test_median {test_median:.0f}px >= {PASS_TEST_MEDIAN_PX:.0f}px")
    if gap is None:
        reasons.append("gap unavailable")
    elif gap >= PASS_GAP_PX:
        reasons.append(f"gap {gap:.0f}px >= {PASS_GAP_PX:.0f}px")
    return ("FAIL", reasons) if reasons else ("PASS", [])


# --- history.csv (the one file this script writes) -------------------------


def append_history(
    path: Path,
    *,
    note: str,
    model_id: str,
    calib_median: float | None,
    test_median: float | None,
    test_median_interp: float | None,
    test_median_extrap: float | None,
    gap: float | None,
    worst_point_label: str,
    verdict: str,
) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(_HISTORY_HEADER)
        writer.writerow(
            [
                datetime.now(UTC).isoformat(),
                note,
                model_id,
                "" if calib_median is None else f"{calib_median:.1f}",
                "" if test_median is None else f"{test_median:.1f}",
                "" if test_median_interp is None else f"{test_median_interp:.1f}",
                "" if test_median_extrap is None else f"{test_median_extrap:.1f}",
                "" if gap is None else f"{gap:.1f}",
                worst_point_label,
                verdict,
            ]
        )


# --- CLI ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help="calibration dataset JSON (omit with --predictions)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="held-out test dataset JSON (omit with --predictions)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="calibration model JSON to score both files with (omit with --predictions)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="pre-computed external predictions JSON, replacing --test/--model; "
        "see the module docstring for the format and what this mode does not report",
    )
    parser.add_argument("--note", required=True, help="what changed in this run")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="also score rejected samples that have usable features (see the module docstring)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.predictions is not None:
        if args.calib is not None or args.test is not None or args.model is not None:
            parser.error("--predictions cannot be combined with --calib/--test/--model")
        return _run_predictions(args)

    missing = [
        name
        for name, value in (("--calib", args.calib), ("--test", args.test), ("--model", args.model))
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return _run_classic(args)


def _run_predictions(args: argparse.Namespace) -> int:
    result = load_predictions(args.predictions)

    geometry_warnings = []
    geometry = result.screen_geometry
    if geometry.width_px != SCREEN_WIDTH_PX or geometry.height_px != SCREEN_HEIGHT_PX:
        geometry_warnings.append(
            f"predictions file screen_geometry ({geometry.width_px}x{geometry.height_px}) does not "
            f"match SCREEN_WIDTH_PX/SCREEN_HEIGHT_PX ({SCREEN_WIDTH_PX}x{SCREEN_HEIGHT_PX}); "
            f"pixel math below uses the file's own geometry, not the constants"
        )

    pred_errors = compute_prediction_errors(result, include_rejected=args.include_rejected)
    print(
        format_predictions_report(
            predictions=pred_errors,
            source_label=str(args.predictions),
            geometry_warnings=geometry_warnings,
        )
    )

    test_median = overall_median(pred_errors)
    test_interp = overall_median(pred_errors, only_extrap=False)
    test_extrap = overall_median(pred_errors, only_extrap=True)
    worst = worst_point(per_point_rows(pred_errors))
    worst_label = "n/a" if worst is None else worst.name

    append_history(
        args.history,
        note=args.note,
        model_id=f"predictions:{args.predictions.stem}",
        calib_median=None,
        test_median=test_median,
        test_median_interp=test_interp,
        test_median_extrap=test_extrap,
        gap=None,
        worst_point_label=worst_label,
        verdict="N/A",
    )
    print(f"History appended to: {args.history}")
    return 0


def _run_classic(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    calib_result = load_calibration(args.calib)
    test_result = load_test(args.test)

    geometry_warnings = []
    for label, geometry in (
        ("calibration file", calib_result.screen_geometry),
        ("test file", test_result.screen_geometry),
    ):
        if geometry.width_px != SCREEN_WIDTH_PX or geometry.height_px != SCREEN_HEIGHT_PX:
            geometry_warnings.append(
                f"{label} screen_geometry ({geometry.width_px}x{geometry.height_px}) does not "
                f"match SCREEN_WIDTH_PX/SCREEN_HEIGHT_PX ({SCREEN_WIDTH_PX}x{SCREEN_HEIGHT_PX}); "
                f"pixel math below uses the file's own geometry, not the constants"
            )

    calib_errors = compute_calibration_errors(
        model, calib_result, include_rejected=args.include_rejected
    )
    test_errors = compute_test_errors(model, test_result, include_rejected=args.include_rejected)

    print(
        format_report(
            model=model, calib=calib_errors, test=test_errors, geometry_warnings=geometry_warnings
        )
    )

    calib_median = overall_median(calib_errors)
    test_median = overall_median(test_errors)
    test_interp = overall_median(test_errors, only_extrap=False)
    test_extrap = overall_median(test_errors, only_extrap=True)
    gap = None if calib_median is None or test_median is None else test_median - calib_median
    verdict, _ = compute_verdict(calib_median, test_median, gap)
    worst = worst_point(per_point_rows(test_errors))
    worst_label = "n/a" if worst is None else worst.name

    append_history(
        args.history,
        note=args.note,
        model_id=model.model_id,
        calib_median=calib_median,
        test_median=test_median,
        test_median_interp=test_interp,
        test_median_extrap=test_extrap,
        gap=gap,
        worst_point_label=worst_label,
        verdict=verdict,
    )
    print(f"History appended to: {args.history}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
