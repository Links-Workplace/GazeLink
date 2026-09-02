"""Fresh live measurements for a calibration candidate, without OS input.

This is intentionally separate from training: every prediction is made by a
fully trained candidate on observations captured *after* its dataset was
closed.  It reports measurement rather than inventing a product pass/fail
threshold before M2 accuracy requirements are calibrated on real hardware.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import median

import numpy as np

from gazelink.calibration_ui import CalibrationTimingSettings
from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry
from gazelink.gaze_engine import GazeEstimationResult

_DEFAULT_VALIDATION_TIMING = CalibrationTimingSettings()
LIVE_VALIDATION_LOG_DIR = Path(".gazelink") / "validation_logs"


@dataclass(frozen=True, slots=True)
class LiveValidationTarget:
    """A known screen coordinate displayed after model training."""

    name: str
    screen_position: GazePoint


def default_live_validation_targets() -> tuple[LiveValidationTarget, ...]:
    """Return a compact set that includes centre and the reported upper area.

    The non-centre positions intentionally sit between the 9 calibration
    targets.  They expose interpolation problems instead of merely replaying
    the exact labels used for training.  Centre remains included because a
    fresh centre observation catches the most obvious live offset.
    """

    return (
        LiveValidationTarget("CENTER", GazePoint(0.50, 0.50)),
        LiveValidationTarget("UP_CENTER", GazePoint(0.50, 0.18)),
        LiveValidationTarget("UP_RIGHT", GazePoint(0.82, 0.18)),
        LiveValidationTarget("LEFT_CENTER", GazePoint(0.18, 0.50)),
        LiveValidationTarget("RIGHT_CENTER", GazePoint(0.82, 0.50)),
    )


class LiveValidationPhase(StrEnum):
    STABILIZING = "STABILIZING"
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class LiveValidationMeasurement:
    target: LiveValidationTarget
    predicted_normalized: GazePoint
    median_error_px: float
    p95_error_px: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class LiveValidationView:
    target: LiveValidationTarget | None
    target_number: int
    total_targets: int
    phase: LiveValidationPhase
    accepted_samples: int
    required_samples: int
    feedback: str
    measurements: tuple[LiveValidationMeasurement, ...]

    @property
    def complete(self) -> bool:
        return self.phase is LiveValidationPhase.COMPLETE


@dataclass(frozen=True, slots=True)
class LiveValidationReportPaths:
    text_path: Path
    json_path: Path


@dataclass(frozen=True, slots=True)
class LiveValidationCandidate:
    """One inert model measured against the same fresh camera observations."""

    model_id: str
    label: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.label:
            raise ContractValidationError("live validation candidate requires model_id and label")


@dataclass(frozen=True, slots=True)
class LiveValidationCandidateView:
    candidate: LiveValidationCandidate
    view: LiveValidationView


@dataclass(frozen=True, slots=True)
class LiveValidationComparisonView:
    """Synchronized validation results for comparable model candidates."""

    candidates: tuple[LiveValidationCandidateView, ...]
    recommended_model_id: str | None

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ContractValidationError("live validation comparison requires candidates")
        primary = self.candidates[0].view
        if any(
            item.view.target != primary.target
            or item.view.target_number != primary.target_number
            or item.view.phase is not primary.phase
            for item in self.candidates[1:]
        ):
            raise ContractValidationError("candidate validation views must stay synchronized")
        candidate_ids = {item.candidate.model_id for item in self.candidates}
        if self.recommended_model_id is not None and self.recommended_model_id not in candidate_ids:
            raise ContractValidationError("recommendation must name a measured candidate")

    @property
    def primary(self) -> LiveValidationView:
        return self.candidates[0].view

    @property
    def complete(self) -> bool:
        return self.primary.complete


def format_live_validation_summary(view: LiveValidationView) -> str:
    """Format aggregate-only data for the operator and a text artifact."""

    if not view.measurements:
        return "No validation point has been measured yet."
    lines = ["Fresh validation measurements (not control):"]
    for measurement in view.measurements:
        target = measurement.target.screen_position
        predicted = measurement.predicted_normalized
        lines.append(
            f"{measurement.target.name}: target=({target.x:.2f}, {target.y:.2f}) "
            f"predicted=({predicted.x:.2f}, {predicted.y:.2f}) "
            f"median={measurement.median_error_px:.0f}px "
            f"p95={measurement.p95_error_px:.0f}px"
        )
    return "\n".join(lines)


def format_live_validation_comparison_summary(view: LiveValidationComparisonView) -> str:
    """Format aggregate-only side-by-side results for live model selection."""

    lines = ["Fresh live comparison (not control):"]
    for candidate_view in view.candidates:
        lines.append(
            f"{candidate_view.candidate.label} [{candidate_view.candidate.model_id[:12]}]:"
        )
        for measurement in candidate_view.view.measurements:
            target = measurement.target.screen_position
            predicted = measurement.predicted_normalized
            lines.append(
                f"  {measurement.target.name}: target=({target.x:.2f}, {target.y:.2f}) "
                f"predicted=({predicted.x:.2f}, {predicted.y:.2f}) "
                f"median={measurement.median_error_px:.0f}px "
                f"p95={measurement.p95_error_px:.0f}px"
            )
    if view.recommended_model_id is None:
        lines.append("Recommendation: none; no candidate was no-worse at every measured target.")
    else:
        winner = next(
            item.candidate.label
            for item in view.candidates
            if item.candidate.model_id == view.recommended_model_id
        )
        lines.append(f"Recommendation: {winner}; it was no-worse at every measured target.")
    return "\n".join(lines)


def write_live_validation_report(
    view: LiveValidationView,
    *,
    model_id: str,
    geometry: ScreenGeometry,
    directory: Path = LIVE_VALIDATION_LOG_DIR,
    generated_at: datetime | None = None,
) -> LiveValidationReportPaths:
    """Persist reviewable aggregate validation evidence, never frames/features.

    The report contains target coordinates, aggregate predictions and errors;
    it deliberately excludes camera images, landmarks and per-frame eye/head
    features.  It is written as both readable text and lossless JSON so the
    user can share a path rather than transcribing a diagnostic window.
    """

    if not view.complete:
        raise ContractValidationError("only a completed validation can be reported")
    if not isinstance(model_id, str) or not model_id:
        raise ContractValidationError("model_id must be a non-empty string")
    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be a ScreenGeometry")
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": timestamp.isoformat(),
        "model_id": model_id,
        "screen_geometry": geometry.to_dict(),
        "measurements": [
            {
                "target_name": measurement.target.name,
                "target_normalized": measurement.target.screen_position.to_dict(),
                "predicted_normalized": measurement.predicted_normalized.to_dict(),
                "median_error_px": measurement.median_error_px,
                "p95_error_px": measurement.p95_error_px,
                "sample_count": measurement.sample_count,
            }
            for measurement in view.measurements
        ],
    }
    header = [
        "GAZELINK -- Live calibration validation",
        f"Generated: {timestamp.isoformat()}",
        f"Model ID: {model_id}",
        f"Screen: {geometry.screen_id} {geometry.width_px}x{geometry.height_px}",
        "No OS input was emitted.",
        "",
        format_live_validation_summary(view),
        "",
    ]
    text_path = directory / f"validation_{stamp}.txt"
    json_path = directory / f"validation_{stamp}.json"
    text_path.write_text("\n".join(header), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return LiveValidationReportPaths(text_path=text_path, json_path=json_path)


def write_live_validation_comparison_report(
    view: LiveValidationComparisonView,
    *,
    geometry: ScreenGeometry,
    directory: Path = LIVE_VALIDATION_LOG_DIR,
    generated_at: datetime | None = None,
) -> LiveValidationReportPaths:
    """Persist aggregate comparisons only; never frames, landmarks, or features."""

    if not view.complete:
        raise ContractValidationError("only a completed validation can be reported")
    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be ScreenGeometry")
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": timestamp.isoformat(),
        "screen_geometry": geometry.to_dict(),
        "recommended_model_id": view.recommended_model_id,
        "candidates": [
            {
                "model_id": candidate_view.candidate.model_id,
                "label": candidate_view.candidate.label,
                "measurements": [
                    {
                        "target_name": measurement.target.name,
                        "target_normalized": measurement.target.screen_position.to_dict(),
                        "predicted_normalized": measurement.predicted_normalized.to_dict(),
                        "median_error_px": measurement.median_error_px,
                        "p95_error_px": measurement.p95_error_px,
                        "sample_count": measurement.sample_count,
                    }
                    for measurement in candidate_view.view.measurements
                ],
            }
            for candidate_view in view.candidates
        ],
    }
    header = [
        "GAZELINK -- Live calibration model comparison",
        f"Generated: {timestamp.isoformat()}",
        f"Screen: {geometry.screen_id} {geometry.width_px}x{geometry.height_px}",
        "No OS input was emitted.",
        "",
        format_live_validation_comparison_summary(view),
        "",
    ]
    text_path = directory / f"validation_comparison_{stamp}.txt"
    json_path = directory / f"validation_comparison_{stamp}.json"
    text_path.write_text("\n".join(header), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return LiveValidationReportPaths(text_path=text_path, json_path=json_path)


def _pixel_error(predicted: GazePoint, target: GazePoint, geometry: ScreenGeometry) -> float:
    return math.hypot(
        (predicted.x - target.x) * (geometry.width_px - 1),
        (predicted.y - target.y) * (geometry.height_px - 1),
    )


class LiveValidationController:
    """Qt-free state machine for fresh candidate-model measurements."""

    def __init__(
        self,
        geometry: ScreenGeometry,
        *,
        targets: tuple[LiveValidationTarget, ...] = default_live_validation_targets(),
        timing: CalibrationTimingSettings = _DEFAULT_VALIDATION_TIMING,
    ) -> None:
        if not isinstance(geometry, ScreenGeometry):
            raise ContractValidationError("geometry must be a ScreenGeometry")
        if not targets or any(not isinstance(target, LiveValidationTarget) for target in targets):
            raise ContractValidationError("targets must contain LiveValidationTarget values")
        if not isinstance(timing, CalibrationTimingSettings):
            raise ContractValidationError("timing must be CalibrationTimingSettings")
        self._geometry = geometry
        self._targets = targets
        self._timing = timing
        self._target_index = 0
        self._phase = LiveValidationPhase.STABILIZING
        self._phase_started_ms: float | None = None
        self._last_frame_id: int | None = None
        self._last_candidate_ms: float | None = None
        self._candidates: list[GazePoint] = []
        self._measurements: list[LiveValidationMeasurement] = []
        self._feedback = "Look at the highlighted point and hold still."

    def ingest(
        self, result: GazeEstimationResult, *, now_monotonic_ms: float
    ) -> LiveValidationView:
        """Collect only a fresh vetted base-model prediction.

        A rejected gaze estimate resets the current target rather than
        converting absent tracking into a fake coordinate or a silent pass.
        """

        if not isinstance(result, GazeEstimationResult):
            raise ContractValidationError("result must be a GazeEstimationResult")
        if not math.isfinite(now_monotonic_ms) or now_monotonic_ms < 0.0:
            raise ContractValidationError("now_monotonic_ms must be a non-negative finite number")
        if self._phase is LiveValidationPhase.COMPLETE:
            return self.view()
        if result.sample is None:
            self._reset_target(now_monotonic_ms)
            self._feedback = "Tracking was withheld; keep both eyes visible and retry this point."
            return self.view()
        if result.sample.source_frame_id == self._last_frame_id:
            return self.view()
        self._last_frame_id = result.sample.source_frame_id
        if self._phase_started_ms is None:
            self._phase_started_ms = now_monotonic_ms
        if self._phase is LiveValidationPhase.STABILIZING:
            elapsed = now_monotonic_ms - self._phase_started_ms
            if elapsed < self._timing.stabilization_ms:
                remaining_s = (self._timing.stabilization_ms - elapsed) / 1000
                self._feedback = f"Hold still; capture starts in {remaining_s:.1f}s."
                return self.view()
            self._phase = LiveValidationPhase.COLLECTING
            self._phase_started_ms = now_monotonic_ms
            self._last_candidate_ms = None
        if (
            self._last_candidate_ms is not None
            and now_monotonic_ms - self._last_candidate_ms < self._timing.min_sample_interval_ms
        ):
            return self.view()
        self._candidates.append(result.sample.raw_normalized)
        self._last_candidate_ms = now_monotonic_ms
        elapsed = now_monotonic_ms - self._phase_started_ms
        if len(self._candidates) < 5 or elapsed < self._timing.capture_window_ms:
            self._feedback = "Collecting fresh validation samples; keep looking at the point."
            return self.view()
        self._complete_target()
        return self.view()

    def restart(self) -> LiveValidationView:
        self._target_index = 0
        self._phase = LiveValidationPhase.STABILIZING
        self._phase_started_ms = None
        self._last_frame_id = None
        self._last_candidate_ms = None
        self._candidates.clear()
        self._measurements.clear()
        self._feedback = "Validation restarted. Look at the highlighted point and hold still."
        return self.view()

    def view(self) -> LiveValidationView:
        target = (
            None
            if self._phase is LiveValidationPhase.COMPLETE
            else self._targets[self._target_index]
        )
        return LiveValidationView(
            target=target,
            target_number=len(self._measurements) + (0 if target is None else 1),
            total_targets=len(self._targets),
            phase=self._phase,
            accepted_samples=len(self._candidates),
            required_samples=5,
            feedback=self._feedback,
            measurements=tuple(self._measurements),
        )

    def _reset_target(self, now_monotonic_ms: float) -> None:
        self._phase = LiveValidationPhase.STABILIZING
        self._phase_started_ms = now_monotonic_ms
        self._last_candidate_ms = None
        self._candidates.clear()

    def _complete_target(self) -> None:
        target = self._targets[self._target_index]
        x = median(point.x for point in self._candidates)
        y = median(point.y for point in self._candidates)
        prediction = GazePoint(x, y)
        errors = [
            _pixel_error(point, target.screen_position, self._geometry)
            for point in self._candidates
        ]
        self._measurements.append(
            LiveValidationMeasurement(
                target=target,
                predicted_normalized=prediction,
                median_error_px=float(median(errors)),
                p95_error_px=float(np.percentile(errors, 95)),
                sample_count=len(self._candidates),
            )
        )
        self._target_index += 1
        self._candidates.clear()
        self._last_candidate_ms = None
        if self._target_index == len(self._targets):
            self._phase = LiveValidationPhase.COMPLETE
            self._feedback = (
                "Validation measured. Review errors, then accept or reject the candidate."
            )
            return
        self._phase = LiveValidationPhase.STABILIZING
        self._phase_started_ms = None
        self._feedback = "Point measured. Move your gaze to the next highlighted point."


class LiveValidationComparisonController:
    """Measure candidate models in lockstep on the exact same fresh frames."""

    def __init__(
        self,
        geometry: ScreenGeometry,
        *,
        candidates: tuple[LiveValidationCandidate, ...],
        targets: tuple[LiveValidationTarget, ...] = default_live_validation_targets(),
        timing: CalibrationTimingSettings = _DEFAULT_VALIDATION_TIMING,
    ) -> None:
        if not candidates or len({candidate.model_id for candidate in candidates}) != len(
            candidates
        ):
            raise ContractValidationError("comparison candidates must have unique model identities")
        self._candidates = candidates
        self._controllers = tuple(
            LiveValidationController(geometry, targets=targets, timing=timing) for _ in candidates
        )

    def ingest(
        self,
        results: Mapping[str, GazeEstimationResult],
        *,
        now_monotonic_ms: float,
    ) -> LiveValidationComparisonView:
        if set(results) != {candidate.model_id for candidate in self._candidates}:
            raise ContractValidationError(
                "comparison results must cover every candidate exactly once"
            )
        values = tuple(results[candidate.model_id] for candidate in self._candidates)
        if any(result.sample is None for result in values):
            withheld = next(result for result in values if result.sample is None)
            views = tuple(
                controller.ingest(withheld, now_monotonic_ms=now_monotonic_ms)
                for controller in self._controllers
            )
        else:
            views = tuple(
                controller.ingest(result, now_monotonic_ms=now_monotonic_ms)
                for controller, result in zip(self._controllers, values, strict=True)
            )
        return self._view(views)

    def restart(self) -> LiveValidationComparisonView:
        return self._view(tuple(controller.restart() for controller in self._controllers))

    def view(self) -> LiveValidationComparisonView:
        return self._view(tuple(controller.view() for controller in self._controllers))

    def _view(self, views: tuple[LiveValidationView, ...]) -> LiveValidationComparisonView:
        candidate_views = tuple(
            LiveValidationCandidateView(candidate, view)
            for candidate, view in zip(self._candidates, views, strict=True)
        )
        recommended = _live_recommendation(candidate_views)
        return LiveValidationComparisonView(candidate_views, recommended)


def _live_recommendation(
    candidates: tuple[LiveValidationCandidateView, ...],
) -> str | None:
    """Recommend only a candidate that does not regress any measured target."""

    if not candidates or not candidates[0].view.complete:
        return None
    if len(candidates) == 1:
        return candidates[0].candidate.model_id
    winners = [
        candidate
        for candidate in candidates
        if all(candidate is other or _dominates(candidate.view, other.view) for other in candidates)
    ]
    return winners[0].candidate.model_id if len(winners) == 1 else None


def _dominates(left: LiveValidationView, right: LiveValidationView) -> bool:
    """Return whether one model is no worse at every target and better once."""

    if len(left.measurements) != len(right.measurements):
        return False
    no_worse = True
    strictly_better = False
    for left_measurement, right_measurement in zip(
        left.measurements, right.measurements, strict=True
    ):
        if left_measurement.target != right_measurement.target:
            return False
        if (
            left_measurement.median_error_px > right_measurement.median_error_px
            or left_measurement.p95_error_px > right_measurement.p95_error_px
        ):
            no_worse = False
        if (
            left_measurement.median_error_px < right_measurement.median_error_px
            or left_measurement.p95_error_px < right_measurement.p95_error_px
        ):
            strictly_better = True
    return no_worse and strictly_better
