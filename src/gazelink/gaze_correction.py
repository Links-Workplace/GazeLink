"""Pure, conservative local residual correction for a calibrated gaze model."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gazelink.calibration import FEATURE_SCHEMA_VERSION
from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    GazeSample,
    JSONValue,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_engine import GazeEstimator, normalized_to_pixel
from gazelink.gaze_features import GazeFeatureVector, from_observation

CORRECTION_FILE_VERSION = 1
DEFAULT_CORRECTION_DIR = Path(".gazelink/calibration")


def _finite(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ContractValidationError(f"{field_name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ContractValidationError(f"{field_name} must be <= {maximum}")
    return result


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{field_name} must be an integer >= 1")
    return value


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class CorrectionSettings:
    """Centralized, explicitly unvalidated local-correction thresholds."""

    influence_radius: float = 0.25  # UNVALIDATED PLACEHOLDER, normalized screen distance.
    shrinkage: float = 1.0  # UNVALIDATED PLACEHOLDER, unitless conservative prior.
    max_total_offset: float = 0.20  # UNVALIDATED PLACEHOLDER, normalized screen distance.
    conflict_residual_distance: float = 0.12  # UNVALIDATED PLACEHOLDER.
    capture_window_ms: float = 400.0  # UNVALIDATED PLACEHOLDER.
    min_capture_samples: int = 5  # UNVALIDATED PLACEHOLDER.
    max_iris_spread: float = 0.08  # UNVALIDATED PLACEHOLDER, eye-relative ratio.
    max_openness_spread: float = 0.15  # UNVALIDATED PLACEHOLDER, ratio.
    max_head_pose_spread_deg: float = 5.0  # UNVALIDATED PLACEHOLDER, degrees.

    def __post_init__(self) -> None:
        for field_name in (
            "influence_radius",
            "shrinkage",
            "max_total_offset",
            "conflict_residual_distance",
            "capture_window_ms",
            "max_iris_spread",
            "max_openness_spread",
            "max_head_pose_spread_deg",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name, minimum=1e-12),
            )
        _positive_int(self.min_capture_samples, "min_capture_samples")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "influence_radius": self.influence_radius,
            "shrinkage": self.shrinkage,
            "max_total_offset": self.max_total_offset,
            "conflict_residual_distance": self.conflict_residual_distance,
            "capture_window_ms": self.capture_window_ms,
            "min_capture_samples": self.min_capture_samples,
            "max_iris_spread": self.max_iris_spread,
            "max_openness_spread": self.max_openness_spread,
            "max_head_pose_spread_deg": self.max_head_pose_spread_deg,
        }

    @classmethod
    def from_dict(cls, value: object) -> CorrectionSettings:
        data = _mapping(value, "correction settings")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CorrectionSample:
    sample_id: str
    features: GazeFeatureVector
    base_raw_prediction: GazePoint
    true_target: GazePoint
    residual: GazePoint
    screen_geometry: ScreenGeometry
    base_model_id: str
    feature_schema_version: int
    captured_at_utc: str
    capture_sample_count: int
    source_frame_ids: tuple[int, ...]
    feature_spread: tuple[float, ...]

    def __post_init__(self) -> None:
        _nonempty(self.sample_id, "sample_id")
        if not isinstance(self.features, GazeFeatureVector):
            raise ContractValidationError("features must be a GazeFeatureVector")
        for field_name in ("base_raw_prediction", "true_target", "residual"):
            if not isinstance(getattr(self, field_name), GazePoint):
                raise ContractValidationError(f"{field_name} must be a GazePoint")
        if not 0.0 <= self.true_target.x <= 1.0 or not 0.0 <= self.true_target.y <= 1.0:
            raise ContractValidationError("true_target must be within normalized screen bounds")
        expected_x = self.true_target.x - self.base_raw_prediction.x
        expected_y = self.true_target.y - self.base_raw_prediction.y
        if not math.isclose(self.residual.x, expected_x, abs_tol=1e-12) or not math.isclose(
            self.residual.y, expected_y, abs_tol=1e-12
        ):
            raise ContractValidationError("residual must equal true_target - base_raw_prediction")
        if not isinstance(self.screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be ScreenGeometry")
        _nonempty(self.base_model_id, "base_model_id")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ContractValidationError("correction feature schema is incompatible")
        _nonempty(self.captured_at_utc, "captured_at_utc")
        _positive_int(self.capture_sample_count, "capture_sample_count")
        if (
            len(self.source_frame_ids) != self.capture_sample_count
            or len(set(self.source_frame_ids)) != len(self.source_frame_ids)
            or any(
                isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0
                for frame_id in self.source_frame_ids
            )
        ):
            raise ContractValidationError(
                "source_frame_ids must be unique non-negative IDs matching capture_sample_count"
            )
        if len(self.feature_spread) != GazeFeatureVector.size:
            raise ContractValidationError("feature_spread must match the feature vector size")
        object.__setattr__(
            self,
            "feature_spread",
            tuple(_finite(value, "feature_spread", minimum=0.0) for value in self.feature_spread),
        )
        expected_id = _correction_identity(self._identity_payload())
        if self.sample_id != expected_id:
            raise ContractValidationError("sample_id does not match correction contents")

    def _identity_payload(self) -> dict[str, JSONValue]:
        return {
            "features": self.features.to_dict(),
            "base_raw_prediction": self.base_raw_prediction.to_dict(),
            "true_target": self.true_target.to_dict(),
            "screen_geometry": self.screen_geometry.to_dict(),
            "base_model_id": self.base_model_id,
            "feature_schema_version": self.feature_schema_version,
            "captured_at_utc": self.captured_at_utc,
            "capture_sample_count": self.capture_sample_count,
            "source_frame_ids": list(self.source_frame_ids),
            "feature_spread": list(self.feature_spread),
        }

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "sample_id": self.sample_id,
            **self._identity_payload(),
            "residual": self.residual.to_dict(),
        }

    @classmethod
    def create(
        cls,
        *,
        features: GazeFeatureVector,
        base_raw_prediction: GazePoint,
        true_target: GazePoint,
        screen_geometry: ScreenGeometry,
        base_model_id: str,
        capture_sample_count: int,
        source_frame_ids: tuple[int, ...],
        feature_spread: tuple[float, ...],
        captured_at_utc: str | None = None,
    ) -> CorrectionSample:
        captured = captured_at_utc or datetime.now(UTC).isoformat()
        payload: dict[str, JSONValue] = {
            "features": features.to_dict(),
            "base_raw_prediction": base_raw_prediction.to_dict(),
            "true_target": true_target.to_dict(),
            "screen_geometry": screen_geometry.to_dict(),
            "base_model_id": base_model_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "captured_at_utc": captured,
            "capture_sample_count": capture_sample_count,
            "source_frame_ids": list(source_frame_ids),
            "feature_spread": list(feature_spread),
        }
        return cls(
            sample_id=_correction_identity(payload),
            features=features,
            base_raw_prediction=base_raw_prediction,
            true_target=true_target,
            residual=GazePoint(
                true_target.x - base_raw_prediction.x,
                true_target.y - base_raw_prediction.y,
            ),
            screen_geometry=screen_geometry,
            base_model_id=base_model_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            captured_at_utc=captured,
            capture_sample_count=capture_sample_count,
            source_frame_ids=source_frame_ids,
            feature_spread=feature_spread,
        )

    @classmethod
    def from_dict(cls, value: object) -> CorrectionSample:
        data = _mapping(value, "correction sample")
        spread = data.get("feature_spread")
        if not isinstance(spread, list):
            raise ContractValidationError("feature_spread must be a list")
        source_frame_ids = data.get("source_frame_ids")
        if not isinstance(source_frame_ids, list):
            raise ContractValidationError("source_frame_ids must be a list")
        return cls(
            sample_id=data.get("sample_id"),  # type: ignore[arg-type]
            features=GazeFeatureVector.from_dict(data.get("features")),
            base_raw_prediction=GazePoint.from_dict(data.get("base_raw_prediction")),
            true_target=GazePoint.from_dict(data.get("true_target")),
            residual=GazePoint.from_dict(data.get("residual")),
            screen_geometry=ScreenGeometry.from_dict(data.get("screen_geometry")),
            base_model_id=data.get("base_model_id"),  # type: ignore[arg-type]
            feature_schema_version=data.get("feature_schema_version"),  # type: ignore[arg-type]
            captured_at_utc=data.get("captured_at_utc"),  # type: ignore[arg-type]
            capture_sample_count=data.get("capture_sample_count"),  # type: ignore[arg-type]
            source_frame_ids=tuple(source_frame_ids),
            feature_spread=tuple(spread),
        )


@dataclass(frozen=True, slots=True)
class CorrectionApplication:
    corrected: GazePoint
    contributing_corrections: int
    reason_codes: tuple[ReasonCode, ...] = ()


@dataclass(frozen=True, slots=True)
class LocalCorrectionEngine:
    base_model_id: str
    screen_geometry: ScreenGeometry
    corrections: tuple[CorrectionSample, ...] = ()
    settings: CorrectionSettings = CorrectionSettings()
    enabled: bool = True
    feature_schema_version: int = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.base_model_id, "base_model_id")
        if not isinstance(self.screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be ScreenGeometry")
        if not isinstance(self.settings, CorrectionSettings):
            raise ContractValidationError("settings must be CorrectionSettings")
        if not isinstance(self.enabled, bool):
            raise ContractValidationError("enabled must be bool")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ContractValidationError("correction engine feature schema is incompatible")
        for correction in self.corrections:
            self._require_compatible(correction)

    def correct(self, raw: GazePoint) -> CorrectionApplication:
        if not isinstance(raw, GazePoint):
            raise ContractValidationError("raw must be GazePoint")
        if not self.enabled or not self.corrections:
            return CorrectionApplication(raw, 0)
        weighted: list[tuple[float, CorrectionSample]] = []
        for correction in self.corrections:
            distance = math.hypot(
                raw.x - correction.base_raw_prediction.x,
                raw.y - correction.base_raw_prediction.y,
            )
            if distance >= self.settings.influence_radius:
                continue
            ratio = distance / self.settings.influence_radius
            weight = (1.0 - (ratio * ratio)) ** 2
            weighted.append((weight, correction))
        if not weighted:
            return CorrectionApplication(raw, 0)
        residuals = [correction.residual for _weight, correction in weighted]
        conflict = max(
            (
                math.hypot(left.x - right.x, left.y - right.y)
                for index, left in enumerate(residuals)
                for right in residuals[index + 1 :]
            ),
            default=0.0,
        )
        if conflict > self.settings.conflict_residual_distance:
            return CorrectionApplication(raw, 0, (ReasonCode.CORRECTION_CONFLICT,))
        total_weight = sum(weight for weight, _correction in weighted)
        denominator = self.settings.shrinkage + total_weight
        delta_x = (
            sum(weight * correction.residual.x for weight, correction in weighted) / denominator
        )
        delta_y = (
            sum(weight * correction.residual.y for weight, correction in weighted) / denominator
        )
        magnitude = math.hypot(delta_x, delta_y)
        if magnitude > self.settings.max_total_offset:
            scale = self.settings.max_total_offset / magnitude
            delta_x *= scale
            delta_y *= scale
        return CorrectionApplication(GazePoint(raw.x + delta_x, raw.y + delta_y), len(weighted))

    def apply_to_sample(self, sample: GazeSample) -> GazeSample:
        if sample.screen_id != self.screen_geometry.screen_id:
            return replace(
                sample,
                corrected_normalized=sample.raw_normalized,
                filtered_normalized=sample.raw_normalized,
                reason_codes=(*sample.reason_codes, ReasonCode.CORRECTION_INCOMPATIBLE),
            )
        application = self.correct(sample.raw_normalized)
        reasons = list(sample.reason_codes)
        reasons.extend(application.reason_codes)
        if (
            application.corrected.x < 0.0
            or application.corrected.x > 1.0
            or application.corrected.y < 0.0
            or application.corrected.y > 1.0
        ):
            for reason in (ReasonCode.OUT_OF_RANGE, ReasonCode.CLAMPED_TO_SCREEN):
                if reason not in reasons:
                    reasons.append(reason)
        return replace(
            sample,
            corrected_normalized=application.corrected,
            filtered_normalized=application.corrected,
            screen_position=normalized_to_pixel(application.corrected, self.screen_geometry),
            reason_codes=tuple(reasons),
            valid_for_control=False,
        )

    def with_correction(self, correction: CorrectionSample) -> LocalCorrectionEngine:
        self._require_compatible(correction)
        return replace(self, corrections=(*self.corrections, correction))

    def undo_last(self) -> LocalCorrectionEngine:
        return replace(self, corrections=self.corrections[:-1])

    def clear(self) -> LocalCorrectionEngine:
        return replace(self, corrections=())

    def disable(self) -> LocalCorrectionEngine:
        return replace(self, enabled=False)

    def enable(self) -> LocalCorrectionEngine:
        return replace(self, enabled=True)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "file_version": CORRECTION_FILE_VERSION,
            "base_model_id": self.base_model_id,
            "screen_geometry": self.screen_geometry.to_dict(),
            "feature_schema_version": self.feature_schema_version,
            "enabled": self.enabled,
            "settings": self.settings.to_dict(),
            "corrections": [correction.to_dict() for correction in self.corrections],
        }

    @classmethod
    def from_dict(cls, value: object) -> LocalCorrectionEngine:
        data = _mapping(value, "local correction engine")
        if data.get("file_version") != CORRECTION_FILE_VERSION:
            raise ContractValidationError("unsupported correction file version")
        serialized = data.get("corrections")
        if not isinstance(serialized, list):
            raise ContractValidationError("corrections must be a list")
        return cls(
            base_model_id=data.get("base_model_id"),  # type: ignore[arg-type]
            screen_geometry=ScreenGeometry.from_dict(data.get("screen_geometry")),
            feature_schema_version=data.get("feature_schema_version"),  # type: ignore[arg-type]
            enabled=data.get("enabled"),  # type: ignore[arg-type]
            settings=CorrectionSettings.from_dict(data.get("settings")),
            corrections=tuple(CorrectionSample.from_dict(item) for item in serialized),
        )

    def _require_compatible(self, correction: CorrectionSample) -> None:
        if not isinstance(correction, CorrectionSample):
            raise ContractValidationError("correction must be CorrectionSample")
        if (
            correction.base_model_id != self.base_model_id
            or correction.feature_schema_version != self.feature_schema_version
            or correction.screen_geometry != self.screen_geometry
        ):
            raise ContractValidationError("correction is incompatible with this base model")


@dataclass(frozen=True, slots=True)
class CorrectionCaptureResult:
    candidate: CorrectionSample | None
    reason_codes: tuple[ReasonCode, ...]


@dataclass(frozen=True, slots=True)
class CorrectionValidationResult:
    candidate: CorrectionSample
    before_error_normalized: float
    after_error_normalized: float
    accepted: bool
    reason_codes: tuple[ReasonCode, ...]


def capture_correction(
    observations: Sequence[VisionObservation],
    *,
    true_target: GazePoint,
    estimator: GazeEstimator,
    now_monotonic_ms: float,
    settings: CorrectionSettings | None = None,
    captured_at_utc: str | None = None,
) -> CorrectionCaptureResult:
    capture_settings = settings or CorrectionSettings()
    aggregate, spread, frame_ids, reasons = _aggregate_capture(
        observations, now_monotonic_ms=now_monotonic_ms, settings=capture_settings
    )
    if aggregate is None:
        return CorrectionCaptureResult(None, reasons)
    if not 0.0 <= true_target.x <= 1.0 or not 0.0 <= true_target.y <= 1.0:
        raise ContractValidationError("true_target must be within normalized screen bounds")
    raw = estimator.predict_features(aggregate)
    candidate = CorrectionSample.create(
        features=aggregate,
        base_raw_prediction=raw,
        true_target=true_target,
        screen_geometry=estimator.screen_geometry,
        base_model_id=estimator.model_id,
        capture_sample_count=len(frame_ids),
        source_frame_ids=frame_ids,
        feature_spread=spread,
        captured_at_utc=captured_at_utc,
    )
    return CorrectionCaptureResult(candidate, ())


def validate_candidate(
    candidate: CorrectionSample,
    observations: Sequence[VisionObservation],
    *,
    validation_target: GazePoint,
    estimator: GazeEstimator,
    active_engine: LocalCorrectionEngine,
    now_monotonic_ms: float,
    settings: CorrectionSettings | None = None,
) -> CorrectionValidationResult:
    capture_settings = settings or active_engine.settings
    aggregate, _spread, frame_ids, reasons = _aggregate_capture(
        observations, now_monotonic_ms=now_monotonic_ms, settings=capture_settings
    )
    if aggregate is None:
        return CorrectionValidationResult(candidate, math.inf, math.inf, False, reasons)
    if set(frame_ids) & set(candidate.source_frame_ids):
        return CorrectionValidationResult(
            candidate,
            math.inf,
            math.inf,
            False,
            (ReasonCode.CORRECTION_INSUFFICIENT,),
        )
    raw = estimator.predict_features(aggregate)
    staged = active_engine.with_correction(candidate)
    application = staged.correct(raw)
    before = math.hypot(raw.x - validation_target.x, raw.y - validation_target.y)
    after = math.hypot(
        application.corrected.x - validation_target.x,
        application.corrected.y - validation_target.y,
    )
    if application.reason_codes:
        return CorrectionValidationResult(candidate, before, after, False, application.reason_codes)
    if after >= before:
        return CorrectionValidationResult(
            candidate, before, after, False, (ReasonCode.CORRECTION_NO_IMPROVEMENT,)
        )
    return CorrectionValidationResult(candidate, before, after, True, ())


def accept_candidate(
    validation: CorrectionValidationResult, engine: LocalCorrectionEngine
) -> LocalCorrectionEngine:
    if not validation.accepted:
        raise ContractValidationError("an unvalidated correction cannot be accepted")
    return engine.with_correction(validation.candidate)


def pixel_to_normalized(point: PixelPoint, geometry: ScreenGeometry) -> GazePoint:
    if not isinstance(point, PixelPoint):
        raise ContractValidationError("point must be PixelPoint")
    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be ScreenGeometry")
    if point.x_px >= geometry.width_px or point.y_px >= geometry.height_px:
        raise ContractValidationError("pixel target must be within screen bounds")
    return GazePoint(
        point.x_px / max(1, geometry.width_px - 1),
        point.y_px / max(1, geometry.height_px - 1),
    )


class CorrectionStore:
    def __init__(self, directory: Path = DEFAULT_CORRECTION_DIR) -> None:
        self._directory = directory

    def save(self, engine: LocalCorrectionEngine) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"corrections_{engine.base_model_id}.json"
        self._write(path, engine.to_dict())
        self._write(self._directory / "latest_corrections.json", engine.to_dict())
        return path

    def load(self, *, base_model_id: str, screen_geometry: ScreenGeometry) -> LocalCorrectionEngine:
        empty = LocalCorrectionEngine(base_model_id, screen_geometry)
        path = self._directory / "latest_corrections.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            loaded = LocalCorrectionEngine.from_dict(value)
        except (OSError, ValueError, TypeError, ContractValidationError):
            return empty
        if loaded.base_model_id != base_model_id or loaded.screen_geometry != screen_geometry:
            return empty
        return loaded

    def _write(self, path: Path, value: Mapping[str, JSONValue]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _aggregate_capture(
    observations: Sequence[VisionObservation],
    *,
    now_monotonic_ms: float,
    settings: CorrectionSettings,
) -> tuple[
    GazeFeatureVector | None,
    tuple[float, ...],
    tuple[int, ...],
    tuple[ReasonCode, ...],
]:
    now = _finite(now_monotonic_ms, "now_monotonic_ms", minimum=0.0)
    vectors: list[GazeFeatureVector] = []
    frame_ids: list[int] = []
    for observation in observations:
        if not isinstance(observation, VisionObservation):
            raise ContractValidationError("observations must contain VisionObservation values")
        age = now - observation.observed_at_monotonic_ms
        if (
            observation.tracking_state is TrackingState.TRACKED
            and 0.0 <= age <= settings.capture_window_ms
        ):
            features = from_observation(observation)
            if features is not None:
                vectors.append(features)
                frame_ids.append(observation.frame_id)
    if len(vectors) < settings.min_capture_samples:
        return None, (), tuple(frame_ids), (ReasonCode.CORRECTION_INSUFFICIENT,)
    matrix = np.asarray([vector.values for vector in vectors], dtype=np.float64)
    spread = tuple(float(value) for value in np.ptp(matrix, axis=0))
    if (
        max(spread[0:4]) > settings.max_iris_spread
        or max(spread[4:6]) > settings.max_openness_spread
        or max(spread[6:9]) > settings.max_head_pose_spread_deg
    ):
        return None, spread, tuple(frame_ids), (ReasonCode.CORRECTION_UNSTABLE,)
    median = tuple(float(value) for value in np.median(matrix, axis=0))
    return GazeFeatureVector(median), spread, tuple(frame_ids), ()


def _correction_identity(payload: Mapping[str, JSONValue]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
