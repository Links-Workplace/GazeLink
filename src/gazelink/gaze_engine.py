"""Calibration training, live gaze estimation, screen mapping, and persistence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gazelink.calibration import FEATURE_SCHEMA_VERSION, CalibrationSessionResult
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
from gazelink.gaze_features import (
    GazeFeatureVector,
    from_calibration_sample,
    from_observation,
)
from gazelink.gaze_model import (
    DEFAULT_RIDGE_LAMBDA,
    GazeModelKind,
    LabeledGazeRow,
    ModelComparison,
    RegressionModel,
    compare_models,
    fit_model,
)

DEFAULT_MAX_GAZE_SAMPLE_AGE_MS = 250.0  # UNVALIDATED PLACEHOLDER for M2 live estimation.
DEFAULT_CALIBRATION_DIR = Path(".gazelink/calibration")
MODEL_FILE_VERSION = 1
PENDING_VALIDATION_FILE = "pending_validation.json"


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return value


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _positive_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ContractValidationError(f"{field_name} must be a positive finite number")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    regression: RegressionModel
    feature_schema_version: int
    screen_geometry: ScreenGeometry
    camera_id: str
    calibration_id: str
    trained_at_utc: str
    model_id: str
    file_version: int = MODEL_FILE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.regression, RegressionModel):
            raise ContractValidationError("regression must be a RegressionModel")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported feature schema {self.feature_schema_version}; expected "
                f"{FEATURE_SCHEMA_VERSION}"
            )
        if not isinstance(self.screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be ScreenGeometry")
        _nonempty(self.camera_id, "camera_id")
        _nonempty(self.calibration_id, "calibration_id")
        _nonempty(self.trained_at_utc, "trained_at_utc")
        _nonempty(self.model_id, "model_id")
        if self.file_version != MODEL_FILE_VERSION:
            raise ContractValidationError(f"unsupported model file version {self.file_version}")
        expected = _model_identity(
            self.regression,
            self.feature_schema_version,
            self.screen_geometry,
            self.camera_id,
            self.calibration_id,
        )
        if self.model_id != expected:
            raise ContractValidationError("model_id does not match the serialized model contents")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "file_version": self.file_version,
            "model_id": self.model_id,
            "regression": self.regression.to_dict(),
            "feature_schema_version": self.feature_schema_version,
            "screen_geometry": self.screen_geometry.to_dict(),
            "camera_id": self.camera_id,
            "calibration_id": self.calibration_id,
            "trained_at_utc": self.trained_at_utc,
        }

    @classmethod
    def create(
        cls,
        *,
        regression: RegressionModel,
        screen_geometry: ScreenGeometry,
        camera_id: str,
        calibration_id: str,
        trained_at_utc: str | None = None,
    ) -> CalibrationModel:
        identity = _model_identity(
            regression,
            FEATURE_SCHEMA_VERSION,
            screen_geometry,
            camera_id,
            calibration_id,
        )
        return cls(
            regression=regression,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            screen_geometry=screen_geometry,
            camera_id=camera_id,
            calibration_id=calibration_id,
            trained_at_utc=trained_at_utc or datetime.now(UTC).isoformat(),
            model_id=identity,
        )

    @classmethod
    def from_dict(cls, value: object) -> CalibrationModel:
        data = _mapping(value, "calibration model")
        return cls(
            file_version=data.get("file_version"),  # type: ignore[arg-type]
            model_id=data.get("model_id"),  # type: ignore[arg-type]
            regression=RegressionModel.from_dict(data.get("regression")),
            feature_schema_version=data.get("feature_schema_version"),  # type: ignore[arg-type]
            screen_geometry=ScreenGeometry.from_dict(data.get("screen_geometry")),
            camera_id=data.get("camera_id"),  # type: ignore[arg-type]
            calibration_id=data.get("calibration_id"),  # type: ignore[arg-type]
            trained_at_utc=data.get("trained_at_utc"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CalibrationTrainingResult:
    model: CalibrationModel
    comparison: ModelComparison
    promotable: bool
    quality_reasons: tuple[str, ...]
    candidate_models: tuple[CalibrationModel, ...]


@dataclass(frozen=True, slots=True)
class PendingCalibration:
    """A trained candidate which is deliberately not active yet.

    M2 has no calibrated product-accuracy threshold.  A model which merely
    beats the trivial "always center" predictor may still be visibly wrong
    for the person who just calibrated.  The candidate therefore remains
    separate from ``latest_model.json`` until a fresh, live validation run is
    reviewed and explicitly accepted.  Only model metadata is persisted here;
    the lossless calibration dataset remains in its own timestamped artifact.
    """

    model: CalibrationModel
    candidate_path: Path
    candidate_models: tuple[CalibrationModel, ...]

    def __post_init__(self) -> None:
        if not self.candidate_models or self.model not in self.candidate_models:
            raise ContractValidationError("pending calibration must include its selected model")


@dataclass(frozen=True, slots=True)
class GazeEstimationResult:
    """Typed success/rejection result; rejected input never gets a fake coordinate."""

    sample: GazeSample | None
    reason_codes: tuple[ReasonCode, ...]

    def __post_init__(self) -> None:
        if self.sample is not None and not isinstance(self.sample, GazeSample):
            raise ContractValidationError("sample must be GazeSample or None")
        if any(not isinstance(reason, ReasonCode) for reason in self.reason_codes):
            raise ContractValidationError("reason_codes must contain ReasonCode values")
        if self.sample is None and not self.reason_codes:
            raise ContractValidationError("a rejected estimation requires a reason code")


class CalibrationEngine:
    def __init__(self, *, ridge_lambda: float = DEFAULT_RIDGE_LAMBDA) -> None:
        self._ridge_lambda = _positive_finite(ridge_lambda, "ridge_lambda")

    def train(self, result: CalibrationSessionResult) -> CalibrationTrainingResult:
        if not isinstance(result, CalibrationSessionResult):
            raise ContractValidationError("result must be CalibrationSessionResult")
        if result.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ContractValidationError("calibration feature schema is incompatible")
        target_positions = {
            target.index: GazePoint(target.screen_position.x, target.screen_position.y)
            for target in result.targets
        }
        rows: list[LabeledGazeRow] = []
        for sample in result.samples:
            features = from_calibration_sample(sample)
            if features is not None:
                rows.append(
                    LabeledGazeRow(
                        features=features,
                        target_index=sample.target_index,
                        target=target_positions[sample.target_index],
                    )
                )
        if len(rows) < 2 or len({row.target_index for row in rows}) != len(result.targets):
            raise ContractValidationError(
                "training requires usable accepted samples for every calibration target"
            )
        comparison = compare_models(rows, result.screen_geometry, ridge_lambda=self._ridge_lambda)
        calibration_id = _calibration_identity(result)
        linear_model = CalibrationModel.create(
            regression=fit_model(
                GazeModelKind.LINEAR,
                [row.features for row in rows],
                [row.target for row in rows],
                ridge_lambda=self._ridge_lambda,
            ),
            screen_geometry=result.screen_geometry,
            camera_id=result.camera_id,
            calibration_id=calibration_id,
        )
        advanced_model = CalibrationModel.create(
            regression=fit_model(
                GazeModelKind.POLYNOMIAL_RIDGE,
                [row.features for row in rows],
                [row.target for row in rows],
                ridge_lambda=self._ridge_lambda,
            ),
            screen_geometry=result.screen_geometry,
            camera_id=result.camera_id,
            calibration_id=calibration_id,
        )
        model = linear_model if comparison.selected_kind is GazeModelKind.LINEAR else advanced_model
        selected_metrics = (
            comparison.baseline
            if comparison.selected_kind.value == "LINEAR"
            else comparison.advanced
        )
        quality_reasons: list[str] = []
        if selected_metrics.median_error_px >= comparison.center_baseline.median_error_px:
            quality_reasons.append("median_error_did_not_beat_center_baseline")
        if selected_metrics.p95_error_px >= comparison.center_baseline.p95_error_px:
            quality_reasons.append("p95_error_did_not_beat_center_baseline")
        return CalibrationTrainingResult(
            model=model,
            comparison=comparison,
            promotable=not quality_reasons,
            quality_reasons=tuple(quality_reasons),
            candidate_models=(linear_model, advanced_model),
        )


class GazeEstimator:
    def __init__(
        self,
        model: CalibrationModel,
        *,
        live_screen_geometry: ScreenGeometry,
        max_sample_age_ms: float = DEFAULT_MAX_GAZE_SAMPLE_AGE_MS,
    ) -> None:
        if not isinstance(model, CalibrationModel):
            raise ContractValidationError("model must be a CalibrationModel")
        if not isinstance(live_screen_geometry, ScreenGeometry):
            raise ContractValidationError("live_screen_geometry must be ScreenGeometry")
        self._model = model
        self._live_geometry = live_screen_geometry
        self._max_sample_age_ms = _positive_finite(max_sample_age_ms, "max_sample_age_ms")

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def screen_geometry(self) -> ScreenGeometry:
        return self._live_geometry

    def predict_features(self, features: GazeFeatureVector) -> GazePoint:
        """Predict base-model gaze for an already-vetted aggregate feature vector."""

        if self._live_geometry != self._model.screen_geometry:
            raise ContractValidationError("calibration model is incompatible with live geometry")
        return self._model.regression.predict(features)

    def estimate(
        self, observation: VisionObservation, *, now_monotonic_ms: float
    ) -> GazeEstimationResult:
        if not isinstance(observation, VisionObservation):
            raise ContractValidationError("observation must be VisionObservation")
        if self._live_geometry != self._model.screen_geometry:
            return GazeEstimationResult(None, (ReasonCode.CALIBRATION_INVALID,))
        now = _positive_finite(now_monotonic_ms, "now_monotonic_ms")
        age_ms = now - observation.observed_at_monotonic_ms
        if age_ms < 0.0 or age_ms > self._max_sample_age_ms:
            return GazeEstimationResult(None, (ReasonCode.CALIBRATION_STALE,))
        if observation.tracking_state is not TrackingState.TRACKED:
            observation_reasons = observation.reason_codes or (ReasonCode.LOW_CONFIDENCE,)
            return GazeEstimationResult(None, observation_reasons)
        features = from_observation(observation)
        if features is None:
            return GazeEstimationResult(None, (ReasonCode.CALIBRATION_INVALID,))
        raw = self._model.regression.predict(features)
        output_reasons: list[ReasonCode] = []
        if not 0.0 <= raw.x <= 1.0 or not 0.0 <= raw.y <= 1.0:
            output_reasons.extend((ReasonCode.OUT_OF_RANGE, ReasonCode.CLAMPED_TO_SCREEN))
        pixel = normalized_to_pixel(raw, self._live_geometry)
        sample = GazeSample(
            source_frame_id=observation.frame_id,
            sampled_at_monotonic_ms=now,
            raw_normalized=raw,
            corrected_normalized=raw,
            filtered_normalized=raw,
            screen_position=pixel,
            screen_id=self._live_geometry.screen_id,
            confidence=observation.overall_confidence,
            valid_for_control=False,
            reason_codes=tuple(output_reasons),
        )
        return GazeEstimationResult(sample, tuple(output_reasons))


def normalized_to_pixel(point: GazePoint, geometry: ScreenGeometry) -> PixelPoint:
    if not isinstance(point, GazePoint):
        raise ContractValidationError("point must be GazePoint")
    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be ScreenGeometry")
    x = min(1.0, max(0.0, point.x))
    y = min(1.0, max(0.0, point.y))
    return PixelPoint(round(x * (geometry.width_px - 1)), round(y * (geometry.height_px - 1)))


class CalibrationStore:
    """JSON persistence with recoverable reads and no biometric image data."""

    def __init__(self, directory: Path = DEFAULT_CALIBRATION_DIR) -> None:
        self._directory = directory

    def save_dataset(self, result: CalibrationSessionResult) -> Path:
        return self._save_timestamped("dataset", result.to_dict())

    def save_model(self, model: CalibrationModel) -> Path:
        timestamped = self._save_timestamped("model", model.to_dict())
        self._write_json(self._directory / "latest_model.json", model.to_dict())
        return timestamped

    def save_pending_model(self, model: CalibrationModel) -> PendingCalibration:
        """Store a candidate without replacing the previously active model."""

        candidate_path = self._save_timestamped("candidate_model", model.to_dict())
        self._write_json(
            self._directory / PENDING_VALIDATION_FILE,
            {
                "file_version": MODEL_FILE_VERSION,
                "candidate_model_file": candidate_path.name,
                "model_id": model.model_id,
                "screen_geometry": model.screen_geometry.to_dict(),
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return PendingCalibration(
            model=model, candidate_path=candidate_path, candidate_models=(model,)
        )

    def save_pending_models(
        self, models: tuple[CalibrationModel, ...], *, selected_model_id: str
    ) -> PendingCalibration:
        """Store comparable candidates from one closed calibration session."""

        if not models or len({model.model_id for model in models}) != len(models):
            raise ContractValidationError("pending candidates must have unique model identities")
        selected = next((model for model in models if model.model_id == selected_model_id), None)
        if selected is None:
            raise ContractValidationError("selected model must be one of the pending candidates")
        if any(
            model.screen_geometry != selected.screen_geometry
            or model.camera_id != selected.camera_id
            or model.calibration_id != selected.calibration_id
            for model in models
        ):
            raise ContractValidationError("pending candidates must share calibration identity")
        paths = tuple(
            self._save_timestamped("candidate_model", model.to_dict()) for model in models
        )
        selected_path = paths[models.index(selected)]
        self._write_json(
            self._directory / PENDING_VALIDATION_FILE,
            {
                "file_version": MODEL_FILE_VERSION,
                "candidate_model_file": selected_path.name,
                "model_id": selected.model_id,
                "candidate_models": [
                    {
                        "candidate_model_file": path.name,
                        "model_id": model.model_id,
                        "kind": model.regression.kind.value,
                    }
                    for model, path in zip(models, paths, strict=True)
                ],
                "screen_geometry": selected.screen_geometry.to_dict(),
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return PendingCalibration(
            model=selected, candidate_path=selected_path, candidate_models=models
        )

    def load_pending_model(self, live_geometry: ScreenGeometry) -> PendingCalibration | None:
        """Load candidate models awaiting live validation, fail-closed."""

        manifest_path = self._directory / PENDING_VALIDATION_FILE
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = _mapping(manifest_data, "pending validation")
            expected_model_id = _nonempty(manifest.get("model_id"), "model_id")
            entries = manifest.get("candidate_models")
            if isinstance(entries, list):
                loaded: list[tuple[CalibrationModel, Path]] = []
                for entry in entries:
                    entry_data = _mapping(entry, "pending candidate")
                    filename = _nonempty(
                        entry_data.get("candidate_model_file"), "candidate_model_file"
                    )
                    candidate_path = self._directory / filename
                    if candidate_path.name != filename:
                        return None
                    model = CalibrationModel.from_dict(
                        json.loads(candidate_path.read_text(encoding="utf-8"))
                    )
                    if model.model_id != _nonempty(entry_data.get("model_id"), "model_id"):
                        return None
                    loaded.append((model, candidate_path))
                if not loaded:
                    return None
                selected_pair = next(
                    ((item, path) for item, path in loaded if item.model_id == expected_model_id),
                    None,
                )
                if selected_pair is None:
                    return None
                model, candidate_path = selected_pair
                candidate_models = tuple(item for item, _path in loaded)
            else:
                filename = _nonempty(manifest.get("candidate_model_file"), "candidate_model_file")
                candidate_path = self._directory / filename
                if candidate_path.name != filename:
                    return None
                model = CalibrationModel.from_dict(
                    json.loads(candidate_path.read_text(encoding="utf-8"))
                )
                candidate_models = (model,)
        except (OSError, ValueError, TypeError, ContractValidationError):
            return None
        if (
            model.model_id != expected_model_id
            or model.screen_geometry != live_geometry
            or any(
                item.screen_geometry != live_geometry
                or item.camera_id != model.camera_id
                or item.calibration_id != model.calibration_id
                for item in candidate_models
            )
        ):
            return None
        return PendingCalibration(
            model=model, candidate_path=candidate_path, candidate_models=candidate_models
        )

    def promote_pending_model(
        self, live_geometry: ScreenGeometry, *, model_id: str | None = None
    ) -> Path | None:
        """Make the reviewed candidate active and archive its manifest."""

        pending = self.load_pending_model(live_geometry)
        if pending is None:
            return None
        model = next(
            (
                candidate
                for candidate in pending.candidate_models
                if candidate.model_id == (model_id or pending.model.model_id)
            ),
            None,
        )
        if model is None:
            return None
        model_path = self.save_model(model)
        self._archive_pending_manifest("accepted_pending_validation")
        return model_path

    def reject_pending_model(self) -> Path | None:
        """Archive a rejected candidate manifest without deleting diagnostics."""

        return self._archive_pending_manifest("rejected_pending_validation")

    def load_latest_model(self, live_geometry: ScreenGeometry) -> CalibrationModel | None:
        path = self._directory / "latest_model.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            model = CalibrationModel.from_dict(data)
        except (OSError, ValueError, TypeError, ContractValidationError):
            return None
        return model if model.screen_geometry == live_geometry else None

    def quarantine_latest_model(self) -> Path | None:
        """Move a previously active model aside after a failed calibration gate."""

        path = self._directory / "latest_model.json"
        if not path.exists():
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self._directory / f"rejected_latest_model_{stamp}.json"
        path.replace(destination)
        return destination

    def _archive_pending_manifest(self, prefix: str) -> Path | None:
        path = self._directory / PENDING_VALIDATION_FILE
        if not path.exists():
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self._directory / f"{prefix}_{stamp}.json"
        path.replace(destination)
        return destination

    def _save_timestamped(self, prefix: str, value: Mapping[str, JSONValue]) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._directory / f"{prefix}_{stamp}.json"
        self._write_json(path, value)
        return path

    def _write_json(self, path: Path, value: Mapping[str, JSONValue]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _model_identity(
    regression: RegressionModel,
    feature_schema_version: int,
    geometry: ScreenGeometry,
    camera_id: str,
    calibration_id: str,
) -> str:
    core = {
        "regression": regression.to_dict(),
        "feature_schema_version": feature_schema_version,
        "screen_geometry": geometry.to_dict(),
        "camera_id": camera_id,
        "calibration_id": calibration_id,
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_identity(result: CalibrationSessionResult) -> str:
    encoded = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
