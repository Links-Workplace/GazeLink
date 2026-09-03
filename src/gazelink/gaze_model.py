"""Small-data, NumPy-only gaze regression and target-held-out benchmarking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gazelink.domain import ContractValidationError, GazePoint, JSONValue, ScreenGeometry
from gazelink.gaze_features import GazeFeatureVector

DEFAULT_RIDGE_LAMBDA = 1e-3  # UNVALIDATED PLACEHOLDER: tune with varied real sessions.
_POLYNOMIAL_SOURCE_INDICES = (0, 1, 2, 3, 4, 5)
_ALL_FEATURE_INDICES = tuple(range(GazeFeatureVector.size))
_X_AXIS_FEATURE_INDICES = (0, 2, 6)
# head_pitch_deg (7) is deliberately excluded here. On a 2026-09-02 nine-point
# session it correlated 0.72 with left_iris_y: on this ultrawide screen the
# calibration head naturally pitches with gaze, so least-squares regression
# splits weight between the two. Leave-one-target-out CV on that dataset
# showed dropping pitch cut worst-case (P95) normalized Y error from 0.40 to
# 0.27 -- the coupling doesn't hold once live head movement differs from the
# calibration session. Adding it back with AXIS_IRIS_LIDS made that profile's
# P95 worse (0.57 -> 0.58), so it stays there; each profile's feature set is
# an evidence-based choice, not meant to mirror the other.
_Y_AXIS_IRIS_FEATURE_INDICES = (1, 3)
_Y_AXIS_IRIS_LIDS_FEATURE_INDICES = (1, 3, 4, 5, 7)


class GazeModelKind(StrEnum):
    LINEAR = "LINEAR"
    POLYNOMIAL_RIDGE = "POLYNOMIAL_RIDGE"


class GazeModelProfile(StrEnum):
    """Persisted feature routing for independent screen axes."""

    FULL = "FULL"
    AXIS_IRIS = "AXIS_IRIS"
    AXIS_IRIS_LIDS = "AXIS_IRIS_LIDS"


def _finite(value: object, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractValidationError(f"{field_name} must be finite and >= {minimum}")
    return result


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class LabeledGazeRow:
    features: GazeFeatureVector
    target_index: int
    target: GazePoint

    def __post_init__(self) -> None:
        if not isinstance(self.features, GazeFeatureVector):
            raise ContractValidationError("features must be a GazeFeatureVector")
        if isinstance(self.target_index, bool) or not isinstance(self.target_index, int):
            raise ContractValidationError("target_index must be an integer")
        if not isinstance(self.target, GazePoint):
            raise ContractValidationError("target must be a GazePoint")


@dataclass(frozen=True, slots=True)
class RegressionModel:
    """Fitted coefficients for independent normalized X and Y regressors."""

    kind: GazeModelKind
    coefficients_x: tuple[float, ...]
    coefficients_y: tuple[float, ...]
    ridge_lambda: float
    profile: GazeModelProfile = GazeModelProfile.FULL
    offsets_x: tuple[float, ...] = ()
    scales_x: tuple[float, ...] = ()
    offsets_y: tuple[float, ...] = ()
    scales_y: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GazeModelKind):
            raise ContractValidationError("kind must be a GazeModelKind")
        if not isinstance(self.profile, GazeModelProfile):
            raise ContractValidationError("profile must be a GazeModelProfile")
        indices_x, indices_y = _profile_indices(self.profile, self.kind)
        expected_x = _design_width(self.kind, len(indices_x))
        expected_y = _design_width(self.kind, len(indices_y))
        if len(self.coefficients_x) != expected_x or len(self.coefficients_y) != expected_y:
            raise ContractValidationError(
                f"{self.kind.value}/{self.profile.value} coefficients must have lengths "
                f"x={expected_x}, y={expected_y}"
            )
        object.__setattr__(
            self,
            "coefficients_x",
            tuple(_finite(value, "coefficients_x") for value in self.coefficients_x),
        )
        object.__setattr__(
            self,
            "coefficients_y",
            tuple(_finite(value, "coefficients_y") for value in self.coefficients_y),
        )
        object.__setattr__(
            self, "ridge_lambda", _finite(self.ridge_lambda, "ridge_lambda", minimum=0.0)
        )
        for axis, indices in (("x", indices_x), ("y", indices_y)):
            offsets = getattr(self, f"offsets_{axis}") or (0.0,) * len(indices)
            scales = getattr(self, f"scales_{axis}") or (1.0,) * len(indices)
            if len(offsets) != len(indices) or len(scales) != len(indices):
                raise ContractValidationError(
                    f"{axis}-axis preprocessing must match its selected feature count"
                )
            object.__setattr__(
                self,
                f"offsets_{axis}",
                tuple(_finite(value, f"offsets_{axis}") for value in offsets),
            )
            validated_scales = tuple(_finite(value, f"scales_{axis}") for value in scales)
            if any(value <= 0.0 for value in validated_scales):
                raise ContractValidationError(f"scales_{axis} must be > 0")
            object.__setattr__(self, f"scales_{axis}", validated_scales)

    def predict(self, features: GazeFeatureVector) -> GazePoint:
        indices_x, indices_y = _profile_indices(self.profile, self.kind)
        design_x = _design_row(features, self.kind, indices_x, self.offsets_x, self.scales_x)
        design_y = _design_row(features, self.kind, indices_y, self.offsets_y, self.scales_y)
        return GazePoint(
            float(design_x @ np.asarray(self.coefficients_x, dtype=np.float64)),
            float(design_y @ np.asarray(self.coefficients_y, dtype=np.float64)),
        )

    @property
    def label(self) -> str:
        return f"{self.kind.value}/{self.profile.value}"

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "kind": self.kind.value,
            "coefficients_x": list(self.coefficients_x),
            "coefficients_y": list(self.coefficients_y),
            "ridge_lambda": self.ridge_lambda,
        }
        # Preserve the serialized identity of legacy FULL models. New
        # axis-routed models persist their complete preprocessing contract.
        if self.profile is not GazeModelProfile.FULL:
            payload.update(
                {
                    "profile": self.profile.value,
                    "offsets_x": list(self.offsets_x),
                    "scales_x": list(self.scales_x),
                    "offsets_y": list(self.offsets_y),
                    "scales_y": list(self.scales_y),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, value: object) -> RegressionModel:
        data = _mapping(value, "regression model")
        coefficients_x = data.get("coefficients_x")
        coefficients_y = data.get("coefficients_y")
        if not isinstance(coefficients_x, list) or not isinstance(coefficients_y, list):
            raise ContractValidationError("regression coefficients must be lists")
        kind = data.get("kind")
        if not isinstance(kind, str):
            raise ContractValidationError("regression kind must be a string")
        return cls(
            kind=GazeModelKind(kind),
            coefficients_x=tuple(coefficients_x),
            coefficients_y=tuple(coefficients_y),
            ridge_lambda=data.get("ridge_lambda"),  # type: ignore[arg-type]
            profile=GazeModelProfile(data.get("profile", GazeModelProfile.FULL.value)),
            offsets_x=tuple(data.get("offsets_x", ())),
            scales_x=tuple(data.get("scales_x", ())),
            offsets_y=tuple(data.get("offsets_y", ())),
            scales_y=tuple(data.get("scales_y", ())),
        )


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    median_error_normalized: float
    p95_error_normalized: float
    median_error_px: float
    p95_error_px: float
    median_abs_error_x_normalized: float
    p95_abs_error_x_normalized: float
    median_abs_error_y_normalized: float
    p95_abs_error_y_normalized: float
    per_target_error_normalized: tuple[tuple[int, float], ...]
    prediction_latency_ms: float

    def __post_init__(self) -> None:
        for name in (
            "median_error_normalized",
            "p95_error_normalized",
            "median_error_px",
            "p95_error_px",
            "median_abs_error_x_normalized",
            "p95_abs_error_x_normalized",
            "median_abs_error_y_normalized",
            "p95_abs_error_y_normalized",
            "prediction_latency_ms",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0))


@dataclass(frozen=True, slots=True)
class ModelComparison:
    center_baseline: ModelMetrics
    baseline: ModelMetrics
    advanced: ModelMetrics
    selected_kind: GazeModelKind


def fit_model(
    kind: GazeModelKind,
    features: Sequence[GazeFeatureVector],
    targets: Sequence[GazePoint],
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    profile: GazeModelProfile = GazeModelProfile.FULL,
) -> RegressionModel:
    """Fit one model using closed-form small-matrix linear algebra."""

    if not isinstance(kind, GazeModelKind):
        raise ContractValidationError("kind must be a GazeModelKind")
    if not isinstance(profile, GazeModelProfile):
        raise ContractValidationError("profile must be a GazeModelProfile")
    if len(features) != len(targets) or len(features) < 2:
        raise ContractValidationError("training requires equal feature/target lists of length >= 2")
    indices_x, indices_y = _profile_indices(profile, kind)
    normalize = profile is not GazeModelProfile.FULL
    offsets_x, scales_x = _preprocessing(features, indices_x, normalize=normalize)
    offsets_y, scales_y = _preprocessing(features, indices_y, normalize=normalize)
    design_x = np.vstack(
        [_design_row(vector, kind, indices_x, offsets_x, scales_x) for vector in features]
    )
    design_y = np.vstack(
        [_design_row(vector, kind, indices_y, offsets_y, scales_y) for vector in features]
    )
    x = np.asarray([target.x for target in targets], dtype=np.float64)
    y = np.asarray([target.y for target in targets], dtype=np.float64)
    regularization = _finite(ridge_lambda, "ridge_lambda", minimum=0.0)
    if kind is GazeModelKind.LINEAR:
        coefficients_x = np.linalg.lstsq(design_x, x, rcond=None)[0]
        coefficients_y = np.linalg.lstsq(design_y, y, rcond=None)[0]
        stored_lambda = 0.0
    else:
        coefficients_x = _ridge_fit(design_x, x, regularization)
        coefficients_y = _ridge_fit(design_y, y, regularization)
        stored_lambda = regularization
    return RegressionModel(
        kind=kind,
        coefficients_x=tuple(float(value) for value in coefficients_x),
        coefficients_y=tuple(float(value) for value in coefficients_y),
        ridge_lambda=stored_lambda,
        profile=profile,
        offsets_x=offsets_x,
        scales_x=scales_x,
        offsets_y=offsets_y,
        scales_y=scales_y,
    )


def compare_models(
    rows: Sequence[LabeledGazeRow],
    geometry: ScreenGeometry,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
) -> ModelComparison:
    """Run leave-one-target-out CV and select the conservative winner."""

    center_baseline = evaluate_center_baseline(rows, geometry)
    baseline = cross_validate(
        rows, geometry, GazeModelKind.LINEAR, profile=GazeModelProfile.AXIS_IRIS
    )
    advanced = cross_validate(
        rows,
        geometry,
        GazeModelKind.POLYNOMIAL_RIDGE,
        ridge_lambda=ridge_lambda,
        profile=GazeModelProfile.AXIS_IRIS_LIDS,
    )
    selected = (
        GazeModelKind.POLYNOMIAL_RIDGE
        if advanced_improves_conservatively(baseline, advanced)
        else GazeModelKind.LINEAR
    )
    return ModelComparison(
        center_baseline=center_baseline,
        baseline=baseline,
        advanced=advanced,
        selected_kind=selected,
    )


def advanced_improves_conservatively(baseline: ModelMetrics, advanced: ModelMetrics) -> bool:
    """Return whether polynomial wins without sacrificing either gaze axis.

    A combined Euclidean error can hide a regression in one output axis.  That
    is unsafe for screen mapping: a model which improves horizontal movement
    while making vertical gaze less accurate is not a conservative upgrade.
    The comparison is intentionally dimensionless so it behaves consistently
    across screen geometries.  It is model selection evidence only, not a
    product accuracy threshold; fresh live validation remains required before
    any candidate is activated.
    """

    return (
        advanced.median_error_normalized < baseline.median_error_normalized
        and advanced.p95_error_normalized <= baseline.p95_error_normalized
        and advanced.median_abs_error_x_normalized <= baseline.median_abs_error_x_normalized
        and advanced.p95_abs_error_x_normalized <= baseline.p95_abs_error_x_normalized
        and advanced.median_abs_error_y_normalized <= baseline.median_abs_error_y_normalized
        and advanced.p95_abs_error_y_normalized <= baseline.p95_abs_error_y_normalized
    )


def evaluate_center_baseline(
    rows: Sequence[LabeledGazeRow], geometry: ScreenGeometry
) -> ModelMetrics:
    """Measure the trivial predictor that always returns screen center.

    A learned model that cannot beat this baseline on held-out targets has
    not demonstrated that the eye features contain useful screen-position
    information and must not become the active calibration.
    """

    if not rows:
        raise ContractValidationError("center baseline requires at least one row")
    errors: list[float] = []
    pixel_errors: list[float] = []
    x_errors: list[float] = []
    y_errors: list[float] = []
    grouped: dict[int, list[float]] = {}
    for row in rows:
        error = math.hypot(0.5 - row.target.x, 0.5 - row.target.y)
        pixel_error = math.hypot(
            (0.5 - row.target.x) * max(1, geometry.width_px - 1),
            (0.5 - row.target.y) * max(1, geometry.height_px - 1),
        )
        errors.append(error)
        pixel_errors.append(pixel_error)
        x_errors.append(abs(0.5 - row.target.x))
        y_errors.append(abs(0.5 - row.target.y))
        grouped.setdefault(row.target_index, []).append(error)
    return ModelMetrics(
        median_error_normalized=float(np.median(errors)),
        p95_error_normalized=float(np.percentile(errors, 95)),
        median_error_px=float(np.median(pixel_errors)),
        p95_error_px=float(np.percentile(pixel_errors, 95)),
        median_abs_error_x_normalized=float(np.median(x_errors)),
        p95_abs_error_x_normalized=float(np.percentile(x_errors, 95)),
        median_abs_error_y_normalized=float(np.median(y_errors)),
        p95_abs_error_y_normalized=float(np.percentile(y_errors, 95)),
        per_target_error_normalized=tuple(
            (index, float(np.median(values))) for index, values in sorted(grouped.items())
        ),
        prediction_latency_ms=0.0,
    )


def cross_validate(
    rows: Sequence[LabeledGazeRow],
    geometry: ScreenGeometry,
    kind: GazeModelKind,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    profile: GazeModelProfile = GazeModelProfile.FULL,
) -> ModelMetrics:
    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be ScreenGeometry")
    target_indices = sorted({row.target_index for row in rows})
    if len(target_indices) < 2:
        raise ContractValidationError("cross-validation requires at least two target regions")
    errors: list[float] = []
    pixel_errors: list[float] = []
    x_errors: list[float] = []
    y_errors: list[float] = []
    per_target: list[tuple[int, float]] = []
    for held_out in target_indices:
        train = [row for row in rows if row.target_index != held_out]
        validation = [row for row in rows if row.target_index == held_out]
        if not train or not validation:
            raise ContractValidationError("every target fold requires training and validation rows")
        model = fit_model(
            kind,
            [row.features for row in train],
            [row.target for row in train],
            ridge_lambda=ridge_lambda,
            profile=profile,
        )
        fold_errors: list[float] = []
        for row in validation:
            predicted = model.predict(row.features)
            normalized_error = math.hypot(predicted.x - row.target.x, predicted.y - row.target.y)
            pixel_error = math.hypot(
                (predicted.x - row.target.x) * max(1, geometry.width_px - 1),
                (predicted.y - row.target.y) * max(1, geometry.height_px - 1),
            )
            errors.append(normalized_error)
            pixel_errors.append(pixel_error)
            x_errors.append(abs(predicted.x - row.target.x))
            y_errors.append(abs(predicted.y - row.target.y))
            fold_errors.append(normalized_error)
        per_target.append((held_out, float(np.median(fold_errors))))
    full_model = fit_model(
        kind,
        [row.features for row in rows],
        [row.target for row in rows],
        ridge_lambda=ridge_lambda,
        profile=profile,
    )
    latency = measure_prediction_latency_ms(full_model, rows[0].features)
    return ModelMetrics(
        median_error_normalized=float(np.median(errors)),
        p95_error_normalized=float(np.percentile(errors, 95)),
        median_error_px=float(np.median(pixel_errors)),
        p95_error_px=float(np.percentile(pixel_errors, 95)),
        median_abs_error_x_normalized=float(np.median(x_errors)),
        p95_abs_error_x_normalized=float(np.percentile(x_errors, 95)),
        median_abs_error_y_normalized=float(np.median(y_errors)),
        p95_abs_error_y_normalized=float(np.percentile(y_errors, 95)),
        per_target_error_normalized=tuple(per_target),
        prediction_latency_ms=latency,
    )


def measure_prediction_latency_ms(
    model: RegressionModel, features: GazeFeatureVector, *, repetitions: int = 64
) -> float:
    if repetitions < 1:
        raise ContractValidationError("repetitions must be >= 1")
    samples: list[float] = []
    for _ in range(repetitions):
        started = perf_counter_ns()
        model.predict(features)
        samples.append((perf_counter_ns() - started) / 1_000_000)
    return float(np.median(samples))


def _profile_indices(
    profile: GazeModelProfile, kind: GazeModelKind
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if profile is GazeModelProfile.FULL:
        indices = (
            _ALL_FEATURE_INDICES if kind is GazeModelKind.LINEAR else _POLYNOMIAL_SOURCE_INDICES
        )
        return indices, indices
    if profile is GazeModelProfile.AXIS_IRIS:
        return _X_AXIS_FEATURE_INDICES, _Y_AXIS_IRIS_FEATURE_INDICES
    return _X_AXIS_FEATURE_INDICES, _Y_AXIS_IRIS_LIDS_FEATURE_INDICES


def _preprocessing(
    features: Sequence[GazeFeatureVector], indices: tuple[int, ...], *, normalize: bool
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not normalize:
        return (0.0,) * len(indices), (1.0,) * len(indices)
    matrix = np.asarray(
        [[vector.values[index] for index in indices] for vector in features], dtype=np.float64
    )
    offsets = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    return tuple(float(value) for value in offsets), tuple(float(value) for value in scales)


def _ridge_fit(
    design: NDArray[np.float64], target: NDArray[np.float64], regularization: float
) -> NDArray[np.float64]:
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _design_width(kind: GazeModelKind, source_count: int) -> int:
    if kind is GazeModelKind.LINEAR:
        return 1 + source_count
    return 1 + source_count + (source_count * (source_count + 1) // 2)


def _design_row(
    features: GazeFeatureVector,
    kind: GazeModelKind,
    indices: tuple[int, ...],
    offsets: tuple[float, ...],
    scales: tuple[float, ...],
) -> NDArray[np.float64]:
    if not isinstance(features, GazeFeatureVector):
        raise ContractValidationError("features must be a GazeFeatureVector")
    source = [
        (features.values[index] - offset) / scale
        for index, offset, scale in zip(indices, offsets, scales, strict=True)
    ]
    if kind is GazeModelKind.LINEAR:
        return np.asarray((1.0, *source), dtype=np.float64)
    expanded: list[float] = [1.0, *source]
    for left in range(len(source)):
        for right in range(left, len(source)):
            expanded.append(source[left] * source[right])
    return np.asarray(expanded, dtype=np.float64)
