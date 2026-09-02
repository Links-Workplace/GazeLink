"""Property-focused tests for the small-data gaze regressors."""

from __future__ import annotations

import math

import numpy as np

from gazelink.domain import GazePoint, ScreenGeometry
from gazelink.gaze_features import GazeFeatureVector
from gazelink.gaze_model import (
    GazeModelKind,
    LabeledGazeRow,
    ModelMetrics,
    RegressionModel,
    advanced_improves_conservatively,
    compare_models,
    cross_validate,
    fit_model,
)


def _geometry() -> ScreenGeometry:
    return ScreenGeometry("primary", 1920, 1080, 1.0)


def _vector(index: int) -> GazeFeatureVector:
    return GazeFeatureVector(
        (
            (index % 5) / 4,
            ((index * 2) % 7) / 6,
            ((index * 3) % 11) / 10,
            ((index * 5) % 13) / 12,
            0.35 + ((index % 4) * 0.1),
            0.4 + (((index + 2) % 4) * 0.1),
            float((index % 7) - 3),
            float((index % 5) - 2),
            float((index % 3) - 1),
        )
    )


def _linear_target(features: GazeFeatureVector) -> GazePoint:
    values = features.values
    return GazePoint(
        0.1 + (0.35 * values[0]) + (0.2 * values[2]) + (0.01 * values[6]),
        0.2 + (0.25 * values[1]) + (0.15 * values[3]) - (0.01 * values[7]),
    )


def _cv_rows() -> tuple[LabeledGazeRow, ...]:
    rows: list[LabeledGazeRow] = []
    for target_index in range(9):
        x = (target_index % 3) / 2
        y = (target_index // 3) / 2
        for sample_index in range(3):
            jitter = (sample_index - 1) * 0.004
            rows.append(
                LabeledGazeRow(
                    features=GazeFeatureVector(
                        (
                            x + jitter,
                            y - jitter,
                            x - jitter,
                            y + jitter,
                            0.5,
                            0.55,
                            x * 4,
                            y * 3,
                            jitter,
                        )
                    ),
                    target_index=target_index,
                    target=GazePoint(x, y),
                )
            )
    return tuple(rows)


def test_linear_model_recovers_a_known_mapping_on_hand_pinned_rows() -> None:
    features = tuple(_vector(index) for index in range(24))
    targets = tuple(_linear_target(vector) for vector in features)
    model = fit_model(GazeModelKind.LINEAR, features, targets)
    for vector, expected in zip(features, targets, strict=True):
        predicted = model.predict(vector)
        assert predicted.x == pytest_approx(expected.x)
        assert predicted.y == pytest_approx(expected.y)


def pytest_approx(value: float) -> object:
    # Kept local so the model fixture stays dependency-obvious and deterministic.
    import pytest

    return pytest.approx(value, abs=1e-9)


def test_leave_one_target_out_metrics_are_finite_and_cover_every_region() -> None:
    rows = _cv_rows()
    for kind in (GazeModelKind.LINEAR, GazeModelKind.POLYNOMIAL_RIDGE):
        metrics = cross_validate(rows, _geometry(), kind)
        assert math.isfinite(metrics.median_error_normalized)
        assert math.isfinite(metrics.p95_error_px)
        assert metrics.prediction_latency_ms >= 0.0
        assert tuple(index for index, _error in metrics.per_target_error_normalized) == tuple(
            range(9)
        )


def test_model_comparison_returns_one_of_the_measured_candidates() -> None:
    comparison = compare_models(_cv_rows(), _geometry())
    assert comparison.selected_kind in {
        GazeModelKind.LINEAR,
        GazeModelKind.POLYNOMIAL_RIDGE,
    }


def _metrics(
    *,
    median: float,
    p95: float,
    x_median: float,
    x_p95: float,
    y_median: float,
    y_p95: float,
) -> ModelMetrics:
    return ModelMetrics(
        median_error_normalized=median,
        p95_error_normalized=p95,
        median_error_px=median * 1000,
        p95_error_px=p95 * 1000,
        median_abs_error_x_normalized=x_median,
        p95_abs_error_x_normalized=x_p95,
        median_abs_error_y_normalized=y_median,
        p95_abs_error_y_normalized=y_p95,
        per_target_error_normalized=(),
        prediction_latency_ms=0.0,
    )


def test_advanced_model_is_rejected_when_it_regresses_the_vertical_axis() -> None:
    baseline = _metrics(median=0.20, p95=0.40, x_median=0.10, x_p95=0.20, y_median=0.10, y_p95=0.20)
    advanced = _metrics(median=0.15, p95=0.35, x_median=0.05, x_p95=0.10, y_median=0.11, y_p95=0.19)

    assert not advanced_improves_conservatively(baseline, advanced)


def test_advanced_model_must_not_regress_either_axis_to_be_selected() -> None:
    baseline = _metrics(median=0.20, p95=0.40, x_median=0.10, x_p95=0.20, y_median=0.10, y_p95=0.20)
    advanced = _metrics(median=0.15, p95=0.35, x_median=0.05, x_p95=0.10, y_median=0.08, y_p95=0.15)

    assert advanced_improves_conservatively(baseline, advanced)


def test_stronger_ridge_shrinks_non_intercept_coefficients() -> None:
    features = tuple(_vector(index) for index in range(24))
    targets = tuple(_linear_target(vector) for vector in features)
    weak = fit_model(GazeModelKind.POLYNOMIAL_RIDGE, features, targets, ridge_lambda=1e-9)
    strong = fit_model(GazeModelKind.POLYNOMIAL_RIDGE, features, targets, ridge_lambda=10.0)
    weak_norm = np.linalg.norm((*weak.coefficients_x[1:], *weak.coefficients_y[1:]))
    strong_norm = np.linalg.norm((*strong.coefficients_x[1:], *strong.coefficients_y[1:]))
    assert strong_norm < weak_norm


def test_regression_model_round_trips_without_prediction_drift() -> None:
    features = tuple(_vector(index) for index in range(24))
    targets = tuple(_linear_target(vector) for vector in features)
    model = fit_model(GazeModelKind.LINEAR, features, targets)
    loaded = RegressionModel.from_dict(model.to_dict())
    assert loaded == model
    assert loaded.predict(features[3]) == model.predict(features[3])
