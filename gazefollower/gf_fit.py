"""Offline fitting and evaluation for the GazeFollower experiment (M2-05).

Everything the live library does after the model output -- the two OpenCV
SVRs, the cm->pixel conversion, the heuristic filter -- is reproduced here
on recorded rows, so every arm of the experiment is trained and scored on
exactly the same samples and only the columns / scaling / parameters differ.

Phase 0 (``run_phase0``):
  1. fit each configuration of the sweep on protocol A's accepted rows
     (ALL of them -- no head filter, so arm A stays a faithful baseline);
  2. score every configuration on TUNE's collecting rows and pick the best by
     the pre-declared metric (median Euclidean error, tie-break median |dy|);
  3. score the chosen model and the library-default model ONCE on T1;
  4. report P0-INFO (error + bias + slope, per axis), P0-CONFOUND
     (descriptive pitch/target correlation on A) and P0-VALID (recording
     integrity), and write a predictions JSON per scored model for
     ``analyze.py --predictions``.

Gate names are P0-* on purpose: G1-G11 are the product goals in the targets
document, and reusing those labels for Phase-0 gates made two different things
share one name.

T1 is never consulted during selection: ``select_config`` sees TUNE metrics
only, and a test permutes T1's labels to prove the choice cannot move.

No ``gazefollower`` import except, lazily, for the filter replay.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_geometry as GEO  # noqa: E402
import gf_select as SEL  # noqa: E402
import gf_report as RP  # noqa: E402
import gf_schema as S  # noqa: E402

# calibration/SVRCalibration.py::_set_svm_params -- the library's fixed choice.
LIBRARY_SVR = {"C": 1.0, "gamma": 0.005, "P": 0.001}
SVR_MAX_ITER = 10000
SVR_TERM_EPS = 1e-4

# Pre-declared selection metric for the Phase-0 sweep (plan §5).
SELECTION_METRIC = "median_euclid_px"
SELECTION_TIEBREAK = "y_median_abs_px"

# G1 category thresholds (plan §5). Half the project's 120 px test threshold.
P0_SLOPE_TOL = 0.3
P0_MEDIAN_ABS_DY_PX = 60.0
P0_ABS_BIAS_Y_PX = 60.0
P0_MIN_IMPROVEMENT = 0.25  # 25 % lower median |dy| than the library default, on TUNE

# P0-VALID thresholds (recording integrity, not product goals).
MIN_COLLECTING_ROWS_PER_TARGET = 20
T2_MIN_PITCH_IQR = 0.03


@dataclass(frozen=True)
class FitConfig:
    name: str
    feature_scaling: str = "none"  # "none" | "zscore"
    label_scaling: str = "none"  # "none" | "zscore"
    C: float = LIBRARY_SVR["C"]
    gamma: float | str = LIBRARY_SVR["gamma"]  # float or "auto"
    P: float = LIBRARY_SVR["P"]
    head_names: tuple[str, ...] = ()
    kind: str = "svr"  # "svr" | "ridge"
    alpha: float = 0.0  # ridge only

    def __post_init__(self) -> None:
        if self.feature_scaling not in ("none", "zscore"):
            raise ValueError(f"feature_scaling {self.feature_scaling!r}")
        if self.label_scaling not in ("none", "zscore"):
            raise ValueError(f"label_scaling {self.label_scaling!r}")
        if self.kind not in ("svr", "ridge"):
            raise ValueError(f"kind {self.kind!r}")
        if isinstance(self.gamma, str) and self.gamma != "auto":
            raise ValueError("gamma must be a float or 'auto'")
        for name in self.head_names:
            if name not in H.HEAD6_NAMES:
                raise ValueError(f"unknown head feature {name!r}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["head_names"] = list(self.head_names)
        return value


def library_default_config(head_names: tuple[str, ...] = ()) -> FitConfig:
    return FitConfig(name="lib-default" if not head_names else "lib-default+head", head_names=head_names)


def phase0_grid(head_names: tuple[str, ...] = ()) -> list[FitConfig]:
    """The sweep from plan §5. The library default is always the first entry."""

    grid: list[FitConfig] = [library_default_config(head_names)]
    for fs in ("none", "zscore"):
        for ls in ("none", "zscore"):
            for c in (1.0, 10.0, 100.0):
                for gamma in (0.005, 0.0005, "auto"):
                    name = f"svr_f{fs}_l{ls}_C{c:g}_g{gamma}"
                    if (fs, ls, c, gamma) == ("none", "none", 1.0, 0.005):
                        continue  # that is lib-default
                    grid.append(FitConfig(name=name, feature_scaling=fs, label_scaling=ls, C=c, gamma=gamma, head_names=head_names))
    grid.append(FitConfig(name="svr_fzscore_lzscore_C10_g0.005_P0.1", feature_scaling="zscore", label_scaling="zscore", C=10.0, gamma=0.005, P=0.1, head_names=head_names))
    for alpha in (0.01, 1.0, 100.0):
        grid.append(FitConfig(name=f"ridge_a{alpha:g}", feature_scaling="zscore", label_scaling="zscore", kind="ridge", alpha=alpha, head_names=head_names))
    return grid


def resolve_gamma(config: FitConfig, X_scaled: np.ndarray) -> float:
    """'auto' = 1/(d * mean column variance): 1/d after z-scoring."""

    if not isinstance(config.gamma, str):
        return float(config.gamma)
    d = X_scaled.shape[1]
    var = float(np.mean(np.var(X_scaled, axis=0)))
    if var <= 0:
        var = 1.0
    return 1.0 / (d * var)


# --- Fitted model ------------------------------------------------------------


def _new_svr(C_: float, gamma: float, P: float) -> Any:
    import cv2  # noqa: PLC0415

    svr = cv2.ml.SVM.create()
    svr.setType(cv2.ml.SVM_EPS_SVR)
    svr.setKernel(cv2.ml.SVM_RBF)
    svr.setC(C_)
    svr.setGamma(gamma)
    svr.setP(P)
    svr.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, SVR_MAX_ITER, SVR_TERM_EPS))
    return svr


@dataclass
class FittedModel:
    config: FitConfig
    schema: S.FeatureSchema
    _svr_x: Any = None
    _svr_y: Any = None
    _ridge: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        config: FitConfig,
        X_raw: np.ndarray,
        Y_cm: np.ndarray,
        *,
        rig: C.RigGeometry,
        train_meta: Mapping[str, Any],
    ) -> "FittedModel":
        X_raw = np.asarray(X_raw, dtype=np.float64)
        Y_cm = np.asarray(Y_cm, dtype=np.float64)
        if X_raw.ndim != 2 or Y_cm.shape != (X_raw.shape[0], 2):
            raise ValueError("X must be (n, d) and Y must be (n, 2)")
        if not (np.all(np.isfinite(X_raw)) and np.all(np.isfinite(Y_cm))):
            raise ValueError("training rows contain NaN/inf: exclude invalid rows before fitting")
        if X_raw.shape[0] < 4:
            raise ValueError("need at least 4 training rows")

        base_dim = X_raw.shape[1] - len(config.head_names)
        fx = S.Standardizer.fit(X_raw) if config.feature_scaling == "zscore" else S.Standardizer.identity(X_raw.shape[1])
        fy = S.Standardizer.fit(Y_cm) if config.label_scaling == "zscore" else S.Standardizer.identity(2)
        Xs = fx.transform(X_raw)
        Ys = fy.transform(Y_cm)
        gamma = resolve_gamma(config, Xs)
        schema = S.FeatureSchema(
            base_dim=base_dim,
            head_names=tuple(config.head_names),
            feature_scaling=config.feature_scaling,
            label_scaling=config.label_scaling,
            features=fx,
            labels=fy,
            svr={"C": float(config.C), "gamma": float(gamma), "P": float(config.P)},
            builder_version=H.BUILDER_VERSION if config.head_names else None,
            rig=rig.to_dict(),
            train=dict(train_meta, n_rows=int(X_raw.shape[0]), config=config.to_dict()),
            versions=S.environment_versions(),
        )
        model = cls(config=config, schema=schema)
        if config.kind == "svr":
            X32 = Xs.astype(np.float32)
            import cv2  # noqa: PLC0415

            model._svr_x = _new_svr(config.C, gamma, config.P)
            model._svr_y = _new_svr(config.C, gamma, config.P)
            model._svr_x.train(X32, cv2.ml.ROW_SAMPLE, Ys[:, 0].astype(np.float32).reshape(-1, 1))
            model._svr_y.train(X32, cv2.ml.ROW_SAMPLE, Ys[:, 1].astype(np.float32).reshape(-1, 1))
        else:
            model._ridge = _ridge_fit(Xs, Ys, config.alpha)
        return model

    def predict_cm(self, X_raw: np.ndarray) -> np.ndarray:
        """(n, 2) centimetres. Rows with NaN input give NaN output (loss)."""

        X_raw = np.asarray(X_raw, dtype=np.float64)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        self.schema.check_columns(X_raw.shape[1], H.BUILDER_VERSION if self.schema.head_names else None)
        out = np.full((X_raw.shape[0], 2), np.nan)
        valid = np.all(np.isfinite(X_raw), axis=1)
        if not np.any(valid):
            return out
        Xs = self.schema.features.transform(X_raw[valid])
        if self.config.kind == "svr":
            X32 = Xs.astype(np.float32)
            px = self._svr_x.predict(X32)[1].ravel()
            py = self._svr_y.predict(X32)[1].ravel()
            Zs = np.stack([px, py], axis=1).astype(np.float64)
        else:
            Zs = _ridge_predict(self._ridge, Xs)
        out[valid] = self.schema.labels.inverse(Zs)
        return out

    def support_activation(self, X_raw: np.ndarray) -> np.ndarray | None:
        """Per row: how strongly this model's calibration covers that input.

        An RBF SVR answers with its constant bias once a query sits far from
        every support vector, so a model used outside the conditions it was
        calibrated in does not degrade gracefully -- its output stops moving
        and freezes on a fixed wrong point.  That failure is invisible from
        the outside: the camera works, the face is found, and the dot simply
        stops following the eye.

        1.0 means the input sits on a support vector, 0.0 that the prediction
        carries no calibration information at all.  Measured: healthy sessions
        run 0.86-0.98, while every recording whose dot had frozen sat at
        0.000 across every sample.

        Returns None for a model with no kernel to measure (ridge), rather
        than inventing a number that would read as confidence.
        """

        if self.config.kind != "svr" or self._svr_x is None:
            return None
        X_raw = np.asarray(X_raw, dtype=np.float64)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        self.schema.check_columns(
            X_raw.shape[1], H.BUILDER_VERSION if self.schema.head_names else None
        )
        vectors = self._svr_x.getSupportVectors()
        if vectors is None or vectors.size == 0:
            return None
        gamma = float(self.schema.svr["gamma"])
        out = np.full(X_raw.shape[0], np.nan)
        valid = np.all(np.isfinite(X_raw), axis=1)
        if not np.any(valid):
            return out
        scaled = self.schema.features.transform(X_raw[valid]).astype(np.float32)
        activation = np.empty(scaled.shape[0], dtype=np.float64)
        # Chunked so a long recording never builds an (n x sv x dim) tensor.
        for start in range(0, scaled.shape[0], 256):
            block = scaled[start : start + 256]
            squared = ((block[:, None, :] - vectors[None, :, :]) ** 2).sum(-1)
            activation[start : start + block.shape[0]] = np.exp(-gamma * squared).max(axis=1)
        out[valid] = activation
        return out

    def predict_norm(self, X_raw: np.ndarray, rig: C.RigGeometry) -> np.ndarray:
        cm = self.predict_cm(X_raw)
        out = np.full_like(cm, np.nan)
        for i, (cx, cy) in enumerate(cm):
            if np.isfinite(cx) and np.isfinite(cy):
                out[i] = rig.cm_to_norm(cx, cy)
        return out

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.schema.save(directory / "schema.json")
        (directory / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2) + "\n", encoding="utf-8")
        if self.config.kind == "svr":
            self._svr_x.save(str(directory / "svr_x.xml"))
            self._svr_y.save(str(directory / "svr_y.xml"))
        else:
            np.savez(directory / "ridge.npz", **self._ridge)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "FittedModel":
        directory = Path(directory)
        schema = S.FeatureSchema.load(directory / "schema.json")
        config = FitConfig(**_config_from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8"))))
        model = cls(config=config, schema=schema)
        if config.kind == "svr":
            import cv2  # noqa: PLC0415

            model._svr_x = cv2.ml.SVM.load(str(directory / "svr_x.xml"))
            model._svr_y = cv2.ml.SVM.load(str(directory / "svr_y.xml"))
            for svr in (model._svr_x, model._svr_y):
                if svr.getVarCount() != schema.n_columns:
                    raise ValueError(
                        f"{directory}: saved SVR has {svr.getVarCount()} inputs, schema says {schema.n_columns}"
                    )
        else:
            with np.load(directory / "ridge.npz") as data:
                model._ridge = {key: data[key] for key in data.files}
            if model._ridge["w"].shape[0] != schema.n_columns:
                raise ValueError(f"{directory}: saved ridge has {model._ridge['w'].shape[0]} inputs, schema says {schema.n_columns}")
        return model


def _config_from_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value["head_names"] = tuple(value.get("head_names", ()))
    return value


def _ridge_fit(Xs: np.ndarray, Ys: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Centred ridge: w = (XᵀX + αI)⁻¹ Xᵀ y on centred data, intercept from means."""

    x_mean = Xs.mean(axis=0)
    y_mean = Ys.mean(axis=0)
    Xc = Xs - x_mean
    Yc = Ys - y_mean
    d = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ Yc)  # (d, 2)
    return {"w": w, "x_mean": x_mean, "y_mean": y_mean}


def _ridge_predict(ridge: Mapping[str, np.ndarray], Xs: np.ndarray) -> np.ndarray:
    return (Xs - ridge["x_mean"]) @ ridge["w"] + ridge["y_mean"]


# --- Row selection -----------------------------------------------------------


def training_rows(rec: S.Recording, head_names: Sequence[str]) -> tuple[np.ndarray, int]:
    """Mask of rows to train on, and how many accepted rows the head filter removed.

    Calibration protocols: the controller's accepted rows. Head columns can
    only be used where they are valid, so a head-using arm additionally
    requires ``head_valid`` -- and REPORTS the count it dropped. An arm with
    no head columns is never filtered by head validity (plan B2: arm A stays
    the library's own population).
    """

    mask = rec.rows_accepted().copy()
    dropped = 0
    if head_names:
        before = int(mask.sum())
        mask &= rec.rows_head_valid()
        dropped = before - int(mask.sum())
    return mask, dropped


def scoring_rows(rec: S.Recording, head_names: Sequence[str]) -> tuple[np.ndarray, int]:
    """Rows a model can actually be asked to predict on.

    A strict subset of :meth:`Recording.rows_eligible`. Everything eligible
    but absent from this mask is a LOSS and is reported as one -- see
    :func:`eligible_rows`, which supplies the denominator.
    """

    mask = rec.rows_collecting().copy()
    dropped = 0
    if head_names:
        before = int(mask.sum())
        mask &= rec.rows_head_valid()
        dropped = before - int(mask.sum())
    return mask, dropped


def eligible_rows(rec: S.Recording) -> np.ndarray:
    """The denominator: every row that should have produced a measurement.

    Independent of tracking success and of which arm is being scored, so all
    arms are judged against the same opportunity count.
    """

    return rec.rows_eligible()


def design_matrix(rec: S.Recording, mask: np.ndarray, head_names: Sequence[str]) -> np.ndarray:
    return S.assemble(rec.features[mask], rec.head[mask] if head_names else None, head_names, H.HEAD6_NAMES)


# --- Metrics -----------------------------------------------------------------


@dataclass
class AxisMetrics:
    """Per-axis error summary.

    Bias travels with spread and slope everywhere: a slope near 1 with a large
    constant bias is not accuracy, which is exactly how the earlier X result
    (slope 1.096, bias -349 px) read as "almost perfect".
    """

    mean_abs_px: float
    median_abs_px: float
    p90_abs_px: float
    p95_abs_px: float
    max_abs_px: float
    bias_px: float  # median signed error
    slope: float | None
    intercept: float | None

    def flat(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_mean_abs_px": self.mean_abs_px,
            f"{prefix}_median_abs_px": self.median_abs_px,
            f"{prefix}_p90_abs_px": self.p90_abs_px,
            f"{prefix}_p95_abs_px": self.p95_abs_px,
            f"{prefix}_max_abs_px": self.max_abs_px,
            f"{prefix}_bias_px": self.bias_px,
            f"{prefix}_slope": self.slope,
            f"{prefix}_intercept": self.intercept,
        }


@dataclass
class EvalMetrics:
    """Accuracy over the rows that produced a prediction, availability over
    every row that SHOULD have.

    Both live in one object on purpose. Scoring only the rows that carried a
    gaze sample reports accuracy among the survivors of a loss while the loss
    itself reads as zero; the eligible denominator is what makes coverage mean
    availability.
    """

    n_eligible: int  # rows that should have produced a measurement
    n_valid: int  # of those, rows with a usable prediction
    n_lost: int  # n_eligible - n_valid
    coverage: float  # n_valid / n_eligible
    n_fresh: int  # valid rows whose sample was fresh
    coverage_fresh: float
    longest_gap_ms: float  # longest unplanned run of lost rows
    mean_euclid_px: float
    median_euclid_px: float
    p90_euclid_px: float
    p95_euclid_px: float
    max_euclid_px: float
    x: AxisMetrics
    y: AxisMetrics
    off_screen_x: int
    off_screen_y: int
    per_target: list[dict[str, Any]]

    def flat(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_eligible": self.n_eligible,
            "n_valid": self.n_valid,
            "n_lost": self.n_lost,
            "coverage": self.coverage,
            "n_fresh": self.n_fresh,
            "coverage_fresh": self.coverage_fresh,
            "longest_gap_ms": self.longest_gap_ms,
            "mean_euclid_px": self.mean_euclid_px,
            "median_euclid_px": self.median_euclid_px,
            "p90_euclid_px": self.p90_euclid_px,
            "p95_euclid_px": self.p95_euclid_px,
            "max_euclid_px": self.max_euclid_px,
            "off_screen_x": self.off_screen_x,
            "off_screen_y": self.off_screen_y,
        }
        out.update(self.x.flat("x"))
        out.update(self.y.flat("y"))
        return out


def _linfit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2 or len(set(xs)) < 2:
        return None, None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def _axis_metrics(d: np.ndarray, targets: Sequence[float], preds: Sequence[float]) -> AxisMetrics:
    a = np.abs(d)
    slope, intercept = _linfit(targets, preds)
    return AxisMetrics(
        mean_abs_px=float(np.mean(a)),
        median_abs_px=float(np.median(a)),
        p90_abs_px=float(np.percentile(a, 90)),
        p95_abs_px=float(np.percentile(a, 95)),
        max_abs_px=float(np.max(a)),
        bias_px=float(np.median(d)),
        slope=slope,
        intercept=intercept,
    )


def _empty_axis() -> AxisMetrics:
    nan = float("nan")
    return AxisMetrics(nan, nan, nan, nan, nan, nan, None, None)


def _longest_gap_ms(valid: np.ndarray, timestamp_ns: np.ndarray | None) -> float:
    """Longest consecutive run of invalid rows, in milliseconds."""

    if timestamp_ns is None or len(valid) == 0:
        return 0.0
    ts_ms = np.asarray(timestamp_ns, dtype=np.float64) / 1e6
    longest = 0.0
    start: float | None = None
    for i, ok in enumerate(valid):
        if not ok and start is None:
            start = ts_ms[i]
        elif ok and start is not None:
            longest = max(longest, ts_ms[i] - start)
            start = None
    if start is not None:
        longest = max(longest, ts_ms[-1] - start)
    return float(longest)


def evaluate(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    target_id: np.ndarray,
    width_px: int,
    height_px: int,
    target_names: Mapping[int, str] | None = None,
    *,
    fresh: np.ndarray | None = None,
    timestamp_ns: np.ndarray | None = None,
) -> EvalMetrics:
    """Errors in logical px (analyze.py convention: normalised * (size - 1)).

    Every row passed in is ELIGIBLE: it should have produced a measurement.
    Rows whose prediction is NaN are counted as lost, never dropped.
    """

    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    target_norm = np.asarray(target_norm, dtype=np.float64)
    target_id = np.asarray(target_id)
    n = pred_norm.shape[0]
    valid = np.all(np.isfinite(pred_norm), axis=1)
    dx = (pred_norm[:, 0] - target_norm[:, 0]) * (width_px - 1)
    dy = (pred_norm[:, 1] - target_norm[:, 1]) * (height_px - 1)
    euclid = np.hypot(dx, dy)

    per_target: list[dict[str, Any]] = []
    tx_list: list[float] = []
    ty_list: list[float] = []
    px_list: list[float] = []
    py_list: list[float] = []
    for tid in sorted(set(int(v) for v in target_id)):
        rows = valid & (target_id == tid)
        if not np.any(rows):
            continue
        tx, ty = float(target_norm[rows][0, 0]), float(target_norm[rows][0, 1])
        px_med, py_med = float(np.median(pred_norm[rows, 0])), float(np.median(pred_norm[rows, 1]))
        per_target.append(
            {
                "target_id": tid,
                "name": (target_names or {}).get(tid, f"T{tid}"),
                "tx": tx,
                "ty": ty,
                "pred_x_median": px_med,
                "pred_y_median": py_med,
                "n": int(rows.sum()),
                "median_euclid_px": float(np.median(euclid[rows])),
                "median_dx_px": float(np.median(dx[rows])),
                "median_dy_px": float(np.median(dy[rows])),
            }
        )
        tx_list.append(tx)
        ty_list.append(ty)
        px_list.append(px_med)
        py_list.append(py_med)

    n_valid = int(valid.sum())
    if n_valid:
        e = euclid[valid]
        ax = _axis_metrics(dx[valid], tx_list, px_list)
        ay = _axis_metrics(dy[valid], ty_list, py_list)
        stats = {
            "mean": float(np.mean(e)),
            "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)),
            "max": float(np.max(e)),
        }
        off_x = int(np.sum((pred_norm[valid, 0] < 0) | (pred_norm[valid, 0] > 1)))
        off_y = int(np.sum((pred_norm[valid, 1] < 0) | (pred_norm[valid, 1] > 1)))
    else:
        nan = float("nan")
        ax = ay = _empty_axis()
        stats = {"mean": nan, "median": nan, "p90": nan, "p95": nan, "max": nan}
        off_x = off_y = 0
    fresh_mask = valid if fresh is None else (valid & np.asarray(fresh, dtype=bool))
    n_fresh = int(fresh_mask.sum())
    return EvalMetrics(
        n_eligible=int(n),
        n_valid=n_valid,
        n_lost=int(n - n_valid),
        coverage=(n_valid / n) if n else float("nan"),
        n_fresh=n_fresh,
        coverage_fresh=(n_fresh / n) if n else float("nan"),
        longest_gap_ms=_longest_gap_ms(valid, timestamp_ns),
        mean_euclid_px=stats["mean"],
        median_euclid_px=stats["median"],
        p90_euclid_px=stats["p90"],
        p95_euclid_px=stats["p95"],
        max_euclid_px=stats["max"],
        x=ax,
        y=ay,
        off_screen_x=off_x,
        off_screen_y=off_y,
        per_target=per_target,
    )


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# --- Filter replay -----------------------------------------------------------


def replay_heuristic_filter(pred_px: np.ndarray, valid: np.ndarray, filter_factory: Any = None) -> np.ndarray:
    """Run the library's HeuristicFilter over a prediction sequence, in order.

    ``process_frame`` only calls the filter on frames with a gaze sample, so
    rows with ``valid=False`` are skipped and get NaN, exactly as live. The
    factory defaults to the library's class (lazy import: it initialises the
    library's native components); tests inject a stand-in.
    """

    if filter_factory is None:

        def filter_factory() -> Any:  # type: ignore[no-redef]
            from gazefollower.filter import HeuristicFilter  # noqa: PLC0415

            return HeuristicFilter()

    flt = filter_factory()
    out = np.full_like(np.asarray(pred_px, dtype=np.float64), np.nan)
    for i, ok in enumerate(valid):
        if ok and np.all(np.isfinite(pred_px[i])):
            fx, fy = flt.filter_values([float(pred_px[i, 0]), float(pred_px[i, 1])])
            out[i] = (fx, fy)
    return out


# --- Phase 0 -----------------------------------------------------------------


def _finite_or_inf(value: Any) -> float:
    """A metric that is NaN/None/inf must sort last, never first.

    NaN compares false against everything, so leaving it in a sort key makes
    the winner depend on input order rather than on the metric.
    """

    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return f if math.isfinite(f) else float("inf")


def _sweep_sort_key(tune: Mapping[str, Any]) -> tuple[float, float]:
    return (_finite_or_inf(tune.get(SELECTION_METRIC)), _finite_or_inf(tune.get(SELECTION_TIEBREAK)))


def screen_sweep(
    models: Mapping[str, "FittedModel"],
    rec: S.Recording,
    rig: C.RigGeometry,
    head_names: Sequence[str],
    *,
    thresholds: SEL.ScreeningThresholds,
) -> list[SEL.CandidateReport]:
    """Screen every candidate on the VALIDATION recording.

    Screening runs on TUNE, never on the test set: a threshold tuned or a
    model chosen against the final numbers would make those numbers a
    training score.
    """

    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    eligible = rec.rows_eligible()
    reports: list[SEL.CandidateReport] = []
    for name, model in models.items():
        try:
            pred_all = np.full((rec.n_rows, 2), np.nan)
            mask, _ = scoring_rows(rec, head_names)
            if np.any(mask):
                pred_all[mask] = model.predict_norm(design_matrix(rec, mask, head_names), rig)
        except Exception as exc:  # noqa: BLE001 - a broken candidate is a rejection, not a crash
            reports.append(SEL.screen_candidate(name, pred_norm=np.zeros((0, 2)), target_norm=np.zeros((0, 2)),
                                                fold_id=np.zeros(0, dtype=int), width_px=width, height_px=height,
                                                fit_failed=True, fit_error=repr(exc)))
            continue
        reports.append(
            SEL.screen_candidate(
                name,
                pred_norm=pred_all[eligible],
                target_norm=rec.target_xy[eligible],
                fold_id=rec.target_id[eligible],
                width_px=width,
                height_px=height,
                thresholds=thresholds,
            )
        )
    return reports


def select_config(sweep: Sequence[Mapping[str, Any]]) -> str:
    """Best configuration by the pre-declared TUNE metric. Pure; TUNE only.

    Never reads T1: the caller passes TUNE metrics, and a test permutes T1's
    labels to prove the choice cannot move.
    """

    if not sweep:
        raise ValueError("empty sweep")
    ranked = sorted(sweep, key=lambda entry: _sweep_sort_key(entry["tune"]))
    best = ranked[0]
    if not math.isfinite(_finite_or_inf(best["tune"].get(SELECTION_METRIC))):
        raise ValueError("no configuration produced a finite selection metric on TUNE")
    return str(best["name"])


def p0_info_category(t1: Mapping[str, Any], lib_tune: Mapping[str, Any], best_tune: Mapping[str, Any]) -> dict[str, Any]:
    """Did calibration-layer scaling recover the vertical axis?

    Error, bias and slope are judged together, and a failed sweep is reported
    as "no configuration found WITH THESE FEATURES AND THIS FITTER" -- never
    as "the embedding carries no vertical information", which the measurement
    cannot support.
    """

    slope = t1.get("y_slope")
    med_dy = t1.get("y_median_abs_px")
    bias = t1.get("y_bias_px")
    fixed = (
        slope is not None
        and abs(slope - 1.0) <= P0_SLOPE_TOL
        and med_dy is not None
        and med_dy <= P0_MEDIAN_ABS_DY_PX
        and bias is not None
        and abs(bias) <= P0_ABS_BIAS_Y_PX
    )
    lib_dy = lib_tune.get("y_median_abs_px")
    best_dy = best_tune.get("y_median_abs_px")
    improved = lib_dy is not None and best_dy is not None and lib_dy > 0 and (lib_dy - best_dy) / lib_dy >= P0_MIN_IMPROVEMENT
    if fixed:
        category = "fixed_by_calibration"
        statement = "Y meets slope, median |dy| and bias bounds on T1 with the selected configuration."
    elif improved:
        category = "partial_improvement"
        statement = "Y improved on TUNE by >= 25 % vs the library default but does not meet all G1 bounds on T1."
    else:
        category = "no_improvement"
        statement = (
            "No configuration in the sweep extracts usable vertical information WITH THESE FEATURES AND THIS FITTER. "
            "This is not evidence that the embedding carries no vertical information."
        )
    return {
        "category": category,
        "statement": statement,
        "t1_y_slope": slope,
        "t1_y_median_abs_px": med_dy,
        "t1_y_bias_px": bias,
        "t1_x_slope": t1.get("x_slope"),
        "t1_x_median_abs_px": t1.get("x_median_abs_px"),
        "t1_x_bias_px": t1.get("x_bias_px"),
        "tune_lib_y_median_abs_px": lib_dy,
        "tune_best_y_median_abs_px": best_dy,
        "thresholds": {
            "slope_tol": P0_SLOPE_TOL,
            "median_abs_dy_px": P0_MEDIAN_ABS_DY_PX,
            "abs_bias_y_px": P0_ABS_BIAS_Y_PX,
            "min_improvement": P0_MIN_IMPROVEMENT,
        },
    }


def p0_confound(rec_a: S.Recording) -> dict[str, Any]:
    mask = rec_a.rows_accepted() & rec_a.rows_head_valid()
    pitch = rec_a.head[mask, H.HEAD6_NAMES.index("pitch_a")]
    yaw = rec_a.head[mask, H.HEAD6_NAMES.index("yaw_ratio")]
    ty = rec_a.target_xy[mask, 1]
    tx = rec_a.target_xy[mask, 0]
    return {
        "n_rows": int(mask.sum()),
        "corr_pitch_a_target_y": pearson(pitch, ty),
        "corr_yaw_ratio_target_x": pearson(yaw, tx),
        "pitch_a_iqr": _iqr(pitch),
        "note": "descriptive only; a correlation here is not a cause of the compression",
    }


def _iqr(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return None
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def p0_valid(rec: S.Recording, rec_t1: S.Recording | None = None) -> dict[str, Any]:
    """Recording validity (plan gate P0-VALID). Reports, never silently fixes.

    Named P0-VALID, not G3: G1-G11 belong to the product goals document, and
    reusing those names for Phase-0 gates made two different things share one
    label.

    A check that cannot be evaluated is ``None`` (unknown), never ``True``.
    An unknown check makes the recording NOT valid: a run whose integrity we
    cannot establish must not support a conclusion. Expected targets come from
    the recording's own metadata, so a run that stopped after one target fails
    instead of certifying the single target it managed to collect.
    """

    integrity = dict(rec.meta.get("integrity", {}))
    expected = [int(t["index"]) for t in rec.targets]
    out: dict[str, Any] = {
        "protocol": rec.protocol,
        "n_rows": rec.n_rows,
        "expected_targets": len(expected),
        "subscriber_errors": integrity.get("subscriber_errors"),
        "watchdog_tripped": integrity.get("watchdog_tripped"),
        "aborted": integrity.get("aborted"),
        "finished": integrity.get("finished"),
        "fps_median": integrity.get("fps_median"),
    }
    gaze_rows = np.asarray(rec.gaze_status, dtype=bool)
    out["head_invalid_rate_among_gaze_rows"] = (
        float(np.mean(~rec.rows_head_valid()[gaze_rows])) if np.any(gaze_rows) else None
    )
    eligible = rec.rows_eligible()
    out["coverage"] = float(np.mean(gaze_rows[eligible])) if np.any(eligible) else None

    checks: dict[str, bool | None] = {}
    if rec.protocol in S.CALIBRATION_PROTOCOLS:
        counts = {tid: int(np.sum(rec.rows_accepted() & (rec.target_id == tid))) for tid in expected}
        out["accepted_rows_per_target"] = counts
        checks["every_expected_target_has_45"] = bool(expected) and all(
            counts.get(tid, 0) == C.N_FRAMES_PER_POINT for tid in expected
        )
    else:
        counts = {tid: int(np.sum(rec.rows_collecting() & (rec.target_id == tid))) for tid in expected}
        out["collecting_rows_per_target"] = counts
        checks["every_expected_target_has_20"] = bool(expected) and all(
            counts.get(tid, 0) >= MIN_COLLECTING_ROWS_PER_TARGET for tid in expected
        )
    out["missing_targets"] = [tid for tid in expected if counts.get(tid, 0) == 0]

    rate = out["head_invalid_rate_among_gaze_rows"]
    checks["head_invalid_below_2pct"] = None if rate is None else rate < 0.02
    fps = out["fps_median"]
    checks["fps_median_at_least_25"] = None if fps is None else fps >= 25
    errs = out["subscriber_errors"]
    checks["no_subscriber_errors"] = None if errs is None else errs == 0
    wd = out["watchdog_tripped"]
    checks["watchdog_not_tripped"] = None if wd is None else not wd
    aborted = out["aborted"]
    checks["not_aborted"] = None if aborted is None else not aborted
    finished = out["finished"]
    checks["ran_to_completion"] = None if finished is None else bool(finished)

    if rec.protocol == "T2" and rec_t1 is not None:
        idx = H.HEAD6_NAMES.index("pitch_a")
        iqr_t2 = _iqr(rec.head[rec.rows_head_valid(), idx])
        iqr_t1 = _iqr(rec_t1.head[rec_t1.rows_head_valid(), idx])
        out["pitch_a_iqr_t2"] = iqr_t2
        out["pitch_a_iqr_t1"] = iqr_t1
        checks["t2_pitch_iqr_3x_t1_and_min"] = (
            None
            if iqr_t2 is None or iqr_t1 is None
            else (iqr_t2 >= 3 * iqr_t1 and iqr_t2 >= T2_MIN_PITCH_IQR)
        )

    out["checks"] = checks
    out["unknown_checks"] = [name for name, value in checks.items() if value is None]
    out["failed_checks"] = [name for name, value in checks.items() if value is False]
    # Unknown counts against validity: absence of evidence is not validity.
    out["valid"] = bool(checks) and all(value is True for value in checks.values())
    return out


def predictions_rows(
    rec: S.Recording,
    pred_norm_all: np.ndarray,
    filtered_norm_all: np.ndarray | None,
    rig: C.RigGeometry,
    *,
    arm: str,
    config_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """analyze.py rows for every row of a timed protocol, plus arrival timings.

    Scored value = the PRE-filter prediction (like the live capture's
    ``_pick_point``); the filtered one rides along as ``filtered_x/y``.
    """

    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    fps = rec.meta.get("integrity", {}).get("fps_median")
    rows: list[dict[str, Any]] = []
    by_target: dict[int, list[tuple[float, float, float]]] = {}
    for i in range(rec.n_rows):
        tid = int(rec.target_id[i])
        if tid < 0 or not np.all(np.isfinite(pred_norm_all[i])):
            continue
        phase = str(rec.phase[i])
        if phase not in (S.PHASE_STABILIZING, S.PHASE_COLLECTING):
            continue
        pnp = rec.pnp_deg[i]
        has_pose = bool(np.all(np.isfinite(pnp)))
        raw = rec.raw_cm[i]
        raw_norm = rig.cm_to_norm(float(raw[0]), float(raw[1])) if np.all(np.isfinite(raw)) else (None, None)
        filt = filtered_norm_all[i] if filtered_norm_all is not None else None
        rows.append(
            C.make_sample(
                target_index=tid,
                predicted_x=float(pred_norm_all[i, 0]),
                predicted_y=float(pred_norm_all[i, 1]),
                timestamp_ms=float(rec.elapsed_ms[i]),
                accepted=True,
                collection_phase=phase,
                fps=None if fps is None else float(fps),
                head_yaw_deg=float(pnp[0]) if has_pose else None,
                head_pitch_deg=float(pnp[1]) if has_pose else None,
                head_roll_deg=float(pnp[2]) if has_pose else None,
                raw_x=raw_norm[0],
                raw_y=raw_norm[1],
                filtered_x=None if filt is None or not np.all(np.isfinite(filt)) else float(filt[0]),
                filtered_y=None if filt is None or not np.all(np.isfinite(filt)) else float(filt[1]),
                head_valid=bool(rec.head_valid[i]),
                tracking_state=str(rec.tracking_state[i]),
                arm=arm,
                config=config_name,
            )
        )
        by_target.setdefault(tid, []).append((float(rec.elapsed_ms[i]), float(pred_norm_all[i, 0]), float(pred_norm_all[i, 1])))
    timings = []
    targets = {int(t["index"]): t for t in rec.targets}
    for tid, seq in sorted(by_target.items()):
        pos = targets[tid]["screen_position"]
        seq.sort(key=lambda r: r[0])
        timings.append({"target_index": tid, "time_to_target_ms": C.arrival_times_ms(seq, float(pos["x"]), float(pos["y"]), width, height)})
    return rows, timings


def _score_timed(model: FittedModel, rec: S.Recording, rig: C.RigGeometry, head_names: Sequence[str]) -> tuple[EvalMetrics, np.ndarray, int]:
    """Score one timed protocol.

    Predictions are produced for the rows that CAN be predicted; the metrics
    are computed over every ELIGIBLE row, so a frame the tracker lost enters
    as NaN and is counted in ``n_lost`` and ``coverage`` instead of quietly
    leaving the denominator.
    """

    mask, dropped = scoring_rows(rec, head_names)
    pred_all = np.full((rec.n_rows, 2), np.nan)
    if np.any(mask):
        X = design_matrix(rec, mask, head_names)
        pred_all[mask] = model.predict_norm(X, rig)
    # Predictions for stabilizing rows too (arrival timings), which the
    # accuracy metrics never see.
    extra = (rec.phase == S.PHASE_STABILIZING) & np.asarray(rec.gaze_status, dtype=bool)
    if head_names:
        extra &= rec.rows_head_valid()
    if np.any(extra):
        pred_all[extra] = model.predict_norm(design_matrix(rec, extra, head_names), rig)
    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    names = {int(t["index"]): str(t["name"]) for t in rec.targets}
    eligible = eligible_rows(rec)
    metrics = evaluate(
        pred_all[eligible],
        rec.target_xy[eligible],
        rec.target_id[eligible],
        width,
        height,
        names,
        fresh=rec.rows_fresh()[eligible],
        timestamp_ns=rec.timestamp_ns[eligible],
    )
    return metrics, pred_all, dropped


def protocol_report(
    rec: S.Recording,
    pred_norm_all: np.ndarray,
    rig: C.RigGeometry,
    *,
    eye_distance_cm: float | None,
    camera_pitch_deg: float | None = None,
    eye_source: str = "nominal",
) -> dict[str, Any]:
    """Zones, target x condition matrix, proximity split and estimated degrees.

    Built on SEGMENTS so a target the tracker followed for many frames cannot
    outvote one it barely held, and on ELIGIBLE rows so a lost frame lands in
    the coverage rather than vanishing from it.
    """

    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    eligible = rec.rows_eligible()
    valid = eligible & np.all(np.isfinite(pred_norm_all), axis=1)
    dx = (pred_norm_all[:, 0] - rec.target_xy[:, 0]) * (width - 1)
    dy = (pred_norm_all[:, 1] - rec.target_xy[:, 1]) * (height - 1)

    degrees = None
    degrees_summary = None
    if eye_distance_cm is not None:
        frame = GEO.ScreenFrame(rig, camera_pitch_deg=camera_pitch_deg)
        degrees = GEO.angular_errors_for_rows(
            frame, rec.target_xy, pred_norm_all, eye_distance_cm=eye_distance_cm
        )
        degrees_summary = GEO.summarise_degrees(
            degrees[valid], source=eye_source, tilt_corrected=frame.tilt_corrected, scale_validated=False
        )

    segments = RP.segments_from_rows(
        target_id=rec.target_id,
        target_xy=rec.target_xy,
        eligible=eligible,
        valid=valid,
        dx_px=dx,
        dy_px=dy,
        degrees=degrees,
        targets_meta=rec.targets,
        condition=str(rec.meta.get("condition", "static_neutral")),
    )
    return RP.build_report(
        protocol=rec.protocol,
        segments=segments,
        targets_meta=rec.targets,
        conditions=(str(rec.meta.get("condition", "static_neutral")),),
        degrees_summary=degrees_summary,
    )


def goal_status(eye_distance_cm: float | None, camera_pitch_deg: float | None) -> dict[str, Any]:
    """Where each product goal stands after a Phase-0 recording.

    Phase 0 measures accuracy on a still head and nothing else, so most goals
    are NOT MEASURED -- the correct result to report, not a gap to paper over.
    Naming all of them keeps a partial run from reading as progress against
    the whole list.
    """

    angles_possible = eye_distance_cm is not None
    if angles_possible:
        tilt = "is modelled." if camera_pitch_deg is not None else "is NOT modelled."
        g1_why = (
            "Angles come from an ESTIMATED eye position; the scale is unvalidated "
            f"and the camera tilt {tilt}"
        )
    else:
        g1_why = "No eye distance was declared, so no angle can be computed."
    return {
        "G1_accuracy_deg": {"status": "NOT VERIFIED" if angles_possible else "NOT MEASURED", "why": g1_why},
        "G2_p90_deg": {
            "status": "NOT VERIFIED" if angles_possible else "NOT MEASURED",
            "why": "Rests on the same estimated geometry as G1.",
        },
        "G3_head_range": {
            "status": "NOT MEASURED",
            "why": "Phase 0 records a still head only; the static-pose and motion protocols are not part of it.",
        },
        "G4_availability": {
            "status": "MEASURED (this condition only)",
            "why": "Coverage is reported over eligible rows, for the still-head condition.",
        },
        "G5_rate": {
            "status": "PARTIAL",
            "why": "Median frame rate is reported; the 5-second-window criterion is not implemented.",
        },
        "G6_latency": {
            "status": "NOT MEASURED",
            "why": "The library timestamps a frame on RECEIPT, not exposure, so only software latency could be reported.",
        },
        "G7_calibration_time": {
            "status": "PARTIAL",
            "why": "Protocol A's duration is recorded; end-to-end calibration time is not.",
        },
        "G8_stability_20min": {"status": "NOT MEASURED", "why": "No 20-minute protocol in Phase 0."},
        "G9_recovery": {"status": "NOT MEASURED", "why": "No recovery protocol in Phase 0."},
        "G10_cpu": {"status": "NOT MEASURED", "why": "No CPU sampling in Phase 0."},
        "G11_repeatability": {
            "status": "NOT MEASURED",
            "why": "One session; three across at least two days are required.",
        },
        "decision": "NOT MEASURED",
        "note": "Phase 0 asks only whether calibration-layer scaling recovers the vertical axis.",
    }


def existing_model_dirs(recording_dir: Path) -> list[str]:
    """Names of the fitted models already saved under this recording."""

    root = Path(recording_dir) / "models"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def refuse_to_overwrite_models(recording_dir: Path) -> None:
    """Stop before fitting if this recording already has saved models.

    ``phase0`` writes every swept configuration into
    ``<recording>/models/<name>``, replacing whatever was there under the same
    name.  A model directory is not scratch space: it is what a profile points
    at and what a live session loads, so replacing it in place destroys the
    thing someone is relying on, and the replacement is not obviously
    different -- same path, same filenames, same config name, different
    weights.

    This is not hypothetical.  Refitting round18 with head features to run an
    experiment silently replaced the baseline every accuracy number that day
    had been measured against; it was recoverable only because the fit is
    deterministic and the numbers could be reproduced afterwards to prove the
    restore was faithful.  Refusing costs one flag; not refusing cost that.

    Raised before any fitting happens, so a refusal never wastes the sweep.
    """

    existing = existing_model_dirs(recording_dir)
    if not existing:
        return
    more = f", ... ({len(existing)} total)" if len(existing) > 3 else ""
    shown = ", ".join(existing[:3]) + more
    raise FileExistsError(
        f"{Path(recording_dir) / 'models'} already holds fitted models: {shown}. "
        "Fitting again would replace them in place, and anything pointing at them -- a saved "
        "profile, a live session -- would silently load different weights from the same path. "
        "Pass --overwrite-models to replace them on purpose, or --no-save-models to fit and "
        "score without writing any model."
    )


def run_phase0(
    recording_dir: Path,
    out_dir: Path,
    *,
    head_names: tuple[str, ...] = (),
    grid: Sequence[FitConfig] | None = None,
    filter_factory: Any = None,
    replay_filter: bool = True,
    save_models: bool = True,
    overwrite_models: bool = False,
    camera_pitch_deg: float | None = None,
    preset: str | None = None,
    thresholds: SEL.ScreeningThresholds | None = None,
    screening: bool = True,
) -> dict[str, Any]:
    recording_dir = Path(recording_dir)
    out_dir = Path(out_dir)
    if save_models and not overwrite_models:
        refuse_to_overwrite_models(recording_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_a = S.Recording.load(recording_dir, "A")
    rec_tune = S.Recording.load(recording_dir, "TUNE")
    rec_t1 = S.Recording.load(recording_dir, "T1")
    rec_t2 = S.Recording.load(recording_dir, "T2") if (recording_dir / "T2.npz").exists() else None
    rec_grid16 = S.Recording.load(recording_dir, "GRID16") if (recording_dir / "GRID16.npz").exists() else None
    rig = C.RigGeometry.from_dict(rec_a.meta["rig"])
    # The declared eye distance is what makes an (estimated) angle possible at
    # all; without it the report stays in pixels and says so.
    setup = rec_a.meta.get("setup") or {}
    eye_distance_cm = (setup.get("declared") or {}).get("eye_distance_cm")
    if grid is not None:
        grid = list(grid)
    else:
        import gf_presets as PRE  # noqa: PLC0415

        grid = PRE.sweep_with_presets(head_names)

    # --- rows from A: all accepted (no head filter unless head columns used)
    train_mask, train_dropped = training_rows(rec_a, head_names)
    X_train = design_matrix(rec_a, train_mask, head_names)
    Y_train = rec_a.label_cm[train_mask]
    column_std = np.std(X_train, axis=0)
    mean_sq_dist = float(np.mean(np.sum((X_train - X_train.mean(axis=0)) ** 2, axis=1))) * 2  # E||u-v||^2

    train_meta = {"protocol": "A", "round_id": rec_a.round_id, "head_dropped": train_dropped, "session": rec_a.meta.get("session")}
    sweep: list[dict[str, Any]] = []
    models: dict[str, FittedModel] = {}
    for config in grid:
        model = FittedModel.fit(config, X_train, Y_train, rig=rig, train_meta=train_meta)
        tune_metrics, _, tune_dropped = _score_timed(model, rec_tune, rig, head_names)
        sweep.append(
            {
                "name": config.name,
                "config": config.to_dict(),
                "gamma_resolved": model.schema.svr["gamma"],
                "effective_gamma_d2": model.schema.svr["gamma"] * mean_sq_dist if config.kind == "svr" else None,
                "tune": tune_metrics.flat(),
                "tune_per_target": tune_metrics.per_target,
                "tune_head_dropped": tune_dropped,
            }
        )
        models[config.name] = model
        if save_models:
            model.save(recording_dir / "models" / config.name)

    # --- selection on TUNE only, and only after a safety screen -------------
    # The old path ranked on the TUNE median alone. That is how a candidate
    # with a 142,863 px P90 on the held-out set was chosen: the median could
    # not see the tail. Screening now happens first, and a preset is screened
    # like any other candidate rather than trusted because it was named.
    thresholds = thresholds or SEL.ScreeningThresholds()
    screen_reports = screen_sweep(models, rec_tune, rig, head_names, thresholds=thresholds) if screening else []
    if screening:
        outcome = SEL.select(screen_reports, thresholds=thresholds, preset=preset)
    else:
        outcome = SEL.SelectionOutcome(select_config(sweep), "auto", [], SELECTION_METRIC, thresholds,
                                       None, "screening disabled: ranked on the TUNE median alone")
    lib_name = library_default_config(head_names).name
    if outcome.refused:
        # Report everything measured, and score the library default on the
        # test set for reference, but never present a chosen configuration.
        selected = lib_name
        Log_no_calibration = True
    else:
        selected = outcome.selected
        Log_no_calibration = False
    lib_tune = next(e["tune"] for e in sweep if e["name"] == lib_name)
    best_tune = next(e["tune"] for e in sweep if e["name"] == selected)

    # --- T1 scored once per reported model
    t1_reports: dict[str, Any] = {}
    for name in dict.fromkeys([selected, lib_name]):
        model = models[name]
        metrics, pred_all, dropped = _score_timed(model, rec_t1, rig, head_names)
        filtered = None
        if replay_filter:
            pred_px = pred_all * np.array([rig.device_w_px, rig.device_h_px])
            valid = np.all(np.isfinite(pred_all), axis=1)
            filt_px = replay_heuristic_filter(pred_px, valid, filter_factory)
            filtered = filt_px / np.array([rig.device_w_px, rig.device_h_px])
            eligible = eligible_rows(rec_t1)
            width = int(rec_t1.meta["target_geometry"]["width_px"])
            height = int(rec_t1.meta["target_geometry"]["height_px"])
            filt_metrics = evaluate(
                filtered[eligible],
                rec_t1.target_xy[eligible],
                rec_t1.target_id[eligible],
                width,
                height,
                fresh=rec_t1.rows_fresh()[eligible],
                timestamp_ns=rec_t1.timestamp_ns[eligible],
            ).flat()
        else:
            filt_metrics = None
        rows, timings = predictions_rows(rec_t1, pred_all, filtered, rig, arm="A", config_name=name)
        pred_path = C.write_predictions_json(
            out_dir / f"predictions_T1_{name}.json",
            screen_geometry=rec_t1.meta["target_geometry"],
            targets=rec_t1.targets,
            samples=rows,
            run={"engine": "gazefollower-1.0.2", "arm": "A", "config": name, "recording_round": rec_t1.round_id, "phase": "phase0", "scored_value": "pre-filter calibrated"},
            timings=timings,
        )
        entry: dict[str, Any] = {
            "t1": metrics.flat(),
            "t1_per_target": metrics.per_target,
            "t1_after_filter": filt_metrics,
            "t1_head_dropped": dropped,
            "predictions_json": str(pred_path),
            "t1_report": protocol_report(
                rec_t1, pred_all, rig, eye_distance_cm=eye_distance_cm, camera_pitch_deg=camera_pitch_deg
            ),
        }
        # The 16-point grid is scored BESIDE the held-out set, never merged
        # into it: eight of its targets sit beside a calibration point.
        if rec_grid16 is not None:
            g_metrics, g_pred, g_dropped = _score_timed(model, rec_grid16, rig, head_names)
            g_rows, g_timings = predictions_rows(rec_grid16, g_pred, None, rig, arm="A", config_name=name)
            g_path = C.write_predictions_json(
                out_dir / f"predictions_GRID16_{name}.json",
                screen_geometry=rec_grid16.meta["target_geometry"],
                targets=rec_grid16.targets,
                samples=g_rows,
                run={
                    "engine": "gazefollower-1.0.2",
                    "arm": "A",
                    "config": name,
                    "recording_round": rec_grid16.round_id,
                    "phase": "phase0",
                    "target_set": "GRID16",
                    "scored_value": "pre-filter calibrated",
                },
                timings=g_timings,
            )
            entry["grid16"] = g_metrics.flat()
            entry["grid16_head_dropped"] = g_dropped
            entry["grid16_predictions_json"] = str(g_path)
            entry["grid16_report"] = protocol_report(
                rec_grid16, g_pred, rig, eye_distance_cm=eye_distance_cm, camera_pitch_deg=camera_pitch_deg
            )
        t1_reports[name] = entry

    # --- ridge R² of target_y from features on TUNE (diagnostic)
    ridge_r2 = _ridge_r2_diagnostic(rec_a, rec_tune, head_names)

    report = {
        "recording_dir": str(recording_dir),
        "head_names": list(head_names),
        "train_rows": int(train_mask.sum()),
        "train_head_dropped": train_dropped,
        "feature_dim": rec_a.feature_dim,
        "column_std_summary": {"min": float(column_std.min()), "median": float(np.median(column_std)), "max": float(column_std.max())},
        "mean_pairwise_sq_dist": mean_sq_dist,
        "selection": {
            "metric": SELECTION_METRIC,
            "tiebreak": SELECTION_TIEBREAK,
            "selected": None if outcome.refused else selected,
            "library_default": lib_name,
            "no_acceptable_calibration": outcome.refused,
            "scored_for_reference_only": lib_name if outcome.refused else None,
            "screening": outcome.to_dict(),
        },
        "sweep": sweep,
        "t1": t1_reports,
        "p0_info": p0_info_category(t1_reports[selected]["t1"], lib_tune, best_tune),
        "p0_confound": p0_confound(rec_a),
        "p0_valid": {
            "A": p0_valid(rec_a),
            "TUNE": p0_valid(rec_tune),
            "GRID16": p0_valid(rec_grid16) if rec_grid16 is not None else None,
            "T1": p0_valid(rec_t1),
            "T2": p0_valid(rec_t2, rec_t1) if rec_t2 is not None else None,
        },
        "setup": setup or None,
        "eye_distance_cm": eye_distance_cm,
        "camera_pitch_deg": camera_pitch_deg,
        "target_proximity": {
            "T1": rec_t1.meta.get("target_proximity"),
            "GRID16": rec_grid16.meta.get("target_proximity") if rec_grid16 is not None else None,
        },
        "goal_status": goal_status(eye_distance_cm, camera_pitch_deg),
        "ridge_r2_diagnostic": ridge_r2,
    }
    (out_dir / "phase0.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (out_dir / "phase0_report.md").write_text(format_phase0_report(report), encoding="utf-8")
    return report


def _ridge_r2_diagnostic(rec_a: S.Recording, rec_tune: S.Recording, head_names: Sequence[str]) -> dict[str, Any]:
    """How much of target_y (and x) a linear model explains on TUNE. Descriptive."""

    try:
        mask_a, _ = training_rows(rec_a, head_names)
        mask_t, _ = scoring_rows(rec_tune, head_names)
        Xa = S.Standardizer.fit(design_matrix(rec_a, mask_a, head_names))
        Za = Xa.transform(design_matrix(rec_a, mask_a, head_names))
        Zt = Xa.transform(design_matrix(rec_tune, mask_t, head_names))
        Ya = rec_a.target_xy[mask_a]
        Yt = rec_tune.target_xy[mask_t]
        ridge = _ridge_fit(Za, Ya, 1.0)
        pred = _ridge_predict(ridge, Zt)
        out = {}
        for axis, name in enumerate(("x", "y")):
            ss_res = float(np.sum((Yt[:, axis] - pred[:, axis]) ** 2))
            ss_tot = float(np.sum((Yt[:, axis] - Yt[:, axis].mean()) ** 2))
            out[f"r2_{name}"] = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
        return out
    except Exception as exc:  # noqa: BLE001 - diagnostic must never abort the report
        return {"error": repr(exc)}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _outcome_from_dict(value: Mapping[str, Any]) -> SEL.SelectionOutcome:
    """Rebuild a SelectionOutcome from its serialised form, for rendering."""

    reports = []
    for r in value.get("reports", []):
        reports.append(
            SEL.CandidateReport(
                name=r["name"],
                passed=r["passed"],
                failures=[SEL.Failure(**f) for f in r.get("failures", [])],
                stats=r.get("stats", {}),
                per_fold=r.get("per_fold", []),
            )
        )
    return SEL.SelectionOutcome(
        selected=value.get("selected"),
        mode=value.get("mode", "auto"),
        reports=reports,
        ranking_metric=value.get("ranking_metric", "median_px"),
        thresholds=SEL.ScreeningThresholds(**value.get("thresholds", {})),
        preset_requested=value.get("preset_requested"),
        preset_note=value.get("preset_note", ""),
    )


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def format_phase0_report(report: Mapping[str, Any]) -> str:
    lines = ["# Phase 0 report", ""]
    lines.append(f"recording: `{report['recording_dir']}`  features: {report['feature_dim']}  train rows: {report['train_rows']} (head-dropped: {report['train_head_dropped']}, arm A is NOT head-filtered)")
    lines.append(f"column std: min {_fmt(report['column_std_summary']['min'], 4)} / median {_fmt(report['column_std_summary']['median'], 4)} / max {_fmt(report['column_std_summary']['max'], 4)}; mean pairwise ||u-v||^2 = {_fmt(report['mean_pairwise_sq_dist'], 2)}")
    lines.append("")
    sel = report["selection"]
    if sel.get("no_acceptable_calibration"):
        lines.append("## Sweep on TUNE -> **NO ACCEPTABLE CALIBRATION**")
        lines.append("")
        lines.append(
            f"No candidate passed screening. `{sel['scored_for_reference_only']}` is scored on the test "
            "set below FOR REFERENCE ONLY; it is not a chosen configuration and its numbers must not be "
            "reported as an accepted result."
        )
    else:
        lines.append(
            f"## Sweep on TUNE (screened, then ranked by {sel['metric']}) -> selected **{sel['selected']}**"
        )
    lines.append("")
    lines.append("| config | euclid med | P90 | \\|dx\\| | bias x | slope x | \\|dy\\| | bias y | slope y | off y | cover | gamma*d^2 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for entry in sorted(report["sweep"], key=lambda e: _sweep_sort_key(e["tune"])):
        t = entry["tune"]
        lines.append(
            f"| {entry['name']} | {_fmt(t['median_euclid_px'])} | {_fmt(t['p90_euclid_px'])} | "
            f"{_fmt(t['x_median_abs_px'])} | {_fmt(t['x_bias_px'])} | {_fmt(t['x_slope'], 3)} | "
            f"{_fmt(t['y_median_abs_px'])} | {_fmt(t['y_bias_px'])} | {_fmt(t['y_slope'], 3)} | "
            f"{t['off_screen_y']} | {_fmt(100 * t['coverage'], 1)}% | {_fmt(entry['effective_gamma_d2'], 3)} |"
        )
    lines.append("")
    lines.append("## T1 (scored ONCE per reported model)")
    lines.append("")
    for name, rep in report["t1"].items():
        t = rep["t1"]
        lines.append(f"### {name}")
        lines.append(
            f"- pre-filter: euclid mean {_fmt(t['mean_euclid_px'])} / median {_fmt(t['median_euclid_px'])} / "
            f"P90 {_fmt(t['p90_euclid_px'])} / P95 {_fmt(t['p95_euclid_px'])} / max {_fmt(t['max_euclid_px'])} px"
        )
        lines.append(
            f"  X: |dx| median {_fmt(t['x_median_abs_px'])}, P90 {_fmt(t['x_p90_abs_px'])}, "
            f"bias {_fmt(t['x_bias_px'])}, slope {_fmt(t['x_slope'], 3)}"
        )
        lines.append(
            f"  Y: |dy| median {_fmt(t['y_median_abs_px'])}, P90 {_fmt(t['y_p90_abs_px'])}, "
            f"bias {_fmt(t['y_bias_px'])}, slope {_fmt(t['y_slope'], 3)}"
        )
        lines.append(
            f"  coverage {_fmt(100 * t['coverage'], 1)}% ({t['n_valid']}/{t['n_eligible']}), "
            f"fresh {_fmt(100 * t['coverage_fresh'], 1)}%, lost {t['n_lost']}, "
            f"longest gap {_fmt(t['longest_gap_ms'])} ms | off-screen x/y {t['off_screen_x']}/{t['off_screen_y']}"
        )
        f = rep.get("t1_after_filter")
        if f:
            lines.append(
                f"- after HeuristicFilter: euclid {_fmt(f['median_euclid_px'])}px (P90 {_fmt(f['p90_euclid_px'])}) | "
                f"X: |dx| {_fmt(f['x_median_abs_px'])}, bias {_fmt(f['x_bias_px'])}, slope {_fmt(f['x_slope'], 3)} | "
                f"Y: |dy| {_fmt(f['y_median_abs_px'])}, bias {_fmt(f['y_bias_px'])}, slope {_fmt(f['y_slope'], 3)}"
            )
        lines.append(f"- predictions JSON: `{rep['predictions_json']}`")
        lines.append("")
        lines.append("| target | tx | ty | pred x | pred y | dx | dy | euclid | n |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for pt in rep["t1_per_target"]:
            lines.append(f"| {pt['name']} | {_fmt(pt['tx'], 3)} | {_fmt(pt['ty'], 3)} | {_fmt(pt['pred_x_median'], 3)} | {_fmt(pt['pred_y_median'], 3)} | {_fmt(pt['median_dx_px'])} | {_fmt(pt['median_dy_px'])} | {_fmt(pt['median_euclid_px'])} | {pt['n']} |")
        lines.append("")
    info = report["p0_info"]
    lines.append(f"## P0-INFO \u2014 {info['category']}")
    lines.append(info["statement"])
    lines.append(
        f"T1 Y: slope {_fmt(info['t1_y_slope'], 3)}, median |dy| {_fmt(info['t1_y_median_abs_px'])}px, "
        f"bias {_fmt(info['t1_y_bias_px'])}px | X: slope {_fmt(info['t1_x_slope'], 3)}, "
        f"median |dx| {_fmt(info['t1_x_median_abs_px'])}px, bias {_fmt(info['t1_x_bias_px'])}px"
    )
    lines.append(
        f"TUNE median |dy|: library default {_fmt(info['tune_lib_y_median_abs_px'])}px "
        f"-> selected {_fmt(info['tune_best_y_median_abs_px'])}px"
    )
    lines.append("")
    confound = report["p0_confound"]
    lines.append("## P0-CONFOUND \u2014 descriptive, not causal")
    lines.append(
        f"rows {confound['n_rows']}: corr(pitch_a, target_y) = {_fmt(confound['corr_pitch_a_target_y'], 3)}; "
        f"corr(yaw_ratio, target_x) = {_fmt(confound['corr_yaw_ratio_target_x'], 3)}; "
        f"pitch_a IQR {_fmt(confound['pitch_a_iqr'], 4)}"
    )
    lines.append("")
    lines.append("## P0-VALID \u2014 recording integrity (an unknown check counts as invalid)")
    for proto, valid in report["p0_valid"].items():
        if valid is None:
            lines.append(f"- {proto}: not recorded")
            continue
        verdict = "VALID" if valid["valid"] else "INVALID"
        detail = ""
        if valid["failed_checks"]:
            detail += f" FAILED: {', '.join(valid['failed_checks'])}."
        if valid["unknown_checks"]:
            detail += f" UNKNOWN: {', '.join(valid['unknown_checks'])}."
        if valid.get("missing_targets"):
            detail += f" missing targets: {valid['missing_targets']}."
        coverage = valid["coverage"]
        head_bad = valid["head_invalid_rate_among_gaze_rows"]
        lines.append(
            f"- {proto}: {verdict} \u2014 rows {valid['n_rows']}, expected targets {valid['expected_targets']}, "
            f"fps {_fmt(valid['fps_median'])}, coverage {_fmt(None if coverage is None else 100 * coverage, 1)}%, "
            f"head-invalid {_fmt(None if head_bad is None else 100 * head_bad, 2)}%.{detail}"
        )
    lines.append("")
    proximity = report.get("target_proximity") or {}
    for label, summary in proximity.items():
        if summary:
            lines.append(
                f"{label}: {summary['n_near_calibration']}/{summary['n']} targets sit closer than "
                f"{summary['threshold_px']:.0f}px to a calibration point (min {summary['min_px']:.0f}px)"
            )
    lines.append("")

    for name, rep in report["t1"].items():
        for key, title in (("t1_report", "T1 (held-out)"), ("grid16_report", "GRID16")):
            block = rep.get(key)
            if block:
                lines.append(f"### {title} zones and matrix -- {name}")
                lines.append(RP.format_report(block))

    goals = report.get("goal_status") or {}
    if goals:
        lines.append("## Product goals after this recording")
        lines.append("")
        lines.append("| goal | status | why |")
        lines.append("|---|---|---|")
        for key, value in goals.items():
            if isinstance(value, dict):
                lines.append(f"| {key} | **{value['status']}** | {value['why']} |")
        lines.append(f"| decision | **{goals.get('decision')}** | {goals.get('note')} |")
        lines.append("")

    screening = (report.get("selection") or {}).get("screening")
    if screening and screening.get("reports"):
        lines.append(SEL.format_selection(_outcome_from_dict(screening)))

    r2 = report["ridge_r2_diagnostic"]
    lines.append(f"ridge R2 on TUNE (diagnostic): x {_fmt(r2.get('r2_x'), 3)}, y {_fmt(r2.get('r2_y'), 3)}")
    lines.append("")
    if report.get("eye_distance_cm") is None:
        degrees_note = "No degrees: no eye distance was declared, and pixels alone cannot give an angle."
    else:
        tilt = "modelled" if report.get("camera_pitch_deg") is not None else "NOT modelled"
        degrees_note = (
            f"Degrees are ESTIMATED from a declared eye distance of {report['eye_distance_cm']} cm, "
            f"camera tilt {tilt}; the accuracy goal stays NOT VERIFIED."
        )
    lines.append(f"Units: logical px (analyze.py geometry). {degrees_note} Mean, median and P90 are all reported.")
    return "\n".join(lines) + "\n"


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p0 = sub.add_parser("phase0", help="sweep on TUNE, score T1 once, report G1-G3")
    p0.add_argument("--recording", type=Path, required=True, help="recordings/round0")
    p0.add_argument("--out", type=Path, required=True, help="results directory (no embeddings written here)")
    p0.add_argument("--head", choices=("none", "head3", "head6"), default="none", help="append head columns (default none: arm A)")
    p0.add_argument("--no-filter-replay", action="store_true", help="skip the HeuristicFilter replay (avoids importing gazefollower)")
    p0.add_argument("--no-save-models", action="store_true")
    p0.add_argument(
        "--overwrite-models",
        action="store_true",
        help=(
            "replace fitted models already saved under this recording. Without it, fitting a "
            "recording that already has models is refused before the sweep starts, so a model "
            "a profile or a live session points at cannot be replaced by accident."
        ),
    )
    p0.add_argument("--preset", default=None, help="use this named preset instead of the ranked winner (still screened)")
    p0.add_argument("--no-screening", action="store_true", help="rank on the median alone, as the old selector did (diagnostic)")
    p0.add_argument("--list-presets", action="store_true", help="print the preset registry and exit")
    p0.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=None,
        help="measured upward tilt of the camera; without it the tilt stays an unmodelled error",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phase0":
        if args.list_presets:
            import gf_presets as PRE  # noqa: PLC0415

            print(PRE.describe_all())
            return 0
        head = {"none": (), "head3": H.HEAD3_NAMES, "head6": H.HEAD6_NAMES}[args.head]
        try:
            report = run_phase0(
                args.recording,
                args.out,
                head_names=head,
                replay_filter=not args.no_filter_replay,
                save_models=not args.no_save_models,
                overwrite_models=args.overwrite_models,
                camera_pitch_deg=args.camera_pitch_deg,
                preset=args.preset,
                screening=not args.no_screening,
            )
        except FileExistsError as exc:
            # A refusal is an answer, not a crash: print it as one, so the
            # operator reads what to do instead of reading a stack trace.
            print(f"REFUSED: {exc}")
            return 2
        print((args.out / "phase0_report.md").read_text(encoding="utf-8"))
        sel = report["selection"]
        if sel.get("no_acceptable_calibration"):
            print(f"NO ACCEPTABLE CALIBRATION -- {sel['screening']['preset_note']}")
        else:
            print(f"selected: {sel['selected']} ({sel['screening']['mode']})  P0-INFO: {report['p0_info']['category']}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
