"""The fitted gaze model: configuration, prediction, save and load (ARCH-01 stage D).

Moved out of ``gf_fit`` (the fitting and evaluation TOOL) so the live session
can load and run a model without importing the tool. What stays in ``gf_fit``
is the orchestration: sweeps, selection, scoring and reports.

``FittedModel.fit`` stays with the class: it is how a model is constructed
from rows, not how an experiment chooses one. Behaviour, numerics and the
on-disk format are unchanged; ``gf_fit`` re-exports every name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gazelink_core.calibration import schema as S
from gazelink_core.domain import common as C
from gazelink_core.gaze import head_features as H

# calibration/SVRCalibration.py::_set_svm_params -- the library's fixed choice.
LIBRARY_SVR = {"C": 1.0, "gamma": 0.005, "P": 0.001}
SVR_MAX_ITER = 10000
SVR_TERM_EPS = 1e-4


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
