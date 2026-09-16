"""On-disk formats for the experiment: recordings, feature schemas, scaling.

Three things live here, all pure (numpy only, no gazefollower import):

``Recording``      -- one protocol's per-frame rows as arrays + a meta dict,
                      saved as ``<protocol>.npz`` + ``<protocol>.meta.json``.
                      Invalid values are NaN, never zero, so an accidental use
                      of a row that should have been excluded propagates
                      loudly instead of training on silent zeros.
``FeatureSchema``  -- what a fitted model expects: column names and order,
                      scaling statistics learned from the TRAINING rows only,
                      SVR parameters, builder version, rig geometry, library
                      versions. A model whose schema does not match the
                      caller's columns must refuse to run (``check_columns``).
``Standardizer``   -- per-column z-scoring with the constant-column rule
                      (std 0 -> 1, so the column stays 0 after centring).

Privacy note: a recording's ``features`` array is the 258-d model embedding
of the face/eye crops, and a fitted SVR's support vectors ARE training rows.
Both belong under ``recordings/`` and nowhere else (plan §7).
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

RECORDING_FORMAT_VERSION = "rec-1"
SCHEMA_VERSION = "schema-1"

# Below this fraction of a column's own magnitude, its spread is floating-point
# noise rather than signal, and the column is treated as constant.
CONSTANT_COLUMN_RTOL = 1e-9
# A calibration is 35 seconds at one head pose, so a column can sit almost
# still for the whole of it and still move later in ordinary use. Dividing by
# that tiny observed spread turns a small real change into an enormous
# z-score: in round5's model, column 140 had a training std of 0.000327, and a
# later shift of 0.28 -- unremarkable in the raw units -- became z=860 and
# carried the sample outside the RBF kernel on its own. The SVR then answered
# with its constant bias and the on-screen point froze.
#
# So no column may claim a spread far below the feature space's own scale.
# The floor is relative to the widest column, which keeps it meaningful
# whatever the embedding's units. Measured on round5/A: it lifts 144 of 258
# columns, improves round6/T1 median error from 209.3px to 181.8px, and lifts
# kernel activation on later sessions from 0.000 (frozen output) to ~0.77.
VARIANCE_FLOOR_FRACTION = 0.01

# Protocol identifiers. Calibration-style protocols store exactly
# N_FRAMES_PER_POINT accepted rows per target; timed ones store whatever the
# window yields and the fitter scores the COLLECTING phase.
CALIBRATION_PROTOCOLS = ("A", "B")
TIMED_PROTOCOLS = ("TUNE", "GRID16", "T1", "T2", "T3", "FULL", "MOVE")
ALL_PROTOCOLS = CALIBRATION_PROTOCOLS + TIMED_PROTOCOLS

# Phase vocabulary, per row.
PHASE_WARMUP = "WARMUP"  # calibration: the unstored centre point
PHASE_PREPARE = "PREPARE"  # calibration: 1.5 s after onset, nothing stored
PHASE_COLLECT = "COLLECT"  # calibration: frames counted toward the 45
PHASE_WAIT = "WAIT"  # calibration: 0.5 s after the 45th, nothing stored
PHASE_STABILIZING = "STABILIZING"  # timed: settle window
PHASE_COLLECTING = "COLLECTING"  # timed: scored window (analyze.py's RESTING_PHASE)
PHASE_IDLE = "IDLE"  # no target shown (countdowns, warm-up seconds)

# Measurement contract: a gaze output older than this is not a fresh
# measurement, so it must not be counted toward the sampling rate or toward
# availability. A repeated frame or a frozen cursor is not a new measurement.
FRESHNESS_LIMIT_MS = 100.0
# An unplanned gap at least this long, while the eyes are open and the head is
# inside test conditions, fails availability on its own.
DROPOUT_FAIL_MS = 1000.0

# TrackingState names as the library spells them (misc.Enumeration).
TRACKING_STATES = ("SUCCESS", "FACE_MISSING", "OUT_OF_BOUNDARIES", "FAILURE", "UNKNOWN")


@dataclass
class Recording:
    """Per-frame rows of one protocol. All arrays share the first dimension."""

    protocol: str
    round_id: int
    meta: dict[str, Any]
    frame_seq: np.ndarray  # int64
    timestamp_ns: np.ndarray  # int64, library time_ns() per frame
    elapsed_ms: np.ndarray  # float64, since current target onset (NaN when idle)
    target_id: np.ndarray  # int32, index into meta["targets"]; -1 when idle
    block: np.ndarray  # int32, pose block for protocol B; 0 otherwise
    phase: np.ndarray  # unicode, one of the PHASE_* values
    target_xy: np.ndarray  # (n, 2) float64 normalised; NaN when idle
    label_cm: np.ndarray  # (n, 2) float64 library label space; NaN when idle
    features: np.ndarray  # (n, d) float32 model output; NaN row when gaze_status False
    head: np.ndarray  # (n, 6) float64 head6; NaN row when head_valid False
    head_valid: np.ndarray  # bool
    pnp_deg: np.ndarray  # (n, 3) float64 approximate yaw/pitch/roll; NaN when unavailable
    raw_cm: np.ndarray  # (n, 2) float64 model res[:2], diagnostic only
    openness: np.ndarray  # (n, 2) float64 left/right eye polygon area px^2
    tracking_state: np.ndarray  # unicode, TRACKING_STATES
    gaze_status: np.ndarray  # bool, gaze_info.status
    accepted: np.ndarray  # bool, the controller's gate for this row (see gf_record)

    def __post_init__(self) -> None:
        n = len(self.frame_seq)
        for name in (
            "timestamp_ns",
            "elapsed_ms",
            "target_id",
            "block",
            "phase",
            "target_xy",
            "label_cm",
            "features",
            "head",
            "head_valid",
            "pnp_deg",
            "raw_cm",
            "openness",
            "tracking_state",
            "gaze_status",
            "accepted",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Recording.{name} has {len(getattr(self, name))} rows, expected {n}")
        if self.protocol not in ALL_PROTOCOLS:
            raise ValueError(f"unknown protocol {self.protocol!r}")

    @property
    def n_rows(self) -> int:
        return int(len(self.frame_seq))

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    @property
    def targets(self) -> list[dict[str, Any]]:
        return list(self.meta.get("targets", []))

    def save(self, directory: Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz_path = directory / f"{self.protocol}.npz"
        meta_path = directory / f"{self.protocol}.meta.json"
        np.savez_compressed(
            npz_path,
            frame_seq=self.frame_seq,
            timestamp_ns=self.timestamp_ns,
            elapsed_ms=self.elapsed_ms,
            target_id=self.target_id,
            block=self.block,
            phase=self.phase,
            target_xy=self.target_xy,
            label_cm=self.label_cm,
            features=self.features,
            head=self.head,
            head_valid=self.head_valid,
            pnp_deg=self.pnp_deg,
            raw_cm=self.raw_cm,
            openness=self.openness,
            tracking_state=self.tracking_state,
            gaze_status=self.gaze_status,
            accepted=self.accepted,
        )
        meta = dict(self.meta)
        meta.update(
            {
                "recording_format": RECORDING_FORMAT_VERSION,
                "protocol": self.protocol,
                "round_id": int(self.round_id),
                "n_rows": self.n_rows,
                "feature_dim": self.feature_dim,
            }
        )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return npz_path, meta_path

    @classmethod
    def load(cls, directory: Path, protocol: str) -> "Recording":
        directory = Path(directory)
        meta = json.loads((directory / f"{protocol}.meta.json").read_text(encoding="utf-8"))
        if meta.get("recording_format") != RECORDING_FORMAT_VERSION:
            raise ValueError(
                f"{directory}/{protocol}: recording format {meta.get('recording_format')!r} "
                f"!= {RECORDING_FORMAT_VERSION!r}"
            )
        with np.load(directory / f"{protocol}.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return cls(protocol=protocol, round_id=int(meta["round_id"]), meta=meta, **arrays)

    # Row selections used by the fitter. Kept here so every arm applies the
    # same definition and a test can pin it.
    def rows_accepted(self) -> np.ndarray:
        return np.asarray(self.accepted, dtype=bool)

    def rows_eligible(self) -> np.ndarray:
        """Every row that SHOULD have produced a measurement.

        The denominator. Deliberately does NOT require ``gaze_status``: a
        frame the tracker lost is exactly what availability has to count. An
        earlier version filtered on gaze status here, which made tracking loss
        vanish from the report (accuracy conditioned on the survivors while
        ``n_lost`` read 0). Coverage is computed against this mask.

        The collecting phase is named differently for the two protocol kinds:
        a calibration protocol keeps showing a point until 45 frames are
        ACCEPTED, so its eligible window is every frame shown while it was
        trying -- which is what reveals how much tracking loss stretched the
        calibration out.
        """

        if self.protocol in CALIBRATION_PROTOCOLS:
            return self.phase == PHASE_COLLECT
        return self.phase == PHASE_COLLECTING

    def rows_collecting(self) -> np.ndarray:
        """The eligible rows that actually carry a gaze sample.

        A strict subset of :meth:`rows_eligible`. Use it for feature
        extraction; never as the denominator of an accuracy or coverage
        report, or the loss disappears.
        """

        return self.rows_eligible() & np.asarray(self.gaze_status, dtype=bool)

    def rows_head_valid(self) -> np.ndarray:
        return np.asarray(self.head_valid, dtype=bool)

    def sample_age_ms(self) -> np.ndarray:
        """Age of each row's gaze sample, in ms, from the previous valid one.

        Freshness, per the measurement contract: a prediction older than
        ``FRESHNESS_LIMIT_MS`` is not a new measurement even if a value is on
        screen. Rows before the first valid sample get ``inf``.
        """

        ts_ms = np.asarray(self.timestamp_ns, dtype=np.float64) / 1e6
        valid = np.asarray(self.gaze_status, dtype=bool)
        ages = np.full(len(ts_ms), np.inf)
        last = None
        for i, (t, ok) in enumerate(zip(ts_ms, valid)):
            if ok:
                last = t
                ages[i] = 0.0
            elif last is not None:
                ages[i] = t - last
        return ages

    def rows_fresh(self, limit_ms: float = FRESHNESS_LIMIT_MS) -> np.ndarray:
        return self.sample_age_ms() <= limit_ms


class RecordingBuilder:
    """Accumulates per-frame rows in Python lists, then freezes to arrays.

    The recorder appends from the camera thread; ``append`` does no numpy
    work beyond copying small vectors so it stays cheap there.
    """

    def __init__(self, protocol: str, round_id: int, meta: Mapping[str, Any], feature_dim: int | None = None):
        if protocol not in ALL_PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}")
        self.protocol = protocol
        self.round_id = int(round_id)
        self.meta = dict(meta)
        self._feature_dim = feature_dim
        self._rows: list[dict[str, Any]] = []

    @property
    def feature_dim(self) -> int | None:
        return self._feature_dim

    def append(
        self,
        *,
        frame_seq: int,
        timestamp_ns: int,
        elapsed_ms: float,
        target_id: int,
        block: int,
        phase: str,
        target_xy: tuple[float, float] | None,
        label_cm: tuple[float, float] | None,
        features: np.ndarray | None,
        head: np.ndarray | None,
        pnp_deg: tuple[float, float, float] | None,
        raw_cm: tuple[float, float] | None,
        openness: tuple[float, float],
        tracking_state: str,
        gaze_status: bool,
        accepted: bool,
    ) -> None:
        if features is not None:
            dim = int(np.asarray(features).shape[0])
            if self._feature_dim is None:
                self._feature_dim = dim
            elif dim != self._feature_dim:
                raise ValueError(f"feature dim changed mid-recording: {dim} != {self._feature_dim}")
        if accepted and (features is None or not gaze_status):
            raise ValueError("a row cannot be accepted without a valid gaze sample")
        self._rows.append(
            {
                "frame_seq": int(frame_seq),
                "timestamp_ns": int(timestamp_ns),
                "elapsed_ms": float("nan") if elapsed_ms is None else float(elapsed_ms),
                "target_id": int(target_id),
                "block": int(block),
                "phase": str(phase),
                "target_xy": None if target_xy is None else (float(target_xy[0]), float(target_xy[1])),
                "label_cm": None if label_cm is None else (float(label_cm[0]), float(label_cm[1])),
                "features": None if features is None else np.asarray(features, dtype=np.float32).copy(),
                "head": None if head is None else np.asarray(head, dtype=np.float64).copy(),
                "pnp_deg": None if pnp_deg is None else tuple(float(v) for v in pnp_deg),
                "raw_cm": None if raw_cm is None else (float(raw_cm[0]), float(raw_cm[1])),
                "openness": (float(openness[0]), float(openness[1])),
                "tracking_state": str(tracking_state),
                "gaze_status": bool(gaze_status),
                "accepted": bool(accepted),
            }
        )

    def __len__(self) -> int:
        return len(self._rows)

    def freeze(self) -> Recording:
        n = len(self._rows)
        dim = self._feature_dim or 0
        nan2 = np.full((n, 2), np.nan)

        def col(key: str, dtype: Any) -> np.ndarray:
            return np.asarray([row[key] for row in self._rows], dtype=dtype)

        def pair(key: str) -> np.ndarray:
            out = nan2.copy()
            for i, row in enumerate(self._rows):
                if row[key] is not None:
                    out[i] = row[key]
            return out

        features = np.full((n, dim), np.nan, dtype=np.float32)
        head = np.full((n, 6), np.nan, dtype=np.float64)
        pnp = np.full((n, 3), np.nan, dtype=np.float64)
        head_valid = np.zeros(n, dtype=bool)
        for i, row in enumerate(self._rows):
            if row["features"] is not None:
                features[i] = row["features"]
            if row["head"] is not None:
                head[i] = row["head"]
                head_valid[i] = True
            if row["pnp_deg"] is not None:
                pnp[i] = row["pnp_deg"]
        return Recording(
            protocol=self.protocol,
            round_id=self.round_id,
            meta=self.meta,
            frame_seq=col("frame_seq", np.int64),
            timestamp_ns=col("timestamp_ns", np.int64),
            elapsed_ms=col("elapsed_ms", np.float64),
            target_id=col("target_id", np.int32),
            block=col("block", np.int32),
            phase=col("phase", "U12"),
            target_xy=pair("target_xy"),
            label_cm=pair("label_cm"),
            features=features,
            head=head,
            head_valid=head_valid,
            pnp_deg=pnp,
            raw_cm=pair("raw_cm"),
            openness=np.asarray([row["openness"] for row in self._rows], dtype=np.float64).reshape(n, 2),
            tracking_state=col("tracking_state", "U18"),
            gaze_status=col("gaze_status", bool),
            accepted=col("accepted", bool),
        )


# --- Feature schema and scaling -------------------------------------------


def base_column_names(base_dim: int) -> tuple[str, ...]:
    return tuple(f"f{i:03d}" for i in range(base_dim))


def assemble(
    features: np.ndarray, head: np.ndarray | None, head_names: Sequence[str], all_head_names: Sequence[str]
) -> np.ndarray:
    """Concatenate base features with the selected head columns, in schema order.

    ``head`` is the full head6 matrix (n x 6); ``head_names`` selects and
    orders the columns to append. With no head names the result is the base
    matrix unchanged, so arms A/B and C share one code path.
    """

    base = np.asarray(features, dtype=np.float64)
    if base.ndim != 2:
        raise ValueError("features must be 2-D (rows x columns)")
    if not head_names:
        return base
    if head is None:
        raise ValueError("head columns requested but no head matrix given")
    head = np.asarray(head, dtype=np.float64)
    if head.shape[0] != base.shape[0]:
        raise ValueError("features and head have different row counts")
    index = {name: i for i, name in enumerate(all_head_names)}
    cols = [index[name] for name in head_names]
    return np.concatenate([base, head[:, cols]], axis=1)


@dataclass
class Standardizer:
    """Per-column z-score with statistics learned from the training rows only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("need a 2-D matrix with at least two rows to standardise")
        if not np.all(np.isfinite(X)):
            raise ValueError("training matrix contains NaN/inf: exclude invalid rows before fitting")
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A column that is constant in exact arithmetic can still show a tiny
        # non-zero std from floating-point summation (45 copies of 0.1 give
        # ~4e-17). Testing `std > 0` would then divide by that dust and turn a
        # constant column into values of +/-1. Scale the floor to the column's
        # own magnitude instead.
        dust = CONSTANT_COLUMN_RTOL * np.maximum(1.0, np.abs(mean))
        std = np.where(std > dust, std, 1.0)
        # Guard the near-constant columns too, not only the exactly-constant
        # ones: see VARIANCE_FLOOR_FRACTION.
        widest = float(std.max())
        if np.isfinite(widest) and widest > 0.0:
            std = np.maximum(std, VARIANCE_FLOOR_FRACTION * widest)
        return cls(mean=mean, std=std)

    @classmethod
    def identity(cls, n_cols: int) -> "Standardizer":
        return cls(mean=np.zeros(n_cols), std=np.ones(n_cols))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} columns, got {X.shape[-1]}")
        return (X - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Standardizer":
        return cls(mean=np.asarray(value["mean"], dtype=np.float64), std=np.asarray(value["std"], dtype=np.float64))


@dataclass
class FeatureSchema:
    """Everything a fitted model needs to refuse mismatched input."""

    base_dim: int
    head_names: tuple[str, ...]
    feature_scaling: str  # "none" | "zscore"
    label_scaling: str  # "none" | "zscore"
    features: Standardizer
    labels: Standardizer
    svr: dict[str, float]  # C, gamma, P
    builder_version: str | None
    rig: dict[str, Any]
    train: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def columns(self) -> tuple[str, ...]:
        return base_column_names(self.base_dim) + tuple(self.head_names)

    @property
    def n_columns(self) -> int:
        return self.base_dim + len(self.head_names)

    def check_columns(self, n_columns: int, builder_version: str | None) -> None:
        if n_columns != self.n_columns:
            raise ValueError(f"schema expects {self.n_columns} columns, caller has {n_columns}")
        if self.head_names and builder_version != self.builder_version:
            raise ValueError(
                f"schema built with head builder {self.builder_version!r}, caller has {builder_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_dim": self.base_dim,
            "head_names": list(self.head_names),
            "columns": list(self.columns),
            "feature_scaling": self.feature_scaling,
            "label_scaling": self.label_scaling,
            "features": self.features.to_dict(),
            "labels": self.labels.to_dict(),
            "svr": dict(self.svr),
            "builder_version": self.builder_version,
            "rig": dict(self.rig),
            "train": dict(self.train),
            "versions": dict(self.versions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureSchema":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema version {value.get('schema_version')!r} != {SCHEMA_VERSION!r}")
        schema = cls(
            base_dim=int(value["base_dim"]),
            head_names=tuple(value["head_names"]),
            feature_scaling=str(value["feature_scaling"]),
            label_scaling=str(value["label_scaling"]),
            features=Standardizer.from_dict(value["features"]),
            labels=Standardizer.from_dict(value["labels"]),
            svr={k: float(v) for k, v in value["svr"].items()},
            builder_version=value.get("builder_version"),
            rig=dict(value["rig"]),
            train=dict(value.get("train", {})),
            versions=dict(value.get("versions", {})),
        )
        if list(value.get("columns", [])) != list(schema.columns):
            raise ValueError("schema column list does not match base_dim + head_names")
        return schema

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureSchema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def environment_versions() -> dict[str, Any]:
    """Library versions worth pinning inside a schema. cv2 is imported lazily."""

    versions: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import cv2  # noqa: PLC0415

        versions["cv2"] = cv2.__version__
    except Exception:  # noqa: BLE001 - absence is itself the information
        versions["cv2"] = None
    return versions
