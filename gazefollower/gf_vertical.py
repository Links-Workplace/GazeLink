"""Why the calibrated model's vertical axis is compressed, and whether the
calibration layer can fix it (M2-05, milestone M2).

The measured problem
--------------------
The configuration Phase 0 selected on round18 (``svr_fzscore_lnone_C100_g0.0005``)
reproduces the target's horizontal position well and its vertical position
badly::

    axis   slope   median |error|   bias
    X      1.095      72.9 px      -12.5 px
    Y      0.740     122.3 px     +120.5 px

A slope below 1 means the prediction under-responds: the target moves 100 px
down and the prediction follows only 74 px.  ``gf_fit.py phase0`` swept ~40
configurations and returned ``no_improvement``.

This module asks whether that verdict is a property of the FEATURES or a
property of the SEARCH, and separates the two questions:

``sweep``      Can any calibration-layer setting -- feature/label scaling, SVR
               C / gamma / epsilon, extra head-pose columns -- recover the
               vertical axis?  Fit on A, compare on TUNE, score the chosen
               candidate on the held-out protocols exactly once.
``signal``     Independently of any SVR: how much vertical information is
               recoverable from these features at all?  Grouped
               cross-validation over calibration targets with several
               regressor families, reported per axis so X is the control.
``geometry``   Descriptive statistics of the feature space (column spread,
               variance share, between/within-target separability) in the raw
               and scaled spaces the kernel actually sees.

Method rules this module enforces in code
-----------------------------------------
* Candidates are fitted on protocol A and compared on TUNE ONLY.  The
  selection rule is declared in :data:`RULE` before any test protocol is
  touched, and :func:`select` sees TUNE metrics only -- it is not passed the
  test metrics at all.
* Test protocols are scored once, at the end, for the chosen candidate and for
  the baseline (whose test numbers are already published in
  ``results/round18/phase0.json``, so they leak nothing new).
* No evaluation label is ever used to fit anything.  ``label_cm`` from A is
  the only supervision.
* Slope and bias are first-class: :class:`AxisReport` carries them next to the
  error percentiles, because a candidate that lowers median error while making
  the slope worse is not an improvement in the thing being investigated.

What the round18 run found (recorded here so the next run starts informed)
-------------------------------------------------------------------------
Feature scaling is the dominant lever on the vertical axis, and it is
categorical: of 864 swept SVR candidates with ``feature_scaling="zscore"``,
NONE reached ``|1 - y_slope| <= 0.1`` on TUNE; of the 864 with
``feature_scaling="none"``, 523 did.  Extra head columns -- ``pitch_a``,
``eye_mid_y``, ``iod_norm``, the ``pnp_deg`` pitch -- moved the median TUNE
y_slope by less than 0.05 inside either family, so the "vertical gaze is
confounded with head pitch" hypothesis was not supported by this data.

Because the horizontal axis wants the opposite setting, no SINGLE
configuration served both axes; the per-axis arm (two SVRs with different
hyper-parameters, which the library's structure already allows) is what
carried the slope fix onto the held-out protocols.

``RULE`` below is left exactly as it was declared, because the run's most
useful methodological result is that its objective was the wrong one:
measured over 400 candidates on round18/T1, TUNE ``y_slope`` predicts held-out
``y_slope`` with Spearman +0.955, while TUNE ``y_median_abs_px`` predicts
held-out ``y_median_abs_px`` with only +0.730.  Minimising the vertical MEDIAN
on TUNE therefore selected a candidate whose advantage was largely noise, and
it did not survive to either test protocol.  Anyone re-running this should
treat the slope as the transferable quantity -- but that observation is itself
post-hoc and wants prospective confirmation on a fresh session.

Units
-----
Errors, bias and percentiles are logical screen pixels, using the recording's
own ``target_geometry`` and analyze.py's ``normalised * (size - 1)``
convention, identical to :func:`gf_fit.evaluate`.  Slope is dimensionless
(normalised prediction per normalised target, 1.0 = perfect).  Angles from
``pnp_deg`` are degrees; ``head6`` columns are the dimensionless ratios
documented in :mod:`gf_head_features`.

Safety / privacy
----------------
Offline replay only.  This module never imports ``gazefollower``, never opens
a camera, never emits OS input, and never writes into ``recordings/`` -- it
reads recordings and writes its artefacts wherever ``--out`` points.  It logs
aggregate statistics, never per-frame feature values.

Usage::

    python gf_vertical.py --recording recordings/round18 --out results/vertical
    python gf_vertical.py --recording recordings/round18 \
        --confirm recordings/round19 --out results/vertical
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_fit as F  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_schema as S  # noqa: E402

# --- What may be appended to the 258-column embedding -------------------------

# ``gf_schema.assemble`` appends head6 columns; the pnp angles are not part of
# head6, so they are appended here.  Both are named in one namespace so a
# candidate can ask for any mixture.
PNP_COLUMN_NAMES: tuple[str, ...] = ("pnp_yaw", "pnp_pitch", "pnp_roll")
PNP_INDEX: dict[str, int] = {name: i for i, name in enumerate(PNP_COLUMN_NAMES)}
EXTRA_COLUMN_NAMES: tuple[str, ...] = tuple(H.HEAD6_NAMES) + PNP_COLUMN_NAMES

# The Phase-0 selection on round18, and the configuration this investigation
# is trying to beat.  Declared as a candidate so it is swept like any other.
BASELINE = "base|f=zscore|l=none|C=100|g=0.0005|P=0.001"


# --- Metrics ------------------------------------------------------------------


@dataclass(frozen=True)
class AxisReport:
    """One axis of one scored protocol, in logical screen pixels.

    ``slope``/``intercept`` are the least-squares fit of the per-target MEDIAN
    prediction against the target position, in normalised units -- the same
    quantity :func:`gf_fit._axis_metrics` reports, so numbers here are directly
    comparable with ``phase0.json``.  ``bias_px`` is the median signed error
    over valid rows.

    ``degenerate`` is non-None when the input cannot support the statistic.
    Everything else is then None rather than a number computed from nothing:
    an empty or all-lost protocol must read as "not measured", never as a
    silent 0.0 or an average over one point.
    """

    axis: str  # "x" | "y"
    n_eligible: int
    n_valid: int
    coverage: float | None
    n_target_positions: int
    slope: float | None
    intercept: float | None
    bias_px: float | None
    median_abs_px: float | None
    p90_abs_px: float | None
    degenerate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_slope_bias(
    targets: Sequence[float], preds: Sequence[float]
) -> tuple[float | None, float | None]:
    """Least-squares slope/intercept of ``preds`` on ``targets``.

    Returns ``(None, None)`` when the fit is not defined -- fewer than two
    points, or every point at the same target position.  A single target
    position cannot distinguish a perfect model from a constant one, and
    returning 0.0 there would read as "fully compressed" when the truth is
    "not measurable".
    """

    t = np.asarray(targets, dtype=np.float64)
    p = np.asarray(preds, dtype=np.float64)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError("targets and preds must be 1-D arrays of the same length")
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size < 2:
        return None, None
    sxx = float(((t - t.mean()) ** 2).sum())
    if sxx <= 0.0:
        return None, None
    slope = float(((t - t.mean()) * (p - p.mean())).sum() / sxx)
    return slope, float(p.mean() - slope * t.mean())


def axis_report(
    axis: str,
    target_norm: np.ndarray,
    pred_norm: np.ndarray,
    target_id: np.ndarray,
    size_px: int,
) -> AxisReport:
    """Slope, bias, median |error|, P90 and coverage for one axis.

    Every row passed in is ELIGIBLE: it should have produced a measurement.
    Rows whose prediction is not finite are counted as lost (they lower
    ``coverage``) and are excluded from the error statistics -- never dropped
    from the denominator.
    """

    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    col = 0 if axis == "x" else 1
    t = np.asarray(target_norm, dtype=np.float64)
    p = np.asarray(pred_norm, dtype=np.float64)
    tid = np.asarray(target_id)
    if t.ndim != 2 or p.ndim != 2 or t.shape[1] < 2 or p.shape[1] < 2:
        raise ValueError("target_norm and pred_norm must be (n, 2) arrays")
    if not (t.shape[0] == p.shape[0] == tid.shape[0]):
        raise ValueError("target_norm, pred_norm and target_id must have the same length")
    if size_px <= 1:
        raise ValueError("size_px must be greater than 1")

    n_eligible = int(t.shape[0])
    if n_eligible == 0:
        return AxisReport(axis, 0, 0, None, 0, None, None, None, None, None, "no_eligible_rows")

    valid = np.isfinite(p[:, col]) & np.isfinite(t[:, col])
    n_valid = int(valid.sum())
    coverage = n_valid / n_eligible
    if n_valid == 0:
        return AxisReport(
            axis, n_eligible, 0, coverage, 0, None, None, None, None, None, "no_valid_predictions"
        )

    d = (p[valid, col] - t[valid, col]) * (size_px - 1)
    positions: list[float] = []
    medians: list[float] = []
    for value in sorted({int(v) for v in tid[valid]}):
        rows = valid & (tid == value)
        positions.append(float(t[rows, col][0]))
        medians.append(float(np.median(p[rows, col])))
    n_positions = len({round(v, 9) for v in positions})
    slope, intercept = fit_slope_bias(positions, medians)
    return AxisReport(
        axis=axis,
        n_eligible=n_eligible,
        n_valid=n_valid,
        coverage=coverage,
        n_target_positions=n_positions,
        slope=slope,
        intercept=intercept,
        bias_px=float(np.median(d)),
        median_abs_px=float(np.median(np.abs(d))),
        p90_abs_px=float(np.percentile(np.abs(d), 90)),
        degenerate=None if slope is not None else "single_target_position",
    )


def score_axes(rec: S.Recording, pred_norm: np.ndarray) -> dict[str, AxisReport]:
    """Both axes of one protocol, over its eligible rows."""

    eligible = F.eligible_rows(rec)
    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    return {
        "x": axis_report(
            "x", rec.target_xy[eligible], pred_norm[eligible], rec.target_id[eligible], width
        ),
        "y": axis_report(
            "y", rec.target_xy[eligible], pred_norm[eligible], rec.target_id[eligible], height
        ),
    }


def r_squared(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    """Coefficient of determination, or None when it is not defined."""

    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(p)
    if ok.sum() < 2:
        return None
    t, p = t[ok], p[ok]
    denom = float(((t - t.mean()) ** 2).sum())
    if denom <= 0.0:
        return None
    return float(1.0 - ((t - p) ** 2).sum() / denom)


# --- Design matrices ----------------------------------------------------------


def extra_matrix(rec: S.Recording, mask: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """The requested head6 / pnp columns for the masked rows, in ``names`` order."""

    n = int(np.asarray(mask, dtype=bool).sum())
    if not names:
        return np.empty((n, 0), dtype=np.float64)
    head6 = {name: i for i, name in enumerate(H.HEAD6_NAMES)}
    cols: list[np.ndarray] = []
    for name in names:
        if name in head6:
            cols.append(np.asarray(rec.head, dtype=np.float64)[mask, head6[name]])
        elif name in PNP_INDEX:
            cols.append(np.asarray(rec.pnp_deg, dtype=np.float64)[mask, PNP_INDEX[name]])
        else:
            raise ValueError(f"unknown extra column {name!r}; known: {EXTRA_COLUMN_NAMES}")
    return np.stack(cols, axis=1)


def usable_rows(rec: S.Recording, base_mask: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """``base_mask`` narrowed to rows whose requested extra columns are finite.

    A head-using candidate can only be trained or scored where its head
    columns exist; the rows it loses are reported as reduced coverage rather
    than quietly leaving the denominator.
    """

    mask = np.asarray(base_mask, dtype=bool).copy()
    if not names:
        return mask
    head6 = set(H.HEAD6_NAMES)
    if any(name in head6 for name in names):
        mask &= rec.rows_head_valid()
    if any(name in PNP_INDEX for name in names):
        mask &= np.all(np.isfinite(np.asarray(rec.pnp_deg, dtype=np.float64)), axis=1)
    return mask


def design(rec: S.Recording, mask: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """The 258-column embedding plus the requested extra columns."""

    base = np.asarray(rec.features, dtype=np.float64)[mask]
    if not names:
        return base
    return np.concatenate([base, extra_matrix(rec, mask, names)], axis=1)


# --- Candidates ---------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One calibration-layer setting.

    ``extra_names`` may mix head6 and pnp columns.  Because
    :class:`gf_fit.FitConfig` only knows the head6 names, the extra columns are
    concatenated here and the fit is handed a ``FitConfig`` with no
    ``head_names``: the model is then a plain (258 + k)-column model, and its
    schema records the width, so a mismatched input is still refused.  Nothing
    is written into ``recordings/models``.
    """

    name: str
    feature_scaling: str = "none"
    label_scaling: str = "none"
    C: float = 1.0
    gamma: float | str = 0.005
    P: float = 0.001
    kind: str = "svr"
    alpha: float = 0.0
    extra_names: tuple[str, ...] = ()

    def fit_config(self) -> F.FitConfig:
        return F.FitConfig(
            name=self.name,
            feature_scaling=self.feature_scaling,
            label_scaling=self.label_scaling,
            C=self.C,
            gamma=self.gamma,
            P=self.P,
            kind=self.kind,
            alpha=self.alpha,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["extra_names"] = list(self.extra_names)
        return value


def candidate_name(
    fs: str, ls: str, c: float, gamma: float | str, p: float, extras: Sequence[str]
) -> str:
    tag = "base" if not extras else "+".join(extras)
    return f"{tag}|f={fs}|l={ls}|C={c:g}|g={gamma}|P={p:g}"


#: Extra-column sets swept.  The hypothesis under test is that vertical gaze
#: is confounded with head pitch and eyelid position, so handing the model the
#: pitch / eye-height / distance columns should let it separate them.
EXTRA_SETS: tuple[tuple[str, ...], ...] = (
    (),
    ("pitch_a",),
    ("eye_mid_y",),
    ("iod_norm",),
    ("pnp_pitch",),
    ("pitch_a", "eye_mid_y"),
    ("pitch_a", "eye_mid_y", "iod_norm"),
    ("pitch_a", "eye_mid_y", "iod_norm", "pnp_pitch"),
    tuple(H.HEAD6_NAMES),
)

SWEEP_C: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
SWEEP_GAMMA: tuple[float | str, ...] = (0.005, 0.0005, 5e-05, "auto")
SWEEP_P: tuple[float, ...] = (0.001, 0.01, 0.1)
SWEEP_RIDGE_ALPHA: tuple[float, ...] = (0.01, 1.0, 100.0, 10000.0)


def build_grid(quick: bool = False) -> list[Candidate]:
    """The candidate grid: scaling x C x gamma x epsilon x extra columns, plus ridge."""

    extras = EXTRA_SETS[:3] if quick else EXTRA_SETS
    cs = SWEEP_C[:2] if quick else SWEEP_C
    gammas = SWEEP_GAMMA[:2] if quick else SWEEP_GAMMA
    ps = SWEEP_P[:1] if quick else SWEEP_P
    alphas = SWEEP_RIDGE_ALPHA[:1] if quick else SWEEP_RIDGE_ALPHA
    grid: list[Candidate] = []
    for extra in extras:
        for fs in ("none", "zscore"):
            for ls in ("none", "zscore"):
                for c in cs:
                    for gamma in gammas:
                        for p in ps:
                            grid.append(
                                Candidate(
                                    name=candidate_name(fs, ls, c, gamma, p, extra),
                                    feature_scaling=fs,
                                    label_scaling=ls,
                                    C=c,
                                    gamma=gamma,
                                    P=p,
                                    extra_names=extra,
                                )
                            )
                for alpha in alphas:
                    tag = "base" if not extra else "+".join(extra)
                    grid.append(
                        Candidate(
                            name=f"{tag}|ridge|f={fs}|l={ls}|a={alpha:g}",
                            feature_scaling=fs,
                            label_scaling=ls,
                            kind="ridge",
                            alpha=alpha,
                            extra_names=extra,
                        )
                    )
    return grid


@dataclass
class Fitted:
    """A fitted candidate and the extra columns its inputs must carry."""

    candidate: Candidate
    model: F.FittedModel

    def predict_norm(self, rec: S.Recording, rig: C.RigGeometry) -> np.ndarray:
        """(n_rows, 2) normalised predictions; NaN where the row is unusable."""

        mask = usable_rows(rec, rec.rows_collecting(), self.candidate.extra_names)
        out = np.full((rec.n_rows, 2), np.nan)
        if np.any(mask):
            out[mask] = self.model.predict_norm(design(rec, mask, self.candidate.extra_names), rig)
        return out


def fit_candidate(rec_a: S.Recording, candidate: Candidate, rig: C.RigGeometry) -> Fitted:
    """Fit one candidate on the calibration protocol's accepted rows."""

    mask = usable_rows(rec_a, rec_a.rows_accepted(), candidate.extra_names)
    X = design(rec_a, mask, candidate.extra_names)
    Y = np.asarray(rec_a.label_cm, dtype=np.float64)[mask]
    meta = {
        "protocol": rec_a.protocol,
        "round_id": rec_a.round_id,
        "session": rec_a.meta.get("session"),
        "extra_names": list(candidate.extra_names),
        "tool": "gf_vertical",
    }
    model = F.FittedModel.fit(candidate.fit_config(), X, Y, rig=rig, train_meta=meta)
    return Fitted(candidate=candidate, model=model)


# --- Pre-declared selection rule ---------------------------------------------


@dataclass(frozen=True)
class SelectionRule:
    """How a candidate is chosen, declared before any test protocol is read.

    The question is whether the VERTICAL axis can be improved, so the
    objective is the vertical error -- but only among candidates that do not
    buy it by giving up the horizontal axis or by flattening the slope
    further.  Both guards are relative to the baseline measured on the SAME
    comparison protocol, so neither depends on an absolute threshold.
    """

    min_coverage: float = 0.95
    max_x_median_ratio: float = 1.25  # x median |error| vs the baseline's
    require_slope_no_worse: bool = True  # |1 - slope_y| <= the baseline's
    objective: str = "y_median_abs_px"
    tiebreak: str = "abs_slope_error_y"


RULE = SelectionRule()


def _slope_error(report: Mapping[str, Any]) -> float:
    slope = report["y"]["slope"]
    return float("inf") if slope is None else abs(1.0 - float(slope))


def _metric(report: Mapping[str, Any], axis: str, key: str) -> float:
    value = report[axis][key]
    return float("inf") if value is None else float(value)


def eligible(
    entry: Mapping[str, Any], baseline: Mapping[str, Any], rule: SelectionRule = RULE
) -> tuple[bool, str]:
    """Does this candidate pass the rule's guards?  Returns (ok, reason)."""

    report = entry["tune"]
    for axis in ("x", "y"):
        cov = report[axis]["coverage"]
        if cov is None or cov < rule.min_coverage:
            return False, f"coverage_{axis}"
    limit = rule.max_x_median_ratio * _metric(baseline, "x", "median_abs_px")
    if _metric(report, "x", "median_abs_px") > limit:
        return False, "x_regression"
    if rule.require_slope_no_worse and _slope_error(report) > _slope_error(baseline):
        return False, "y_slope_regression"
    return True, "ok"


def select(
    sweep: Sequence[Mapping[str, Any]], baseline_name: str = BASELINE, rule: SelectionRule = RULE
) -> dict[str, Any]:
    """Choose one candidate from TUNE metrics alone.

    ``sweep`` entries carry a ``tune`` report and nothing from any test
    protocol, so this function cannot see the test set even by accident.
    """

    entries = list(sweep)
    if not entries:
        raise ValueError("empty sweep: nothing to select from")
    try:
        baseline = next(e for e in entries if e["name"] == baseline_name)["tune"]
    except StopIteration:  # pragma: no cover - guarded by the caller's grid
        raise ValueError(f"baseline {baseline_name!r} is not in the sweep") from None

    survivors: list[Mapping[str, Any]] = []
    rejected: dict[str, int] = {}
    for entry in entries:
        ok, reason = eligible(entry, baseline, rule)
        if ok:
            survivors.append(entry)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    if not survivors:
        return {
            "selected": None,
            "refused": True,
            "reason": "no candidate passed the guards",
            "rejected": rejected,
            "n_candidates": len(entries),
            "rule": asdict(rule),
        }
    best = min(
        survivors, key=lambda e: (_metric(e["tune"], "y", "median_abs_px"), _slope_error(e["tune"]))
    )
    return {
        "selected": best["name"],
        "refused": False,
        "reason": None,
        "rejected": rejected,
        "n_candidates": len(entries),
        "n_survivors": len(survivors),
        "rule": asdict(rule),
    }


def select_per_axis(
    sweep: Sequence[Mapping[str, Any]], rule: SelectionRule = RULE
) -> dict[str, Any]:
    """Choose an X candidate and a Y candidate independently, on TUNE only.

    The library already fits the two axes with two separate SVRs, so nothing
    stops them using different hyper-parameters.  This arm tests whether the
    X/Y trade-off the single-config sweep runs into is forced or merely an
    artefact of making one setting serve both axes.
    """

    entries = [
        e
        for e in sweep
        if all(
            e["tune"][a]["coverage"] not in (None,)
            and e["tune"][a]["coverage"] >= rule.min_coverage
            for a in ("x", "y")
        )
    ]
    if not entries:
        return {
            "x": None,
            "y": None,
            "refused": True,
            "reason": "no candidate met the coverage floor",
        }
    x_best = min(entries, key=lambda e: _metric(e["tune"], "x", "median_abs_px"))
    y_best = min(
        entries, key=lambda e: (_slope_error(e["tune"]), _metric(e["tune"], "y", "median_abs_px"))
    )
    return {"x": x_best["name"], "y": y_best["name"], "refused": False, "reason": None}


def compose(pred_x: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
    """X column from one model's prediction, Y column from another's."""

    a = np.asarray(pred_x, dtype=np.float64)
    b = np.asarray(pred_y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("both predictions must be (n, 2) arrays of the same shape")
    return np.stack([a[:, 0], b[:, 1]], axis=1)


# --- Is the vertical signal there at all? -------------------------------------
#
# These regressors deliberately do NOT go through the SVR: the point is to ask
# what the features support, not what one fitter extracts.  All are closed
# form and depend on no package beyond numpy, so the answer is reproducible.


def _standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (train - mean) / std, (other - mean) / std


def ridge_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, alpha: float) -> np.ndarray:
    Z_tr, Z_te = _standardise(X_tr, X_te)
    mean = float(y_tr.mean())
    d = Z_tr.shape[1]
    w = np.linalg.solve(Z_tr.T @ Z_tr + alpha * np.eye(d), Z_tr.T @ (y_tr - mean))
    return Z_te @ w + mean


def kernel_ridge_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, alpha: float, gamma_scale: float
) -> np.ndarray:
    """RBF kernel ridge; ``gamma = gamma_scale / n_columns`` after standardising."""

    Z_tr, Z_te = _standardise(X_tr, X_te)
    gamma = gamma_scale / Z_tr.shape[1]

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))

    mean = float(y_tr.mean())
    n = Z_tr.shape[0]
    weights = np.linalg.solve(kernel(Z_tr, Z_tr) + alpha * np.eye(n), y_tr - mean)
    return kernel(Z_te, Z_tr) @ weights + mean


def knn_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, k: int) -> np.ndarray:
    Z_tr, Z_te = _standardise(X_tr, X_te)
    k = min(k, Z_tr.shape[0])
    out = np.empty(Z_te.shape[0])
    for i, row in enumerate(Z_te):
        dist = ((Z_tr - row) ** 2).sum(axis=1)
        out[i] = float(y_tr[np.argpartition(dist, k - 1)[:k]].mean())
    return out


Regressor = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

REGRESSORS: dict[str, Regressor] = {
    "ridge_a10": lambda a, b, c: ridge_predict(a, b, c, 10.0),
    "ridge_a1000": lambda a, b, c: ridge_predict(a, b, c, 1000.0),
    "kernel_ridge_g1": lambda a, b, c: kernel_ridge_predict(a, b, c, 1.0, 1.0),
    "kernel_ridge_g0.1": lambda a, b, c: kernel_ridge_predict(a, b, c, 1.0, 0.1),
    "knn_k15": lambda a, b, c: knn_predict(a, b, c, 15),
}


def grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: Sequence[Sequence[int]],
    predict: Regressor,
) -> np.ndarray:
    """Out-of-fold predictions; rows in no fold stay NaN.

    ``groups`` labels each row with the calibration target it came from, and a
    fold is a set of those labels.  Splitting by target matters: consecutive
    rows of one target are frames of one ~1.5 s fixation, so a random row
    split would put near-duplicate rows on both sides and report memorisation
    as generalisation.
    """

    out = np.full(len(y), np.nan)
    for fold in folds:
        test = np.isin(groups, list(fold))
        train = ~test
        if not test.any() or train.sum() < 2:
            continue
        if len(np.unique(y[train])) < 2:
            continue  # the label is constant in the training half: nothing to learn
        out[test] = predict(X[train], y[train], X[test])
    return out


def target_folds(rec: S.Recording, mask: np.ndarray, axis: str | None = None) -> list[list[int]]:
    """Cross-validation folds over calibration targets.

    ``axis=None``  leave one TARGET out.
    ``axis='x'``   leave one whole COLUMN of the grid out.
    ``axis='y'``   leave one whole ROW of the grid out.

    The row/column folds exist because a 3x3 calibration grid has only three
    distinct positions per axis: leaving one target out still leaves two other
    targets at the same height, so a leave-one-target-out score for Y is
    partly interpolation between rows that were seen.  Holding out the whole
    row removes that.
    """

    tid = np.asarray(rec.target_id)[mask].astype(int)
    pos = np.asarray(rec.target_xy, dtype=np.float64)[mask]
    ids = sorted({int(v) for v in tid})
    if axis is None:
        return [[i] for i in ids]
    col = 0 if axis == "x" else 1
    levels: dict[float, list[int]] = {}
    for i in ids:
        value = round(float(pos[tid == i][0, col]), 6)
        levels.setdefault(value, []).append(i)
    return [sorted(v) for _, v in sorted(levels.items())]


def signal_probe(rec_a: S.Recording, rec_tune: S.Recording) -> dict[str, Any]:
    """How much of each axis is recoverable from the features, without an SVR.

    X is the control: whatever the numbers are, the question is whether Y is
    materially worse than X on the same rows with the same regressor.
    """

    mask = rec_a.rows_accepted()
    X = design(rec_a, mask, ())
    T = np.asarray(rec_a.target_xy, dtype=np.float64)[mask]
    tid = np.asarray(rec_a.target_id)[mask].astype(int)
    width = int(rec_a.meta["target_geometry"]["width_px"])
    height = int(rec_a.meta["target_geometry"]["height_px"])

    schemes = {
        "leave_one_target_out": target_folds(rec_a, mask, None),
        "leave_one_row_out": target_folds(rec_a, mask, "y"),
        "leave_one_column_out": target_folds(rec_a, mask, "x"),
    }
    within: dict[str, Any] = {}
    for scheme, folds in schemes.items():
        block: dict[str, Any] = {"n_folds": len(folds)}
        for axis, col, size in (("x", 0, width), ("y", 1, height)):
            per_axis: dict[str, Any] = {}
            for name, fn in REGRESSORS.items():
                pred = grouped_cv(X, T[:, col], tid, folds, fn)
                ok = np.isfinite(pred)
                # axis_report takes an (n, 2) prediction; only ``col`` is filled,
                # so the other axis stays NaN and is never scored here.
                two = np.full((len(pred), 2), np.nan)
                two[:, col] = pred
                report = axis_report(axis, T, two, tid, size)
                per_axis[name] = {
                    "r2": r_squared(T[ok, col], pred[ok]),
                    "slope": report.slope,
                    "median_abs_px": report.median_abs_px,
                    "coverage": report.coverage,
                }
            block[axis] = per_axis
        within[scheme] = block

    # The honest transfer test: fit on A, predict TUNE's different points.
    tune_mask = rec_tune.rows_collecting()
    X_tune = design(rec_tune, tune_mask, ())
    T_tune = np.asarray(rec_tune.target_xy, dtype=np.float64)[tune_mask]
    tid_tune = np.asarray(rec_tune.target_id)[tune_mask].astype(int)
    across: dict[str, Any] = {}
    for axis, col, size in (("x", 0, width), ("y", 1, height)):
        per_axis = {}
        for name, fn in REGRESSORS.items():
            pred = fn(X, T[:, col], X_tune)
            two = np.full((len(pred), 2), np.nan)
            two[:, col] = pred
            report = axis_report(axis, T_tune, two, tid_tune, size)
            per_axis[name] = {
                "r2": r_squared(T_tune[:, col], pred),
                "slope": report.slope,
                "median_abs_px": report.median_abs_px,
            }
        across[axis] = per_axis
    return {
        "within_calibration": within,
        "across_protocol_A_to_TUNE": across,
        "note": (
            "kNN can look perfect on a 3x3 grid because a held-out target shares its "
            "row or column position with targets that were seen; read it with the "
            "leave_one_row_out / leave_one_column_out schemes, not alone."
        ),
    }


# --- Descriptive geometry -----------------------------------------------------


def geometry_probe(rec_a: S.Recording) -> dict[str, Any]:
    """Descriptive statistics of the feature space the kernel actually sees.

    Purely descriptive: it says how the embedding is shaped, not why any
    configuration wins.
    """

    mask = rec_a.rows_accepted()
    X = design(rec_a, mask, ())
    T = np.asarray(rec_a.target_xy, dtype=np.float64)[mask]
    std = X.std(axis=0)
    scaler = S.Standardizer.fit(X)
    Z = scaler.transform(X)

    def correlations(target: np.ndarray) -> np.ndarray:
        centred = X - X.mean(axis=0)
        t = target - target.mean()
        ok = std > 1e-12
        out = np.zeros(X.shape[1])
        if t.std() > 0:
            out[ok] = (centred[:, ok] * t[:, None]).mean(axis=0) / (std[ok] * t.std())
        return out

    top = {
        "x": np.argsort(-np.abs(correlations(T[:, 0])))[:20],
        "y": np.argsort(-np.abs(correlations(T[:, 1])))[:20],
    }
    var_raw, var_scaled = X.var(axis=0), Z.var(axis=0)

    def separability(matrix: np.ndarray, level: np.ndarray) -> float:
        grand = matrix.mean(axis=0)
        between = within = 0.0
        for value in np.unique(level):
            rows = level == value
            between += int(rows.sum()) * float(((matrix[rows].mean(axis=0) - grand) ** 2).sum())
            within += float(((matrix[rows] - matrix[rows].mean(axis=0)) ** 2).sum())
        return between / within if within > 0 else float("inf")

    lx, ly = np.round(T[:, 0], 6), np.round(T[:, 1], 6)
    return {
        "n_columns": int(X.shape[1]),
        "n_constant_columns": int((std <= 1e-12).sum()),
        "column_std": {
            "min": float(std.min()),
            "median": float(np.median(std)),
            "max": float(std.max()),
        },
        "variance_floor": {
            "fraction": S.VARIANCE_FLOOR_FRACTION,
            "value": float(scaler.std.min()),
            "n_columns_clamped": int(np.isclose(scaler.std, scaler.std.min()).sum()),
            "note": (
                "gf_schema.Standardizer floors every column at 1% of the widest, "
                "so 'zscore' is not a plain z-score"
            ),
        },
        "variance_share_top20": {
            "x_raw": float(var_raw[top["x"]].sum() / var_raw.sum()),
            "x_scaled": float(var_scaled[top["x"]].sum() / var_scaled.sum()),
            "y_raw": float(var_raw[top["y"]].sum() / var_raw.sum()),
            "y_scaled": float(var_scaled[top["y"]].sum() / var_scaled.sum()),
        },
        "between_within_ratio": {
            "by_target_x_level_raw": separability(X, lx),
            "by_target_x_level_scaled": separability(Z, lx),
            "by_target_y_level_raw": separability(X, ly),
            "by_target_y_level_scaled": separability(Z, ly),
        },
        "max_abs_column_correlation": {
            "with_target_x": float(np.abs(correlations(T[:, 0])).max()),
            "with_target_y": float(np.abs(correlations(T[:, 1])).max()),
        },
    }


def head_drift(recordings: Mapping[str, S.Recording]) -> dict[str, Any]:
    """Head pose per protocol: does the pose at test time match calibration?

    A model that leans on head pitch to place the vertical coordinate will
    show a vertical bias when the pose drifts between calibration and use, so
    the drift is worth measuring next to the bias.
    """

    out: dict[str, Any] = {}
    for label, rec in recordings.items():
        mask = rec.rows_collecting()
        head = np.asarray(rec.head, dtype=np.float64)[mask]
        pnp = np.asarray(rec.pnp_deg, dtype=np.float64)[mask]
        entry: dict[str, Any] = {"n_rows": int(mask.sum())}
        for name in ("pitch_a", "eye_mid_y", "iod_norm"):
            col = head[:, H.HEAD6_NAMES.index(name)]
            entry[name] = {"mean": float(np.nanmean(col)), "sd": float(np.nanstd(col))}
        entry["pnp_pitch_deg"] = {
            "mean": float(np.nanmean(pnp[:, 1])),
            "sd": float(np.nanstd(pnp[:, 1])),
        }
        out[label] = entry
    return out


def pitch_target_correlation(rec_a: S.Recording) -> dict[str, Any]:
    """How strongly head pitch tracked the target during calibration.

    Descriptive.  A high value means the two cannot be separated FROM THIS
    RECORDING, not that one causes the other.
    """

    mask = rec_a.rows_accepted()
    head = np.asarray(rec_a.head, dtype=np.float64)[mask]
    pnp = np.asarray(rec_a.pnp_deg, dtype=np.float64)[mask]
    target = np.asarray(rec_a.target_xy, dtype=np.float64)[mask]
    return {
        "n_rows": int(mask.sum()),
        "corr_pitch_a_target_y": F.pearson(head[:, H.HEAD6_NAMES.index("pitch_a")], target[:, 1]),
        "corr_eye_mid_y_target_y": F.pearson(
            head[:, H.HEAD6_NAMES.index("eye_mid_y")], target[:, 1]
        ),
        "corr_pnp_pitch_target_y": F.pearson(pnp[:, 1], target[:, 1]),
        "corr_yaw_ratio_target_x": F.pearson(
            head[:, H.HEAD6_NAMES.index("yaw_ratio")], target[:, 0]
        ),
        "note": "descriptive only; a correlation here is a confound, not a cause",
    }


# --- Orchestration ------------------------------------------------------------


@dataclass
class Inputs:
    recording_dir: Path
    rec_a: S.Recording
    rec_tune: S.Recording
    rec_dev: S.Recording  # the development test protocol (already inspected)
    rig: C.RigGeometry
    confirm: S.Recording | None = None
    confirm_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def load_inputs(recording_dir: Path, confirm_dir: Path | None = None) -> Inputs:
    recording_dir = Path(recording_dir)
    rec_a = S.Recording.load(recording_dir, "A")
    rec_tune = S.Recording.load(recording_dir, "TUNE")
    rec_dev = S.Recording.load(recording_dir, "T1")
    rig = C.RigGeometry.from_dict(rec_a.meta["rig"])
    warnings: list[str] = []
    confirm = None
    if confirm_dir is not None:
        confirm_dir = Path(confirm_dir)
        confirm = S.Recording.load(confirm_dir, "T1")
        if confirm.meta.get("session") != rec_a.meta.get("session"):
            warnings.append(
                f"confirmation recording {confirm_dir.name} is session "
                f"{confirm.meta.get('session')!r}, calibration is {rec_a.meta.get('session')!r}: "
                "a model fitted on this calibration is being used across sessions"
            )
    return Inputs(recording_dir, rec_a, rec_tune, rec_dev, rig, confirm, confirm_dir, warnings)


def run_sweep(
    inputs: Inputs, grid: Sequence[Candidate]
) -> tuple[list[dict[str, Any]], dict[str, Fitted]]:
    """Fit every candidate on A and score it on TUNE.  No test protocol here."""

    sweep: list[dict[str, Any]] = []
    fitted: dict[str, Fitted] = {}
    for candidate in grid:
        try:
            model = fit_candidate(inputs.rec_a, candidate, inputs.rig)
        except Exception as exc:  # noqa: BLE001 - a refused candidate is data, not a crash
            sweep.append(
                {
                    "name": candidate.name,
                    "candidate": candidate.to_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        reports = score_axes(inputs.rec_tune, model.predict_norm(inputs.rec_tune, inputs.rig))
        sweep.append(
            {
                "name": candidate.name,
                "candidate": candidate.to_dict(),
                "tune": {axis: report.to_dict() for axis, report in reports.items()},
            }
        )
        fitted[candidate.name] = model
    return [e for e in sweep if "tune" in e], fitted


def score_once(fitted: Fitted, rec: S.Recording, rig: C.RigGeometry) -> dict[str, Any]:
    return {
        axis: report.to_dict()
        for axis, report in score_axes(rec, fitted.predict_norm(rec, rig)).items()
    }


def score_hybrid(
    x_fit: Fitted, y_fit: Fitted, rec: S.Recording, rig: C.RigGeometry
) -> dict[str, Any]:
    pred = compose(x_fit.predict_norm(rec, rig), y_fit.predict_norm(rec, rig))
    return {axis: report.to_dict() for axis, report in score_axes(rec, pred).items()}


def analyse(inputs: Inputs, *, quick: bool = False) -> dict[str, Any]:
    """The whole investigation, in the order the method requires."""

    grid = build_grid(quick=quick)
    if not any(c.name == BASELINE for c in grid):
        grid = [
            Candidate(BASELINE, "zscore", "none", 100.0, 0.0005, 0.001),
            *grid,
        ]
    sweep, fitted = run_sweep(inputs, grid)

    # --- selection: TUNE only ------------------------------------------------
    single = select(sweep, BASELINE, RULE)
    hybrid = select_per_axis(sweep, RULE)
    by_name = {e["name"]: e for e in sweep}

    chosen: dict[str, Any] = {"arm": None, "single": single, "per_axis": hybrid}
    if not single["refused"] and not hybrid["refused"]:
        single_y = _metric(by_name[single["selected"]]["tune"], "y", "median_abs_px")
        hybrid_pred = compose(
            fitted[hybrid["x"]].predict_norm(inputs.rec_tune, inputs.rig),
            fitted[hybrid["y"]].predict_norm(inputs.rec_tune, inputs.rig),
        )
        hybrid_tune = {a: r.to_dict() for a, r in score_axes(inputs.rec_tune, hybrid_pred).items()}
        chosen["per_axis_tune"] = hybrid_tune
        hybrid_ok, hybrid_reason = eligible({"tune": hybrid_tune}, by_name[BASELINE]["tune"], RULE)
        chosen["per_axis_passes_guards"] = hybrid_ok
        chosen["per_axis_guard_reason"] = hybrid_reason
        hybrid_y = _metric(hybrid_tune, "y", "median_abs_px")
        chosen["arm"] = "per_axis" if (hybrid_ok and hybrid_y < single_y) else "single"
    elif not single["refused"]:
        chosen["arm"] = "single"

    # --- held-out scoring: once, after the choice is fixed -------------------
    held_out: dict[str, Any] = {}
    protocols: list[tuple[str, S.Recording]] = [
        (f"{inputs.recording_dir.name}/T1 (development)", inputs.rec_dev)
    ]
    if inputs.confirm is not None and inputs.confirm_dir is not None:
        protocols.append((f"{inputs.confirm_dir.name}/T1 (confirmation)", inputs.confirm))
    for label, rec in protocols:
        entry: dict[str, Any] = {}
        if BASELINE in fitted:
            entry["baseline"] = score_once(fitted[BASELINE], rec, inputs.rig)
        if chosen["arm"] == "single" and not single["refused"]:
            entry["chosen_single"] = score_once(fitted[single["selected"]], rec, inputs.rig)
        if not hybrid["refused"]:
            entry["per_axis"] = score_hybrid(
                fitted[hybrid["x"]], fitted[hybrid["y"]], rec, inputs.rig
            )
        if not single["refused"] and chosen["arm"] != "single":
            entry["best_single"] = score_once(fitted[single["selected"]], rec, inputs.rig)
        held_out[label] = entry

    recordings = {
        "A": inputs.rec_a,
        "TUNE": inputs.rec_tune,
        f"{inputs.recording_dir.name}/T1": inputs.rec_dev,
    }
    if inputs.confirm is not None and inputs.confirm_dir is not None:
        recordings[f"{inputs.confirm_dir.name}/T1"] = inputs.confirm

    return {
        "tool": "gf_vertical",
        "task": "M2-05",
        "recording_dir": str(inputs.recording_dir),
        "confirm_dir": str(inputs.confirm_dir) if inputs.confirm_dir else None,
        "warnings": inputs.warnings,
        "baseline": BASELINE,
        "rule": asdict(RULE),
        "n_candidates": len(sweep),
        "sweep": sweep,
        "selection": chosen,
        "held_out": held_out,
        "signal": signal_probe(inputs.rec_a, inputs.rec_tune),
        "geometry": geometry_probe(inputs.rec_a),
        "head_drift": head_drift(recordings),
        "pitch_confound": pitch_target_correlation(inputs.rec_a),
    }


# --- Reporting ----------------------------------------------------------------


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _regressor_line(name: str, x: Mapping[str, Any], y: Mapping[str, Any]) -> str:
    """One row of the signal-probe table: the same statistic for X and for Y."""

    def half(side: Mapping[str, Any]) -> str:
        return (
            f"R2 {_num(side['r2'], 3):>7s} slope {_num(side['slope'], 3):>6s} "
            f"med {_num(side['median_abs_px']):>7s}"
        )

    return f"    {name:<18s} X {half(x)}  |  Y {half(y)}"


def _axis_line(label: str, report: Mapping[str, Any]) -> str:
    x, y = report["x"], report["y"]
    return (
        f"  {label:<34s} "
        f"X slope {_num(x['slope'], 3):>6s} med {_num(x['median_abs_px']):>6s} "
        f"P90 {_num(x['p90_abs_px']):>6s} bias {_num(x['bias_px']):>7s} | "
        f"Y slope {_num(y['slope'], 3):>6s} med {_num(y['median_abs_px']):>6s} "
        f"P90 {_num(y['p90_abs_px']):>6s} bias {_num(y['bias_px']):>7s} | "
        f"cov {_num(y['coverage'], 3)}"
    )


def format_report(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("M2-05 vertical-axis investigation (gf_vertical)")
    add(f"calibration/tuning: {result['recording_dir']}  (fit on A, compared on TUNE)")
    add(f"baseline: {result['baseline']}")
    add(f"candidates scored on TUNE: {result['n_candidates']}")
    for warning in result.get("warnings", []):
        add(f"WARNING: {warning}")
    add("")
    add("Errors are logical screen px. Slope is dimensionless (1.0 = perfect).")
    add("")

    sweep = {e["name"]: e for e in result["sweep"]}
    add("TUNE (round18/TUNE, 8 points) -- selection evidence, never a test set")
    if result["baseline"] in sweep:
        add(_axis_line("baseline", sweep[result["baseline"]]["tune"]))
    sel = result["selection"]
    if sel["single"].get("selected"):
        add(
            _axis_line(
                f"best single: {sel['single']['selected'][:28]}",
                sweep[sel["single"]["selected"]]["tune"],
            )
        )
    if sel.get("per_axis_tune"):
        add(_axis_line("per-axis hybrid", sel["per_axis_tune"]))
        add(f"    hybrid X from: {sel['per_axis']['x']}")
        add(f"    hybrid Y from: {sel['per_axis']['y']}")
    add(
        f"    chosen arm: {sel['arm']}  "
        f"(rule: {result['rule']['objective']}, guards on X and slope)"
    )
    add(f"    rejected by guard: {sel['single'].get('rejected')}")
    add("")

    for label, entry in result["held_out"].items():
        add(f"{label} -- scored once, after the choice was fixed")
        for name, report in entry.items():
            add(_axis_line(name, report))
        add("")

    add("Is the vertical signal present at all? (no SVR; grouped CV on A's targets)")
    signal = result["signal"]
    for scheme, block in signal["within_calibration"].items():
        add(f"  {scheme} ({block['n_folds']} folds)")
        for regressor in REGRESSORS:
            add(_regressor_line(regressor, block["x"][regressor], block["y"][regressor]))
    add("  A -> TUNE transfer (fit on A's 9 points, predict TUNE's 8)")
    for regressor in REGRESSORS:
        across = signal["across_protocol_A_to_TUNE"]
        add(_regressor_line(regressor, across["x"][regressor], across["y"][regressor]))
    add(f"  note: {signal['note']}")
    add("")

    geo = result["geometry"]
    add("Feature-space geometry (descriptive)")
    add(
        f"  {geo['n_columns']} columns, {geo['n_constant_columns']} constant; std min "
        f"{geo['column_std']['min']:.3g} median {geo['column_std']['median']:.3g} "
        f"max {geo['column_std']['max']:.3g}"
    )
    floor = geo["variance_floor"]
    add(
        f"  variance floor {floor['value']:.4g} clamps {floor['n_columns_clamped']} "
        f"of {geo['n_columns']} columns ({floor['note']})"
    )
    share = geo["variance_share_top20"]
    add(
        f"  variance share of the 20 columns most correlated with the target: "
        f"X raw {share['x_raw']:.3f} -> scaled {share['x_scaled']:.3f}; "
        f"Y raw {share['y_raw']:.3f} -> scaled {share['y_scaled']:.3f}"
    )
    bw = geo["between_within_ratio"]
    add(
        f"  between/within separability by target level: "
        f"X raw {bw['by_target_x_level_raw']:.2f} -> scaled {bw['by_target_x_level_scaled']:.2f}; "
        f"Y raw {bw['by_target_y_level_raw']:.2f} -> scaled {bw['by_target_y_level_scaled']:.2f}"
    )
    corr = geo["max_abs_column_correlation"]
    add(
        f"  max |corr| of a single column with the target: "
        f"X {corr['with_target_x']:.3f}, Y {corr['with_target_y']:.3f}"
    )
    add("")

    add("Head pose per protocol (drift between calibration and use)")
    for label, entry in result["head_drift"].items():
        add(
            f"  {label:<24s} pitch_a {entry['pitch_a']['mean']:+.4f}"
            f"+-{entry['pitch_a']['sd']:.4f}  "
            f"eye_mid_y {entry['eye_mid_y']['mean']:.4f}  iod {entry['iod_norm']['mean']:.4f}  "
            f"pnp_pitch {entry['pnp_pitch_deg']['mean']:+.2f}deg"
            f"+-{entry['pnp_pitch_deg']['sd']:.2f}"
        )
    conf = result["pitch_confound"]
    add(
        f"  on A: corr(pitch_a, target_y) {_num(conf['corr_pitch_a_target_y'], 3)}, "
        f"corr(pnp_pitch, target_y) {_num(conf['corr_pnp_pitch_target_y'], 3)}, "
        f"corr(eye_mid_y, target_y) {_num(conf['corr_eye_mid_y_target_y'], 3)} -- {conf['note']}"
    )
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M2-05: can the calibration layer recover the compressed vertical axis?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recording",
        type=Path,
        default=Path(__file__).resolve().parent / "recordings" / "round18",
        help="round directory holding A / TUNE / T1 (fit on A, compare on TUNE)",
    )
    parser.add_argument(
        "--confirm",
        type=Path,
        default=None,
        help="optional second round directory whose T1 is scored once as confirmation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "vertical",
        help="output directory for the JSON and text report (never inside recordings/)",
    )
    parser.add_argument("--quick", action="store_true", help="small grid, for a smoke run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    if "recordings" in Path(out_dir).resolve().parts:
        print("refusing to write inside recordings/: it is read-only", file=sys.stderr)
        return 2
    inputs = load_inputs(args.recording, args.confirm)
    result = analyse(inputs, quick=args.quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vertical.json").write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    report = format_report(result)
    (out_dir / "vertical_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nwrote {out_dir / 'vertical.json'} and {out_dir / 'vertical_report.txt'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
