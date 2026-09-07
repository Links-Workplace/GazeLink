"""Screen calibration candidates for safety before ranking them for accuracy.

A median hides its own tail. In the central-band run the automatic selector
chose ``ridge_a100`` on a TUNE median of 138 px; on the independent test set
the same model produced a Euclidean P90 of 142,863 px, a maximum of 170,788 px
and 102 predictions off the screen. Half the time it was fine, and the other
half it put the cursor in another county. Ranking by a central statistic
cannot see that, so a candidate must first PASS A SCREEN and only then be
ranked.

Three ideas the screen is built on:

**Tails, not centres.** Rejection looks at P90/P95/P99, the maximum, and how
often the prediction leaves the screen. Those are the numbers a person
actually feels.

**Normalised units.** Thresholds are fractions of the screen diagonal, not
pixels, so moving from a 5120 x 1440 ultrawide to a 1920 x 1080 monitor does
not silently change the standard. An angular gate can be layered on when the
geometry is known, and is the one the specification's goal is written in.

**Per-fold, not just aggregate.** Every target is a fold. A candidate that is
excellent on nine targets and catastrophic on the tenth is rejected, because
the aggregate would have hidden exactly that.

When nothing passes, the answer is "no acceptable calibration" -- never the
least bad candidate. A gaze cursor that is usually right and occasionally
1,000 px away is worse than one that reports it cannot calibrate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

SCREEN_VERSION = "screen-1"


@dataclass(frozen=True)
class ScreeningThresholds:
    """Rejection rules, in units that survive a change of screen.

    Defaults are PROPOSED EXPERIMENTAL GATES, not validated product limits.
    Each is expressed as a fraction of the screen diagonal so the same numbers
    mean the same thing on any display; the pixel equivalents are printed in
    every report.

    Rationale for the defaults, on the 4096 x 1152 logical geometry these were
    developed against (diagonal 4255 px):

    * ``max_p90_frac`` 0.12 -> 511 px. The project's own 120 px reference is a
      median target; a P90 four times that is already a poor experience but is
      still measuring the same phenomenon rather than a broken model.
    * ``max_p99_frac`` 0.25 -> 1064 px. Beyond a quarter of the diagonal the
      prediction is not a noisy estimate of the target any more.
    * ``max_error_frac`` 1.0 -> one diagonal. A single error larger than the
      whole screen is a numerical failure, not an accuracy problem.
    * ``max_offscreen_rate`` 0.02. Some overshoot at the edges is normal; 2 %
      of samples leaving the display is not.
    * ``min_valid_rate`` 0.95 matches the availability goal in the targets
      document.
    * ``max_fold_median_frac`` 0.35 -> 1489 px. Catches the "excellent on nine
      targets, hopeless on the tenth" shape that an aggregate conceals.
    """

    max_p90_frac: float = 0.12
    max_p95_frac: float = 0.18
    max_p99_frac: float = 0.25
    max_error_frac: float = 1.0
    max_offscreen_rate: float = 0.02
    min_valid_rate: float = 0.95
    max_fold_median_frac: float = 0.35
    # Optional angular gate, applied only when measured geometry is supplied.
    # The specification's accuracy goal is a MEAN below 1.5 deg; this is a
    # tail gate and is deliberately looser.
    max_p95_deg: float | None = None
    require_finite: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def in_pixels(self, diagonal_px: float) -> dict[str, float]:
        return {
            "max_p90_px": self.max_p90_frac * diagonal_px,
            "max_p95_px": self.max_p95_frac * diagonal_px,
            "max_p99_px": self.max_p99_frac * diagonal_px,
            "max_error_px": self.max_error_frac * diagonal_px,
            "max_fold_median_px": self.max_fold_median_frac * diagonal_px,
            "diagonal_px": diagonal_px,
        }


@dataclass
class Failure:
    rule: str
    observed: float
    limit: float
    unit: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.rule}: {self.observed:.4g} > {self.limit:.4g} {self.unit}{(' (' + self.detail + ')') if self.detail else ''}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateReport:
    """One candidate's screening outcome, with everything that decided it."""

    name: str
    passed: bool
    failures: list[Failure] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    per_fold: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [str(f) for f in self.failures]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "failures": [f.to_dict() for f in self.failures],
            "reasons": self.reasons,
            "stats": self.stats,
            "per_fold": self.per_fold,
        }


def screen_diagonal_px(width_px: int, height_px: int) -> float:
    return float(math.hypot(width_px - 1, height_px - 1))


def screen_candidate(
    name: str,
    *,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    fold_id: np.ndarray,
    width_px: int,
    height_px: int,
    thresholds: ScreeningThresholds = ScreeningThresholds(),
    degrees: np.ndarray | None = None,
    fit_failed: bool = False,
    fit_error: str = "",
) -> CandidateReport:
    """Does this candidate's behaviour clear the safety bar?

    ``pred_norm`` covers every ELIGIBLE row, so a row the model could not
    predict arrives as NaN and counts against availability rather than
    silently leaving the denominator.
    """

    diagonal = screen_diagonal_px(width_px, height_px)
    limits = thresholds.in_pixels(diagonal)
    failures: list[Failure] = []

    if fit_failed:
        return CandidateReport(
            name=name,
            passed=False,
            failures=[Failure("fit_failed", 1.0, 0.0, "boolean", fit_error or "the fit did not produce a model")],
            stats={"fit_failed": True, "fit_error": fit_error},
        )

    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    target_norm = np.asarray(target_norm, dtype=np.float64)
    fold_id = np.asarray(fold_id)
    n_eligible = int(pred_norm.shape[0])
    if n_eligible == 0:
        return CandidateReport(
            name=name,
            passed=False,
            failures=[Failure("no_rows", 0.0, 1.0, "rows", "nothing to screen")],
            stats={"n_eligible": 0},
        )

    finite = np.all(np.isfinite(pred_norm), axis=1)
    n_valid = int(finite.sum())
    valid_rate = n_valid / n_eligible

    dx = (pred_norm[:, 0] - target_norm[:, 0]) * (width_px - 1)
    dy = (pred_norm[:, 1] - target_norm[:, 1]) * (height_px - 1)
    euclid = np.hypot(dx, dy)
    euclid_valid = euclid[finite]

    offscreen = np.zeros(n_eligible, dtype=bool)
    offscreen[finite] = (
        (pred_norm[finite, 0] < 0.0)
        | (pred_norm[finite, 0] > 1.0)
        | (pred_norm[finite, 1] < 0.0)
        | (pred_norm[finite, 1] > 1.0)
    )
    offscreen_rate = float(offscreen.sum()) / n_eligible

    def pct(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q)) if values.size else float("nan")

    stats: dict[str, Any] = {
        "n_eligible": n_eligible,
        "n_valid": n_valid,
        "valid_rate": valid_rate,
        "offscreen_rate": offscreen_rate,
        "n_offscreen": int(offscreen.sum()),
        "median_px": pct(euclid_valid, 50),
        "mean_px": float(np.mean(euclid_valid)) if euclid_valid.size else float("nan"),
        "p90_px": pct(euclid_valid, 90),
        "p95_px": pct(euclid_valid, 95),
        "p99_px": pct(euclid_valid, 99),
        "max_px": float(np.max(euclid_valid)) if euclid_valid.size else float("nan"),
        "diagonal_px": diagonal,
        "limits_px": limits,
    }
    for key in ("median", "p90", "p95", "p99", "max"):
        stats[f"{key}_frac"] = stats[f"{key}_px"] / diagonal

    # --- availability and numerical soundness -------------------------------
    if thresholds.require_finite and n_valid < n_eligible:
        # Missing predictions are an availability problem, judged by rate.
        pass
    if valid_rate < thresholds.min_valid_rate:
        failures.append(
            Failure("valid_rate", valid_rate, thresholds.min_valid_rate, "fraction",
                    f"{n_eligible - n_valid} of {n_eligible} rows had no usable prediction")
        )
    if n_valid == 0:
        failures.append(Failure("no_valid_predictions", 0.0, 1.0, "rows"))
        return CandidateReport(name=name, passed=False, failures=failures, stats=stats)

    # --- tails ---------------------------------------------------------------
    for key, frac_limit, label in (
        ("p90", thresholds.max_p90_frac, "max_p90_frac"),
        ("p95", thresholds.max_p95_frac, "max_p95_frac"),
        ("p99", thresholds.max_p99_frac, "max_p99_frac"),
        ("max", thresholds.max_error_frac, "max_error_frac"),
    ):
        observed = stats[f"{key}_frac"]
        if not math.isfinite(observed) or observed > frac_limit:
            failures.append(
                Failure(label, observed, frac_limit, "of screen diagonal",
                        f"{stats[f'{key}_px']:.0f} px vs limit {frac_limit * diagonal:.0f} px")
            )

    if offscreen_rate > thresholds.max_offscreen_rate:
        failures.append(
            Failure("max_offscreen_rate", offscreen_rate, thresholds.max_offscreen_rate, "fraction",
                    f"{int(offscreen.sum())} of {n_eligible} predictions left the display")
        )

    # --- angular gate, only with measured geometry ---------------------------
    if thresholds.max_p95_deg is not None and degrees is not None:
        deg = np.asarray(degrees, dtype=np.float64)
        deg = deg[np.isfinite(deg)]
        if deg.size:
            observed = float(np.percentile(deg, 95))
            stats["p95_deg"] = observed
            if observed > thresholds.max_p95_deg:
                failures.append(Failure("max_p95_deg", observed, thresholds.max_p95_deg, "degrees"))

    # --- per fold: one catastrophic target is enough to reject ---------------
    per_fold: list[dict[str, Any]] = []
    worst_fold_frac, worst_fold = 0.0, None
    for fid in sorted({int(v) for v in fold_id}):
        rows = fold_id == fid
        f_valid = rows & finite
        entry: dict[str, Any] = {
            "fold": fid,
            "n_eligible": int(rows.sum()),
            "n_valid": int(f_valid.sum()),
            "offscreen": int(offscreen[rows].sum()),
        }
        if f_valid.any():
            fe = euclid[f_valid]
            entry["median_px"] = float(np.median(fe))
            entry["max_px"] = float(np.max(fe))
            entry["median_frac"] = entry["median_px"] / diagonal
            if entry["median_frac"] > worst_fold_frac:
                worst_fold_frac, worst_fold = entry["median_frac"], fid
        else:
            entry["median_px"] = float("nan")
            entry["median_frac"] = float("inf")
            worst_fold_frac, worst_fold = float("inf"), fid
        per_fold.append(entry)

    if worst_fold is not None and worst_fold_frac > thresholds.max_fold_median_frac:
        failures.append(
            Failure("max_fold_median_frac", worst_fold_frac, thresholds.max_fold_median_frac,
                    "of screen diagonal", f"worst fold is target {worst_fold}")
        )
    stats["worst_fold"] = worst_fold
    stats["worst_fold_median_frac"] = worst_fold_frac

    return CandidateReport(name=name, passed=not failures, failures=failures, stats=stats, per_fold=per_fold)


@dataclass
class SelectionOutcome:
    """The chosen candidate, or an explicit refusal, plus every report."""

    selected: str | None
    mode: str  # "auto" | "preset"
    reports: list[CandidateReport]
    ranking_metric: str
    thresholds: ScreeningThresholds
    preset_requested: str | None = None
    preset_note: str = ""

    @property
    def accepted(self) -> list[CandidateReport]:
        return [r for r in self.reports if r.passed]

    @property
    def refused(self) -> bool:
        return self.selected is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_version": SCREEN_VERSION,
            "selected": self.selected,
            "mode": self.mode,
            "refused": self.refused,
            "ranking_metric": self.ranking_metric,
            "n_candidates": len(self.reports),
            "n_accepted": len(self.accepted),
            "thresholds": self.thresholds.to_dict(),
            "preset_requested": self.preset_requested,
            "preset_note": self.preset_note,
            "reports": [r.to_dict() for r in self.reports],
        }


def select(
    reports: Sequence[CandidateReport],
    *,
    ranking_metric: str = "median_px",
    thresholds: ScreeningThresholds = ScreeningThresholds(),
    preset: str | None = None,
) -> SelectionOutcome:
    """Screen first, then rank the survivors. Refuse when none survive.

    ``preset`` names a configuration to use instead of the ranked winner. It
    is still screened: a preset that behaves catastrophically is reported as
    such rather than used because somebody named it.
    """

    reports = list(reports)
    accepted = [r for r in reports if r.passed]

    if preset is not None:
        match = next((r for r in reports if r.name == preset), None)
        if match is None:
            return SelectionOutcome(None, "preset", reports, ranking_metric, thresholds, preset,
                                    f"preset {preset!r} was not among the candidates")
        if not match.passed:
            return SelectionOutcome(None, "preset", reports, ranking_metric, thresholds, preset,
                                    f"preset {preset!r} failed screening: {'; '.join(match.reasons)}")
        return SelectionOutcome(preset, "preset", reports, ranking_metric, thresholds, preset,
                                f"preset {preset!r} passed screening and was used as requested")

    if not accepted:
        return SelectionOutcome(None, "auto", reports, ranking_metric, thresholds, None,
                                "no candidate passed screening; no acceptable calibration")

    def key(r: CandidateReport) -> float:
        value = r.stats.get(ranking_metric)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("inf")
        return value if math.isfinite(value) else float("inf")

    winner = sorted(accepted, key=key)[0]
    return SelectionOutcome(winner.name, "auto", reports, ranking_metric, thresholds, None,
                            f"{len(accepted)} of {len(reports)} candidates passed screening")


def format_selection(outcome: SelectionOutcome, *, max_rejected: int = 12) -> str:
    """A report a person can audit: what passed, what did not, and why."""

    lines: list[str] = []
    t = outcome.thresholds
    diagonal = next((r.stats.get("diagonal_px") for r in outcome.reports if r.stats.get("diagonal_px")), None)
    lines.append(f"## Calibration selection ({outcome.mode})\n")
    if outcome.refused:
        lines.append(f"**NO ACCEPTABLE CALIBRATION** -- {outcome.preset_note}\n")
    else:
        lines.append(f"selected **{outcome.selected}** -- {outcome.preset_note}\n")
    lines.append(f"Screened {len(outcome.reports)} candidates; {len(outcome.accepted)} passed. "
                 f"Ranking metric among survivors: `{outcome.ranking_metric}`.\n")
    if diagonal:
        px = t.in_pixels(diagonal)
        lines.append(f"Thresholds (fractions of the {diagonal:.0f} px diagonal): "
                     f"P90 <= {t.max_p90_frac:.0%} ({px['max_p90_px']:.0f} px), "
                     f"P95 <= {t.max_p95_frac:.0%}, P99 <= {t.max_p99_frac:.0%} ({px['max_p99_px']:.0f} px), "
                     f"max <= {t.max_error_frac:.0%}, off-screen <= {t.max_offscreen_rate:.1%}, "
                     f"valid >= {t.min_valid_rate:.0%}, worst fold median <= {t.max_fold_median_frac:.0%}.\n")

    if outcome.accepted:
        lines.append("| candidate | median px | P90 | P95 | P99 | max | off-screen | worst fold |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(outcome.accepted, key=lambda r: r.stats.get("median_px", float("inf"))):
            s = r.stats
            lines.append(
                f"| {r.name} | {s['median_px']:.0f} | {s['p90_px']:.0f} | {s['p95_px']:.0f} | "
                f"{s['p99_px']:.0f} | {s['max_px']:.0f} | {s['offscreen_rate']:.2%} | "
                f"{s.get('worst_fold_median_frac', float('nan')):.0%} |"
            )
        lines.append("")

    rejected = [r for r in outcome.reports if not r.passed]
    if rejected:
        lines.append(f"### Rejected ({len(rejected)})\n")
        for r in rejected[:max_rejected]:
            lines.append(f"- **{r.name}** -- {'; '.join(r.reasons)}")
        if len(rejected) > max_rejected:
            lines.append(f"- ... and {len(rejected) - max_rejected} more")
        lines.append("")
    return "\n".join(lines) + "\n"
