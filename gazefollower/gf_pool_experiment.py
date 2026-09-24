"""Can the pooled model be made more accurate from recordings already on disk?

Offline, no camera, no OS input. Never overwrites a model or a profile.
Every fold model is written under ``results/pool_experiment/`` only.

Pre-registered (plan reviewed by Opus before any result, 15.9):

* Held-out unit: a SITTING -- recordings whose time spans are separated by less
  than ``SITTING_GAP_S``. ``meta.session`` is a per-launch timestamp and splits
  one sitting into several ids, so it is not used for grouping. The whole
  sitting is removed from training. ``--group day`` gives the secondary
  leave-one-day-out check, which with 5 folds is expected to be inconclusive.
* Training data: the pooled recipe (``gf_pool.build``, every poolable
  protocol, the main compatibility key, round44 excluded as in the active
  artifact), refitted per fold. The saved artifact is never evaluated.
* Scored: T1 and T3 only, filtered with the active profile's filter (the gate),
  unfiltered reported beside it. A fold needs >= ``MIN_FOLD_SEGMENTS``.
* Arms:
    - baseline: the recipe.
    - H1 (position balance): rows per rounded screen position capped at the
      median per-position count of that fold's pool, spread evenly over the
      recordings that show the position. Tests balance across positions.
    - H2 (per-sitting offset, promotable): a (dx, dy) offset from the sitting's
      own calibration protocol A (one vote per A target, median residual),
      ridge-shrunk toward zero with a prior of ``H2_PRIOR_TARGETS`` targets.
      Only folds whose sitting has a full A (>= 9 targets). A positions never
      coincide with T1/T3 positions, so support is never scored.
    - H2b (offset + horizontal gain, DESCRIPTIVE ONLY, never promoted).
* Gate, per arm against baseline on the same folds:
    per fold = median over presentations of (baseline - arm) euclid px on
    frames both predicted; across folds = median; 95 % percentile bootstrap
    over folds (B=10000, seed 0); sign count. IMPROVEMENT requires median > 0,
    CI low > 0, >= 2/3 of folds better, no measured band zone WORSE (zone 4
    has no T1/T3 data and is reported as not measured), and the bottom row
    (zones 6-8) with no zone WORSE and a paired median >= 0. REGRESSION when
    median < 0 and CI high < 0, or any zone WORSE. Otherwise INSUFFICIENT.
* All of this data was already used to choose C/gamma: it is development data,
  and T1 positions recur across sittings, so absolute accuracy is optimistic.

Usage (from gazefollower/):
    python gf_pool_experiment.py                # sittings (primary)
    python gf_pool_experiment.py --group day    # secondary
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_filter_benchmark as B  # noqa: E402
from gazelink_core.calibration import correction as CORR  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_pool as POOL  # noqa: E402
import gf_presets as PRE  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_recal_compare as RC  # noqa: E402
import gf_report as R  # noqa: E402
import gf_schema as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "pool_experiment"

CONFIG_NAME = "svr_fzscore_lnone_C100_g0.0005"
ALWAYS_EXCLUDED_ROUNDS = ("round44",)  # as in the active artifact's recipe
SCORED_PROTOCOLS = ("T1", "T3")
BAND_X = (0.30, 0.70)
SITTING_GAP_S = 600.0
MIN_FOLD_SEGMENTS = 5
H2_PRIOR_TARGETS = 9.0
H2_MIN_SUPPORT_TARGETS = 9
POSITION_DECIMALS_CM = 1  # the same rounding gf_pool uses to count positions
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0
BOTTOM_ROW_ZONES = (6, 7, 8)
UNMEASURABLE_ZONES = (4,)


# --- grouping ----------------------------------------------------------------


@dataclass
class Item:
    directory: Path
    protocol: str
    start_ns: int
    end_ns: int
    key: tuple[Any, ...]
    day: str


def inventory(root: Path | None = None) -> list[Item]:
    items = []
    for directory, protocol in POOL.discover(root):
        meta = json.loads((directory / f"{protocol}.meta.json").read_text(encoding="utf-8"))
        try:
            rec = S.Recording.load(directory, protocol)
        except Exception:  # noqa: BLE001 - unreadable recordings are simply not items
            continue
        if rec.n_rows == 0:
            continue
        ts = np.asarray(rec.timestamp_ns, dtype=np.int64)
        items.append(
            Item(
                directory=directory,
                protocol=protocol,
                start_ns=int(ts.min()),
                end_ns=int(ts.max()),
                key=POOL.compatibility_key(meta),
                day=str(meta.get("session") or "")[:8],
            )
        )
    return sorted(items, key=lambda i: i.start_ns)


def sittings(items: list[Item], gap_s: float = SITTING_GAP_S) -> list[list[Item]]:
    """Time-contiguous groups: a new group starts after a gap longer than gap_s."""

    groups: list[list[Item]] = []
    last_end: int | None = None
    for item in sorted(items, key=lambda i: i.start_ns):
        if last_end is None or (item.start_ns - last_end) / 1e9 > gap_s:
            groups.append([])
        groups[-1].append(item)
        last_end = item.end_ns if last_end is None else max(last_end, item.end_ns)
    return groups


def days(items: list[Item]) -> list[list[Item]]:
    by: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by[item.day].append(item)
    return [by[d] for d in sorted(by)]


# --- H1: position balance ------------------------------------------------------


def position_balanced_rows(
    Y_cm: np.ndarray, source_of_row: np.ndarray, *, cap: int | None = None
) -> np.ndarray:
    """Indices keeping at most ``cap`` rows per rounded position, spread over sources.

    ``cap`` defaults to the median per-position count. Each position's budget
    is split evenly between the recordings that show it (earlier sources get
    the remainder); within a recording rows are taken evenly spaced in order.
    A recording with fewer rows than its share gives what it has.
    """

    pos = np.round(Y_cm, POSITION_DECIMALS_CM)
    keys, inverse = np.unique(pos, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse, minlength=len(keys))
    if cap is None:
        cap = int(np.median(counts))
    keep: list[np.ndarray] = []
    for k in range(len(keys)):
        rows = np.flatnonzero(inverse == k)
        if rows.size <= cap:
            keep.append(rows)
            continue
        sources = np.unique(source_of_row[rows])
        share, extra = divmod(cap, len(sources))
        for j, src in enumerate(sources):
            mine = rows[source_of_row[rows] == src]
            want = min(mine.size, share + (1 if j < extra else 0))
            if want <= 0:
                continue
            pick = np.linspace(0, mine.size - 1, want).round().astype(int)
            keep.append(mine[np.unique(pick)])
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


# --- H2: per-sitting correction ---------------------------------------------


def support_residuals(pred: np.ndarray, rec: S.Recording, rows: np.ndarray) -> np.ndarray:
    """One (target_x, target_y, median pred_x, median pred_y) per A target, normalized."""

    out = []
    for tid in np.unique(rec.target_id[rows]):
        if tid < 0:
            continue
        m = rows & (rec.target_id == tid) & np.all(np.isfinite(pred), axis=1)
        if not m.any():
            continue
        tx, ty = np.median(rec.target_xy[m], axis=0)
        px, py = np.median(pred[m], axis=0)
        out.append((tx, ty, px, py))
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def ridge_offset(
    support: np.ndarray, prior_targets: float = H2_PRIOR_TARGETS
) -> tuple[float, float]:
    """Offset to SUBTRACT from predictions, shrunk toward zero: sum(r) / (n + prior)."""

    if support.size == 0:
        return 0.0, 0.0
    r = support[:, 2:4] - support[:, 0:2]
    shrunk = r.sum(axis=0) / (len(r) + prior_targets)
    return float(shrunk[0]), float(shrunk[1])


def ridge_x_gain(
    support: np.ndarray, prior_targets: float = H2_PRIOR_TARGETS
) -> tuple[float, float]:
    """Map pred_x -> a + g * (pred_x - 0.5) + 0.5, with a ridge prior at a=0, g=1."""

    if support.size == 0:
        return 0.0, 1.0
    u = support[:, 2] - 0.5
    y = support[:, 0] - 0.5
    X = np.c_[np.ones_like(u), u]
    lam = prior_targets * np.eye(2)
    prior = np.array([0.0, 1.0])
    coef = np.linalg.solve(X.T @ X + lam, X.T @ y + lam @ prior)
    return float(coef[0]), float(coef[1])


# --- scoring -------------------------------------------------------------------


def presentations(rec: S.Recording, rows: np.ndarray) -> int:
    pid = np.concatenate(([0], np.cumsum(np.diff(rec.target_id) != 0)))
    return int(len({int(p) for p in pid[rows] if rec.target_id[np.flatnonzero(pid == p)[0]] >= 0}))


def paired_fold(
    base: np.ndarray, arm: np.ndarray, rec: S.Recording
) -> tuple[list[R.SegmentResult], list[R.SegmentResult], list[float]]:
    eligible = FIT.eligible_rows(rec)
    both = eligible & np.all(np.isfinite(base), axis=1) & np.all(np.isfinite(arm), axis=1)
    sb = RC.paired_segments(base, rec, both)
    sa = RC.paired_segments(arm, rec, both)
    if len(sb) != len(sa):
        raise AssertionError("paired segmentation differs between arms")
    diffs = [b.median_euclid_px - a.median_euclid_px for b, a in zip(sb, sa, strict=True)]
    for seg in sb + sa:
        seg.target_name = f"r{rec.round_id}{rec.protocol}:" + seg.target_name
    return sb, sa, diffs


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(values)
    meds = np.median(arr[rng.integers(0, len(arr), size=(BOOTSTRAP_B, len(arr)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def verdict(
    fold_medians: list[float], zone_gate: dict[str, Any], bottom_diffs: list[float]
) -> dict[str, Any]:
    n = len(fold_medians)
    med = statistics.median(fold_medians) if n else None
    lo, hi = bootstrap_ci(fold_medians)
    better = sum(1 for v in fold_medians if v > 0)
    worse_zones = [
        z["zone"]
        for z in zone_gate["zones"]
        if z["zone"] not in UNMEASURABLE_ZONES and z.get("verdict") == "WORSE"
    ]
    bottom_worse = [z for z in worse_zones if z in BOTTOM_ROW_ZONES]
    bottom_med = statistics.median(bottom_diffs) if bottom_diffs else None
    out = {
        "folds": n,
        "median_fold_improvement_px": med,
        "ci95": [lo, hi],
        "folds_better": f"{better}/{n}",
        "zones_worse": worse_zones,
        "bottom_row_paired_median_px": bottom_med,
    }
    if n and worse_zones or (med is not None and hi is not None and med < 0 and hi < 0):
        out["outcome"] = "REGRESSION"
    elif (
        n >= 2
        and med is not None
        and med > 0
        and lo is not None
        and lo > 0
        and better >= math.ceil(2 * n / 3)
        and not bottom_worse
        and bottom_med is not None
        and bottom_med >= 0
    ):
        out["outcome"] = "IMPROVEMENT"
    else:
        out["outcome"] = "INSUFFICIENT"
    return out


# --- the experiment ------------------------------------------------------------


@dataclass
class ArmResult:
    base_segments: list[R.SegmentResult] = field(default_factory=list)
    arm_segments: list[R.SegmentResult] = field(default_factory=list)
    fold_medians: list[float] = field(default_factory=list)
    fold_ids: list[str] = field(default_factory=list)
    bottom_diffs: list[float] = field(default_factory=list)


def fit_or_load(path: Path, X: np.ndarray, Y: np.ndarray, rig: C.RigGeometry, meta: dict) -> Any:
    if (path / "schema.json").exists():
        return CORR.load_with_correction(path, FIT.FittedModel.load)
    cfg = next(c for c in PRE.sweep_with_presets(()) if c.name == CONFIG_NAME)
    model = FIT.FittedModel.fit(cfg, X, Y, rig=rig, train_meta=meta)
    model.save(path)
    return model


def run(group: str, *, limit: int | None = None, profile_name: str = "pooled_trial") -> dict:
    items = [i for i in inventory() if i.directory.name not in ALWAYS_EXCLUDED_ROUNDS]
    keys = defaultdict(int)
    for i in items:
        keys[i.key] += 1
    main_key = max(keys, key=keys.get)
    items = [i for i in items if i.key == main_key]
    groups = sittings(items) if group == "sitting" else days(items)
    settings = PROF.load(profile_name).filter_settings()

    arms = {"H1": ArmResult(), "H2": ArmResult(), "H2b_descriptive": ArmResult()}
    folds_log: list[dict[str, Any]] = []
    slope_points: list[tuple[float, float]] = []
    fold_band_slopes: list[float] = []
    done = 0
    for gi, members in enumerate(groups):
        checks = [m for m in members if m.protocol in SCORED_PROTOCOLS]
        if not checks:
            continue
        recs = []
        rec_items = {}
        for m in checks:
            rec = S.Recording.load(m.directory, m.protocol)
            if presentations(rec, FIT.eligible_rows(rec)) >= MIN_FOLD_SEGMENTS:
                recs.append(rec)
                rec_items[id(rec)] = m
        if not recs:
            folds_log.append({"group": gi, "skipped": "fewer than MIN_FOLD_SEGMENTS"})
            continue
        fold_id = f"{group}{gi:02d}_" + "+".join(sorted({m.directory.name for m in members}))
        exclude = [(m.directory, m.protocol) for m in members] + [
            (HERE / "recordings" / r, p)
            for r in ALWAYS_EXCLUDED_ROUNDS
            for p in POOL.POOLABLE_PROTOCOLS
        ]
        pool = POOL.build((), exclude=exclude, require=main_key)
        held = {(m.directory.resolve(), m.protocol) for m in members}
        leaked = [s for s in pool.sources if (s.directory.resolve(), s.protocol) in held]
        if leaked:
            raise AssertionError(f"held-out recordings in the training pool: {leaked}")
        source_of_row = np.repeat(np.arange(len(pool.sources)), [s.rows for s in pool.sources])
        meta = {"fitted_by": f"pool_experiment {fold_id}", "held_out": sorted(str(h) for h in held)}
        mdir = OUT / "models" / fold_id
        base = fit_or_load(mdir / "baseline", pool.X, pool.Y_cm, pool.rig, meta)
        keep = position_balanced_rows(pool.Y_cm, source_of_row)
        h1 = fit_or_load(
            mdir / "h1", pool.X[keep], pool.Y_cm[keep], pool.rig, dict(meta, rows=int(keep.size))
        )

        # H2 support is chosen PER SCORED RECORDING: the latest full A (>= 9
        # targets) in the same sitting that ends before that recording starts,
        # as a user would calibrate first. Not applicable to day folds, where
        # one A would be applied to checks hours later.
        support_cache: dict[str, np.ndarray | None] = {}

        def support_for(
            scored: Item,
            members: list[Item] = members,
            support_cache: dict[str, np.ndarray | None] = support_cache,
            base: Any = base,
            pool: POOL.Pool = pool,
        ) -> tuple[np.ndarray | None, str | None]:
            if group != "sitting":
                return None, None
            full = []
            for m in members:
                if m.protocol != "A" or m.end_ns >= scored.start_ns:
                    continue
                key = f"{m.directory.name}/A"
                if key not in support_cache:
                    arec = S.Recording.load(m.directory, "A")
                    amask, _ = FIT.training_rows(arec, ())
                    n_t = len({int(t) for t in arec.target_id[amask] if t >= 0})
                    support_cache[key] = (
                        support_residuals(B.predict(base, arec, pool.rig), arec, amask)
                        if n_t >= H2_MIN_SUPPORT_TARGETS
                        else None
                    )
                if support_cache[key] is not None:
                    full.append(m)
            if not full:
                return None, None
            chosen = max(full, key=lambda m: m.end_ns)
            key = f"{chosen.directory.name}/A"
            return support_cache[key], key

        fold_rec = {
            "fold": fold_id,
            "train_rows": pool.rows,
            "h1_rows": int(keep.size),
            "scored": [f"{r.meta.get('round_id')}/{r.protocol}" for r in recs],
            "h2_support_by_scored": {},
        }
        per_arm_fold: dict[str, list[float]] = defaultdict(list)
        fold_band_points: list[tuple[float, float]] = []
        for rec in recs:
            rig = C.RigGeometry.from_dict(rec.meta["rig"])
            raw_b = B.predict(base, rec, rig)
            filt_b, _ = B.replay(raw_b, rec.timestamp_ns, settings)
            candidates = {"H1": B.predict(h1, rec, rig)}
            support, support_key = support_for(rec_items[id(rec)])
            fold_rec["h2_support_by_scored"][f"{rec.meta.get('round_id')}/{rec.protocol}"] = (
                support_key
            )
            if support is not None:
                ox, oy = ridge_offset(support)
                candidates["H2"] = raw_b - np.array([ox, oy])
                a, g = ridge_x_gain(support)
                gx = raw_b.copy()
                gx[:, 0] = a + g * (raw_b[:, 0] - 0.5) + 0.5
                gx[:, 1] = raw_b[:, 1] - oy
                candidates["H2b_descriptive"] = gx
                fold_rec.setdefault("h2_offset_norm", []).append([ox, oy])
                fold_rec.setdefault("h2b_a_g", []).append([a, g])
            eligible = FIT.eligible_rows(rec)
            for seg in RC.paired_segments(
                filt_b, rec, eligible & np.all(np.isfinite(filt_b), axis=1)
            ):
                w = int(rec.meta["target_geometry"]["width_px"]) - 1
                point = (seg.target_x, seg.target_x + seg.median_dx_px / w)
                slope_points.append(point)
                if BAND_X[0] <= seg.target_x <= BAND_X[1]:
                    fold_band_points.append(point)
            for name, raw_a in candidates.items():
                filt_a, _ = B.replay(raw_a, rec.timestamp_ns, settings)
                sb, sa, diffs = paired_fold(filt_b, filt_a, rec)
                arms[name].base_segments += sb
                arms[name].arm_segments += sa
                per_arm_fold[name] += diffs
                arms[name].bottom_diffs += [
                    d
                    for d, s in zip(diffs, sb, strict=True)
                    if (R.T.band_zone_of(s.target_x, s.target_y) in BOTTOM_ROW_ZONES)
                ]
        fb = np.asarray(fold_band_points)
        if len(fb) >= 3 and np.ptp(fb[:, 0]) > 0:
            band_slope = float(np.polyfit(fb[:, 0], fb[:, 1], 1)[0])
            fold_band_slopes.append(band_slope)
            fold_rec["baseline_band_x_slope"] = band_slope
        for name, diffs in per_arm_fold.items():
            if diffs:
                arms[name].fold_medians.append(float(statistics.median(diffs)))
                arms[name].fold_ids.append(fold_id)
                fold_rec[f"{name}_median_improvement_px"] = float(statistics.median(diffs))
        folds_log.append(fold_rec)
        print(json.dumps(fold_rec), flush=True)
        done += 1
        if limit is not None and done >= limit:
            break

    results: dict[str, Any] = {}
    for name, arm in arms.items():
        if not arm.fold_medians:
            results[name] = {"folds": 0, "outcome": "INSUFFICIENT"}
            continue
        before = R.by_band_zone(arm.base_segments)
        after = R.by_band_zone(arm.arm_segments)
        gate = R.compare_band_zones(before, after)
        v = verdict(arm.fold_medians, gate, arm.bottom_diffs)
        # The fold statistic on its own, so a zone-rule REGRESSION is not read
        # as "the arm hurts" when the folds themselves are inconclusive.
        v["fold_level_outcome"] = verdict(
            arm.fold_medians, {"zones": []}, arm.bottom_diffs or [0.0]
        )["outcome"]
        if name.endswith("descriptive"):
            v["outcome"] = "DESCRIPTIVE (never promoted): " + v["outcome"]
        results[name] = {
            **v,
            "fold_ids": arm.fold_ids,
            "zones_before": before,
            "zones_after": after,
            "zone_gate": gate["zones"],
        }
    sp = np.asarray(slope_points)
    slope = None
    if len(sp) >= 3:
        slope = float(np.polyfit(sp[:, 0], sp[:, 1], 1)[0])
    band_lo, band_hi = bootstrap_ci(fold_band_slopes)
    band_slope = {
        "per_fold_median": statistics.median(fold_band_slopes) if fold_band_slopes else None,
        "fold_bootstrap_ci95": [band_lo, band_hi],
        "folds_below_1": f"{sum(1 for v in fold_band_slopes if v < 1)}/{len(fold_band_slopes)}",
        "note": "targets inside x 0.30-0.70 only; the all-target slope is pulled by edge targets",
    }
    return {
        "group": group,
        "units": "logical px (device = logical x dpi_scale)",
        "gate_path": "filtered with profile " + profile_name,
        "sitting_gap_s": SITTING_GAP_S,
        "n_groups": len(groups),
        "folds": folds_log,
        "baseline_x_slope_all_targets_pooled": slope,
        "baseline_x_slope_band_only": band_slope,
        "arms": results,
        "notes": [
            "development data: C/gamma were chosen on these recordings",
            "T1 positions recur across sittings: absolute accuracy optimistic",
            "zone 4 has no T1/T3 data and cannot be measured",
            "H2b is descriptive only",
        ],
        "os_input": "none",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline pooled-model accuracy experiment.")
    ap.add_argument("--group", choices=("sitting", "day"), default="sitting")
    ap.add_argument("--limit", type=int, default=None, help="stop after N folds (smoke test)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    result = run(args.group, limit=args.limit)
    target = OUT / f"{args.group}{'_limit' + str(args.limit) if args.limit else ''}.json"
    target.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    for name, r in result["arms"].items():
        print(
            name,
            {
                k: r.get(k)
                for k in (
                    "outcome",
                    "folds",
                    "median_fold_improvement_px",
                    "ci95",
                    "folds_better",
                    "zones_worse",
                    "bottom_row_paired_median_px",
                )
            },
        )
    print("baseline x slope, all targets pooled:", result["baseline_x_slope_all_targets_pooled"])
    print("baseline x slope, band only per fold:", result["baseline_x_slope_band_only"])
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
