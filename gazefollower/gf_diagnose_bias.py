"""Where does a horizontal offset come from? Trace it stage by stage.

A constant X bias on the held-out targets can be born in four places, and
they call for different fixes, so this script asks each stage separately:

1. RAW     -- the model's own 2-D output (``res[:2]``, cm). If the raw output
              is already shifted relative to the target, the calibration
              inherits it and the fix belongs upstream.
2. FIT     -- residuals of the calibration on its OWN training rows. A model
              that fits protocol A well but is off on T1 has a
              generalisation problem, not a fitting one.
3. SPACE   -- is the offset uniform across the screen (a constant shift), or
              does it grow toward the edges (a scale error)? Per-target dx
              against target x answers that.
4. TIME    -- TUNE was recorded right after A, T1 later. If the offset grows
              across TUNE -> GRID16 -> T1, the person drifted after calibrating
              and the calibration went stale.

Everything is in the recording; nothing here touches a camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_fit as F  # noqa: E402
import gf_schema as S  # noqa: E402


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _per_target(rec: S.Recording, pred_norm: np.ndarray, width: int) -> list[dict]:
    rows = []
    for t in rec.targets:
        tid = int(t["index"])
        m = rec.rows_collecting() & (rec.target_id == tid) & np.all(np.isfinite(pred_norm), axis=1)
        if not m.any():
            continue
        tx = float(t["screen_position"]["x"])
        px = float(np.median(pred_norm[m, 0]))
        rows.append({"name": t["name"], "tx": tx, "px": px, "dx_px": (px - tx) * (width - 1), "n": int(m.sum())})
    return rows


def main(round_dir: Path) -> None:
    rec_a = S.Recording.load(round_dir, "A")
    rig = C.RigGeometry.from_dict(rec_a.meta["rig"])
    width = int(rec_a.meta["target_geometry"]["width_px"])
    tests = {p: S.Recording.load(round_dir, p) for p in ("TUNE", "GRID16", "T1") if (round_dir / f"{p}.npz").exists()}

    print(f"# Horizontal-offset diagnosis: {round_dir.name}\n")

    # ---- 1. RAW: does the model's own output carry the shift? -------------
    print("## 1. Raw model output (before any calibration)\n")
    print("Per protocol: slope/intercept of raw_x (cm, camera frame) against the")
    print("target's label x (cm). A perfect model would give slope 1, intercept 0.\n")
    print("| protocol | axis | rows | slope | intercept (cm) | corr | raw span (cm) | label span (cm) | span ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rec in [("A", rec_a)] + list(tests.items()):
        m = (rec.rows_accepted() if name == "A" else rec.rows_collecting()) & np.all(np.isfinite(rec.raw_cm), axis=1)
        for axis, label in ((0, "X"), (1, "Y")):
            raw = rec.raw_cm[m, axis]
            lab = rec.label_cm[m, axis]
            slope, intercept = _linfit(lab, raw)
            corr = float(np.corrcoef(lab, raw)[0, 1])
            rspan, lspan = raw.max() - raw.min(), lab.max() - lab.min()
            print(
                f"| {name} | {label} | {int(m.sum())} | {slope:.3f} | {intercept:+.2f} | {corr:+.3f} | "
                f"{rspan:.1f} | {lspan:.1f} | 1:{lspan / rspan:.0f} |"
            )
    print()
    print("Reading: a raw span of a few cm against a label span of ~110 cm means the")
    print("model's own X output is nearly flat and the calibration does all the work")
    print("of stretching it. Any intercept here is inherited by the calibration.\n")

    # ---- 2. FIT: does the calibration fit its own training rows? ----------
    print("## 2. Calibration fit on its own training rows (protocol A)\n")
    mask, _ = F.training_rows(rec_a, ())
    X = F.design_matrix(rec_a, mask, ())
    Y = rec_a.label_cm[mask]
    results = {}
    for cfg in (F.library_default_config(), F.FitConfig(name="zscore_C10_gauto", feature_scaling="zscore", label_scaling="zscore", C=10.0, gamma="auto")):
        model = F.FittedModel.fit(cfg, X, Y, rig=rig, train_meta={})
        results[cfg.name] = model
        pred_cm = model.predict_cm(X)
        res_x = pred_cm[:, 0] - Y[:, 0]
        pred_norm = np.array([rig.cm_to_norm(*p) for p in pred_cm])
        lab_norm = np.array([rig.cm_to_norm(*p) for p in Y])
        dx_px = (pred_norm[:, 0] - lab_norm[:, 0]) * (width - 1)
        print(f"- **{cfg.name}**: training residual X median {np.median(np.abs(res_x)):.2f} cm ({np.median(np.abs(dx_px)):.0f} px), "
              f"bias {np.median(dx_px):+.0f} px")
    print()
    print("Reading: a small training residual with a large held-out offset means the")
    print("calibration learned protocol A faithfully and something CHANGED afterwards.\n")

    # ---- 3 & 4. SPACE and TIME: per target, per protocol -------------------
    print("## 3. Offset across the screen and across time\n")
    for cfg_name, model in results.items():
        print(f"### {cfg_name}\n")
        print("| protocol (in recording order) | median dx (px) | slope of dx vs target x | dx at left edge | dx at right edge | interpretation |")
        print("|---|---|---|---|---|---|")
        for pname, rec in tests.items():
            m, _ = F.scoring_rows(rec, ())
            pred_all = np.full((rec.n_rows, 2), np.nan)
            if m.any():
                pred_all[m] = model.predict_norm(F.design_matrix(rec, m, ()), rig)
            rows = _per_target(rec, pred_all, width)
            if len(rows) < 3:
                continue
            tx = np.array([r["tx"] for r in rows])
            dx = np.array([r["dx_px"] for r in rows])
            slope, intercept = _linfit(tx, dx)
            left, right = slope * 0.0 + intercept, slope * 1.0 + intercept
            if abs(slope) < 100:
                kind = "constant shift"
            elif slope * np.sign(intercept + slope / 2) > 0:
                kind = "scale error (grows toward one edge)"
            else:
                kind = "scale error"
            print(f"| {pname} | {np.median(dx):+.0f} | {slope:+.0f} px per screen-width | {left:+.0f} | {right:+.0f} | {kind} |")
        print()

    # ---- per-target table for T1, the held-out set --------------------------
    if "T1" in tests:
        rec = tests["T1"]
        model = results["lib-default"]
        m, _ = F.scoring_rows(rec, ())
        pred_all = np.full((rec.n_rows, 2), np.nan)
        pred_all[m] = model.predict_norm(F.design_matrix(rec, m, ()), rig)
        print("### T1 per target (library default)\n")
        print("| target | target x | predicted x | dx (px) |")
        print("|---|---|---|---|")
        for r in sorted(_per_target(rec, pred_all, width), key=lambda r: r["tx"]):
            print(f"| {r['name']} | {r['tx']:.3f} | {r['px']:.3f} | {r['dx_px']:+.0f} |")
        print()

    # ---- 5. head position at calibration vs at test ------------------------
    print("## 4. Did the head sit somewhere else at test time than at calibration?\n")
    import gf_head_features as H  # noqa: PLC0415

    def head_stats(rec: S.Recording, mask: np.ndarray) -> dict:
        m = mask & rec.rows_head_valid()
        h = rec.head[m]
        return {
            "eye_x_px": float(np.median(h[:, H.HEAD6_NAMES.index("eye_mid_x")]) * 640),
            "eye_y_px": float(np.median(h[:, H.HEAD6_NAMES.index("eye_mid_y")]) * 480),
            "iod_px": float(np.median(h[:, H.HEAD6_NAMES.index("iod_norm")]) * 640),
            "yaw_ratio": float(np.median(h[:, H.HEAD6_NAMES.index("yaw_ratio")])),
        }

    base = head_stats(rec_a, rec_a.rows_accepted())
    print("| protocol | eye x (px) | d from A | eye y (px) | d | iod (px) | d | yaw_ratio | d |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| A (calibration) | {base['eye_x_px']:.1f} | — | {base['eye_y_px']:.1f} | — | {base['iod_px']:.1f} | — | {base['yaw_ratio']:+.3f} | — |")
    for pname, rec in tests.items():
        s = head_stats(rec, rec.rows_collecting())
        print(
            f"| {pname} | {s['eye_x_px']:.1f} | {s['eye_x_px'] - base['eye_x_px']:+.1f} | "
            f"{s['eye_y_px']:.1f} | {s['eye_y_px'] - base['eye_y_px']:+.1f} | "
            f"{s['iod_px']:.1f} | {s['iod_px'] - base['iod_px']:+.1f} | "
            f"{s['yaw_ratio']:+.3f} | {s['yaw_ratio'] - base['yaw_ratio']:+.3f} |"
        )
    print()
    print("Reading: the model receives the face box as an input, so a few pixels of")
    print("head translation between calibration and test can move the prediction")
    print("even when the gaze did not. A shift in eye x that lines up with the sign")
    print("of the X offset is the thing to look at.")
    print()
    print_saturation(round_dir)


def saturation_check(rec: S.Recording, axis: int, coords: tuple[float, ...]) -> list[tuple[float, float, float, int]]:
    """Median raw output at each target level along one axis.

    Distinguishes two very different reasons for a small raw span. A signal
    that is merely WEAK rises linearly with the target, just with a small
    gain: noise-limited, and more pixels on the eyes would help. A signal
    that SATURATES rises in the middle and goes flat at the edges: the model
    has run out of range, and only a narrower screen region would help.
    Returns (level, median raw cm, IQR, n) per level.
    """

    m = rec.rows_collecting() & np.all(np.isfinite(rec.raw_cm), axis=1)
    out = []
    for level in coords:
        rows = m & (np.abs(rec.target_xy[:, axis] - level) < 1e-6)
        if rows.sum() < 5:
            continue
        v = rec.raw_cm[rows, axis]
        q75, q25 = np.percentile(v, [75, 25])
        out.append((level, float(np.median(v)), float(q75 - q25), int(rows.sum())))
    return out


def print_saturation(round_dir: Path) -> None:
    path = round_dir / "GRID16.npz"
    if not path.exists():
        return
    rec = S.Recording.load(round_dir, "GRID16")
    coords = (0.05, 0.35, 0.65, 0.95)
    print("## 5. Weak or saturated? Raw output per target level on GRID16\n")
    print("If the steps between levels are roughly equal, the signal is linear and")
    print("just weak. If the outer steps are much smaller than the inner one, the")
    print("model has run out of range at the screen edges.\n")
    for axis, label in ((0, "X"), (1, "Y")):
        levels = saturation_check(rec, axis, coords)
        if len(levels) < 4:
            continue
        print(f"### {label}\n")
        print("| target level | median raw (cm) | IQR (cm) | n | step from previous (cm) |")
        print("|---|---|---|---|---|")
        prev = None
        steps = []
        for level, med, iqr, n in levels:
            step = "" if prev is None else f"{med - prev:+.2f}"
            if prev is not None:
                steps.append(med - prev)
            print(f"| {level:.2f} | {med:+.2f} | {iqr:.2f} | {n} | {step} |")
            prev = med
        outer = (abs(steps[0]) + abs(steps[2])) / 2
        inner = abs(steps[1])
        ratio = outer / inner if inner > 1e-9 else float("inf")
        verdict = "linear (weak, not saturated)" if 0.5 <= ratio <= 2.0 else ("SATURATED at the edges" if ratio < 0.5 else "steeper at the edges")
        print(f"\nouter/inner step ratio {ratio:.2f} -> **{verdict}**; total raw span {levels[-1][1] - levels[0][1]:+.2f} cm "
              f"across {coords[-1] - coords[0]:.0%} of the screen\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "recordings" / "round1")
