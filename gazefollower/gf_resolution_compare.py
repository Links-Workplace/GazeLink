"""Does the higher-resolution crop give MGazeNet better information? Paired.

Reads one ``--capture dual`` round (gf_record): ``roundN/{A,T1}`` holds the hi
pipeline (1440x1080 crop), ``roundN/lo/{A,T1}`` the same frames reduced to
640x480. The rows are identical by construction and checked here.

For each arm, separately and on the SAME calibration frames:

* a model with the production configuration, fitted on that arm's features;
* a small C/gamma grid scored by leave-one-calibration-point-out on the
  calibration alone, so neither arm is stuck with settings tuned on the other
  (every existing configuration was chosen on 640x480-library features).

Then both arms predict the SAME test frames (T1, targets never used for
training) and are compared per band zone with ``gf_recal_compare`` -- paired,
because both predicted exactly those frames. A presentation bootstrap puts an
interval on each zone's difference.

A frame counts only where BOTH arms produced a valid sample and both eyes are
open by the 640x480-unit blink rule (the hi arm reports openness in those
units, see gf_capture), so neither arm is scored on frames the other lost.

Also reported: per-frame processing time and delivered frame rate from the
capture, against limits fixed here BEFORE the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_presets as PRE  # noqa: E402
import gf_recal_compare as RC  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402
import gf_targets as T  # noqa: E402

PRODUCTION_CONFIG = "svr_fzscore_lnone_C100_g0.0005"
GRID = (
    "svr_fzscore_lnone_C10_g0.0005",
    "svr_fzscore_lnone_C100_g0.0005",
    "svr_fzscore_lnone_C10_g0.005",
    "svr_fzscore_lnone_C100_g0.005",
    "svr_fzscore_lnone_C10_gauto",
    "svr_fzscore_lnone_C100_gauto",
)
# Fixed before any dual recording exists. Processing is what the hi arm adds
# to every frame; the frame rate is what the person would feel.
LATENCY_LIMITS = {"hi_ms_p95_max": 25.0, "delivered_fps_min": 28.0}
PAIRED_FIELDS = ("frame_seq", "timestamp_ns", "target_id", "phase")


def load_pair(round_dir: Path, protocol: str) -> tuple[S.Recording, S.Recording]:
    hi = S.Recording.load(round_dir, protocol)
    lo = S.Recording.load(round_dir / R.DUAL_LO_SUBDIR, protocol)
    for name in PAIRED_FIELDS:
        if not np.array_equal(getattr(hi, name), getattr(lo, name)):
            raise ValueError(f"{protocol}: hi and lo rows differ in {name}; not a paired recording")
    return hi, lo


def both_valid(hi: S.Recording, lo: S.Recording) -> np.ndarray:
    ok = np.ones(len(hi.target_id), dtype=bool)
    for rec in (hi, lo):
        ok &= np.asarray(rec.gaze_status, dtype=bool)
        ok &= np.all(np.isfinite(rec.features), axis=1)
        ok &= np.all(rec.openness > C.BLINK_THRESHOLD, axis=1)
    return ok


def config_named(name: str) -> Any:
    return next(c for c in PRE.sweep_with_presets(()) if c.name == name)


def fit_arm(rec: S.Recording, mask: np.ndarray, name: str, rig: C.RigGeometry) -> FIT.FittedModel:
    return FIT.FittedModel.fit(
        config_named(name),
        rec.features[mask],
        rec.label_cm[mask],
        rig=rig,
        train_meta={"protocol": "A", "round_id": rec.round_id, "fitted_by": "gf_resolution_compare",
                    "pipeline": (rec.meta.get("capture") or {}).get("pipeline")},
    )


def lopo_error_px(rec: S.Recording, mask: np.ndarray, name: str, rig: C.RigGeometry) -> float:
    """Median over calibration points of the held-out point's median error."""

    width = int(rec.meta["target_geometry"]["width_px"]) - 1
    height = int(rec.meta["target_geometry"]["height_px"]) - 1
    errors = []
    for point in np.unique(rec.target_id[mask]):
        train = mask & (rec.target_id != point)
        test = mask & (rec.target_id == point)
        if train.sum() < 20 or test.sum() == 0:
            continue
        model = fit_arm(rec, train, name, rig)
        pred = model.predict_norm(rec.features[test], rig)
        d = np.hypot((pred[:, 0] - rec.target_xy[test, 0]) * width,
                     (pred[:, 1] - rec.target_xy[test, 1]) * height)
        errors.append(float(np.nanmedian(d)))
    return float(np.median(errors)) if errors else float("nan")


def predict_masked(model: FIT.FittedModel, rec: S.Recording, mask: np.ndarray, rig: C.RigGeometry) -> np.ndarray:
    out = np.full((len(rec.target_id), 2), np.nan)
    out[mask] = model.predict_norm(rec.features[mask], rig)
    return out


def zone_bootstrap(runs: list[tuple[np.ndarray, np.ndarray, S.Recording, np.ndarray]],
                   *, n: int = 2000, seed: int = 0) -> dict[int, dict[str, float]]:
    """Per zone: median presentation error lo - hi (positive = hi better), 90% interval.

    ``runs`` is (lo predictions, hi predictions, recording, scored mask) per test recording.
    """

    rng = np.random.default_rng(seed)
    out: dict[int, dict[str, float]] = {}
    by_zone: dict[int, list[tuple[float, float]]] = {}
    for pred_lo, pred_hi, rec, mask in runs:
        segs_lo = RC.paired_segments(pred_lo, rec, mask)
        segs_hi = RC.paired_segments(pred_hi, rec, mask)
        for a, b in zip(segs_lo, segs_hi, strict=True):
            zone = T.band_zone_of(a.target_x, a.target_y)
            if zone is not None:
                by_zone.setdefault(zone, []).append((a.median_euclid_px, b.median_euclid_px))
    for zone, pairs in sorted(by_zone.items()):
        arr = np.array(pairs)
        diff = arr[:, 0] - arr[:, 1]
        boots = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(n)]
        out[zone] = {
            "presentations": len(diff),
            "lo_median_px": float(np.median(arr[:, 0])),
            "hi_median_px": float(np.median(arr[:, 1])),
            "lo_minus_hi_px": float(np.median(diff)),
            "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
        }
    return out


def latency_verdict(capture: dict[str, Any]) -> dict[str, Any]:
    stats = (capture or {}).get("stats") or {}
    p95 = stats.get("hi_ms_p95")
    fps = stats.get("delivered_fps_median")
    return {
        "hi_ms_p95": p95,
        "lo_ms_p95": stats.get("lo_ms_p95"),
        "delivered_fps_median": fps,
        "limits": LATENCY_LIMITS,
        "hi_within_limit": None if p95 is None else p95 <= LATENCY_LIMITS["hi_ms_p95_max"],
        "fps_within_limit": None if fps is None else fps >= LATENCY_LIMITS["delivered_fps_min"],
        "note": "both pipelines ran on every frame, so the delivered rate is lower than hi alone would get",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument(
        "--extra-test-rounds",
        type=Path,
        nargs="*",
        default=(),
        help="more dual rounds from the SAME sitting whose T1 is added to the test set "
        "(their A, if any, is not used). One T1 gives only two presentations per zone.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cal_hi, cal_lo = load_pair(args.round_dir, "A")
    tests = [load_pair(d, "T1") for d in (args.round_dir, *args.extra_test_rounds)]
    rig = C.RigGeometry.from_dict(cal_hi.meta["rig"])
    train = cal_hi.rows_accepted() & cal_lo.rows_accepted() & both_valid(cal_hi, cal_lo)
    masks = [FIT.eligible_rows(hi) & both_valid(hi, lo) for hi, lo in tests]
    print(f"calibration rows used by both arms: {int(train.sum())} "
          f"(hi accepted {int(cal_hi.rows_accepted().sum())}, lo accepted {int(cal_lo.rows_accepted().sum())})")
    for (hi, _), m in zip(tests, masks, strict=True):
        print(f"test rows scored by both arms in round{hi.round_id}: {int(m.sum())} of "
              f"{int(FIT.eligible_rows(hi).sum())} eligible")

    report: dict[str, Any] = {"round_dir": str(args.round_dir),
                              "extra_test_rounds": [str(d) for d in args.extra_test_rounds],
                              "train_rows": int(train.sum()),
                              "test_rows": [int(m.sum()) for m in masks]}
    grid = {arm: {name: lopo_error_px(rec, train, name, rig) for name in GRID}
            for arm, rec in (("lo", cal_lo), ("hi", cal_hi))}
    chosen = {arm: min(grid[arm], key=grid[arm].get) for arm in grid}
    report["lopo_calibration_px"] = grid
    report["chosen_by_lopo"] = chosen
    print("\nleave-one-calibration-point-out median error, logical px (calibration only):")
    for name in GRID:
        print(f"  {name:34} lo {grid['lo'][name]:7.1f}   hi {grid['hi'][name]:7.1f}")

    report["comparisons"] = {}
    for label, names in (("production config", (PRODUCTION_CONFIG, PRODUCTION_CONFIG)),
                         ("each arm's LOPO choice", (chosen["lo"], chosen["hi"]))):
        m_lo = fit_arm(cal_lo, train, names[0], rig)
        m_hi = fit_arm(cal_hi, train, names[1], rig)
        runs = []
        for (hi, lo), m in zip(tests, masks, strict=True):
            p_lo = predict_masked(m_lo, lo, m, rig)
            p_hi = predict_masked(m_hi, hi, m, rig)
            runs.append((p_lo, p_hi, hi, m))
        zones = RC.zone_comparison([(a, b, rec) for a, b, rec, _ in runs], target_half_device_px=40.0)
        boot = zone_bootstrap(runs)
        report["comparisons"][label] = {"configs": {"lo": names[0], "hi": names[1]}, "zones": zones, "bootstrap": boot}
        print(f"\n## {label}: lo={names[0]}  hi={names[1]}  (old = lo, new = hi)")
        print(RC.format_zone_comparison(zones))
        print("\nper zone, presentation medians (positive = hi better), 90% bootstrap interval:")
        for zone, b in boot.items():
            print(f"  zone {zone} ({T.band_zone_label(zone)}): n={b['presentations']} lo {b['lo_median_px']:.0f} "
                  f"hi {b['hi_median_px']:.0f}  diff {b['lo_minus_hi_px']:+.0f} [{b['ci90'][0]:+.0f}, {b['ci90'][1]:+.0f}]")

    report["latency"] = [latency_verdict(hi.meta.get("capture") or {}) for hi, _ in tests]
    print("\nlatency:", json.dumps(report["latency"], default=float))
    report["units"] = "logical px (device = logical x dpi_scale)"
    report["scope"] = ("one session: a paired go/no-go on whether hi features carry more information. "
                       "Not a production model and not a selection test.")
    out = args.out or Path(__file__).resolve().parent / "results" / "resolution" / f"{args.round_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
