"""Score one recording twice, with two models, on exactly the same frames.

Every previous model comparison in this project compared two RUNS, so it
carried the run-to-run difference -- different eye movement, different head
pose, different instant -- on top of whatever the models did.  On the rounds
recorded two hours apart that difference was larger than the effect being
looked for, which is why those comparisons were unreadable.

Here both models predict the SAME rows of ONE recording.  The eye movement,
the instant, the head pose and the filter are identical on both sides, so the
difference that remains is the model's.  Used to answer one question: did
recalibrating recover the accuracy, or did it not?

Read-only.  No camera, no OS input, no model is refitted.
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
import gf_filter_benchmark as B  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_schema as S  # noqa: E402


def euclid_px(pred_norm: np.ndarray, rec: S.Recording, rows: np.ndarray) -> np.ndarray:
    """Per-row euclidean error in logical px over the selected rows."""

    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    dx = (pred_norm[rows, 0] - rec.target_xy[rows, 0]) * (width - 1)
    dy = (pred_norm[rows, 1] - rec.target_xy[rows, 1]) * (height - 1)
    return np.hypot(dx, dy)


def score_one(
    model_dir: Path,
    rec: S.Recording,
    rig: C.RigGeometry,
    settings: GF.FilterSettings | None,
) -> dict[str, Any]:
    """Unfiltered and filtered metrics for one model on one recording."""

    model = FIT.FittedModel.load(model_dir)
    raw = B.predict(model, rec, rig)
    eligible = FIT.eligible_rows(rec)
    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    names = {int(t["index"]): str(t["name"]) for t in rec.targets}

    def metrics(points: np.ndarray) -> dict[str, Any]:
        return FIT.evaluate(
            points[eligible],
            rec.target_xy[eligible],
            rec.target_id[eligible],
            width,
            height,
            names,
            fresh=rec.rows_fresh()[eligible],
            timestamp_ns=rec.timestamp_ns[eligible],
        ).flat()

    out: dict[str, Any] = {
        "model": str(model_dir),
        "unfiltered": metrics(raw),
        # Support activation separates "the model no longer covers these
        # inputs" -- where it answers with its bias, wrongly but steadily --
        # from "the model covers them and is simply less accurate".
        "support": B.support_activation(model, rec, eligible),
        "jitter_unfiltered": B.jitter_metrics(raw, rec, width, height),
        "_raw": raw,
    }
    if settings is not None:
        filtered, _ = B.replay(raw, rec.timestamp_ns, settings)
        out["filtered"] = metrics(filtered)
        out["jitter_filtered"] = B.jitter_metrics(filtered, rec, width, height)
        out["_filtered"] = filtered
    return out


def paired_difference(old: np.ndarray, new: np.ndarray, rec: S.Recording) -> dict[str, Any]:
    """Old minus new on the rows BOTH models predicted.

    Restricting to rows both produced keeps the comparison paired: a row one
    model lost and the other did not would otherwise be scored on one side
    only, which is a coverage difference reported as an accuracy difference.
    """

    eligible = FIT.eligible_rows(rec)
    old_ok = np.all(np.isfinite(old), axis=1)
    new_ok = np.all(np.isfinite(new), axis=1)
    both = eligible & old_ok & new_ok
    if not np.any(both):
        return {"n_paired": 0}
    e_old = euclid_px(old, rec, both)
    e_new = euclid_px(new, rec, both)
    delta = e_old - e_new
    return {
        "n_paired": int(both.sum()),
        "n_eligible": int(eligible.sum()),
        "n_old_only": int(np.sum(eligible & old_ok & ~new_ok)),
        "n_new_only": int(np.sum(eligible & new_ok & ~old_ok)),
        "median_euclid_old_px": float(np.median(e_old)),
        "median_euclid_new_px": float(np.median(e_new)),
        "median_paired_delta_px": float(np.median(delta)),
        "mean_paired_delta_px": float(np.mean(delta)),
        # A sign count, not a p-value: it says how consistent the direction is
        # across frames, which a median difference alone does not.
        "fraction_of_frames_new_is_better": float(np.mean(delta > 0)),
    }


def _row(label: str, m: dict[str, Any]) -> str:
    def f(key: str, digits: int = 1) -> str:
        v = m.get(key)
        return "n/a" if v is None else f"{v:.{digits}f}"

    return (
        f"| {label} | {f('median_euclid_px')} | {f('x_median_abs_px')} | "
        f"{f('y_median_abs_px')} | {f('x_slope', 3)} | {f('x_intercept')} | "
        f"{f('y_slope', 3)} | {f('y_intercept')} | {f('coverage', 3)} |"
    )


def report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {result['recording']} / {result['protocol']}: old model vs new model")
    lines.append("")
    lines.append("Same frames, same filter, same training configuration on both sides.")
    lines.append("Errors in logical px (analyze.py geometry).")
    lines.append("")
    for band in ("unfiltered", "filtered"):
        if band not in result["old"]:
            continue
        lines.append(f"## {band}")
        lines.append("")
        lines.append(
            "| model | median euclid | med abs dx | med abs dy | x slope | x intercept "
            "| y slope | y intercept | coverage |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines.append(_row("old", result["old"][band]))
        lines.append(_row("new", result["new"][band]))
        lines.append("")
        paired = result["paired"][band]
        if paired.get("n_paired"):
            lines.append(
                f"paired on {paired['n_paired']} frames both models predicted: "
                f"median old-minus-new {paired['median_paired_delta_px']:+.1f} px, "
                f"new better on {paired['fraction_of_frames_new_is_better'] * 100:.0f}% of frames"
            )
            lines.append("")
    for label in ("old", "new"):
        sup = result[label]["support"]
        if sup.get("available"):
            lines.append(
                f"{label} support activation: median {sup['median_activation']:.3f}, "
                f"min {sup['min_activation']:.3f}, "
                f"{sup['fraction_below_0p01'] * 100:.1f}% of frames below 0.01"
            )
        else:
            lines.append(f"{label} support activation: unavailable ({sup.get('reason')})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--protocol", default="T1")
    parser.add_argument("--old", type=Path, required=True, help="model directory")
    parser.add_argument("--new", type=Path, required=True, help="model directory")
    parser.add_argument("--filter", default="one-euro", choices=[k.value for k in GF.FilterKind])
    parser.add_argument("--one-euro-min-cutoff-hz", type=float, default=0.4)
    parser.add_argument("--one-euro-beta", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    rec = S.Recording.load(str(args.recording), args.protocol)
    rig = C.RigGeometry.from_dict(rec.meta["rig"])
    settings = GF.FilterSettings(
        width_px=rig.device_w_px,
        height_px=rig.device_h_px,
        kind=GF.FilterKind(args.filter),
        one_euro_min_cutoff_hz=args.one_euro_min_cutoff_hz,
        one_euro_beta_hz_per_px_s=args.one_euro_beta,
    )
    old = score_one(args.old, rec, rig, settings)
    new = score_one(args.new, rec, rig, settings)
    result = {
        "recording": str(args.recording),
        "protocol": args.protocol,
        "filter": {
            "kind": args.filter,
            "one_euro_min_cutoff_hz": args.one_euro_min_cutoff_hz,
            "one_euro_beta_hz_per_px_s": args.one_euro_beta,
        },
        "old": old,
        "new": new,
        "paired": {
            "unfiltered": paired_difference(old["_raw"], new["_raw"], rec),
            "filtered": paired_difference(old["_filtered"], new["_filtered"], rec),
        },
    }
    print(report(result))
    if args.json_out:
        clean = {
            key: (
                {k: v for k, v in value.items() if not k.startswith("_")}
                if isinstance(value, dict)
                else value
            )
            for key, value in result.items()
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(clean, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
