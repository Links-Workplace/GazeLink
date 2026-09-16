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
import gf_report as R  # noqa: E402
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


# Logical px are what every report in this project uses; the plan's button
# sizes are written in DEVICE px. The factor is read from the recording, not
# assumed, and printed next to every number that depends on it.
def device_per_logical(rec: S.Recording) -> float:
    geometry = rec.meta.get("target_geometry") or {}
    scale = geometry.get("dpi_scale")
    return float(scale) if scale else 1.0


def paired_segments(pred: np.ndarray, rec: S.Recording, rows: np.ndarray) -> list[R.SegmentResult]:
    """One segment per PRESENTATION, over the rows given.

    A presentation is a contiguous run of one target. The recording carries
    no presentation id, and a target shown twice must count twice -- that is
    the module's one-vote-per-segment contract -- so runs are split wherever
    the target changes.

    ``rows`` is the mask BOTH models predicted. Building both sides from the
    same mask is what keeps the zone comparison paired: a frame one model lost
    would otherwise be scored on one side only, a coverage difference reported
    as an accuracy difference.
    """

    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    dx = (pred[:, 0] - rec.target_xy[:, 0]) * (width - 1)
    dy = (pred[:, 1] - rec.target_xy[:, 1]) * (height - 1)
    names = {int(t["index"]): str(t["name"]) for t in rec.targets}
    near = {int(t["index"]): bool(t.get("near_calibration", False)) for t in rec.targets}
    out: list[R.SegmentResult] = []
    if not np.any(rows):
        return out
    # Presentations are numbered over EVERY row of the recording -- a new one
    # starts only where the target changes, idle rows (-1) included -- and the
    # mask is applied afterwards.
    #
    # Splitting on gaps in the MASKED rows instead was wrong, and it failed in
    # the direction that matters: rows drop out of the mask for tracking loss,
    # a blink, or one model not predicting, so a single presentation with two
    # dropouts became three segments. That cleared ``min_segments`` and
    # marked a zone "measured" on the evidence of one showing of one target.
    presentation = np.concatenate(([0], np.cumsum(np.diff(rec.target_id) != 0)))
    for pid in np.unique(presentation[rows]):
        frames = np.flatnonzero(rows & (presentation == pid))
        target = int(rec.target_id[frames[0]])
        if target < 0:
            continue
        err = np.hypot(dx[frames], dy[frames])
        out.append(
            R.SegmentResult(
                target_id=target,
                target_name=names.get(target, f"T{target}"),
                target_x=float(rec.target_xy[frames[0], 0]),
                target_y=float(rec.target_xy[frames[0], 1]),
                condition=rec.protocol,
                near_calibration=near.get(target, False),
                n_eligible=int(frames.size),
                n_valid=int(frames.size),
                median_dx_px=float(np.median(dx[frames])),
                median_dy_px=float(np.median(dy[frames])),
                median_euclid_px=float(np.median(err)),
                euclid_px=err,
            )
        )
    return out


def zone_comparison(
    runs: list[tuple[np.ndarray, np.ndarray, S.Recording]],
    *,
    target_half_device_px: float | None = None,
    limits: R.ZoneLimits | None = None,
) -> dict[str, Any]:
    """The per-zone gate, on the frames both models predicted.

    ``runs`` is one (old predictions, new predictions, recording) per
    recording. Pairing is enforced WITHIN each recording -- a frame counts only
    if both models predicted it -- and the segments are then pooled, so a zone
    covered once per session becomes a zone covered once per session per
    recording. That is what moves a zone from UNVERIFIED to measured without
    ever inventing a target that was not shown.
    """

    old_segments: list[R.SegmentResult] = []
    new_segments: list[R.SegmentResult] = []
    n_paired = 0
    scales = set()
    for old, new, rec in runs:
        eligible = FIT.eligible_rows(rec)
        both = eligible & np.all(np.isfinite(old), axis=1) & np.all(np.isfinite(new), axis=1)
        n_paired += int(both.sum())
        scales.add(device_per_logical(rec))
        tag = f"r{rec.round_id}:"
        for seg in paired_segments(old, rec, both):
            seg.target_name = tag + seg.target_name
            old_segments.append(seg)
        for seg in paired_segments(new, rec, both):
            seg.target_name = tag + seg.target_name
            new_segments.append(seg)
    if len(scales) != 1:
        raise ValueError(f"recordings disagree on the pixel scale: {sorted(scales)}")
    scale = scales.pop()
    half_logical = None if target_half_device_px is None else target_half_device_px / scale
    before = R.by_band_zone(old_segments, limits=limits)
    after = R.by_band_zone(new_segments, limits=limits)
    return {
        "n_recordings": len(runs),
        "n_paired": n_paired,
        "units": f"logical px; device px = logical x {scale:g}",
        "target_half_logical_px": half_logical,
        "before": before,
        "after": after,
        "gate": R.compare_band_zones(before, after, limits=limits, target_half_px=half_logical),
    }


def format_zone_comparison(result: dict[str, Any]) -> str:
    lines = [
        "",
        f"## By band zone (paired, {result['n_paired']} frames over "
        f"{result['n_recordings']} recording(s); {result['units']})",
        "",
        "### old",
        *R.format_band_zones(result["before"]),
        "",
        "### new",
        *R.format_band_zones(result["after"]),
        "",
        "### gate",
    ]
    gate = result["gate"]
    for row in gate["zones"]:
        extra = "; ".join(row.get("reasons", [])) or row.get("why", "")
        lines.append(f"- zone {row['zone']} ({row.get('label')}): **{row['verdict']}** {extra}")
    lines.append("")
    lines.append(
        f"regressed: {gate['regressed'] or 'none'}   unverified: {gate['unverified'] or 'none'}   "
        f"**adopt: {gate['adopt']}**"
    )
    lines.append(f"({gate['note']})")
    return "\n".join(lines)


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
    parser.add_argument(
        "--by-zone",
        action="store_true",
        help="also break the paired comparison down by band zone and run the per-zone "
        "regression gate. A zone that got worse blocks adoption even when the average "
        "improved; a zone with too little data is UNVERIFIED and never a pass.",
    )
    parser.add_argument(
        "--zone-recordings",
        type=Path,
        nargs="*",
        default=(),
        help="more recordings to pool into the --by-zone comparison, each paired frame by "
        "frame. The main --recording report above is unchanged.",
    )
    parser.add_argument(
        "--target-half-device-px",
        type=float,
        default=None,
        help="half-size, in DEVICE px, of the smallest target the zones must support. A zone "
        "whose frame-level P90 crosses it has stopped being usable whatever the percentages say.",
    )
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
    if args.by_zone:
        runs = {"raw": [(old["_raw"], new["_raw"], rec)],
                "filtered": [(old["_filtered"], new["_filtered"], rec)]}
        for extra in args.zone_recordings:
            if extra.resolve() == args.recording.resolve():
                continue
            other = S.Recording.load(str(extra), args.protocol)
            other_rig = C.RigGeometry.from_dict(other.meta["rig"])
            o = score_one(args.old, other, other_rig, settings)
            n = score_one(args.new, other, other_rig, settings)
            runs["raw"].append((o["_raw"], n["_raw"], other))
            runs["filtered"].append((o["_filtered"], n["_filtered"], other))
        result["by_zone"] = {
            band: zone_comparison(items, target_half_device_px=args.target_half_device_px)
            for band, items in runs.items()
        }
        for band, zones in result["by_zone"].items():
            print(f"\n# {band}")
            print(format_zone_comparison(zones))
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
