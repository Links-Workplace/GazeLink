"""Aggregation that cannot hide a failing axis, zone or condition (M0).

The measurement contract asks for three things this module provides, and each
exists because the obvious alternative conceals something:

* **Per zone, weighted by SEGMENT and not by frame.** A target the tracker
  followed for 200 frames must not outvote one it managed 20 frames on --
  otherwise the easy positions decide the average.
* **A full target x condition matrix, where a missing cell is not a pass.**
  An absent segment is reported as MISSING, never averaged away.
* **The two target sets apart, and inside the 16-point grid the targets that
  sit beside a calibration point apart from the rest.** Accuracy measured next
  to a training point is interpolation; folding it in flatters the result.

Every number is in logical pixels. Degrees appear only when an eye-position
estimate was available, and always carry the estimate's caveat.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_targets as T  # noqa: E402

MISSING = "MISSING"


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


@dataclass
class SegmentResult:
    """One presentation of one target under one condition.

    A segment is the unit of aggregation, so every presentation counts once
    regardless of how many frames it produced.
    """

    target_id: int
    target_name: str
    target_x: float
    target_y: float
    condition: str
    near_calibration: bool
    n_eligible: int
    n_valid: int
    median_dx_px: float
    median_dy_px: float
    median_euclid_px: float
    median_deg: float | None = None

    @property
    def coverage(self) -> float:
        return self.n_valid / self.n_eligible if self.n_eligible else float("nan")

    @property
    def zone(self) -> int:
        return T.zone_of(self.target_x, self.target_y)


def segments_from_rows(
    *,
    target_id: np.ndarray,
    target_xy: np.ndarray,
    eligible: np.ndarray,
    valid: np.ndarray,
    dx_px: np.ndarray,
    dy_px: np.ndarray,
    degrees: np.ndarray | None,
    targets_meta: Sequence[Mapping[str, Any]],
    condition: str = "static_neutral",
    presentation: np.ndarray | None = None,
) -> list[SegmentResult]:
    """Group per-frame rows into one result per target presentation.

    ``presentation`` distinguishes repeat showings of the same target; without
    it every appearance of a target id is treated as one segment.
    """

    meta_by_id = {int(t["index"]): t for t in targets_meta}
    keys = target_id if presentation is None else np.stack([target_id, presentation], axis=1)
    unique = (
        sorted({int(v) for v in target_id})
        if presentation is None
        else sorted({(int(a), int(b)) for a, b in keys})
    )
    out: list[SegmentResult] = []
    for key in unique:
        if presentation is None:
            tid, mask = int(key), (target_id == key)
        else:
            tid, pres = key
            mask = (target_id == tid) & (presentation == pres)
        rows = mask & eligible
        if not np.any(rows):
            continue
        good = rows & valid
        meta = meta_by_id.get(tid, {})
        pos = meta.get("screen_position", {})
        deg = None
        if degrees is not None and np.any(good) and np.any(np.isfinite(degrees[good])):
            deg = float(np.nanmedian(degrees[good]))
        out.append(
            SegmentResult(
                target_id=tid,
                target_name=str(meta.get("name", f"T{tid}")),
                target_x=float(pos.get("x", target_xy[rows][0, 0])),
                target_y=float(pos.get("y", target_xy[rows][0, 1])),
                condition=condition,
                near_calibration=bool(meta.get("near_calibration", False)),
                n_eligible=int(rows.sum()),
                n_valid=int(good.sum()),
                median_dx_px=float(np.median(dx_px[good])) if np.any(good) else float("nan"),
                median_dy_px=float(np.median(dy_px[good])) if np.any(good) else float("nan"),
                median_euclid_px=float(np.median(np.hypot(dx_px[good], dy_px[good]))) if np.any(good) else float("nan"),
                median_deg=deg,
            )
        )
    return out


def aggregate(segments: Sequence[SegmentResult]) -> dict[str, Any]:
    """Summary over segments, each weighted equally."""

    if not segments:
        return {"n_segments": 0, "note": "no segments"}
    euclid = np.array([s.median_euclid_px for s in segments])
    dx = np.array([s.median_dx_px for s in segments])
    dy = np.array([s.median_dy_px for s in segments])
    degrees = np.array([s.median_deg if s.median_deg is not None else np.nan for s in segments])
    eligible = int(sum(s.n_eligible for s in segments))
    valid = int(sum(s.n_valid for s in segments))
    return {
        "n_segments": len(segments),
        "n_eligible": eligible,
        "n_valid": valid,
        "coverage": valid / eligible if eligible else None,
        "euclid_px": _stats(euclid),
        "abs_dx_px": _stats(np.abs(dx)),
        "abs_dy_px": _stats(np.abs(dy)),
        "bias_dx_px": float(np.median(dx[np.isfinite(dx)])) if np.any(np.isfinite(dx)) else None,
        "bias_dy_px": float(np.median(dy[np.isfinite(dy)])) if np.any(np.isfinite(dy)) else None,
        "degrees": _stats(degrees) if np.any(np.isfinite(degrees)) else None,
        "weighting": "one vote per segment, not per frame",
    }


def by_zone(segments: Sequence[SegmentResult], n_zones: int = 16) -> list[dict[str, Any]]:
    """One row per screen zone; a zone with no segment is MISSING, not a pass."""

    grouped: dict[int, list[SegmentResult]] = {}
    for segment in segments:
        grouped.setdefault(segment.zone, []).append(segment)
    rows: list[dict[str, Any]] = []
    for zone in range(n_zones):
        members = grouped.get(zone, [])
        row: dict[str, Any] = {"zone": zone, "row": zone // 4, "col": zone % 4}
        if not members:
            row["status"] = MISSING
            rows.append(row)
            continue
        row["status"] = "measured"
        row.update(aggregate(members))
        row["targets"] = [s.target_name for s in members]
        rows.append(row)
    return rows


def target_condition_matrix(
    segments: Sequence[SegmentResult], conditions: Sequence[str], targets_meta: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Full target x condition grid. Every empty cell is reported explicitly."""

    grouped: dict[tuple[int, str], list[SegmentResult]] = {}
    for segment in segments:
        grouped.setdefault((segment.target_id, segment.condition), []).append(segment)
    cells: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for meta in targets_meta:
        tid = int(meta["index"])
        for condition in conditions:
            members = grouped.get((tid, condition), [])
            cell: dict[str, Any] = {
                "target_id": tid,
                "target_name": str(meta.get("name", f"T{tid}")),
                "condition": condition,
                "near_calibration": bool(meta.get("near_calibration", False)),
            }
            if members:
                cell["status"] = "measured"
                cell["n_presentations"] = len(members)
                cell.update(aggregate(members))
            else:
                cell["status"] = MISSING
                missing.append({"target_id": tid, "condition": condition})
            cells.append(cell)
    return {
        "cells": cells,
        "missing": missing,
        "complete": not missing,
        "note": "A cell without a segment is MISSING and cannot count as a pass.",
    }


def split_by_calibration_proximity(segments: Sequence[SegmentResult]) -> dict[str, Any]:
    """Held-out-like targets apart from those beside a calibration point."""

    far = [s for s in segments if not s.near_calibration]
    near = [s for s in segments if s.near_calibration]
    return {
        "far_from_calibration": aggregate(far),
        "near_calibration": aggregate(near),
        "note": (
            "near_calibration targets sit closer to a training point than the "
            "project's held-out threshold; accuracy there is interpolation and "
            "no generalisation claim may rest on it."
        ),
    }


def presentation_counts(segments: Sequence[SegmentResult], required: int = 2) -> dict[str, Any]:
    """Does every target x condition pair have the required repeat showings?"""

    counts: dict[tuple[int, str], int] = {}
    for segment in segments:
        counts[(segment.target_id, segment.condition)] = counts.get((segment.target_id, segment.condition), 0) + 1
    short = [
        {"target_id": tid, "condition": condition, "presentations": n}
        for (tid, condition), n in sorted(counts.items())
        if n < required
    ]
    return {"required": required, "under_required": short, "satisfied": not short}


def build_report(
    *,
    protocol: str,
    segments: Sequence[SegmentResult],
    targets_meta: Sequence[Mapping[str, Any]],
    conditions: Sequence[str] = ("static_neutral",),
    degrees_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "overall": aggregate(segments),
        "by_zone": by_zone(segments),
        "matrix": target_condition_matrix(segments, conditions, targets_meta),
        "by_calibration_proximity": split_by_calibration_proximity(segments),
        "presentations": presentation_counts(segments),
        "degrees": degrees_summary,
        "units": "logical pixels (analyze.py geometry); degrees only when an eye position was estimated",
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def format_report(report: Mapping[str, Any]) -> str:
    lines = [f"## {report['protocol']}", ""]
    overall = report["overall"]
    if overall.get("n_segments"):
        e, dx, dy = overall["euclid_px"], overall["abs_dx_px"], overall["abs_dy_px"]
        lines.append(
            f"segments {overall['n_segments']}, coverage {_fmt(None if overall['coverage'] is None else 100 * overall['coverage'])}% "
            f"({overall['n_valid']}/{overall['n_eligible']} frames)"
        )
        lines.append(
            f"euclid mean {_fmt(e['mean'])} / median {_fmt(e['median'])} / P90 {_fmt(e['p90'])} / max {_fmt(e['max'])} px"
        )
        lines.append(
            f"X |dx| median {_fmt(dx['median'])}, P90 {_fmt(dx['p90'])}, bias {_fmt(overall['bias_dx_px'])} | "
            f"Y |dy| median {_fmt(dy['median'])}, P90 {_fmt(dy['p90'])}, bias {_fmt(overall['bias_dy_px'])}"
        )
        if overall.get("degrees"):
            d = overall["degrees"]
            lines.append(f"degrees (ESTIMATED) mean {_fmt(d['mean'], 2)} / median {_fmt(d['median'], 2)} / P90 {_fmt(d['p90'], 2)}")
    else:
        lines.append("no segments")
    lines.append("")

    proximity = report["by_calibration_proximity"]
    for label, key in (("far from calibration", "far_from_calibration"), ("beside a calibration point", "near_calibration")):
        block = proximity[key]
        if block.get("n_segments"):
            lines.append(
                f"- {label}: {block['n_segments']} segments, euclid median {_fmt(block['euclid_px']['median'])} px, "
                f"|dy| median {_fmt(block['abs_dy_px']['median'])} px"
            )
        else:
            lines.append(f"- {label}: none")
    lines.append("")

    lines.append("### Screen zones (equal weight per segment)")
    lines.append("")
    lines.append("| zone | r,c | status | segments | euclid med | |dx| med | |dy| med | coverage |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in report["by_zone"]:
        if row["status"] == MISSING:
            lines.append(f"| {row['zone']} | {row['row']},{row['col']} | **MISSING** | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['zone']} | {row['row']},{row['col']} | measured | {row['n_segments']} | "
            f"{_fmt(row['euclid_px']['median'])} | {_fmt(row['abs_dx_px']['median'])} | "
            f"{_fmt(row['abs_dy_px']['median'])} | {_fmt(None if row['coverage'] is None else 100 * row['coverage'])}% |"
        )
    lines.append("")

    matrix = report["matrix"]
    if matrix["complete"]:
        lines.append(f"target x condition matrix: complete ({len(matrix['cells'])} cells)")
    else:
        lines.append(f"target x condition matrix: **INCOMPLETE** -- {len(matrix['missing'])} cell(s) missing:")
        for cell in matrix["missing"][:20]:
            lines.append(f"  - target {cell['target_id']} x {cell['condition']}")
        if len(matrix["missing"]) > 20:
            lines.append(f"  ... and {len(matrix['missing']) - 20} more")
    presentations = report["presentations"]
    if not presentations["satisfied"]:
        lines.append(
            f"presentations: **{len(presentations['under_required'])} pair(s) below the required "
            f"{presentations['required']} showings**"
        )
    if report.get("degrees"):
        lines.append("")
        lines.append(f"degrees: {report['degrees'].get('caveat', '')}")
    return "\n".join(lines) + "\n"


def save_report(path: Path, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)!r}")
