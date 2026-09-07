"""Score the 24-inch sessions against the gates fixed in PROTOCOL_24IN.md.

The gates live in code so the verdict is computed, not argued after the fact.
Every threshold carries its source, and a threshold with no measurement behind
it reports UNKNOWN rather than passing by default.

Angular units are primary here. A pixel threshold silently changes meaning
when the screen does, which is exactly the mistake this milestone exists to
avoid; the specification's goal is written in degrees and so are the gates.

Usage:
    python gf_session.py report --rounds recordings/s1 recordings/s2 recordings/s3 \
        --viewing-distance-cm 73 --out results/24in
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_fit as F  # noqa: E402
import gf_schema as S  # noqa: E402
import gf_select as SEL  # noqa: E402

PROTOCOL_VERSION = "24in-1"

# --- Gate thresholds, from PROTOCOL_24IN.md section 6 ------------------------
GATE2_MAX_SESSION_MEDIAN_DEG = 3.0
GATE2_MAX_MEDIAN_SPREAD_DEG = 1.5
GATE2_MIN_AVAILABILITY = 0.95
GATE3_MAX_MEAN_DEG = 2.0
GATE3_MAX_P90_DEG = 4.0
GATE3_MAX_REGION_MEDIAN_DEG = 3.0
GATE3_MAX_OFFSCREEN = 0.02
GATE3_MAX_MOVE_DEGRADATION_DEG = 1.0
GATE3_MIN_FPS = 25.0
GATE4_MAX_MEAN_DEG = 1.5  # specification 3.2 / 12
GATE4_MIN_FPS = 30.0
GATE4_MAX_LATENCY_P95_MS = 50.0

UNKNOWN = "UNKNOWN"


@dataclass
class GateResult:
    name: str
    status: str  # "PASS" | "FAIL" | "UNKNOWN"
    checks: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "checks": self.checks, "note": self.note}


def _check(label: str, observed: Any, limit: Any, ok: bool | None, unit: str, source: str) -> dict[str, Any]:
    return {
        "check": label,
        "observed": observed,
        "limit": limit,
        "unit": unit,
        "source": source,
        "status": UNKNOWN if ok is None else ("PASS" if ok else "FAIL"),
    }


def _combine(checks: Sequence[Mapping[str, Any]]) -> str:
    """A gate fails on any failure, and is UNKNOWN if anything is unmeasured.

    Absence of evidence never passes: an unmeasured check leaves the gate
    undecided rather than quietly satisfied.
    """

    if any(c["status"] == "FAIL" for c in checks):
        return "FAIL"
    if any(c["status"] == UNKNOWN for c in checks):
        return UNKNOWN
    return "PASS" if checks else UNKNOWN


@dataclass
class SessionScore:
    """One session's measured behaviour, in degrees where geometry allows."""

    round_id: str
    protocol: str
    n_eligible: int
    n_valid: int
    availability: float
    fps_median: float | None
    offscreen_rate: float
    mean_deg: float | None
    median_deg: float | None
    p90_deg: float | None
    p95_deg: float | None
    p99_deg: float | None
    max_deg: float | None
    median_px: float
    p90_px: float
    by_region: dict[str, dict[str, float]] = field(default_factory=dict)
    by_target: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        return d


def score_protocol(
    round_dir: Path,
    protocol: str,
    model: F.FittedModel,
    rig: C.RigGeometry,
    geometry: GD.ViewingGeometry,
    head_names: Sequence[str] = (),
) -> SessionScore | None:
    """Score one protocol, in pixels and in degrees, overall and by region."""

    path = round_dir / f"{protocol}.npz"
    if not path.exists():
        return None
    rec = S.Recording.load(round_dir, protocol)
    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])

    eligible = rec.rows_eligible()
    pred = np.full((rec.n_rows, 2), np.nan)
    mask, _ = F.scoring_rows(rec, head_names)
    if np.any(mask):
        pred[mask] = model.predict_norm(F.design_matrix(rec, mask, head_names), rig)

    p, t = pred[eligible], rec.target_xy[eligible]
    tid = rec.target_id[eligible]
    valid = np.all(np.isfinite(p), axis=1)
    n_eligible = int(p.shape[0])
    n_valid = int(valid.sum())

    dx = (p[:, 0] - t[:, 0]) * (width - 1)
    dy = (p[:, 1] - t[:, 1]) * (height - 1)
    euclid = np.hypot(dx, dy)

    # Degrees from the measured geometry. Small-angle at the centre understates
    # the edges, so the angle is computed from the actual displacement in cm.
    cm_x = dx * (geometry.screen_w_cm / width)
    cm_y = dy * (geometry.screen_h_cm / height)
    cm = np.hypot(cm_x, cm_y)
    deg = np.degrees(np.arctan2(cm, geometry.viewing_distance_cm))

    offscreen = np.zeros(n_eligible, dtype=bool)
    offscreen[valid] = (
        (p[valid, 0] < 0) | (p[valid, 0] > 1) | (p[valid, 1] < 0) | (p[valid, 1] > 1)
    )

    def q(values: np.ndarray, pct: float) -> float | None:
        v = values[valid]
        return float(np.percentile(v, pct)) if v.size else None

    regions: dict[str, dict[str, float]] = {}
    by_target: list[dict[str, Any]] = []
    meta_by_id = {int(x["index"]): x for x in rec.targets}
    for target_id in sorted({int(v) for v in tid}):
        rows = (tid == target_id) & valid
        meta = meta_by_id.get(target_id, {})
        region = meta.get("region") or GD.region_of(
            float(meta.get("screen_position", {}).get("x", 0.5)),
            float(meta.get("screen_position", {}).get("y", 0.5)),
        )
        entry = {
            "target_id": target_id,
            "name": meta.get("name", f"T{target_id}"),
            "region": region,
            "n": int(rows.sum()),
            "median_px": float(np.median(euclid[rows])) if rows.any() else None,
            "median_deg": float(np.median(deg[rows])) if rows.any() else None,
            "mean_deg": float(np.mean(deg[rows])) if rows.any() else None,
        }
        by_target.append(entry)
        if rows.any():
            regions.setdefault(region, {"medians": [], "means": []})
            regions[region]["medians"].append(entry["median_deg"])
            regions[region]["means"].append(entry["mean_deg"])

    by_region = {
        r: {
            "median_deg": float(np.median(v["medians"])),
            "mean_deg": float(np.mean(v["means"])),
            "n_targets": len(v["medians"]),
        }
        for r, v in regions.items()
    }

    integrity = rec.meta.get("integrity", {})
    return SessionScore(
        round_id=round_dir.name,
        protocol=protocol,
        n_eligible=n_eligible,
        n_valid=n_valid,
        availability=(n_valid / n_eligible) if n_eligible else 0.0,
        fps_median=integrity.get("fps_median"),
        offscreen_rate=float(offscreen.sum()) / n_eligible if n_eligible else 0.0,
        mean_deg=float(np.mean(deg[valid])) if n_valid else None,
        median_deg=q(deg, 50),
        p90_deg=q(deg, 90),
        p95_deg=q(deg, 95),
        p99_deg=q(deg, 99),
        max_deg=float(np.max(deg[valid])) if n_valid else None,
        median_px=q(euclid, 50) or float("nan"),
        p90_px=q(euclid, 90) or float("nan"),
        by_region=by_region,
        by_target=by_target,
    )


# --- the four gates ----------------------------------------------------------


def gate1_calibration_safety(outcomes: Sequence[SEL.SelectionOutcome]) -> GateResult:
    checks = []
    for i, o in enumerate(outcomes, 1):
        checks.append(
            _check(f"session {i}: a candidate passed screening", not o.refused, True,
                   not o.refused, "boolean", "gf_select.py thresholds")
        )
    note = "Screening runs on TUNE only, so the test set never influences the choice."
    if any(o.refused for o in outcomes):
        note += " At least one session produced NO ACCEPTABLE CALIBRATION."
    return GateResult("Gate 1 - calibration safety and numerical stability", _combine(checks), checks, note)


def gate2_repeatability(full: Sequence[SessionScore], gate1: GateResult) -> GateResult:
    checks = [_check("Gate 1 passed in every session", gate1.status, "PASS",
                     gate1.status == "PASS" if gate1.status != UNKNOWN else None, "status", "Gate 1")]
    medians = [s.median_deg for s in full if s.median_deg is not None]
    for s in full:
        checks.append(
            _check(f"{s.round_id}: median error", s.median_deg, GATE2_MAX_SESSION_MEDIAN_DEG,
                   None if s.median_deg is None else s.median_deg <= GATE2_MAX_SESSION_MEDIAN_DEG,
                   "deg", "proposed")
        )
        checks.append(
            _check(f"{s.round_id}: availability", s.availability, GATE2_MIN_AVAILABILITY,
                   s.availability >= GATE2_MIN_AVAILABILITY, "fraction", "targets document G4")
        )
    if len(medians) >= 2:
        spread = max(medians) - min(medians)
        checks.append(_check("spread of session medians", spread, GATE2_MAX_MEDIAN_SPREAD_DEG,
                             spread <= GATE2_MAX_MEDIAN_SPREAD_DEG, "deg", "proposed"))
    else:
        checks.append(_check("spread of session medians", None, GATE2_MAX_MEDIAN_SPREAD_DEG, None, "deg", "proposed"))
    n = len(full)
    if n < 3:
        checks.append(_check("three independent sessions recorded", n, 3, None, "sessions", "protocol section 4"))
    return GateResult("Gate 2 - repeatability across three sessions", _combine(checks), checks,
                      f"{n} session(s) supplied.")


def gate3_interaction_readiness(full: Sequence[SessionScore], move: Sequence[SessionScore]) -> GateResult:
    checks = []
    means = [s.mean_deg for s in full if s.mean_deg is not None]
    p90s = [s.p90_deg for s in full if s.p90_deg is not None]
    pooled_mean = float(np.mean(means)) if means else None
    pooled_p90 = float(np.max(p90s)) if p90s else None
    checks.append(_check("pooled mean error (FULL)", pooled_mean, GATE3_MAX_MEAN_DEG,
                         None if pooled_mean is None else pooled_mean <= GATE3_MAX_MEAN_DEG, "deg", "proposed"))
    checks.append(_check("worst session P90 (FULL)", pooled_p90, GATE3_MAX_P90_DEG,
                         None if pooled_p90 is None else pooled_p90 <= GATE3_MAX_P90_DEG, "deg", "proposed"))
    for region in ("centre", "edge-h", "edge-v", "corner"):
        values = [s.by_region[region]["median_deg"] for s in full if region in s.by_region]
        worst = max(values) if values else None
        checks.append(_check(f"{region}: worst session median", worst, GATE3_MAX_REGION_MEDIAN_DEG,
                             None if worst is None else worst <= GATE3_MAX_REGION_MEDIAN_DEG, "deg", "proposed"))
    off = max((s.offscreen_rate for s in full), default=None)
    checks.append(_check("off-screen rate", off, GATE3_MAX_OFFSCREEN,
                         None if off is None else off <= GATE3_MAX_OFFSCREEN, "fraction", "proposed"))
    fps = [s.fps_median for s in full if s.fps_median is not None]
    worst_fps = min(fps) if fps else None
    checks.append(_check("frame rate", worst_fps, GATE3_MIN_FPS,
                         None if worst_fps is None else worst_fps >= GATE3_MIN_FPS, "per second", "proposed"))
    if move and full:
        fm = [s.median_deg for s in full if s.median_deg is not None]
        mm = [s.median_deg for s in move if s.median_deg is not None]
        if fm and mm:
            degradation = float(np.median(mm) - np.median(fm))
            checks.append(_check("MOVE degradation vs FULL", degradation, GATE3_MAX_MOVE_DEGRADATION_DEG,
                                 degradation <= GATE3_MAX_MOVE_DEGRADATION_DEG, "deg", "proposed"))
    else:
        checks.append(_check("MOVE degradation vs FULL", None, GATE3_MAX_MOVE_DEGRADATION_DEG, None, "deg", "proposed"))
    return GateResult("Gate 3 - readiness for a supervised interaction trial", _combine(checks), checks,
                      "Large-target trials are informative below these limits even though they miss the product goal.")


def gate4_specification(full: Sequence[SessionScore], latency_p95_ms: float | None = None) -> GateResult:
    checks = []
    for s in full:
        checks.append(_check(f"{s.round_id}: mean error", s.mean_deg, GATE4_MAX_MEAN_DEG,
                             None if s.mean_deg is None else s.mean_deg <= GATE4_MAX_MEAN_DEG,
                             "deg", "specification 3.2 / 12"))
    for region in ("centre", "edge-h", "edge-v", "corner"):
        values = [s.by_region[region]["mean_deg"] for s in full if region in s.by_region]
        worst = max(values) if values else None
        checks.append(_check(f"{region}: worst session mean", worst, GATE4_MAX_MEAN_DEG,
                             None if worst is None else worst <= GATE4_MAX_MEAN_DEG, "deg",
                             "proposed reading of 'across the screen'"))
    fps = [s.fps_median for s in full if s.fps_median is not None]
    worst_fps = min(fps) if fps else None
    checks.append(_check("frame rate", worst_fps, GATE4_MIN_FPS,
                         None if worst_fps is None else worst_fps >= GATE4_MIN_FPS, "per second", "specification 3.2"))
    checks.append(_check("latency P95", latency_p95_ms, GATE4_MAX_LATENCY_P95_MS,
                         None if latency_p95_ms is None else latency_p95_ms <= GATE4_MAX_LATENCY_P95_MS,
                         "ms", "specification 3.2"))
    return GateResult("Gate 4 - specification accuracy target", _combine(checks), checks,
                      "Latency is NOT MEASURABLE on this hardware: the library timestamps a frame on "
                      "receipt, not exposure. Gate 4 therefore cannot be fully closed here.")


def evaluate(
    full: Sequence[SessionScore],
    move: Sequence[SessionScore],
    outcomes: Sequence[SEL.SelectionOutcome],
    latency_p95_ms: float | None = None,
) -> dict[str, Any]:
    g1 = gate1_calibration_safety(outcomes)
    g2 = gate2_repeatability(full, g1)
    g3 = gate3_interaction_readiness(full, move)
    g4 = gate4_specification(full, latency_p95_ms)
    gates = [g1, g2, g3, g4]
    statuses = {g.name: g.status for g in gates}
    if g1.status == "FAIL":
        decision = "NO ACCEPTABLE CALIBRATION - report and stop"
    elif all(g.status == "PASS" for g in gates):
        decision = "PASS - meets the specification on this monitor; proceed to the interaction trial"
    elif g1.status == "PASS" and g2.status == "PASS" and g3.status == "PASS":
        decision = "PARTIAL - a supervised interaction trial is justified; the accuracy goal is not met"
    elif g1.status == "PASS" and g2.status == "PASS":
        decision = "PARTIAL - calibration is stable but not accurate enough for interaction"
    elif UNKNOWN in statuses.values():
        decision = "NOT MEASURED - not enough evidence to decide"
    else:
        decision = "FAIL"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "gates": [g.to_dict() for g in gates],
        "decision": decision,
        "sessions_full": [s.to_dict() for s in full],
        "sessions_move": [s.to_dict() for s in move],
    }


def format_report(result: Mapping[str, Any]) -> str:
    lines = [f"# 24-inch full-screen sessions ({result['protocol_version']})", ""]
    lines.append(f"**Decision: {result['decision']}**")
    lines.append("")
    full = result["sessions_full"]
    if full:
        lines.append("## Sessions (FULL, neutral posture)\n")
        lines.append("| session | n | avail | fps | mean deg | median deg | P90 | P95 | P99 | max | off-screen |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in full:
            f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
            lines.append(
                f"| {s['round_id']} | {s['n_eligible']} | {s['availability']:.1%} | {f(s['fps_median'],1)} | "
                f"{f(s['mean_deg'])} | {f(s['median_deg'])} | {f(s['p90_deg'])} | {f(s['p95_deg'])} | "
                f"{f(s['p99_deg'])} | {f(s['max_deg'])} | {s['offscreen_rate']:.2%} |"
            )
        lines.append("")
        lines.append("## By screen region (median degrees)\n")
        regions = sorted({r for s in full for r in s["by_region"]})
        lines.append("| session | " + " | ".join(regions) + " |")
        lines.append("|---" * (len(regions) + 1) + "|")
        for s in full:
            cells = [f"{s['by_region'][r]['median_deg']:.2f}" if r in s["by_region"] else "n/a" for r in regions]
            lines.append(f"| {s['round_id']} | " + " | ".join(cells) + " |")
        lines.append("")
    if result["sessions_move"]:
        lines.append("## MOVE (natural head movement), reported separately\n")
        lines.append("| session | mean deg | median deg | P90 |")
        lines.append("|---|---|---|---|")
        for s in result["sessions_move"]:
            f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            lines.append(f"| {s['round_id']} | {f(s['mean_deg'])} | {f(s['median_deg'])} | {f(s['p90_deg'])} |")
        lines.append("")
    lines.append("## Gates\n")
    for g in result["gates"]:
        lines.append(f"### {g['name']} -- **{g['status']}**\n")
        if g["note"]:
            lines.append(f"{g['note']}\n")
        lines.append("| check | observed | limit | unit | source | status |")
        lines.append("|---|---|---|---|---|---|")
        for c in g["checks"]:
            obs = c["observed"]
            obs = "n/a" if obs is None else (f"{obs:.3f}" if isinstance(obs, float) else str(obs))
            lines.append(f"| {c['check']} | {obs} | {c['limit']} | {c['unit']} | {c['source']} | {c['status']} |")
        lines.append("")
    lines.append("Angles use the measured viewing distance and physical screen size; "
                 "an unmeasured check is UNKNOWN and never counts as a pass.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("report", help="score sessions against the preregistered gates")
    r.add_argument("--rounds", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--preset", default="central-band-svr")
    r.add_argument("--viewing-distance-cm", type=float, default=None)
    r.add_argument("--latency-p95-ms", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import gf_presets as PRE  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    full_scores, move_scores, outcomes = [], [], []
    for round_dir in args.rounds:
        rec_a = S.Recording.load(round_dir, "A")
        rig = C.RigGeometry.from_dict(rec_a.meta["rig"])
        declared = (rec_a.meta.get("setup") or {}).get("declared", {})
        viewing = args.viewing_distance_cm or declared.get("viewing_distance_cm")
        if viewing is None:
            print(f"{round_dir}: no viewing distance recorded or supplied; angles cannot be computed")
            return 2
        geom = GD.ViewingGeometry(rig.screen_w_cm, rig.screen_h_cm, rig.device_w_px, rig.device_h_px, float(viewing))
        mask, _ = F.training_rows(rec_a, ())
        X, Y = F.design_matrix(rec_a, mask, ()), rec_a.label_cm[mask]

        # Screen candidates on TUNE, exactly as phase 0 does.
        grid = PRE.sweep_with_presets()
        models = {}
        for cfg in grid:
            try:
                models[cfg.name] = F.FittedModel.fit(cfg, X, Y, rig=rig, train_meta={"protocol": "A"})
            except Exception:  # noqa: BLE001 - a broken candidate is screened out, not fatal
                continue
        rec_tune = S.Recording.load(round_dir, "TUNE")
        reports = F.screen_sweep(models, rec_tune, rig, (), thresholds=SEL.ScreeningThresholds())
        outcome = SEL.select(reports, preset=args.preset)
        outcomes.append(outcome)
        if outcome.refused:
            print(f"{round_dir}: NO ACCEPTABLE CALIBRATION -- {outcome.preset_note}")
            continue
        model = models[outcome.selected]
        for protocol, bucket in (("FULL", full_scores), ("MOVE", move_scores)):
            score = score_protocol(round_dir, protocol, model, rig, geom)
            if score is not None:
                bucket.append(score)

    result = evaluate(full_scores, move_scores, outcomes, args.latency_p95_ms)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sessions.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    text = format_report(result)
    (args.out / "sessions_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
