"""Replay logged dwell sessions through the real selection engine. No camera.

Why: offline gaze error looks acceptable while live selection is weak. Before
changing the model or the dwell, this says -- per trial -- what the logged
gaze actually did against the logged buttons, and how much of that a log can
support at all.

What it is:
  * The UNMODIFIED ``gf_dwell.DwellEngine``, driven with the logged time as
    its clock. No simulated dwell of its own, no OS input, no camera.
  * The requested target is used ONLY by the evaluator after the engine has
    run; the engine never sees it.

What it is not: a model of the person. A replay is a fixed path; it cannot say
how someone would have looked had the feedback been different. Replay input is
cut at the first logged row AFTER the logged activation (a sparse log can only
reach the dwell threshold on a later row than the live firing); with
engine_calls the cut is the logged time itself (+ AGREE_EPS_MS). An activation
agrees only if the same button fires with a lag in [-AGREE_EPS_MS, cut - logged]:
a replay that fires EARLIER than the live run does not agree. Path statistics
use nothing after the logged activation.

Inputs, best first:
  1. ``trial["engine_calls"]``: every input the live engine received (time
     to 1e-4 ms, points to 1e-8, so exact up to that rounding), as
     ``[t_ms, x|None, y|None, fresh, update_id|None]``. ALL calls are
     replayed, repeats of the same ``update_id`` included, with their original
     time and ``fresh``. De-duplication by ``update_id`` feeds sample counts
     only; durations are always measured in time, never in sample counts.
  2. ``trial["path"]`` (+ ``raw_path`` rows carrying ``update_id``): the older
     ~20 Hz log of FILTERED points that were present. It omits frames with no
     point, so the engine's resets on those frames are invisible. Such trials
     are replayed but flagged ``partial``, and anything a gap could change is
     classified ``undetermined_from_log`` rather than guessed.

Usage (from gazefollower/):
    python gf_selection_replay.py results/select_compare/20260915_073919.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_dwell as D  # noqa: E402

# A path row further than this multiple of the median row spacing from the
# previous one means frames without a point were left out of the log between
# them (the logger writes a present point at least every 50 ms).
GAP_FACTOR = 2.0
# Rounding slack between the logged activation time and the firing call's time.
AGREE_EPS_MS = 1.0

CLASSES = (
    "success",
    "wrong_activation",
    "never_entered",
    "single_short_entry",
    "reentry_no_complete",
    "entered_short_partial_log",
    "undetermined_from_log",
    "not_scored",
)
NO_SELECTION_MECHANISMS = (
    "never_entered",
    "single_short_entry",
    "reentry_no_complete",
    "entered_short_partial_log",
)


def cue_neighbour(buttons: list[D.Button]) -> str | None:
    """The button directly under a top-centre prompt: spans x=0.5, highest on screen.

    The practice prompt was drawn centred at the top (gf_record.prompt_layout
    "top"). A wrong activation on this button is CONSISTENT with reading the
    prompt; it is not proof of it.
    """

    under = [b for b in buttons if b.x0 <= 0.5 <= b.x1]
    return min(under, key=lambda b: b.y0).key if under else None


@dataclass(frozen=True)
class Call:
    """One input to ``DwellEngine.update``: time (ms), point or None, fresh."""

    t_ms: float
    point: tuple[float, float] | None
    fresh: bool = True
    update_id: int | None = None


@dataclass
class TrialReplay:
    index: int
    requested: str
    logged_activation: str | None
    logged_activation_ms: float | None
    replay_activation: str | None = None
    replay_activation_ms: float | None = None
    replay_lag_ms: float | None = None
    agrees: bool = False
    previous_requested: str | None = None
    wrong_on_previous_target: bool = False
    source: str = "path"
    partial: bool = True
    calls: int = 0
    unique_updates: int | None = None
    median_spacing_ms: float | None = None
    max_gap_ms: float | None = None
    gaps_over_threshold: int = 0
    frames: int = 0
    frames_without_point: int = 0
    first_entry_ms: float | None = None
    time_in_target_frac: float = 0.0
    longest_stay_ms: float = 0.0
    entries: int = 0
    target_switches: int = 0
    most_time_on: str | None = None
    most_time_frac: float = 0.0
    cue_neighbour_time_frac: float = 0.0
    wrong_on_cue_neighbour: bool = False
    paused: bool = False
    classification: str = "not_scored"
    reasons: list[str] = field(default_factory=list)


# --- reading logs -------------------------------------------------------------


def reports_in(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(label, dwell-practice report) pairs from a comparison or a single report."""

    if "blocks" in doc:
        return [
            (f"block{b['block']}:{b['arm']}:{b['report'].get('profile')}", b["report"])
            for b in doc["blocks"]
            if b.get("report")
        ]
    return [(f"single:{doc.get('profile')}", doc)]


def calls_for(trial: dict[str, Any]) -> tuple[list[Call], str]:
    """The engine inputs for one trial, and which log they came from."""

    if trial.get("engine_calls"):
        out = []
        for row in trial["engine_calls"]:
            t, x, y, fresh = row[0], row[1], row[2], bool(row[3])
            uid = row[4] if len(row) > 4 else None
            point = None if x is None or y is None else (float(x), float(y))
            out.append(Call(float(t), point, fresh, uid))
        return out, "engine_calls"
    ids: dict[float, int] = {}
    for row in trial.get("raw_path") or []:
        if len(row) >= 4 and row[3] is not None:
            ids[float(row[0])] = int(row[3])
    calls = []
    for row in trial.get("path") or []:
        t, x, y = float(row[0]), float(row[1]), float(row[2])
        calls.append(Call(t, (x, y), True, ids.get(t)))
    return calls, "path"


# --- the replay ---------------------------------------------------------------


def replay_calls(
    buttons: list[D.Button], dwell_ms: float, calls: Iterable[Call]
) -> tuple[str | None, float | None]:
    """First activation the real engine produces over these calls.

    A fresh engine per trial matches the live one only when the trial's
    neutral gate was reached: the gate's last call is a dead-space point,
    which clears current target, progress and latch. ``classify`` checks it.
    """

    engine = D.DwellEngine(buttons, D.DwellConfig(dwell_ms=dwell_ms))
    for c in calls:
        fired = engine.update(c.t_ms / 1000.0, c.point, fresh=c.fresh)
        if fired is not None:
            return fired.button, c.t_ms
    return None, None


def hovered(buttons: list[D.Button], c: Call) -> str | None:
    if c.point is None or not c.fresh:
        return None
    for b in buttons:
        if b.contains(c.point):
            return b.key
    return None


def describe(
    trial: dict[str, Any],
    buttons: list[D.Button],
    dwell_ms: float,
    cue_key: str | None = None,
    previous_requested: str | None = None,
) -> TrialReplay:
    r = TrialReplay(
        index=int(trial["index"]),
        requested=trial["requested"],
        logged_activation=trial.get("activated"),
        logged_activation_ms=trial.get("activated_at_ms"),
        frames=int(trial.get("frames", 0)),
        frames_without_point=int(trial.get("frames_without_point", 0)),
        previous_requested=previous_requested,
        paused=bool(trial.get("pauses")),
    )
    if not trial.get("scored", True) or trial.get("aborted"):
        r.reasons.append("trial not scored or aborted")
        return r
    calls, r.source = calls_for(trial)
    r.partial = r.source != "engine_calls"
    r.calls = len(calls)
    ids = [c.update_id for c in calls if c.update_id is not None]
    r.unique_updates = len(set(ids)) if ids else None
    if not calls:
        r.classification = "undetermined_from_log"
        r.reasons.append("no logged samples")
        return r

    cutoff = float("inf")
    if r.logged_activation_ms is not None:
        cutoff = r.logged_activation_ms + AGREE_EPS_MS
        if r.partial:
            later = [c.t_ms for c in calls if c.t_ms > r.logged_activation_ms + AGREE_EPS_MS]
            if later:
                cutoff = min(later)
    r.replay_activation, r.replay_activation_ms = replay_calls(
        buttons, dwell_ms, [c for c in calls if c.t_ms <= cutoff]
    )
    if r.replay_activation_ms is not None and r.logged_activation_ms is not None:
        r.replay_lag_ms = r.replay_activation_ms - r.logged_activation_ms
    if r.logged_activation is None:
        r.agrees = r.replay_activation is None
    else:
        r.agrees = (
            r.replay_activation == r.logged_activation
            and r.replay_lag_ms is not None
            and r.replay_lag_ms >= -AGREE_EPS_MS
        )

    # Described only up to the logged activation: the trial is over for the
    # person there, and what follows is not evidence about selection.
    end_ms = r.logged_activation_ms if r.logged_activation_ms is not None else float("inf")
    window = [c for c in calls if c.t_ms <= end_ms] or calls[:1]

    times = [c.t_ms for c in window]
    spacings = [b - a for a, b in zip(times, times[1:], strict=False)]
    if spacings:
        r.median_spacing_ms = statistics.median(spacings)
        r.max_gap_ms = max(spacings)
        r.gaps_over_threshold = sum(1 for s in spacings if s > GAP_FACTOR * r.median_spacing_ms)

    held: dict[str, float] = {}
    total = 0.0
    run_start = 0.0
    prev: str | None = None
    for i, c in enumerate(window):
        h = hovered(buttons, c)
        dt = (window[i + 1].t_ms - c.t_ms) if i + 1 < len(window) else 0.0
        total += dt
        if h is not None:
            held[h] = held.get(h, 0.0) + dt
            if prev is not None and h != prev:
                r.target_switches += 1
        if h == r.requested:
            if prev != r.requested:
                r.entries += 1
                run_start = c.t_ms
                if r.first_entry_ms is None:
                    r.first_entry_ms = c.t_ms
            # Held time as the engine measures it: from the first call on the
            # target to this one.
            r.longest_stay_ms = max(r.longest_stay_ms, c.t_ms - run_start)
        prev = h
    if total > 0:
        r.time_in_target_frac = held.get(r.requested, 0.0) / total
        if held:
            key, t = max(held.items(), key=lambda kv: kv[1])
            r.most_time_on, r.most_time_frac = key, t / total
        if cue_key is not None and cue_key != r.requested:
            r.cue_neighbour_time_frac = held.get(cue_key, 0.0) / total
    r.wrong_on_cue_neighbour = (
        cue_key is not None and r.logged_activation == cue_key and r.requested != cue_key
    )
    # The competing explanation for a wrong activation: the gaze still on the
    # previous trial's target (a single dead-space point opens the neutral gate).
    r.wrong_on_previous_target = (
        previous_requested is not None
        and r.logged_activation == previous_requested
        and r.requested != previous_requested
    )

    r.classification = classify(r, trial, dwell_ms)
    return r


def classify(r: TrialReplay, trial: dict[str, Any], dwell_ms: float) -> str:
    """Outcome, with no-selection split by mechanism only where the log supports it."""

    if trial.get("pauses"):
        r.reasons.append("a pause reset the engine and restarted the trial clock")
        return "undetermined_from_log"
    if not trial.get("neutral_reached", True):
        r.reasons.append("neutral start not reached: engine state at trial start unknown")
        return "undetermined_from_log"
    if r.logged_activation == r.requested:
        if not r.agrees:
            r.reasons.append("logged success the replay does not reproduce (log too sparse)")
        return "success"
    if r.logged_activation is not None:
        if not r.agrees:
            r.reasons.append("logged wrong activation the replay does not reproduce")
        return "wrong_activation"
    if not r.agrees:
        r.reasons.append(f"replay fired {r.replay_activation} but the live run did not")
        return "undetermined_from_log"
    resolution = (r.max_gap_ms or 0.0) if r.partial else 0.0
    if r.partial and r.frames_without_point:
        # The path log leaves out frames without a point, and each one resets
        # the live engine. Where they fell is unknown, so no mechanism.
        r.reasons.append(
            f"{r.frames_without_point} frame(s) without a point are not in the log; "
            "their resets cannot be placed"
        )
        return "undetermined_from_log"
    if r.partial and r.max_gap_ms is not None and r.max_gap_ms >= dwell_ms:
        r.reasons.append(f"a {r.max_gap_ms:.0f} ms gap is as long as the dwell")
        return "undetermined_from_log"
    if r.entries == 0:
        if r.partial:
            r.reasons.append("never LOGGED in target; brief unlogged entries are not ruled out")
        return "never_entered"
    if r.longest_stay_ms + 2.0 * resolution >= dwell_ms:
        r.reasons.append(
            f"longest stay {r.longest_stay_ms:.0f} ms is within log resolution of {dwell_ms:.0f} ms"
        )
        return "undetermined_from_log"
    if r.partial and r.gaps_over_threshold:
        r.reasons.append(
            f"{r.gaps_over_threshold} gap(s) may hide resets or longer stays; entries unreliable"
        )
        return "undetermined_from_log"
    if r.partial:
        # Between logged rows the display loop fed the engine unlogged points;
        # one of them outside the target resets progress. Entries are a lower
        # bound and a logged stay is neither bound (covered by the 2x gap
        # margin above), so re-entry vs single entry is not separable here --
        # only that no logged stay came within log resolution of the dwell.
        r.reasons.append("path log: entries are a lower bound; re-entry not separable")
        return "entered_short_partial_log"
    return "reentry_no_complete" if r.entries >= 2 else "single_short_entry"


# --- summary ------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def replay_report(report: dict[str, Any], label: str) -> dict[str, Any]:
    buttons = [D.Button(b["key"], b["x0"], b["y0"], b["x1"], b["y1"]) for b in report["buttons"]]
    dwell_ms = float(report["dwell_ms"])
    cue_key = cue_neighbour(buttons)
    raw_trials = report.get("trials", [])
    trials = [
        describe(t, buttons, dwell_ms, cue_key, raw_trials[i - 1]["requested"] if i else None)
        for i, t in enumerate(raw_trials)
    ]
    scored = [t for t in trials if t.classification != "not_scored"]
    comparable = [t for t in scored if not t.paused]
    counts = {c: sum(1 for t in trials if t.classification == c) for c in CLASSES}
    times = [
        float(t.logged_activation_ms)
        for t in scored
        if t.classification == "success" and t.logged_activation_ms is not None
    ]
    wrong_on: dict[str, int] = {}
    for t in scored:
        if t.classification == "wrong_activation" and t.logged_activation:
            wrong_on[t.logged_activation] = wrong_on.get(t.logged_activation, 0) + 1
    summ = report.get("summary", {})
    # Rest: each rest starts from engine.reset(), so a fresh engine per rest is
    # exact. Only possible when the harness logged rest_engine_calls.
    rest_replayed = None
    if report.get("rest_engine_calls"):
        rest_replayed = 0
        for rest in report["rest_engine_calls"]:
            engine = D.DwellEngine(buttons, D.DwellConfig(dwell_ms=dwell_ms))
            for c in calls_for({"engine_calls": rest["engine_calls"]})[0]:
                if engine.update(c.t_ms / 1000.0, c.point, fresh=c.fresh) is not None:
                    rest_replayed += 1
    idle = float(summ.get("idle_seconds") or 0.0)
    unintended = int(summ.get("unintended_activations") or 0)
    return {
        "label": label,
        "profile": report.get("profile"),
        "dwell_ms": dwell_ms,
        "scored": len(scored),
        "counts": counts,
        # Paused trials are left out: a pause resets the engine unlogged and
        # restarts the trial clock, so their replay is not comparable.
        "agreement": {
            "agree": sum(1 for t in comparable if t.agrees),
            "of": len(comparable),
            "paused_excluded": len(scored) - len(comparable),
            "partial_trials": sum(1 for t in comparable if t.partial),
            "activations_reproduced": sum(
                1 for t in comparable if t.logged_activation is not None and t.agrees
            ),
            "activations_logged": sum(1 for t in comparable if t.logged_activation is not None),
            "non_activations_reproduced": sum(
                1 for t in comparable if t.logged_activation is None and t.agrees
            ),
            "non_activations_logged": sum(1 for t in comparable if t.logged_activation is None),
            "replay_lag_ms": sorted(
                t.replay_lag_ms for t in comparable if t.replay_lag_ms is not None
            ),
        },
        "success_time_ms": {
            "n": len(times),
            "median": statistics.median(times) if times else None,
            "p90": percentile(times, 0.9),
            "denominator": "successful trials only; trial window open to activation",
        },
        "wrong_activation_targets": wrong_on,
        "cue_neighbour": cue_key,
        "wrong_on_cue_neighbour": sum(1 for t in scored if t.wrong_on_cue_neighbour),
        "wrong_on_previous_target": sum(1 for t in scored if t.wrong_on_previous_target),
        "wrong_on_both": sum(
            1 for t in scored if t.wrong_on_previous_target and t.wrong_on_cue_neighbour
        ),
        "rest": {
            "unintended": unintended,
            "idle_seconds": idle,
            "per_minute": unintended / (idle / 60.0) if idle else None,
            "targets": [u.get("key") for u in summ.get("unintended_detail", [])],
            "replayed_unintended": rest_replayed,
            "note": "configured rest length; replayed_unintended is None without rest_engine_calls",
        },
        "trials": [asdict(t) for t in trials],
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, dict[str, Any]] = {}
    for res in results:
        p = by_profile.setdefault(
            str(res["profile"]),
            {
                "counts": {c: 0 for c in CLASSES},
                "success_ms": [],
                "rest": 0,
                "rest_s": 0.0,
                "agree": 0,
                "of": 0,
                "wrong_on": {},
                "wrong_on_cue_neighbour": 0,
                "wrong_on_previous_target": 0,
                "wrong_on_both": 0,
                "act_rep": 0,
                "act_log": 0,
                "non_rep": 0,
                "non_log": 0,
                "max_lag_ms": 0.0,
                "reentry_class_reachable": False,
            },
        )
        for c, n in res["counts"].items():
            p["counts"][c] += n
        p["success_ms"] += [
            t["logged_activation_ms"] for t in res["trials"] if t["classification"] == "success"
        ]
        p["rest"] += res["rest"]["unintended"]
        p["rest_s"] += res["rest"]["idle_seconds"]
        p["wrong_on_cue_neighbour"] += res["wrong_on_cue_neighbour"]
        p["wrong_on_previous_target"] += res["wrong_on_previous_target"]
        p["wrong_on_both"] += res["wrong_on_both"]
        a = res["agreement"]
        p["act_rep"] += a["activations_reproduced"]
        p["act_log"] += a["activations_logged"]
        p["non_rep"] += a["non_activations_reproduced"]
        p["non_log"] += a["non_activations_logged"]
        p["max_lag_ms"] = max([p["max_lag_ms"], *a["replay_lag_ms"]])
        p["reentry_class_reachable"] |= any(
            not t["partial"] for t in res["trials"] if t["classification"] != "not_scored"
        )
        p["agree"] += res["agreement"]["agree"]
        p["of"] += res["agreement"]["of"]
        for k, n in res["wrong_activation_targets"].items():
            p["wrong_on"][k] = p["wrong_on"].get(k, 0) + n
    out = {}
    for name, p in by_profile.items():
        ms = p.pop("success_ms")
        out[name] = {
            **p,
            "success_median_ms": statistics.median(ms) if ms else None,
            "success_p90_ms": percentile(ms, 0.9),
            "no_selection_by_mechanism": sum(p["counts"][c] for c in NO_SELECTION_MECHANISMS),
            "rest_per_minute": p["rest"] / (p["rest_s"] / 60.0) if p["rest_s"] else None,
        }
    return out


def format_text(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = []
    for res in results:
        a = res["agreement"]
        lines.append(
            f"\n## {res['label']}  dwell {res['dwell_ms']:.0f} ms  replay agrees "
            f"{a['agree']}/{a['of']} (partial log on {a['partial_trials']})"
        )
        lines.append(
            "| # | requested | logged | replay | class | 1st entry ms | in-target | longest ms "
            "| entries | max gap ms | most time on | reasons |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in res["trials"]:
            lines.append(
                f"| {t['index']} | {t['requested']} | {t['logged_activation']} | "
                f"{t['replay_activation']} | {t['classification']} | {t['first_entry_ms']} | "
                f"{t['time_in_target_frac']:.2f} | {t['longest_stay_ms']:.0f} | {t['entries']} | "
                f"{t['max_gap_ms']} | {t['most_time_on']} {t['most_time_frac']:.2f} | "
                f"{'; '.join(t['reasons'])} |"
            )
        lines.append(f"rest: {res['rest']['unintended']} unintended on {res['rest']['targets']}")
    lines.append("\n## by profile")
    for name, p in summary.items():
        lines.append(
            f"- {name}: {p['counts']}; replay reproduces {p['act_rep']}/{p['act_log']} "
            f"activations (lag <= {p['max_lag_ms']:.0f} ms) and {p['non_rep']}/{p['non_log']} "
            f"non-activations; wrong on {p['wrong_on']} ({p['wrong_on_cue_neighbour']} on the "
            f"button under the top prompt, {p['wrong_on_previous_target']} on the previous "
            f"trial's target, {p['wrong_on_both']} on both; carry-over across block starts "
            f"not counted); re-entry class reachable: {p['reentry_class_reachable']}; "
            f"success median {p['success_median_ms']} ms, p90 "
            f"{p['success_p90_ms']} ms; rest activations/min {p['rest_per_minute']}"
        )
    lines.append(
        "\nFixed-path replay. Logs without engine_calls omit frames with no point, so "
        "'undetermined_from_log' is a real outcome. No FPS or latency is reported: the logs "
        "carry no capture timestamps."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay logged dwell sessions. No camera, no OS input."
    )
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "results" / "selection_diag"
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for log in args.logs:
        doc = json.loads(log.read_text(encoding="utf-8"))
        results = [replay_report(rep, label) for label, rep in reports_in(doc)]
        summary = summarise(results)
        payload = {"source": str(log), "results": results, "summary": summary, "os_input": "none"}
        target = args.out / f"{log.stem}.replay.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(format_text(results, summary))
        print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
