"""Live selection, two profiles, same task: does a model difference show up
where it matters -- in what a person can select?

Error tables say where the gaze estimate is; they cannot say whether a button
gets selected. This runs the dwell practice (``gf_dwell_practice``) on the
nine band zones, alternating two profiles in ABBA order so that learning or
fatigue over the session does not favour whichever profile ran first:

    block 1  A    block 2  B    block 3  B    block 4  A

A is the reference (default: the ACTIVE profile), B the candidate. The
candidate runs as its own profile; nothing here activates, edits or replaces
either one.

Every trial that was scored stays in the denominator: a timeout is a
non-selection, not a trial that did not happen. Per-zone counts are small by
design -- this is a short comparison -- and the report says so rather than
turning four trials into a percentage that looks like a measurement.

No OS input. The targets live inside this window only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_dwell as D  # noqa: E402
import gf_profile as PROF  # noqa: E402

ABBA = ("A", "B", "B", "A")


def make_trial_profile(
    base: PROF.Profile,
    *,
    name: str,
    model_dir: str,
    note: str,
    root: Path | None = None,
    require_model: bool = True,
    capture_pipeline: str | None = None,
) -> Path:
    """Save ``base`` with a different model, and change nothing else.

    Everything except the model is carried over as-is -- filter, cursor,
    gesture rule, rig, screen, x range -- so that a difference in the live
    comparison can only come from the model. The new profile is saved but
    NOT activated, and an existing profile of the same name is never replaced.
    """

    trial = replace(
        base,
        name=name,
        model_dir=model_dir,
        scores={"trial_of": base.name, "note": note},
        capture_pipeline=capture_pipeline or base.capture_pipeline,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    if require_model and not trial.model_path().exists():
        raise FileNotFoundError(f"model directory does not exist: {trial.model_path()}")
    if require_model:
        PROF.require_capture_match(trial)  # never save a profile that cannot run
    return PROF.save(trial, root=root)


def block_plan(reference: str, candidate: str, *, seed: int) -> list[dict[str, Any]]:
    """ABBA, each block with its own shuffled order (seeded, reproducible)."""

    # Seeds s, s, s+1, s+1 over A B B A: each arm runs the SAME two orders, so a
    # difference cannot come from one arm drawing an easier sequence.
    names = {"A": reference, "B": candidate}
    seeds = (seed, seed, seed + 1, seed + 1)
    return [
        {"block": i + 1, "arm": arm, "profile": names[arm], "seed": seeds[i]}
        for i, arm in enumerate(ABBA)
    ]


def zone_of_key(key: str) -> int:
    return D.ZONE_KEYS.index(key)


def per_zone(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Counts per zone over SCORED trials. Timeouts stay in the denominator."""

    rows = []
    for zone, key in enumerate(D.ZONE_KEYS):
        mine = [t for t in trials if t["requested"] == key and t.get("scored", True)]
        here = [t for t in trials if t["requested"] == key]
        correct = [t for t in mine if t.get("activated") == key]
        wrong = [t for t in mine if t.get("activated") not in (None, key)]
        times = [t["activated_at_ms"] for t in correct if t.get("activated_at_ms") is not None]
        rows.append(
            {
                "zone": zone,
                "key": key,
                "trials": len(mine),
                "correct": len(correct),
                "wrong": len(wrong),
                "no_selection": len(mine) - len(correct) - len(wrong),
                "skipped_no_neutral": sum(
                    1 for t in here if not t.get("scored", True) and not t.get("aborted")
                ),
                "aborted": sum(1 for t in here if t.get("aborted")),
                "median_time_ms": statistics.median(times) if times else None,
            }
        )
    return rows


def summarise(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool each arm's blocks; one table per arm, and the difference per zone."""

    out: dict[str, Any] = {}
    for arm in ("A", "B"):
        mine = [b for b in blocks if b["arm"] == arm]
        reports = [b["report"] for b in mine if b.get("report")]
        trials = [t for r in reports for t in r.get("trials", [])]
        zones = per_zone(trials)
        scored = sum(z["trials"] for z in zones)
        correct = sum(z["correct"] for z in zones)
        times = [
            t["activated_at_ms"]
            for t in trials
            if t.get("scored", True)
            and t.get("activated") == t["requested"]
            and t.get("activated_at_ms") is not None
        ]
        out[arm] = {
            "profile": mine[0]["profile"] if mine else None,
            "blocks": len(reports),
            "aborted_blocks": sum(1 for r in reports if r.get("aborted")),
            "scored_trials": scored,
            "correct": correct,
            "wrong": sum(z["wrong"] for z in zones),
            "no_selection": sum(z["no_selection"] for z in zones),
            "skipped_no_neutral": sum(z["skipped_no_neutral"] for z in zones),
            "unintended_rest": sum(
                r.get("summary", {}).get("unintended_activations", 0) for r in reports
            ),
            "median_time_ms": statistics.median(times) if times else None,
            "zones": zones,
        }
    completed = [b for b in blocks if b.get("report") and not b["report"].get("aborted")]
    out["complete"] = len(completed) == len(ABBA)
    out["note"] = (
        (
            ""
            if out["complete"]
            else f"INCOMPLETE ABBA ({len(completed)}/{len(ABBA)} blocks finished): NOT comparable. "
        )
        + "Preliminary. A few trials per zone cannot separate 95% from 80%; this shows whether a "
        "difference is visible in selection at all, and where. Adoption is not decided here."
    )
    return out


def format_summary(summary: dict[str, Any]) -> str:
    a, b = summary["A"], summary["B"]

    def frac(z: dict[str, Any]) -> str:
        return f"{z['correct']}/{z['trials']}" if z["trials"] else "--"

    def ms(v: float | None) -> str:
        return "--" if v is None else f"{v:.0f}"

    lines = [
        f"A = {a['profile']}   B = {b['profile']}   (ABBA)",
        "",
        "| zone | A correct | A wrong | A none | A skip | A ms | B correct | B wrong | B none | B skip | B ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for za, zb in zip(a["zones"], b["zones"], strict=True):
        lines.append(
            f"| {za['key']} | {frac(za)} | {za['wrong']} | {za['no_selection']} | "
            f"{za['skipped_no_neutral']} | {ms(za['median_time_ms'])} | {frac(zb)} | {zb['wrong']} | "
            f"{zb['no_selection']} | {zb['skipped_no_neutral']} | {ms(zb['median_time_ms'])} |"
        )
    for arm, s in (("A", a), ("B", b)):
        lines.append(
            f"\n{arm}: {s['correct']}/{s['scored_trials']} correct, {s['wrong']} wrong target, "
            f"{s['no_selection']} no selection, {s['skipped_no_neutral']} skipped, "
            f"{s['unintended_rest']} unintended while resting, median {ms(s['median_time_ms'])} ms"
        )
    lines.append("\n" + summary["note"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-profile", help="save a trial profile; does not activate it")
    make.add_argument("--base", default=None, help="profile to copy (default: the active one)")
    make.add_argument("--name", required=True)
    make.add_argument("--model-dir", required=True, help="relative to gazefollower/")
    make.add_argument("--note", default="")
    make.add_argument(
        "--capture-pipeline",
        choices=PROF.KNOWN_CAPTURES,
        default=None,
        help="capture the model was trained on (default: the base profile's)",
    )
    run = sub.add_parser("run", help="ABBA live selection comparison")
    run.add_argument("--reference", default=None, help="A (default: the active profile)")
    run.add_argument("--candidate", required=True, help="B")
    run.add_argument("--trials-per-block", type=int, default=18)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--dwell-ms", type=float, default=900.0)
    run.add_argument("--trial-timeout-s", type=float, default=8.0)
    run.add_argument("--idle-seconds", type=float, default=20.0)
    run.add_argument("--out", type=Path, default=None)
    run.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="an earlier result file: blocks it finished are kept and only the rest are run. "
        "A block is kept only if its profile, arm, seed and trial count match.",
    )
    block = sub.add_parser("run-block", help=argparse.SUPPRESS)
    block.add_argument("--profile", required=True)
    block.add_argument("--trials", type=int, required=True)
    block.add_argument("--seed", type=int, required=True)
    block.add_argument("--dwell-ms", type=float, required=True)
    block.add_argument("--trial-timeout-s", type=float, required=True)
    block.add_argument("--idle-seconds", type=float, required=True)
    block.add_argument("--report", type=Path, required=True)
    return parser


def resumable_blocks(
    path: Path,
    plan: list[dict[str, Any]],
    trials_per_block: int,
    *,
    settings: dict[str, float],
    models: dict[str, str],
) -> dict[int, dict[str, Any]]:
    """Finished blocks from an earlier result file, keyed by block number.

    Kept only when arm, profile, seed, trial count, the task settings (dwell,
    timeout, rest) AND the model the profile points at all match, and the
    block finished without a stop. Anything else is run again rather than
    trusted: a block run at another dwell, or on a model the name no longer
    points at, would confound the comparison without saying so.
    """

    old = json.loads(Path(path).read_text(encoding="utf-8"))
    wanted = {p["block"]: p for p in plan}
    kept: dict[int, dict[str, Any]] = {}
    for block in old.get("blocks", []):
        step = wanted.get(block.get("block"))
        report = block.get("report") or {}
        if step is None or any(block.get(k) != step[k] for k in ("arm", "profile", "seed")):
            continue
        same_task = (
            report.get("dwell_ms") == settings["dwell_ms"]
            and report.get("trial_timeout_s") == settings["trial_timeout_s"]
            and (report.get("summary") or {}).get("idle_seconds") == settings["idle_seconds"]
            and report.get("model") == models.get(step["profile"])
        )
        if (
            same_task
            and not report.get("aborted")
            and len(report.get("trials", [])) == trials_per_block
        ):
            kept[step["block"]] = block
    return kept


def run_block_in_child(
    step: dict[str, Any], args: argparse.Namespace, report_path: Path
) -> dict[str, Any]:
    """One block in its OWN process.

    Measured on the first live run: the camera library keeps sampling state
    past shutdown, so the second block in the same process printed "Please do
    not call start_sampling repeatedly", saw no frames and stopped. A fresh
    process per block starts the library clean.
    """

    report_path.unlink(missing_ok=True)
    command = [
        sys.executable, str(Path(__file__).resolve()), "run-block",
        "--profile", step["profile"],
        "--trials", str(args.trials_per_block),
        "--seed", str(step["seed"]),
        "--dwell-ms", str(args.dwell_ms),
        "--trial-timeout-s", str(args.trial_timeout_s),
        "--idle-seconds", str(args.idle_seconds),
        "--report", str(report_path),
    ]  # fmt: skip
    subprocess.run(command, check=False)
    if not report_path.exists():
        # A crash or a kill is a stop, never a silent skip to the next block.
        return {"aborted": True, "error": "block process ended without a report"}
    return json.loads(report_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-profile":
        base = PROF.load(args.base) if args.base else PROF.load_active()
        if base is None:
            raise SystemExit("no active profile to copy; pass --base")
        path = make_trial_profile(
            base,
            name=args.name,
            model_dir=args.model_dir,
            note=args.note,
            capture_pipeline=args.capture_pipeline,
        )
        print(f"saved {path} (not activated; active is still {PROF.active_name()!r})")
        return 0

    import gf_live as L  # noqa: PLC0415 - pulls in the camera stack

    if args.command == "run-block":
        import gf_dwell_practice as P  # noqa: PLC0415

        report = P.run_practice(
            L.resolve_profile(args.profile),
            layout="zones",
            n_trials=args.trials,
            seed=args.seed,
            dwell_ms=args.dwell_ms,
            trial_timeout_s=args.trial_timeout_s,
            idle_seconds=args.idle_seconds,
            bar=None,
            show_bias=False,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        return 0

    reference = args.reference or PROF.active_name()
    if reference is None:
        raise SystemExit("no active profile; pass --reference")
    if args.trials_per_block % len(D.ZONE_KEYS):
        raise SystemExit("--trials-per-block must be a multiple of 9, or zones get unequal counts")
    if reference == args.candidate:
        raise SystemExit("reference and candidate are the same profile")
    profiles = {n: L.resolve_profile(n) for n in (reference, args.candidate)}
    for p in profiles.values():
        L.check_rig(p, p.rig_geometry(), allow_mismatch=False)
    if profiles[reference].filter != profiles[args.candidate].filter:
        raise SystemExit(
            "the two profiles use different filters; the comparison would not isolate the model"
        )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = (
        args.out or Path(__file__).resolve().parent / "results" / "select_compare" / f"{stamp}.json"
    )
    plan = block_plan(reference, args.candidate, seed=args.seed)
    settings = {
        "dwell_ms": args.dwell_ms,
        "trial_timeout_s": args.trial_timeout_s,
        "idle_seconds": args.idle_seconds,
    }
    models = {name: str(p.model_path()) for name, p in profiles.items()}
    kept = (
        resumable_blocks(
            args.resume, plan, args.trials_per_block, settings=settings, models=models
        )
        if args.resume
        else {}
    )
    blocks: list[dict[str, Any]] = []

    def save() -> None:
        # Written after EVERY block, so a crash loses at most the block running.
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plan": plan,
            "trials_per_block": args.trials_per_block,
            "settings": settings,
            "models": models,
            "blocks": blocks,
            "summary": summarise(blocks),
            "os_input": "none",
        }
        out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    for step in plan:
        if step["block"] in kept:
            print(f"\n=== block {step['block']} of {len(plan)}: kept from {args.resume.name} ===")
            blocks.append(kept[step["block"]])
            continue
        # The profile name is not shown: the person selecting should not know
        # which arm is running.
        print(f"\n=== block {step['block']} of {len(plan)} ===")
        report = run_block_in_child(step, args, out.with_suffix(f".block{step['block']}.json"))
        blocks.append({**step, "report": report})
        save()
        if report.get("aborted"):
            # Stop means stop: the next block must not reopen the camera.
            print("session stopped; remaining blocks not run")
            break
    save()
    summary = summarise(blocks)
    print(format_summary(summary))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
