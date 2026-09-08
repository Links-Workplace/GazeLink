"""Dwell practice: can this accuracy make useful, reliable choices?

A named target is requested, all targets are drawn identically, and whichever
one the gaze actually rests on is the one that fills and activates.  The
question is not how many pixels the error is -- that is already measured --
but whether a person can put a selection where they meant to.

What is deliberately NOT done here: the requested target is never given to the
dwell engine, never changes how a target is drawn, and never influences what
activates.  A selection mechanism that can see the answer is not measuring
selection.  `gf_dwell.DwellEngine.update` has no parameter for it and a test
asserts so.

Failures stay in the report.  A miss, a wrong selection and a lost face mean
different things and each is worth more than an average that hides all three.

No OS input.  Nothing here moves the cursor or emits a click; the targets live
inside this window only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_dwell as D  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402

# Worst horizontal/vertical bias measured on this rig, as screen fractions.
# Used only when a live measurement is unavailable, and the report says which
# of the two it used -- an assumed number and a measured one must never be
# indistinguishable in the output.
FALLBACK_BIAS = (0.056, 0.387)


@dataclass
class Trial:
    index: int
    requested: str
    activated: str | None = None
    activated_at_ms: float | None = None
    activated_at_point: tuple[float, float] | None = None
    extra_activations: list[str] = field(default_factory=list)
    frames: int = 0
    frames_without_point: int = 0
    # Where the gaze actually sat during the scored window. Without this the
    # report can say a trial went wrong but not how far wrong or in which
    # direction, which is the difference between "the model is biased" and
    # "the protocol let the previous target fire" -- and the first run could
    # not tell them apart.
    median_point: tuple[float, float] | None = None
    # The gaze had to be seen away from every target before the trial was
    # scored. False means it never got there within the allowance; the trial
    # still ran and says so, rather than hanging until it did.
    neutral_reached: bool = True
    neutral_wait_ms: float = 0.0
    # False when the neutral start was never achieved. Such a trial is NOT
    # scored: running it anyway would re-admit the very contamination the
    # gate exists to remove -- the gaze still resting on the previous
    # target when the window opens -- and then count it as a normal trial.
    scored: bool = True
    activations_before_start: list[str] = field(default_factory=list)
    # (ms since the window opened, x, y) sampled through the trial. A
    # single position at the moment of firing cannot say whether the gaze
    # reached the requested target and left, or never arrived -- and the
    # operator reported seeing the dot land correctly on trials the log
    # scored as deep inside the opposite target. One of those is wrong and
    # only the path can say which.
    path: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.activated == self.requested

    @property
    def lost_fraction(self) -> float:
        return self.frames_without_point / self.frames if self.frames else 0.0


def alternating_order(keys: list[str], n_trials: int) -> list[str]:
    """Strict left, right, left, right ...

    When the operator decides to alternate deliberately, the alternation IS
    the expected answer: no shuffled order is needed and none should be used.
    Any departure from the pattern is then a failure of the system rather than
    a mismatch with an order the operator was never following.

    This removes the confound that made the shuffled runs unreadable: that
    order alternated 79% of the time by construction, so "always alternate"
    and "follow the prompt" produced nearly identical statistics and could not
    be told apart.
    """

    return [keys[i % len(keys)] for i in range(n_trials)]


def trial_order(keys: list[str], n_trials: int, seed: int) -> list[str]:
    """A shuffled order in which every target appears about equally often.

    Not uniform random: with few trials that leaves one target under-tested
    and the score then describes the targets that happened to come up. Each
    full cycle of the targets is shuffled on its own, so the counts stay level
    while the order stays unpredictable.
    """

    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n_trials:
        block = list(keys)
        rng.shuffle(block)
        out.extend(block)
    return out[:n_trials]


def score(
    trials: list[Trial],
    *,
    unintended: int,
    idle_seconds: float,
    buttons: list[D.Button] | None = None,
) -> dict[str, Any]:
    """Aggregate, keeping the three failure kinds apart.

    A wrong selection and a non-selection are not the same event: one acts
    against the person's intent, the other only fails to act. Pooling them
    into "accuracy" would hide exactly the distinction that decides whether
    this is safe to build on.
    """

    scored = [t for t in trials if t.scored]
    skipped = [t for t in trials if not t.scored]
    done = [t for t in scored if t.activated is not None]
    correct = [t for t in done if t.correct]
    wrong = [t for t in done if not t.correct]
    none = [t for t in scored if t.activated is None]
    times = [t.activated_at_ms for t in correct if t.activated_at_ms is not None]
    lost = [t.lost_fraction for t in trials]
    return {
        "trials": len(trials),
        "scored_trials": len(scored),
        "skipped_no_neutral": len(skipped),
        "correct_first": len(correct),
        "wrong": len(wrong),
        "no_selection": len(none),
        "wrong_keys": sorted({f"{t.requested}->{t.activated}" for t in wrong}),
        "repeat_activations": sum(len(t.extra_activations) for t in trials),
        "time_to_select_ms": {
            "median": statistics.median(times) if times else None,
            "p90": float(np.percentile(times, 90)) if times else None,
            "max": max(times) if times else None,
        },
        "tracking_lost_fraction": {
            "median": statistics.median(lost) if lost else None,
            "max": max(lost) if lost else None,
        },
        "unintended_activations": unintended,
        "idle_seconds": idle_seconds,
        # Where in (or around) the requested target the gaze sat. NOT a bias
        # measurement: the operator is free to look anywhere inside a target,
        # so a non-zero value here is expected and proves nothing about the
        # model. Measuring bias needs an explicit centre marker to look at.
        # Kept because it helps explain which selections happened and why.
        "gaze_offset_from_centre": _offset_summary(scored, buttons),
        "neutral_gate_missed": len(skipped),
        "neutral_wait_ms_median": (
            statistics.median([t.neutral_wait_ms for t in trials]) if trials else None
        ),
        "activations_before_start": sum(len(t.activations_before_start) for t in trials),
    }


def _offset_summary(trials: list[Trial], buttons: list[D.Button] | None) -> dict[str, Any]:
    """Median signed offset of the gaze from the requested target's centre.

    Signed, and per axis: the direction is the whole point. An unsigned
    average would say the gaze was 0.05 away without saying whether it sat
    consistently to one side, which is what would make it correctable.
    """

    if not buttons:
        return {}
    centres = {b.key: b.centre for b in buttons}
    dx = [
        t.median_point[0] - centres[t.requested][0]
        for t in trials
        if t.median_point is not None and t.requested in centres
    ]
    dy = [
        t.median_point[1] - centres[t.requested][1]
        for t in trials
        if t.median_point is not None and t.requested in centres
    ]
    if not dx:
        return {}
    return {
        "n": len(dx),
        "dx_median": float(np.median(dx)),
        "dy_median": float(np.median(dy)),
        "dx_min": float(np.min(dx)),
        "dx_max": float(np.max(dx)),
    }


def verdict(summary: dict[str, Any], *, bar: int | None) -> tuple[bool | None, str]:
    """Pass, fail, or no bar -- and what kind of failure it was.

    The kind matters more than the score: a wrong-neighbour failure and a
    never-settles failure lead to different next steps, so the verdict names
    which one happened rather than only reporting a fraction.
    """

    if bar is None:
        return None, "diagnostic layout: reported as a limit, not as pass or fail"
    correct = summary["correct_first"]
    total = summary.get("scored_trials", summary["trials"])
    skipped = summary.get("skipped_no_neutral", 0)
    if total < bar:
        return False, (
            f"only {total} trial(s) were scored ({skipped} skipped for no neutral start), which "
            f"is fewer than the bar of {bar}. Not a dwell result: the session could not be "
            "started cleanly often enough to judge it"
        )
    if correct >= bar:
        return True, f"{correct}/{total} first-try correct, at or above the bar of {bar}"
    reasons = []
    if summary["wrong"] > summary["no_selection"]:
        reasons.append(
            f"mostly WRONG TARGET ({summary['wrong']} of {total}): the bias reaches past the "
            "dead zone. Selection is not viable at this spacing -- widen it or use fewer targets"
        )
    elif summary["no_selection"] > 0:
        reasons.append(
            f"mostly NO SELECTION ({summary['no_selection']} of {total}): the point never "
            "settles inside the target. The bias is larger than the target, or the dwell never "
            "completes"
        )
    if (summary["tracking_lost_fraction"]["median"] or 0.0) > 0.2:
        reasons.append(
            "tracking was lost for a large share of frames: this is a capture or seating "
            "problem, not a dwell problem"
        )
    if summary["unintended_activations"] > 0:
        reasons.append(
            f"{summary['unintended_activations']} unintended activation(s) while resting: the "
            "dwell time is too short, or resting gaze lands on a target and there is nowhere "
            "neutral to look"
        )
    if skipped:
        reasons.append(
            f"{skipped} trial(s) never reached a neutral start and were skipped rather than scored"
        )
    return False, f"{correct}/{total} first-try correct, below the bar of {bar}. " + "; ".join(
        reasons or ["no single dominant failure kind"]
    )


def measure_bias(profile: PROF.Profile) -> tuple[tuple[float, float], str]:
    """Worst per-target bias for this profile, as screen fractions.

    Measured from the most recent held-out recording rather than assumed, so
    a layout that is too tight for today's conditions is refused before the
    session instead of explained afterwards.
    """

    import gf_filter_benchmark as B  # noqa: PLC0415
    import gf_recal_compare as CMP  # noqa: PLC0415

    root = R.RECORDINGS_DIR
    candidates = sorted(
        (p for p in root.glob("round*") if (p / "T1.npz").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for recording in candidates:
        try:
            rec = S.Recording.load(recording, "T1")
            rig = C.RigGeometry.from_dict(rec.meta["rig"])
            scored = CMP.score_one(profile.model_path(), rec, rig, profile.filter_settings())
            pred = scored["_filtered"]
            worst_x = worst_y = 0.0
            for win in B._presentation_windows(
                np.flatnonzero(rec.rows_collecting()), rec.target_id
            ):
                idx = win[np.all(np.isfinite(pred[win]), axis=1)]
                if idx.size < 5:
                    continue
                worst_x = max(
                    worst_x, abs(float(np.median(pred[idx, 0]) - rec.target_xy[idx[0], 0]))
                )
                worst_y = max(
                    worst_y, abs(float(np.median(pred[idx, 1]) - rec.target_xy[idx[0], 1]))
                )
            if worst_x > 0.0:
                return (worst_x, worst_y), f"measured on {recording.name}/T1"
        except Exception:  # noqa: BLE001 - a sizing hint must not stop the session
            continue
    return FALLBACK_BIAS, "ASSUMED (no held-out recording could be scored)"


def run_practice(
    profile: PROF.Profile,
    *,
    layout: str = "a",
    n_trials: int = 20,
    seed: int = 1,
    dwell_ms: float = 900.0,
    trial_timeout_s: float = 8.0,
    idle_seconds: float = 60.0,
    neutral_timeout_s: float = 5.0,
    pattern: str = "shuffled",
    bar: int | None = 18,
    out: Path | None = None,
    headless: bool = False,
) -> dict[str, Any]:
    import gf_fit as FIT  # noqa: PLC0415

    rig = profile.rig_geometry()
    buttons = D.LAYOUTS[layout]()
    (bias_x, bias_y), bias_source = measure_bias(profile)
    warnings = D.layout_warnings(buttons, bias_x=bias_x, bias_y=bias_y)
    print(f"layout {layout}: {[b.key for b in buttons]}")
    print(f"bias used for sizing: |dx| {bias_x:.3f} |dy| {bias_y:.3f}  ({bias_source})")
    for line in warnings:
        print(f"  WARNING: {line}")
    if not warnings:
        print("  layout clears the measured bias on both axes")

    model = FIT.FittedModel.load(profile.model_path())
    gf = L.build_gaze_follower(rig)
    runner = L.LiveRunner(model, None, rig, profile.filter_settings())
    engine = D.DwellEngine(buttons, D.DwellConfig(dwell_ms=dwell_ms))
    menu = GEST.RecoveryMenu(
        [
            GEST.MenuOption("continue", "Keep going"),
            GEST.MenuOption("pause", "Pause"),
            GEST.MenuOption("exit", "Stop the session"),
        ]
    )
    armed = {"live": False}
    gf.add_subscriber(lambda face, gaze: runner.on_frame(face, gaze) if armed["live"] else None)
    display = R.Display(rig.device_w_px, rig.device_h_px, headless=headless, origin=(0, 0))

    keys = [b.key for b in buttons]
    order = (
        alternating_order(keys, n_trials)
        if pattern == "alternate"
        else trial_order(keys, n_trials, seed)
    )
    trials: list[Trial] = []
    unintended = 0
    aborted = False

    def poll(now: float) -> str | None:
        """Drain gestures into the menu; returns a chosen action, if any."""

        for when, event in runner.drain_gesture_events():
            taken = menu.handle(event, when)
            if taken is not None and taken.key != "continue":
                return taken.key
        menu.tick(now)
        return None

    try:
        gf.camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        if R._sleep_with_escape(display, R.CAMERA_WARMUP_S):
            return {"aborted": True}
        armed["live"] = True
        if (
            display.wait_for_key(
                [
                    f"Dwell practice -- layout {layout.upper()}, {n_trials} trials",
                    "",
                    "Look at the target named at the top left and rest on it",
                    f"until the ring fills ({dwell_ms:.0f} ms).",
                    "",
                    "Close your eyes about half a second to open the menu.",
                ]
            )
            == "abort"
        ):
            return {"aborted": True}

        for index, requested in enumerate(order):
            # The engine is deliberately NOT reset between trials. Resetting
            # clears the latch, so a target the operator is still resting on
            # from the previous trial starts filling again and fires before
            # they have moved: measured on the first run, 8 of 11 wrong
            # activations landed on the PREVIOUS trial's target and 7 fired at
            # the earliest instant possible. Carrying the latch is what makes
            # "one activation per entry" mean anything across a session.
            trial = Trial(index=index, requested=requested)
            flash: str | None = None

            # Neutral gate: the gaze must be seen away from every target
            # before the trial is scored, so each trial starts from the same
            # place. Bounded -- if it never gets there the trial still runs
            # and the report says the gate was not met, rather than waiting
            # for something that may not be reachable.
            gate_started = time.monotonic()
            while time.monotonic() - gate_started < neutral_timeout_s:
                now = time.monotonic()
                if display.poll_escape():
                    aborted = True
                    break
                state = runner.state
                point = R.visible_point(state.point, state.updated_s, now, R.OVERLAY_STALE_S)
                # Keep driving the engine: this is what lets the latch clear
                # naturally once the gaze genuinely leaves the old target.
                fired = engine.update(now, point, fresh=point is not None)
                if fired is not None:
                    trial.activations_before_start.append(fired.button)
                if point is not None and engine.find(point) is None:
                    trial.neutral_reached = True
                    break
                trial.neutral_reached = False
                display.draw_practice(
                    buttons,
                    point,
                    hovered=engine.hovered,
                    progress=engine.progress,
                    prompt=[
                        f"trial {index + 1}/{n_trials}",
                        "LOOK AWAY from both targets to begin",
                    ]
                    + L._menu_lines(menu),
                    tracking=point is not None,
                )
                time.sleep(0.005)
            trial.neutral_wait_ms = (time.monotonic() - gate_started) * 1000.0
            if aborted:
                trials.append(trial)
                break
            if not trial.neutral_reached:
                # Not scored, and said so. The alternative -- running it and
                # counting it -- is what let the previous target decide the
                # answer in the first run.
                trial.scored = False
                trials.append(trial)
                display.draw_message(
                    [
                        f"trial {index + 1}: no neutral start",
                        "",
                        "Skipped, and recorded as skipped.",
                    ]
                )
                time.sleep(1.0)
                continue

            seen: list[tuple[float, float]] = []
            started = time.monotonic()
            while True:
                now = time.monotonic()
                if display.poll_escape():
                    aborted = True
                    break
                action = poll(now)
                if action == "exit":
                    aborted = True
                    break
                if action == "pause":
                    engine.reset()
                    if display.wait_for_key(["Paused", "", "Press a key to continue"]) == "abort":
                        aborted = True
                        break
                    started = time.monotonic()
                if now - started > trial_timeout_s:
                    break
                state = runner.state
                point = R.visible_point(state.point, state.updated_s, now, R.OVERLAY_STALE_S)
                trial.frames += 1
                if point is None:
                    trial.frames_without_point += 1
                else:
                    seen.append(point)
                    # ~20 Hz is enough to see a saccade and a settle without
                    # turning the report into a frame dump.
                    elapsed_ms = (now - started) * 1000.0
                    if not trial.path or elapsed_ms - trial.path[-1][0] >= 50.0:
                        trial.path.append(
                            (round(elapsed_ms, 1), round(point[0], 4), round(point[1], 4))
                        )
                fired = engine.update(now, point, fresh=point is not None)
                if fired is not None:
                    if trial.activated is None:
                        trial.activated = fired.button
                        trial.activated_at_ms = (now - started) * 1000.0
                        trial.activated_at_point = fired.position
                        flash = fired.button
                    else:
                        trial.extra_activations.append(fired.button)
                display.draw_practice(
                    buttons,
                    point,
                    hovered=engine.hovered,
                    progress=engine.progress,
                    prompt=[
                        f"trial {index + 1}/{n_trials}    LOOK AT:  {requested}",
                        f"dwell {dwell_ms:.0f} ms    Esc to stop",
                    ]
                    + L._menu_lines(menu),
                    flash=flash,
                    tracking=point is not None,
                )
                if (
                    trial.activated is not None
                    and now - started > (trial.activated_at_ms or 0) / 1000.0 + 0.4
                ):
                    break
                time.sleep(0.005)
            if seen:
                trial.median_point = (
                    float(np.median([p[0] for p in seen])),
                    float(np.median([p[1] for p in seen])),
                )
            trials.append(trial)
            if aborted:
                break

        rest_prompt = [
            f"Rest block -- {idle_seconds:.0f} seconds",
            "",
            "Look around the screen and rest your eyes.",
            "Do NOT try to select anything.",
            "Anything that activates now is an unintended activation.",
        ]
        wanted = not aborted and idle_seconds > 0
        if wanted and display.wait_for_key(rest_prompt) != "abort":
            engine.reset()
            until = time.monotonic() + idle_seconds
            while time.monotonic() < until:
                now = time.monotonic()
                if display.poll_escape():
                    break
                state = runner.state
                point = R.visible_point(state.point, state.updated_s, now, R.OVERLAY_STALE_S)
                if engine.update(now, point, fresh=point is not None) is not None:
                    unintended += 1
                display.draw_practice(
                    buttons,
                    point,
                    hovered=engine.hovered,
                    progress=engine.progress,
                    prompt=[
                        f"REST -- {until - now:.0f}s left, select nothing",
                        f"unintended so far: {unintended}",
                    ],
                    tracking=point is not None,
                )
                time.sleep(0.005)
    finally:
        display.close()
        R.shutdown_library(gf)

    summary = score(trials, unintended=unintended, idle_seconds=idle_seconds, buttons=buttons)
    passed, why = verdict(summary, bar=bar)
    report = {
        "profile": profile.name,
        "model": str(profile.model_path()),
        "layout": layout,
        "buttons": [asdict(b) for b in buttons],
        "bias_used": {"dx": bias_x, "dy": bias_y, "source": bias_source},
        "layout_warnings": warnings,
        "dwell_ms": dwell_ms,
        "trial_timeout_s": trial_timeout_s,
        "neutral_timeout_s": neutral_timeout_s,
        "seed": seed,
        "pattern": pattern,
        "order": order,
        "aborted": aborted,
        "summary": summary,
        "passed": passed,
        "verdict": why,
        "trials": [asdict(t) for t in trials],
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "confidence_note": (
            "The harness has no per-frame confidence signal, so activation is gated on gaze "
            "status, both eyes above the blink threshold, and a point fresher than "
            f"{R.OVERLAY_STALE_S}s. That is weaker than the product's stated 0.85 control gate."
        ),
        "os_input": "none",
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def format_report(report: dict[str, Any]) -> str:
    if report.get("aborted") and not report.get("trials"):
        return "aborted; nothing measured"
    s = report["summary"]
    t = s["time_to_select_ms"]
    lines = [
        "",
        f"dwell practice -- layout {report['layout'].upper()}, dwell {report['dwell_ms']:.0f} ms",
        "",
        f"  first-try correct   {s['correct_first']}/{s.get('scored_trials', s['trials'])}"
        f"   (scored trials; {s.get('skipped_no_neutral', 0)} skipped for no neutral start)",
        f"  wrong target        {s['wrong']}"
        + (f"   {s['wrong_keys']}" if s["wrong_keys"] else ""),
        f"  no selection        {s['no_selection']}",
        f"  repeat activations  {s['repeat_activations']}",
        f"  unintended (rest)   {s['unintended_activations']} in {s['idle_seconds']:.0f}s",
        f"  neutral gate missed {s.get('neutral_gate_missed', 0)}"
        f"   (activations before a trial started: {s.get('activations_before_start', 0)})",
    ]
    off = s.get("gaze_offset_from_centre") or {}
    if off:
        lines.append(
            f"  gaze vs target centre: dx {off['dx_median']:+.3f} "
            f"(range {off['dx_min']:+.3f}..{off['dx_max']:+.3f})  dy {off['dy_median']:+.3f}"
        )
        lines.append(
            "     (context for the selections, NOT a bias measurement -- looking anywhere "
            "inside a target is allowed)"
        )
    if t["median"] is not None:
        lines.append(
            f"  time to select      median {t['median']:.0f} ms  p90 {t['p90']:.0f} ms  "
            f"max {t['max']:.0f} ms"
        )
    lost = s["tracking_lost_fraction"]
    if lost["median"] is not None:
        lines.append(
            f"  frames with no point  median {lost['median'] * 100:.0f}%  worst trial "
            f"{lost['max'] * 100:.0f}%"
        )
    lines += ["", f"  {report['verdict']}", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--layout", choices=sorted(D.LAYOUTS), default="a")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--pattern",
        choices=("shuffled", "alternate"),
        default="shuffled",
        help=(
            "alternate: strict LEFT, RIGHT, LEFT, RIGHT. Use it when you intend to "
            "alternate deliberately -- then the pattern itself is the expected answer "
            "and any departure from it is the system getting it wrong."
        ),
    )
    parser.add_argument("--dwell-ms", type=float, default=900.0)
    parser.add_argument("--trial-timeout-s", type=float, default=8.0)
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--neutral-timeout-s", type=float, default=5.0)
    parser.add_argument("--bar", type=int, default=None, help="pass mark (layout a defaults to 18)")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = L.resolve_profile(args.profile)
    L.check_rig(profile, profile.rig_geometry(), allow_mismatch=False)
    bar = args.bar if args.bar is not None else (18 if args.layout == "a" else None)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = args.out or (
        Path(__file__).resolve().parent / "results" / "dwell" / f"{args.layout}_{stamp}.json"
    )
    report = run_practice(
        profile,
        layout=args.layout,
        n_trials=args.trials,
        seed=args.seed,
        dwell_ms=args.dwell_ms,
        trial_timeout_s=args.trial_timeout_s,
        idle_seconds=args.idle_seconds,
        neutral_timeout_s=args.neutral_timeout_s,
        pattern=args.pattern,
        bar=bar,
        out=out,
    )
    print(format_report(report))
    print(f"wrote {out}")
    return 0 if report.get("passed") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
