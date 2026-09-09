"""What does a wink actually look like, on this person, on this camera?

Every wink threshold in this project so far was reasoned from summaries --
medians, minima, counts -- and every one of them was wrong. A median cannot
tell you how long a wink lasts, and a minimum cannot tell you whether the
other eye came with it. So this records EVERY FRAME and labels it, and the
rule is derived afterwards from the frames rather than argued from the
aggregates.

Two phases, so the data is labelled:

    WINK    the operator winks the right eye, whenever they like
    BLINK   the operator blinks normally and deliberately does NOT wink

Anything that separates the two phases is a candidate rule; anything that does
not is not, however sensible it sounds. There is no threshold in this file and
nothing here decides whether a wink happened -- it measures, and
``gf_wink_probe.py --analyse`` reads the result back.

Shows both eyes live and large, because a person cannot tell a gesture that
was not made from a gesture the camera could not see, and being unable to tell
those apart is what made the last three attempts feel arbitrary.

Privacy: two floats per frame in memory, written as aggregates plus the raw
series of those same two numbers. No frame, image, landmark or embedding is
stored. No OS input of any kind.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "wink"
PHASES = (
    ("WINK", "Wink your RIGHT eye. Take your time, about ten of them."),
    ("BLINK", "Blink normally. Do NOT wink."),
)


@dataclass
class Sample:
    """One frame. Two numbers and where they came from."""

    t: float
    phase: str
    left: float
    right: float
    left_ratio: float
    right_ratio: float

    def as_row(self) -> list[Any]:
        return [
            round(self.t, 3),
            self.phase,
            round(self.left, 1),
            round(self.right, 1),
            round(self.left_ratio, 4),
            round(self.right_ratio, 4),
        ]


@dataclass
class Run:
    """A stretch of consecutive frames where one eye was closing."""

    phase: str
    frames: int
    ms: float
    deepest_right: float
    deepest_left: float
    # The whole question: at the deepest point, how much more open was the
    # other eye? A blink should sit near 1; a wink should not.
    openness_gap: float = field(default=0.0)


# A floor rather than a special case for zero. Both eyes at zero is a blink,
# and dividing by zero to get "infinitely asymmetric" said the opposite --
# measured in the first probe run, a two-eyed closure reported a gap of inf.
GAP_FLOOR = 1e-4


def openness_gap(left_ratio: float, right_ratio: float) -> float:
    """How much more open the left eye was, at the right eye's deepest point.

    Near 1 for a blink, because both eyes go together. Large for a wink. Small
    when the LEFT is the one that closed, which is a left wink and not this
    gesture. Monotone across all three, with no case that reads as its
    opposite.
    """

    return left_ratio / max(right_ratio, GAP_FLOOR)


def closure_runs(samples: list[Sample], *, closing_under: float) -> list[Run]:
    """Maximal stretches where the RIGHT eye was below ``closing_under``.

    Deliberately not "below and the left one open": that condition is exactly
    what previous versions assumed and got wrong, so imposing it here would
    hide the evidence needed to test it.
    """

    runs: list[Run] = []
    current: list[Sample] = []
    # The trailing None closes a run that was still going when the phase
    # ended: a wink held through the timer is still a wink, and dropping it
    # would bias the durations toward the short ones.
    padded: list[Sample | None] = [*samples, None]
    for sample in padded:
        if sample is not None and sample.right_ratio < closing_under:
            current.append(sample)
            continue
        if current:
            deepest = min(current, key=lambda s: s.right_ratio)
            runs.append(
                Run(
                    phase=current[0].phase,
                    frames=len(current),
                    ms=(current[-1].t - current[0].t) * 1000.0,
                    deepest_right=deepest.right_ratio,
                    deepest_left=deepest.left_ratio,
                    openness_gap=openness_gap(deepest.left_ratio, deepest.right_ratio),
                )
            )
            current = []
    return runs


def separation(runs: list[Run]) -> dict[str, Any]:
    """What, if anything, tells the two phases apart.

    Reports the two groups side by side and does NOT propose a threshold: a
    number chosen to fit the data it was derived from is not a measurement,
    and every threshold in this project that was picked that way has since had
    to be picked again.
    """

    winks = [r for r in runs if r.phase == "WINK"]
    blinks = [r for r in runs if r.phase == "BLINK"]

    def describe(group: list[Run]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        gaps = sorted(r.openness_gap for r in group)
        return {
            "n": len(group),
            "ms": {
                "min": round(min(r.ms for r in group), 1),
                "median": round(sorted(r.ms for r in group)[len(group) // 2], 1),
                "max": round(max(r.ms for r in group), 1),
            },
            "deepest_right": {
                "min": round(min(r.deepest_right for r in group), 4),
                "median": round(sorted(r.deepest_right for r in group)[len(group) // 2], 4),
                "max": round(max(r.deepest_right for r in group), 4),
            },
            "left_at_the_deepest_point": {
                "min": round(min(r.deepest_left for r in group), 4),
                "median": round(sorted(r.deepest_left for r in group)[len(group) // 2], 4),
                "max": round(max(r.deepest_left for r in group), 4),
            },
            "openness_gap_left_over_right": {
                "min": round(gaps[0], 2),
                "median": round(gaps[len(gaps) // 2], 2),
                "max": round(gaps[-1], 2),
            },
        }

    return {"WINK": describe(winks), "BLINK": describe(blinks)}


def run_probe(
    profile: PROF.Profile,
    *,
    seconds_per_phase: float = 20.0,
    closing_under: float = 0.7,
    headless: bool = False,
    advance_timeout_s: float | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """Record both phases and write everything. Decides nothing."""

    rig = profile.rig_geometry()
    gf = L.build_gaze_follower(rig)
    # No model: this measures eyelids, and loading one would make a failure to
    # predict look like a failure to see the eyes.
    runner = L.LiveRunner(None, None, rig, None)
    live = {"on": False}
    gf.add_subscriber(lambda face, gaze: runner.on_frame(face, gaze) if live["on"] else None)
    display = R.Display(rig.device_w_px, rig.device_h_px, headless=headless, origin=(0, 0))
    samples: list[Sample] = []
    aborted = False
    started = time.monotonic()

    try:
        gf.camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        if R._sleep_with_escape(display, R.CAMERA_WARMUP_S):
            return {"aborted": True}
        live["on"] = True
        for phase, instruction in PHASES:
            if (
                display.wait_for_key(
                    [
                        f"{phase}  --  {seconds_per_phase:.0f} seconds",
                        "",
                        instruction,
                        "",
                        "Both eyes are shown live. Nothing is decided here and",
                        "nothing is clicked; this only records what the camera sees.",
                        "",
                        "Esc stops everything.",
                    ],
                    timeout_s=advance_timeout_s,
                )
                == "abort"
            ):
                aborted = True
                break
            until = time.monotonic() + seconds_per_phase
            while time.monotonic() < until:
                now = time.monotonic()
                if display.poll_escape():
                    aborted = True
                    break
                state = runner.state
                if state.openness is not None and state.openness_ratio is not None:
                    samples.append(
                        Sample(
                            t=now - started,
                            phase=phase,
                            left=state.openness[0],
                            right=state.openness[1],
                            left_ratio=state.openness_ratio[0],
                            right_ratio=state.openness_ratio[1],
                        )
                    )
                ratio = state.openness_ratio or (1.0, 1.0)
                display.draw_message(
                    [
                        f"{phase}   {until - now:.0f}s left",
                        "",
                        instruction,
                        "",
                        f"LEFT   {ratio[0] * 100:5.0f}%   {'#' * int(min(40, ratio[0] * 40))}",
                        f"RIGHT  {ratio[1] * 100:5.0f}%   {'#' * int(min(40, ratio[1] * 40))}",
                        "",
                        f"raw px^2:  L {state.openness[0]:.0f}  R {state.openness[1]:.0f}"
                        if state.openness
                        else "no face",
                        f"samples {len(samples)}",
                    ]
                )
                time.sleep(0.005)
            if aborted:
                break
    finally:
        display.close()
        R.shutdown_library(gf)

    runs = closure_runs(samples, closing_under=closing_under)
    result = {
        "profile": profile.name,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "os_input": "none",
        "seconds_per_phase": seconds_per_phase,
        "closing_under": closing_under,
        "aborted": aborted,
        "frames": len(samples),
        "frames_per_phase": {
            phase: sum(1 for s in samples if s.phase == phase) for phase, _ in PHASES
        },
        "gate": {
            "shut_fraction": GEST.OpennessGateConfig().shut_fraction,
            "absolute_threshold": C.BLINK_THRESHOLD,
        },
        "runs": [
            {
                "phase": r.phase,
                "frames": r.frames,
                "ms": round(r.ms, 1),
                "deepest_right": round(r.deepest_right, 4),
                "left_there": round(r.deepest_left, 4),
                "gap": round(r.openness_gap, 2),
            }
            for r in runs
        ],
        "separation": separation(runs),
        "columns": ["t", "phase", "left_px2", "right_px2", "left_ratio", "right_ratio"],
        "series": [s.as_row() for s in samples],
    }
    path = out or RESULTS_DIR / f"wink_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print(format_report(result))
    return result


def format_report(result: dict[str, Any]) -> str:
    lines = [
        "",
        f"frames {result['frames']}  {result['frames_per_phase']}",
        f"closure runs: {len(result['runs'])}",
        "",
    ]
    sep = result["separation"]
    for phase in ("WINK", "BLINK"):
        group = sep.get(phase, {})
        if not group.get("n"):
            lines.append(f"{phase}: no closures recorded at all")
            continue
        ms, deep, other, gap = (
            group["ms"],
            group["deepest_right"],
            group["left_at_the_deepest_point"],
            group["openness_gap_left_over_right"],
        )
        lines.append(
            f"{phase}: {group['n']} closures, "
            f"{ms['median']:.0f} ms median ({ms['min']:.0f}-{ms['max']:.0f})"
        )
        lines.append(
            f"        right reaches {deep['median']:.3f}, left is at {other['median']:.3f} there, "
            f"gap x{gap['median']:.1f}  (x{gap['min']:.1f} to x{gap['max']:.1f})"
        )
    lines += [
        "",
        "The gap is left divided by right at the deepest frame. A blink closes",
        "both eyes and sits near 1; a wink does not. If the two phases overlap",
        "on every column, no threshold on these numbers can separate them and",
        "the signal is not the place to look.",
    ]
    return "\n".join(lines)


def analyse(path: Path) -> str:
    return format_report(json.loads(Path(path).read_text(encoding="utf-8")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seconds-per-phase", type=float, default=20.0)
    parser.add_argument(
        "--closing-under",
        type=float,
        default=0.7,
        help="what counts as the eye starting to close, for grouping frames into runs. "
        "Deliberately loose: this is not a decision threshold.",
    )
    parser.add_argument("--advance-timeout-s", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--analyse", type=Path, default=None, help="re-read a saved probe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyse is not None:
        print(analyse(args.analyse))
        return 0
    run_probe(
        L.resolve_profile(args.profile),
        seconds_per_phase=args.seconds_per_phase,
        closing_under=args.closing_under,
        advance_timeout_s=args.advance_timeout_s,
        out=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
