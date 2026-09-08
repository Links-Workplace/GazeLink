"""Does a deliberate eye closure separate from an ordinary blink, here?

Hands-free recalibration needs a signal the operator can produce on purpose
and never produces by accident.  The obvious candidate is "both eyes shut
while the face is still tracked", and the plan for it carries thresholds
(400 ms natural blink, 1500 ms confirm) that were **guessed**, in a different
codebase, against a different eye signal, and never once measured on a person.

Building a menu on an unmeasured signal is how a system ends up either
ignoring deliberate input or firing on its own.  So this measures first.

What it measures: for each frame, whether the face is tracked and whether both
eye-openness values sit below the blink threshold.  It then extracts the
CLOSURE RUNS -- maximal stretches of consecutive shut frames -- from two
recorded blocks, and reports how their durations compare.

The signal here is eye polygon AREA in px^2 (``BLINK_THRESHOLD`` = 10.0), an
absolute measure that grows and shrinks with how close you sit and how large
the image is.  That is a known weakness worth seeing in the numbers: the
product's own gesture code deliberately avoided openness for this reason.

Privacy: keeps two floats and two booleans per frame in memory for the length
of the run.  No frame, image, landmark or embedding is stored, and the file it
writes contains aggregate durations only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402


@dataclass
class Sample:
    t: float
    face: bool
    shut: bool
    left: float
    right: float


@dataclass
class Block:
    """One instruction the operator followed, and the frames it produced."""

    name: str
    instruction: str
    seconds: float
    samples: list[Sample] = field(default_factory=list)


def closure_runs(samples: list[Sample], *, bridge_ms: float = 0.0) -> list[float]:
    """Durations of maximal runs of shut-with-face-present, in milliseconds.

    A run ends when the eyes open OR when the face is lost.  Losing the face
    has to end it: "I cannot see you" is not evidence that your eyes are shut,
    and letting it extend a run is exactly how looking away would be read as a
    deliberate hold.

    ``bridge_ms`` ignores re-openings shorter than itself.  Measured on a real
    person, a two-second deliberate closure did NOT arrive as one run: the
    openness estimate jitters back across the threshold mid-hold, so the hold
    came apart into fragments (11 runs, longest 1125 ms, for closures held
    about 2000 ms).  A confirm threshold applied to unbridged runs would
    therefore almost never fire.  Bridging is measured here rather than
    assumed -- the whole point is to find out which value, if any, recovers
    the hold without also fusing separate blinks into one.

    A gap ends the run when the FACE is lost regardless of ``bridge_ms``:
    bridging is for a flickering openness estimate, not for absence.
    """

    runs: list[float] = []
    start: float | None = None
    last_shut: float | None = None
    previous: float | None = None
    for s in samples:
        holding = s.face and s.shut
        if holding:
            if start is None:
                start = s.t
            last_shut = s.t
        elif start is not None:
            gap_ms = (s.t - last_shut) * 1000.0 if last_shut is not None else 0.0
            # Absence is never bridged; only a flickering openness estimate is.
            if not s.face or gap_ms > bridge_ms:
                runs.append((last_shut - start) * 1000.0 if last_shut is not None else 0.0)
                start = None
                last_shut = None
        previous = s.t
    if start is not None and last_shut is not None:
        runs.append((last_shut - start) * 1000.0)
    del previous
    return [r for r in runs if r > 0.0]


def area_percentiles(samples: list[Sample]) -> dict[str, float | None]:
    """Where the eye-area signal actually sits, so a threshold can be chosen.

    The inherited threshold (10.0 px^2) was not derived from any measurement
    on this rig; the first probe found open eyes reaching down to 10-12 px^2,
    i.e. no margin at all at the boundary.  Reporting the distribution turns
    "the threshold jitters" into "here is where to put it".
    """

    values = [min(s.left, s.right) for s in samples if s.face]
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{int(q)}": float(np.percentile(array, q)) for q in (1, 5, 10, 25, 50, 75, 90, 99)}


BRIDGE_LADDER = (0.0, 100.0, 150.0, 200.0, 300.0, 400.0)


def runs_summary(runs: list[float]) -> dict[str, Any]:
    return {
        "closures": len(runs),
        "min": min(runs) if runs else None,
        "median": float(np.median(runs)) if runs else None,
        "max": max(runs) if runs else None,
        "all": [round(r, 1) for r in runs],
    }


def summarise(block: Block, bridges: tuple[float, ...] = BRIDGE_LADDER) -> dict[str, Any]:
    return {
        "block": block.name,
        "frames": len(block.samples),
        "frames_with_face": sum(1 for s in block.samples if s.face),
        "area_px2": area_percentiles(block.samples),
        "threshold_px2": C.BLINK_THRESHOLD,
        "by_bridge": {
            str(int(b)): runs_summary(closure_runs(block.samples, bridge_ms=b)) for b in bridges
        },
    }


def verdict(
    natural: dict[str, Any],
    deliberate: dict[str, Any],
    bridges: tuple[float, ...] = BRIDGE_LADDER,
) -> tuple[bool, str]:
    """The smallest bridge, if any, at which holds clear every natural blink.

    The comparison is the LONGEST natural blink against the SHORTEST
    deliberate hold.  Medians would flatter the signal: what decides a
    threshold is whether the worst case of one class crosses the best case of
    the other, because that crossing is a false activation on a real person.

    The smallest working bridge is preferred, not the largest: bridging is
    permission to ignore evidence that the eyes reopened, and every extra
    millisecond of it makes a pair of ordinary blinks likelier to fuse into
    one apparent hold.
    """

    if not natural["frames_with_face"] or not deliberate["frames_with_face"]:
        return False, "the face was not tracked in one of the blocks; nothing to compare"
    for b in bridges:
        key = str(int(b))
        blink = natural["by_bridge"][key]
        hold = deliberate["by_bridge"][key]
        if hold["closures"] == 0:
            continue
        if blink["max"] is None:
            return True, (
                f"at bridge {key} ms: no natural blink registered at all, and deliberate holds "
                f"ran {hold['min']:.0f}-{hold['max']:.0f} ms. Clean separation."
            )
        if hold["min"] > blink["max"]:
            return True, (
                f"at bridge {key} ms: longest natural blink {blink['max']:.0f} ms, shortest "
                f"deliberate hold {hold['min']:.0f} ms. A confirm threshold between them exists."
            )
    worst = deliberate["by_bridge"][str(int(bridges[-1]))]
    if worst["closures"] == 0:
        return False, (
            "no closure was detected at all during the deliberate block. The openness signal "
            "does not register your eyes shutting, so a gesture cannot be built on it as it "
            "stands."
        )
    return False, (
        f"no bridge up to {int(bridges[-1])} ms separates the two: at every value some natural "
        "blink is as long as the shortest deliberate hold. A duration threshold on this signal "
        "would either miss real holds or fire on ordinary blinking."
    )


class ProbeRunner:
    """Records the eye signal only. No prediction, no model, no storage."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.block: Block | None = None
        self.errors = 0

    def on_frame(self, face_info: Any, gaze_info: Any) -> None:
        try:
            now = self.clock()
            left = float(getattr(face_info, "left_eye_openness", 0.0) or 0.0)
            right = float(getattr(face_info, "right_eye_openness", 0.0) or 0.0)
            face = bool(getattr(face_info, "status", False))
            shut = not (left > C.BLINK_THRESHOLD and right > C.BLINK_THRESHOLD)
            with self.lock:
                if self.block is not None:
                    self.block.samples.append(Sample(now, face, shut, left, right))
        except Exception:  # noqa: BLE001 - never raise into the camera thread
            with self.lock:
                self.errors += 1


BLOCKS = (
    Block(
        "natural",
        "Look around the screen and BLINK NORMALLY. Do not hold your eyes shut.",
        12.0,
    ),
    Block(
        "deliberate",
        "Close your eyes for about TWO SECONDS, then open. Repeat until the time is up.",
        15.0,
    ),
)


def run_probe(profile: PROF.Profile, *, headless: bool = False) -> dict[str, Any]:
    rig = profile.rig_geometry()
    gf = L.build_gaze_follower(rig)
    runner = ProbeRunner()
    gf.add_subscriber(lambda face, gaze: runner.on_frame(face, gaze))
    display = R.Display(rig.device_w_px, rig.device_h_px, headless=headless, origin=(0, 0))
    blocks = [Block(b.name, b.instruction, b.seconds) for b in BLOCKS]
    try:
        gf.camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        if R._sleep_with_escape(display, R.CAMERA_WARMUP_S):
            return {"aborted": True}
        for block in blocks:
            if (
                display.wait_for_key(
                    [f"{block.name}  ({block.seconds:.0f} s)", "", block.instruction]
                )
                == "abort"
            ):
                return {"aborted": True}
            with runner.lock:
                runner.block = block
            display.draw_message([block.instruction])
            deadline = time.monotonic() + block.seconds
            while time.monotonic() < deadline:
                if display.poll_escape():
                    return {"aborted": True}
                left = deadline - time.monotonic()
                display.draw_message([block.instruction, "", f"{left:.0f} s"])
                time.sleep(0.05)
            with runner.lock:
                runner.block = None
    finally:
        display.close()
        R.shutdown_library(gf)
    summaries = {b.name: summarise(b) for b in blocks}
    ok, message = verdict(summaries["natural"], summaries["deliberate"])
    return {
        "aborted": False,
        "threshold_px2": C.BLINK_THRESHOLD,
        "blocks": summaries,
        "separated": ok,
        "verdict": message,
        "errors": runner.errors,
    }


def report(result: dict[str, Any]) -> str:
    if result.get("aborted"):
        return "aborted; nothing measured"
    lines = ["", "eye-closure signal probe", ""]
    for name in ("natural", "deliberate"):
        b = result["blocks"][name]
        lines.append(f"{name}: {b['frames']} frames, {b['frames_with_face']} with a face")
        area = b["area_px2"]
        if area:
            lines.append(
                f"  eye area px^2 (min of the two eyes): p1 {area['p1']:.0f}  p5 {area['p5']:.0f}  "
                f"p25 {area['p25']:.0f}  median {area['p50']:.0f}  p90 {area['p90']:.0f}"
                f"   [threshold {b['threshold_px2']:.0f}]"
            )
        lines.append("  bridge ms |  closures |   min |median|   max")
        for key, r in b["by_bridge"].items():
            if r["closures"]:
                lines.append(
                    f"  {key:>9s} | {r['closures']:9d} | {r['min']:5.0f} |{r['median']:5.0f} "
                    f"|{r['max']:6.0f}"
                )
            else:
                lines.append(f"  {key:>9s} | {0:9d} |     - |    - |     -")
        lines.append("")
    lines.append(("SEPARATED: " if result["separated"] else "NOT SEPARATED: ") + result["verdict"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the aggregate JSON here")
    args = parser.parse_args(argv)
    profile = L.resolve_profile(args.profile)
    result = run_probe(profile)
    print(report(result))
    if args.out and not result.get("aborted"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if result.get("separated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
