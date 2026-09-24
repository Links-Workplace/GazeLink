"""Deterministic behaviour trace of the live session (ARCH-01 stage C).

Records, for a scripted scenario, everything the session does that a person or
Windows could observe: every input event handed to a sender (with fake time),
every draw call, every printed line and the return code. The trace was
captured from the pre-refactor ``run_live`` and is the equivalence oracle for
stages E-G (TECHNICAL_SPEC 17, "behaviour equivalence").

Determinism:
* time is fake and INJECTED (``LiveEnvironment.clock``). It moves only when
  something sleeps through it (the loop's 5 ms tick, the click adapter's
  double-click gap). Nothing patches ``time``: a trace that still matches
  proves every control-time read in the session goes through the injected
  clock. (The golden file was first captured with ``time`` patched
  process-wide, before the seam existed.)
* the pipeline is a script keyed on TIME, not on how often its state is read,
  so a refactor that reads the snapshot a different number of times per frame
  produces the same trace.
* no camera, no window, no real OS input: every sender is a recorder.

Scenario inputs are synthetic (no recordings, no biometric data).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gf_gesture as GEST
from gazelink_core.domain.clock import FakeClock

from fake_live_env import FakeWorld

T0 = 1000.0
TESTS_DIR = Path(__file__).resolve().parent
MONITOR = SimpleNamespace(origin=(0, 0), width_px=5120, height_px=1440, name="FAKE")
DESKTOP = {"x": 0, "y": 0, "width": 5120, "height": 1440}


Point = tuple[float, float]


@dataclass
class Script:
    """What the pipeline reports, as functions of seconds since start."""

    gaze: Callable[[float], Point | None] = lambda t: (0.5, 0.5)
    face: Callable[[float], bool] = lambda t: True
    steady: Callable[[float], bool] = lambda t: True
    # (seconds, event) -- delivered at most one per drain, like the detector.
    gestures: list[tuple[float, GEST.Event]] = field(default_factory=list)
    # (seconds, aim or None, age_s) -- aim is where the wink was aimed.
    winks: list[tuple[float, Point | None, float]] = field(default_factory=list)
    space_down: list[tuple[float, float]] = field(default_factory=list)
    escape_at: float | None = None
    # Window handle in front, as a function of seconds.
    foreground: Callable[[float], int] = lambda t: 4242


class ScriptedPipeline:
    """Stands in for LiveRunner. Answers from the script at the fake time."""

    def __init__(self, script: Script, clock: FakeClock) -> None:
        self.script = script
        self._clock = clock
        self._gestures = sorted(script.gestures, key=lambda g: g[0])
        self._winks = sorted(script.winks, key=lambda w: w[0])
        self.errors = 0
        self.last_error = None
        self.winks_dropped_mid_frame = 0

    def clock(self) -> float:
        return self._clock.now()

    def _t(self) -> float:
        return self._clock.now() - T0

    @property
    def state(self) -> SimpleNamespace:
        t = self._t()
        present = self.script.face(t)
        point = self.script.gaze(t) if present else None
        return SimpleNamespace(
            point=point,
            raw_model=None,
            unfiltered=point,
            tracking=point is not None,
            eyes_steady=bool(present and point is not None and self.script.steady(t)),
            face_present=present,
            openness=(150.0, 130.0) if present else (0.0, 0.0),
            openness_ratio=(1.0, 1.0) if present else None,
            head_pitch=None,
            head=None,
            updated_s=self._clock.now(),
            frames=int(t * 30.0),
            fps=30.0,
        )

    def on_observation(self, *a: object) -> None:
        return None

    def drain_gesture_events(self) -> list[tuple[float, GEST.Event]]:
        t = self._t()
        if self._gestures and self._gestures[0][0] <= t:
            when, event = self._gestures.pop(0)
            return [(T0 + when, event)]
        return []

    def drain_wink_events(self) -> list[tuple[float, Point | None, GEST.Eye]]:
        t = self._t()
        out = []
        while self._winks and self._winks[0][0] <= t:
            # (when, aim, age) or (when, aim, age, eye). LEFT by default: the
            # ordinary click, which is what every scenario captured before
            # 24.9 meant by "a wink" (it pins ``wink_click`` to reproduce it).
            when, aim, age, *rest = self._winks.pop(0)
            eye = rest[0] if rest else GEST.Eye.LEFT
            out.append((T0 + when - age, aim, eye))
        return out

    def cancel_wink(self) -> int:
        # Winks already due are the "fired" queue; they are dropped. Winks
        # scheduled later are not yet made and survive, as a closure that
        # starts after the mode change would.
        t = self._t()
        dropped = 0
        while self._winks and self._winks[0][0] <= t:
            self._winks.pop(0)
            dropped += 1
        return dropped


def _pt(p: Any) -> Any:
    return None if p is None else (float(p[0]), float(p[1]))


class RecordingDisplay:
    def __init__(self, trace: list[Any]) -> None:
        self.trace = trace
        self.overlay = None
        self.font_name = "fake"

    def draw_message(self, lines: Any) -> None:
        self.trace.append(("message", tuple(lines)))

    def poll_escape(self) -> bool:
        return False

    def draw_live(self, point, raw, unfiltered, hud, *, tracking, zones=(), active_zone=None):  # noqa: ANN001
        self.trace.append(
            ("live", _pt(point), _pt(raw), _pt(unfiltered), tuple(hud), bool(tracking),
             tuple(z.key for z in zones), active_zone)
        )

    def draw_board(self, buttons, labels, *, hovered, progress, centre=(), point=None,  # noqa: ANN001
                   tracking=True, ready=True):
        self.trace.append(
            ("board", tuple(b.key for b in buttons), tuple(sorted(dict(labels).items())), hovered,
             None if progress is None else float(progress), tuple(centre), _pt(point),
             bool(tracking), bool(ready))
        )

    def draw_scan(self, items, index, *, centre=(), typed="", parked=False, point=None,  # noqa: ANN001
                  tracking=True):
        self.trace.append(
            ("scan", tuple(items), index, tuple(centre), typed, bool(parked), _pt(point),
             bool(tracking))
        )

    def close(self) -> None:
        self.trace.append(("close",))


@dataclass
class Trace:
    returncode: Any
    inputs: list[Any]
    draws: list[Any]
    stdout: list[str]

    def to_golden(self) -> dict[str, Any]:
        draws_json = json.dumps(self.draws, ensure_ascii=False, default=repr)
        kinds: list[list[Any]] = []
        for d in self.draws:
            if kinds and kinds[-1][0] == d[0]:
                kinds[-1][1] += 1
            else:
                kinds.append([d[0], 1])
        return {
            "returncode": self.returncode,
            "inputs": self.inputs,
            "stdout": self.stdout,
            "draw_count": len(self.draws),
            "draw_kinds": kinds,
            "draw_sha256": hashlib.sha256(draws_json.encode("utf-8")).hexdigest(),
        }


def _normalise(text: str) -> list[str]:
    return [line.replace(str(TESTS_DIR), "<tests>") for line in text.splitlines()]


def profile() -> Any:
    import gf_profile as PROF  # noqa: PLC0415

    return PROF.Profile(
        name="trace",
        model_dir=str(TESTS_DIR),
        rig={
            "camera_x_cm": 60.0, "camera_y_cm": 63.6, "screen_w_cm": 120.0,
            "screen_h_cm": 33.75, "device_w_px": 5120, "device_h_px": 1440,
        },
        filter={"kind": "one-euro"},
        cursor={"smoothing": 1.0, "dead_zone_px": 0, "max_step_px": 5000},
    )


# The live session's environment is installed here and ONLY here, so that the
# refactor changes this function and not the scenarios or the golden file.
def run_scenario(script: Script, **live_kwargs: Any) -> Trace:
    import gf_live as L  # noqa: PLC0415

    clock = FakeClock(T0)
    inputs: list[Any] = []
    draws: list[Any] = []
    display = RecordingDisplay(draws)
    pipeline = ScriptedPipeline(script, clock)

    def stamp(kind: str, values: tuple) -> None:
        inputs.append((kind, round((clock.now() - T0) * 1000.0, 3), *values))

    world = FakeWorld(
        runner=pipeline,
        display=display,
        clock=clock,
        escape=lambda: script.escape_at is not None and clock.now() - T0 >= script.escape_at,
        key_down=lambda vk: any(a <= clock.now() - T0 < b for a, b in script.space_down),
        foreground=lambda: script.foreground(clock.now() - T0),
        title=lambda hwnd: f"window {hwnd}",
        shutdown_result="library shut",
        on_send=stamp,
    )
    out = io.StringIO()
    returncode: Any = None
    kwargs: dict[str, Any] = {
        "move_cursor": True, "confirmed": True, "click_by": "wink", "skip_model_check": True,
    }
    kwargs.update(live_kwargs)
    with contextlib.redirect_stdout(out):
        try:
            returncode = L.run_live(profile(), env=world.environment(), **kwargs)
        except SystemExit as exc:
            returncode = f"SystemExit: {exc}"
    return Trace(returncode, inputs, draws, _normalise(out.getvalue()))
