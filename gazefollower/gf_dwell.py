"""Dwell selection: resting the gaze on a target activates it, once.

Pure state machine and geometry -- no camera, no pygame, no OS input.  Fed
from the same filtered point the live view draws, so what activates is exactly
what the operator can see.

Why the shape of this is what it is
-----------------------------------

Measured on this rig, the error is **bias-dominated**: the gaze cluster is
tight (within-fixation P95 of 58-77 px) but sits in the wrong place, by up to
446 px.  Dwell integrates over its window, so it removes scatter and removes
none of the bias.  Two consequences run through everything here:

* Sizing targets on the median error (189 px) would fail at exactly the
  positions where the bias is worst.
* A systematic offset fails in two different ways with two different
  thresholds.  It MISSES once it exceeds a target's half-width, and it reaches
  the NEIGHBOUR only once it exceeds half-width plus the gap -- the gap alone
  is not the margin, because half the target lies between its centre and the
  dead space.  Missing is tolerable; activating the wrong thing confidently is
  not, so layouts are sized to keep the second case out of reach even when
  they accept the first.

The horizontal and vertical axes are not comparable: worst bias is 0.056 of
screen width but 0.387 of screen height.  That is why the layouts here are
full-height columns rather than a grid.

Safety rules, each with a test
------------------------------

* One activation per entry.  After firing, a target is latched until the point
  is seen somewhere else; a time-only cooldown is not enough.  This is the
  "repeated or stuck action" failure in CLAUDE.md 4.1 and the "event old or
  duplicate" refusal in TECHNICAL_SPEC.md 8.3, and it is the same latch that
  `gf_gesture.EyeCloseDetector` already needed after one five-second hold
  produced three confirms.
* Losing the point never clears the latch.  Absence is not evidence of having
  looked away, and treating it as such would let a blink re-arm the target the
  operator is still resting on.
* A stale point cancels progress.  A frozen prediction that keeps "resting" on
  a target would otherwise activate it while the camera is dead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Phase(StrEnum):
    IDLE = "idle"  # no usable point, or the point is in dead space
    FILLING = "filling"  # resting on a target, accumulating toward activation
    LATCHED = "latched"  # already activated; will not fire again until it exits


@dataclass(frozen=True)
class Button:
    """A target in normalised screen coordinates."""

    key: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(f"button {self.key!r} has an empty or inverted rectangle")

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class DwellConfig:
    """Durations in milliseconds.

    ``dwell_ms`` starts at the value the product already declares --
    ``GestureTimingConfig.dwell_duration_ms = 900`` in src/gazelink/config.py
    -- rather than a second number invented here.  It is meant to be tuned
    against a real person; TECHNICAL_SPEC.md 12 requires dwell to be
    adjustable, so it is a parameter and never a literal in the loop.
    """

    dwell_ms: float = 900.0
    stale_after_s: float = 0.25

    def __post_init__(self) -> None:
        if self.dwell_ms <= 0.0:
            raise ValueError("dwell_ms must be positive")
        if self.stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")


@dataclass(frozen=True)
class Activation:
    """One selection.

    Field names deliberately mirror ``InteractionEvent`` in
    src/gazelink/domain.py so that porting this to the product later is a
    mapping rather than a rewrite.  Nothing is imported across the two trees;
    they are kept separate on purpose.
    """

    button: str
    occurred_at_monotonic_ms: float
    position: tuple[float, float]
    dwell_ms: float
    event_type: str = "DWELL_SELECT"
    source: str = "DWELL"
    requires_active_control: bool = True


@dataclass
class DwellEngine:
    """Turns a stream of gaze points into at most one activation per entry.

    ``update`` is called once per displayed frame.  It is deliberately given
    no information about which target the operator was ASKED to look at: a
    selection mechanism that can see the answer is not measuring selection.
    """

    buttons: list[Button]
    config: DwellConfig = field(default_factory=DwellConfig)

    def __post_init__(self) -> None:
        keys = [b.key for b in self.buttons]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate button keys: {keys}")
        self._current: str | None = None
        self._start_s: float | None = None
        self._latched: str | None = None
        self._progress: float = 0.0

    # -- state the display reads -------------------------------------------

    @property
    def progress(self) -> float:
        """0..1 for the ring. Zero unless a target is actively filling."""

        return self._progress

    @property
    def hovered(self) -> str | None:
        return self._current

    @property
    def phase(self) -> Phase:
        if self._latched is not None and self._latched == self._current:
            return Phase.LATCHED
        return Phase.FILLING if self._current is not None else Phase.IDLE

    def reset(self) -> None:
        """Forget everything in flight.

        Called on pause, resume, profile switch and recalibration: dwell
        progress accumulated against one model must not carry into another.
        """

        self._current = None
        self._start_s = None
        self._latched = None
        self._progress = 0.0

    # -- the machine --------------------------------------------------------

    def find(self, point: tuple[float, float]) -> Button | None:
        for button in self.buttons:
            if button.contains(point):
                return button
        return None

    def update(
        self,
        now_s: float,
        point: tuple[float, float] | None,
        *,
        fresh: bool = True,
    ) -> Activation | None:
        """One frame. Returns an activation only on the frame it fires.

        ``point`` is None when there is no usable gaze -- tracking lost, a
        blink, a prediction that did not resolve.  ``fresh`` is False when the
        point exists but its source frame is older than ``stale_after_s``.
        """

        if point is None or not fresh:
            # Cancel the fill, but keep the latch: absence is not evidence
            # that the operator looked away, and clearing it here would let a
            # blink re-arm the target they are still resting on.
            self._current = None
            self._start_s = None
            self._progress = 0.0
            return None

        button = self.find(point)
        if button is None:
            # Dead space. Seeing the point outside every target IS evidence of
            # having left, so this is where the latch clears.
            self._current = None
            self._start_s = None
            self._latched = None
            self._progress = 0.0
            return None

        if button.key != self._current:
            # Entered a different target: progress never carries across.
            self._current = button.key
            self._start_s = now_s
            self._progress = 0.0
            if self._latched is not None and self._latched != button.key:
                self._latched = None
            return None

        if self._latched == button.key:
            self._progress = 1.0
            return None

        if self._start_s is None:
            self._start_s = now_s
        held_ms = (now_s - self._start_s) * 1000.0
        self._progress = min(1.0, max(0.0, held_ms / self.config.dwell_ms))
        if held_ms < self.config.dwell_ms:
            return None

        self._latched = button.key
        self._progress = 1.0
        return Activation(
            button=button.key,
            occurred_at_monotonic_ms=now_s * 1000.0,
            position=point,
            dwell_ms=held_ms,
        )


# --- layouts -----------------------------------------------------------------

# The validated band. Outside it the model is not merely less accurate: on
# round4 the targets beyond the band read 964 px against 234 px inside it, so
# nothing is ever placed there.
BAND = (0.30, 0.70)


def column_layout(
    keys: list[str],
    *,
    band: tuple[float, float] = BAND,
    gap: float,
    top: float = 0.05,
    bottom: float = 0.95,
) -> list[Button]:
    """Evenly spaced full-height columns inside the band.

    Full height because the vertical bias reaches 0.387 of the screen height:
    a target short enough to sit in a grid row would be missed vertically at
    the positions where the bias is worst, whatever its width.
    """

    if len(keys) < 1:
        raise ValueError("a layout needs at least one target")
    lo, hi = band
    span = hi - lo
    total_gap = gap * (len(keys) - 1)
    width = (span - total_gap) / len(keys)
    if width <= 0.0:
        raise ValueError(f"{len(keys)} targets with gap {gap} do not fit in a band {span:.2f} wide")
    out = []
    for i, key in enumerate(keys):
        x0 = lo + i * (width + gap)
        out.append(Button(key, x0, top, x0 + width, bottom))
    return out


def layout_a(gap: float = 0.08) -> list[Button]:
    """Two columns with a dead zone ~1.4x the worst measured horizontal bias.

    The pass/fail layout: a horizontally biased point should land in the gap
    and activate nothing, rather than confidently activating the other target.
    """

    return column_layout(["LEFT", "RIGHT"], gap=gap)


def layout_b(gap: float = 0.05) -> list[Button]:
    """Three columns whose half-width is deliberately UNDER the worst bias.

    Expected to produce MISSES: half-width 0.050 against a worst measured bias
    of 0.056. The neighbour stays out of reach (that needs 0.100), so this
    finds the point where selection stops working, not where it goes wrong.
    """

    return column_layout(["LEFT", "MIDDLE", "RIGHT"], gap=gap)


LAYOUTS = {"a": layout_a, "b": layout_b}


def layout_warnings(buttons: list[Button], *, bias_x: float, bias_y: float) -> list[str]:
    """Where this layout is too tight for the bias actually measured.

    Two different failures, with two different thresholds, and confusing them
    is easy: a target is MISSED once the bias exceeds its half-width, but the
    NEIGHBOUR is only reached once the bias exceeds half-width plus the gap.
    The gap alone is not the margin -- half the target sits between its centre
    and the dead space, and that half counts.

    Missing is the tolerable failure and hitting the neighbour is the
    dangerous one, so they are reported separately rather than as one score.
    Reported before a run, not inferred from a bad result afterwards.
    """

    out: list[str] = []
    ordered = sorted(buttons, key=lambda b: b.x0)
    for button in ordered:
        half = button.width / 2.0
        if half <= bias_x:
            out.append(
                f"MISS RISK: {button.key} half-width {half:.3f} is at or under the worst "
                f"horizontal bias {bias_x:.3f}; expect non-activations"
            )
        half_h = button.height / 2.0
        if half_h <= bias_y:
            out.append(
                f"MISS RISK: {button.key} half-height {half_h:.3f} is at or under the worst "
                f"vertical bias {bias_y:.3f}; expect non-activations"
            )
    for left, right in zip(ordered, ordered[1:], strict=False):
        gap = right.x0 - left.x1
        reach = left.width / 2.0 + gap
        if reach <= bias_x:
            out.append(
                f"NEIGHBOUR RISK: from {left.key} the next target starts {reach:.3f} from its "
                f"centre, at or under the worst horizontal bias {bias_x:.3f}; a biased point "
                f"can activate {right.key} instead"
            )
    return out
