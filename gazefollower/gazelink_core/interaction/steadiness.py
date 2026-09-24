"""The point a DWELL may use while an eyelid is on its way down.

Why this exists
---------------

``InteractionController`` gates the gaze on ``eyes_steady`` before anything may
use it: an eye halfway behind its lid still passes the blink gate, so the point
it produces is real but wrong, and following it makes the pointer lurch as a
wink begins. That gate is right for the POINTER and for winks.

It is wrong for a dwell, and measurably so. ``DwellEngine.update(point=None)``
clears ``_start_s`` -- the fill restarts from zero, with no tolerance for a gap.
Measured on ``recordings/round47/T1`` (904 frames, 32.5 ms apart) replayed
through this person's own gate from ``profiles/group1_corrected_trial.json``
(``shut_fraction`` 0.55, ``steady_fraction`` 0.80, ``adapt`` 0.02):

* ``eyes_steady`` is true on **93.7%** of frames -- which sounds like nothing.
* But a 900 ms fill at 32.3 fps needs **29 consecutive** steady frames, and the
  steady runs had a **median of 5**. Only **7 of 31** were long enough to
  finish one dwell.
* The scattered 6.3% is what made the desk bar feel dead while the drawn dot
  sat exactly on the tile: the dot comes from the ungated point, the fill does
  not. ``gf_dwell_practice`` has no such gate, fed the engine every frame, and
  scored 17/18 on the same model and the same screen.

So the gaps are bridged rather than the gate removed.

Where the number comes from
---------------------------

The gaps are short: median 1 frame (32 ms), P90 3 frames (101 ms), max 9 frames
(292 ms). That maximum is a natural blink, and this project has already measured
both ends of that scale on this person: **natural blinks reached 297 ms** and
**deliberate holds started at 1172 ms**, with the confirm threshold at 800 ms.

``DEFAULT_HOLD_MS`` therefore sits above the longest natural blink and far below
the shortest deliberate close: a blink must not cancel a fill the person is
still committed to, and a deliberate close must never be bridged, because that
is the gesture that opens the menu. It is a parameter and never a literal in
the loop (TECHNICAL_SPEC 12, CLAUDE.md 4.2).

The budget, and why holding is not free
---------------------------------------

A plain "hold for up to N ms" can be chained: one steady frame re-enables
another full N ms, so a fill could complete on almost nothing but held frames.
Instead the hold draws on a **leaky budget** -- held frames spend it, steady
frames refill it, both at real time and capped at ``hold_ms``. Over any window
the held time therefore cannot exceed the steady time, and a 900 ms fill always
contains at least 550 ms the eyes were genuinely open for.

Pure state: no camera, no clock of its own, no OS input. The caller passes time.
"""

from __future__ import annotations

# Above the longest natural blink measured on this person (297 ms) and far below
# the shortest deliberate hold (1172 ms; the confirm sits at 800 ms).
DEFAULT_HOLD_MS = 350.0


class SteadyHold:
    """Bridges brief unsteadiness for a dwell, on a budget it has to earn.

    One instance per thing that fills. It owns exactly two pieces of state --
    the last point the eyes were open for, and how much holding is left -- and
    nothing else reads or writes them.
    """

    def __init__(self, hold_ms: float = DEFAULT_HOLD_MS) -> None:
        if not hold_ms > 0.0:
            raise ValueError("hold_ms must be positive")
        self.hold_s = float(hold_ms) / 1000.0
        self._point: tuple[float, float] | None = None
        self._budget_s = self.hold_s
        self._last_s: float | None = None
        # Reported, not merely enforced: a session has to be able to say how
        # much of its dwelling happened behind an eyelid.
        self.held_frames = 0
        self.bridged_gaps = 0
        self.exhausted = 0
        self._in_gap = False

    def reset(self) -> None:
        """Forget the point and refill the budget. For a mode change or a loss."""

        self._point = None
        self._budget_s = self.hold_s
        self._last_s = None
        self._in_gap = False

    @property
    def holding(self) -> bool:
        return self._in_gap and self._point is not None

    def update(
        self,
        now_s: float,
        *,
        steady_point: tuple[float, float] | None,
        have_prediction: bool,
    ) -> tuple[float, float] | None:
        """The point a dwell may use this frame.

        ``steady_point`` is the gaze when the eyes are open, else None.
        ``have_prediction`` is whether a point existed at all: no face and no
        prediction is a tracking loss, never a blink, and nothing is bridged
        through it -- CLAUDE.md 4.1 requires a loss to stop things, not to be
        smoothed over.
        """

        dt_s = 0.0 if self._last_s is None else max(0.0, now_s - self._last_s)
        self._last_s = now_s

        if steady_point is not None:
            # Eyes open: the point is the truth, and steadiness earns budget back.
            self._point = steady_point
            self._budget_s = min(self.hold_s, self._budget_s + dt_s)
            self._in_gap = False
            return steady_point

        if not have_prediction:
            # A tracking loss. Not a blink: nothing is bridged through it, and
            # the budget resets because recovery starts from a clean sheet.
            self._point = None
            self._budget_s = self.hold_s
            self._in_gap = False
            return None

        if self._point is None:
            # Nothing steady seen yet, or the budget already ran out inside this
            # closure. Deliberately NOT a refill: refilling here was the
            # chaining hole -- every frame of a long close handed back a full
            # hold, so one steady frame bought another 350 ms indefinitely.
            # ``HoldingIsEarnedTests`` caught it and fails if it comes back.
            self._in_gap = False
            return None

        if self._budget_s - dt_s <= 0.0:
            # Spent. A long closure reaches here, and it must: the deliberate
            # hold is a GESTURE, and bridging it would let the menu's own
            # signal finish a dwell on the way.
            self._budget_s = 0.0
            if self._in_gap:
                self._in_gap = False
                self.exhausted += 1
            self._point = None
            return None

        self._budget_s -= dt_s
        if not self._in_gap:
            self._in_gap = True
            self.bridged_gaps += 1
        self.held_frames += 1
        return self._point

    def summary(self) -> dict[str, object]:
        return {
            "hold_ms": round(self.hold_s * 1000.0, 1),
            "frames bridged behind an eyelid": self.held_frames,
            "gaps bridged": self.bridged_gaps,
            "gaps too long to bridge": self.exhausted,
        }
