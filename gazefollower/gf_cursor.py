"""The Windows cursor adapter: the first code here that touches the real OS.

Everything else in this directory is forbidden from emitting input.  This file
is the single exception, and it is written to be easy to audit: one function
calls Windows, it moves the pointer and nothing else, and it refuses unless it
was switched on explicitly.

**No clicks.**  Nothing here presses a button, and there is no code path that
could.  Moving a pointer is recoverable by moving it back; a click is not, and
selection at rest still produced one unintended activation per minute.

Safety, and why each rule is here rather than "later"
-----------------------------------------------------

* **Off unless asked.**  ``enabled`` defaults to False and the CLI needs
  ``--i-mean-it`` on top of ``--move-cursor``.  CLAUDE.md 4.1 requires OS
  input to be opt-in and separated from simulation; the default path here is
  a dry run that prints where the cursor *would* go.
* **Freeze on loss.**  No point, a stale point, a blink, a lost face: the
  cursor stops where it is.  It is never moved to a guess, and never left
  drifting toward one.  A frozen pointer is recoverable; a pointer wandering
  on bad data is not.
* **A verified ruler.**  Refuses to start unless ``gf_screen_check`` passes.
  M3-00: no real cursor on an unverified ruler.
* **A pause that always works.**  The hand-free gesture toggles movement, and
  Esc stops everything. The gaze dot keeps being drawn while paused, so the
  operator can see tracking is alive without the pointer moving.
* **Rate limited.**  A per-frame cap on how far the pointer may jump. A
  prediction that leaps across the screen is far more likely to be an error
  than an intention, and an unbounded jump is what makes a bad frame
  unrecoverable rather than merely wrong.
* **Released on exit.**  The pointer is put back where it started when the
  session ends, including on an exception.
"""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CursorLimits:
    """Bounds on what a single frame is allowed to do.

    ``max_step_px`` is the one that matters: the gaze point can jump the width
    of the screen between frames when a prediction goes wrong, and a pointer
    that follows it there has taken the desktop away from the operator. The
    cap turns a bad frame into a small wrong movement instead.
    """

    max_step_px: int = 400
    min_interval_s: float = 1.0 / 60.0
    # Smoothing applied to the POINTER only. The gaze point itself is left
    # exactly as the profile produces it, because that filter is part of the
    # configuration verified at 20/20 on the selection task and changing it
    # would invalidate that result. What is smoothed here is presentation.
    #
    # 0.35 at ~32 Hz follows a deliberate look within a few frames while
    # absorbing the micro-movement the eye makes constantly and never stops
    # making: a pointer that reproduces every one of those is accurate and
    # unusable at the same time.
    smoothing: float = 0.35
    # Below this the pointer simply does not move. A gaze never holds
    # perfectly still, so without a floor the pointer jitters forever around
    # a target the person is looking at steadily.
    dead_zone_px: int = 12


class CursorAdapter:
    """Moves the real pointer, or pretends to.

    ``enabled=False`` is a full simulation: it computes and records every
    position but calls nothing. The two paths differ in exactly one line, so
    what is tested in simulation is what runs for real.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        limits: CursorLimits | None = None,
        setter: Any = None,
        getter: Any = None,
        clock: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limits = limits or CursorLimits()
        self._set = setter or _set_cursor_pos
        self._get = getter or _get_cursor_pos
        self.moves = 0
        self.frozen_frames = 0
        self.clamped_steps = 0
        self.last: tuple[int, int] | None = None
        self._smooth: tuple[float, float] | None = None
        self.held_frames = 0
        self.held_for_click_frames = 0
        self.held_until_s: float | None = None
        self.origin: tuple[int, int] | None = None
        self.paused = False
        import time  # noqa: PLC0415

        self._now = clock or time.monotonic

    def hold_for(self, seconds: float) -> None:
        """Stop following the gaze for a moment, starting now.

        Called after a click so the pointer stays where the click landed. The
        person needs a beat to see what happened, and a second click meant as
        a double has to reach the same pixel.
        """

        if seconds > 0.0:
            self.held_until_s = self._now() + seconds

    def release_hold(self) -> None:
        """Follow the gaze again at once, whatever time was left."""

        self.held_until_s = None

    def jump_to(self, target: tuple[int, int]) -> tuple[int, int]:
        """Put the pointer exactly here, ignoring smoothing and the dead zone.

        Those exist to make FOLLOWING a gaze comfortable: 60% of the way per
        frame, and no movement at all under six pixels. Both are wrong for
        placing a click, which has one destination and one chance to reach it
        -- passed through ``update`` the pointer stops short and the click
        lands somewhere between where it was and where it was aimed.

        The jump limit does not apply either: this is not a gaze estimate that
        might be wild, it is a point the caller has already decided on.
        """

        placed = (int(target[0]), int(target[1]))
        if self.enabled:
            self._set(*placed)
        self.last = placed
        # Dropped so the next frame of following starts from here rather than
        # sliding back from wherever the smoothing had got to.
        self._smooth = (float(placed[0]), float(placed[1]))
        self.moves += 1
        return placed

    def __enter__(self) -> CursorAdapter:
        self.origin = self._get() if self.enabled else None
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def release(self) -> None:
        """Put the pointer back where it was found.

        Called on normal exit and on exception alike: a session that crashed
        must not leave the pointer parked wherever the last gaze happened to
        land, with the operator unable to reach the thing that would fix it.
        """

        if self.enabled and self.origin is not None:
            # Suppressed on purpose: this runs during shutdown, including
            # shutdown caused by an exception. A failure to put the pointer
            # back must not replace the original error with its own.
            with contextlib.suppress(Exception):
                self._set(*self.origin)
        self.origin = None

    def _limit(self, target: tuple[int, int]) -> tuple[int, int]:
        if self.last is None:
            return target
        dx = target[0] - self.last[0]
        dy = target[1] - self.last[1]
        cap = self.limits.max_step_px
        if abs(dx) <= cap and abs(dy) <= cap:
            return target
        self.clamped_steps += 1
        return (
            self.last[0] + max(-cap, min(cap, dx)),
            self.last[1] + max(-cap, min(cap, dy)),
        )

    def update(self, target: tuple[int, int] | None) -> tuple[int, int] | None:
        """One frame. ``None`` means freeze: no tracking, stale, or blinking.

        Returns the position the pointer holds afterwards, which is the
        previous one whenever the frame was refused. Freezing is a decision
        and is counted, so "the cursor did not move" can be told apart from
        "nothing was asked of it".
        """

        if self.held_until_s is not None and self._now() < self.held_until_s:
            # Held on purpose after a click, not frozen for want of a point.
            # Two reasons: the click must land where it was aimed rather than
            # where the gaze had already moved on to, and a second click for a
            # DOUBLE has to reach the same pixel -- which it cannot if the
            # pointer is chasing the eye between the two.
            self.held_for_click_frames += 1
            return self.last

        if target is None or self.paused:
            self.frozen_frames += 1
            # The smoothing history is dropped, not kept: resuming from a
            # position the gaze left long ago would slide the pointer across
            # the screen from a stale start.
            self._smooth = None
            return self.last

        if self._smooth is None:
            self._smooth = (float(target[0]), float(target[1]))
        else:
            a = self.limits.smoothing
            self._smooth = (
                self._smooth[0] + a * (target[0] - self._smooth[0]),
                self._smooth[1] + a * (target[1] - self._smooth[1]),
            )
        smoothed = (int(round(self._smooth[0])), int(round(self._smooth[1])))

        if self.last is not None:
            dx = smoothed[0] - self.last[0]
            dy = smoothed[1] - self.last[1]
            if abs(dx) < self.limits.dead_zone_px and abs(dy) < self.limits.dead_zone_px:
                self.held_frames += 1
                return self.last

        limited = self._limit(smoothed)
        if self.enabled:
            self._set(*limited)
        self.last = limited
        self.moves += 1
        return limited

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "moves": self.moves,
            "frozen_frames": self.frozen_frames,
            "clamped_steps": self.clamped_steps,
            "held_by_dead_zone": self.held_frames,
            "smoothing": self.limits.smoothing,
            "dead_zone_px": self.limits.dead_zone_px,
            "max_step_px": self.limits.max_step_px,
            "clicks": 0,
            "note": "this adapter cannot click; there is no code path that presses a button",
        }


def _set_cursor_pos(x: int, y: int) -> None:
    """The only line in this project that moves the real pointer."""

    ctypes.windll.user32.SetCursorPos(int(x), int(y))  # type: ignore[attr-defined]


def _get_cursor_pos() -> tuple[int, int]:
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    point = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))  # type: ignore[attr-defined]
    return (int(point.x), int(point.y))
