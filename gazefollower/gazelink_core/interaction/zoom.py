"""Precise selection: one stable magnification, and the checks before a click.

Pure state machine. No camera, no pygame, no ctypes, no PIL -- the screen is
read through ``capture_port.ScreenCapturePort``, which a fake implements in
tests.

What this is for, and what it is not
------------------------------------

It does not improve the gaze. It turns a limited prediction into an exact
click: the region is shown large enough that the gaze CAN land where the
person means, and the click is sent back to the original coordinates.

The ordinary route does not come through here. A click is a look and a
confirm. ``open`` is reached from one tile in the secondary menu, and the
person is expected to leave again immediately.

Rules that this module exists to enforce
----------------------------------------

* **The picture is frozen.** It is taken once per level and never re-read while
  the person is choosing. Nothing on screen follows the gaze here.
* **Entering never emits.** Opening the magnifier is an internal mode change;
  ``ActionRouter`` refuses every click in ``UiMode.ZOOM`` except ``ZOOM_CLICK``,
  which only the explicit confirm target produces.
* **The overlay must be layered before anything is captured.** Our own window
  is excluded from a grab only while it is layered. Unlayered, the magnifier
  would photograph its own picture and recurse on it. Refused, not risked.
* **One coordinate space.** Device rectangles are built by mapping normalised
  points through the SAME function the pointer uses, so the captured region and
  the click cannot drift apart.
* **Freshness, twice.** The foreground window must be what it was AND must not
  be ours, and the pixels around the target must be unchanged. Until the
  thresholds are measured the comparison is exact, which is the safe side and
  refuses on an animated area.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from gazelink_core.interaction.capture_port import Capture, DeviceRect, ScreenCapturePort

# How much of the screen the first magnification shows, as a fraction of
# width. The panel cannot be wider than the validated band (0.40), so this is
# the only lever there is: 0.10 in a 0.40 panel is four times.
SPAN_NEAR = 0.10
SPAN_DEEP = 0.05
SPAN_OVERVIEW = 1.0
# The neighbourhood compared before a click, in device pixels.
FRESH_RADIUS_PX = 24


class Phase(StrEnum):
    CLOSED = "closed"
    CHOOSING = "choosing"
    DONE = "done"


class Refusal(StrEnum):
    NONE = "none"
    NOT_LAYERED = "the overlay is not layered, so a capture would include it"
    NO_CAPTURE = "the screen could not be read"
    NO_POINT = "no point has been chosen inside the magnifier"
    FOREGROUND_CHANGED = "the window in front changed"
    FOREGROUND_IS_OURS = "our own window is in front"
    CONTENT_CHANGED = "the content under the point changed"


@dataclass(frozen=True)
class Region:
    """A square of the screen in normalised coordinates, clamped on screen."""

    cx: float
    cy: float
    span: float

    def clamped(self) -> Region:
        half = self.span / 2.0
        return Region(
            min(max(self.cx, half), 1.0 - half),
            min(max(self.cy, half), 1.0 - half),
            self.span,
        )

    @property
    def corners(self) -> tuple[tuple[float, float], tuple[float, float]]:
        c = self.clamped()
        half = c.span / 2.0
        return (c.cx - half, c.cy - half), (c.cx + half, c.cy + half)

    def point_at(self, fx: float, fy: float) -> tuple[float, float]:
        """A fraction of the PANEL back to a normalised screen point."""

        (x0, y0), (x1, y1) = self.corners
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)


@dataclass(frozen=True)
class Outcome:
    """What ``confirm`` decided, described rather than done."""

    allowed: bool
    reason: Refusal
    at_device: tuple[int, int] | None = None


class ZoomSelection:
    """One magnifier session: open, look, confirm, leave."""

    def __init__(
        self,
        *,
        capture: ScreenCapturePort,
        to_device: Callable[[tuple[float, float]], tuple[int, int]],
        overlay_is_layered: Callable[[], bool],
        foreground: Callable[[], int],
        our_windows: Callable[[], set[int]] | None = None,
        fresh_radius_px: int = FRESH_RADIUS_PX,
    ) -> None:
        self._capture_port = capture
        self._to_device = to_device
        self._overlay_is_layered = overlay_is_layered
        self._foreground = foreground
        self._our_windows = our_windows or (lambda: set())
        self._fresh_radius_px = fresh_radius_px
        self.phase = Phase.CLOSED
        self.region = Region(0.5, 0.5, SPAN_NEAR)
        self.point: tuple[float, float] | None = None
        self.refused = Refusal.NONE
        self._frame: Capture | None = None
        self._entry_window: int | None = None
        self._history: list[Region] = []
        # Counted, because "it refused" and "it was never asked" need
        # different fixes and look identical from outside.
        self.refusals: dict[Refusal, int] = {r: 0 for r in Refusal if r is not Refusal.NONE}
        self.captures = 0

    # -- the frozen picture --------------------------------------------------

    @property
    def frame(self) -> Capture | None:
        """The picture being shown. Replaced only on a deliberate step."""

        return self._frame

    def device_rect(self, region: Region) -> DeviceRect:
        """The region in device pixels, through the pointer's own mapping."""

        (x0, y0), (x1, y1) = region.corners
        ax, ay = self._to_device((x0, y0))
        bx, by = self._to_device((x1, y1))
        left, right = sorted((ax, bx))
        top, bottom = sorted((ay, by))
        return DeviceRect(left, top, max(right, left + 1), max(bottom, top + 1))

    def _refuse(self, reason: Refusal) -> Outcome:
        self.refusals[reason] += 1
        self.refused = reason
        return Outcome(False, reason)

    def _freeze(self, region: Region) -> Outcome:
        if not self._overlay_is_layered():
            return self._refuse(Refusal.NOT_LAYERED)
        try:
            frame = self._capture_port.grab(self.device_rect(region))
        except Exception:  # noqa: BLE001 - a failed grab never ends a session
            return self._refuse(Refusal.NO_CAPTURE)
        self._frame = frame
        self.captures += 1
        self.region = region.clamped()
        self.point = None
        self.refused = Refusal.NONE
        return Outcome(True, Refusal.NONE)

    # -- the session ---------------------------------------------------------

    def open(self, at: tuple[float, float] | None) -> Outcome:
        """Magnify around the place the person was last looking at content."""

        centre = at or (0.5, 0.5)
        self._history = []
        self._entry_window = self._foreground()
        outcome = self._freeze(Region(centre[0], centre[1], SPAN_NEAR))
        self.phase = Phase.CHOOSING if outcome.allowed else Phase.CLOSED
        return outcome

    def look(self, fx: float, fy: float) -> None:
        """Where inside the panel the gaze has settled. Emits nothing."""

        if self.phase is not Phase.CHOOSING:
            return
        self.point = (min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0))

    def deeper(self) -> Outcome:
        """One more magnification, centred on the point if there is one."""

        if self.phase is not Phase.CHOOSING or self.region.span <= SPAN_DEEP:
            return Outcome(False, self.refused)
        centre = (
            self.region.point_at(*self.point)
            if self.point
            else (self.region.cx, self.region.cy)
        )
        self._history.append(self.region)
        return self._freeze(Region(centre[0], centre[1], SPAN_DEEP))

    def overview(self) -> Outcome:
        """The whole screen inside the band, for a target outside it."""

        if self.phase is not Phase.CHOOSING:
            return Outcome(False, self.refused)
        self._history.append(self.region)
        return self._freeze(Region(0.5, 0.5, SPAN_OVERVIEW))

    def back(self) -> Outcome:
        """One step out. Always available, and never emits anything."""

        if not self._history:
            return Outcome(False, self.refused)
        return self._freeze(self._history.pop())

    def cancel(self) -> None:
        """Leave with nothing done. The picture is dropped with it."""

        self.phase = Phase.CLOSED
        self.point = None
        self._frame = None
        self._history = []
        self.refused = Refusal.NONE

    # -- the one click -------------------------------------------------------

    def confirm(self) -> Outcome:
        """May the click go, and where. Decides; never sends."""

        if self.phase is not Phase.CHOOSING:
            return self._refuse(Refusal.NO_POINT)
        if self.point is None:
            return self._refuse(Refusal.NO_POINT)
        if self._frame is None:
            return self._refuse(Refusal.NO_CAPTURE)
        now_window = self._foreground()
        if now_window in self._our_windows():
            # Our own overlay is in front. The window check would pass against
            # itself and mean nothing, so it is refused outright.
            return self._refuse(Refusal.FOREGROUND_IS_OURS)
        if self._entry_window is not None and now_window != self._entry_window:
            return self._refuse(Refusal.FOREGROUND_CHANGED)
        target = self.region.point_at(*self.point)
        device = self._to_device(target)
        if not self._fresh_at(device):
            return self._refuse(Refusal.CONTENT_CHANGED)
        self.phase = Phase.DONE
        self.refused = Refusal.NONE
        return Outcome(True, Refusal.NONE, at_device=device)

    def _fresh_at(self, device: tuple[int, int]) -> bool:
        """Did the pixels around the click change since the picture was taken?

        Exact comparison on purpose. The patient threshold has never been
        measured on the two cases that decide it -- a blinking caret and a page
        that scrolled one line -- so until it is, this refuses on an animated
        area rather than clicking on a page that moved.
        """

        frame = self._frame
        if frame is None:
            return False
        if not frame.rect.contains(*device):
            return False
        box = frame.rect.around(device[0], device[1], self._fresh_radius_px)
        try:
            before = frame.crop_bytes(box)
            after = self._capture_port.grab(box).crop_bytes(box)
        except Exception:  # noqa: BLE001 - a failed check is a refusal
            return False
        return before == after

    def summary(self) -> dict[str, object]:
        return {
            "captures": self.captures,
            "refused": {str(r): n for r, n in self.refusals.items() if n},
            "note": "this module decides; the executor is the only thing that clicks",
        }
