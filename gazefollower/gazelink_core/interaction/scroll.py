"""Gaze-driven scrolling: the zones, and the repeat that drives them.

No camera, no OS input, no drawing.  Everything here is arithmetic over
(time, which zone the gaze is in), so the whole behaviour can be tested
deterministically -- which matters more here than for a click, because a
scroll REPEATS.  A click that misfires costs one click; a repeat that misfires
keeps going until something stops it, and the thing that stops it has to be
tested, not assumed.

Why the zones are horizontal bands, which the rest of this project says not to
do.  ``gf_dwell`` measured the bias on this rig at 0.056 of screen width but
**0.387 of screen height**, and concluded that targets should be full-height
columns rather than a grid.  Bands at the top and bottom are the opposite of
that advice, and they are used here anyway for two reasons:

* scrolling is REVERSIBLE.  The cost of the weak axis is "I looked and nothing
  happened", or "it went the wrong way and I scrolled back" -- not a click on
  the wrong thing.  This is the one capability that can afford that axis.
* the middle of the screen is a large dead zone, so ordinary reading never
  scrolls, and the failure mode of a large bias is a MISS rather than a
  reversal.

That is an argument for trying it, not a proof that it works.  Whether the
bands are actually reachable is a measurement the first live run has to make,
which is why the geometry here is parameters and not constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gazelink_core.interaction import dwell as D

# The validated horizontal band, same one gf_dwell uses. Outside it the model
# is not merely less accurate; nothing is ever placed there.
BAND = D.BAND

UP = "SCROLL UP"
DOWN = "SCROLL DOWN"
TILE = "SCROLL"


@dataclass(frozen=True)
class ScrollConfig:
    """How long before scrolling starts, and how fast it then goes.

    Both numbers are OPENING GUESSES.  Nothing on this rig has ever measured a
    comfortable scroll rate, and no measurement in the repository constrains
    them, so they are exposed as flags in the same way ``--wink-hold-ms`` was
    in section 56: the report prints what happened, and the numbers move when
    a measurement says so.
    """

    # Long enough that crossing a band on the way somewhere else does not
    # scroll; short enough that asking to scroll does not feel like waiting.
    arm_ms: float = 500.0
    # One wheel notch per this long. A notch is three lines on most Windows
    # setups. Opened at 500 ms, which the operator read as "very slow" on the
    # first live run -- about twelve seconds to move one screenful. 150 ms was
    # tried next and kept: roughly three and a half seconds a screenful, which
    # is reading pace. Chosen by using it, the same way the cursor smoothing
    # in the profile was, and not reasoned about from anything.
    repeat_ms: float = 150.0
    # How far the top and bottom bands reach in from the edge, and how far
    # they sit from it. Reachability is the open question, so this is a knob.
    band_height: float = 0.20
    band_margin: float = 0.04

    def __post_init__(self) -> None:
        for name in ("arm_ms", "repeat_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not 0.0 < self.band_height < 0.5:
            raise ValueError("band_height must be between 0 and 0.5")
        if not 0.0 <= self.band_margin < 0.5:
            raise ValueError("band_margin must be at least 0 and under 0.5")
        if self.band_margin + self.band_height >= 0.5:
            # Otherwise the two bands meet in the middle and there is nowhere
            # left to rest the gaze without scrolling.
            raise ValueError("the bands would leave no dead zone in the middle")


def scroll_zones(config: ScrollConfig | None = None, *, band: tuple[float, float] = BAND):
    """The two bands, top and bottom, across the validated horizontal band."""

    cfg = config or ScrollConfig()
    x0, x1 = band
    top = cfg.band_margin
    return [
        D.Button(UP, x0, top, x1, top + cfg.band_height),
        D.Button(DOWN, x0, 1.0 - top - cfg.band_height, x1, 1.0 - top),
    ]


def scroll_tile(config: ScrollConfig | None = None, *, band: tuple[float, float] = BAND):  # noqa: ARG001
    """The way in and the way out, as one target.

    Deliberately NOT in the middle of the screen and NOT inside either band.
    The middle is where the gaze rests to STOP scrolling, so a tile there would
    turn every stop into an exit; and a tile inside a band could not be looked
    at without scrolling first.  It sits at the side, vertically centred, in
    the dead zone.
    """

    x0, x1 = band
    width = (x1 - x0) * 0.25
    return D.Button(TILE, x0, 0.42, x0 + width, 0.58)


@dataclass
class ScrollRepeater:
    """One zone at a time, and it stops the moment it is not being asked.

    The dwell engine cannot do this and must not be made to: it fires once per
    entry and latches, which is the rule that keeps a click from repeating
    (``gf_dwell`` module docstring, and ``test_holding_far_longer_still
    _activates_only_once``).  Scrolling needs the opposite behaviour, so it
    gets its own state rather than a weakened version of that one.
    """

    config: ScrollConfig = field(default_factory=ScrollConfig)
    _zone: str | None = None
    _since_s: float | None = None
    _last_tick_s: float | None = None

    @property
    def armed(self) -> bool:
        """Past the wait, so the next tick is due on the clock alone."""

        return self._last_tick_s is not None

    @property
    def zone(self) -> str | None:
        return self._zone

    def progress(self, now_s: float) -> float:
        """0..1 through the wait before scrolling starts, for the screen."""

        if self._since_s is None or self.armed:
            return 1.0 if self.armed else 0.0
        waited = (now_s - self._since_s) * 1000.0
        return min(1.0, max(0.0, waited / self.config.arm_ms))

    def stop(self) -> None:
        """Forget the wait as well as the repeat.

        Both, deliberately: leaving the wait behind would mean that glancing
        away and back resumes instantly, and the wait exists precisely so that
        a glance is not an instruction.
        """

        self._zone = None
        self._since_s = None
        self._last_tick_s = None

    def update(self, now_s: float, zone: str | None, *, usable: bool = True) -> int:
        """One frame. Returns the notches to send: +1 up, -1 down, 0 nothing.

        ``usable`` is the caller's answer to "is there a trustworthy gaze
        point right now" -- fresh, face present, not paused.  Passed in rather
        than worked out here, for the same reason ``ClickAdapter`` takes
        ``armed``: a module that can decide its own permission is a module
        that can be wrong about it alone.
        """

        if not usable or zone not in (UP, DOWN):
            # Includes the gaze resting in the middle, which is how a person
            # stops: not a special case, just the absence of a request.
            self.stop()
            return 0
        if zone != self._zone:
            self._zone = zone
            self._since_s = now_s
            self._last_tick_s = None
            return 0
        step = 1 if zone == UP else -1
        if self._last_tick_s is None:
            if self._since_s is None:
                self._since_s = now_s
                return 0
            if (now_s - self._since_s) * 1000.0 < self.config.arm_ms:
                return 0
            # The first notch lands the moment the wait is over, so the person
            # sees that the request registered without waiting a second time.
            self._last_tick_s = now_s
            return step
        if (now_s - self._last_tick_s) * 1000.0 < self.config.repeat_ms:
            return 0
        self._last_tick_s = now_s
        return step
