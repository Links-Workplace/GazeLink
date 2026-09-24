"""What the magnifier needs from the screen, and nothing about how to get it.

The contract only. The library that actually reads pixels lives in
``platform.screen_capture``; ``interaction`` may not import it, and the
architecture test now enforces that by name (``PIL`` is a forbidden root).

Everything here is in DEVICE pixels of the virtual desktop, because that is
the space ``platform.screen_check.to_desktop_pixels`` puts a click in. A
magnifier that captured in one space and clicked in another would be confident
and wrong, which is the worst thing this module could be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DeviceRect:
    """A rectangle in device pixels, right and bottom exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"empty or inverted rectangle: {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def around(self, x: int, y: int, radius: int) -> DeviceRect:
        """A small box centred on a point, clipped to this rectangle.

        Used for the freshness check. A NEIGHBOURHOOD and not the whole
        region: a blinking caret in one corner, or a clock in the taskbar,
        must not block a click on a button at the other end.
        """

        if radius < 1:
            raise ValueError("a neighbourhood needs a positive radius")
        x0 = max(self.x0, min(x - radius, self.x1 - 1))
        y0 = max(self.y0, min(y - radius, self.y1 - 1))
        x1 = min(self.x1, max(x + radius, x0 + 1))
        y1 = min(self.y1, max(y + radius, y0 + 1))
        return DeviceRect(x0, y0, x1, y1)


@runtime_checkable
class Capture(Protocol):
    """One frozen picture of a region. Held in memory, never written out."""

    @property
    def rect(self) -> DeviceRect: ...

    def crop_bytes(self, rect: DeviceRect) -> bytes:
        """The pixels of ``rect``, which must lie inside ``self.rect``."""


@runtime_checkable
class ScreenCapturePort(Protocol):
    def grab(self, rect: DeviceRect) -> Capture:
        """Read that region of the screen right now."""
