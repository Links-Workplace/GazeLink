"""Monotonic time as an injected dependency (TECHNICAL_SPEC 4.5).

Every control-time comparison in a session reads ONE clock. Production uses
:class:`MonotonicClock`; tests and replay use :class:`FakeClock`, which moves
only when something sleeps or the test advances it.

Seconds, float, monotonic. Never wall-clock/UTC -- that is for reports only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    """The process monotonic clock, looked up at call time.

    Looked up rather than bound at construction so a characterisation harness
    that replaces ``time.monotonic`` for a whole process sees one timeline.
    """

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """Deterministic time. ``sleep`` advances it; hooks run after each advance."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = float(start)
        self.on_advance: list[Callable[[float], None]] = []

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not run backwards")
        self._now += float(seconds)
        for hook in list(self.on_advance):
            hook(self._now)
