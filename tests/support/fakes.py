"""In-memory test doubles that never touch a camera or operating-system input."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

FrameT = TypeVar("FrameT")


class FakeClock:
    """A monotonic clock controlled entirely by a test.

    Times are expressed in seconds.  Production code must use its own monotonic
    clock; this fake makes elapsed-time and cooldown tests reproducible.
    """

    def __init__(self, *, initial_seconds: float = 0.0) -> None:
        if initial_seconds < 0:
            raise ValueError("initial_seconds must be non-negative")
        self._seconds = initial_seconds

    def now(self) -> float:
        """Return the current synthetic monotonic time in seconds."""

        return self._seconds

    def advance(self, seconds: float) -> float:
        """Advance time by a non-negative duration and return the new time."""

        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self._seconds += seconds
        return self._seconds


class FakeCamera(Generic[FrameT]):
    """A bounded, in-memory camera source for synthetic frame objects.

    The fake accepts arbitrary synthetic objects so the test layer never needs
    to keep a face image or a biometric recording on disk.
    """

    def __init__(self, frames: Iterable[FrameT] = ()) -> None:
        self._frames: deque[FrameT] = deque(frames)
        self._is_open = False
        self.open_count = 0
        self.close_count = 0

    @property
    def is_open(self) -> bool:
        """Whether the synthetic source is currently available for reads."""

        return self._is_open

    @property
    def pending_frame_count(self) -> int:
        """Return the number of in-memory frames that have not been read."""

        return len(self._frames)

    def open(self) -> None:
        """Open the synthetic source; no device is enumerated or acquired."""

        if not self._is_open:
            self._is_open = True
            self.open_count += 1

    def close(self) -> None:
        """Close the synthetic source without persisting queued objects."""

        if self._is_open:
            self._is_open = False
            self.close_count += 1

    def read(self) -> FrameT | None:
        """Return the next queued frame, or ``None`` at end of stream."""

        if not self._is_open:
            raise RuntimeError("FakeCamera must be opened before read")
        if not self._frames:
            return None
        return self._frames.popleft()

    def append(self, frame: FrameT) -> None:
        """Append a synthetic frame to the in-memory stream."""

        self._frames.append(frame)


@dataclass(frozen=True, slots=True)
class FakeInputEvent:
    """A requested input operation recorded only in memory."""

    name: str
    arguments: tuple[object, ...]


class FakeInput:
    """Record requested input commands without moving a real cursor or clicking."""

    def __init__(self) -> None:
        self._events: list[FakeInputEvent] = []

    @property
    def events(self) -> tuple[FakeInputEvent, ...]:
        """Return an immutable snapshot of the command log."""

        return tuple(self._events)

    def emit(self, name: str, *arguments: object) -> None:
        """Record a command request; this has no external side effect."""

        self._events.append(FakeInputEvent(name=name, arguments=arguments))

    def clear(self) -> None:
        """Discard the in-memory command history."""

        self._events.clear()
