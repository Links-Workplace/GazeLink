"""Ordered, individually guarded, idempotent cleanup (TECHNICAL_SPEC §4.5).

Every acquisition registers its undo the moment it succeeds. ``close`` then runs
the undos last-in first-out, each in its own guard, so one failing step never
stops the steps after it. A report is not a cleanup step: callers print it
after ``close`` returns, and a failing report cannot skip a release.

``BaseException`` from a step (KeyboardInterrupt, SystemExit) is also contained
until every step has been attempted, and is then re-raised. An Esc or Ctrl+C
arriving during shutdown must not leave a mouse button held.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CleanupFailure:
    step: str
    error: str


@dataclass
class CleanupStack:
    """Undo steps, run once, in reverse order of registration."""

    _steps: list[tuple[str, Callable[[], object]]] = field(default_factory=list)
    failures: list[CleanupFailure] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    closed: bool = False

    def push(self, name: str, undo: Callable[[], object]) -> None:
        if self.closed:
            # Registered after shutdown began: run it now rather than never.
            self._run(name, undo)
            return
        self._steps.append((name, undo))

    def close(self) -> list[CleanupFailure]:
        """Run every step once. Idempotent; never raises an ``Exception``."""

        if self.closed:
            return self.failures
        self.closed = True
        deferred: BaseException | None = None
        while self._steps:
            name, undo = self._steps.pop()
            try:
                self._run(name, undo)
            except BaseException as exc:  # noqa: BLE001 - finish every step first
                if deferred is None:
                    deferred = exc
        if deferred is not None:
            raise deferred
        return self.failures

    def _run(self, name: str, undo: Callable[[], object]) -> None:
        try:
            undo()
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
            self.failures.append(CleanupFailure(name, repr(exc)))
        else:
            self.completed.append(name)
