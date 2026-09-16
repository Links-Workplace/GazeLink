"""Process-wide interlock in front of every real Windows input call.

Disarmed by default. The real senders (``gf_click._send``/``_send_wheel``,
``gf_keys._send_key``, ``gf_cursor._set_cursor_pos``) call :func:`require`
before touching ``SendInput``/``SetCursorPos``. Only a command-line entry point
arms it, inside :func:`armed`, after the operator's explicit confirmation
(``--i-mean-it``).

Why here and not in the test suite: ``unittest discover`` does not guarantee
any package-level test file is loaded, and a harness that forgets to fake one
sender used to let a real wheel notch escape (tests/test_live_loop.py). With
the interlock in the sender itself, a forgotten fake raises instead of reaching
Windows, whatever launched the process.

RELEASES are never blocked (TECHNICAL_SPEC 17): a button-up or key-up must be
deliverable during shutdown even after the scope has closed, or a held button
would stay held. Callers mark them with ``release=True``. Moving the pointer is
not a release, including the restore on exit.

This is a local interlock, not business policy: whether an action is allowed
is decided by the SafetyController and the ActionRouter before any sender runs.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator

_lock = threading.Lock()
_depth = 0
_reason: str | None = None


class RealInputNotArmed(RuntimeError):
    """A real OS input call was attempted without an armed entry point."""


def is_armed() -> bool:
    with _lock:
        return _depth > 0


def reason() -> str | None:
    with _lock:
        return _reason if _depth > 0 else None


def require(kind: str, *, release: bool = False) -> None:
    """Raise unless real input is armed. Releases always pass."""

    if release:
        return
    with _lock:
        if _depth > 0:
            return
    # Adapters count a failed send and carry on, so a refusal can be silent.
    # Setting this variable makes every refusal auditable across a whole run
    # (used to prove the test suite never relies on a real sender).
    audit = os.environ.get("GAZELINK_INPUT_REFUSAL_LOG")
    if audit:
        with contextlib.suppress(OSError), open(audit, "a", encoding="utf-8") as log:
            log.write(f"{kind}\n")
    raise RealInputNotArmed(
        f"refused real OS input ({kind}): no entry point armed real input. "
        "Tests must inject fake senders; tools arm only after --i-mean-it."
    )


@contextlib.contextmanager
def armed(why: str) -> Iterator[None]:
    """Allow real input for the duration of the block. Nested scopes count."""

    global _depth, _reason
    if not why:
        raise ValueError("arming real input needs a stated reason")
    with _lock:
        _depth += 1
        if _depth == 1:
            _reason = why
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                _reason = None
