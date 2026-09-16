"""Everything a live session needs from outside the process, in one object.

The session never reaches for a camera, a window, the screen, the keyboard
state, a model file or a Windows sender directly: it asks this. Production
builds it from the real adapters at the entry point; tests build it from fakes
and pass ``real_input=False``, and a session refuses to run enabled input
adapters on real senders unless an entry point armed real input
(``gazelink_core.platform.real_input``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gazelink_core.domain.clock import Clock


@dataclass
class LiveEnvironment:
    clock: Clock
    # Tracking source: opens the camera/library for a profile and returns an
    # object with subscribe(sink), start(), stop() and close() -> report. Sinks
    # receive FrameObservation only.
    open_source: Callable[[Any], Any]
    warm_up: Callable[[Any, float], bool]
    make_runner: Callable[..., Any]
    load_model: Callable[[Any], Any]
    # Display
    make_display: Callable[..., Any]
    # Screen geometry and verification
    pick_monitor: Callable[[Any], Any]
    virtual_desktop: Callable[[], dict[str, int]]
    ensure_dpi_aware: Callable[[], tuple[bool, str]]
    check_screen: Callable[[Any], Any]
    # Keyboard state read by the loop (Esc, SPACE)
    escape_is_down: Callable[[], bool]
    key_is_down: Callable[[int], bool]
    # Windows input senders handed to the adapters
    click_sender: Callable[[int], None]
    wheel_sender: Callable[[int], None]
    key_sender: Callable[[int, int, int], None]
    foreground_window: Callable[[], int]
    window_title: Callable[[int], str]
    set_pointer: Callable[[int, int], None]
    get_pointer: Callable[[], tuple[int, int]]
    # True only when the senders above reach Windows.
    real_input: bool
