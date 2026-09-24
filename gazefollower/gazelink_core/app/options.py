"""Everything a live session can be told, with its defaults, in one place.

``LiveOptions`` replaces the 25 keyword arguments of ``run_live`` (ARCH-01
stage C/G). Precedence is unchanged: an explicit command-line flag, then the
profile (``gf_live.main`` resolves that), then the defaults below. Units are in
the field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gazelink_core.interaction.menu import MenuModel
from gazelink_core.platform.cursor import CursorLimits

# Defaults owned by the component they tune, read here rather than repeated.
_CURSOR = CursorLimits()
_MENU_DWELL_MS: float = float(MenuModel.__dataclass_fields__["dwell_ms"].default)  # type: ignore[arg-type]


@dataclass(frozen=True)
class LiveOptions:
    monitor: Any = None
    show_unfiltered: bool = False
    skip_model_check: bool = False
    headless: bool = False
    max_seconds: float | None = None
    # OS input needs BOTH: the operator asking (move_cursor) and confirming.
    move_cursor: bool = False
    confirmed: bool = False
    cursor_smoothing: float = _CURSOR.smoothing
    cursor_dead_zone_px: int = _CURSOR.dead_zone_px
    cursor_max_step_px: int = _CURSOR.max_step_px
    click_by: str = "off"  # "off" | "wink"
    # What a LEFT wink does at the start; the menu toggles it. A right wink is
    # always a right click. Single by default (operator, 24.9.2026).
    wink_click: str = "single"  # "single" | "double"
    wink_hold_ms: float | None = None
    scroll_arm_ms: float | None = None
    scroll_repeat_ms: float | None = None
    start_scrolling: bool = False
    toggle_by: str = "key"  # "key" (SPACE) | "eyes" (long close)
    start_active: bool = True
    hold_after_click_s: float = 1.5
    # The desk interface: one bar, a secondary menu, the magnifier, the gaze
    # keyboard and drag. OFF by default, so every existing command and every
    # golden scenario runs the path they always ran.
    desk: bool = False
    menu_enabled: bool = True
    menu_dwell_ms: float = _MENU_DWELL_MS
    scan_ms: float | None = None
    scan_settle_ms: float | None = None
    wink_lag_ms: float | None = None

    @property
    def input_enabled(self) -> bool:
        """Click, wheel and keyboard adapters are enabled only in wink mode with the pointer."""

        return self.click_by == "wink" and self.move_cursor

    @property
    def controlled(self) -> bool:
        """A PAUSED/ACTIVE control machine exists (wink mode)."""

        return self.click_by == "wink"

    def validate(self) -> None:
        """Refuse combinations that cannot mean anything, before anything opens."""

        if self.click_by not in ("off", "wink"):
            raise SystemExit(f"unknown click mode {self.click_by!r}: use 'off' or 'wink'")
        if self.toggle_by not in ("key", "eyes"):
            raise SystemExit(f"unknown toggle {self.toggle_by!r}: use 'key' or 'eyes'")
        if self.wink_click not in ("single", "double"):
            raise SystemExit(f"unknown wink action {self.wink_click!r}: use 'single' or 'double'")
        if self.click_by == "wink" and not self.move_cursor:
            raise SystemExit(
                "--click-by wink needs --move-cursor: a click that lands wherever the pointer "
                "was last left is not a click at what you are looking at."
            )
        if self.menu_enabled and self.toggle_by == "eyes" and self.click_by == "wink":
            # One gesture cannot mean two things. The long close is what opens the
            # menu, and it is also what --toggle-by eyes uses to switch PAUSED and
            # ACTIVE: with both on, every attempt to open the menu would also flip
            # the mode. Refused rather than silently given a precedence, the same
            # way --click-by wink refuses to run without --move-cursor.
            raise SystemExit(
                "--toggle-by eyes and the menu are the same gesture: a long close cannot both "
                "open the menu and switch PAUSED/ACTIVE. Use --no-menu to keep the eyes toggle, "
                "or leave the default and switch with SPACE. The menu's own PAUSE tile switches "
                "the mode without a key, and it opens while paused."
            )
