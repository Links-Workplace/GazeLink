"""Is the ruler verified? Read-only checks before any real cursor is moved.

M3-00 blocks M3-02 with one sentence: no real cursor on an unverified ruler.
This is that verification, and it emits NOTHING -- it reads what Windows says
about the desktop and compares it against what the profile was fitted for.
No cursor is moved here, no input is synthesised, nothing is written.

Why a gaze point cannot simply be scaled to the screen
------------------------------------------------------

The model outputs a fraction of the screen it was calibrated on.  Turning that
into a Windows cursor position needs three things to be true at once, and each
fails silently and differently:

* **The same monitor.**  Size alone does not identify a display: two monitors
  can share a resolution.  A profile fitted on one and replayed on another
  maps confidently to the wrong place.
* **DPI awareness.**  With display scaling on (1.25 here), a process that has
  not declared itself DPI-aware is fed *virtualised* coordinates by Windows.
  ``GetSystemMetrics`` then returns 4096x1152 for a 5120x1440 panel and the
  cursor lands 20% short of where it was aimed -- consistently, and with no
  error anywhere.
* **The desktop origin.**  On a multi-monitor desktop a monitor's top-left is
  not (0, 0), and it can be negative.  A fraction of *this* monitor has to be
  offset into virtual-desktop space before it means anything.

Each is checked separately and reported separately, because "the cursor is in
the wrong place" is the same symptom for all three.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

from gazelink_core.calibration import profile as PROF  # noqa: E402
from gazelink_core.platform import display as GD  # noqa: E402

# GetSystemMetrics indices for the whole virtual desktop.
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# DPI_AWARENESS_CONTEXT values, as reported by GetAwarenessFromDpiAwarenessContext.
_AWARENESS = {
    0: "UNAWARE -- Windows virtualises coordinates for this process",
    1: "SYSTEM_AWARE -- correct only while every display shares one scale factor",
    2: "PER_MONITOR_AWARE -- coordinates are real physical pixels",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class ScreenReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name, ok, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


def dpi_awareness() -> tuple[int | None, str]:
    """What Windows thinks this process is, for scaling purposes."""

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        context = user32.GetThreadDpiAwarenessContext()
        value = int(user32.GetAwarenessFromDpiAwarenessContext(context))
        return value, _AWARENESS.get(value, f"unknown awareness value {value}")
    except Exception as exc:  # noqa: BLE001 - absence is a finding, not a crash
        return None, f"could not be determined ({exc!r})"


# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
_PER_MONITOR_V2 = ctypes.c_void_p(-4)


def ensure_per_monitor_dpi_aware() -> tuple[bool, str]:
    """Declare this process DPI-aware, before it asks Windows anything.

    Left undeclared, Windows hands a scaled process virtualised coordinates:
    on a 125% display it reports 4096x1152 for a 5120x1440 panel, and a cursor
    aimed at the right edge lands a fifth of the screen short -- consistently,
    with no error raised anywhere. Declaring awareness removes the entire
    class of failure rather than compensating for it.

    Must be called before any window is created or any metric read; Windows
    refuses to change it afterwards, and that refusal is reported rather than
    swallowed.
    """

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if user32.SetProcessDpiAwarenessContext(_PER_MONITOR_V2):
            return True, "declared PER_MONITOR_AWARE_V2"
        return False, "Windows refused the change (already set for this process)"
    except Exception as exc:  # noqa: BLE001 - reported, never fatal on its own
        return False, f"could not be declared ({exc!r})"


def virtual_desktop() -> dict[str, int]:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    return {
        "x": user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
        "y": user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
        "width": user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
        "height": user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
    }


def check_profile_screen(profile: PROF.Profile) -> ScreenReport:
    """Every condition that must hold before a gaze fraction becomes a pixel."""

    report = ScreenReport()

    value, description = dpi_awareness()
    # SYSTEM_AWARE is accepted only because this rig has one display; it would
    # be wrong the moment a second monitor with a different scale appeared.
    report.add(
        "dpi awareness",
        value in (1, 2),
        description
        + ("" if value != 1 else " (acceptable here only because a single display is present)"),
    )

    monitors = GD.list_monitors()
    report.add(
        "single display",
        len(monitors) == 1,
        f"{len(monitors)} display(s) reported: " + ", ".join(m.name for m in monitors),
    )

    rig = profile.rig_geometry()
    monitor = GD.pick_monitor(None)
    same_size = (monitor.width_px, monitor.height_px) == (rig.device_w_px, rig.device_h_px)
    report.add(
        "resolution matches the profile",
        same_size,
        f"profile {rig.device_w_px}x{rig.device_h_px}, display now "
        f"{monitor.width_px}x{monitor.height_px}",
    )

    screen = profile.screen or {}
    report.add(
        "display identity recorded",
        bool(screen.get("screen_id")),
        f"profile was fitted on {screen.get('screen_id')!r} via {screen.get('connector')!r}"
        if screen.get("screen_id")
        else "the profile carries no screen identity, so only its SIZE can be compared. "
        "Two monitors can share a resolution, which is exactly the failure M3-00 names",
    )

    # Physical millimetres are the identity check available at runtime: the
    # panel's model string is not exposed by the display library, but its size
    # in millimetres is, and it differs between panels that share a
    # resolution. Half a millimetre of rounding is not a different monitor.
    want_mm = (screen.get("width_mm"), screen.get("height_mm"))
    got_mm = (monitor.width_mm, monitor.height_mm)
    if all(v is not None for v in want_mm + got_mm):
        same_panel = all(
            abs(float(a) - float(b)) <= 1.0 for a, b in zip(want_mm, got_mm, strict=True)
        )
        report.add(
            "physical size matches the profile",
            same_panel,
            f"profile {want_mm[0]}x{want_mm[1]} mm, display now {got_mm[0]}x{got_mm[1]} mm",
        )
    else:
        report.add(
            "physical size matches the profile",
            False,
            "the display reports no physical size, so the panel cannot be identified at all",
        )

    desktop = virtual_desktop()
    matches_desktop = (desktop["width"], desktop["height"]) == (
        monitor.width_px,
        monitor.height_px,
    )
    report.add(
        "windows agrees with the display library",
        matches_desktop,
        f"virtual desktop {desktop['width']}x{desktop['height']} at "
        f"({desktop['x']}, {desktop['y']}); display library says "
        f"{monitor.width_px}x{monitor.height_px} at {monitor.origin}",
    )

    report.add(
        "desktop origin known",
        True,
        f"a gaze fraction must be offset by ({desktop['x']}, {desktop['y']}) to become a "
        "virtual-desktop coordinate",
    )
    return report


def to_desktop_pixels(
    point: tuple[float, float], monitor: GD.MonitorInfo, desktop: dict[str, int]
) -> tuple[int, int]:
    """A gaze fraction of THIS monitor as a virtual-desktop pixel.

    Pure arithmetic and deliberately separate from anything that could move a
    cursor, so the mapping can be tested without the ability to emit input.
    Clamped to the monitor: a prediction outside it is a prediction to ignore,
    never a cursor thrown to the far edge of another screen.
    """

    x = min(1.0, max(0.0, float(point[0])))
    y = min(1.0, max(0.0, float(point[1])))
    ox, oy = monitor.origin
    return (
        int(round(ox + x * (monitor.width_px - 1))),
        int(round(oy + y * (monitor.height_px - 1))),
    )


def format_report(report: ScreenReport) -> str:
    lines = ["", "screen verification (read-only; nothing was moved or written)", ""]
    for c in report.checks:
        lines.append(f"  [{'ok ' if c.ok else 'NO '}] {c.name}")
        lines.append(f"         {c.detail}")
    lines.append("")
    if report.verified:
        lines.append("  VERIFIED: a gaze fraction can be mapped to this desktop.")
    else:
        failed = [c.name for c in report.checks if not c.ok]
        lines.append(f"  NOT VERIFIED: {', '.join(failed)}.")
        lines.append("  M3-00 blocks a real cursor on an unverified ruler.")
    lines.append("")
    return "\n".join(lines)
