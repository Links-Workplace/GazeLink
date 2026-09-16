"""Compatibility name and command line for ``gazelink_core.platform.screen_check`` (ADR-0002).

Imported, ``gf_screen_check`` IS the core module (same object). Run as a script,
it is the read-only screen check: the command line lives here, not in the core.
"""

import argparse
import sys

from gazelink_core.calibration import profile as PROF
from gazelink_core.platform import display as GD
from gazelink_core.platform import screen_check as _module

__doc__ = _module.__doc__
ensure_per_monitor_dpi_aware = _module.ensure_per_monitor_dpi_aware
check_profile_screen = _module.check_profile_screen
format_report = _module.format_report
virtual_desktop = _module.virtual_desktop
to_desktop_pixels = _module.to_desktop_pixels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(argv)
    # Declared FIRST, before anything reads a metric or opens a window.
    declared, how = ensure_per_monitor_dpi_aware()
    print(f"dpi awareness: {how}")
    profile = PROF.resolve_profile(args.profile)
    report = check_profile_screen(profile)
    print(format_report(report))
    monitor = GD.pick_monitor(None)
    desktop = virtual_desktop()
    print("  mapping examples (gaze fraction -> desktop pixel):")
    for label, pt in (
        ("centre", (0.5, 0.5)),
        ("LEFT button", (0.38, 0.5)),
        ("RIGHT button", (0.62, 0.5)),
    ):
        print(f"    {label:14s} {pt} -> {to_desktop_pixels(pt, monitor, desktop)}")
    print()
    return 0 if report.verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
sys.modules[__name__] = _module
