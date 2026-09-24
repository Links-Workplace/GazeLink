"""Free-running live gaze view: open a saved profile and watch the point.

Every live view so far was a side effect of recording a protocol.  The dot was
drawn only while a target was on screen, so watching it meant spending a
protocol, and it stopped when the targets ran out.  That is a measurement
tool, not a way to use the system: you cannot look at your own screen with it,
and you cannot leave it running.

This opens the profile's model, shows the filtered gaze point until you stop
it, and writes nothing -- no recording, no embeddings, no round number spent.

Safety: OS input is OFF by default, and every path to it is opt-in on the
command line.  With no such flag the point is drawn inside this window only and
nothing moves the pointer or emits a click.  ``--move-cursor`` moves the real
pointer and is REFUSED unless ``--i-mean-it`` is also given; ``--click-by wink``
arms real clicks the same way.  Releasing a button or key is always allowed, so
shutdown can never leave one held.

Privacy: no frame, face image or embedding is written to disk.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_click as CK  # noqa: E402
import gf_common as C  # noqa: E402
import gf_cursor as CUR  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_keys as KEYS  # noqa: E402
import gf_overlay as OV  # noqa: E402
import gf_profile as PROF  # noqa: E402
from gazelink_core.app import environment as ENV  # noqa: E402
from gazelink_core.app import live_session as SESSION  # noqa: E402
from gazelink_core.app import telemetry as TELEMETRY  # noqa: E402
from gazelink_core.app.options import LiveOptions  # noqa: E402
from gazelink_core.calibration import correction as CORR  # noqa: E402
from gazelink_core.calibration import model as MODEL  # noqa: E402
from gazelink_core.domain.clock import MonotonicClock  # noqa: E402
from gazelink_core.domain.observation import HeadPolicy  # noqa: E402
from gazelink_core.gaze import pipeline as PIPE  # noqa: E402
from gazelink_core.gaze import visibility as VIS  # noqa: E402
from gazelink_core.interaction.safety import SafetyController  # noqa: E402
from gazelink_core.platform import real_input as RI  # noqa: E402
from gazelink_core.tracking import face_landmarks as FL  # noqa: E402
from gazelink_core.tracking import gazefollower_library as LIB  # noqa: E402
from gazelink_core.tracking import gazefollower_source as SRC  # noqa: E402
from gazelink_core.ui import presenter as PRESENTER  # noqa: E402
from gazelink_core.ui import pygame_display as UI  # noqa: E402

# A distinct exit code so a caller can tell "the operator asked to
# recalibrate" from "the view was closed" without parsing stdout.
RECALIBRATE_REQUESTED = SESSION.RECALIBRATE_REQUESTED
# The one place the session's defaults live; the command line reads them from here.
_DEFAULT = LiveOptions()


# The per-frame pipeline lives in the core (ARCH-01 stage E). These names stay
# for every tool and test that uses them.
LiveState = PIPE.LiveState
TIMING_BUFFER = PIPE.TIMING_BUFFER


class LiveRunner(PIPE.FramePipeline):
    """``FramePipeline`` fed straight from a library callback.

    For tools that still subscribe to the library object themselves. The
    library's objects are translated by the tracking adapter (``observe``) and
    never read here; a session built on ``GazeFollowerSource`` calls
    ``on_observation`` instead.
    """

    def on_frame(self, face_info: Any, gaze_info: Any) -> None:
        """Called on the camera thread; must never raise into the library."""

        self._guarded(
            lambda: self._process(
                SRC.observe(
                    face_info,
                    gaze_info,
                    observed_s=self.clock(),
                    head_builder=self.head_builder or FL.head6_from_face,
                    head_policy=HeadPolicy.GAZE_OR_FACE,
                )
            )
        )


# Diagnostics and presentation helpers moved to the core (ARCH-01 stage F).
_ratio_summary = TELEMETRY.ratio_summary
_pitch_summary = TELEMETRY.pitch_summary
ratio_watcher = TELEMETRY.ratio_watcher
_menu_lines = PRESENTER.menu_lines


def state_face_ok(runner: Any, *, stale_after_s: float = VIS.OVERLAY_STALE_S) -> bool:
    """Is a face in front of the camera right now?

    Not the same question as "is there a usable gaze point": closing the eyes
    ends the point and not the face, and a caller that cannot tell them apart
    treats every deliberate close as a tracking failure and pauses in the same
    frame the close armed it.

    Frames must still be arriving, or a dead camera leaves the last answer
    standing and reads as a face for ever.
    """

    # The rule lives in SafetyController; this asks it on the runner's OWN
    # clock (``updated_s`` was stamped with it).
    safety = SafetyController(
        control=None,
        input_enabled=False,
        clock=getattr(runner, "clock", time.monotonic),
        stale_after_s=stale_after_s,
    )
    return safety.evaluate_face(runner)


# Moved to the profile module so the screen check can use it without importing
# this application (the gf_screen_check -> gf_live cycle, ARCH-01).
resolve_profile = PROF.resolve_profile


def check_rig(profile: PROF.Profile, rig: C.RigGeometry, *, allow_mismatch: bool) -> None:
    """Refuse a screen the profile was not fitted for.

    The model's output is mapped through the screen geometry, so running a
    profile against a different display does not degrade gracefully -- it
    produces a confidently wrong point with nothing on screen to say so.
    """

    mismatch = PROF.rig_mismatch(profile, rig)
    if not mismatch:
        return
    detail = "; ".join(mismatch)
    if allow_mismatch:
        print(f"WARNING: profile {profile.name!r} was fitted for a different rig ({detail})")
        return
    raise SystemExit(
        f"profile {profile.name!r} was fitted for a different rig ({detail}). "
        "The prediction is mapped through the screen geometry, so this would place the point "
        "confidently in the wrong spot. Recalibrate for this screen, or pass --allow-rig-mismatch "
        "if you know the difference does not matter."
    )


# Moved to the tracking adapter module (ARCH-01 stage D). Kept under this
# name: every live tool opens its camera through ``gf_live.build_gaze_follower``.
build_gaze_follower = LIB.build_gaze_follower


def real_environment() -> ENV.LiveEnvironment:
    """The real world for a live session, every name looked up at call time."""

    import gf_screen_check as SC  # noqa: PLC0415

    clock = MonotonicClock()
    return ENV.LiveEnvironment(
        clock=clock,
        open_source=lambda profile: SRC.GazeFollowerSource(
            profile,
            clock=clock,
            head_policy=HeadPolicy.GAZE_OR_FACE,
            # Looked up at call time, so a tool that replaces the builder
            # replaces it here as well.
            open_library=lambda p: build_gaze_follower(p),
            shutdown_library=lambda gf: LIB.shutdown_library(gf),
        ).open(),
        warm_up=lambda display, seconds: UI.sleep_with_escape(display, seconds),
        make_runner=lambda *a, **kw: LiveRunner(*a, **kw),
        # Honours a correction.json beside the model, so the point drawn here
        # and the point scored afterwards are the same thing. Loading the bare
        # model would silently show the UNCORRECTED prediction under the
        # corrected model's name.
        load_model=lambda path: CORR.load_with_correction(path, MODEL.FittedModel.load),
        make_display=lambda *a, **kw: UI.Display(*a, **kw),
        pick_monitor=lambda selector: GD.pick_monitor(selector),
        virtual_desktop=lambda: SC.virtual_desktop(),
        ensure_dpi_aware=lambda: SC.ensure_per_monitor_dpi_aware(),
        check_screen=lambda profile: SC.check_profile_screen(profile),
        escape_is_down=lambda: OV.escape_is_down(),
        key_is_down=lambda vk: OV.key_is_down(vk),
        click_sender=lambda flag: CK._send(flag),
        wheel_sender=lambda delta: CK._send_wheel(delta),
        key_sender=lambda vk, scan, flags: KEYS._send_key(vk, scan, flags),
        foreground_window=lambda: KEYS.foreground_window(),
        window_title=lambda hwnd: KEYS.window_title(hwnd),
        set_pointer=lambda x, y: CUR._set_cursor_pos(x, y),
        get_pointer=lambda: CUR._get_cursor_pos(),
        real_input=True,
    )


def run_live(
    profile: PROF.Profile, *, env: ENV.LiveEnvironment | None = None, **options: Any
) -> int:
    """Run one live session. A thin wrapper: see ``gazelink_core.app.live_session``.

    ``options`` are the fields of ``LiveOptions`` (an unknown name is a
    TypeError). Every call that leaves the process goes through ``env``; the
    default is the real world, and tests inject fakes.
    """

    session_env = env if env is not None else real_environment()
    return SESSION.LiveSession(profile, LiveOptions(**options), session_env).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None, help="profile name (default: the active one)")
    parser.add_argument(
        "--monitor", default=None, help="which display: an index, or part of its name"
    )
    parser.add_argument(
        "--show-unfiltered", action="store_true", help="also draw the unfiltered prediction"
    )
    parser.add_argument(
        "--move-cursor",
        action="store_true",
        help="move the REAL Windows pointer with your gaze. Requires --i-mean-it. No clicks.",
    )
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="confirms --move-cursor. Without it, --move-cursor is refused rather than simulated.",
    )
    parser.add_argument(
        "--cursor-smoothing",
        type=float,
        default=None,
        help="0..1 per frame toward the gaze. Lower is calmer and laggier; 1 disables smoothing.",
    )
    parser.add_argument(
        "--cursor-dead-zone-px",
        type=int,
        default=None,
        help="movement smaller than this does not move the pointer at all. Raise it if the "
        "pointer shivers while you hold still.",
    )
    parser.add_argument("--cursor-max-step-px", type=int, default=None)
    parser.add_argument(
        "--click-by",
        choices=("off", "wink"),
        default="off",
        help="'wink' emits a REAL left click where you were looking when you winked your "
        "right eye. Needs --move-cursor and --i-mean-it. Starts PAUSED; close BOTH eyes to "
        "switch. On the desktop a click cannot be taken back.",
    )
    parser.add_argument(
        "--wink-hold-ms",
        type=float,
        default=None,
        help="how long a wink must be held to count, overriding the profile. Lower it if "
        "the session report shows frames matching the rule but few winks firing; raise it "
        "if clicks arrive that you did not mean. The report prints both numbers.",
    )
    parser.add_argument(
        "--scroll-arm-ms",
        type=float,
        default=None,
        help="how long the gaze must rest on a scroll band before it starts scrolling. "
        "Raise it if crossing a band scrolls when you did not mean to; lower it if asking "
        "to scroll feels like waiting. This number has never been measured on this rig.",
    )
    parser.add_argument(
        "--scroll-repeat-ms",
        type=float,
        default=None,
        help="one wheel notch this often while the gaze stays on a band. Also unmeasured: "
        "the default is deliberately slower than a hand, because overshooting a page costs "
        "a look back the other way.",
    )
    parser.add_argument(
        "--menu-dwell-ms",
        type=float,
        default=_DEFAULT.menu_dwell_ms,
        help="how long to rest on a menu tile to choose it. The same 900 ms the dwell "
        "engine has used since it was written; nothing has measured it against a person "
        "on these four tiles yet.",
    )
    parser.add_argument(
        "--scan-ms",
        type=float,
        default=None,
        help="how long the scanning keyboard holds each group or key before moving on. "
        "An OPENING GUESS at 1200 ms: it has never been measured on this rig. Lower it if "
        "waiting for the letter is the slow part; raise it if the highlight moves past "
        "before a wink can land. The report prints how often the sweep parked itself.",
    )
    parser.add_argument(
        "--scan-settle-ms",
        type=float,
        default=None,
        help="how long the FIRST group or key of a new screen must have been up before a "
        "wink may take it - the descent into a group, the return to the groups after a "
        "key, a layout change, the keyboard opening. 350 ms by default, just above the "
        "soonest an unwanted second wink can come out of one closure. It is NOT charged "
        "on an ordinary step of the sweep, where --wink-lag-ms already decides. Raise it "
        "if a key is taken that was never watched; lower it if deliberate winks are being "
        "refused (the report counts both).",
    )
    parser.add_argument(
        "--wink-lag-ms",
        type=float,
        default=None,
        help="how long after the eyelid starts moving the wink is stamped. Defaults to "
        "--wink-hold-ms, which is the part of it this repository knows; the camera and "
        "landmark model add more and that has not been measured. Raise it if the letter "
        "typed is consistently the one AFTER the one chosen.",
    )
    parser.add_argument(
        "--no-menu",
        dest="menu",
        action="store_false",
        help="do not open a menu on a long close. Needed with --toggle-by eyes, which "
        "uses the same gesture. Without the menu there is no right click, no keyboard "
        "and no way into scrolling except --start-scrolling.",
    )
    parser.add_argument(
        "--start-scrolling",
        action="store_true",
        help="open already in scroll mode, so the bands work straight away and the menu "
        "is only needed to come BACK to the pointer. Leave the pointer on the page you "
        "want to read before starting: it is frozen where it is found.",
    )
    parser.add_argument(
        "--wink-click",
        choices=("single", "double"),
        default="single",
        help="what a LEFT wink does at the start: 'single' (default) or 'double'. The menu "
        "tile 'לחיצה כפולה' switches it during the session. A RIGHT wink is always a right "
        "click. Double exists because asking for two winks inside Windows' 500 ms window is "
        "a timing test a person can fail through no fault of their own.",
    )
    parser.add_argument(
        "--toggle-by",
        choices=("key", "eyes"),
        default="key",
        help="what switches PAUSED <-> ACTIVE. 'key' is SPACE and works whoever has focus, "
        "which the overlay never does. 'eyes' is the two-eyed close, which is also what "
        "opens the menu -- so it needs --no-menu, and it gives up the right click, the "
        "keyboard and every navigation command with it.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="open PAUSED instead of ready to click. The default is ACTIVE, because the "
        "paused start left no way in on a desktop: the practice window's toggle was a gaze "
        "panel and there is nowhere to draw one here.",
    )
    parser.add_argument(
        "--hold-after-click-s",
        type=float,
        default=_DEFAULT.hold_after_click_s,
        help="how long the pointer stays where it clicked before following the gaze again. "
        "This is also the window in which a second wink becomes a double click, because both "
        "halves have to land on the same pixel.",
    )
    parser.add_argument(
        "--desk",
        action="store_true",
        help="the desk interface: one bar (click, scroll, keyboard, more), a pause target "
        "that never moves, the magnifier, the gaze keyboard and drag. Off by default; "
        "without it this runs exactly the path it has always run.",
    )
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--allow-rig-mismatch", action="store_true")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop after this long (for unattended checks)",
    )
    parser.add_argument("--list", action="store_true", help="print the saved profiles and exit")
    return parser


def _setting(override: Any, profile: PROF.Profile, key: str, fallback: Any) -> Any:
    """A flag beats the profile; the profile beats the built-in default."""

    if override is not None:
        return override
    return (profile.cursor or {}).get(key, fallback)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        active = PROF.active_name()
        for name in PROF.list_profiles():
            print(f"{'*' if name == active else ' '} {name}")
        return 0
    profile = resolve_profile(args.profile)
    monitor = GD.pick_monitor(args.monitor) if args.monitor is not None else None
    device = (
        C.RigGeometry(
            profile.rig["camera_x_cm"],
            profile.rig["camera_y_cm"],
            profile.rig["screen_w_cm"],
            profile.rig["screen_h_cm"],
            monitor.width_px,
            monitor.height_px,
        )
        if monitor is not None
        else profile.rig_geometry()
    )
    check_rig(profile, device, allow_mismatch=args.allow_rig_mismatch)
    # Real input is armed only here, only for this run, and only after the
    # operator typed both flags. Everything else in the process -- tests,
    # probes, a programmatic caller -- stays disarmed and a real send raises.
    arming = (
        RI.armed("gf_live --move-cursor --i-mean-it")
        if args.move_cursor and args.i_mean_it
        else contextlib.nullcontext()
    )
    with arming:
        return _run_live_from_args(profile, monitor, args)


def _run_live_from_args(profile: PROF.Profile, monitor: Any, args: argparse.Namespace) -> int:
    return run_live(
        profile,
        monitor=monitor,
        show_unfiltered=args.show_unfiltered,
        skip_model_check=args.skip_model_check,
        move_cursor=args.move_cursor,
        confirmed=args.i_mean_it,
        # The profile carries what was found comfortable; a flag overrides it
        # for one run without editing the saved setup.
        cursor_smoothing=_setting(
            args.cursor_smoothing, profile, "smoothing", _DEFAULT.cursor_smoothing
        ),
        cursor_dead_zone_px=int(
            _setting(
                args.cursor_dead_zone_px, profile, "dead_zone_px", _DEFAULT.cursor_dead_zone_px
            )
        ),
        cursor_max_step_px=int(
            _setting(args.cursor_max_step_px, profile, "max_step_px", _DEFAULT.cursor_max_step_px)
        ),
        click_by=args.click_by,
        wink_click=args.wink_click,
        wink_hold_ms=args.wink_hold_ms,
        scroll_arm_ms=args.scroll_arm_ms,
        scroll_repeat_ms=args.scroll_repeat_ms,
        start_scrolling=args.start_scrolling,
        desk=args.desk,
        menu_enabled=args.menu,
        menu_dwell_ms=args.menu_dwell_ms,
        scan_ms=args.scan_ms,
        scan_settle_ms=args.scan_settle_ms,
        wink_lag_ms=args.wink_lag_ms,
        toggle_by=args.toggle_by,
        start_active=not args.start_paused,
        hold_after_click_s=args.hold_after_click_s,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
