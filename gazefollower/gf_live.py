"""Free-running live gaze view: open a saved profile and watch the point.

Every live view so far was a side effect of recording a protocol.  The dot was
drawn only while a target was on screen, so watching it meant spending a
protocol, and it stopped when the targets ran out.  That is a measurement
tool, not a way to use the system: you cannot look at your own screen with it,
and you cannot leave it running.

This opens the profile's model, shows the filtered gaze point until you stop
it, and writes nothing -- no recording, no embeddings, no round number spent.

Safety and privacy: no OS input of any kind.  The point is drawn inside this
window only; nothing here moves the cursor or emits a click.  No frame, face
image or embedding is written to disk.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_cursor as CUR  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402

# A distinct exit code so a caller can tell "the operator asked to
# recalibrate" from "the view was closed" without parsing stdout.
RECALIBRATE_REQUESTED = 10


def _menu_lines(menu: GEST.RecoveryMenu) -> list[str]:
    if not menu.open:
        return []
    return ["", "MENU  (short close cycles, long close confirms)"] + [
        f"  {'>' if i == menu.index else ' '} {option.label}"
        for i, option in enumerate(menu.options)
    ]


@dataclass
class LiveState:
    """What the display thread reads. Replaced as a whole, never mutated."""

    point: tuple[float, float] | None = None
    raw_model: tuple[float, float] | None = None
    unfiltered: tuple[float, float] | None = None
    tracking: bool = False
    updated_s: float | None = None
    frames: int = 0
    fps: float | None = None


class LiveRunner:
    """Predict and filter every frame, and keep nothing.

    Deliberately not a ProtocolRunner: there is no target, no gate, no
    accepted-row accounting and no RecordingBuilder here.  Reusing that class
    with a fake one-target protocol would keep a storage path alive in a mode
    whose whole promise is that it stores nothing.
    """

    def __init__(
        self,
        model: Any,
        model_y: Any,
        rig: C.RigGeometry,
        settings: GF.FilterSettings | None,
        *,
        clock: Any = time.monotonic,
        head_builder: Any = H.build,
        gesture: GEST.GestureConfig | None = None,
    ) -> None:
        self.model = model
        self.model_y = model_y
        self.rig = rig
        self.clock = clock
        self.head_builder = head_builder
        self.filter = None if settings is None else GF.GazePointFilter(settings)
        # Driven here, on the camera thread, so every frame is seen exactly
        # once. Driving it from the display loop instead would resample the
        # same frame many times over -- the display redraws far faster than
        # the camera delivers -- and a hold would appear to last longer than
        # it did.
        self.detector = GEST.EyeCloseDetector(gesture)
        self.gesture_events: list[tuple[float, GEST.Event]] = []
        self.lock = threading.Lock()
        self.state = LiveState()
        self.frames = 0
        self.errors = 0
        self.last_error: str | None = None
        self._frame_times: list[float] = []

    def on_frame(self, face_info: Any, gaze_info: Any) -> None:
        """Called on the camera thread; must never raise into the library."""

        try:
            self._on_frame(face_info, gaze_info)
        except Exception as exc:  # noqa: BLE001 - a view must not kill the camera thread
            with self.lock:
                self.errors += 1
                self.last_error = repr(exc)

    def _on_frame(self, face_info: Any, gaze_info: Any) -> None:
        now_s = self.clock()
        gaze_status = bool(getattr(gaze_info, "status", False))
        features = getattr(gaze_info, "features", None) if gaze_status else None
        if features is not None:
            features = np.asarray(features, dtype=np.float64).reshape(-1)
        left = float(getattr(face_info, "left_eye_openness", 0.0) or 0.0)
        right = float(getattr(face_info, "right_eye_openness", 0.0) or 0.0)
        head = self.head_builder(face_info) if gaze_status else None

        # Same gate as the recorder's overlay: a blink is not a gaze sample,
        # and predicting through one puts the point somewhere the eye is not.
        valid = gaze_status and left > C.BLINK_THRESHOLD and right > C.BLINK_THRESHOLD
        # The gesture reads the FACE, not the gaze: it has to keep working
        # when the gaze is unusable, because "eyes shut" is precisely when
        # there is no gaze.
        face_present = bool(getattr(face_info, "status", False))
        eyes_shut = not (left > C.BLINK_THRESHOLD and right > C.BLINK_THRESHOLD)
        event = self.detector.update(now_s, face_present=face_present, eyes_shut=eyes_shut)
        raw = (
            R.predict_overlay_point(self.model, self.model_y, features, head, self.rig)
            if valid
            else None
        )

        point = None
        if raw is None:
            if self.filter is not None:
                self.filter.reset()
        elif self.filter is None:
            point = raw
        else:
            try:
                point = self.filter.update(raw, now_s)
            except ValueError:
                self.filter.reset()

        raw_model = None
        raw_cm = getattr(gaze_info, "raw_gaze_coordinates", None) if gaze_status else None
        if raw_cm is not None:
            try:
                raw_model = self.rig.cm_to_norm(float(raw_cm[0]), float(raw_cm[1]))
            except (TypeError, ValueError):
                raw_model = None

        with self.lock:
            if event is not GEST.Event.NONE:
                self.gesture_events.append((now_s, event))
            self.frames += 1
            self._frame_times.append(now_s)
            if len(self._frame_times) > 60:
                del self._frame_times[:-60]
            self.state = LiveState(
                point=point,
                raw_model=raw_model,
                unfiltered=raw,
                tracking=valid,
                updated_s=now_s,
                frames=self.frames,
                fps=self._fps(),
            )

    def drain_gesture_events(self) -> list[tuple[float, GEST.Event]]:
        with self.lock:
            events = self.gesture_events
            self.gesture_events = []
        return events

    def _fps(self, window: int = 30) -> float | None:
        times = self._frame_times[-window:]
        if len(times) < 2:
            return None
        span = times[-1] - times[0]
        return None if span <= 0 else (len(times) - 1) / span


def resolve_profile(name: str | None) -> PROF.Profile:
    profile = PROF.load(name) if name else PROF.load_active()
    if profile is None:
        known = PROF.list_profiles()
        hint = (
            f" Known profiles: {', '.join(known)}."
            if known
            else " No profiles have been saved yet."
        )
        raise SystemExit(
            f"no active profile to run.{hint} Activate one with gf_profile, or pass --profile."
        )
    return profile


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


def build_gaze_follower(rig: C.RigGeometry) -> Any:
    """Open the library for a free-running session on this rig.

    Shared with the gesture probe so there is exactly one description of how
    the library is configured for a no-protocol session; two copies would
    drift, and the probe would then be measuring a different pipeline from
    the one the view actually runs.
    """

    import gazefollower  # noqa: F401, PLC0415 - initialises native components
    from gazefollower import GazeFollower  # noqa: PLC0415
    from gazefollower.misc import DefaultConfig  # noqa: PLC0415

    config = DefaultConfig()
    config.cali_mode = 9
    config.camera_position = (rig.camera_x_cm, rig.camera_y_cm)
    config.screen_physical_size = (rig.screen_w_cm, rig.screen_h_cm)
    config.screen_size = np.array([rig.device_w_px, rig.device_h_px])
    return GazeFollower(config=config, calibration=R.make_pass_through_calibration())


def run_live(
    profile: PROF.Profile,
    *,
    monitor: Any = None,
    show_unfiltered: bool = False,
    skip_model_check: bool = False,
    headless: bool = False,
    max_seconds: float | None = None,
    move_cursor: bool = False,
    confirmed: bool = False,
    cursor_smoothing: float = 0.35,
    cursor_dead_zone_px: int = 12,
    cursor_max_step_px: int = 400,
) -> int:
    import gf_fit as FIT  # noqa: PLC0415
    import gf_screen_check as SC  # noqa: PLC0415

    cursor_monitor = desktop = None
    if move_cursor:
        # Two separate gates, on purpose. The first is the operator saying
        # they meant it; the second is the machine agreeing the ruler is
        # trustworthy. Neither substitutes for the other.
        if not confirmed:
            raise SystemExit(
                "--move-cursor moves the REAL Windows pointer. Add --i-mean-it to confirm. "
                "Without it this runs as a simulation and prints where the cursor would go."
            )
        declared, how = SC.ensure_per_monitor_dpi_aware()
        print(f"dpi awareness: {how}")
        report = SC.check_profile_screen(profile)
        if not report.verified:
            print(SC.format_report(report))
            raise SystemExit(
                "the screen is not verified, so a gaze fraction cannot be trusted as a pixel. "
                "M3-00: no real cursor on an unverified ruler. Nothing was moved."
            )
        cursor_monitor = GD.pick_monitor(None)
        desktop = SC.virtual_desktop()

    rig = profile.rig_geometry()
    settings = profile.filter_settings()
    model_dir = profile.model_path()
    if not model_dir.exists():
        raise SystemExit(
            f"profile {profile.name!r} points at a model that is not there: {model_dir}"
        )
    model = FIT.FittedModel.load(model_dir)
    print(f"profile {profile.name}: model {model_dir} ({len(model.schema.columns)} columns)")

    gf = build_gaze_follower(rig)
    runner = LiveRunner(model, None, rig, settings)
    # The harmless option is first, so a confirm that fires when it should not
    # costs nothing. Recalibration is reachable only by cycling to it first.
    options = [GEST.MenuOption("dismiss", "Keep watching")]
    if move_cursor:
        # The pause sits before recalibration: when the pointer is moving,
        # stopping it is the thing most likely to be wanted in a hurry, and it
        # costs nothing if chosen by accident.
        options.append(GEST.MenuOption("pause", "Pause / resume the cursor"))
    options.append(GEST.MenuOption("recalibrate", "Recalibrate for this screen"))
    menu = GEST.RecoveryMenu(options)
    chosen: list[str] = []
    preflight: list[np.ndarray] = []
    armed = {"live": False}

    def subscriber(face_info: Any, gaze_info: Any) -> None:
        if not armed["live"]:
            if getattr(gaze_info, "status", False) and len(preflight) < R.PREFLIGHT_MAX_FRAMES:
                row = R._overlay_design_row(model, face_info, gaze_info)
                if row is not None:
                    preflight.append(row)
            return
        runner.on_frame(face_info, gaze_info)

    gf.add_subscriber(subscriber)
    origin = (0, 0) if monitor is None else monitor.origin
    display = R.Display(
        rig.device_w_px,
        rig.device_h_px,
        headless=headless,
        origin=origin,
        overlay_available=True,
        show_unfiltered_overlay=show_unfiltered,
    )
    cursor = CUR.CursorAdapter(
        enabled=move_cursor,
        limits=CUR.CursorLimits(
            max_step_px=cursor_max_step_px,
            smoothing=cursor_smoothing,
            dead_zone_px=cursor_dead_zone_px,
        ),
    )
    try:
        gf.camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        if R._sleep_with_escape(display, R.CAMERA_WARMUP_S):
            return 1
        if not skip_model_check:
            # Always report, even with nothing to check.  Printing nothing
            # when no frames arrived reads exactly like a pass, and "the
            # check did not run" is the one outcome the operator most needs
            # to hear -- it is also what a covered camera looks like.
            #
            # "No frames" is reported separately rather than by passing None
            # to preflight_verdict: there, None means "this model has no
            # kernel", so reusing it would blame the model for an empty
            # camera and print a reason that is simply untrue.
            ok = True
            if not preflight:
                print(
                    "model check: no warm-up frames carried a tracked face, so the model was "
                    "not checked against today's conditions. If the point misbehaves, that is "
                    "the first thing to suspect."
                )
            else:
                ok, message = R.preflight_verdict(model.support_activation(np.vstack(preflight)))
                print(f"model check ({len(preflight)} warm-up frames): {message}")
            if not ok:
                display.draw_message(
                    ["Model does not fit current conditions.", "", "See the terminal."]
                )
                time.sleep(3.0)
                return 2
        armed["live"] = True
        cursor.__enter__()
        started = time.monotonic()
        while True:
            if display.poll_escape():
                break
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                break
            for when, event in runner.drain_gesture_events():
                taken = menu.handle(event, when)
                if taken is None or taken.key == "dismiss":
                    continue
                if taken.key == "pause" and cursor is not None:
                    cursor.paused = not cursor.paused
                    print(f"cursor {'paused' if cursor.paused else 'resumed'}")
                    continue
                chosen.append(taken.key)
            menu.tick(time.monotonic())
            if chosen:
                break
            state = runner.state
            fresh = R.visible_point(
                state.point, state.updated_s, time.monotonic(), R.OVERLAY_STALE_S
            )
            raw_fresh = R.visible_point(
                state.raw_model, state.updated_s, time.monotonic(), R.OVERLAY_STALE_S
            )
            unfiltered = R.visible_point(
                state.unfiltered, state.updated_s, time.monotonic(), R.OVERLAY_STALE_S
            )
            # None whenever the point is missing or stale: the adapter reads
            # that as "freeze", never as "move somewhere plausible".
            if cursor.enabled and cursor_monitor is not None and desktop is not None:
                cursor.update(
                    None if fresh is None else SC.to_desktop_pixels(fresh, cursor_monitor, desktop)
                )
            display.draw_live(
                fresh,
                raw_fresh,
                unfiltered,
                [
                    f"profile {profile.name}   frames {state.frames}"
                    + (f"   fps {state.fps:.1f}" if state.fps else "   fps --"),
                    f"filter {settings.kind.value} fc={settings.one_euro_min_cutoff_hz} "
                    f"beta={settings.one_euro_beta_hz_per_px_s}",
                    (
                        f"CURSOR {'PAUSED' if cursor.paused else 'MOVING'} -- Esc stops everything"
                        if cursor.enabled
                        else "Esc to stop.  Nothing is recorded and no OS input is sent."
                    ),
                    "Close your eyes about half a second to open the menu.",
                ]
                + _menu_lines(menu),
                tracking=fresh is not None,
            )
            time.sleep(0.005)
    finally:
        # Released before anything else can fail: the pointer goes back where
        # it was found even if the library shutdown then misbehaves.
        cursor.release()
        display.close()
        print(R.shutdown_library(gf))
        if cursor.enabled or cursor.moves:
            print(f"cursor: {cursor.summary()}")
    state = runner.state
    tracked = "unknown"
    if state.frames:
        last = "tracking" if state.tracking else "not tracking"
        tracked = f"{state.frames} frames, last frame {last}"
    print(f"live view ended: {tracked}")
    if state.frames == 0:
        print(
            "no frames reached the view at all -- the camera produced nothing usable. "
            "Check it is uncovered, the room is lit, and no other application is holding it."
        )
    if runner.errors:
        print(f"{runner.errors} frame errors, last: {runner.last_error}")
    if chosen:
        print(f"gesture chose: {chosen[0]}")
        return RECALIBRATE_REQUESTED
    return 0


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
    return run_live(
        profile,
        monitor=monitor,
        show_unfiltered=args.show_unfiltered,
        skip_model_check=args.skip_model_check,
        move_cursor=args.move_cursor,
        confirmed=args.i_mean_it,
        # The profile carries what was found comfortable; a flag overrides it
        # for one run without editing the saved setup.
        cursor_smoothing=_setting(args.cursor_smoothing, profile, "smoothing", 0.35),
        cursor_dead_zone_px=int(_setting(args.cursor_dead_zone_px, profile, "dead_zone_px", 12)),
        cursor_max_step_px=int(_setting(args.cursor_max_step_px, profile, "max_step_px", 400)),
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
