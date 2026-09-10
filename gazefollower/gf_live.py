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
import dataclasses
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_click as CK  # noqa: E402
import gf_common as C  # noqa: E402
import gf_control as CTL  # noqa: E402
import gf_cursor as CUR  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_dwell as D  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_overlay as OV  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402
import gf_scroll as SCR  # noqa: E402

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
    # Whether a FACE is in the frame, which is not the same question as
    # whether there is a usable gaze point. Closing the eyes ends the point
    # and not the face, and a caller that cannot tell the two apart treats
    # every deliberate close as a tracking failure.
    face_present: bool = False
    # The raw per-eye openness the gesture layer is gated on. Exposed so a
    # screen can SHOW it: measured on round34/35/36, this signal never once
    # fell to the blink threshold across forty seconds, which cannot be true
    # of a person's eyelids and means every gesture silently did nothing.
    openness: tuple[float, float] | None = None
    # Each eye as a fraction of its own recent baseline, which is what the
    # gesture layer actually judges. Shown on screen so a gesture that the
    # signal could not see is visible as such.
    openness_ratio: tuple[float, float] | None = None
    # Both eyes open enough for the gaze estimate to be worth moving a pointer
    # with. False through the whole of a wink, including the half-closed part
    # at each end where the blink gate still passes and the prediction is
    # already made from an eye behind its own lid.
    eyes_steady: bool = False
    # Chin-up head pitch (head6 "pitch_a", nose offset over inter-ocular
    # distance; larger = chin higher). Recorded next to the ratios because
    # eye openness is polygon AREA in px^2 -- a PROJECTED area -- so lifting
    # the chin shrinks it with the eye wide open, and the gate's baseline
    # adapts at 0.02 a frame and only while the ratio is already above 0.55.
    # If a pitch change pushes both eyes under that, they read as shut, the
    # baseline stops recovering, and the asymmetry rule cannot pass. This is
    # the number that says whether that is what happens.
    head_pitch: float | None = None
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
        wink: GEST.WinkConfig | None = None,
        gate: GEST.OpennessGateConfig | None = None,
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
        self.wink_detector = GEST.RightWinkDetector(wink)
        # One gate per eye, for the GESTURE path only. `valid` below keeps the
        # absolute threshold, because it decides what counts as a gaze sample
        # and every measurement in this project was made against it.
        self.left_gate = GEST.OpennessGate(gate)
        self.right_gate = GEST.OpennessGate(gate)
        self.gesture_events: list[tuple[float, GEST.Event]] = []
        self.wink_events: list[tuple[float, tuple[float, float] | None]] = []
        self.lock = threading.Lock()
        self.state = LiveState()
        self.frames = 0
        self._last_steady_point: tuple[float, float] | None = None
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
        # Named for the PERSON from here down. The camera faces them, so the
        # library's "left" is the eye on the left of the IMAGE, which is their
        # right one. Every gesture in this project watched the wrong eye.
        left, right = GEST.eyes_as_the_person_has_them(
            float(getattr(face_info, "left_eye_openness", 0.0) or 0.0),
            float(getattr(face_info, "right_eye_openness", 0.0) or 0.0),
        )
        head = self.head_builder(face_info) if gaze_status else None

        # Same gate as the recorder's overlay: a blink is not a gaze sample,
        # and predicting through one puts the point somewhere the eye is not.
        valid = gaze_status and left > C.BLINK_THRESHOLD and right > C.BLINK_THRESHOLD
        # The gesture reads the FACE, not the gaze: it has to keep working
        # when the gaze is unusable, because "eyes shut" is precisely when
        # there is no gaze.
        face_present = bool(getattr(face_info, "status", False))
        # BOTH eyes, not either. Written as "not (left and right)" this was
        # true during a one-eyed wink as well, so a wink drove the mode menu
        # and a wink-to-click would have fired two mechanisms from one
        # gesture. The two signals are now exclusive by construction.
        # Judged against each eye's own baseline. Measured live on 9.9: over
        # 627 frames the openness never once reached the absolute threshold of
        # 10.0 -- minima of 27.0 and 64.5 against medians of 191.5 and 160.0 --
        # so every eyelid gesture was silently impossible. The same minima are
        # 0.14 and 0.40 of their own baselines, which is readable.
        left_open = self.left_gate.is_open(left)
        right_open = self.right_gate.is_open(right)
        eyes_shut = not left_open and not right_open
        event = self.detector.update(now_s, face_present=face_present, eyes_shut=eyes_shut)
        # Ratios, not the open/shut booleans: the operator's left eye narrows
        # whenever the right one closes, so a rule that needed it OPEN rejected
        # every real wink. Comparing the two depths does not.
        left_ratio = self.left_gate.ratio if self.left_gate.ratio is not None else 1.0
        right_ratio = self.right_gate.ratio if self.right_gate.ratio is not None else 1.0
        winked = self.wink_detector.update(
            now_s,
            face_present=face_present,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
        )
        # An eye on its way down still passes the blink gate, so a prediction
        # is still produced -- from an eye already half behind its lid. That
        # is what makes the pointer wander at the start of a wink, well before
        # anything registers the closure.
        steady = self.left_gate.config.steady_fraction
        eyes_steady = valid and left_ratio >= steady and right_ratio >= steady
        # ``head`` above is built only when the gaze is usable, and the whole
        # question here is what the head was doing while an eye was SHUT, so
        # the pose is taken again for any frame carrying a face. It is used
        # for reporting only -- the prediction below still takes ``head``.
        pose = head
        if pose is None and face_present:
            pose = self.head_builder(face_info)
        head_pitch = float(pose[2]) if pose is not None and len(pose) > 2 else None
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
            if winked:
                # The point from before the eye began to close, not merely the
                # last one that passed the blink gate: the half-closed frames
                # pass it too, and they are the ones that put the pointer
                # somewhere the person was never looking.
                self.wink_events.append((now_s, self._last_steady_point))
            if eyes_steady and point is not None:
                self._last_steady_point = point
            self.frames += 1
            self._frame_times.append(now_s)
            if len(self._frame_times) > 60:
                del self._frame_times[:-60]
            self.state = LiveState(
                face_present=face_present,
                openness=(left, right),
                openness_ratio=(self.left_gate.ratio or 0.0, self.right_gate.ratio or 0.0),
                eyes_steady=eyes_steady,
                head_pitch=head_pitch,
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

    def drain_wink_events(self) -> list[tuple[float, tuple[float, float] | None]]:
        """Deliberate right winks, each with the gaze point it was aimed at."""

        with self.lock:
            events = self.wink_events
            self.wink_events = []
        return events

    def _fps(self, window: int = 30) -> float | None:
        times = self._frame_times[-window:]
        if len(times) < 2:
            return None
        span = times[-1] - times[0]
        return None if span <= 0 else (len(times) - 1) / span


def _ratio_summary(values: list[tuple[float, float]], rule: GEST.WinkConfig) -> str:
    """How close the eyes came to the wink rule over a whole session.

    A session that recorded no winks has to be able to say WHY: the eye never
    closed far enough, the other eye came with it, or the rule was never the
    problem. Without this the only report was "0 winks", which says none of
    those three.
    """

    if not values:
        return "no frames"
    probe = GEST.RightWinkDetector(rule)
    # Rows are (left, right) or (left, right, pitch): the pitch was added later
    # and every earlier caller still passes the pair.
    right = [row[1] for row in values]
    left = [row[0] for row in values]
    deep = min(right)
    matching = sum(1 for row in values if probe.looks_like_a_wink(row[0], row[1]))
    at_deepest = min(values, key=lambda row: row[1])
    return (
        f"{len(values)} frames; right reached {deep:.3f} (rule needs under {rule.shut_ratio}), "
        f"left was {at_deepest[0]:.3f} there (needs {rule.asymmetry}x the right, so over "
        f"{at_deepest[1] * rule.asymmetry:.3f}); left reached {min(left):.3f}; "
        f"{matching} frames matched the rule"
    )


def _pitch_summary(
    values: list[tuple[float, float, float | None]], rule: GEST.WinkConfig, bands: int = 5
) -> list[str]:
    """Does the wink rule pass or fail depending on where the head is?

    Reported live: raising the chin freezes the pointer but produces no click,
    and lowering it toward the lens makes the same wink work. That is a
    testable claim, because eye openness here is polygon AREA in px^2 -- a
    PROJECTED area, which shrinks as the eye is seen more edge-on -- while the
    gate normalises it against a baseline that moves at 0.02 a frame and only
    while the ratio is already above 0.55. If lifting the chin pushes both
    eyes under that, they read as shut together, the baseline stops recovering
    and the asymmetry rule cannot pass however hard the person winks.

    So: split the session by head pitch and show, per band, how deep the right
    eye got and how often the rule was satisfied. If the matching frames sit
    in one band and the rest sit in another, the camera angle is the story.
    """

    known = [row for row in values if len(row) > 2 and row[2] is not None]
    if not known:
        return ["  head pitch             : no head pose on any frame"]
    probe = GEST.RightWinkDetector(rule)
    pitches = [row[2] for row in known]
    lo, hi = min(pitches), max(pitches)
    lines = [
        f"  head pitch             : {len(known)} frames with a pose, "
        f"{lo:+.3f} to {hi:+.3f} (higher = chin up)",
    ]
    matched = [row[2] for row in known if probe.looks_like_a_wink(row[0], row[1])]
    if matched:
        ordered = sorted(matched)
        lines.append(
            f"    frames matching the rule: {len(matched)}, pitch "
            f"{ordered[0]:+.3f} to {ordered[-1]:+.3f}, median {ordered[len(ordered) // 2]:+.3f}"
        )
    else:
        lines.append("    frames matching the rule: none, so there is no pitch to compare")
    if hi - lo < 1e-6:
        lines.append("    the head never moved enough to split into bands")
        return lines
    width = (hi - lo) / bands
    lines.append(
        f"    {'pitch band':>18} {'frames':>7} {'deepest R':>10} "
        f"{'median R':>9} {'matched':>8}"
    )
    for b in range(bands):
        low = lo + b * width
        high = hi if b == bands - 1 else low + width
        rows = [(row[0], row[1]) for row in known if low <= row[2] <= high]
        if not rows:
            continue
        rights = sorted(rv for _lv, rv in rows)
        hits = sum(1 for lv, rv in rows if probe.looks_like_a_wink(lv, rv))
        lines.append(
            f"    {low:+.3f}..{high:+.3f} {len(rows):>7} {rights[0]:>10.3f} "
            f"{rights[len(rights) // 2]:>9.3f} {hits:>8}"
        )
    return lines


def ratio_watcher(config: GEST.WinkConfig | None = None) -> Any:
    """A per-frame view of what the wink rule makes of the current eyes.

    Exists so a live view can SHOW it. Reported live: 'I winked' against a
    session that recorded zero winks, with nothing on screen or in the log
    that could say whether the signal ever came near the rule.
    """

    detector = GEST.RightWinkDetector(config)

    def matches(state: Any) -> bool:
        ratio = getattr(state, "openness_ratio", None)
        return bool(ratio is not None and detector.looks_like_a_wink(*ratio))

    return matches


def state_face_ok(runner: Any, *, stale_after_s: float = R.OVERLAY_STALE_S) -> bool:
    """Is a face in front of the camera right now?

    Not the same question as "is there a usable gaze point": closing the eyes
    ends the point and not the face, and a caller that cannot tell them apart
    treats every deliberate close as a tracking failure and pauses in the same
    frame the close armed it.

    Frames must still be arriving, or a dead camera leaves the last answer
    standing and reads as a face for ever.
    """

    state = runner.state
    if state.updated_s is None:
        return False
    if time.monotonic() - state.updated_s > stale_after_s:
        return False
    return bool(state.face_present)


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
    click_by: str = "off",
    wink_click: str = "double",
    wink_hold_ms: float | None = None,
    scroll_arm_ms: float | None = None,
    scroll_repeat_ms: float | None = None,
    scroll_toggle_ms: float = 1500.0,
    start_scrolling: bool = False,
    toggle_by: str = "key",
    start_active: bool = True,
    hold_after_click_s: float = 1.5,
) -> int:
    import gf_fit as FIT  # noqa: PLC0415
    import gf_screen_check as SC  # noqa: PLC0415

    if click_by not in ("off", "wink"):
        raise SystemExit(f"unknown click mode {click_by!r}: use 'off' or 'wink'")
    if toggle_by not in ("key", "eyes"):
        raise SystemExit(f"unknown toggle {toggle_by!r}: use 'key' or 'eyes'")
    if wink_click not in ("single", "double"):
        raise SystemExit(f"unknown wink action {wink_click!r}: use 'single' or 'double'")
    if click_by == "wink" and not move_cursor:
        raise SystemExit(
            "--click-by wink needs --move-cursor: a click that lands wherever the pointer "
            "was last left is not a click at what you are looking at."
        )
    # Built once and used everywhere, so the readout, the detector and the
    # report cannot end up judging by three different rules.
    wink_cfg = profile.wink_config()
    if wink_hold_ms is not None:
        wink_cfg = dataclasses.replace(wink_cfg, hold_ms=wink_hold_ms)
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
    # The eyelid rules come from the PROFILE: they belong to this face and
    # this camera geometry, not to whoever last edited the defaults.
    runner = LiveRunner(model, None, rig, settings, wink=wink_cfg, gate=profile.gate_config())
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
        # A fullscreen window IS the thing that gets clicked, so a real click
        # over it never reaches the application underneath. Clicking on the
        # desktop needs a window the mouse passes through.
        click_through=click_by == "wink",
    )
    cursor = CUR.CursorAdapter(
        enabled=move_cursor,
        limits=CUR.CursorLimits(
            max_step_px=cursor_max_step_px,
            smoothing=cursor_smoothing,
            dead_zone_px=cursor_dead_zone_px,
        ),
    )
    # Clicking on the DESKTOP is not clicking in a practice window. There the
    # only action was a counter this window drew on itself; here a click can
    # close, delete or send, and cannot be taken back. So it is off unless
    # asked for twice, it starts PAUSED whatever the cursor flag says, and the
    # mode is an explicit state rather than a side effect of the pause menu.
    clicker = CK.ClickAdapter(enabled=click_by == "wink" and move_cursor)
    # Scrolling rides on the same two gates as clicking: it is OS input, so it
    # needs the operator to have asked for it and the pointer to be allowed to
    # move. The wheel goes to whatever is under the pointer.
    scroller = CK.ScrollAdapter(enabled=click_by == "wink" and move_cursor)
    scroll_cfg = SCR.ScrollConfig(
        **{
            name: value
            for name, value in (
                ("arm_ms", scroll_arm_ms),
                ("repeat_ms", scroll_repeat_ms),
            )
            if value is not None
        }
    )
    scroll_bands = SCR.scroll_zones(scroll_cfg)
    # With --start-scrolling there is no tile at all. It sits in the middle of
    # the band, which is exactly where the gaze RESTS to stop scrolling, so a
    # session that is only for reading kept being thrown out of scroll mode by
    # the act of stopping. Reported as "it still sends me to the scroll
    # square". Esc is the way out of a reading session.
    scroll_tile = None if start_scrolling else SCR.scroll_tile(scroll_cfg)
    repeater = SCR.ScrollRepeater(scroll_cfg)
    # Its own engine over its own single target, so the way in and out can
    # never be selected by anything else and nothing else can be selected by
    # it -- the same separation gf_click_practice uses for its MODE tile. The
    # dwell is longer than the scroll wait, so crossing the tile on the way
    # somewhere else cannot flip the mode.
    tile_engine = (
        None
        if scroll_tile is None
        else D.DwellEngine([scroll_tile], D.DwellConfig(dwell_ms=scroll_toggle_ms))
    )
    # Starting ACTIVE is the operator's decision, asked for directly after
    # the paused start left them with no way in: the toggle they had in the
    # practice window was a gaze panel, and there is nowhere to put one on a
    # desktop. The two OS gates still stand in front of this -- nothing runs
    # without --move-cursor and --i-mean-it -- and Esc still stops it. What is
    # given up is the beat between launching and the first click being
    # possible, so the window opens ready to click.
    control = (
        CTL.ToggleMachine(start=CTL.ToggleMachine.ACTIVE if start_active else CTL.Mode.PAUSED)
        if click_by == "wink"
        else None
    )
    tally = {
        "clicks": 0,
        "doubles": 0,
        # Every wink the detector produced, before any of the reasons one
        # might not become a click. Without it, "nothing happened" cannot be
        # told apart from "the wink was seen and then dropped", and the two
        # need opposite fixes.
        "winks": 0,
        "winks_with_no_aim": 0,
        # Scrolling. Counted separately from clicks throughout: "the wheel
        # turned" and "a click happened" are different outcomes and the report
        # exists to tell them apart.
        "notches": 0,
        "scroll_sessions": 0,
        "scroll_cancelled": 0,
        "winks_while_scrolling": 0,
        "winks_dropped_at_scroll_edge": 0,
        "suppressed_while_paused": 0,
        "cancelled": 0,
    }
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
        clicker.__enter__()
        if control is not None:
            cursor.paused = not control.mode.cursor_enabled
            print(f"starting {control.label}")
        started = time.monotonic()
        wink_rule_matches = ratio_watcher(wink_cfg)
        # Every frame's ratios, so a session that recorded no winks can still
        # say how close the signal came. Two floats per frame, nothing else.
        ratios_seen: list[tuple[float, float, float | None]] = []
        # Edge, not level: a key held for half a second is one instruction,
        # and reading the level would toggle the mode on every frame it is
        # down -- sixty times a second, ending wherever the release happened.
        space_was_down = False
        # Scroll mode is ORTHOGONAL to the ToggleMachine on purpose. Adding a
        # third Mode would have meant changing the boolean flip in four places
        # in gf_control and the meaning of ``cursor_enabled`` everywhere, and
        # breaking five tests, to express something that is not a third kind
        # of "how much control does the person have".
        # Opening straight into scroll mode skips the tile, for when the
        # session is only for reading. The pointer is left exactly where it
        # was found -- there is no content point yet to jump back to, and the
        # place the mouse was last put IS the content the person means.
        scrolling = bool(start_scrolling) and click_by == "wink"
        # Where the pointer was while the gaze was last on CONTENT, in pixels.
        content_px: tuple[int, int] | None = None
        seen_in: dict[str, int] = {
            SCR.UP: 0,
            SCR.DOWN: 0,
            "the middle": 0,
            "the tile": 0,
            "no point": 0,
        }
        while True:
            # A click-through window never takes focus, so pygame stops seeing
            # key presses the moment the overlay starts working. The stop is
            # read from the keyboard directly, or it would fail exactly while
            # the dangerous mode was on.
            if display.poll_escape() or (control is not None and OV.escape_is_down()):
                break
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                break
            # Read ONCE per frame and used everywhere below. The runner builds
            # a fresh LiveState on every access, so reading it twice in a frame
            # can answer the same question two different ways.
            state = runner.state
            if control is not None:
                # ONE tick per frame, always, whether or not anything happened.
                # The machine is a per-frame machine: it clears the guard that
                # follows a tracking loss on the first clean frame it is GIVEN.
                # Calling it only when there was an event meant that a face
                # coming back, with no key pressed, never produced a call --
                # so it sat in WAITING for ever with the face plainly in view.
                # That is what the operator saw.
                events: list[GEST.Event] = []
                if toggle_by == "eyes":
                    events = [e for _when, e in runner.drain_gesture_events()]
                else:
                    # Drained and dropped, so they cannot pile up and fire in
                    # a burst if the toggle is ever switched back.
                    runner.drain_gesture_events()
                    down = OV.key_is_down(OV.VK_SPACE)
                    if down and not space_was_down:
                        events = [GEST.Event.CONFIRM]
                    space_was_down = down
                face_ok = state_face_ok(runner)
                if state.openness_ratio is not None:
                    # getattr: a test double may carry only the ratios, and a
                    # missing pose must read as "unknown", never as a number.
                    ratios_seen.append(
                        (*state.openness_ratio, getattr(state, "head_pitch", None))
                    )
                for event in events or [GEST.Event.NONE]:
                    transition = control.update(event, tracking_ok=face_ok)
                    if transition.changed:
                        print(control.label)
                    if transition.cancel_selection:
                        tally["cancelled"] += 1
                cursor.paused = not control.mode.cursor_enabled
            else:
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
                # None means freeze. An eye on its way down still passes the
                # blink gate, so `fresh` is a real point made from an eye that
                # is already half behind its lid -- and following it is what
                # makes the pointer lurch the moment a wink starts, before
                # anything has registered a closure at all. Holding still from
                # the first sign of a closure is also what lets the click land
                # where the person was looking rather than where the estimate
                # slid to on the way down.
                steady = control is None or state.eyes_steady
                # Only a steady point may ask for anything. A half-closed eye
                # still produces a point, and it must not be able to request a
                # scroll any more than it may move the pointer.
                aim = fresh if steady else None
                now = time.monotonic()
                if control is not None:
                    over_tile = (
                        scroll_tile is not None and aim is not None and scroll_tile.contains(aim)
                    )
                    zone = next(
                        (z.key for z in scroll_bands if aim is not None and z.contains(aim)),
                        None,
                    )
                    if not face_ok or not control.mode.cursor_enabled:
                        # The wheel stops either way: nothing repeats while
                        # the face is gone or the mode is paused.
                        repeater.stop()
                        if tile_engine is not None:
                            tile_engine.reset()
                        # Whether the MODE also ends depends on there being a
                        # way back into it. With the tile, a pause cancels it
                        # and the person asks again -- input that repeats
                        # should be asked for deliberately.
                        #
                        # Without the tile there is no way to ask, and
                        # cancelling was a trap: ``state_face_ok`` is false
                        # for the first frames of EVERY session, before the
                        # camera has delivered anything, so --start-scrolling
                        # was switched off a moment after it started and the
                        # bands were never drawn at all. Reported as "I cannot
                        # see the scroll bands". The launch flag IS the
                        # deliberate request, and it stands for the session;
                        # resting on a band for the wait is still required
                        # before anything moves.
                        if scrolling and tile_engine is not None:
                            scrolling = False
                            tally["scroll_cancelled"] += 1
                            print("scroll mode off - tracking or mode was lost")
                    elif (
                        tile_engine is not None
                        and tile_engine.update(now, aim, fresh=aim is not None) is not None
                    ):
                        # One activation per entry: the engine latches until
                        # the gaze is seen elsewhere, so resting on the tile
                        # cannot flip the mode straight back again.
                        scrolling = not scrolling
                        repeater.stop()
                        # A wink that fired while scrolling must not arrive at
                        # the click path now that scrolling is over, and one
                        # that fired on the way in must not click either. The
                        # event carries a COPY of its point and sits in the
                        # queue until this loop drains it, so the queue is
                        # dropped rather than trusted.
                        dropped = len(runner.drain_wink_events())
                        tally["winks_dropped_at_scroll_edge"] += dropped
                        print(f"scroll mode {'on' if scrolling else 'off'}")
                        if scrolling:
                            tally["scroll_sessions"] += 1
                            if content_px is not None:
                                # Back to the content. Looking at the tile
                                # dragged the pointer to the tile, and the
                                # wheel goes to whatever is under the pointer,
                                # so freezing it where it happens to be would
                                # scroll the tile's own corner of the screen.
                                cursor.release_hold()
                                cursor.jump_to(content_px)
                    if scrolling:
                        # Where the gaze actually went while scroll mode was
                        # on. This is the reachability question and nothing
                        # else answers it: "I looked up and nothing happened"
                        # and "the band never saw me" are the same sentence
                        # from the person and different numbers here.
                        if aim is None:
                            seen_in["no point"] += 1
                        elif zone is not None:
                            seen_in[zone] += 1
                        elif over_tile:
                            seen_in["the tile"] += 1
                        else:
                            seen_in["the middle"] += 1
                        # Frozen for the whole of scroll mode, so the wheel
                        # keeps going to the window the person aimed at.
                        cursor.paused = True
                        notches = repeater.update(now, zone, usable=aim is not None)
                        if notches and scroller.scroll(notches, armed=True):
                            tally["notches"] += abs(notches)
                    elif aim is not None and not over_tile and zone is None:
                        # The pointer's position while the gaze is on CONTENT
                        # -- not the tile, not a band. Taking "where it was
                        # before the dwell started" instead would be too late:
                        # by then the gaze is already ON the tile and the
                        # pointer has followed it there.
                        content_px = SC.to_desktop_pixels(aim, cursor_monitor, desktop)
                if not scrolling:
                    cursor.update(
                        SC.to_desktop_pixels(fresh, cursor_monitor, desktop)
                        if fresh is not None and steady
                        else None
                    )
                for _when, aimed_at in runner.drain_wink_events():
                    if control is None:
                        continue
                    tally["winks"] += 1
                    if scrolling:
                        # The eyes are driving the wheel, not choosing a
                        # target. Without this a wink mid-scroll would pass
                        # ``selection_armed`` -- which is still true, because
                        # the mode is still ACTIVE -- and double click on
                        # whatever the page had scrolled under the pointer.
                        tally["winks_while_scrolling"] += 1
                        continue
                    if not control.mode.selection_armed:
                        tally["suppressed_while_paused"] += 1
                        continue
                    if aimed_at is None:
                        # The gaze is invalid while an eye is shut, and with
                        # nothing remembered from before it there is no place
                        # the wink can honestly mean. Counted, because "no
                        # wink was seen" and "a wink was seen and had nowhere
                        # to go" are different failures with different fixes.
                        tally["winks_with_no_aim"] += 1
                        continue
                    # Put the pointer where the eye was when the wink STARTED,
                    # then click there. Clicking wherever the pointer drifted
                    # to is how a click lands next to what was being looked at.
                    landing = SC.to_desktop_pixels(aimed_at, cursor_monitor, desktop)
                    # Release first: the pointer must be allowed to reach the
                    # new place before the hold pins it there, or a second
                    # wink would click wherever the FIRST one landed.
                    cursor.release_hold()
                    # jump_to, not update: smoothing would leave the pointer
                    # 40% short of the target and the dead zone would refuse
                    # small corrections outright, so the click would land
                    # between where the pointer was and where it was aimed.
                    cursor.jump_to(landing)
                    # One wink, one whole action. Asking the person to make two
                    # winks inside the system's 500 ms window is a timing test
                    # they can fail through no fault of their own, and failing
                    # it silently produces two clicks that open nothing.
                    landed = (
                        clicker.double_click(armed=True, at=cursor.last)
                        if wink_click == "double"
                        else clicker.click(armed=True, at=cursor.last)
                    )
                    if landed:
                        tally["clicks"] += 1
                        tally["doubles"] = clicker.doubles
                        # Hold after, not before: the click has landed, and
                        # now the person needs a moment to see what happened
                        # and to wink again if they meant a double.
                        cursor.hold_for(hold_after_click_s)
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
                        f"{control.label}   clicks {tally['clicks']}"
                        f" (double {tally['doubles']})"
                        + ("   HELD" if cursor.held_until_s else "")
                        + "   -- Esc stops everything"
                        if control is not None
                        else (
                            f"CURSOR {'PAUSED' if cursor.paused else 'MOVING'}"
                            " -- Esc stops everything"
                            if cursor.enabled
                            else "Esc to stop.  Nothing is recorded and no OS input is sent."
                        )
                    ),
                    (
                        (
                            (
                                "SCROLLING.  Look UP or DOWN to scroll, at the middle to stop, "
                                "at SCROLL to come back."
                            )
                            if scrolling
                            else (
                                f"SPACE switches PAUSED/ACTIVE.  Wink RIGHT to {wink_click}-click."
                                "  Rest on SCROLL to scroll."
                                if toggle_by == "key"
                                else "Close BOTH eyes to switch PAUSED/ACTIVE."
                                f"  Wink RIGHT to {wink_click}-click.  Rest on SCROLL to scroll."
                            )
                        )
                        if control is not None
                        else "Close your eyes about half a second to open the menu."
                    ),
                ]
                # The same readout the practice window has, and the reason
                # this view could not diagnose itself: a wink the signal never
                # saw and a wink that was seen and dropped look identical
                # without it.
                + (
                    [
                        f"your eyes  L{state.openness_ratio[0] * 100:.0f}%"
                        f"  R{state.openness_ratio[1] * 100:.0f}%"
                        + ("   <<< WINK" if wink_rule_matches(state) else "")
                        + f"   winks {tally['winks']}"
                    ]
                    if control is not None and state.openness_ratio is not None
                    else []
                )
                + _menu_lines(menu),
                tracking=fresh is not None,
                # Shown only in scroll mode. Bands on screen the rest of the
                # time would be clutter over whatever the person is reading,
                # and the tile is drawn always so there is a visible way IN.
                zones=(
                    ([*scroll_bands] if scrolling else [])
                    + ([scroll_tile] if scroll_tile is not None else [])
                )
                if control is not None
                else [],
                active_zone=repeater.zone if scrolling else None,
            )
            time.sleep(0.005)
    finally:
        if control is not None:
            # Printed rather than left to be guessed at. Three separate
            # sessions were spent arguing about whether a click had happened,
            # because nothing said what the mode was, whether the overlay was
            # transparent, or whether the pointer had been allowed to move.
            print("\nsession:")
            print(f"  mode at the end        : {control.label}")
            print(f"  toggled by             : {toggle_by}")
            print(f"  clicks                 : {tally['clicks']} (double {tally['doubles']})")
            print(f"  winks detected         : {tally['winks']}")
            print(f"  winks with nowhere to go: {tally['winks_with_no_aim']}")
            print(f"  winks while paused     : {tally['suppressed_while_paused']}")
            print(f"  wink action            : {wink_click}")
            print(f"  wink rule              : {wink_cfg}")
            print(
                f"  scrolling              : {tally['notches']} notches over "
                f"{tally['scroll_sessions']} entries "
                f"({tally['scroll_cancelled']} ended by a pause or a lost face)"
            )
            print(f"  scroll rule            : {scroll_cfg}")
            print(f"  winks while scrolling  : {tally['winks_while_scrolling']} (none clicked)")
            print(
                f"  winks dropped entering/leaving scroll: "
                f"{tally['winks_dropped_at_scroll_edge']}"
            )
            print(f"  scroll adapter         : {scroller.summary()}")
            looked = sum(seen_in.values())
            if looked:
                where = ", ".join(
                    f"{name} {count} ({100 * count / looked:.0f}%)"
                    for name, count in seen_in.items()
                    if count
                )
                print(f"  while scrolling, the gaze was in: {where}")
            else:
                print("  while scrolling, the gaze was in: scroll mode was never entered")
            print(f"  eye ratios             : {_ratio_summary(ratios_seen, wink_cfg)}")
            for line in _pitch_summary(ratios_seen, wink_cfg):
                print(line)
            print(f"  overlay                : {display.overlay}")
            print(f"  cursor                 : {cursor.summary()}")
            print(f"  click adapter          : {clicker.summary()}")
        # The button first: a held left button turns every later pointer move
        # into a drag over whatever is on screen, and unlike a stray click it
        # does not stop happening.
        clicker.release()
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
        "--scroll-toggle-ms",
        type=float,
        default=1500.0,
        help="how long to rest on the SCROLL tile to enter or leave scroll mode. Longer "
        "than the band wait on purpose, so crossing the tile cannot flip the mode.",
    )
    parser.add_argument(
        "--start-scrolling",
        action="store_true",
        help="open already in scroll mode, so the bands work straight away and the SCROLL "
        "tile is only needed to come BACK to the pointer. Leave the mouse on the page you "
        "want to read before starting: the pointer is frozen where it is found.",
    )
    parser.add_argument(
        "--wink-click",
        choices=("single", "double"),
        default="double",
        help="what ONE wink does. 'double' is the default because it is what opens things, "
        "and because asking for two winks inside Windows' 500 ms window is a timing test a "
        "person can fail through no fault of their own.",
    )
    parser.add_argument(
        "--toggle-by",
        choices=("key", "eyes"),
        default="key",
        help="what switches PAUSED <-> ACTIVE. 'key' is SPACE and works whoever has focus, "
        "which the overlay never does. 'eyes' is the two-eyed close; there is no MODE panel "
        "on the desktop, so if that close does not register there is no way back in.",
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
        default=1.5,
        help="how long the pointer stays where it clicked before following the gaze again. "
        "This is also the window in which a second wink becomes a double click, because both "
        "halves have to land on the same pixel.",
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
        click_by=args.click_by,
        wink_click=args.wink_click,
        wink_hold_ms=args.wink_hold_ms,
        scroll_arm_ms=args.scroll_arm_ms,
        scroll_repeat_ms=args.scroll_repeat_ms,
        scroll_toggle_ms=args.scroll_toggle_ms,
        start_scrolling=args.start_scrolling,
        toggle_by=args.toggle_by,
        start_active=not args.start_paused,
        hold_after_click_s=args.hold_after_click_s,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
