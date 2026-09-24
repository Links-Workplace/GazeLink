"""Gaze, pause and a real left click, on buttons that only change a counter.

This is the first screen in the project where a click can actually reach
Windows, so everything it does is chosen to make that reversible:

* the only actions are "this button's counter went up" and "this button
  flashed", drawn by this window on itself;
* the window is fullscreen on the verified monitor, so a click that lands
  anywhere lands on this window and on nothing else;
* clicking is off unless BOTH ``--click`` and ``--i-mean-it`` are given, the
  same two-gate pattern the cursor adapter uses -- one is the operator saying
  they meant it, the other is the machine agreeing the ruler is trustworthy;
* Esc stops everything, and the pointer is put back where it was found.

What triggers a click is DWELL, and only dwell.  The eyelids are the mode
switch -- paused, move-only, move-and-select -- and never a click, because a
gesture that both changed mode and clicked would fire two things at once and
neither would be attributable.  Winking as a click mode is a separate
experiment, to be measured against dwell rather than run alongside it.

The click lands on the button that was SELECTED, not wherever the pointer
happened to drift to: the pointer follows the gaze through smoothing and a
dead zone, so at the instant the ring completes it can still sit a few pixels
outside the target it filled.  Selecting one button and clicking beside it
would be the worst kind of near miss, because it looks like it worked.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_click as CK  # noqa: E402
import gf_common as C  # noqa: E402
import gf_control as CTL  # noqa: E402
import gf_cursor as CUR  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_dwell as D  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402
import gf_screen_check as SC  # noqa: E402
from gazelink_core.calibration import correction as CORR  # noqa: E402
from gazelink_core.platform import real_input as RI  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "click"

# How long the gaze must rest on the toggle zone to change mode. Far longer
# than a click dwell on purpose: the mode changes twice in a session and a
# click happens constantly, so this one can afford to be slow and must not be
# reachable by a glance on the way between the two buttons.
TOGGLE_DWELL_MS = 1500.0


def toggle_button(buttons: list[D.Button]) -> D.Button:
    """The dead zone between the outermost targets, made into a target.

    Placed there because it is the one region inside the calibrated band that
    is not already a click target, and because a gaze that lands there was
    already going to click nothing -- that gap exists so a horizontally biased
    point activates neither side. Turning it into the mode control spends a
    space that was deliberately empty, and the long dwell is what keeps it
    from being triggered while crossing.
    """

    ordered = sorted(buttons, key=lambda b: b.x0)
    left, right = ordered[0], ordered[-1]
    y0 = min(b.y0 for b in buttons)
    y1 = max(b.y1 for b in buttons)
    return D.Button("MODE", left.x1, y0, right.x0, y1)


def button_centre_px(button: D.Button, monitor: Any, desktop: dict[str, int]) -> tuple[int, int]:
    """The desktop pixel at the middle of a button."""

    return SC.to_desktop_pixels(
        ((button.x0 + button.x1) / 2.0, (button.y0 + button.y1) / 2.0), monitor, desktop
    )


def inside(
    button: D.Button, point_px: tuple[int, int], monitor: Any, desktop: dict[str, int]
) -> bool:
    """Is a desktop pixel within this button's rectangle?"""

    x0, y0 = SC.to_desktop_pixels((button.x0, button.y0), monitor, desktop)
    x1, y1 = SC.to_desktop_pixels((button.x1, button.y1), monitor, desktop)
    return x0 <= point_px[0] <= x1 and y0 <= point_px[1] <= y1


def where_to_click(
    button: D.Button,
    pointer: tuple[int, int] | None,
    monitor: Any,
    desktop: dict[str, int],
) -> tuple[int, int]:
    """Where the click for ``button`` must land.

    The pointer if it is already inside -- moving it would be a visible twitch
    for nothing -- and the button's centre otherwise. Never anywhere else: a
    selection that clicks outside the thing it selected is exactly the near
    miss this function exists to prevent.
    """

    if pointer is not None and inside(button, pointer, monitor, desktop):
        return pointer
    return button_centre_px(button, monitor, desktop)


def _openness_summary(values: list[tuple[float, float]]) -> dict[str, Any]:
    """What the eyelid signal did, so a silent gesture can be explained.

    ``reached_shut`` is the question: did either eye ever fall below the
    threshold the gesture layer is gated on? A run where it never did is a run
    where no eyelid gesture could have worked, whatever the person did.
    """

    if not values:
        return {"frames": 0, "reached_shut": False}
    left = [v[0] for v in values]
    right = [v[1] for v in values]
    shut = [v for v in values if v[0] <= C.BLINK_THRESHOLD or v[1] <= C.BLINK_THRESHOLD]
    return {
        "frames": len(values),
        "threshold": C.BLINK_THRESHOLD,
        "left": {"median": statistics.median(left), "min": min(left)},
        "right": {"median": statistics.median(right), "min": min(right)},
        "frames_at_or_below_threshold": len(shut),
        "reached_shut": bool(shut),
    }


def _ratio_summary(
    values: list[tuple[float, float]], wink: GEST.WinkConfig | None = None
) -> dict[str, Any]:
    """How far each eye actually closed, as a fraction of its own baseline.

    ``deepest`` is the whole question after a run where a wink was made and
    nothing happened: if the right eye never went under the gate fraction, the
    gate is set too high; if it did and no click followed, the fault is later
    in the chain. One number decides which.
    """

    if not values:
        return {"frames": 0}
    left = [v[0] for v in values]
    right = [v[1] for v in values]
    cut = GEST.OpennessGateConfig().shut_fraction
    detector = GEST.RightWinkDetector(wink)
    left_detector = GEST.WinkDetector(wink, eye=GEST.Eye.LEFT)
    return {
        "frames": len(values),
        "gate_shut_under": cut,
        "deepest_left": min(left),
        "deepest_right": min(right),
        "frames_left_under_gate": sum(1 for v in left if v < cut),
        "frames_right_under_gate": sum(1 for v in right if v < cut),
        # Counted with the SAME test the detector uses. Counting it separately
        # is how the previous report could say 112 wink-like frames while the
        # detector fired zero times, and neither number was wrong.
        "frames_looking_like_a_left_wink": sum(
            1 for lv, rv in values if left_detector.looks_like_a_wink(lv, rv)
        ),
        "frames_looking_like_a_right_wink": sum(
            1 for lv, rv in values if detector.looks_like_a_wink(lv, rv)
        ),
        "wink_rule": {
            "right_under": detector.config.shut_ratio,
            "left_at_least_this_many_times_more_open": detector.config.asymmetry,
            "held_ms": detector.config.hold_ms,
        },
    }


@dataclass
class Tally:
    """What happened, kept apart by kind rather than pooled into a score."""

    clicks: int = 0
    per_button: dict[str, int] = field(default_factory=dict)
    suppressed_while_unarmed: int = 0
    # Winks that landed on no button at all. Kept apart from a click: "it did
    # not fire" and "it fired somewhere wrong" are different failures.
    winks_on_no_button: int = 0
    cancelled_selections: int = 0
    pointer_moved_to_centre: int = 0
    # Which eye fired each wink. The eye picks the button (left = left click,
    # right = right click), and the LEFT wink was never measured on this person,
    # so this practice is where it is first seen working -- in simulation.
    winks_by_eye: dict[str, int] = field(default_factory=dict)
    click_points: list[dict[str, int | str]] = field(default_factory=list)
    mode_changes: list[dict[str, Any]] = field(default_factory=list)

    def record_click(self, key: str, landing: tuple[int, int], eye: str | None = None) -> None:
        self.clicks += 1
        self.per_button[key] = self.per_button.get(key, 0) + 1
        # The desktop pixel each click actually went to. Recorded because
        # "it selected LEFT" and "it clicked inside LEFT" are two claims, and
        # a near miss satisfies the first while failing the second.
        point: dict[str, int | str] = {"key": key, "x": int(landing[0]), "y": int(landing[1])}
        if eye is not None:
            point["eye"] = eye
        self.click_points.append(point)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clicks": self.clicks,
            "per_button": dict(self.per_button),
            "winks_by_eye": dict(self.winks_by_eye),
            # Activations the dwell engine produced while selection was NOT
            # armed. Reported because it is the number that says what would
            # have happened had the mode been wrong: a paused run showing zero
            # clicks proves nothing on its own.
            "suppressed_while_unarmed": self.suppressed_while_unarmed,
            "winks_on_no_button": self.winks_on_no_button,
            "cancelled_selections": self.cancelled_selections,
            "pointer_moved_to_centre": self.pointer_moved_to_centre,
            "click_points": list(self.click_points),
            "mode_changes": list(self.mode_changes),
        }


def run_click_practice(
    profile: PROF.Profile,
    *,
    layout: str = "a",
    dwell_ms: float = 900.0,
    click_by: str = "dwell",
    toggle_by: str = "gaze",
    hold_ms: float = 800.0,
    toggle_ms: float = TOGGLE_DWELL_MS,
    advance_timeout_s: float | None = None,
    click: bool = False,
    confirmed: bool = False,
    max_seconds: float | None = None,
    headless: bool = False,
    out: Path | None = None,
) -> dict[str, Any]:
    """Run the window. Returns the report and writes it next to the others."""

    import gf_fit as FIT  # noqa: PLC0415

    if click_by not in ("dwell", "wink"):
        raise SystemExit(f"unknown click mode {click_by!r}: use 'dwell' or 'wink'")
    if toggle_by not in ("gaze", "key", "eyes"):
        raise SystemExit(f"unknown toggle {toggle_by!r}: use 'gaze', 'key' or 'eyes'")
    if click and not confirmed:
        raise SystemExit(
            "--click emits REAL left clicks. Add --i-mean-it to confirm. Without it this "
            "runs as a simulation and records where the clicks would have gone."
        )
    _, how = SC.ensure_per_monitor_dpi_aware()
    print(f"dpi awareness: {how}")
    report = SC.check_profile_screen(profile)
    if click and not report.verified:
        print(SC.format_report(report))
        raise SystemExit(
            "the screen is not verified, so a gaze fraction cannot be trusted as a pixel. "
            "M3-00: no real click on an unverified ruler. Nothing was clicked."
        )
    monitor = GD.pick_monitor(None)
    desktop = SC.virtual_desktop()

    rig = profile.rig_geometry()
    buttons = D.LAYOUTS[layout]()
    by_key = {b.key: b for b in buttons}
    model = CORR.load_with_correction(profile.model_path(), FIT.FittedModel.load)
    gf = L.build_gaze_follower(profile)
    # The eyelid rules come from the PROFILE, so they belong to this person
    # and this camera rather than to whoever last edited the defaults.
    wink_cfg = profile.wink_config()
    gate_cfg = profile.gate_config()
    runner = L.LiveRunner(
        model,
        None,
        rig,
        profile.filter_settings(),
        gesture=GEST.GestureConfig(confirm_ms=hold_ms),
        wink=wink_cfg,
        gate=gate_cfg,
    )
    engine = D.DwellEngine(buttons, D.DwellConfig(dwell_ms=dwell_ms))
    mode_target = toggle_button(buttons)
    # Its own engine, over its own single target, so the click targets and the
    # mode control can never select each other.
    toggle_engine = D.DwellEngine([mode_target], D.DwellConfig(dwell_ms=toggle_ms))
    drawn = [*buttons, mode_target] if toggle_by == "gaze" else list(buttons)
    # Two states, one gesture. The three-mode menu asked the user to remember
    # a sequence before they could click at all; the guards it carried are
    # kept, the choreography is not.
    control = CTL.ToggleMachine()
    # Only ever asked whether the CURRENT frame looks like a wink, for the
    # readout. The detector that decides lives on the camera thread.
    # One per eye: the readout has to light for a LEFT wink too. With only the
    # right-eye rule here, left winks were detected and counted (11 in the run
    # of 24.9) while the screen showed nothing -- which reads as "the left eye
    # is not detected" when it was.
    wink_rules = {eye: GEST.WinkDetector(wink_cfg, eye=eye) for eye in GEST.Eye}
    tally = Tally()
    seen = {"frames": 0, "with_point": 0, "with_face": 0}
    # The gesture signal, kept for the whole run. Section 35 measured that it
    # sometimes never falls anywhere near the blink threshold, and a report
    # that does not carry it cannot tell a gesture that was not made from a
    # gesture the signal could not see.
    openness_seen: list[tuple[float, float]] = []
    # The RATIO is what the gate judges, so it is the number that decides
    # whether a wink was seen. Kept per frame so the deepest point of an
    # actual wink is in the report rather than read off a moving screen.
    ratio_seen: list[tuple[float, float]] = []
    cursor_cfg = profile.cursor or {}
    live = {"on": False}
    gf.add_subscriber(lambda face, gaze: runner.on_frame(face, gaze) if live["on"] else None)
    display = R.Display(rig.device_w_px, rig.device_h_px, headless=headless, origin=(0, 0))
    started = time.monotonic()
    aborted = False

    with (
        CUR.CursorAdapter(
            enabled=click,
            limits=CUR.CursorLimits(
                smoothing=float(cursor_cfg.get("smoothing", 0.6)),
                dead_zone_px=int(cursor_cfg.get("dead_zone_px", 6)),
                max_step_px=int(cursor_cfg.get("max_step_px", 400)),
            ),
        ) as pointer,
        CK.ClickAdapter(enabled=click) as clicker,
    ):
        try:
            gf.camera.start_sampling()
            display.draw_message(["Camera warming up..."])
            if R._sleep_with_escape(display, R.CAMERA_WARMUP_S):
                return {"aborted": True}
            live["on"] = True
            headline = "Click practice" + (
                "  --  REAL CLICKS ARE ON" if click else "  (simulation, no OS input)"
            )
            if (
                display.wait_for_key(
                    [
                        headline,
                        "",
                        "The system starts PAUSED.",
                        "",
                        {
                            "gaze": f"Rest on the middle MODE panel for {toggle_ms / 1000.0:.1f}s"
                            " to switch PAUSED <-> ACTIVE.",
                            "key": "Press SPACE to switch PAUSED <-> ACTIVE.",
                            "eyes": f"Close BOTH eyes for {hold_ms / 1000.0:.1f}s to switch"
                            " PAUSED <-> ACTIVE.",
                        }[toggle_by],
                        "Nothing else changes the mode.",
                        "",
                        (
                            f"While ACTIVE, rest on a button for {dwell_ms:.0f} ms to click it."
                            if click_by == "dwell"
                            else "While ACTIVE, look at a button and WINK your RIGHT eye."
                        ),
                        "Its counter goes up. Nothing else happens.",
                        "",
                        "Esc stops everything.",
                    ],
                    timeout_s=advance_timeout_s,
                )
                == "abort"
            ):
                return {"aborted": True}

            flash: str | None = None
            flash_until = 0.0
            while True:
                now = time.monotonic()
                keys = display.poll_keys()
                if "escape" in keys:
                    aborted = True
                    break
                if max_seconds is not None and now - started > max_seconds:
                    break
                state = runner.state
                point = R.visible_point(state.point, state.updated_s, now, R.OVERLAY_STALE_S)
                have_point = point is not None
                # The mode is interrupted by losing the FACE, never by losing
                # the point. The confirm fires while the eyes are still shut,
                # and the eyes being shut is exactly when there is no point --
                # so driving the interrupt from the point flipped the mode on
                # and straight back off again, in the same frame, every time.
                # Frames must also still be arriving: a dead camera leaves the
                # last face_present standing and would read as a face for ever.
                fresh = state.updated_s is not None and (now - state.updated_s) <= R.OVERLAY_STALE_S
                face_ok = bool(state.face_present) and fresh
                # Counted so a run that drew nothing can be told apart from a
                # run that drew a dot nobody noticed. "I cannot see it" and
                # "it was never there" need different fixes.
                seen["frames"] += 1
                seen["with_point"] += int(have_point)
                seen["with_face"] += int(face_ok)
                if state.openness is not None:
                    openness_seen.append(state.openness)
                if state.openness_ratio is not None:
                    ratio_seen.append(state.openness_ratio)

                def note(change: CTL.Transition, at_s: float) -> None:
                    if change.changed:
                        tally.mode_changes.append(
                            {
                                "at_s": round(at_s - started, 2),
                                "mode": str(change.mode),
                                "why": str(change.reason) if change.reason else None,
                            }
                        )

                transition = None
                if toggle_by == "eyes":
                    for when, event in runner.drain_gesture_events():
                        transition = control.update(event, tracking_ok=face_ok)
                        note(transition, when)
                else:
                    # The eyelid events are still DRAINED, so they cannot pile
                    # up and fire in a burst if the toggle is switched back.
                    runner.drain_gesture_events()
                    asked = False
                    if toggle_by == "key":
                        asked = "space" in keys
                    elif toggle_engine.update(now, point, fresh=have_point) is not None:
                        asked = True
                    if asked:
                        transition = control.update(GEST.Event.CONFIRM, tracking_ok=face_ok)
                        note(transition, now)
                if not face_ok:
                    transition = control.update(GEST.Event.NONE, tracking_ok=False)
                    # Recorded on this path too. Without it a run that paused
                    # itself because the face went reads, in the report, as
                    # having stayed active the whole time.
                    note(transition, now)
                if transition is not None and transition.cancel_selection:
                    engine.reset()
                    tally.cancelled_selections += 1

                mode = control.mode
                target_px = (
                    SC.to_desktop_pixels(point, monitor, desktop)
                    if point is not None and mode.cursor_enabled
                    else None
                )
                held = pointer.update(target_px)

                # In wink mode the dwell engine is never STEPPED, only
                # queried for which button a point is over. A filling ring
                # while the click comes from a wink tells the user that
                # resting will do something, and it will not; and an engine
                # that is not running cannot fire a second click.
                selected: str | None = None
                selected_eye: GEST.Eye | None = None
                if click_by == "dwell":
                    fired = engine.update(now, point, fresh=have_point)
                    if fired is not None:
                        selected = fired.button
                for event in runner.drain_wink_events():
                    _when, aimed_at, *rest = event
                    eye = GEST.Eye(rest[0]) if rest else GEST.Eye.RIGHT
                    tally.winks_by_eye[eye.value] = tally.winks_by_eye.get(eye.value, 0) + 1
                    if click_by != "wink":
                        continue
                    # The gaze point is invalid while an eye is shut, so this
                    # is the last point seen with both eyes open.
                    button = None if aimed_at is None else engine.find(aimed_at)
                    if button is None:
                        tally.winks_on_no_button += 1
                        continue
                    selected = button.key
                    selected_eye = eye

                if selected is not None:
                    if mode.selection_armed:
                        landing = where_to_click(by_key[selected], held, monitor, desktop)
                        if landing != held:
                            pointer.update(landing)
                            tally.pointer_moved_to_centre += 1
                        # The eye picks the button, as in gf_live: right = right.
                        send = (
                            clicker.right_click
                            if selected_eye is GEST.Eye.RIGHT
                            else clicker.click
                        )
                        if send(armed=True):
                            tally.record_click(
                                selected,
                                landing,
                                None if selected_eye is None else selected_eye.value,
                            )
                            flash, flash_until = selected, now + 0.35
                    else:
                        tally.suppressed_while_unarmed += 1

                display.draw_practice(
                    drawn,
                    point,
                    # No ring in wink mode: it would say that resting selects.
                    hovered=(
                        toggle_engine.hovered
                        if toggle_by == "gaze" and toggle_engine.hovered
                        else (engine.hovered if click_by == "dwell" else None)
                    ),
                    progress=(
                        toggle_engine.progress
                        if toggle_by == "gaze" and toggle_engine.hovered
                        else (engine.progress if click_by == "dwell" else 0.0)
                    ),
                    flash=flash if now < flash_until else None,
                    pointer=(
                        None
                        if held is None
                        else (
                            (held[0] - desktop["x"]) / max(1, monitor.width_px - 1),
                            (held[1] - desktop["y"]) / max(1, monitor.height_px - 1),
                        )
                    ),
                    prompt=[
                        control.label,
                        "   ".join(f"{b.key} {tally.per_button.get(b.key, 0)}" for b in buttons),
                        # Shown because a gesture that never fires and a
                        # gesture that fires and is ignored look identical
                        # from the outside. If these numbers do not fall when
                        # the eyes close, no gesture can work and the fault is
                        # upstream of everything on this screen.
                        (
                            f"your eyes  L{state.openness_ratio[0] * 100:.0f}%"
                            f"  R{state.openness_ratio[1] * 100:.0f}%"
                            + "".join(
                                f"   <<< {eye.value.upper()} WINK"
                                for eye, rule in wink_rules.items()
                                if rule.looks_like_a_wink(*state.openness_ratio)
                            )
                            if state.openness_ratio is not None
                            else "eyes --"
                        ),
                        (
                            f"winks seen  left {tally.winks_by_eye.get('left', 0)}"
                            f"   right {tally.winks_by_eye.get('right', 0)}"
                        ),
                        "Esc stops everything",
                    ],
                    tracking=have_point,
                )
                time.sleep(0.005)
        finally:
            display.close()
            R.shutdown_library(gf)

        cursor_summary = pointer.summary()
        click_summary = clicker.summary()

    result = {
        "profile": profile.name,
        "model": str(profile.model_path()),
        "layout": layout,
        "click_by": click_by,
        "toggle_by": toggle_by,
        "wink_rule": {
            "source": "profile" if (profile.gesture or {}).get("wink") else "built-in default",
            **{k: getattr(wink_cfg, k) for k in GEST.WinkConfig.__dataclass_fields__},
        },
        "gate_rule": {
            "source": "profile" if (profile.gesture or {}).get("gate") else "built-in default",
            **{k: getattr(gate_cfg, k) for k in GEST.OpennessGateConfig.__dataclass_fields__},
        },
        "toggle_ms": toggle_ms,
        "hold_ms": hold_ms,
        "dwell_ms": dwell_ms,
        "wink_thresholds_measured": False,
        "os_input": "real left clicks" if click else "none (simulated)",
        "screen_verified": report.verified,
        "dpi_awareness": how,
        "aborted": aborted,
        "seconds": round(time.monotonic() - started, 1),
        "frames": seen["frames"],
        "frames_with_gaze_point": seen["with_point"],
        "frames_with_face": seen["with_face"],
        "eye_openness": _openness_summary(openness_seen),
        "eye_ratio": _ratio_summary(ratio_seen, wink_cfg),
        "tally": tally.to_dict(),
        "cursor": cursor_summary,
        "click_adapter": click_summary,
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    path = out or RESULTS_DIR / f"{layout}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--layout", choices=sorted(D.LAYOUTS), default="a")
    parser.add_argument("--dwell-ms", type=float, default=900.0)
    parser.add_argument(
        "--click-by",
        choices=("dwell", "wink"),
        default="dwell",
        help="what fires a click. 'wink' is a RIGHT-eye wink; its thresholds are not "
        "yet measured on a person, unlike every other threshold in this project.",
    )
    parser.add_argument(
        "--click",
        action="store_true",
        help="emit REAL left clicks on the buttons. Requires --i-mean-it.",
    )
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="confirms --click. Without it, --click is refused rather than simulated.",
    )
    parser.add_argument(
        "--toggle-by",
        choices=("gaze", "key", "eyes"),
        default="gaze",
        help="what switches PAUSED <-> ACTIVE. 'gaze' rests on the middle MODE panel; "
        "'key' is SPACE, for testing the rest of the chain; 'eyes' is the two-eyed close, "
        "which measurement showed does not reach the blink threshold on every session.",
    )
    parser.add_argument("--toggle-ms", type=float, default=TOGGLE_DWELL_MS)
    parser.add_argument(
        "--hold-ms",
        type=float,
        default=800.0,
        help="how long both eyes must stay closed to switch mode. Tune this on yourself: "
        "long enough that no blink reaches it, short enough not to be a chore.",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--advance-timeout-s",
        type=float,
        default=None,
        help="start on its own after this long instead of waiting for a key",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = L.resolve_profile(args.profile)
    # Armed only for this run and only after both flags; see
    # gazelink_core.platform.real_input.
    arming = (
        RI.armed("gf_click_practice --click --i-mean-it")
        if args.click and args.i_mean_it
        else contextlib.nullcontext()
    )
    with arming:
        _practice_from_args(profile, args)
    return 0


def _practice_from_args(profile: PROF.Profile, args: argparse.Namespace) -> None:
    run_click_practice(
        profile,
        layout=args.layout,
        dwell_ms=args.dwell_ms,
        click_by=args.click_by,
        toggle_by=args.toggle_by,
        hold_ms=args.hold_ms,
        toggle_ms=args.toggle_ms,
        click=args.click,
        confirmed=args.i_mean_it,
        max_seconds=args.max_seconds,
        advance_timeout_s=args.advance_timeout_s,
        out=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
