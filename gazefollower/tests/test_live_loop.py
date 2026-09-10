"""The live view's loop, driven with fakes. No camera, no window, no OS input.

Every piece here was already tested on its own. What is tested here is the
ASSEMBLY, because that is where this project keeps losing sessions: the mode
machine, the face-presence check and the click adapter were each correct while
the loop that drives them was not.

The case that motivated the file: the machine is a per-frame machine, and the
loop called it only when something happened. A face coming back with no key
pressed produced no call at all, so the view sat in WAITING FOR YOUR FACE with
the face plainly in view, and nothing on screen or in the code said why.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_control as CTL  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402

MONITOR = SimpleNamespace(origin=(0, 0), width_px=5120, height_px=1440, name="FAKE")
DESKTOP = {"x": 0, "y": 0, "width": 5120, "height": 1440}
ON_SCREEN = (0.5, 0.5)


def _profile() -> PROF.Profile:
    return PROF.Profile(
        name="t",
        # An existing directory: run_live checks the path before loading, and
        # FittedModel.load is faked, so any real directory will do.
        model_dir=str(Path(__file__).resolve().parent),
        rig={
            "camera_x_cm": 60.0,
            "camera_y_cm": 63.6,
            "screen_w_cm": 120.0,
            "screen_h_cm": 33.75,
            "device_w_px": 5120,
            "device_h_px": 1440,
        },
        filter={"kind": "one-euro"},
        cursor={"smoothing": 1.0, "dead_zone_px": 0, "max_step_px": 5000},
    )


class _Display:
    def __init__(self, *a: object, **kw: object) -> None:
        self.frames = 0
        self.huds: list[list[str]] = []
        self.overlay = None
        self.zones: list[list[str]] = []
        self.active_zones: list[str | None] = []

    def draw_message(self, *a: object, **kw: object) -> None:
        return None

    def poll_escape(self) -> bool:
        return False

    def draw_live(  # noqa: ANN001
        self, point, raw, unfiltered, hud, *, tracking, zones=(), active_zone=None
    ) -> None:
        self.frames += 1
        self.huds.append(list(hud))
        self.zones.append([z.key for z in zones])
        self.active_zones.append(active_zone)

    def close(self) -> None:
        return None


class _Runner:
    """No face for the first ``blind_reads`` looks, then a face, always."""

    def __init__(
        self,
        *,
        blind_reads: int = 0,
        winks: list[Any] | None = None,
        points: list[Any] | None = None,
        wink_after: int | None = None,
        blind_after: int | None = None,
    ) -> None:
        self._reads = 0
        self._blind = blind_reads
        # A face that is there and then goes, which is the opposite of
        # ``blind_reads`` and the only way to test losing one mid-session.
        self._blind_after = blind_after
        self._winks = list(winks or [])
        # Where the gaze is, read by read. The last entry is held once the
        # list runs out, so a test can steer the gaze somewhere and leave it.
        self._points = list(points or [])
        self._wink_after = wink_after
        self.errors = 0
        self.last_error = None

    @property
    def state(self):  # noqa: ANN201
        self._reads += 1
        present = self._reads > self._blind
        if self._blind_after is not None and self._reads > self._blind_after:
            present = False
        here = ON_SCREEN
        if self._points:
            here = self._points[min(self._reads - 1, len(self._points) - 1)]
        return SimpleNamespace(
            point=here if present else None,
            raw_model=None,
            unfiltered=None,
            tracking=present,
            eyes_steady=present,
            face_present=present,
            openness=(150.0, 130.0),
            openness_ratio=(1.0, 1.0),
            updated_s=time.monotonic(),
            frames=self._reads,
            fps=30.0,
        )

    def on_frame(self, *a: object) -> None:
        return None

    def drain_gesture_events(self) -> list[Any]:
        return []

    def drain_wink_events(self) -> list[Any]:
        # Held back until the loop has run far enough, so a test can place a
        # wink AFTER the gaze has arrived somewhere rather than before it.
        if self._wink_after is not None and self._reads <= self._wink_after:
            return []
        out, self._winks = self._winks, []
        return out


def _run(
    test: unittest.TestCase,
    *,
    blind_reads: int = 0,
    winks: list[Any] | None = None,
    space_down: bool = False,
    start_active: bool = True,
    max_seconds: float = 0.4,
    points: list[Any] | None = None,
    wink_after: int | None = None,
    blind_after: int | None = None,
    scroll_toggle_ms: float = 1500.0,
    start_scrolling: bool = False,
    scroll_arm_ms: float | None = None,
    scroll_repeat_ms: float | None = None,
) -> tuple[_Display, list[int], _Runner]:
    sends: list[int] = []
    # The WHEEL has its own entry point, and leaving it unpatched meant this
    # harness sent a real scroll to Windows: measured, one notch of +120
    # escaped the suite. Recorded here so it is patched by construction and a
    # test can assert on it.
    wheels: list[int] = []
    moves: list[tuple[int, int]] = []
    runner = _Runner(
        blind_reads=blind_reads,
        winks=winks,
        points=points,
        wink_after=wink_after,
        blind_after=blind_after,
    )
    display = _Display()
    gf = SimpleNamespace(
        camera=SimpleNamespace(start_sampling=lambda: None), add_subscriber=lambda fn: None
    )
    import gf_click as CK  # noqa: PLC0415
    import gf_cursor as CUR  # noqa: PLC0415
    import gf_display as GD  # noqa: PLC0415
    import gf_fit as FIT  # noqa: PLC0415
    import gf_overlay as OV  # noqa: PLC0415
    import gf_record as R  # noqa: PLC0415
    import gf_screen_check as SC  # noqa: PLC0415

    originals = {
        "build": L.build_gaze_follower,
        "runner": L.LiveRunner,
        "display": R.Display,
        "sleep": R._sleep_with_escape,
        "shutdown": R.shutdown_library,
        "visible": R.visible_point,
        "monitor": GD.pick_monitor,
        "desktop": SC.virtual_desktop,
        "to_px": SC.to_desktop_pixels,
        "dpi": SC.ensure_per_monitor_dpi_aware,
        "check": SC.check_profile_screen,
        "send": CK._send,
        "wheel": CK._send_wheel,
        "set": CUR._set_cursor_pos,
        "get": CUR._get_cursor_pos,
        "esc": OV.escape_is_down,
        "key": OV.key_is_down,
        "load": FIT.FittedModel.load,
    }

    def restore() -> None:
        L.build_gaze_follower = originals["build"]
        L.LiveRunner = originals["runner"]
        R.Display = originals["display"]
        R._sleep_with_escape = originals["sleep"]
        R.shutdown_library = originals["shutdown"]
        R.visible_point = originals["visible"]
        GD.pick_monitor = originals["monitor"]
        SC.virtual_desktop = originals["desktop"]
        SC.to_desktop_pixels = originals["to_px"]
        SC.ensure_per_monitor_dpi_aware = originals["dpi"]
        SC.check_profile_screen = originals["check"]
        CK._send = originals["send"]
        CK._send_wheel = originals["wheel"]
        CUR._set_cursor_pos = originals["set"]
        CUR._get_cursor_pos = originals["get"]
        OV.escape_is_down = originals["esc"]
        OV.key_is_down = originals["key"]
        FIT.FittedModel.load = originals["load"]

    test.addCleanup(restore)
    L.build_gaze_follower = lambda rig: gf
    L.LiveRunner = lambda *a, **kw: runner
    R.Display = lambda *a, **kw: display
    R._sleep_with_escape = lambda *a, **kw: False
    R.shutdown_library = lambda *a, **kw: None
    R.visible_point = lambda point, updated, now, stale=0.0: point
    GD.pick_monitor = lambda selector=None: MONITOR
    SC.virtual_desktop = lambda: DESKTOP
    SC.ensure_per_monitor_dpi_aware = lambda: (True, "PER_MONITOR (fake)")
    SC.check_profile_screen = lambda profile: SimpleNamespace(verified=True)
    CK._send = sends.append
    CK._send_wheel = wheels.append
    CUR._set_cursor_pos = lambda x, y: moves.append((x, y))
    CUR._get_cursor_pos = lambda: (0, 0)
    OV.escape_is_down = lambda *a, **kw: False
    OV.key_is_down = lambda vk, *a, **kw: space_down
    FIT.FittedModel.load = staticmethod(
        lambda path: SimpleNamespace(
            schema=SimpleNamespace(columns=range(4), head_names=()),
            support_activation=lambda design: [1.0],
        )
    )

    L.run_live(
        _profile(),
        move_cursor=True,
        confirmed=True,
        click_by="wink",
        start_active=start_active,
        skip_model_check=True,
        max_seconds=max_seconds,
        scroll_toggle_ms=scroll_toggle_ms,
        start_scrolling=start_scrolling,
        scroll_arm_ms=scroll_arm_ms,
        scroll_repeat_ms=scroll_repeat_ms,
    )
    # Hung on the runner rather than added to the tuple: the existing call
    # sites all unpack exactly three values.
    runner.pointer_moves = moves
    runner.wheels = wheels
    return display, sends, runner


def _last_mode_line(display: _Display) -> str:
    for hud in reversed(display.huds):
        for line in hud:
            if "PAUSED" in line or "ACTIVE" in line or "WAITING" in line:
                return line
    return ""


class PerFrameTickTests(unittest.TestCase):
    """The mode machine has to be called every frame, not only when something
    happened. Observed live: 'starting ACTIVE' in the terminal and PAUSED on
    the screen, with the face plainly in view and no way back in."""

    def test_a_face_that_arrives_late_still_ends_up_active(self) -> None:
        display, _, _ = _run(self, blind_reads=5)
        line = _last_mode_line(display)
        self.assertIn(
            "ACTIVE",
            line,
            f"it never recovered once the face appeared; the screen still said: {line!r}",
        )

    def test_it_says_waiting_rather_than_paused_while_the_face_is_gone(self) -> None:
        display, _, _ = _run(self, blind_reads=10_000)
        self.assertIn("WAITING", _last_mode_line(display))

    def test_a_face_present_from_the_first_frame_stays_active(self) -> None:
        display, _, _ = _run(self)
        self.assertIn("ACTIVE", _last_mode_line(display))

    def test_starting_paused_does_not_arm_itself(self) -> None:
        display, _, _ = _run(self, start_active=False)
        self.assertNotIn("ACTIVE", _last_mode_line(display))


class DesktopClickTests(unittest.TestCase):
    def test_a_wink_while_active_reaches_windows(self) -> None:
        _, sends, _ = _run(self, winks=[(0.0, ON_SCREEN)])
        import gf_click as CK  # noqa: PLC0415

        self.assertIn(CK.MOUSEEVENTF_LEFTDOWN, sends)
        self.assertEqual(
            sends.count(CK.MOUSEEVENTF_LEFTDOWN),
            sends.count(CK.MOUSEEVENTF_LEFTUP),
            "a press was left without its release",
        )

    def test_a_wink_while_the_face_is_gone_clicks_nothing(self) -> None:
        _, sends, _ = _run(self, blind_reads=10_000, winks=[(0.0, ON_SCREEN)])
        self.assertEqual(sends, [])

    def test_a_wink_aimed_at_nothing_clicks_nothing(self) -> None:
        _, sends, _ = _run(self, winks=[(0.0, None)])
        self.assertEqual(sends, [])

    def test_starting_paused_means_a_wink_does_not_click(self) -> None:
        _, sends, _ = _run(self, start_active=False, winks=[(0.0, ON_SCREEN)])
        self.assertEqual(sends, [])


class MachineContractTests(unittest.TestCase):
    """Stated here because the loop above is what has to honour it."""

    def test_a_machine_that_is_never_ticked_stays_suspended(self) -> None:
        machine = CTL.ToggleMachine(start=CTL.ToggleMachine.ACTIVE)
        machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertTrue(machine.suspended)
        # No further calls: exactly what the loop used to do while the face
        # was fine and no key was pressed.
        self.assertIs(machine.mode, CTL.Mode.PAUSED)

    def test_one_clean_tick_is_all_it_needs(self) -> None:
        machine = CTL.ToggleMachine(start=CTL.ToggleMachine.ACTIVE)
        machine.update(GEST.Event.NONE, tracking_ok=False)
        machine.update(GEST.Event.NONE, tracking_ok=True)
        self.assertIs(machine.mode, CTL.ToggleMachine.ACTIVE)


class NoLurchOnAWinkTests(unittest.TestCase):
    """The pointer must hold still from the first sign of a closure.

    Reported live: "the cursor wobbles, and the moment you wink it jumps".
    An eye on its way down still passes the blink gate, so a point is still
    produced -- from an eye already half behind its lid. Following it moves
    the pointer before anything has registered a closure at all, and the
    click then lands where the estimate slid to.
    """

    def _run_with_steadiness(self, steady: bool) -> list[tuple[int, int]]:
        moves: list[tuple[int, int]] = []
        runner = _Runner()
        runner_state = runner.state

        class _Half(_Runner):
            @property
            def state(self):  # noqa: ANN201
                base = super().state
                base.eyes_steady = steady
                return base

        del runner_state
        display = _Display()
        import gf_click as CK  # noqa: PLC0415
        import gf_cursor as CUR  # noqa: PLC0415
        import gf_display as GD  # noqa: PLC0415
        import gf_fit as FIT  # noqa: PLC0415
        import gf_overlay as OV  # noqa: PLC0415
        import gf_record as R  # noqa: PLC0415
        import gf_screen_check as SC  # noqa: PLC0415

        saved = (
            L.build_gaze_follower,
            L.LiveRunner,
            R.Display,
            R._sleep_with_escape,
            R.shutdown_library,
            R.visible_point,
            GD.pick_monitor,
            SC.virtual_desktop,
            SC.ensure_per_monitor_dpi_aware,
            SC.check_profile_screen,
            CK._send,
            CUR._set_cursor_pos,
            CUR._get_cursor_pos,
            OV.escape_is_down,
            OV.key_is_down,
            FIT.FittedModel.load,
        )

        def restore() -> None:
            (
                L.build_gaze_follower,
                L.LiveRunner,
                R.Display,
                R._sleep_with_escape,
                R.shutdown_library,
                R.visible_point,
                GD.pick_monitor,
                SC.virtual_desktop,
                SC.ensure_per_monitor_dpi_aware,
                SC.check_profile_screen,
                CK._send,
                CUR._set_cursor_pos,
                CUR._get_cursor_pos,
                OV.escape_is_down,
                OV.key_is_down,
                FIT.FittedModel.load,
            ) = saved

        self.addCleanup(restore)
        half = _Half()
        gf = SimpleNamespace(
            camera=SimpleNamespace(start_sampling=lambda: None), add_subscriber=lambda fn: None
        )
        L.build_gaze_follower = lambda rig: gf
        L.LiveRunner = lambda *a, **kw: half
        R.Display = lambda *a, **kw: display
        R._sleep_with_escape = lambda *a, **kw: False
        R.shutdown_library = lambda *a, **kw: None
        R.visible_point = lambda point, updated, now, stale=0.0: point
        GD.pick_monitor = lambda selector=None: MONITOR
        SC.virtual_desktop = lambda: DESKTOP
        SC.ensure_per_monitor_dpi_aware = lambda: (True, "PER_MONITOR (fake)")
        SC.check_profile_screen = lambda profile: SimpleNamespace(verified=True)
        CK._send = lambda flag: None
        CUR._set_cursor_pos = lambda x, y: moves.append((x, y))
        CUR._get_cursor_pos = lambda: (0, 0)
        OV.escape_is_down = lambda *a, **kw: False
        OV.key_is_down = lambda vk, *a, **kw: False
        FIT.FittedModel.load = staticmethod(
            lambda path: SimpleNamespace(
                schema=SimpleNamespace(columns=range(4), head_names=()),
                support_activation=lambda design: [1.0],
            )
        )
        L.run_live(
            _profile(),
            move_cursor=True,
            confirmed=True,
            click_by="wink",
            skip_model_check=True,
            max_seconds=0.3,
        )
        return moves

    def test_a_half_closed_eye_does_not_move_the_pointer(self) -> None:
        moves = self._run_with_steadiness(False)
        # The last move is the pointer being put back where it was found on
        # exit, which is required and is not the loop following a gaze.
        during_the_run = [m for m in moves if m != (0, 0)]
        self.assertEqual(
            during_the_run,
            [],
            "the pointer followed a gaze estimate made from a closing eye",
        )

    def test_open_eyes_do_move_it(self) -> None:
        """The other half: freezing on everything would also pass the first."""

        moves = self._run_with_steadiness(True)
        self.assertTrue([m for m in moves if m != (0, 0)], "it never followed the gaze at all")


class OneWinkOneDoubleTests(unittest.TestCase):
    """One wink has to send the whole double click, not half of it.

    Reported live: the pointer froze correctly, the pair still opened nothing,
    because two separately made winks cannot clear a hold, a reopening and a
    cooldown and still land inside Windows' 500 ms window.
    """

    def test_a_wink_sends_two_complete_click_pairs(self) -> None:
        import gf_click as CK  # noqa: PLC0415

        _, sends, _ = _run(self, winks=[(0.0, ON_SCREEN)])
        self.assertEqual(
            sends.count(CK.MOUSEEVENTF_LEFTDOWN),
            2,
            f"one wink sent {sends.count(CK.MOUSEEVENTF_LEFTDOWN)} press(es), not a double",
        )
        self.assertEqual(
            sends.count(CK.MOUSEEVENTF_LEFTDOWN),
            sends.count(CK.MOUSEEVENTF_LEFTUP),
            "a press was left without its release",
        )


class RatioReadoutTests(unittest.TestCase):
    """A session that recorded no winks has to be able to say why.

    Reported live: "I winked" against a run whose whole account was
    "winks detected: 0" -- which distinguishes none of the three reasons a
    wink can fail to register, and they need opposite fixes.
    """

    def _profile(self) -> PROF.Profile:
        return _profile()

    def test_it_says_so_when_there_is_nothing_to_report(self) -> None:
        self.assertEqual(L._ratio_summary([], self._profile().wink_config()), "no frames")

    def test_a_real_wink_is_counted_as_matching(self) -> None:
        text = L._ratio_summary([(1.0, 1.0), (0.5, 0.003)], self._profile().wink_config())
        self.assertIn("1 frames matched", text)

    def test_eyes_that_never_closed_match_nothing_and_say_how_far_off(self) -> None:
        text = L._ratio_summary([(0.95, 0.92), (0.9, 0.88)], self._profile().wink_config())
        self.assertIn("0 frames matched", text)
        self.assertIn("0.880", text)

    def test_a_blink_matches_nothing_and_shows_the_other_eye_came_too(self) -> None:
        """Which is a different failure from an eye that never closed."""

        text = L._ratio_summary([(1.0, 1.0), (0.05, 0.05)], self._profile().wink_config())
        self.assertIn("0 frames matched", text)
        self.assertIn("left was 0.050", text)

    def test_the_watcher_uses_the_same_rule_the_detector_does(self) -> None:
        """Reporting one rule while judging by another is worse than silence."""

        matches = L.ratio_watcher(self._profile().wink_config())
        self.assertTrue(matches(SimpleNamespace(openness_ratio=(0.5, 0.003))))
        self.assertFalse(matches(SimpleNamespace(openness_ratio=(0.05, 0.05))))
        self.assertFalse(matches(SimpleNamespace(openness_ratio=None)))
