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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_control as CTL  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
from fake_live_env import DESKTOP, MONITOR, FakeWorld  # noqa: E402

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
        # The menu and the keyboard draw their own screens rather than the
        # live view with extra lines, so a test that only watched ``huds``
        # would see nothing at all while either was open.
        self.boards: list[list[str]] = []
        self.board_labels: list[dict[str, str]] = []
        self.board_hovered: list[str | None] = []
        self.board_ready: list[bool] = []
        self.board_centres: list[list[str]] = []
        self.scans: list[list[str]] = []
        self.scan_index: list[int] = []
        self.scan_typed: list[str] = []
        self.scan_centres: list[list[str]] = []
        # Which screen was drawn, in order. Three of them exist now, so "was
        # the menu ever open" and "is it still open" are different questions
        # and a list of boards alone cannot answer the second.
        self.order: list[str] = []
        self.font_name = "fake"

    def draw_message(self, *a: object, **kw: object) -> None:
        return None

    def poll_escape(self) -> bool:
        return False

    def draw_live(  # noqa: ANN001
        self, point, raw, unfiltered, hud, *, tracking, zones=(), active_zone=None
    ) -> None:
        self.frames += 1
        self.order.append("live")
        self.huds.append(list(hud))
        self.zones.append([z.key for z in zones])
        self.active_zones.append(active_zone)

    def draw_board(  # noqa: ANN001
        self, buttons, labels, *, hovered, progress, centre=(), point=None,
        tracking=True, ready=True,
    ) -> None:
        self.frames += 1
        self.order.append("board")
        self.boards.append([b.key for b in buttons])
        self.board_labels.append(dict(labels))
        self.board_hovered.append(hovered)
        self.board_ready.append(bool(ready))
        self.board_centres.append(list(centre))

    def draw_scan(  # noqa: ANN001
        self, items, index, *, centre=(), typed="", parked=False, point=None, tracking=True
    ) -> None:
        self.frames += 1
        self.order.append("scan")
        self.scans.append(list(items))
        self.scan_index.append(index)
        self.scan_typed.append(typed)
        self.scan_centres.append(list(centre))

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
        gestures: list[Any] | None = None,
        gesture_after: int | None = None,
        wink_age_s: float = 0.0,
        wink_gap_s: float = 0.0,
    ) -> None:
        self._reads = 0
        # Seconds between one wink firing and the next, and before the first.
        # The real detector cannot fire two in the same instant -- it wants the
        # eye to open and a cooldown to pass -- and the scanning keyboard now
        # refuses a cell that has not been on screen long enough to be a
        # reaction to. Winks fired all at once were a case that cannot happen.
        self._wink_gap_s = wink_gap_s
        self._wink_armed_s: float | None = None
        self._last_wink_s: float | None = None
        self._wink_age_s = wink_age_s
        self._blind = blind_reads
        # A face that is there and then goes, which is the opposite of
        # ``blind_reads`` and the only way to test losing one mid-session.
        self._blind_after = blind_after
        self._winks = list(winks or [])
        # Where the gaze is, read by read. The last entry is held once the
        # list runs out, so a test can steer the gaze somewhere and leave it.
        self._points = list(points or [])
        self._wink_after = wink_after
        # Long closes, which is how the menu is opened now. Held back the same
        # way winks are, so a test can place one after the gaze has arrived.
        self._gestures = list(gestures or [])
        self._gesture_after = gesture_after
        self._queue: list[Any] = []
        self.errors = 0
        self.last_error = None
        self.cancels = 0

    @property
    def state(self):  # noqa: ANN201
        self._reads += 1
        present = self._reads > self._blind
        if self._blind_after is not None and self._reads > self._blind_after:
            present = False
        if self._wink_after is None or self._reads > self._wink_after:
            # The camera thread's job: fired winks land in the queue. They are
            # dropped from there by cancel_wink, exactly as the real ones are.
            #
            # Stamped with the clock NOW rather than with whatever the test
            # wrote, because that is what the real runner records -- the
            # moment the wink fired. The loop compares it against its own
            # clock to refuse stale ones, so a fixture stamped 0.0 would be
            # refused as thirty years old and every wink test would pass for
            # the wrong reason. A test that WANTS a stale wink passes
            # ``wink_age_s``.
            now_s = time.monotonic()
            if self._wink_armed_s is None:
                self._wink_armed_s = now_s
            since = self._last_wink_s if self._last_wink_s is not None else self._wink_armed_s
            while self._winks and now_s - since >= self._wink_gap_s:
                _when, aim = self._winks.pop(0)
                self._queue.append((now_s - self._wink_age_s, aim))
                self._last_wink_s = since = now_s
                if self._wink_gap_s > 0.0:
                    # One a frame at most once a gap is asked for; without this
                    # a single read would drain the whole list the moment the
                    # first gap elapsed, which is the case being removed.
                    break
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

    def on_observation(self, *a: object) -> None:
        return None

    def drain_gesture_events(self) -> list[Any]:
        """One event a frame, at most, which is what the real one produces.

        The detector fires a confirm and then waits for the eyes to open and a
        cooldown to pass, so two long closes in a single frame cannot happen.
        Handing them over together would test a case that does not exist.
        """

        if self._gesture_after is not None and self._reads <= self._gesture_after:
            return []
        if not self._gestures:
            return []
        return [self._gestures.pop(0)]

    def cancel_wink(self) -> int:
        """The real one also cancels the DETECTOR, which a fake cannot show.

        That half is tested directly against ``LiveRunner`` in
        ``test_live_wink_cancel``; what this stands in for is the queue.
        """

        dropped = len(self._queue)
        self._queue = []
        self.cancels += 1
        return dropped

    def drain_wink_events(self) -> list[Any]:
        # Held back until the loop has run far enough, so a test can place a
        # wink AFTER the gaze has arrived somewhere rather than before it.
        out, self._queue = self._queue, []
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
    start_scrolling: bool = False,
    scroll_arm_ms: float | None = None,
    scroll_repeat_ms: float | None = None,
    gestures: list[Any] | None = None,
    gesture_after: int | None = None,
    wink_age_s: float = 0.0,
    menu_enabled: bool = True,
    menu_dwell_ms: float = 900.0,
    scan_ms: float | None = None,
    scan_settle_ms: float | None = None,
    wink_gap_s: float = 0.0,
    foreground: Any = None,
    events: list[str] | None = None,
    display_error: BaseException | None = None,
    escape_during_warmup: bool = False,
    loop_error: BaseException | None = None,
) -> tuple[_Display, list[int], _Runner]:
    runner = _Runner(
        blind_reads=blind_reads,
        winks=winks,
        points=points,
        wink_after=wink_after,
        blind_after=blind_after,
        gestures=gestures,
        gesture_after=gesture_after,
        wink_age_s=wink_age_s,
        wink_gap_s=wink_gap_s,
    )
    display = _Display()
    # Injected, not patched: every sender, the screen, the keyboard state and
    # the library come from this one fake world (ARCH-01 stage C). The wheel
    # and keyboard senders once escaped a patch-based harness and reached
    # Windows; an injected environment has no second path to forget.
    world = FakeWorld(
        runner=runner,
        display=display,
        key_down=lambda vk: space_down,
        # A stable window handle unless a test says otherwise, so the adapter's
        # target lock has something to lock ON to.
        foreground=foreground or (lambda: 4242),
        title=lambda hwnd: "",
        warm_up_escape=escape_during_warmup,
        display_error=display_error,
        events=events,
        # Returned by the fake shutdown and printed by the live view, so a test
        # can place the library shutdown in stdout relative to the report.
        shutdown_result="LIBRARY-SHUT" if events is not None else None,
    )
    if events is not None:
        display.close = lambda: events.append("display close")
    if loop_error is not None:
        def failing_draw(*a: object, **kw: object) -> None:
            raise loop_error

        display.draw_live = failing_draw

    try:
        L.run_live(
            _profile(),
            move_cursor=True,
            confirmed=True,
            click_by="wink",
            start_active=start_active,
            skip_model_check=True,
            max_seconds=max_seconds,
            start_scrolling=start_scrolling,
            scroll_arm_ms=scroll_arm_ms,
            scroll_repeat_ms=scroll_repeat_ms,
            menu_enabled=menu_enabled,
            menu_dwell_ms=menu_dwell_ms,
            scan_ms=scan_ms,
            scan_settle_ms=scan_settle_ms,
            env=world.environment(),
        )
    finally:
        # Hung on the runner rather than added to the tuple: the existing call
        # sites all unpack exactly three values.
        runner.subscribers = world.subscribers
        runner.pointer_moves = world.moves
        runner.wheels = world.wheels
        runner.keystrokes = world.keystrokes
        runner.world = world
    # Canary: the fakes were what the session used.
    test.assertEqual(world.calls["open_library"], 1, "the fake library was not the one opened")
    return display, world.sends, runner


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
        world = FakeWorld(runner=_Half(), display=_Display(), title=lambda hwnd: "")
        L.run_live(
            _profile(),
            move_cursor=True,
            confirmed=True,
            click_by="wink",
            skip_model_check=True,
            max_seconds=0.3,
            env=world.environment(),
        )
        self.assertEqual(world.calls["make_runner"], 1, "the fake runner was not used")
        moves.extend(world.moves)
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
