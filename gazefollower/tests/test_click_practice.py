"""The click practice window, driven with fakes. No camera, no OS input.

The point of these tests is the WIRING. Every piece here was already tested on
its own -- the dwell engine fires once per entry, the control machine only
arms in one mode, the click adapter refuses when not armed -- and the project
has repeatedly found that the pieces were right and the assembly was not.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_click_practice as P  # noqa: E402
import gf_dwell as D  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_profile as PROF  # noqa: E402

MONITOR = SimpleNamespace(origin=(0, 0), width_px=5120, height_px=1440, name="FAKE")
DESKTOP = {"x": 0, "y": 0, "width": 5120, "height": 1440}


def _profile() -> PROF.Profile:
    return PROF.Profile(
        name="fake",
        model_dir="models/none",
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
    """Headless: records what it was asked to draw and counts frames."""

    def __init__(self, *a: object, **kw: object) -> None:
        self.frames = 0
        self.prompts: list[list[str]] = []
        self.keys: set[str] = set()

    def draw_message(self, *a: object, **kw: object) -> None:
        return None

    def wait_for_key(self, *a: object, **kw: object) -> str:
        return "ok"

    def poll_escape(self) -> bool:
        return False

    def poll_keys(self) -> set[str]:
        """Delivers the scripted keys once, then nothing."""

        out, self.keys = self.keys, set()
        return out

    def draw_practice(self, *a: object, **kw: object) -> None:
        self.frames += 1
        self.prompts.append(list(kw.get("prompt", [])))

    def close(self) -> None:
        return None


class _Runner:
    """Serves a scripted gaze point and scripted queues of gestures."""

    def __init__(
        self,
        point: tuple[float, float] | None,
        events: list[Any],
        winks: list[Any] | None = None,
        face_present: bool = True,
        face_leaves_after: int | None = None,
        wink_after: int | None = None,
    ) -> None:
        self._wink_after = wink_after
        self._point = point
        self._face_present = face_present
        # Losing the face partway is the case that matters: the system has to
        # be ARMED first for there to be anything to cancel.
        self._face_leaves_after = face_leaves_after
        self._reads = 0
        self._events = events
        self._winks = list(winks or [])

    @property
    def state(self) -> SimpleNamespace:
        """Always fresh: the loop discards a state older than the stale limit."""

        import time as _time  # noqa: PLC0415

        self._reads += 1
        present = self._face_present
        if self._face_leaves_after is not None and self._reads > self._face_leaves_after:
            present = False
        return SimpleNamespace(
            point=self._point if present else None,
            face_present=present,
            openness=(120.0, 110.0),
            openness_ratio=(1.0, 1.0),
            updated_s=_time.monotonic(),
        )

    def on_frame(self, *a: object) -> None:
        return None

    def drain_gesture_events(self) -> list[Any]:
        out, self._events = self._events, []
        return out

    def drain_wink_events(self) -> list[Any]:
        # Held back until the loop has run far enough, so a test can place a
        # wink AFTER the face has gone rather than before it.
        if self._wink_after is not None and self._reads <= self._wink_after:
            return []
        out, self._winks = self._winks, []
        return out


class _Harness:
    """Patches the module's world so the real loop can be run."""

    def __init__(self, test: unittest.TestCase, runner: _Runner, *, sends: list[int]) -> None:
        self.sent = sends
        originals = {
            "build": P.L.build_gaze_follower,
            "runner": P.L.LiveRunner,
            "display": P.R.Display,
            "sleep": P.R._sleep_with_escape,
            "shutdown": P.R.shutdown_library,
            "visible": P.R.visible_point,
            "dpi": P.SC.ensure_per_monitor_dpi_aware,
            "check": P.SC.check_profile_screen,
            "desktop": P.SC.virtual_desktop,
            "monitor": P.GD.pick_monitor,
            "sender": P.CK._send,
            "set_pointer": P.CUR._set_cursor_pos,
            "get_pointer": P.CUR._get_cursor_pos,
        }
        # The pointer was never faked here, so every test in this file moved
        # the REAL Windows pointer. Found by the real-input interlock
        # (ARCH-01): the move now raises unless it is faked, so it is.
        self.pointer_moves: list[tuple[int, int]] = []
        P.CUR._set_cursor_pos = lambda x, y: self.pointer_moves.append((x, y))
        P.CUR._get_cursor_pos = lambda: (0, 0)
        self.display = _Display()
        gf = SimpleNamespace(
            camera=SimpleNamespace(start_sampling=lambda: None),
            add_subscriber=lambda fn: None,
        )
        P.L.build_gaze_follower = lambda rig: gf
        P.L.LiveRunner = lambda *a, **kw: runner
        P.R.Display = lambda *a, **kw: self.display
        P.R._sleep_with_escape = lambda *a, **kw: False
        P.R.shutdown_library = lambda *a, **kw: None
        P.R.visible_point = lambda point, updated, now, stale=0.0: point
        P.SC.ensure_per_monitor_dpi_aware = lambda: (True, "PER_MONITOR (fake)")
        P.SC.check_profile_screen = lambda profile: SimpleNamespace(verified=True)
        P.SC.virtual_desktop = lambda: DESKTOP
        P.GD.pick_monitor = lambda selector=None: MONITOR
        P.CK._send = sends.append
        test.addCleanup(self._restore, originals)

    @staticmethod
    def _restore(originals: dict[str, Any]) -> None:
        P.L.build_gaze_follower = originals["build"]
        P.L.LiveRunner = originals["runner"]
        P.R.Display = originals["display"]
        P.R._sleep_with_escape = originals["sleep"]
        P.R.shutdown_library = originals["shutdown"]
        P.R.visible_point = originals["visible"]
        P.SC.ensure_per_monitor_dpi_aware = originals["dpi"]
        P.SC.check_profile_screen = originals["check"]
        P.SC.virtual_desktop = originals["desktop"]
        P.GD.pick_monitor = originals["monitor"]
        P.CK._send = originals["sender"]
        P.CUR._set_cursor_pos = originals["set_pointer"]
        P.CUR._get_cursor_pos = originals["get_pointer"]


def _run(
    test: unittest.TestCase,
    *,
    point: tuple[float, float] | None,
    events: list[Any],
    winks: list[Any] | None = None,
    click_by: str = "dwell",
    # The existing tests script eyelid CONFIRM events, so they keep exercising
    # that route. The product default is "gaze", covered by its own tests.
    toggle_by: str = "eyes",
    # Short so the run is short. The production value is asserted separately,
    # because a test that also fixed the duration would pass with any value.
    toggle_ms: float = 1.0,
    keys: set[str] | None = None,
    click: bool = True,
    face_present: bool = True,
    face_leaves_after: int | None = None,
    wink_after: int | None = None,
    tmp: Path | None = None,
) -> tuple[dict[str, Any], list[int]]:
    sends: list[int] = []
    runner = _Runner(
        point,
        events,
        winks,
        face_present=face_present,
        face_leaves_after=face_leaves_after,
        wink_after=wink_after,
    )
    harness = _Harness(test, runner, sends=sends)
    harness.display.keys = set(keys or ())
    original_load = None
    import gf_fit as FIT  # noqa: PLC0415

    original_load = FIT.FittedModel.load
    FIT.FittedModel.load = staticmethod(lambda path: object())  # type: ignore[method-assign]
    test.addCleanup(lambda: setattr(FIT.FittedModel, "load", original_load))
    out = (tmp or Path(test.id().replace(".", "_"))).with_suffix(".json")
    report = P.run_click_practice(
        _profile(),
        dwell_ms=1.0,  # the ring completes almost at once, so a rest fires within the run
        click_by=click_by,
        toggle_by=toggle_by,
        toggle_ms=toggle_ms,
        click=click,
        confirmed=click,
        max_seconds=0.6,
        headless=True,
        out=out,
    )
    test.addCleanup(lambda: out.unlink(missing_ok=True))
    return report, sends


class GateTests(unittest.TestCase):
    def test_click_without_the_confirmation_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            P.run_click_practice(_profile(), click=True, confirmed=False)
        self.assertIn("--i-mean-it", str(caught.exception))

    def test_an_unverified_screen_refuses_to_click(self) -> None:
        runner = _Runner((0.38, 0.5), [])
        _Harness(self, runner, sends=[])
        P.SC.check_profile_screen = lambda profile: SimpleNamespace(verified=False)
        P.SC.format_report = lambda report: "fake report"
        with self.assertRaises(SystemExit) as caught:
            P.run_click_practice(_profile(), click=True, confirmed=True)
        self.assertIn("unverified ruler", str(caught.exception))


class ArmingWiringTests(unittest.TestCase):
    """Every piece is tested alone; this is the assembly."""

    def test_resting_on_a_button_while_paused_never_reaches_windows(self) -> None:
        report, sends = _run(self, point=(0.38, 0.5), events=[])
        self.assertEqual(report["tally"]["clicks"], 0)
        self.assertEqual(sends, [], "a click reached Windows while the system was paused")
        self.assertGreater(
            report["tally"]["suppressed_while_unarmed"],
            0,
            "the dwell engine never fired, so this run proves nothing about arming",
        )

    def test_the_same_rest_clicks_once_the_user_arms_selection(self) -> None:
        """Two cycles then a confirm walks paused -> move-only -> move-and-select."""

        report, sends = _run(self, point=(0.38, 0.5), events=[(0.0, GEST.Event.CONFIRM)])
        self.assertGreater(report["tally"]["clicks"], 0, "arming produced no click at all")
        self.assertIn(CK_DOWN := P.CK.MOUSEEVENTF_LEFTDOWN, sends)
        self.assertIn(P.CK.MOUSEEVENTF_LEFTUP, sends)
        self.assertEqual(
            sends.count(CK_DOWN),
            sends.count(P.CK.MOUSEEVENTF_LEFTUP),
            "a press was left without its release",
        )

    def test_simulation_counts_the_click_and_calls_nothing(self) -> None:
        report, sends = _run(
            self, point=(0.38, 0.5), events=[(0.0, GEST.Event.CONFIRM)], click=False
        )
        self.assertGreater(report["tally"]["clicks"], 0)
        self.assertEqual(sends, [], "simulation reached Windows")
        self.assertEqual(report["os_input"], "none (simulated)")


class LossTests(unittest.TestCase):
    def test_losing_the_face_pauses_and_cancels_instead_of_clicking(self) -> None:
        report, sends = _run(
            self,
            point=ON_LEFT,
            events=[(0.0, GEST.Event.CONFIRM)],
            face_leaves_after=3,
            wink_after=6,  # the wink arrives after the face has gone
            click_by="wink",
            winks=[(0.0, ON_LEFT)],
        )
        self.assertEqual(report["tally"]["clicks"], 0)
        self.assertEqual(sends, [])
        self.assertGreater(
            report["tally"]["cancelled_selections"],
            0,
            "the face went and nothing was cancelled",
        )
        modes = [entry["mode"] for entry in report["tally"]["mode_changes"]]
        self.assertEqual(modes[-1], "paused", f"it did not fall back to paused: {modes}")


class LandingTests(unittest.TestCase):
    """A selection that clicks outside the thing it selected looks like it
    worked, which makes it the worst failure available here."""

    def setUp(self) -> None:
        self.button = D.LAYOUTS["a"]()[0]

    def test_a_pointer_already_inside_is_not_nudged(self) -> None:
        pointer = P.button_centre_px(self.button, MONITOR, DESKTOP)
        moved = (pointer[0] + 5, pointer[1] + 5)
        self.assertEqual(P.where_to_click(self.button, moved, MONITOR, DESKTOP), moved)

    def test_a_pointer_outside_lands_on_the_selected_button(self) -> None:
        outside = (MONITOR.width_px - 1, 0)
        landing = P.where_to_click(self.button, outside, MONITOR, DESKTOP)
        self.assertTrue(
            P.inside(self.button, landing, MONITOR, DESKTOP),
            "the click landed outside the button that was selected",
        )

    def test_no_pointer_at_all_still_lands_on_the_button(self) -> None:
        landing = P.where_to_click(self.button, None, MONITOR, DESKTOP)
        self.assertTrue(P.inside(self.button, landing, MONITOR, DESKTOP))

    def test_every_layout_button_can_be_landed_on(self) -> None:
        for name in sorted(D.LAYOUTS):
            for button in D.LAYOUTS[name]():
                landing = P.where_to_click(button, None, MONITOR, DESKTOP)
                self.assertTrue(
                    P.inside(button, landing, MONITOR, DESKTOP),
                    f"layout {name}, button {button.key}",
                )


class SeparationTests(unittest.TestCase):
    def test_the_eyelids_switch_modes_and_never_click(self) -> None:
        """One gesture that both changed mode and clicked would fire two
        things at once and neither would be attributable."""

        source = (Path(__file__).resolve().parent.parent / "gf_click_practice.py").read_text(
            encoding="utf-8"
        )
        body = source.split("fired = engine.update", 1)
        self.assertEqual(len(body), 2, "the dwell call moved; this test needs updating")
        before = body[0]
        self.assertNotIn("clicker.click", before, "something clicks before the dwell engine fires")


class LandingRecordTests(unittest.TestCase):
    def test_every_click_the_loop_made_is_recorded_inside_its_button(self) -> None:
        """ "It selected LEFT" and "it clicked inside LEFT" are two claims.

        Only the second is checked here, and only for the pointer positions
        this harness produces. A pointer lagging OUTSIDE the target at the
        instant the ring completes is the case that would exercise the recentre
        path, and it is NOT covered: the unit tests of where_to_click are what
        stand behind that.
        """

        report, _ = _run(self, point=(0.38, 0.5), events=[(0.0, GEST.Event.CONFIRM)])
        points = report["tally"]["click_points"]
        self.assertTrue(points, "no click was recorded, so this test proves nothing")
        by_key = {b.key: b for b in D.LAYOUTS["a"]()}
        for entry in points:
            button = by_key[str(entry["key"])]
            self.assertTrue(
                P.inside(button, (int(entry["x"]), int(entry["y"])), MONITOR, DESKTOP),
                f"a click for {entry['key']} landed at ({entry['x']}, {entry['y']}), outside it",
            )


# One long close is the whole sequence now. Kept as a name so that if the
# gesture changes again, every test that arms the system changes with it.
ARM = [(0.0, GEST.Event.CONFIRM)]
ON_LEFT = (0.38, 0.5)


class WinkModeTests(unittest.TestCase):
    """Dwell and wink are alternative click modes, never both at once.

    Two live mechanisms would make a double activation impossible to
    attribute, which is the thing the plan asks to avoid before either is
    trusted.
    """

    def test_a_wink_on_a_button_clicks_it_when_armed(self) -> None:
        # A LEFT wink is the ordinary left click.
        report, sends = _run(
            self,
            point=ON_LEFT,
            events=ARM,
            winks=[(0.0, ON_LEFT, P.GEST.Eye.LEFT)],
            click_by="wink",
        )
        self.assertEqual(report["tally"]["per_button"].get("LEFT"), 1)
        self.assertEqual(sends.count(P.CK.MOUSEEVENTF_LEFTDOWN), 1)
        self.assertEqual(report["tally"]["winks_by_eye"], {"left": 1})

    def test_a_right_wink_sends_a_right_click_not_a_left_one(self) -> None:
        # The eye picks the button. The left wink was never measured on this
        # person, so this practice is where the two are first told apart.
        report, sends = _run(
            self,
            point=ON_LEFT,
            events=ARM,
            winks=[(0.0, ON_LEFT, P.GEST.Eye.RIGHT)],
            click_by="wink",
        )
        self.assertEqual(sends.count(P.CK.MOUSEEVENTF_LEFTDOWN), 0)
        self.assertEqual(sends.count(P.CK.MOUSEEVENTF_RIGHTDOWN), 1)
        self.assertEqual(
            sends.count(P.CK.MOUSEEVENTF_RIGHTDOWN), sends.count(P.CK.MOUSEEVENTF_RIGHTUP)
        )
        self.assertEqual(report["tally"]["click_points"][0]["eye"], "right")

    def test_a_wink_while_paused_never_reaches_windows(self) -> None:
        report, sends = _run(
            self, point=ON_LEFT, events=[], winks=[(0.0, ON_LEFT)], click_by="wink"
        )
        self.assertEqual(report["tally"]["clicks"], 0)
        self.assertEqual(sends, [])
        self.assertGreater(report["tally"]["suppressed_while_unarmed"], 0)

    def test_resting_on_a_button_does_not_click_in_wink_mode(self) -> None:
        """Otherwise a dwell and a wink could both fire on one intention."""

        report, _ = _run(self, point=ON_LEFT, events=ARM, winks=[], click_by="wink")
        self.assertEqual(
            report["tally"]["clicks"], 0, "the dwell engine clicked while wink was the click mode"
        )

    def test_a_wink_does_not_click_in_dwell_mode(self) -> None:
        report, _ = _run(self, point=None, events=ARM, winks=[(0.0, ON_LEFT)], click_by="dwell")
        self.assertEqual(report["tally"]["clicks"], 0)

    def test_a_wink_aimed_at_nothing_is_counted_and_not_clicked(self) -> None:
        """Off-target is a different failure from not firing, and is kept apart."""

        report, sends = _run(
            self, point=ON_LEFT, events=ARM, winks=[(0.0, (0.5, 0.5))], click_by="wink"
        )
        self.assertEqual(report["tally"]["clicks"], 0)
        self.assertEqual(report["tally"]["winks_on_no_button"], 1)
        self.assertEqual(sends, [])

    def test_a_wink_with_no_remembered_point_clicks_nothing(self) -> None:
        """The gaze is invalid while an eye is shut; with nothing latched
        before it, there is no place the wink can honestly mean."""

        report, sends = _run(self, point=ON_LEFT, events=ARM, winks=[(0.0, None)], click_by="wink")
        self.assertEqual(report["tally"]["clicks"], 0)
        self.assertEqual(report["tally"]["winks_on_no_button"], 1)
        self.assertEqual(sends, [])

    def test_an_unknown_click_mode_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            P.run_click_practice(_profile(), click_by="blink")
        self.assertIn("dwell", str(caught.exception))


class NoRingInWinkModeTests(unittest.TestCase):
    """A filling ring says resting will select. In wink mode it will not."""

    def test_the_dwell_engine_is_not_stepped_when_the_wink_clicks(self) -> None:
        """Not merely ignored: an engine that runs can fire a second click."""

        stepped = {"n": 0}
        real_update = D.DwellEngine.update

        def counting(self, *a: object, **kw: object):  # noqa: ANN001, ANN202
            stepped["n"] += 1
            return real_update(self, *a, **kw)

        D.DwellEngine.update = counting  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(D.DwellEngine, "update", real_update))
        _run(self, point=ON_LEFT, events=ARM, winks=[], click_by="wink")
        self.assertEqual(stepped["n"], 0, "the dwell engine ran while the wink was the click mode")

    def test_the_dwell_engine_is_stepped_when_dwell_clicks(self) -> None:
        """The other half: this test is worthless if the engine never runs."""

        stepped = {"n": 0}
        real_update = D.DwellEngine.update

        def counting(self, *a: object, **kw: object):  # noqa: ANN001, ANN202
            stepped["n"] += 1
            return real_update(self, *a, **kw)

        D.DwellEngine.update = counting  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(D.DwellEngine, "update", real_update))
        _run(self, point=ON_LEFT, events=ARM, click_by="dwell")
        self.assertGreater(stepped["n"], 0)


class SimplifiedInterfaceTests(unittest.TestCase):
    def test_the_screen_says_only_paused_or_active(self) -> None:
        """The operator asked for one line, not a menu with a highlight."""

        sends: list[int] = []
        runner = _Runner(ON_LEFT, list(ARM), [])
        harness = _Harness(self, runner, sends=sends)
        import gf_fit as FIT  # noqa: PLC0415

        real_load = FIT.FittedModel.load
        FIT.FittedModel.load = staticmethod(lambda path: object())  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(FIT.FittedModel, "load", real_load))
        out = Path(self.id().replace(".", "_")).with_suffix(".json")
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        P.run_click_practice(
            _profile(), dwell_ms=1.0, click_by="wink", click=False, max_seconds=0.4, out=out
        )
        drawn = [line for lines in harness.display.prompts for line in lines]
        self.assertTrue(drawn, "nothing was drawn, so this test proves nothing")
        self.assertTrue(
            any(line in ("PAUSED", "ACTIVE - wink to click") for line in drawn),
            f"the mode line is not one of the two agreed strings: {set(drawn)}",
        )
        self.assertFalse(
            any("highlight" in line.lower() or "chooses" in line.lower() for line in drawn),
            "the menu wording is still on screen",
        )


class ConfirmWhileTheEyesAreShutTests(unittest.TestCase):
    """Reported live: a long close switched to ACTIVE and straight back.

    The confirm fires WHILE the eyes are still shut, by design, so the person
    learns it took effect before they open them. The eyes being shut is also
    exactly when there is no gaze point. Driving the mode interrupt from the
    point therefore paused the system in the very same frame the confirm
    armed it, every time, and the mode flickered instead of changing.
    """

    def test_a_confirm_with_no_gaze_point_still_activates(self) -> None:
        report, _ = _run(
            self,
            point=None,  # the eyes are shut, so there is no point
            events=[(0.0, GEST.Event.CONFIRM)],
            face_present=True,  # but the face has not gone anywhere
            click_by="wink",
        )
        modes = [entry["mode"] for entry in report["tally"]["mode_changes"]]
        self.assertIn(
            str(K_ACTIVE := "move-and-select"),
            modes,
            "the confirm never activated at all",
        )
        self.assertEqual(
            modes[-1],
            K_ACTIVE,
            f"the mode was flipped back within the run: {modes}",
        )

    def test_a_face_that_really_goes_away_still_pauses(self) -> None:
        """Narrowing the interrupt must not remove it."""

        report, sends = _run(
            self,
            point=None,
            events=[(0.0, GEST.Event.CONFIRM)],
            face_present=False,
            click_by="wink",
            winks=[(0.0, ON_LEFT)],
        )
        modes = [entry["mode"] for entry in report["tally"]["mode_changes"]]
        self.assertNotIn("move-and-select", modes, "it armed while the face was gone")
        self.assertEqual(report["tally"]["clicks"], 0)
        self.assertEqual(sends, [])


ON_MODE = (0.5, 0.5)


class ToggleRouteTests(unittest.TestCase):
    """The eyelid signal was measured not to reach the blink threshold on
    three of five recent sessions, so the mode needs a route that does not
    depend on it. Gaze is the signal known to work; the key is for testing."""

    def test_resting_on_the_mode_panel_activates(self) -> None:
        report, _ = _run(self, point=ON_MODE, events=[], toggle_by="gaze", click_by="wink")
        modes = [entry["mode"] for entry in report["tally"]["mode_changes"]]
        self.assertIn("move-and-select", modes, "resting on the mode panel did nothing")

    def test_resting_on_a_click_button_does_not_change_mode(self) -> None:
        """The click targets and the mode control must not select each other."""

        report, _ = _run(self, point=ON_LEFT, events=[], toggle_by="gaze", click_by="wink")
        self.assertEqual(report["tally"]["mode_changes"], [])

    def test_space_toggles_when_the_key_route_is_chosen(self) -> None:
        report, _ = _run(
            self, point=ON_LEFT, events=[], keys={"space"}, toggle_by="key", click_by="wink"
        )
        modes = [entry["mode"] for entry in report["tally"]["mode_changes"]]
        self.assertIn("move-and-select", modes)

    def test_space_does_nothing_on_the_gaze_route(self) -> None:
        report, _ = _run(
            self, point=ON_LEFT, events=[], keys={"space"}, toggle_by="gaze", click_by="wink"
        )
        self.assertEqual(report["tally"]["mode_changes"], [])

    def test_an_eyelid_confirm_does_nothing_on_the_gaze_route(self) -> None:
        """Otherwise two controls are live and a change cannot be attributed."""

        report, _ = _run(
            self,
            point=ON_LEFT,
            events=[(0.0, GEST.Event.CONFIRM)],
            toggle_by="gaze",
            click_by="wink",
        )
        self.assertEqual(report["tally"]["mode_changes"], [])

    def test_the_mode_panel_sits_between_the_click_targets_and_touches_neither(self) -> None:
        buttons = D.LAYOUTS["a"]()
        panel = P.toggle_button(buttons)
        left, right = sorted(buttons, key=lambda b: b.x0)
        self.assertEqual(panel.x0, left.x1)
        self.assertEqual(panel.x1, right.x0)
        for button in buttons:
            self.assertFalse(
                P.inside(button, P.button_centre_px(panel, MONITOR, DESKTOP), MONITOR, DESKTOP),
                f"the mode panel overlaps {button.key}",
            )

    def test_the_mode_dwell_is_far_longer_than_a_click_dwell(self) -> None:
        """A glance on the way between the two buttons must not switch mode."""

        self.assertGreaterEqual(P.TOGGLE_DWELL_MS, 1500.0)

    def test_an_unknown_toggle_route_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            P.run_click_practice(_profile(), toggle_by="nod")
        self.assertIn("gaze", str(caught.exception))


class ProfileRuleTests(unittest.TestCase):
    """The rule must come from the profile, or it belongs to whoever last
    edited the defaults rather than to the person in front of the camera."""

    def _with_rule(self, **wink: float) -> PROF.Profile:
        base = _profile()
        return type(base)(**{**base.__dict__, "gesture": {"wink": wink}})

    def test_the_report_names_where_the_rule_came_from(self) -> None:
        sends: list[int] = []
        runner = _Runner(ON_LEFT, [], [])
        harness = _Harness(self, runner, sends=sends)
        import gf_fit as FIT  # noqa: PLC0415

        real = FIT.FittedModel.load
        FIT.FittedModel.load = staticmethod(lambda path: object())  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(FIT.FittedModel, "load", real))
        out = Path(self.id().replace(".", "_")).with_suffix(".json")
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        report = P.run_click_practice(
            self._with_rule(shut_ratio=0.31, hold_ms=190.0),
            dwell_ms=1.0,
            click_by="wink",
            toggle_by="eyes",
            click=False,
            max_seconds=0.3,
            out=out,
        )
        self.assertEqual(report["wink_rule"]["source"], "profile")
        self.assertEqual(report["wink_rule"]["shut_ratio"], 0.31)
        self.assertEqual(report["wink_rule"]["hold_ms"], 190.0)
        del harness

    def test_a_profile_rule_actually_changes_what_counts_as_a_wink(self) -> None:
        """Reporting the number while judging by another one is worse than
        not reporting it: the run would look explained and not be."""

        import gf_gesture as GEST  # noqa: PLC0415

        strict = GEST.RightWinkDetector(self._with_rule(shut_ratio=0.05).wink_config())
        loose = GEST.RightWinkDetector(self._with_rule(shut_ratio=0.60).wink_config())
        self.assertFalse(strict.looks_like_a_wink(0.9, 0.30))
        self.assertTrue(loose.looks_like_a_wink(0.9, 0.30))


class ProfileReachesTheDetectorTests(unittest.TestCase):
    """Reading the rule and then not handing it to the thing that decides is
    the same failure as not reading it, and it looks explained in the report."""

    def test_the_runner_is_built_with_the_profiles_wink_and_gate(self) -> None:
        seen: dict[str, Any] = {}
        runner = _Runner(ON_LEFT, [], [])
        harness = _Harness(self, runner, sends=[])
        P.L.LiveRunner = lambda *a, **kw: (seen.update(kw), runner)[1]
        import gf_fit as FIT  # noqa: PLC0415

        real = FIT.FittedModel.load
        FIT.FittedModel.load = staticmethod(lambda path: object())  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(FIT.FittedModel, "load", real))
        base = _profile()
        profile = type(base)(**{**base.__dict__, "gesture": {"wink": {"shut_ratio": 0.31}}})
        out = Path(self.id().replace(".", "_")).with_suffix(".json")
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        P.run_click_practice(
            profile, dwell_ms=1.0, click_by="wink", click=False, max_seconds=0.2, out=out
        )
        self.assertIn("wink", seen, "the runner was built without a wink rule at all")
        self.assertEqual(
            seen["wink"].shut_ratio,
            0.31,
            "the profile's rule was read and then not given to the detector",
        )
        self.assertIn("gate", seen)
        del harness
