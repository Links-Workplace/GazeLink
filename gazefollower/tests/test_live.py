"""Free-running live view: no camera, no model fitting, no OS input."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as P  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
SETTINGS = GF.FilterSettings(
    width_px=5120,
    height_px=1440,
    kind=GF.FilterKind.ONE_EURO,
    one_euro_min_cutoff_hz=0.4,
    one_euro_beta_hz_per_px_s=0.0,
)
OPEN = C.BLINK_THRESHOLD + 5.0
SHUT = 0.0


class _Model:
    """A model that answers with a fixed point, or explodes on demand."""

    def __init__(self, point=(0.5, 0.5), raises: bool = False) -> None:
        self.schema = SimpleNamespace(head_names=(), columns=range(4))
        self._point = point
        self._raises = raises
        self.calls = 0

    def predict_norm(self, design, rig):
        self.calls += 1
        if self._raises:
            raise RuntimeError("model blew up")
        return np.array([self._point], dtype=np.float64)


def _frame(openness: float = OPEN, status: bool = True):
    face = SimpleNamespace(left_eye_openness=openness, right_eye_openness=openness)
    gaze = SimpleNamespace(
        status=status,
        features=np.zeros(4, dtype=np.float32),
        raw_gaze_coordinates=None,
    )
    return face, gaze


class LiveRunnerTests(unittest.TestCase):
    def _runner(self, model=None, settings=SETTINGS):
        clock = {"t": 0.0}

        def tick():
            clock["t"] += 1.0 / 30.0
            return clock["t"]

        return L.LiveRunner(
            model or _Model(), None, RIG, settings, clock=tick, head_builder=lambda f: None
        )

    def test_an_open_eyed_frame_produces_a_point(self) -> None:
        runner = self._runner()
        runner.on_frame(*_frame())
        self.assertIsNotNone(runner.state.point)
        self.assertTrue(runner.state.tracking)
        self.assertIsNotNone(runner.state.updated_s)

    def test_a_blink_produces_no_point_and_is_not_reported_as_tracking(self) -> None:
        """Predicting through a blink puts the dot where the eye is not."""

        runner = self._runner()
        runner.on_frame(*_frame())
        runner.on_frame(*_frame(openness=SHUT))
        self.assertIsNone(runner.state.point)
        self.assertFalse(runner.state.tracking)

    def test_a_blink_clears_the_filter_so_the_point_cannot_resume_stale(self) -> None:
        """Without the reset the filter would carry pre-blink history across
        the gap and the point would slide out of a blink from where it was,
        rather than from where the eye now is."""

        runner = self._runner()
        for _ in range(10):
            runner.on_frame(*_frame())
        settled = runner.state.point
        self.assertIsNotNone(settled)
        runner.on_frame(*_frame(openness=SHUT))
        self.assertIsNone(runner.state.point)
        # A far-away model output right after the blink must be followed
        # immediately, not damped toward the pre-blink position.
        runner.model = _Model(point=(0.9, 0.9))
        runner.on_frame(*_frame())
        self.assertAlmostEqual(runner.state.point[0], 0.9, places=6)

    def test_a_frame_with_no_gaze_status_yields_nothing(self) -> None:
        runner = self._runner()
        runner.on_frame(*_frame(status=False))
        self.assertIsNone(runner.state.point)
        self.assertFalse(runner.state.tracking)

    def test_a_model_that_raises_never_reaches_the_camera_thread(self) -> None:
        """on_frame runs on the library's thread; an exception there kills
        tracking for the whole session."""

        runner = self._runner(model=_Model(raises=True))
        runner.on_frame(*_frame())  # must not raise
        self.assertIsNone(runner.state.point)

    def test_a_broken_frame_never_reaches_the_camera_thread(self) -> None:
        runner = self._runner()
        runner.on_frame(None, None)  # must not raise
        self.assertGreaterEqual(runner.errors, 0)

    def test_without_a_filter_the_raw_point_is_shown_unchanged(self) -> None:
        runner = self._runner(settings=None)
        runner.on_frame(*_frame())
        self.assertEqual(runner.state.point, (0.5, 0.5))

    def test_the_frame_window_does_not_grow_without_bound(self) -> None:
        """This view is meant to be left running; an unbounded list is a leak."""

        runner = self._runner()
        for _ in range(500):
            runner.on_frame(*_frame())
        self.assertLessEqual(len(runner._frame_times), 60)
        self.assertEqual(runner.state.frames, 500)


class RigCheckTests(unittest.TestCase):
    def _profile(self) -> P.Profile:
        return P.Profile(
            name="baseline",
            model_dir="somewhere",
            rig=RIG.to_dict(),
            filter=P.filter_to_dict(SETTINGS),
        )

    def test_a_matching_rig_is_accepted(self) -> None:
        L.check_rig(self._profile(), RIG, allow_mismatch=False)  # must not raise

    def test_a_different_screen_is_refused(self) -> None:
        """A profile run against another display does not degrade gracefully:
        it produces a confidently wrong point with nothing on screen saying so."""

        other = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 1920, 1080)
        with self.assertRaises(SystemExit) as caught:
            L.check_rig(self._profile(), other, allow_mismatch=False)
        self.assertIn("device_w_px", str(caught.exception))

    def test_the_refusal_can_be_overridden_deliberately(self) -> None:
        other = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 1920, 1080)
        L.check_rig(self._profile(), other, allow_mismatch=True)  # warns, does not raise


if __name__ == "__main__":
    unittest.main()


def _asym(person_left: float, person_right: float, status: bool = True):
    """A frame, described by the PERSON's eyes rather than the image's.

    The camera faces them, so the library's ``left_eye_openness`` is the eye on
    the left of the IMAGE, which is the person's RIGHT. Writing these fixtures
    in the library's frame is how a right-wink detector came to watch the left
    eye, so they are written in the person's frame and swapped here, once.

    ``status`` on the FACE is what gf_live reads as "a face is present", and
    the gesture detectors reset without it. The older fixture omits it, which
    is why no earlier test in this file ever produced a gesture event.
    """

    face = SimpleNamespace(
        left_eye_openness=person_right, right_eye_openness=person_left, status=True
    )
    gaze = SimpleNamespace(
        status=status, features=np.zeros(4, dtype=np.float32), raw_gaze_coordinates=None
    )
    return face, gaze


class WinkSeparationTests(unittest.TestCase):
    """A wink must not also drive the mode menu.

    ``eyes_shut`` was written as ``not (left > T and right > T)``, which is
    true whenever EITHER eye is shut -- so a one-eyed wink fed the menu
    detector as well, and a wink-to-click would have fired two mechanisms from
    one gesture with no way to attribute either.
    """

    def _runner(self):
        clock = {"t": 0.0}

        def tick():
            clock["t"] += 1.0 / 30.0
            return clock["t"]

        return L.LiveRunner(_Model(), None, RIG, SETTINGS, clock=tick, head_builder=lambda f: None)

    def test_a_long_right_wink_never_reaches_the_menu(self) -> None:
        runner = self._runner()
        for _ in range(60):
            runner.on_frame(*_asym(OPEN, SHUT))
        self.assertEqual(runner.drain_gesture_events(), [], "a wink produced a menu event as well")

    def test_a_long_right_wink_produces_a_wink_event(self) -> None:
        runner = self._runner()
        runner.on_frame(*_asym(OPEN, OPEN))
        for _ in range(60):
            runner.on_frame(*_asym(OPEN, SHUT))
        self.assertTrue(runner.drain_wink_events(), "the wink was never reported")

    def test_closing_both_eyes_still_reaches_the_menu(self) -> None:
        """Narrowing the menu's signal must not remove the menu."""

        runner = self._runner()
        for _ in range(60):
            runner.on_frame(*_asym(SHUT, SHUT))
        self.assertTrue(runner.drain_gesture_events(), "both eyes shut no longer opens the menu")

    def test_the_wink_carries_the_point_from_before_the_eye_shut(self) -> None:
        """The gaze is invalid while an eye is shut, so a wink that carried the
        live point would be aimed at nothing."""

        runner = self._runner()
        runner.on_frame(*_asym(OPEN, OPEN))
        seen = runner.state.point
        self.assertIsNotNone(seen)
        for _ in range(60):
            runner.on_frame(*_asym(OPEN, SHUT))
        events = runner.drain_wink_events()
        self.assertTrue(events)
        self.assertEqual(events[0][1], seen)


class RelativeGateWiringTests(unittest.TestCase):
    """Measured live: openness ran a median near 190 and never fell below 27,
    while the absolute threshold is 10. Every eyelid gesture was impossible."""

    def _runner(self):
        clock = {"t": 0.0}

        def tick():
            clock["t"] += 1.0 / 30.0
            return clock["t"]

        return L.LiveRunner(_Model(), None, RIG, SETTINGS, clock=tick, head_builder=lambda f: None)

    def test_a_closure_that_never_reaches_the_absolute_threshold_still_confirms(self) -> None:
        runner = self._runner()
        for _ in range(30):
            runner.on_frame(*_asym(191.0, 160.0))
        runner.drain_gesture_events()
        for _ in range(60):
            runner.on_frame(*_asym(27.0, 24.0))
        self.assertTrue(
            runner.drain_gesture_events(),
            "a closure to 14% of baseline produced nothing, so no gesture can ever fire",
        )

    def test_a_right_wink_that_never_reaches_it_is_still_a_wink(self) -> None:
        runner = self._runner()
        for _ in range(30):
            runner.on_frame(*_asym(191.0, 160.0))
        for _ in range(60):
            runner.on_frame(*_asym(191.0, 24.0))
        self.assertTrue(runner.drain_wink_events(), "the wink was invisible to the gate")

    def test_a_wink_whose_left_eye_narrows_too_is_still_a_wink(self) -> None:
        """Measured on the operator: the left eye was under the gate on 658
        frames against the right's 331. It closes with the right, every time."""

        runner = self._runner()
        for _ in range(30):
            runner.on_frame(*_asym(191.0, 160.0))
        runner.drain_wink_events()
        for _ in range(60):
            # left to ~30% of baseline, right to ~5%: not open, but far apart
            runner.on_frame(*_asym(57.0, 8.0))
        self.assertTrue(
            runner.drain_wink_events(),
            "a wink was rejected because the left eye narrowed along with the right",
        )

    def test_a_symmetric_blink_is_not_a_wink(self) -> None:
        runner = self._runner()
        for _ in range(30):
            runner.on_frame(*_asym(191.0, 160.0))
        runner.drain_wink_events()
        for _ in range(60):
            runner.on_frame(*_asym(9.5, 8.0))  # both to ~5%
        self.assertEqual(runner.drain_wink_events(), [], "a blink was read as a wink")

    def test_ordinary_open_eyes_are_not_read_as_shut(self) -> None:
        runner = self._runner()
        for _ in range(90):
            runner.on_frame(*_asym(191.0, 160.0))
        self.assertEqual(runner.drain_gesture_events(), [])
        self.assertEqual(runner.drain_wink_events(), [])


class DesktopClickGateTests(unittest.TestCase):
    """Clicking on the DESKTOP is not clicking in a practice window.

    There the only action was a counter the window drew on itself. Here a
    click can close, delete or send, and cannot be taken back.
    """

    def _profile(self):
        import gf_profile as PROF  # noqa: PLC0415

        return PROF.Profile(
            name="t",
            model_dir="m",
            rig={
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            filter={"kind": "one-euro"},
        )

    def test_winking_to_click_without_moving_the_cursor_is_refused(self) -> None:
        """A click that lands wherever the pointer was last left is not a
        click at what you are looking at."""

        with self.assertRaises(SystemExit) as caught:
            L.run_live(self._profile(), click_by="wink", move_cursor=False, confirmed=True)
        self.assertIn("--move-cursor", str(caught.exception))

    def test_an_unknown_click_mode_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            L.run_live(self._profile(), click_by="blink")
        self.assertIn("off", str(caught.exception))

    def test_moving_the_cursor_still_needs_its_own_confirmation(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            L.run_live(self._profile(), click_by="wink", move_cursor=True, confirmed=False)
        self.assertIn("--i-mean-it", str(caught.exception))

    def test_the_cli_carries_the_click_mode_through(self) -> None:
        args = L.build_parser().parse_args(["--click-by", "wink"])
        self.assertEqual(args.click_by, "wink")
        self.assertEqual(L.build_parser().parse_args([]).click_by, "off")


class FacePresenceHelperTests(unittest.TestCase):
    """Closing the eyes ends the gaze point and not the face. A caller that
    cannot tell them apart pauses in the same frame the close armed it."""

    def _runner(self, *, face: bool, age_s: float):
        return SimpleNamespace(
            state=SimpleNamespace(face_present=face, updated_s=time.monotonic() - age_s)
        )

    def test_a_present_face_on_a_fresh_frame_is_present(self) -> None:
        self.assertTrue(L.state_face_ok(self._runner(face=True, age_s=0.0)))

    def test_a_stale_frame_is_not_a_face_however_recently_it_said_so(self) -> None:
        """A dead camera leaves the last answer standing for ever."""

        self.assertFalse(L.state_face_ok(self._runner(face=True, age_s=5.0)))

    def test_a_runner_that_has_never_seen_a_frame_is_not_a_face(self) -> None:
        runner = SimpleNamespace(state=SimpleNamespace(face_present=False, updated_s=None))
        self.assertFalse(L.state_face_ok(runner))


class StartActiveTests(unittest.TestCase):
    """Opening ready to click, at the operator's explicit request.

    The paused start left no way in on a desktop: the practice window's toggle
    was a gaze panel and there is nowhere to draw one over someone else's
    application. The two OS gates still stand in front of this and Esc still
    stops it; what is given up is the beat before the first click is possible.
    """

    def test_the_default_is_active(self) -> None:
        self.assertFalse(L.build_parser().parse_args([]).start_paused)

    def test_paused_can_still_be_asked_for(self) -> None:
        self.assertTrue(L.build_parser().parse_args(["--start-paused"]).start_paused)

    def test_an_active_start_is_armed_and_moving(self) -> None:
        import gf_control as CTL  # noqa: PLC0415

        machine = CTL.ToggleMachine(start=CTL.ToggleMachine.ACTIVE)
        self.assertTrue(machine.mode.selection_armed)
        self.assertTrue(machine.mode.cursor_enabled)

    def test_it_can_still_be_paused_once_running(self) -> None:
        """Starting armed must not cost the way out of being armed."""

        import gf_control as CTL  # noqa: PLC0415
        import gf_gesture as GEST  # noqa: PLC0415

        machine = CTL.ToggleMachine(start=CTL.ToggleMachine.ACTIVE)
        transition = machine.update(GEST.Event.CONFIRM)
        self.assertIs(transition.mode, CTL.Mode.PAUSED)
        self.assertTrue(transition.cancel_selection)

    def test_losing_the_face_still_pauses_an_active_start(self) -> None:
        import gf_control as CTL  # noqa: PLC0415
        import gf_gesture as GEST  # noqa: PLC0415

        machine = CTL.ToggleMachine(start=CTL.ToggleMachine.ACTIVE)
        transition = machine.update(GEST.Event.NONE, tracking_ok=False)
        self.assertIs(transition.mode, CTL.Mode.PAUSED)
