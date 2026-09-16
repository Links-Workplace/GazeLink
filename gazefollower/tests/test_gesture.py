"""Intentional-gesture state machine: no camera, no model, no OS input.

These assert the safety properties, not that the functions return something:
ordinary blinking must never act, a lost face must never accumulate toward a
confirm, and a confirm must never land on a destructive option by default.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_gesture as G  # noqa: E402

HZ = 30.0
STEP = 1.0 / HZ


class _Feeder:
    """Drives a detector frame by frame and collects what it emitted."""

    def __init__(self, detector: G.EyeCloseDetector) -> None:
        self.detector = detector
        self.t = 0.0
        self.events: list[G.Event] = []

    def run(self, spec: str, face: str | None = None) -> list[G.Event]:
        face = face or "+" * len(spec)
        for i, ch in enumerate(spec):
            self.t += STEP
            event = self.detector.update(self.t, face_present=face[i] != "-", eyes_shut=ch == "x")
            if event is not G.Event.NONE:
                self.events.append(event)
        return self.events


def _frames(ms: float) -> int:
    return max(1, int(round(ms / 1000.0 * HZ)))


class ConfigTests(unittest.TestCase):
    def test_a_confirm_at_or_below_the_blink_ceiling_is_refused(self) -> None:
        """Otherwise ordinary blinking would confirm, which is the single
        failure this whole design exists to prevent."""

        with self.assertRaises(ValueError):
            G.GestureConfig(natural_blink_max_ms=800.0, confirm_ms=800.0)
        with self.assertRaises(ValueError):
            G.GestureConfig(natural_blink_max_ms=900.0, confirm_ms=800.0)

    def test_the_defaults_sit_between_the_measured_populations(self) -> None:
        """Measured on this rig: longest natural blink 297 ms, shortest
        deliberate hold 1172 ms. The defaults must fall inside that gap."""

        cfg = G.GestureConfig()
        self.assertGreater(cfg.natural_blink_max_ms, 297.0)
        self.assertLess(cfg.confirm_ms, 1172.0)
        self.assertGreater(cfg.confirm_ms, cfg.natural_blink_max_ms)


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = G.EyeCloseDetector()
        self.feed = _Feeder(self.detector)

    def test_ordinary_blinking_emits_nothing_at_all(self) -> None:
        """Measured natural blinks ran 31-297 ms. None may act."""

        blink = "x" * _frames(200.0)
        gap = "." * _frames(600.0)
        self.assertEqual(self.feed.run((blink + gap) * 6), [])

    def test_a_long_deliberate_hold_confirms(self) -> None:
        self.feed.run("." * 5 + "x" * _frames(1200.0) + "." * 10)
        self.assertIn(G.Event.CONFIRM, self.feed.events)

    def test_a_medium_close_cycles_rather_than_confirming(self) -> None:
        self.feed.run("." * 5 + "x" * _frames(550.0) + "." * 20)
        self.assertEqual(self.feed.events, [G.Event.CYCLE])

    def test_a_flicker_mid_hold_does_not_break_the_hold(self) -> None:
        """The measured signal reopens briefly mid-closure; without bridging
        a two-second hold arrived as fragments and never confirmed."""

        held = "x" * _frames(400.0) + "." + "x" * _frames(400.0) + "." + "x" * _frames(400.0)
        self.feed.run("." * 5 + held + "." * 10)
        self.assertIn(G.Event.CONFIRM, self.feed.events)

    def test_a_real_reopening_is_not_bridged(self) -> None:
        """Two separate short closes must stay two, not fuse into a confirm."""

        short = "x" * _frames(500.0)
        self.feed.run("." * 5 + short + "." * _frames(900.0) + short + "." * 10)
        self.assertNotIn(G.Event.CONFIRM, self.feed.events)

    def test_losing_the_face_mid_hold_aborts_instead_of_confirming(self) -> None:
        """Looking away or leaving must not keep accumulating. Absence is not
        evidence that the eyes are shut."""

        spec = "." * 5 + "x" * _frames(2000.0)
        face = "+" * 5 + "+" * _frames(300.0) + "-" * (_frames(2000.0) - _frames(300.0))
        self.assertEqual(self.feed.run(spec, face=face), [])

    def test_a_hold_does_not_confirm_twice_while_still_shut(self) -> None:
        """Holding for five seconds is one intention, not six."""

        self.feed.run("." * 5 + "x" * _frames(5000.0) + "." * 10)
        self.assertEqual(self.feed.events.count(G.Event.CONFIRM), 1)

    def test_the_cooldown_blocks_an_immediate_second_event(self) -> None:
        cfg = G.GestureConfig(cooldown_ms=1000.0)
        feed = _Feeder(G.EyeCloseDetector(cfg))
        hold = "x" * _frames(900.0)
        feed.run("." * 3 + hold + "." * 2 + hold + "." * 5)
        self.assertEqual(feed.events.count(G.Event.CONFIRM), 1)

    def test_confirm_fires_while_the_eyes_are_still_shut(self) -> None:
        """Waiting for the eyes to open would leave no way to tell a confirm
        from a miss until after the fact."""

        detector = G.EyeCloseDetector()
        t = 0.0
        fired_while_shut = False
        for _ in range(_frames(1500.0)):
            t += STEP
            if detector.update(t, face_present=True, eyes_shut=True) is G.Event.CONFIRM:
                fired_while_shut = True
                break
        self.assertTrue(fired_while_shut)


class MenuTests(unittest.TestCase):
    def _menu(self) -> G.RecoveryMenu:
        return G.RecoveryMenu(
            [
                G.MenuOption("dismiss", "Keep watching"),
                G.MenuOption("recalibrate", "Recalibrate for this screen"),
            ]
        )

    def test_a_confirm_out_of_nowhere_selects_nothing(self) -> None:
        """The menu must be opened deliberately before anything can be taken,
        so a single stray confirm cannot act."""

        menu = self._menu()
        self.assertIsNone(menu.handle(G.Event.CONFIRM, 1.0))
        self.assertTrue(menu.open)

    def test_the_harmless_option_is_the_one_selected_first(self) -> None:
        """A gesture on an imperfect signal will sometimes fire when it should
        not; what it does by default must therefore cost nothing."""

        menu = self._menu()
        menu.handle(G.Event.CYCLE, 1.0)
        self.assertEqual(menu.selected.key, "dismiss")
        taken = menu.handle(G.Event.CONFIRM, 2.0)
        self.assertIsNotNone(taken)
        self.assertEqual(taken.key, "dismiss")

    def test_cycling_reaches_the_other_option_and_wraps(self) -> None:
        menu = self._menu()
        menu.handle(G.Event.CYCLE, 1.0)  # opens
        menu.handle(G.Event.CYCLE, 2.0)
        self.assertEqual(menu.selected.key, "recalibrate")
        menu.handle(G.Event.CYCLE, 3.0)
        self.assertEqual(menu.selected.key, "dismiss")

    def test_an_idle_menu_closes_itself(self) -> None:
        """An accidental opening must not sit there waiting to be confirmed
        by the next stray blink."""

        menu = self._menu()
        menu.handle(G.Event.CYCLE, 1.0)
        self.assertTrue(menu.open)
        menu.handle(G.Event.NONE, 1.0 + menu.timeout_s + 0.1)
        self.assertFalse(menu.open)

    def test_taking_an_option_closes_the_menu(self) -> None:
        menu = self._menu()
        menu.handle(G.Event.CYCLE, 1.0)
        menu.handle(G.Event.CONFIRM, 2.0)
        self.assertFalse(menu.open)
        self.assertEqual(menu.index, 0)


if __name__ == "__main__":
    unittest.main()


class RightWinkTests(unittest.TestCase):
    """A right wink, told apart from a blink and from the menu's gesture.

    These thresholds are the only ones in this module that were NOT measured
    on a person. The tests fix the BEHAVIOUR -- what must and must not fire --
    so that replacing the numbers after a probe run cannot quietly change it.
    """

    def _run(self, detector, frames, *, start=0.0, step=0.016):
        """Feed (left_ratio, right_ratio) frames; return how many times it fired.

        16 ms a frame, because the measured run came in at 63 fps.
        """

        fired = 0
        now = start
        for left, right in frames:
            if detector.update(now, face_present=True, left_ratio=left, right_ratio=right):
                fired += 1
            now += step
        return fired

    def _wink(self, n):
        """Measured: right to 0.003 of baseline, left narrowing to about 0.5."""

        return [(0.5, 0.003)] * n

    def _open(self, n):
        return [(1.0, 1.0)] * n

    def _blink(self, n):
        """Both eyes together, which is what makes it not a wink."""

        return [(0.05, 0.05)] * n

    def test_a_held_right_wink_fires_once(self) -> None:
        detector = G.RightWinkDetector()
        self.assertEqual(self._run(detector, self._open(3) + self._wink(25) + self._open(5)), 1)

    def test_a_short_wink_is_ignored(self) -> None:
        """Below the hold time it is an asymmetry, not an instruction.

        Counted in CAMERA frames, which is what the detector actually sees:
        the hold is 50 ms and the first matching frame only starts the clock,
        so two frames at 30 fps is under it and three is over.
        """

        detector = G.RightWinkDetector()
        frames = self._open(3) + self._wink(2) + self._open(5)
        self.assertEqual(self._run(detector, frames, step=1 / 30), 0)

    def test_three_camera_frames_is_enough(self) -> None:
        """The other side of the same line, so the number cannot drift up
        without this failing: measured winks were three frames long."""

        detector = G.RightWinkDetector()
        frames = self._open(3) + self._wink(3) + self._open(5)
        self.assertEqual(self._run(detector, frames, step=1 / 30), 1)

    def test_a_two_eyed_blink_never_winks(self) -> None:
        detector = G.RightWinkDetector()
        self.assertEqual(self._run(detector, self._open(3) + self._blink(40) + self._open(5)), 0)

    def test_the_tail_of_a_blink_does_not_fire(self) -> None:
        """Ordinary blinks reopen unevenly, so for a FEW FRAMES one eye is
        further shut than the other. That asymmetry is real, which is why the
        hold exists: a tail lasts two or three frames and a wink does not.

        The earlier version of this test appended 25 asymmetric frames -- 400
        ms -- and called them a tail. That is a wink, and the detector was
        right to fire on it; the test was wrong.
        """

        detector = G.RightWinkDetector()
        tail = [(0.5, 0.05)] * 3  # ~48 ms of one eye trailing the other
        frames = self._open(3) + self._blink(10) + tail + self._open(10)
        self.assertEqual(self._run(detector, frames), 0)

    def test_a_deliberate_wink_right_after_a_blink_still_counts(self) -> None:
        """The old veto blocked this for 600 ms, and the operator blinks a lot."""

        detector = G.RightWinkDetector()
        frames = self._open(3) + self._blink(10) + self._wink(20) + self._open(5)
        self.assertEqual(self._run(detector, frames), 1)

    def test_a_wink_well_after_a_blink_still_works(self) -> None:
        detector = G.RightWinkDetector()
        frames = self._blink(10) + self._open(40) + self._wink(20) + self._open(5)
        self.assertEqual(self._run(detector, frames), 1)

    def test_a_wink_the_measured_length_is_long_enough(self) -> None:
        """Measured: 112 wink-like frames over 100 s at 63 fps, so a single
        wink runs about 150 ms. A hold that cannot be reached fires never."""

        detector = G.RightWinkDetector()
        frames = self._open(5) + self._wink(10) + self._open(5)  # 160 ms held
        self.assertEqual(self._run(detector, frames), 1)

    def test_a_left_eye_that_narrows_along_with_the_right_is_still_a_wink(self) -> None:
        """Measured: the left eye was under the gate on 658 frames against the
        right's 331. Requiring it OPEN required something the person cannot do."""

        detector = G.RightWinkDetector()
        frames = self._open(3) + [(0.30, 0.05)] * 15 + self._open(5)
        self.assertEqual(self._run(detector, frames), 1)

    def test_holding_the_wink_does_not_fire_again(self) -> None:
        detector = G.RightWinkDetector()
        self.assertEqual(self._run(detector, self._open(3) + self._wink(120)), 1)

    def test_a_left_wink_is_not_a_right_wink(self) -> None:
        detector = G.RightWinkDetector()
        frames = self._open(3) + [(0.003, 0.5)] * 30 + self._open(5)
        self.assertEqual(self._run(detector, frames), 0)

    def test_losing_the_face_mid_wink_fires_nothing(self) -> None:
        detector = G.RightWinkDetector()
        now = 0.0
        for _ in range(10):
            detector.update(now, face_present=True, left_ratio=0.5, right_ratio=0.003)
            now += 0.016
        fired = detector.update(now, face_present=False, left_ratio=0.5, right_ratio=0.003)
        self.assertFalse(fired)

    def test_two_winks_in_a_row_need_the_eye_to_reopen(self) -> None:
        detector = G.RightWinkDetector()
        frames = self._open(3) + self._wink(25) + self._open(40) + self._wink(25) + self._open(3)
        self.assertEqual(self._run(detector, frames), 2)

    def test_the_config_refuses_nonsense(self) -> None:
        for bad in ({"hold_ms": 0.0}, {"asymmetry": 1.0}, {"shut_ratio": 0.0}):
            with self.assertRaises(ValueError):
                G.WinkConfig(**bad)


class OpennessGateTests(unittest.TestCase):
    """Openness is polygon AREA, so a fixed threshold means different things
    on different days. Measured live 9.9: 627 frames, minima 27.0 and 64.5
    against an absolute threshold of 10.0 -- nothing ever reached it."""

    def test_a_closure_the_absolute_threshold_misses_is_seen(self) -> None:
        gate = G.OpennessGate()
        for _ in range(30):
            gate.is_open(191.0)
        self.assertFalse(
            gate.is_open(27.0), "27 against a baseline of 191 is a closure and was read as open"
        )

    def test_the_same_number_is_open_for_someone_sitting_closer(self) -> None:
        """27 px^2 is a shut eye at one distance and an open one at another."""

        gate = G.OpennessGate()
        for _ in range(30):
            gate.is_open(30.0)
        self.assertTrue(gate.is_open(27.0))

    def test_a_long_closure_cannot_drag_the_baseline_down(self) -> None:
        """Otherwise a held wink slowly becomes an open eye and releases."""

        gate = G.OpennessGate()
        for _ in range(30):
            gate.is_open(200.0)
        before = gate.baseline
        for _ in range(120):
            self.assertFalse(gate.is_open(20.0))
        self.assertEqual(gate.baseline, before)

    def test_opening_wider_moves_the_baseline_at_once(self) -> None:
        """Sitting closer must not read as an eye that never opens."""

        gate = G.OpennessGate()
        gate.is_open(100.0)
        self.assertTrue(gate.is_open(400.0))
        self.assertEqual(gate.baseline, 400.0)

    def test_a_drift_upward_is_followed(self) -> None:
        gate = G.OpennessGate()
        for value in range(100, 200):
            self.assertTrue(gate.is_open(float(value)))
        self.assertGreater(gate.baseline, 190.0)

    def test_nonsense_is_shut_not_open(self) -> None:
        gate = G.OpennessGate()
        gate.is_open(150.0)
        for bad in (0.0, -1.0, float("nan")):
            self.assertFalse(gate.is_open(bad))

    def test_the_config_refuses_nonsense(self) -> None:
        for bad in ({"shut_fraction": 0.0}, {"shut_fraction": 1.0}, {"adapt": 0.0}):
            with self.assertRaises(ValueError):
                G.OpennessGateConfig(**bad)


class MirrorTests(unittest.TestCase):
    """The camera faces the person, so the image's left eye is their right one.

    Measured: the operator winked their RIGHT eye for 100 s and the library's
    LEFT field was under the gate on 658 frames against the right's 331. Every
    gesture in this project had been watching the wrong eye.
    """

    def test_the_library_left_is_the_person_right(self) -> None:
        person_left, person_right = G.eyes_as_the_person_has_them(12.0, 200.0)
        self.assertEqual(person_right, 12.0)
        self.assertEqual(person_left, 200.0)

    def test_translating_twice_gets_back_where_it_started(self) -> None:
        once = G.eyes_as_the_person_has_them(7.0, 9.0)
        self.assertEqual(G.eyes_as_the_person_has_them(*once), (7.0, 9.0))

    def test_a_right_wink_in_the_persons_frame_is_a_wink(self) -> None:
        """The end-to-end statement, in the only frame that matters."""

        detector = G.RightWinkDetector()
        # The library reports a nearly shut LEFT field and an open right one.
        person_left, person_right = G.eyes_as_the_person_has_them(0.03, 0.5)
        self.assertTrue(detector.looks_like_a_wink(person_left, person_right))

    def test_the_same_numbers_unmirrored_are_not_a_wink(self) -> None:
        """Which is exactly what the detector saw before this existed."""

        detector = G.RightWinkDetector()
        self.assertFalse(detector.looks_like_a_wink(0.03, 0.5))


class SteadyBandTests(unittest.TestCase):
    """Between shut and steady is the half-closed band.

    The blink gate still passes there, so a gaze point is produced from an eye
    already going behind its own lid. Reported live: "the cursor wobbles, and
    the moment you wink it jumps" -- long before anything registered a closure.
    """

    def test_steady_sits_above_shut(self) -> None:
        config = G.OpennessGateConfig()
        self.assertGreater(config.steady_fraction, config.shut_fraction)

    def test_a_half_closed_eye_is_neither_shut_nor_steady(self) -> None:
        config = G.OpennessGateConfig()
        midway = (config.shut_fraction + config.steady_fraction) / 2.0
        self.assertGreater(midway, config.shut_fraction, "it would read as shut")
        self.assertLess(midway, config.steady_fraction, "it would read as steady")

    def test_the_two_cannot_be_configured_the_wrong_way_round(self) -> None:
        """Steady under shut would make the band empty and the wobble return."""

        with self.assertRaises(ValueError):
            G.OpennessGateConfig(shut_fraction=0.8, steady_fraction=0.5)

    def test_steady_cannot_exceed_a_fully_open_eye(self) -> None:
        with self.assertRaises(ValueError):
            G.OpennessGateConfig(steady_fraction=1.5)

    def test_a_wide_open_eye_is_steady(self) -> None:
        self.assertGreaterEqual(1.0, G.OpennessGateConfig().steady_fraction)


class DoubleClickReachableTests(unittest.TestCase):
    """Two deliberate winks have to be able to land inside Windows' own
    double-click window, or nothing they aim at will ever open.

    Reported live: the pointer froze correctly and the folder still did not
    open. The cooldown was 700 ms and starts only after the eye reopens, so
    the soonest a second wink could fire was about 900 ms after the first --
    against a system window of 500 ms. The pair was two clicks, every time.
    """

    # What a reopening costs, measured at the 63 fps the live runs report:
    # three or four frames from the deepest point back to open.
    REOPEN_MS = 60.0

    def _soonest_second_wink_ms(self, config: G.WinkConfig) -> float:
        return self.REOPEN_MS + config.cooldown_ms + config.hold_ms

    def test_two_winks_fit_inside_the_system_double_click_window(self) -> None:
        import gf_click as CK  # noqa: PLC0415

        soonest = self._soonest_second_wink_ms(G.WinkConfig())
        self.assertLess(
            soonest,
            CK.system_double_click_ms(),
            "a second wink cannot arrive in time, so a double click is unreachable",
        )

    def test_there_is_headroom_rather_than_a_photo_finish(self) -> None:
        """A pair that only just fits is one the person cannot reliably make."""

        import gf_click as CK  # noqa: PLC0415

        headroom = CK.system_double_click_ms() - self._soonest_second_wink_ms(G.WinkConfig())
        self.assertGreater(headroom, 100.0)

    def test_the_old_cooldown_would_have_failed_this(self) -> None:
        """Pinned so the number cannot drift back without the reason showing."""

        import gf_click as CK  # noqa: PLC0415

        old = G.WinkConfig(cooldown_ms=700.0)
        self.assertGreater(self._soonest_second_wink_ms(old), CK.system_double_click_ms())

    def test_one_closure_still_cannot_fire_twice(self) -> None:
        """The short cooldown must not cost the guard it was standing in for:
        the eye reopening is what actually ends a wink."""

        detector = G.RightWinkDetector()
        fired = 0
        now = 0.0
        for _ in range(120):  # nearly two seconds of one held wink
            if detector.update(now, face_present=True, left_ratio=0.5, right_ratio=0.003):
                fired += 1
            now += 0.016
        self.assertEqual(fired, 1)


class BridgeTests(unittest.TestCase):
    """A noisy frame is not the eye opening.

    Measured over a two-minute session: 103 frames matched the wink rule and
    only TWO winks fired. The runs kept being broken by a frame that popped
    out of the rule and starting again from zero, so most winks never reached
    their own hold time. The eye-close detector has had a bridge since it was
    written; this one did not.
    """

    def _run(self, frames, step: float = 1 / 30) -> int:
        detector = G.RightWinkDetector()
        fired = 0
        now = 0.0
        for left, right in frames:
            if detector.update(now, face_present=True, left_ratio=left, right_ratio=right):
                fired += 1
            now += step
        return fired

    WINK = (0.5, 0.003)
    OPEN = (1.0, 1.0)

    def test_a_wink_broken_by_one_noisy_frame_still_fires(self) -> None:
        """Two frames, a dropped one, two more: under the hold on either side
        of the gap, and over it once the gap is bridged."""

        frames = [self.OPEN] * 3 + [self.WINK] * 2 + [self.OPEN] + [self.WINK] * 2
        self.assertEqual(self._run(frames + [self.OPEN] * 5, step=1 / 30), 1)

    def test_the_same_wink_unbridged_would_not_have(self) -> None:
        """Without the bridge each half is under the hold time on its own."""

        detector = G.RightWinkDetector(G.WinkConfig(bridge_ms=0.0))
        fired = 0
        now = 0.0
        frames = [self.OPEN] * 3 + [self.WINK] * 2 + [self.OPEN] + [self.WINK] * 2
        for left, right in frames + [self.OPEN] * 5:
            if detector.update(now, face_present=True, left_ratio=left, right_ratio=right):
                fired += 1
            now += 1 / 30
        self.assertEqual(fired, 0)

    def test_a_real_reopening_still_ends_the_wink(self) -> None:
        """The bridge must not run two winks together into one.

        Both halves are long enough to fire on their own here, so the count
        says whether they stayed separate. An earlier version made each half
        too short and asserted zero, which passed for the wrong reason.
        """

        frames = [self.OPEN] * 3 + [self.WINK] * 3 + [self.OPEN] * 7 + [self.WINK] * 3
        self.assertEqual(self._run(frames + [self.OPEN] * 5, step=1 / 30), 2)

    def test_a_gap_longer_than_the_bridge_does_not_carry_the_hold(self) -> None:
        """Two sub-threshold winks with a real gap must add up to nothing."""

        frames = [self.OPEN] * 3 + [self.WINK] * 2 + [self.OPEN] * 5 + [self.WINK] * 2
        self.assertEqual(self._run(frames + [self.OPEN] * 5, step=1 / 30), 0)

    def test_a_blink_is_still_not_bridged_into_a_wink(self) -> None:
        frames = [self.OPEN] * 3 + [(0.05, 0.05)] * 12 + [self.OPEN] * 5
        self.assertEqual(self._run(frames), 0)

    def test_an_absurdly_long_bridge_is_refused(self) -> None:
        """A long bridge merges two separate winks into one."""

        with self.assertRaises(ValueError):
            G.WinkConfig(bridge_ms=500.0)

    def test_the_hold_can_be_tuned_below_the_bridge(self) -> None:
        """The bound used to be ``hold_ms``, which made the hold un-tunable
        downwards: asking for 35 ms against the default 40 ms bridge raised an
        error and the session would not start. Reported live, twice."""

        config = G.WinkConfig(hold_ms=35.0)
        self.assertEqual(config.hold_ms, 35.0)
        self.assertGreater(config.bridge_ms, config.hold_ms)

    def test_the_bridge_still_covers_a_dropped_camera_frame(self) -> None:
        """Which is the one thing it is for, so it cannot follow the hold down."""

        self.assertGreater(G.WinkConfig().bridge_ms, 1000.0 / 30.0)

    def test_no_bridge_at_all_is_allowed(self) -> None:
        self.assertEqual(G.WinkConfig(bridge_ms=0.0).bridge_ms, 0.0)

    def test_one_held_wink_still_fires_exactly_once(self) -> None:
        self.assertEqual(self._run([self.OPEN] * 3 + [self.WINK] * 120), 1)


class CancelTests(unittest.TestCase):
    """A wink that is part-way through when the mode changes.

    Dropping the QUEUE of fired winks -- which is what the scroll edge did --
    covers only half of this. At the moment a mode changes the eye may already
    be closed with the hold accumulating: nothing has fired, so there is
    nothing in any queue to drop, and the wink lands a moment later in the new
    mode from a closure begun in the old one.

    Clearing the accumulation alone is not enough either. The eye is still
    shut, so the very next frame starts a fresh hold from the SAME closure and
    one long close is counted twice, once on each side of the change. This
    project shipped that exact bug once, with a latch that cleared when the
    rule stopped matching instead of when the eye opened: one 2.7 second
    closure fired four times.
    """

    WINK = (0.5, 0.003)
    OPEN = (1.0, 1.0)

    def _feed(self, detector, frames, *, start=0.0, step=1 / 30):
        fired = 0
        now = start
        for left, right in frames:
            if detector.update(now, face_present=True, left_ratio=left, right_ratio=right):
                fired += 1
            now += step
        return fired, now

    def test_a_hold_in_progress_does_not_fire_after_a_cancel(self) -> None:
        detector = G.RightWinkDetector()
        # One frame short of the hold: the closure has started and not fired.
        fired, now = self._feed(detector, [self.OPEN, self.WINK])
        self.assertEqual(fired, 0, "the fixture fired before the cancel, so this proves nothing")
        detector.cancel()
        fired, _now = self._feed(detector, [self.WINK] * 60, start=now)
        self.assertEqual(fired, 0, "a closure begun before the change fired after it")

    def test_a_closure_that_already_fired_does_not_fire_again(self) -> None:
        """The other end of the same closure. Once is the whole point."""

        detector = G.RightWinkDetector()
        fired, now = self._feed(detector, [self.OPEN] + [self.WINK] * 4)
        self.assertEqual(fired, 1, "the fixture never fired, so this proves nothing")
        detector.cancel()
        again, _now = self._feed(detector, [self.WINK] * 90, start=now)
        self.assertEqual(again, 0, "one closure was counted twice")

    def test_a_new_wink_needs_the_eye_to_open_first(self) -> None:
        detector = G.RightWinkDetector()
        # Two frames: the closure has STARTED and has not reached the hold, so
        # what follows is about the cancel and not about a wink that already
        # fired and latched on its own.
        fired, now = self._feed(detector, [self.OPEN, self.WINK])
        self.assertEqual(fired, 0, "the fixture fired before the cancel")
        detector.cancel()
        # Still shut: nothing.
        fired, now = self._feed(detector, [self.WINK] * 30, start=now)
        self.assertEqual(fired, 0)
        # Opened, waited out the cooldown, and winked again: that one counts.
        fired, now = self._feed(detector, [self.OPEN] * 30, start=now)
        self.assertEqual(fired, 0)
        fired, _now = self._feed(detector, [self.WINK] * 10, start=now)
        self.assertEqual(fired, 1, "the way back to winking was closed off")

    def test_cancelling_when_nothing_is_in_flight_is_harmless(self) -> None:
        detector = G.RightWinkDetector()
        detector.cancel()
        fired, now = self._feed(detector, [self.OPEN] * 30)
        self.assertEqual(fired, 0)
        fired, _now = self._feed(detector, [self.WINK] * 10, start=now)
        self.assertEqual(fired, 1, "a cancel with nothing to cancel killed the next wink")
