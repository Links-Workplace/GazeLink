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
