"""The click adapter. Every test uses a fake Windows: none emits a real click."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_click as CK  # noqa: E402


class _Fake:
    """Records the flags it was asked to send; can be told to fail."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.sent: list[int] = []
        self.fail_on = fail_on

    def __call__(self, flag: int) -> None:
        if self.fail_on is not None and flag == self.fail_on:
            raise OSError("SendInput refused")
        self.sent.append(flag)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _adapter(fake: _Fake, clock: _Clock | None = None, **kw: object) -> CK.ClickAdapter:
    return CK.ClickAdapter(sender=fake, clock=clock or _Clock(), **kw)  # type: ignore[arg-type]


class ArmingTests(unittest.TestCase):
    def test_an_unarmed_click_sends_nothing(self) -> None:
        fake = _Fake()
        adapter = _adapter(fake, enabled=True)
        self.assertFalse(adapter.click(armed=False))
        self.assertEqual(fake.sent, [])
        self.assertEqual(adapter.summary()["clicks"], 0)
        self.assertEqual(adapter.summary()["refused_not_armed"], 1)

    def test_an_armed_click_sends_a_down_and_an_up_in_that_order(self) -> None:
        fake = _Fake()
        adapter = _adapter(fake, enabled=True)
        self.assertTrue(adapter.click(armed=True))
        self.assertEqual(fake.sent, [CK.MOUSEEVENTF_LEFTDOWN, CK.MOUSEEVENTF_LEFTUP])

    def test_simulation_counts_the_click_and_calls_nothing(self) -> None:
        """The two paths must differ only in whether Windows is called."""

        fake = _Fake()
        adapter = _adapter(fake, enabled=False)
        self.assertTrue(adapter.click(armed=True))
        self.assertEqual(fake.sent, [])
        self.assertEqual(adapter.summary()["clicks"], 1)


class StuckButtonTests(unittest.TestCase):
    """A down with no up turns every later gaze movement into a drag, over
    whatever the user is looking at, with no way to release it by looking."""

    def test_a_failed_release_is_reported_not_swallowed(self) -> None:
        fake = _Fake(fail_on=CK.MOUSEEVENTF_LEFTUP)
        adapter = _adapter(fake, enabled=True)
        self.assertFalse(adapter.click(armed=True))
        self.assertTrue(adapter.summary()["button_stuck"])
        self.assertEqual(adapter.summary()["clicks"], 0)
        self.assertEqual(adapter.summary()["failed"], 1)

    def test_release_retries_the_up_and_clears_the_hold(self) -> None:
        fake = _Fake(fail_on=CK.MOUSEEVENTF_LEFTUP)
        adapter = _adapter(fake, enabled=True)
        adapter.click(armed=True)
        fake.fail_on = None  # the next attempt succeeds
        adapter.release()
        self.assertIn(CK.MOUSEEVENTF_LEFTUP, fake.sent)
        self.assertFalse(adapter.summary()["button_stuck"])

    def test_leaving_the_context_releases_even_after_an_exception(self) -> None:
        fake = _Fake(fail_on=CK.MOUSEEVENTF_LEFTUP)
        with self.assertRaises(RuntimeError), _adapter(fake, enabled=True) as adapter:
            adapter.click(armed=True)
            fake.fail_on = None
            raise RuntimeError("the session fell over")
        self.assertEqual(fake.sent.count(CK.MOUSEEVENTF_LEFTUP), 1)

    def test_a_release_that_cannot_be_delivered_does_not_raise_over_the_error(self) -> None:
        """Shutdown caused by an error must keep reporting that error."""

        fake = _Fake(fail_on=CK.MOUSEEVENTF_LEFTUP)
        adapter = _adapter(fake, enabled=True)
        adapter.click(armed=True)
        adapter.release()  # still failing
        self.assertTrue(adapter.summary()["button_stuck"])

    def test_a_failed_press_is_not_a_press(self) -> None:
        fake = _Fake(fail_on=CK.MOUSEEVENTF_LEFTDOWN)
        adapter = _adapter(fake, enabled=True)
        self.assertFalse(adapter.click(armed=True))
        self.assertEqual(fake.sent, [])
        self.assertFalse(adapter.summary()["button_stuck"])
        adapter.release()
        self.assertEqual(fake.sent, [], "released a button that was never pressed")


class DoubleClickTests(unittest.TestCase):
    def test_two_clicks_too_close_together_are_not_sent_as_a_double(self) -> None:
        """Windows reads a fast pair as a double-click, a different action."""

        fake = _Fake()
        clock = _Clock()
        adapter = _adapter(fake, clock, enabled=True)
        self.assertTrue(adapter.click(armed=True))
        clock.t += 0.1
        self.assertFalse(adapter.click(armed=True))
        self.assertEqual(adapter.summary()["refused_too_soon"], 1)
        self.assertEqual(fake.sent.count(CK.MOUSEEVENTF_LEFTDOWN), 1)

    def test_a_click_after_the_gap_is_allowed(self) -> None:
        fake = _Fake()
        clock = _Clock()
        adapter = _adapter(fake, clock, enabled=True)
        adapter.click(armed=True)
        clock.t += 1.0
        self.assertTrue(adapter.click(armed=True))
        self.assertEqual(adapter.summary()["clicks"], 2)

    def test_a_refused_click_does_not_restart_the_gap(self) -> None:
        """Otherwise a stream of refusals could hold the gate open for ever."""

        fake = _Fake()
        clock = _Clock()
        adapter = _adapter(fake, clock, enabled=True)
        adapter.click(armed=True)
        # Three attempts, all inside the gap, so all refused. An earlier
        # version of this test stepped far enough that one of them LANDED and
        # legitimately restarted the gap -- the test was wrong, not the code.
        for _ in range(3):
            clock.t += 0.1
            self.assertFalse(adapter.click(armed=True))
        self.assertEqual(adapter.summary()["refused_too_soon"], 3)
        clock.t += 0.15  # 0.45 s since the click that landed
        self.assertTrue(
            adapter.click(armed=True),
            "refusals pushed the deadline out, so a stream of them could hold the gate shut",
        )


class SeparationTests(unittest.TestCase):
    def test_the_click_module_never_moves_the_pointer(self) -> None:
        """Where a click lands is decided by the file that moves the pointer."""

        source = Path(__import__("gf_click").__file__).read_text(
            encoding="utf-8"
        )
        for forbidden in ("SetCursorPos", "MOUSEEVENTF_MOVE", "mouse_move"):
            self.assertNotIn(forbidden, source)

    def test_the_cursor_module_still_cannot_click(self) -> None:
        """The audit in test_cursor.py must not be quietly relaxed."""

        source = Path(__import__("gf_cursor").__file__).read_text(
            encoding="utf-8"
        )
        for forbidden in ("mouse_event", "SendInput", "MOUSEEVENTF"):
            self.assertNotIn(forbidden, source)


class DoubleClickTests2(unittest.TestCase):
    """Two winks at the same place are a double click, which is what opens
    most things. A pair at DIFFERENT places is a gesture misfiring twice."""

    def _adapter(self, clock: _Clock) -> tuple[CK.ClickAdapter, _Fake]:
        fake = _Fake()
        return CK.ClickAdapter(sender=fake, clock=clock, enabled=True), fake

    def test_a_second_click_at_the_same_pixel_is_let_through(self) -> None:
        clock = _Clock()
        adapter, fake = self._adapter(clock)
        self.assertTrue(adapter.click(armed=True, at=(1000, 500)))
        clock.t += 0.25
        self.assertTrue(
            adapter.click(armed=True, at=(1000, 500)),
            "the guard refused the second half of a double click",
        )
        self.assertEqual(adapter.summary()["doubles"], 1)
        self.assertEqual(fake.sent.count(CK.MOUSEEVENTF_LEFTDOWN), 2)

    def test_a_second_click_somewhere_else_is_still_refused(self) -> None:
        """Time alone would let a gesture that fired twice while the gaze
        moved count as a double, and open something nobody chose."""

        clock = _Clock()
        adapter, fake = self._adapter(clock)
        adapter.click(armed=True, at=(1000, 500))
        clock.t += 0.25
        self.assertFalse(adapter.click(armed=True, at=(2000, 900)))
        self.assertEqual(adapter.summary()["refused_too_soon"], 1)
        self.assertEqual(adapter.summary()["doubles"], 0)

    def test_a_second_click_after_the_double_window_is_two_clicks(self) -> None:
        clock = _Clock()
        adapter, _ = self._adapter(clock)
        adapter.click(armed=True, at=(1000, 500))
        clock.t += 1.2
        self.assertTrue(adapter.click(armed=True, at=(1000, 500)))
        self.assertEqual(adapter.summary()["doubles"], 0)
        self.assertEqual(adapter.summary()["clicks"], 2)

    def test_a_small_wobble_still_counts_as_the_same_place(self) -> None:
        clock = _Clock()
        adapter, _ = self._adapter(clock)
        adapter.click(armed=True, at=(1000, 500))
        clock.t += 0.2
        self.assertTrue(adapter.click(armed=True, at=(1004, 497)))

    def test_without_a_position_it_falls_back_to_refusing(self) -> None:
        """Not knowing where a click landed is not evidence that it was a
        double, and guessing yes would open things by accident."""

        clock = _Clock()
        adapter, _ = self._adapter(clock)
        adapter.click(armed=True, at=None)
        clock.t += 0.2
        self.assertFalse(adapter.click(armed=True, at=None))

    def test_the_gap_is_reported_against_the_systems_own_setting(self) -> None:
        """Ours can be a double by our rule and still miss Windows' window,
        and then nothing opens with no way to tell why."""

        clock = _Clock()
        adapter, _ = self._adapter(clock)
        adapter.click(armed=True, at=(10, 10))
        clock.t += 0.3
        adapter.click(armed=True, at=(10, 10))
        summary = adapter.summary()
        self.assertAlmostEqual(summary["last_double_gap_ms"], 300.0, places=3)
        self.assertGreater(summary["system_double_click_ms"], 0)
        self.assertTrue(summary["last_double_within_system_window"])

    def test_a_pair_past_the_guard_is_not_refused_and_windows_decides(self) -> None:
        """Between min_gap_s and the system's double-click time, the second
        click simply passes as an ordinary click -- and Windows, which is what
        actually interprets a pair, may still read the two as a double.

        So this counter is bookkeeping, not the mechanism: what matters is
        that no legitimate second click is ever REFUSED, and it is not.
        """

        clock = _Clock()
        adapter, fake = self._adapter(clock)
        adapter.click(armed=True, at=(10, 10))
        clock.t += 0.45  # past min_gap_s, still inside Windows' 500 ms
        self.assertTrue(adapter.click(armed=True, at=(10, 10)))
        self.assertEqual(adapter.summary()["refused_too_soon"], 0)
        self.assertEqual(fake.sent.count(CK.MOUSEEVENTF_LEFTDOWN), 2)

    def test_the_limits_refuse_nonsense(self) -> None:
        for bad in ({"double_window_s": 0.0}, {"same_place_px": -1}, {"min_gap_s": -1.0}):
            with self.assertRaises(ValueError):
                CK.ClickLimits(**bad)


class OneWinkOneDoubleTests(unittest.TestCase):
    """One wink sends the whole double click.

    Two winks the person makes separately have to clear a hold, a reopening
    and a cooldown between them and still land inside Windows' 500 ms window.
    That is a timing test they can fail through no fault of their own, and
    failing it silently produces two clicks that open nothing.
    """

    def _adapter(self) -> tuple[CK.ClickAdapter, _Fake, list[float]]:
        fake = _Fake()
        slept: list[float] = []
        return (
            CK.ClickAdapter(sender=fake, clock=_Clock(), sleep=slept.append, enabled=True),
            fake,
            slept,
        )

    def test_one_call_sends_two_complete_click_pairs(self) -> None:
        adapter, fake, _ = self._adapter()
        self.assertTrue(adapter.double_click(armed=True, at=(10, 10)))
        self.assertEqual(
            fake.sent,
            [
                CK.MOUSEEVENTF_LEFTDOWN,
                CK.MOUSEEVENTF_LEFTUP,
                CK.MOUSEEVENTF_LEFTDOWN,
                CK.MOUSEEVENTF_LEFTUP,
            ],
        )

    def test_the_gap_is_ours_and_sits_well_inside_the_system_window(self) -> None:
        """Chosen rather than hoped for: it cannot fail on a slow frame."""

        adapter, _, slept = self._adapter()
        adapter.double_click(armed=True, at=(10, 10))
        self.assertEqual(slept, [adapter.limits.double_gap_s])
        self.assertLess(adapter.limits.double_gap_s * 1000.0, CK.system_double_click_ms() / 2)

    def test_a_gap_of_zero_would_be_one_long_press_and_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CK.ClickLimits(double_gap_s=0.0)

    def test_a_gap_outside_the_double_window_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CK.ClickLimits(double_gap_s=1.0, double_window_s=0.6)

    def test_an_unarmed_double_sends_nothing(self) -> None:
        adapter, fake, _ = self._adapter()
        self.assertFalse(adapter.double_click(armed=False))
        self.assertEqual(fake.sent, [])

    def test_a_failed_second_half_is_reported_as_the_single_it_was(self) -> None:
        """Counting it as a double would claim something opened when it did not."""

        fake = _Fake()
        slept: list[float] = []
        adapter = CK.ClickAdapter(sender=fake, clock=_Clock(), sleep=slept.append, enabled=True)

        calls = {"n": 0}
        real = fake.__call__

        def flaky(flag: int) -> None:
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError("SendInput refused")
            real(flag)

        adapter._send = flaky
        self.assertFalse(adapter.double_click(armed=True, at=(10, 10)))
        self.assertEqual(adapter.summary()["doubles"], 0)
        self.assertEqual(adapter.summary()["clicks"], 1)

    def test_a_double_never_leaves_the_button_held(self) -> None:
        adapter, fake, _ = self._adapter()
        adapter.double_click(armed=True, at=(10, 10))
        self.assertEqual(
            fake.sent.count(CK.MOUSEEVENTF_LEFTDOWN), fake.sent.count(CK.MOUSEEVENTF_LEFTUP)
        )
        self.assertFalse(adapter.summary()["button_stuck"])


class RightClickTests(unittest.TestCase):
    """A second button, and the release that has to know which one is held.

    Adding a button without adding its release is exactly how the failure this
    module opens by naming -- a button held with no way to let go by looking --
    comes back through a door nobody was watching.
    """

    def _adapter(self, **kw: object):
        fake = _Fake()
        return _adapter(fake, enabled=True, **kw), fake

    def test_an_unarmed_right_click_sends_nothing(self) -> None:
        adapter, fake = self._adapter()
        self.assertFalse(adapter.right_click(armed=False))
        self.assertEqual(fake.sent, [])
        self.assertEqual(adapter.summary()["right_clicks"], 0)
        self.assertEqual(adapter.summary()["refused_not_armed"], 1)

    def test_an_armed_right_click_sends_a_down_and_an_up_in_that_order(self) -> None:
        adapter, fake = self._adapter()
        self.assertTrue(adapter.right_click(armed=True, at=(10, 10)))
        self.assertEqual(fake.sent, [CK.MOUSEEVENTF_RIGHTDOWN, CK.MOUSEEVENTF_RIGHTUP])
        self.assertEqual(adapter.summary()["right_clicks"], 1)

    def test_it_is_counted_apart_from_the_left_ones(self) -> None:
        """"How often did the system commit to something" and "how often did
        it offer a choice" are different questions."""

        adapter, _fake = self._adapter()
        adapter.right_click(armed=True, at=(10, 10))
        self.assertEqual(adapter.summary()["clicks"], 0)
        self.assertEqual(adapter.summary()["right_clicks"], 1)

    def test_simulation_counts_it_and_calls_nothing(self) -> None:
        fake = _Fake()
        adapter = _adapter(fake, enabled=False)
        self.assertTrue(adapter.right_click(armed=True, at=(1, 1)))
        self.assertEqual(fake.sent, [])
        self.assertEqual(adapter.summary()["right_clicks"], 1)

    def test_two_right_clicks_at_the_same_pixel_are_not_a_double(self) -> None:
        """The same-pixel exception exists so a DOUBLE can get through. There
        is no double right click, so the ordinary gap guard stands here."""

        clock = _Clock()
        fake = _Fake()
        adapter = _adapter(fake, clock=clock, enabled=True)
        self.assertTrue(adapter.right_click(armed=True, at=(10, 10)))
        clock.t += 0.05
        self.assertFalse(adapter.right_click(armed=True, at=(10, 10)))
        self.assertEqual(adapter.summary()["refused_too_soon"], 1)

    def test_a_failed_right_release_leaves_the_right_button_named(self) -> None:
        adapter, fake = self._adapter()
        real = fake.__call__
        calls = {"n": 0}

        def flaky(flag: int) -> None:
            calls["n"] += 1
            if flag == CK.MOUSEEVENTF_RIGHTUP:
                raise OSError("SendInput refused")
            real(flag)

        adapter._send = flaky
        self.assertFalse(adapter.right_click(armed=True, at=(1, 1)))
        self.assertTrue(adapter.summary()["button_stuck"])
        self.assertEqual(adapter.summary()["stuck_button"], "right")

    def test_release_sends_the_right_up_not_the_left_one(self) -> None:
        """Written for one button, this sent LEFTUP for a held RIGHT button --
        which is a right button still held, and it would look from here like a
        successful release."""

        adapter, fake = self._adapter()
        real = fake.__call__
        state = {"fail": True}

        def flaky(flag: int) -> None:
            if flag == CK.MOUSEEVENTF_RIGHTUP and state["fail"]:
                raise OSError("SendInput refused")
            real(flag)

        adapter._send = flaky
        adapter.right_click(armed=True, at=(1, 1))
        self.assertTrue(adapter.button_stuck)
        state["fail"] = False
        adapter.release()
        self.assertIn(CK.MOUSEEVENTF_RIGHTUP, fake.sent)
        self.assertNotIn(CK.MOUSEEVENTF_LEFTUP, fake.sent)
        self.assertFalse(adapter.button_stuck)

    def test_a_right_click_never_leaves_a_button_held(self) -> None:
        adapter, fake = self._adapter()
        adapter.right_click(armed=True, at=(1, 1))
        self.assertEqual(
            fake.sent.count(CK.MOUSEEVENTF_RIGHTDOWN), fake.sent.count(CK.MOUSEEVENTF_RIGHTUP)
        )
        self.assertFalse(adapter.button_stuck)


class NoPressOverAHeldButtonTests(unittest.TestCase):
    """Pressing again while a button is down overwrote the record of WHICH
    one was held -- after which even release() could not let go of it.

    Reproduced: a right click whose UP fails, then an ordinary left click,
    and the adapter believed nothing was held while the right button was.
    """

    def _stuck(self):
        fake = _Fake()
        adapter = _adapter(fake, enabled=True)
        real = fake.__call__
        fail = {"on": True}

        def flaky(flag: int) -> None:
            if flag == CK.MOUSEEVENTF_RIGHTUP and fail["on"]:
                raise OSError("SendInput refused")
            real(flag)

        adapter._send = flaky
        adapter.right_click(armed=True, at=(1, 1))
        return adapter, fake, fail

    def test_a_later_click_is_refused_rather_than_losing_the_held_button(self) -> None:
        adapter, fake, _fail = self._stuck()
        self.assertEqual(adapter.stuck_button, "right", "the fixture did not get stuck")
        adapter._last_s = None  # past the gap guard, so only the held button can refuse it
        self.assertFalse(adapter.click(armed=True, at=(50, 50)))
        self.assertEqual(adapter.stuck_button, "right", "the held button was forgotten")
        self.assertEqual(adapter.summary()["refused_button_held"], 1)
        self.assertNotIn(CK.MOUSEEVENTF_LEFTDOWN, fake.sent)

    def test_clicking_works_again_once_the_release_gets_through(self) -> None:
        """The other half. An adapter that refused every click after one
        failure would pass the test above and be useless."""

        adapter, fake, fail = self._stuck()
        fail["on"] = False
        adapter._last_s = None
        self.assertTrue(adapter.click(armed=True, at=(50, 50)))
        self.assertIsNone(adapter.stuck_button)
        self.assertIn(CK.MOUSEEVENTF_RIGHTUP, fake.sent, "the right button was never released")
        self.assertIn(CK.MOUSEEVENTF_LEFTDOWN, fake.sent)
