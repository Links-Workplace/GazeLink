"""Scroll zones, the repeat that drives them, and the wheel adapter.

No camera, no model, no OS input.  These assert the safety properties rather
than that the functions return something: scrolling REPEATS, so what matters
is not that it starts but that it STOPS -- on every one of the ways a person
can stop asking for it -- and that nothing here can press a button.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_click as CK  # noqa: E402
import gf_dwell as D  # noqa: E402
import gf_scroll as S  # noqa: E402

HZ = 30.0
STEP = 1.0 / HZ


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _Fake:
    """A sender that records deltas instead of calling Windows."""

    def __init__(self) -> None:
        self.sent: list[int] = []

    def __call__(self, delta: int) -> None:
        self.sent.append(delta)


def _run(repeater, zone, seconds, *, start=0.0, usable=True, step=STEP):
    """Feed frames at camera rate; return the notches emitted, in order."""

    out, now, until = [], start, start + seconds
    while now <= until:
        notch = repeater.update(now, zone, usable=usable)
        if notch:
            out.append(notch)
        now += step
    return out


class ZoneLayoutTests(unittest.TestCase):
    """Where the targets are is a safety property, not decoration."""

    def test_the_bands_sit_inside_the_calibrated_horizontal_band(self) -> None:
        """Outside it the model is not merely less accurate."""

        for zone in S.scroll_zones():
            self.assertGreaterEqual(zone.x0, D.BAND[0])
            self.assertLessEqual(zone.x1, D.BAND[1])

    def test_there_is_a_dead_zone_between_them_to_rest_in(self) -> None:
        """Resting the gaze must be possible without scrolling."""

        up, down = S.scroll_zones()
        self.assertLess(up.y1, down.y0)
        self.assertGreater(down.y0 - up.y1, 0.3, "the middle is too thin to rest in")

    def test_the_tile_is_outside_both_bands(self) -> None:
        """Otherwise it could not be looked at without scrolling first."""

        tile = S.scroll_tile()
        for zone in S.scroll_zones():
            overlaps = not (tile.y1 <= zone.y0 or tile.y0 >= zone.y1)
            self.assertFalse(overlaps, f"the tile overlaps {zone.key}")

    def test_the_tile_is_not_where_the_gaze_rests_to_stop(self) -> None:
        """A tile in the stop zone would turn every stop into an exit."""

        tile = S.scroll_tile()
        centre = ((D.BAND[0] + D.BAND[1]) / 2.0, 0.5)
        self.assertFalse(tile.contains(centre))

    def test_a_layout_with_no_dead_zone_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            S.ScrollConfig(band_height=0.45, band_margin=0.10)

    def test_the_config_refuses_nonsense(self) -> None:
        for bad in ({"arm_ms": 0.0}, {"repeat_ms": -1.0}, {"band_height": 0.0}):
            with self.assertRaises(ValueError):
                S.ScrollConfig(**bad)


class RepeatTests(unittest.TestCase):
    def test_nothing_happens_before_the_wait_is_over(self) -> None:
        repeater = S.ScrollRepeater(S.ScrollConfig(arm_ms=500.0))
        emitted = _run(repeater, S.UP, 0.4)
        self.assertEqual(emitted, [], "it scrolled before it was asked for long enough")

    def test_it_starts_once_the_wait_is_over(self) -> None:
        repeater = S.ScrollRepeater(S.ScrollConfig(arm_ms=500.0))
        self.assertTrue(_run(repeater, S.UP, 0.7))

    def test_it_repeats_at_the_rate_it_was_given(self) -> None:
        cfg = S.ScrollConfig(arm_ms=200.0, repeat_ms=500.0)
        emitted = _run(S.ScrollRepeater(cfg), S.UP, 3.2)
        self.assertGreaterEqual(len(emitted), 6)
        self.assertLessEqual(len(emitted), 8, f"it scrolled {len(emitted)} times, far too fast")

    def test_up_and_down_carry_opposite_signs(self) -> None:
        cfg = S.ScrollConfig(arm_ms=200.0)
        self.assertEqual(_run(S.ScrollRepeater(cfg), S.UP, 0.5)[0], 1)
        self.assertEqual(_run(S.ScrollRepeater(cfg), S.DOWN, 0.5)[0], -1)


class StoppingTests(unittest.TestCase):
    """The half that matters.

    A scroll that starts is a feature; a scroll that will not stop is the
    "repeated or stuck action" failure in CLAUDE.md 4.1.
    """

    def _armed(self):
        cfg = S.ScrollConfig(arm_ms=200.0, repeat_ms=100.0)
        repeater = S.ScrollRepeater(cfg)
        self.assertTrue(_run(repeater, S.UP, 0.5), "it never started, so stopping proves nothing")
        return repeater

    def test_resting_the_gaze_in_the_middle_stops_it(self) -> None:
        """How a person stops: not a special case, just no longer asking."""

        self.assertEqual(_run(self._armed(), None, 2.0, start=10.0), [])

    def test_an_unusable_point_stops_it(self) -> None:
        """Stale, or no face. The zone still reads UP -- that is the point."""

        self.assertEqual(_run(self._armed(), S.UP, 2.0, start=10.0, usable=False), [])

    def test_stopping_throws_the_wait_away_too(self) -> None:
        """A glance away and back must not resume instantly.

        The wait exists so that a glance is not an instruction.
        """

        cfg = S.ScrollConfig(arm_ms=500.0, repeat_ms=100.0)
        repeater = S.ScrollRepeater(cfg)
        _run(repeater, S.UP, 0.8)
        repeater.update(20.0, None)
        self.assertEqual(
            _run(repeater, S.UP, 0.4, start=21.0), [], "it resumed without waiting again"
        )

    def test_changing_bands_restarts_the_wait(self) -> None:
        cfg = S.ScrollConfig(arm_ms=500.0, repeat_ms=100.0)
        repeater = S.ScrollRepeater(cfg)
        _run(repeater, S.UP, 0.8)
        self.assertEqual(
            _run(repeater, S.DOWN, 0.4, start=20.0), [], "it changed direction without waiting"
        )

    def test_an_explicit_stop_ends_it(self) -> None:
        repeater = self._armed()
        repeater.stop()
        self.assertFalse(repeater.armed)
        self.assertEqual(_run(repeater, S.UP, 0.15, start=10.0), [])

    def test_looking_at_the_tile_stops_scrolling(self) -> None:
        """The tile is not a band, so it is simply not a request to scroll."""

        self.assertEqual(_run(self._armed(), S.TILE, 2.0, start=10.0), [])


class WheelAdapterTests(unittest.TestCase):
    def _adapter(self, fake, clock=None, **kw):
        return CK.ScrollAdapter(sender=fake, clock=clock or _Clock(), **kw)

    def test_an_unarmed_scroll_sends_nothing(self) -> None:
        fake = _Fake()
        adapter = self._adapter(fake, enabled=True)
        self.assertFalse(adapter.scroll(1, armed=False))
        self.assertEqual(fake.sent, [])
        self.assertEqual(adapter.summary()["refused_not_armed"], 1)

    def test_up_and_down_send_opposite_deltas(self) -> None:
        fake, clock = _Fake(), _Clock()
        adapter = self._adapter(fake, clock, enabled=True)
        adapter.scroll(1, armed=True)
        clock.t += 1.0
        adapter.scroll(-1, armed=True)
        self.assertEqual(fake.sent, [CK.WHEEL_DELTA, -CK.WHEEL_DELTA])

    def test_it_refuses_a_second_turn_that_is_too_soon(self) -> None:
        fake, clock = _Fake(), _Clock()
        adapter = self._adapter(fake, clock, enabled=True)
        adapter.scroll(1, armed=True)
        self.assertFalse(adapter.scroll(1, armed=True))
        self.assertEqual(len(fake.sent), 1)
        self.assertEqual(adapter.summary()["refused_too_soon"], 1)

    def test_a_wild_request_is_capped(self) -> None:
        """A bad number from a flag must not jump a screenful at a time."""

        fake = _Fake()
        adapter = self._adapter(fake, enabled=True)
        adapter.scroll(9999, armed=True)
        self.assertEqual(fake.sent, [adapter.limits.max_notches * CK.WHEEL_DELTA])

    def test_disabled_counts_but_calls_nothing(self) -> None:
        fake = _Fake()
        adapter = self._adapter(fake, enabled=False)
        self.assertTrue(adapter.scroll(1, armed=True))
        self.assertEqual(fake.sent, [])
        self.assertEqual(adapter.summary()["notches_up"], 1)

    def test_ready_predicts_what_scroll_then_does(self) -> None:
        fake, clock = _Fake(), _Clock()
        adapter = self._adapter(fake, clock, enabled=True)
        adapter.scroll(1, armed=True)
        for step in (0.01, 0.02, 0.05, 0.1):
            clock.t += step
            self.assertEqual(adapter.ready(), adapter.scroll(1, armed=True))

    def test_a_failed_turn_is_counted_and_not_claimed(self) -> None:
        class _Broken:
            def __call__(self, delta: int) -> None:
                raise OSError("SendInput refused")

        adapter = self._adapter(_Broken(), enabled=True)
        self.assertFalse(adapter.scroll(1, armed=True))
        self.assertEqual(adapter.summary()["failed"], 1)
        self.assertEqual(adapter.summary()["notches_up"], 0)


class SeparationTests(unittest.TestCase):
    """The audit this project applies to every file that can reach the OS."""

    def test_the_scroll_logic_cannot_reach_the_os_at_all(self) -> None:
        source = Path(__import__("gf_scroll").__file__).read_text(
            encoding="utf-8"
        )
        for forbidden in ("SendInput", "MOUSEEVENTF", "SetCursorPos", "ctypes", "windll"):
            self.assertNotIn(forbidden, source)

    def test_the_wheel_never_presses_a_button(self) -> None:
        fake = _Fake()
        adapter = CK.ScrollAdapter(sender=fake, enabled=True)
        adapter.scroll(1, armed=True)
        for delta in fake.sent:
            self.assertNotIn(delta, (CK.MOUSEEVENTF_LEFTDOWN, CK.MOUSEEVENTF_LEFTUP))
        self.assertFalse(hasattr(adapter, "release"), "a wheel has nothing to release")

    def test_the_click_module_still_never_moves_the_pointer(self) -> None:
        """The wheel was added to that file; the existing audit must still hold."""

        source = Path(__import__("gf_click").__file__).read_text(
            encoding="utf-8"
        )
        for forbidden in ("SetCursorPos", "MOUSEEVENTF_MOVE", "mouse_move"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
