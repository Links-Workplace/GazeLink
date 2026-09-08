"""The cursor adapter: never touches the real pointer in these tests.

Every test injects a fake setter/getter. If any of these ever moved the real
cursor, the suite itself would be emitting OS input -- which the project
forbids outright.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_cursor as CU  # noqa: E402


class _Fake:
    """Stands in for Windows. Records instead of acting."""

    def __init__(self, start: tuple[int, int] = (100, 100)) -> None:
        self.pos = start
        self.calls: list[tuple[int, int]] = []

    def set(self, x: int, y: int) -> None:
        self.pos = (x, y)
        self.calls.append((x, y))

    def get(self) -> tuple[int, int]:
        return self.pos


def _adapter(fake: _Fake, **kw) -> CU.CursorAdapter:
    return CU.CursorAdapter(setter=fake.set, getter=fake.get, **kw)


class OptInTests(unittest.TestCase):
    def test_disabled_by_default_and_touches_nothing(self) -> None:
        """OS input must be opt-in and separated from simulation."""

        fake = _Fake()
        adapter = _adapter(fake)
        self.assertFalse(adapter.enabled)
        adapter.update((900, 400))
        self.assertEqual(fake.calls, [], "the real cursor was moved without being enabled")

    def test_simulation_still_computes_the_same_position(self) -> None:
        """The two paths must differ only in whether Windows is called, or
        what is tested in simulation is not what runs for real."""

        off, on = _Fake(), _Fake()
        a, b = _adapter(off), _adapter(on, enabled=True)
        for target in ((300, 300), (420, 380), (500, 500)):
            self.assertEqual(a.update(target), b.update(target))
        self.assertEqual(off.calls, [])
        self.assertEqual(len(on.calls), 3)

    def test_it_cannot_click(self) -> None:
        """There must be no code path that presses a button."""

        source = (Path(__file__).resolve().parent.parent / "gf_cursor.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("mouse_event", "SendInput", "MOUSEEVENTF", "mouse_down", "click("):
            self.assertNotIn(forbidden, source)
        self.assertEqual(_adapter(_Fake(), enabled=True).summary()["clicks"], 0)


class FreezeTests(unittest.TestCase):
    def test_no_point_freezes_instead_of_guessing(self) -> None:
        """Tracking loss must never move the pointer to a guess."""

        fake = _Fake()
        adapter = _adapter(fake, enabled=True)
        adapter.update((300, 300))
        before = fake.pos
        for _ in range(30):
            adapter.update(None)
        self.assertEqual(fake.pos, before)
        self.assertEqual(adapter.frozen_frames, 30)

    def test_freezing_is_counted_not_silent(self) -> None:
        """'The cursor did not move' and 'nothing was asked of it' are
        different states and must be distinguishable afterwards."""

        adapter = _adapter(_Fake())
        adapter.update(None)
        self.assertEqual(adapter.frozen_frames, 1)
        self.assertEqual(adapter.moves, 0)

    def test_pausing_stops_movement_without_losing_position(self) -> None:
        fake = _Fake()
        adapter = _adapter(fake, enabled=True)
        adapter.update((300, 300))
        adapter.paused = True
        self.assertEqual(adapter.update((900, 900)), (300, 300))
        self.assertEqual(fake.pos, (300, 300))
        adapter.paused = False
        self.assertEqual(adapter.update((320, 310)), (320, 310))


class StepLimitTests(unittest.TestCase):
    def test_a_wild_jump_is_capped(self) -> None:
        """A prediction that leaps the width of the screen is far likelier to
        be an error than an intention, and an uncapped jump is what makes a
        bad frame unrecoverable rather than merely wrong."""

        fake = _Fake()
        adapter = _adapter(fake, enabled=True, limits=CU.CursorLimits(max_step_px=100))
        adapter.update((500, 500))
        self.assertEqual(adapter.update((5000, 500)), (600, 500))
        self.assertEqual(adapter.clamped_steps, 1)

    def test_ordinary_movement_is_not_clamped(self) -> None:
        """Smoothing may shorten the step; the cap must not be what did it."""

        adapter = _adapter(_Fake(), enabled=True, limits=CU.CursorLimits(max_step_px=100))
        adapter.update((500, 500))
        adapter.update((540, 470))
        self.assertEqual(adapter.clamped_steps, 0)


class SmoothingTests(unittest.TestCase):
    """The pointer is smoothed; the gaze point is not.

    The profile's filter belongs to the configuration verified at 20/20 on the
    selection task, so it is left alone. What is smoothed here is only how the
    pointer presents that same point.
    """

    def test_the_pointer_eases_toward_a_new_target_rather_than_snapping(self) -> None:
        adapter = _adapter(_Fake(), enabled=True)
        adapter.update((500, 500))
        first = adapter.update((900, 500))
        self.assertGreater(first[0], 500, "the pointer did not move toward the target")
        self.assertLess(first[0], 900, "the pointer snapped instead of easing")

    def test_a_held_gaze_converges_on_the_target(self) -> None:
        """Easing must not mean never arriving."""

        adapter = _adapter(_Fake(), enabled=True)
        adapter.update((500, 500))
        for _ in range(40):
            last = adapter.update((900, 500))
        self.assertLess(abs(last[0] - 900), 5)

    def test_micro_movement_does_not_move_the_pointer_at_all(self) -> None:
        """A gaze never holds perfectly still. Without a floor the pointer
        jitters forever around a target the person is looking at steadily."""

        adapter = _adapter(_Fake(), enabled=True, limits=CU.CursorLimits(dead_zone_px=12))
        adapter.update((500, 500))
        for offset in (2, -3, 1, -2, 3, -1):
            self.assertEqual(adapter.update((500 + offset, 500)), (500, 500))
        self.assertGreater(adapter.held_frames, 0)

    def test_a_deliberate_move_still_gets_through_the_dead_zone(self) -> None:
        """The floor must suppress jitter, not intent."""

        adapter = _adapter(_Fake(), enabled=True, limits=CU.CursorLimits(dead_zone_px=12))
        adapter.update((500, 500))
        for _ in range(10):
            last = adapter.update((800, 500))
        self.assertGreater(last[0], 700)

    def test_freezing_drops_the_smoothing_history(self) -> None:
        """Resuming from a position the gaze left long ago would slide the
        pointer across the screen from a stale start."""

        adapter = _adapter(_Fake(), enabled=True)
        adapter.update((100, 100))
        for _ in range(20):
            adapter.update(None)
        # Deliberately inside the step cap, so this measures the smoothing
        # history and not the cap: eased from the stale point it would land
        # near (170, 135); with the history dropped it goes straight there.
        self.assertEqual(adapter.update((300, 200)), (300, 200))

    def test_the_step_cap_still_applies_after_a_freeze(self) -> None:
        """Dropping the history must not also drop the safety limit: a long
        blink followed by a wild prediction is exactly when an uncapped jump
        would take the desktop away."""

        adapter = _adapter(_Fake(), enabled=True, limits=CU.CursorLimits(max_step_px=400))
        adapter.update((100, 100))
        adapter.update(None)
        self.assertEqual(adapter.update((3000, 800)), (500, 500))
        self.assertEqual(adapter.clamped_steps, 1)

    def test_the_first_frame_is_not_capped_against_nothing(self) -> None:
        adapter = _adapter(_Fake(), enabled=True, limits=CU.CursorLimits(max_step_px=10))
        self.assertEqual(adapter.update((4000, 900)), (4000, 900))

    def test_a_capped_jump_still_moves_toward_the_target(self) -> None:
        """Clamping must not freeze the pointer, only slow it."""

        adapter = _adapter(_Fake(), enabled=True, limits=CU.CursorLimits(max_step_px=50))
        adapter.update((0, 0))
        first = adapter.update((1000, 0))
        second = adapter.update((1000, 0))
        self.assertEqual((first, second), ((50, 0), (100, 0)))


class ReleaseTests(unittest.TestCase):
    def test_the_pointer_is_put_back_where_it_was_found(self) -> None:
        """A session must not leave the pointer parked wherever the last gaze
        landed, with the operator unable to reach what would fix it."""

        fake = _Fake(start=(77, 88))
        with _adapter(fake, enabled=True) as adapter:
            adapter.update((3000, 900))
            self.assertEqual(fake.pos, (3000, 900))
        self.assertEqual(fake.pos, (77, 88))

    def test_the_pointer_is_released_even_when_the_session_raises(self) -> None:
        fake = _Fake(start=(5, 6))
        with self.assertRaises(RuntimeError), _adapter(fake, enabled=True) as adapter:
            adapter.update((2000, 700))
            raise RuntimeError("camera died")
        self.assertEqual(fake.pos, (5, 6))

    def test_release_never_raises_out_of_a_failing_shutdown(self) -> None:
        class Broken(_Fake):
            def set(self, x: int, y: int) -> None:
                raise OSError("display gone")

        adapter = _adapter(Broken(), enabled=True)
        adapter.origin = (1, 2)
        adapter.release()  # must not raise

    def test_a_simulated_session_has_nothing_to_release(self) -> None:
        fake = _Fake(start=(9, 9))
        with _adapter(fake) as adapter:
            adapter.update((400, 400))
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
