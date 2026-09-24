"""Bridging a blink for a dwell, and refusing to bridge anything else.

The defect this closes, measured on ``recordings/round47/T1`` through this
person's own gate: ``eyes_steady`` is true on 93.7% of frames, but a 900 ms fill
needs 29 consecutive and the steady runs had a median of 5 -- only 7 of 31 were
long enough. ``DwellEngine.update(None)`` zeroes ``_start_s``, so the scattered
6.3% meant the bar never filled while the drawn dot sat on the tile.

What must stay true after the fix:

* a blink-length gap is bridged, and a long CLOSE is not -- that is the gesture
  that opens the menu, and finishing a dwell on it would be a second meaning
  for one signal;
* a tracking loss is never bridged (CLAUDE.md 4.1: a loss stops things);
* holding is earned. One steady frame must not buy another full hold, or a fill
  could complete on almost nothing but held frames.

Pure state and a fed clock: no camera, no window, no OS input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gazelink_core.interaction import steadiness as STEADY  # noqa: E402

P = (0.5, 0.84)
FRAME_S = 0.0325  # the measured interval on that recording


class _Feed:
    """Frames at a fixed interval, so a test reads as durations."""

    def __init__(self, hold_ms: float = STEADY.DEFAULT_HOLD_MS) -> None:
        self.hold = STEADY.SteadyHold(hold_ms=hold_ms)
        self.now = 1000.0
        self.out: list[tuple[float, float] | None] = []

    def steady(self, frames: int = 1, point: tuple[float, float] = P) -> None:
        for _ in range(frames):
            self.now += FRAME_S
            self.out.append(
                self.hold.update(self.now, steady_point=point, have_prediction=True)
            )

    def blink(self, frames: int = 1) -> None:
        """A point exists but the eyes are not open: an eyelid on its way down."""

        for _ in range(frames):
            self.now += FRAME_S
            self.out.append(
                self.hold.update(self.now, steady_point=None, have_prediction=True)
            )

    def lost(self, frames: int = 1) -> None:
        for _ in range(frames):
            self.now += FRAME_S
            self.out.append(
                self.hold.update(self.now, steady_point=None, have_prediction=False)
            )

    def ms(self, frames: int) -> float:
        return frames * FRAME_S * 1000.0


class TheMeasuredGapsAreBridgedTests(unittest.TestCase):
    def test_a_one_frame_gap_is_bridged(self) -> None:
        """The median gap on that recording was a single frame."""

        f = _Feed()
        f.steady(10)
        f.blink(1)
        self.assertEqual(f.out[-1], P)

    def test_the_p90_gap_is_bridged(self) -> None:
        f = _Feed()
        f.steady(30)
        f.blink(3)
        self.assertTrue(all(o == P for o in f.out[-3:]))

    def test_the_longest_observed_gap_is_bridged(self) -> None:
        """9 frames, 292 ms: a natural blink, the worst seen in 904 frames."""

        f = _Feed()
        f.steady(40)
        f.blink(9)
        self.assertTrue(all(o == P for o in f.out[-9:]))

    def test_the_point_is_the_last_open_eyed_one_not_a_guess(self) -> None:
        f = _Feed()
        f.steady(5, point=(0.4, 0.8))
        f.steady(5, point=(0.6, 0.9))
        f.blink(2)
        self.assertTrue(all(o == (0.6, 0.9) for o in f.out[-2:]))


class ADeliberateCloseIsNotBridgedTests(unittest.TestCase):
    def test_the_hold_sits_above_a_natural_blink_and_below_a_deliberate_one(self) -> None:
        # Both numbers are measured on this person and recorded in the project:
        # natural blinks reached 297 ms, deliberate holds started at 1172 ms,
        # and the confirm threshold is 800 ms.
        self.assertGreater(STEADY.DEFAULT_HOLD_MS, 297.0)
        self.assertLess(STEADY.DEFAULT_HOLD_MS, 800.0)

    def test_a_confirm_length_close_runs_out_of_budget(self) -> None:
        f = _Feed()
        f.steady(40)
        f.blink(int(0.8 / FRAME_S))  # 800 ms, the confirm threshold
        self.assertIsNone(f.out[-1])

    def test_running_out_is_counted_rather_than_silent(self) -> None:
        f = _Feed()
        f.steady(40)
        f.blink(int(0.8 / FRAME_S))
        self.assertEqual(f.hold.exhausted, 1)


class ALossIsNeverBridgedTests(unittest.TestCase):
    def test_no_prediction_returns_nothing_even_within_the_hold(self) -> None:
        f = _Feed()
        f.steady(40)
        f.lost(1)
        self.assertIsNone(f.out[-1])

    def test_a_loss_forgets_the_point_so_recovery_starts_clean(self) -> None:
        f = _Feed()
        f.steady(40)
        f.lost(2)
        f.blink(1)
        self.assertIsNone(f.out[-1])

    def test_nothing_is_held_before_the_eyes_have_ever_been_open(self) -> None:
        f = _Feed()
        f.blink(3)
        self.assertTrue(all(o is None for o in f.out))


class HoldingIsEarnedTests(unittest.TestCase):
    def test_one_steady_frame_does_not_buy_another_full_hold(self) -> None:
        """The chaining hole: [1 steady, many held] repeated must not hold forever."""

        f = _Feed()
        f.steady(40)
        for _ in range(6):
            f.blink(5)
            f.steady(1)
        f.blink(5)
        # By now far more has been spent than one steady frame per cycle earned.
        self.assertIsNone(f.out[-1])

    def test_sustained_steadiness_refills_it(self) -> None:
        f = _Feed()
        f.steady(40)
        f.blink(9)
        f.steady(40)
        f.blink(9)
        self.assertTrue(all(o == P for o in f.out[-9:]))

    def test_held_time_cannot_exceed_steady_time_over_a_long_window(self) -> None:
        f = _Feed()
        f.steady(1)
        for _ in range(40):
            f.blink(4)
            f.steady(1)
        held = sum(1 for o in f.out if o is not None)
        steady_frames = 41
        # Every non-None frame is either steady or held; the budget rule caps
        # the held share at the steady share.
        self.assertLessEqual(held - steady_frames, steady_frames)


class ResetTests(unittest.TestCase):
    def test_reset_forgets_the_point(self) -> None:
        f = _Feed()
        f.steady(40)
        f.hold.reset()
        f.blink(1)
        self.assertIsNone(f.out[-1])

    def test_reset_refills_the_budget(self) -> None:
        f = _Feed()
        f.steady(40)
        f.blink(int(0.8 / FRAME_S))
        f.hold.reset()
        f.steady(5)
        f.blink(3)
        self.assertTrue(all(o == P for o in f.out[-3:]))


class ItRefusesNonsenseTests(unittest.TestCase):
    def test_a_non_positive_hold_is_refused(self) -> None:
        for bad in (0.0, -1.0):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    STEADY.SteadyHold(hold_ms=bad)


if __name__ == "__main__":
    unittest.main()
