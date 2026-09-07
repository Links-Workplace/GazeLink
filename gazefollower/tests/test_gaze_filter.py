"""Deterministic tests for the live calibrated-point filters."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_gaze_filter as F  # noqa: E402


class GazePointFilterTests(unittest.TestCase):
    def settings(self, kind: F.FilterKind, **changes) -> F.FilterSettings:
        values = {"width_px": 1000, "height_px": 500, "kind": kind, **changes}
        return F.FilterSettings(**values)

    def test_off_is_an_exact_pass_through(self) -> None:
        flt = F.GazePointFilter(self.settings(F.FilterKind.OFF))
        self.assertEqual(flt.update((-0.2, 1.4), 1.0), (-0.2, 1.4))

    def test_all_filters_preserve_the_first_point(self) -> None:
        for kind in (F.FilterKind.EMA, F.FilterKind.ONE_EURO, F.FilterKind.KALMAN):
            with self.subTest(kind=kind):
                flt = F.GazePointFilter(self.settings(kind))
                self.assertEqual(flt.update((0.25, 0.75), 1.0), (0.25, 0.75))

    def test_one_euro_reduces_stationary_noise(self) -> None:
        flt = F.GazePointFilter(self.settings(F.FilterKind.ONE_EURO))
        raw = []
        filtered = []
        for i in range(180):
            x = 0.5 + (0.012 if i % 2 else -0.012)
            raw.append(x)
            filtered.append(flt.update((x, 0.5), i / 30.0)[0])
        raw_span = max(raw[30:]) - min(raw[30:])
        filtered_span = max(filtered[30:]) - min(filtered[30:])
        self.assertLess(filtered_span, raw_span * 0.25)

    def test_one_euro_responds_faster_when_beta_is_higher(self) -> None:
        slow = F.GazePointFilter(
            self.settings(F.FilterKind.ONE_EURO, one_euro_beta_hz_per_px_s=0.0)
        )
        fast = F.GazePointFilter(
            self.settings(F.FilterKind.ONE_EURO, one_euro_beta_hz_per_px_s=0.01)
        )
        for i in range(30):
            slow.update((0.2, 0.5), i / 30.0)
            fast.update((0.2, 0.5), i / 30.0)
        slow_after_step = slow.update((0.8, 0.5), 1.0)[0]
        fast_after_step = fast.update((0.8, 0.5), 1.0)[0]
        self.assertGreater(fast_after_step, slow_after_step)

    def test_variable_frame_rate_uses_elapsed_time(self) -> None:
        short = F.GazePointFilter(self.settings(F.FilterKind.EMA))
        long = F.GazePointFilter(self.settings(F.FilterKind.EMA))
        short.update((0.0, 0.0), 0.0)
        long.update((0.0, 0.0), 0.0)
        short_step = short.update((1.0, 1.0), 0.01)[0]
        long_step = long.update((1.0, 1.0), 0.10)[0]
        self.assertGreater(long_step, short_step)

    def test_gap_and_non_increasing_time_reset_to_current_measurement(self) -> None:
        flt = F.GazePointFilter(self.settings(F.FilterKind.ONE_EURO, reset_gap_ms=100.0))
        flt.update((0.1, 0.1), 1.0)
        self.assertEqual(flt.update((0.9, 0.9), 1.2), (0.9, 0.9))
        self.assertEqual(flt.update((0.2, 0.2), 1.2), (0.2, 0.2))
        self.assertEqual(flt.update((0.7, 0.7), 1.1), (0.7, 0.7))

    def test_invalid_input_raises_and_clears_history(self) -> None:
        flt = F.GazePointFilter(self.settings(F.FilterKind.ONE_EURO))
        flt.update((0.1, 0.1), 1.0)
        with self.assertRaises(ValueError):
            flt.update((math.nan, 0.2), 1.1)
        self.assertEqual(flt.update((0.8, 0.8), 1.2), (0.8, 0.8))

    def test_reset_prevents_a_stale_point_from_dragging_recovery(self) -> None:
        flt = F.GazePointFilter(self.settings(F.FilterKind.ONE_EURO))
        flt.update((0.1, 0.1), 1.0)
        flt.update((0.2, 0.2), 1.03)
        flt.reset()
        self.assertEqual(flt.update((0.9, 0.9), 1.06), (0.9, 0.9))

    def test_bad_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.settings(F.FilterKind.ONE_EURO, one_euro_min_cutoff_hz=0.0)
        with self.assertRaises(ValueError):
            self.settings(F.FilterKind.ONE_EURO, one_euro_beta_hz_per_px_s=-1.0)


if __name__ == "__main__":
    unittest.main()
