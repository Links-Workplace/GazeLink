"""The band-local zone grid and the per-zone regression gate.

The gate exists because an aggregate one can approve a model that improves the
average and ruins one region.  Every test here is a property of that gate --
and each is asserted in both directions, because a gate that refused
everything would pass every "it must refuse" test on its own.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_report as R  # noqa: E402
import gf_targets as T  # noqa: E402


def _seg(name: str, x: float, y: float, errors: list[float] | np.ndarray) -> R.SegmentResult:
    errors = np.asarray(errors, dtype=np.float64)
    return R.SegmentResult(
        target_id=abs(hash(name)) % 10_000,
        target_name=name,
        target_x=x,
        target_y=y,
        condition="c",
        near_calibration=False,
        n_eligible=int(errors.size),
        n_valid=int(errors.size),
        median_dx_px=float(np.median(errors)),
        median_dy_px=float(np.median(errors)),
        median_euclid_px=float(np.median(errors)),
        euclid_px=errors,
    )


def _zone(rows, zone):
    return next(r for r in rows if r.get("zone") == zone)


class BandGridTests(unittest.TestCase):
    def test_the_whole_band_is_two_columns_on_the_global_grid(self) -> None:
        """Why a new grid exists at all."""

        cols = {T.zone_of(x, 0.5) % 4 for x in np.linspace(0.30, 0.70, 41)}
        self.assertEqual(cols, {1, 2})

    def test_the_band_grid_splits_x_inside_the_band(self) -> None:
        cols = {T.band_zone_of(x, 0.5) % T.BAND_COLS for x in np.linspace(0.30, 0.70, 41)}
        self.assertEqual(cols, set(range(T.BAND_COLS)))

    def test_y_is_not_restricted_to_the_band(self) -> None:
        """The reported failure is around y 0.80. A grid that clipped y to
        0.30-0.70 would exclude exactly the region under investigation."""

        self.assertIsNotNone(T.band_zone_of(0.385, 0.86), "the bottom tiles fell off the grid")
        self.assertIsNotNone(T.band_zone_of(0.385, 0.14), "the top tiles fell off the grid")
        self.assertEqual(T.band_zone_of(0.385, 0.86) // T.BAND_COLS, T.BAND_ROWS - 1)

    def test_x_outside_the_band_is_none_not_a_zone(self) -> None:
        self.assertIsNone(T.band_zone_of(0.20, 0.5))
        self.assertIsNone(T.band_zone_of(0.80, 0.5))

    def test_a_prediction_just_off_screen_stays_in_its_edge_row(self) -> None:
        """Dropping it would remove the worst samples from the worst cell."""

        self.assertEqual(T.band_zone_of(0.5, 1.07) // T.BAND_COLS, T.BAND_ROWS - 1)
        self.assertEqual(T.band_zone_of(0.5, -0.05) // T.BAND_COLS, 0)

    def test_the_menu_tiles_land_in_different_zones(self) -> None:
        tiles = [(0.385, 0.14), (0.615, 0.14), (0.385, 0.86), (0.615, 0.86)]
        self.assertEqual(len({T.band_zone_of(x, y) for x, y in tiles}), 4)


class ZoneTableTests(unittest.TestCase):
    def test_an_empty_zone_is_missing_not_zero(self) -> None:
        rows = R.by_band_zone([_seg("a", 0.5, 0.5, [50.0] * 60)])
        empty = [r for r in rows if r.get("zone") is not None and r["n_segments"] == 0]
        self.assertTrue(empty)
        for row in empty:
            self.assertEqual(row["status"], R.MISSING)
            self.assertNotIn("median_px", row)

    def test_too_little_data_is_unverified_never_measured(self) -> None:
        rows = R.by_band_zone([_seg("a", 0.5, 0.5, [50.0] * 60)])
        self.assertEqual(_zone(rows, T.band_zone_of(0.5, 0.5))["status"], R.UNVERIFIED)

    def test_enough_data_is_measured(self) -> None:
        rows = R.by_band_zone(
            [_seg("a", 0.5, 0.5, [50.0] * 30), _seg("b", 0.52, 0.45, [60.0] * 30)]
        )
        self.assertEqual(_zone(rows, T.band_zone_of(0.5, 0.5))["status"], "measured")

    def test_the_p90_is_over_frames_and_sees_what_the_median_hides(self) -> None:
        """Mostly accurate with a frequent large miss: the median stays low,
        and a person still cannot select anything."""

        frames = [40.0] * 80 + [400.0] * 20
        rows = R.by_band_zone([_seg("a", 0.5, 0.5, frames), _seg("b", 0.5, 0.5, frames)])
        row = _zone(rows, T.band_zone_of(0.5, 0.5))
        self.assertLess(row["median_px"], 100.0)
        self.assertGreater(row["p90_frame_px"], 300.0)

    def test_the_median_is_one_vote_per_segment(self) -> None:
        """The module's contract: a long segment must not outvote a short one."""

        long_good = _seg("long", 0.5, 0.5, [10.0] * 500)
        short_bad = _seg("short", 0.5, 0.5, [90.0] * 20)
        other_bad = _seg("other", 0.5, 0.5, [90.0] * 20)
        rows = R.by_band_zone([long_good, short_bad, other_bad])
        self.assertEqual(_zone(rows, T.band_zone_of(0.5, 0.5))["median_px"], 90.0)

    def test_targets_outside_the_band_are_reported_not_dropped(self) -> None:
        rows = R.by_band_zone([_seg("far", 0.10, 0.5, [50.0] * 60)])
        outside = [r for r in rows if r.get("zone") is None]
        self.assertTrue(outside)
        self.assertIn("far", outside[0]["targets"])


def _pair(before_frames, after_frames, *, x=0.5, y=0.5):
    before = R.by_band_zone([_seg("a", x, y, before_frames), _seg("b", x, y, before_frames)])
    after = R.by_band_zone([_seg("a", x, y, after_frames), _seg("b", x, y, after_frames)])
    return before, after


class GateTests(unittest.TestCase):
    def test_a_zone_that_got_much_worse_blocks_adoption(self) -> None:
        before, after = _pair([50.0] * 40, [120.0] * 40)
        result = R.compare_band_zones(before, after)
        self.assertFalse(result["adopt"])
        self.assertIn(T.band_zone_of(0.5, 0.5), result["regressed"])

    def test_a_zone_that_improved_does_not_block(self) -> None:
        """The other half: a gate that refused every change would pass above."""

        before, after = _pair([120.0] * 40, [50.0] * 40)
        result = R.compare_band_zones(before, after)
        self.assertEqual(result["regressed"], [])

    def test_a_better_average_does_not_excuse_one_ruined_zone(self) -> None:
        """The whole reason this gate exists."""

        good_before = [_seg(f"g{i}", 0.35, 0.15, [200.0] * 30) for i in range(2)]
        bad_before = [_seg(f"b{i}", 0.65, 0.85, [40.0] * 30) for i in range(2)]
        good_after = [_seg(f"g{i}", 0.35, 0.15, [30.0] * 30) for i in range(2)]
        bad_after = [_seg(f"b{i}", 0.65, 0.85, [150.0] * 30) for i in range(2)]
        before = R.by_band_zone(good_before + bad_before)
        after = R.by_band_zone(good_after + bad_after)
        result = R.compare_band_zones(before, after)
        self.assertLess(
            np.mean([s.median_euclid_px for s in good_after + bad_after]),
            np.mean([s.median_euclid_px for s in good_before + bad_before]),
            "the fixture must improve the average or it proves nothing",
        )
        self.assertFalse(result["adopt"], "a better average excused a ruined zone")
        self.assertIn(T.band_zone_of(0.65, 0.85), result["regressed"])

    def test_both_the_fraction_and_the_floor_are_required(self) -> None:
        # +50% but only +5 px: noise in a small zone, not a regression.
        before, after = _pair([10.0] * 40, [15.0] * 40)
        self.assertEqual(R.compare_band_zones(before, after)["regressed"], [])
        # +15 px but only +3%: not a regression either.
        before, after = _pair([500.0] * 40, [515.0] * 40)
        self.assertEqual(R.compare_band_zones(before, after)["regressed"], [])

    def test_an_unverified_zone_is_never_a_pass(self) -> None:
        before = R.by_band_zone([_seg("a", 0.5, 0.5, [50.0] * 10)])
        after = R.by_band_zone([_seg("a", 0.5, 0.5, [40.0] * 10)])
        result = R.compare_band_zones(before, after)
        self.assertIn(T.band_zone_of(0.5, 0.5), result["unverified"])
        self.assertFalse(result["adopt"], "a zone with no real data counted as passing")

    def test_crossing_the_operational_limit_fails_even_under_the_percentages(self) -> None:
        """P90 crossing the half-size of the smallest target is what stops a
        person selecting it, whatever the relative numbers say."""

        # 6 of 40 frames high, so the P90 actually sits among them: 35 before
        # (under the limit), 42 after (over it). With 4 of 40 the P90 lands on
        # the low values and neither side crosses anything.
        before, after = _pair([30.0] * 34 + [35.0] * 6, [30.0] * 34 + [42.0] * 6)
        result = R.compare_band_zones(before, after, target_half_px=38.0)
        self.assertIn(T.band_zone_of(0.5, 0.5), result["regressed"])

    def test_already_over_the_limit_is_not_a_new_regression(self) -> None:
        before, after = _pair([50.0] * 40, [51.0] * 40)
        result = R.compare_band_zones(before, after, target_half_px=38.0)
        self.assertEqual(result["regressed"], [])


if __name__ == "__main__":
    unittest.main()
