"""Pure parts of the pooled-model experiment: grouping, balance, corrections, verdict.

Synthetic data only. No camera, no recordings, no model fitting.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_pool_experiment as E  # noqa: E402

S = 1_000_000_000


def item(name: str, protocol: str, start_s: float, end_s: float, day: str = "20260907") -> E.Item:
    return E.Item(Path(name), protocol, int(start_s * S), int(end_s * S), ("k",), day)


class Grouping(unittest.TestCase):
    def test_recordings_close_in_time_share_a_sitting_whatever_their_ids(self) -> None:
        items = [
            item("round3", "A", 0, 60),
            item("round3", "T1", 101, 160),  # different session id in real data
            item("round5", "A", 300, 360),
            item("round7", "T1", 2000, 2060),  # > 10 min later
        ]
        groups = E.sittings(items)
        self.assertEqual(
            [[i.directory.name for i in g] for g in groups],
            [["round3", "round3", "round5"], ["round7"]],
        )

    def test_round_numbers_do_not_decide_order(self) -> None:
        items = [item("round30", "T1", 5000, 5060), item("round4", "A", 4800, 4900)]
        groups = E.sittings(items)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0].directory.name, "round4")

    def test_days_group_by_day(self) -> None:
        items = [item("a", "T1", 0, 1, "20260907"), item("b", "T1", 9e5, 9e5 + 1, "20260908")]
        self.assertEqual(len(E.days(items)), 2)


class PositionBalance(unittest.TestCase):
    def test_overrepresented_positions_are_capped_and_rare_ones_kept(self) -> None:
        Y = np.array([[0.0, 0.0]] * 300 + [[10.0, 0.0]] * 40 + [[20.0, 5.0]] * 50)
        src = np.array([0] * 150 + [1] * 150 + [2] * 40 + [3] * 50)
        keep = E.position_balanced_rows(Y, src)
        counts = {tuple(p): int(np.sum(np.all(Y[keep] == p, axis=1))) for p in np.unique(Y, axis=0)}
        self.assertEqual(counts[(10.0, 0.0)], 40)  # below the median cap: all kept
        self.assertEqual(counts[(0.0, 0.0)], 50)  # capped at the median count
        # The capped position draws evenly from both recordings that show it.
        kept_src = src[keep][np.all(Y[keep] == [0.0, 0.0], axis=1)]
        self.assertEqual(int(np.sum(kept_src == 0)), 25)
        self.assertEqual(int(np.sum(kept_src == 1)), 25)

    def test_deterministic(self) -> None:
        Y = np.repeat(np.arange(5)[:, None] * [1.0, 0.0], [10, 20, 30, 40, 50], axis=0)
        src = np.arange(len(Y)) % 3
        np.testing.assert_array_equal(
            E.position_balanced_rows(Y, src), E.position_balanced_rows(Y, src)
        )


class Corrections(unittest.TestCase):
    def test_offset_is_shrunk_toward_zero_by_the_prior(self) -> None:
        support = np.array([[0.3, 0.3, 0.4, 0.3]] * 9)  # every target reads +0.1 in x
        ox, oy = E.ridge_offset(support, prior_targets=9.0)
        self.assertAlmostEqual(ox, 0.05)  # 9 * 0.1 / (9 + 9)
        self.assertAlmostEqual(oy, 0.0)

    def test_no_support_means_no_correction(self) -> None:
        self.assertEqual(E.ridge_offset(np.zeros((0, 4))), (0.0, 0.0))
        self.assertEqual(E.ridge_x_gain(np.zeros((0, 4))), (0.0, 1.0))

    def test_gain_prior_keeps_three_points_from_fitting_a_full_affine(self) -> None:
        # Three points compressed to half the spread: an unregularised fit
        # would give gain 2; the prior must pull it well back toward 1.
        support = np.array([[0.3, 0.5, 0.4, 0.5], [0.5, 0.5, 0.5, 0.5], [0.7, 0.5, 0.6, 0.5]])
        _a, g = E.ridge_x_gain(support, prior_targets=9.0)
        self.assertGreater(g, 1.0)
        self.assertLess(g, 1.1)


class Verdict(unittest.TestCase):
    ZONES_OK = {"zones": [{"zone": z, "verdict": "ok"} for z in range(9)]}

    def test_consistent_improvement_passes(self) -> None:
        v = E.verdict([10.0, 12.0, 8.0, 9.0, 11.0, 7.0], self.ZONES_OK, [1.0, 2.0])
        self.assertEqual(v["outcome"], "IMPROVEMENT")

    def test_a_worse_zone_is_a_regression_even_when_the_median_improves(self) -> None:
        zones = {"zones": [{"zone": 2, "verdict": "WORSE"}]}
        v = E.verdict([10.0, 12.0, 8.0, 9.0], zones, [1.0])
        self.assertEqual(v["outcome"], "REGRESSION")

    def test_zone_four_is_never_used_to_fail(self) -> None:
        zones = {"zones": [{"zone": 4, "verdict": "WORSE"}]}
        v = E.verdict([10.0, 12.0, 8.0, 9.0], zones, [1.0])
        self.assertEqual(v["outcome"], "IMPROVEMENT")

    def test_mixed_folds_are_insufficient(self) -> None:
        v = E.verdict([10.0, -12.0, 8.0, -9.0, 3.0], self.ZONES_OK, [0.0])
        self.assertEqual(v["outcome"], "INSUFFICIENT")

    def test_bottom_row_getting_worse_blocks_improvement(self) -> None:
        v = E.verdict([10.0, 12.0, 8.0, 9.0], self.ZONES_OK, [-5.0, -3.0])
        self.assertEqual(v["outcome"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
