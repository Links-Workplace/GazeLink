"""Paired old-vs-new scoring; no camera, no model fitting, no OS input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_recal_compare as R  # noqa: E402


def _recording(target_xy: np.ndarray, phases: list[str]) -> SimpleNamespace:
    n = len(phases)
    return SimpleNamespace(
        n_rows=n,
        target_xy=np.asarray(target_xy, dtype=np.float64),
        target_id=np.zeros(n, dtype=int),
        phase=np.asarray(phases),
        meta={"target_geometry": {"width_px": 1001, "height_px": 501}},
        rows_collecting=lambda: np.asarray(phases) == "COLLECTING",
    )


class PairedDifferenceTests(unittest.TestCase):
    """The pairing exists so a coverage difference cannot be reported as an
    accuracy difference, and so the direction of the difference is visible."""

    def setUp(self) -> None:
        self.rec = _recording(np.tile([0.5, 0.5], (4, 1)), ["COLLECTING"] * 4)
        self._eligible = R.FIT.eligible_rows
        R.FIT.eligible_rows = lambda rec: np.ones(rec.n_rows, dtype=bool)  # type: ignore[assignment]

    def tearDown(self) -> None:
        R.FIT.eligible_rows = self._eligible  # type: ignore[assignment]

    def test_a_row_only_one_model_predicted_is_excluded_from_the_pairing(self) -> None:
        old = np.array([[0.6, 0.5], [0.6, 0.5], [0.6, 0.5], [0.6, 0.5]])
        new = np.array([[0.55, 0.5], [0.55, 0.5], [np.nan, np.nan], [0.55, 0.5]])
        result = R.paired_difference(old, new, self.rec)
        self.assertEqual(result["n_paired"], 3)
        self.assertEqual(result["n_old_only"], 1)
        self.assertEqual(result["n_new_only"], 0)

    def test_a_worse_new_model_is_reported_as_worse_not_as_an_improvement(self) -> None:
        """Guards the sign of the difference: old minus new, so positive means
        the new model is better.  A flipped sign would turn a regression into
        a recovery, which is the whole question being asked."""

        old = np.array([[0.55, 0.5]] * 4)
        new = np.array([[0.70, 0.5]] * 4)
        result = R.paired_difference(old, new, self.rec)
        self.assertLess(result["median_paired_delta_px"], 0.0)
        self.assertEqual(result["fraction_of_frames_new_is_better"], 0.0)

    def test_a_better_new_model_is_reported_as_better(self) -> None:
        old = np.array([[0.70, 0.5]] * 4)
        new = np.array([[0.55, 0.5]] * 4)
        result = R.paired_difference(old, new, self.rec)
        self.assertGreater(result["median_paired_delta_px"], 0.0)
        self.assertEqual(result["fraction_of_frames_new_is_better"], 1.0)

    def test_no_shared_rows_reports_nothing_rather_than_a_fabricated_zero(self) -> None:
        old = np.array([[0.6, 0.5], [np.nan, np.nan]])
        new = np.array([[np.nan, np.nan], [0.6, 0.5]])
        rec = _recording(np.tile([0.5, 0.5], (2, 1)), ["COLLECTING"] * 2)
        result = R.paired_difference(old, new, rec)
        self.assertEqual(result["n_paired"], 0)
        self.assertNotIn("median_paired_delta_px", result)


if __name__ == "__main__":
    unittest.main()
