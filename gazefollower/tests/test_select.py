"""Screening must catch the failure the old selector could not see.

The case that motivated this module: a candidate with a good median and a
catastrophic tail. The old selector ranked on the median alone and chose it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_presets as P  # noqa: E402
import gf_select as SEL  # noqa: E402

W, H = 4096, 1152
DIAG = SEL.screen_diagonal_px(W, H)


def _candidate(name: str, offsets_px, *, n_per_target: int = 20, n_targets: int = 10,
               nan_rows: int = 0, offscreen_rows: int = 0) -> SEL.CandidateReport:
    """Build a candidate whose per-target error is dictated by ``offsets_px``."""

    rng = np.random.default_rng(0)
    targets, preds, folds = [], [], []
    for tid in range(n_targets):
        tx, ty = 0.2 + 0.06 * tid, 0.5
        off = offsets_px[tid % len(offsets_px)]
        for _ in range(n_per_target):
            targets.append([tx, ty])
            preds.append([tx + off / (W - 1) + rng.normal(scale=1e-4), ty])
            folds.append(tid)
    pred = np.array(preds)
    target = np.array(targets)
    fold = np.array(folds)
    for i in range(nan_rows):
        pred[i] = [np.nan, np.nan]
    for i in range(offscreen_rows):
        pred[-(i + 1)] = [1.4, 0.5]
    return SEL.screen_candidate(name, pred_norm=pred, target_norm=target, fold_id=fold,
                                width_px=W, height_px=H)


class GoodMedianCatastrophicTailTests(unittest.TestCase):
    """The ridge_a100 shape: fine at the centre, ruinous in the tail."""

    def test_a_good_median_does_not_rescue_a_ruinous_tail(self) -> None:
        # Nine targets at 100 px, one target 150,000 px away.
        offsets = [100.0] * 9 + [150_000.0]
        report = _candidate("ridge_like", offsets)
        self.assertLess(report.stats["median_px"], 200)  # the median looks fine
        self.assertFalse(report.passed)
        rules = {f.rule for f in report.failures}
        self.assertTrue({"max_p90_frac", "max_p99_frac", "max_error_frac"} & rules, report.reasons)

    def test_one_catastrophic_fold_is_enough(self) -> None:
        # A fold-level failure that the tail thresholds alone would let through:
        # one target off by 40 % of the diagonal, the rest excellent.
        offsets = [30.0] * 9 + [0.40 * DIAG]
        report = _candidate("one_bad_target", offsets, n_per_target=5)
        self.assertIn("max_fold_median_frac", {f.rule for f in report.failures}, report.reasons)
        worst = max(report.per_fold, key=lambda f: f["median_frac"])
        self.assertEqual(report.stats["worst_fold"], worst["fold"])

    def test_offscreen_predictions_are_rejected(self) -> None:
        report = _candidate("wanderer", [50.0], n_per_target=20, offscreen_rows=30)
        self.assertIn("max_offscreen_rate", {f.rule for f in report.failures}, report.reasons)
        self.assertGreater(report.stats["offscreen_rate"], 0.02)


class NonFiniteTests(unittest.TestCase):
    def test_missing_predictions_count_against_availability(self) -> None:
        report = _candidate("lossy", [50.0], n_per_target=20, nan_rows=40)
        self.assertIn("valid_rate", {f.rule for f in report.failures}, report.reasons)
        self.assertLess(report.stats["valid_rate"], 0.95)
        self.assertEqual(report.stats["n_eligible"], 200)

    def test_all_predictions_missing_is_rejected_not_crashed(self) -> None:
        pred = np.full((50, 2), np.nan)
        target = np.tile([0.5, 0.5], (50, 1))
        report = SEL.screen_candidate("dead", pred_norm=pred, target_norm=target,
                                      fold_id=np.zeros(50, dtype=int), width_px=W, height_px=H)
        self.assertFalse(report.passed)
        self.assertIn("no_valid_predictions", {f.rule for f in report.failures})

    def test_a_failed_fit_is_reported_not_ranked(self) -> None:
        report = SEL.screen_candidate("broken", pred_norm=np.zeros((0, 2)), target_norm=np.zeros((0, 2)),
                                      fold_id=np.zeros(0, dtype=int), width_px=W, height_px=H,
                                      fit_failed=True, fit_error="SVM training raised")
        self.assertFalse(report.passed)
        self.assertEqual(report.failures[0].rule, "fit_failed")
        self.assertIn("SVM training raised", report.failures[0].detail)

    def test_no_rows_is_rejected(self) -> None:
        report = SEL.screen_candidate("empty", pred_norm=np.zeros((0, 2)), target_norm=np.zeros((0, 2)),
                                      fold_id=np.zeros(0, dtype=int), width_px=W, height_px=H)
        self.assertFalse(report.passed)
        self.assertIn("no_rows", {f.rule for f in report.failures})


class AcceptableCandidateTests(unittest.TestCase):
    def test_a_well_behaved_candidate_passes(self) -> None:
        report = _candidate("good", [60.0, 90.0, 45.0])
        self.assertTrue(report.passed, report.reasons)
        self.assertLess(report.stats["p99_px"], 0.25 * DIAG)
        self.assertEqual(report.stats["offscreen_rate"], 0.0)

    def test_thresholds_are_expressed_as_fractions_of_the_diagonal(self) -> None:
        # The same behaviour must pass or fail identically on a different
        # resolution, which pixel thresholds would not guarantee.
        offsets = [0.05 * DIAG]
        big = _candidate("big", offsets)
        small_diag = SEL.screen_diagonal_px(1920, 1080)
        rng = np.random.default_rng(1)
        n = 200
        target = np.tile([0.5, 0.5], (n, 1))
        pred = target + np.column_stack([np.full(n, 0.05 * small_diag / (1920 - 1)), np.zeros(n)])
        pred += rng.normal(scale=1e-6, size=pred.shape)
        small = SEL.screen_candidate("small", pred_norm=pred, target_norm=target,
                                     fold_id=np.arange(n) % 10, width_px=1920, height_px=1080)
        self.assertEqual(big.passed, small.passed)
        self.assertAlmostEqual(big.stats["median_frac"], small.stats["median_frac"], places=3)


class SelectionTests(unittest.TestCase):
    def test_screening_comes_before_ranking(self) -> None:
        bad = _candidate("best_median_worst_tail", [10.0] * 9 + [150_000.0])
        ok = _candidate("honest", [80.0])
        outcome = SEL.select([bad, ok])
        self.assertLess(bad.stats["median_px"], ok.stats["median_px"])  # would have won on median
        self.assertEqual(outcome.selected, "honest")
        self.assertFalse(outcome.refused)

    def test_no_acceptable_calibration_rather_than_least_bad(self) -> None:
        outcome = SEL.select([
            _candidate("bad_a", [10.0] * 9 + [150_000.0]),
            _candidate("bad_b", [20.0] * 9 + [200_000.0]),
        ])
        self.assertTrue(outcome.refused)
        self.assertIsNone(outcome.selected)
        self.assertIn("no acceptable calibration", outcome.preset_note)

    def test_ranking_among_survivors_uses_the_named_metric(self) -> None:
        outcome = SEL.select([_candidate("worse", [200.0]), _candidate("better", [60.0])])
        self.assertEqual(outcome.selected, "better")
        self.assertEqual(outcome.ranking_metric, "median_px")

    def test_nan_metric_never_wins(self) -> None:
        good = _candidate("good", [70.0])
        weird = _candidate("weird", [70.0])
        weird.stats["median_px"] = float("nan")
        self.assertEqual(SEL.select([weird, good]).selected, "good")


class PresetSelectionTests(unittest.TestCase):
    def test_a_preset_is_used_when_it_passes(self) -> None:
        outcome = SEL.select([_candidate("chosen", [70.0]), _candidate("other", [50.0])], preset="chosen")
        self.assertEqual(outcome.selected, "chosen")
        self.assertEqual(outcome.mode, "preset")
        self.assertIn("passed screening", outcome.preset_note)

    def test_a_preset_that_fails_screening_is_refused_not_forced(self) -> None:
        outcome = SEL.select([_candidate("named", [10.0] * 9 + [150_000.0]), _candidate("fine", [70.0])],
                             preset="named")
        self.assertTrue(outcome.refused)
        self.assertIn("failed screening", outcome.preset_note)

    def test_an_unknown_preset_is_reported(self) -> None:
        outcome = SEL.select([_candidate("a", [70.0])], preset="nope")
        self.assertTrue(outcome.refused)
        self.assertIn("not among the candidates", outcome.preset_note)


class ReportTests(unittest.TestCase):
    def test_report_names_the_reason_for_every_rejection(self) -> None:
        outcome = SEL.select([_candidate("bad", [10.0] * 9 + [150_000.0]), _candidate("good", [70.0])])
        text = SEL.format_selection(outcome)
        self.assertIn("good", text)
        self.assertIn("Rejected", text)
        self.assertIn("bad", text)
        self.assertIn("diagonal", text)

    def test_refusal_is_stated_loudly(self) -> None:
        outcome = SEL.select([_candidate("bad", [10.0] * 9 + [150_000.0])])
        self.assertIn("NO ACCEPTABLE CALIBRATION", SEL.format_selection(outcome))


class PresetRegistryTests(unittest.TestCase):
    def test_the_candidate_is_preserved_with_its_split_labelled(self) -> None:
        p = P.get("central-band-svr")
        self.assertEqual(p.config.feature_scaling, "zscore")
        self.assertEqual(p.config.label_scaling, "zscore")
        self.assertEqual(p.config.C, 100.0)
        self.assertEqual(p.config.gamma, 0.0005)
        self.assertEqual(p.evidence_split, "TUNE")
        self.assertIn("never been scored on an independent", p.caveat)

    def test_the_negative_control_is_kept_and_labelled(self) -> None:
        p = P.get("central-band-ridge")
        self.assertIn("NEGATIVE CONTROL", p.caveat.upper())

    def test_presets_ride_along_in_the_sweep(self) -> None:
        names = {c.name for c in P.sweep_with_presets()}
        self.assertIn("central-band-svr", names)
        self.assertIn("lib-default", names)

    def test_unknown_preset_raises_with_the_known_list(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            P.get("does-not-exist")
        self.assertIn("central-band-svr", str(ctx.exception))

    def test_describe_states_that_a_preset_is_not_a_fitted_model(self) -> None:
        text = P.describe("central-band-svr")
        self.assertIn("fresh calibration", text)
        self.assertIn("TUNE", text)


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
