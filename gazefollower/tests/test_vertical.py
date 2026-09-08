"""gf_vertical on synthetic fixtures.

The point of these tests is that the numbers the M2-05 investigation reports
mean what they say:

* slope and bias are recovered EXACTLY from data with a planted slope and
  bias, so a compression of 0.74 is a real compression and not an artefact of
  the metric;
* degenerate input -- empty, all-lost, one target position -- is REPORTED as
  degenerate and never silently averaged into a number that reads as a
  measurement;
* the selection rule cannot see a test protocol, rejects a candidate that buys
  vertical error with horizontal error, and rejects one that buys it by
  flattening the slope further;
* cross-validation folds hold out whole targets, so memorising one fixation's
  frames cannot be reported as generalisation.

Everything here is synthetic and deterministic. No recording, camera, model
file or OS input is touched.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_fit as F  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_schema as S  # noqa: E402
import gf_vertical as V  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
GEOM = {
    "width_px": 4096,
    "height_px": 1152,
    "dpi_scale": 1.25,
    "screen_id": "test",
    "orientation": "LANDSCAPE",
}
WIDTH, HEIGHT = GEOM["width_px"], GEOM["height_px"]


# --- fixtures -----------------------------------------------------------------


def planted(
    positions: list[float], per_target: int, slope: float, offset: float, axis: str = "y"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Targets and predictions with an EXACTLY known slope and offset.

    ``pred = slope * target + offset`` on the scored axis; the other axis is
    filled with the target itself so it never influences the axis under test.
    """

    col = 0 if axis == "x" else 1
    targets = np.zeros((len(positions) * per_target, 2))
    preds = np.zeros_like(targets)
    tid = np.zeros(len(positions) * per_target, dtype=int)
    row = 0
    for index, value in enumerate(positions):
        for _ in range(per_target):
            targets[row, col] = value
            targets[row, 1 - col] = 0.5
            preds[row, col] = slope * value + offset
            preds[row, 1 - col] = 0.5
            tid[row] = index
            row += 1
    return targets, preds, tid


def make_recording(
    protocol: str,
    points: list[tuple[float, float]],
    per_target: int,
    *,
    head_valid_every: int = 1,
    pnp_valid_every: int = 1,
    dim: int = 6,
) -> S.Recording:
    """A tiny recording whose features carry both target axes linearly."""

    meta = {
        "rig": RIG.to_dict(),
        "targets": [
            {"index": i, "name": f"P{i}", "screen_position": {"x": x, "y": y}}
            for i, (x, y) in enumerate(points)
        ],
        "target_geometry": GEOM,
        "session": "synthetic",
    }
    builder = S.RecordingBuilder(protocol, 0, meta)
    calibration = protocol in S.CALIBRATION_PROTOCOLS
    rng = np.random.default_rng(11)
    seq = 0
    for tid, (nx, ny) in enumerate(points):
        for _ in range(per_target):
            features = rng.normal(scale=0.05, size=dim)
            features[0] = 2.0 * nx
            features[1] = -1.5 * ny
            head = np.zeros(6)
            head[H.HEAD6_NAMES.index("pitch_a")] = -0.6 + 0.3 * ny
            head[H.HEAD6_NAMES.index("eye_mid_y")] = 0.5
            head[H.HEAD6_NAMES.index("iod_norm")] = 0.15
            builder.append(
                frame_seq=seq,
                timestamp_ns=seq * 33_000_000,
                elapsed_ms=1500.0,
                target_id=tid,
                block=0,
                phase=S.PHASE_COLLECT if calibration else S.PHASE_COLLECTING,
                target_xy=(nx, ny),
                label_cm=RIG.norm_to_label_cm(nx, ny),
                features=features,
                head=head if (seq % head_valid_every == 0) else None,
                pnp_deg=(1.0, 20.0 + 5.0 * ny, 0.5) if (seq % pnp_valid_every == 0) else None,
                raw_cm=(0.0, 0.0),
                openness=(100.0, 100.0),
                tracking_state="SUCCESS",
                gaze_status=True,
                accepted=calibration,
            )
            seq += 1
    return builder.freeze()


GRID9 = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.3, 0.5, 0.7)]


def sweep_entry(
    name: str, *, y_slope: float, y_median: float, x_median: float, coverage: float = 1.0
) -> dict:
    """A minimal sweep entry carrying only what :func:`gf_vertical.select` reads."""

    def axis(slope: float, median: float) -> dict:
        return {
            "slope": slope,
            "intercept": 0.0,
            "bias_px": 0.0,
            "median_abs_px": median,
            "p90_abs_px": median * 2,
            "coverage": coverage,
            "n_eligible": 100,
            "n_valid": int(100 * coverage),
            "n_target_positions": 3,
            "axis": "x",
            "degenerate": None,
        }

    return {"name": name, "tune": {"x": axis(1.0, x_median), "y": axis(y_slope, y_median)}}


# --- slope and bias are recovered exactly -------------------------------------


class TestFitSlopeBias(unittest.TestCase):
    def test_recovers_planted_slope_and_intercept(self) -> None:
        targets = [0.1, 0.4, 0.5, 0.9]
        for slope, intercept in ((1.0, 0.0), (0.74, 0.13), (-0.5, 1.0), (2.0, -0.25)):
            preds = [slope * t + intercept for t in targets]
            got_slope, got_intercept = V.fit_slope_bias(targets, preds)
            self.assertAlmostEqual(got_slope, slope, places=12, msg=f"slope {slope}")
            self.assertAlmostEqual(
                got_intercept, intercept, places=12, msg=f"intercept {intercept}"
            )

    def test_compression_reads_as_slope_below_one(self) -> None:
        targets = [0.05, 0.5, 0.95]
        preds = [0.74 * t + 0.13 for t in targets]
        slope, _ = V.fit_slope_bias(targets, preds)
        self.assertLess(slope, 1.0)
        self.assertAlmostEqual(slope, 0.74, places=12)

    def test_single_point_and_single_position_are_not_measurable(self) -> None:
        self.assertEqual(V.fit_slope_bias([0.5], [0.5]), (None, None))
        self.assertEqual(V.fit_slope_bias([], []), (None, None))
        # Four points all at the same target: a perfect model and a constant
        # one are indistinguishable, so the answer must be "unknown", not 0.0.
        self.assertEqual(V.fit_slope_bias([0.5, 0.5, 0.5, 0.5], [0.1, 0.9, 0.4, 0.6]), (None, None))

    def test_ignores_non_finite_pairs(self) -> None:
        slope, intercept = V.fit_slope_bias([0.1, 0.5, 0.9, 0.3], [0.1, 0.5, 0.9, float("nan")])
        self.assertAlmostEqual(slope, 1.0, places=12)
        self.assertAlmostEqual(intercept, 0.0, places=12)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            V.fit_slope_bias([0.1, 0.2], [0.1])


class TestAxisReport(unittest.TestCase):
    def test_pure_offset_is_all_bias_and_slope_one(self) -> None:
        offset = 0.1  # normalised
        targets, preds, tid = planted([0.1, 0.5, 0.9], 20, slope=1.0, offset=offset)
        report = V.axis_report("y", targets, preds, tid, HEIGHT)
        self.assertIsNone(report.degenerate)
        self.assertAlmostEqual(report.slope, 1.0, places=12)
        self.assertAlmostEqual(report.bias_px, offset * (HEIGHT - 1), places=9)
        self.assertAlmostEqual(report.median_abs_px, offset * (HEIGHT - 1), places=9)
        self.assertAlmostEqual(report.p90_abs_px, offset * (HEIGHT - 1), places=9)
        self.assertEqual(report.coverage, 1.0)
        self.assertEqual(report.n_target_positions, 3)

    def test_planted_compression_is_reported_as_compression(self) -> None:
        targets, preds, tid = planted([0.1, 0.5, 0.9], 15, slope=0.74, offset=0.13)
        report = V.axis_report("y", targets, preds, tid, HEIGHT)
        self.assertAlmostEqual(report.slope, 0.74, places=12)
        # The bias is the median signed error, which for a symmetric layout is
        # the error at the middle target: 0.74*0.5 + 0.13 - 0.5.
        self.assertAlmostEqual(report.bias_px, (0.74 * 0.5 + 0.13 - 0.5) * (HEIGHT - 1), places=9)

    def test_x_and_y_are_read_from_the_right_column(self) -> None:
        targets, preds, tid = planted([0.2, 0.6], 10, slope=0.5, offset=0.0, axis="x")
        x_report = V.axis_report("x", targets, preds, tid, WIDTH)
        y_report = V.axis_report("y", targets, preds, tid, HEIGHT)
        self.assertAlmostEqual(x_report.slope, 0.5, places=12)
        # The y column was left equal to its target, so y must read as perfect.
        self.assertAlmostEqual(y_report.bias_px, 0.0, places=9)

    def test_empty_input_is_reported_not_averaged(self) -> None:
        report = V.axis_report(
            "y", np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=int), HEIGHT
        )
        self.assertEqual(report.degenerate, "no_eligible_rows")
        for value in (
            report.slope,
            report.bias_px,
            report.median_abs_px,
            report.p90_abs_px,
            report.coverage,
        ):
            self.assertIsNone(value)
        self.assertEqual(report.n_valid, 0)

    def test_all_lost_predictions_are_reported_not_averaged(self) -> None:
        targets, preds, tid = planted([0.1, 0.9], 10, slope=1.0, offset=0.0)
        preds[:] = np.nan
        report = V.axis_report("y", targets, preds, tid, HEIGHT)
        self.assertEqual(report.degenerate, "no_valid_predictions")
        self.assertIsNone(report.median_abs_px)
        self.assertIsNone(report.bias_px)
        self.assertIsNone(report.slope)
        self.assertEqual(report.coverage, 0.0)
        self.assertEqual(report.n_eligible, 20)

    def test_single_target_position_reports_error_but_no_slope(self) -> None:
        targets, preds, tid = planted([0.5], 20, slope=1.0, offset=0.05)
        report = V.axis_report("y", targets, preds, tid, HEIGHT)
        self.assertEqual(report.degenerate, "single_target_position")
        self.assertIsNone(report.slope)
        self.assertAlmostEqual(report.bias_px, 0.05 * (HEIGHT - 1), places=9)

    def test_lost_rows_lower_coverage_without_diluting_the_error(self) -> None:
        targets, preds, tid = planted([0.1, 0.5, 0.9], 20, slope=1.0, offset=0.1)
        preds[::2, 1] = np.nan  # lose half the rows
        report = V.axis_report("y", targets, preds, tid, HEIGHT)
        self.assertEqual(report.n_eligible, 60)
        self.assertEqual(report.n_valid, 30)
        self.assertAlmostEqual(report.coverage, 0.5, places=12)
        # The surviving rows still carry the full planted offset.
        self.assertAlmostEqual(report.median_abs_px, 0.1 * (HEIGHT - 1), places=9)

    def test_rejects_bad_arguments(self) -> None:
        targets, preds, tid = planted([0.1, 0.9], 4, slope=1.0, offset=0.0)
        with self.assertRaises(ValueError):
            V.axis_report("z", targets, preds, tid, HEIGHT)
        with self.assertRaises(ValueError):
            V.axis_report("y", targets, preds, tid[:-1], HEIGHT)
        with self.assertRaises(ValueError):
            V.axis_report("y", targets, preds, tid, 1)

    def test_agrees_with_gf_fit_evaluate(self) -> None:
        """The wiring test: same input, same slope/bias as the existing fitter.

        If this drifts, numbers from gf_vertical stop being comparable with
        phase0.json and the whole comparison is void.
        """

        rng = np.random.default_rng(3)
        targets, preds, tid = planted([0.12, 0.4, 0.77], 25, slope=0.8, offset=0.06)
        preds[:, 1] += rng.normal(scale=0.01, size=preds.shape[0])
        preds[:, 0] += rng.normal(scale=0.01, size=preds.shape[0])
        mine_y = V.axis_report("y", targets, preds, tid, HEIGHT)
        mine_x = V.axis_report("x", targets, preds, tid, WIDTH)
        theirs = F.evaluate(preds, targets, tid, WIDTH, HEIGHT)
        self.assertAlmostEqual(mine_y.slope, theirs.y.slope, places=10)
        self.assertAlmostEqual(mine_y.bias_px, theirs.y.bias_px, places=8)
        self.assertAlmostEqual(mine_y.median_abs_px, theirs.y.median_abs_px, places=8)
        self.assertAlmostEqual(mine_y.p90_abs_px, theirs.y.p90_abs_px, places=8)
        self.assertAlmostEqual(mine_x.slope, theirs.x.slope, places=10)
        self.assertAlmostEqual(mine_x.bias_px, theirs.x.bias_px, places=8)


class TestRSquared(unittest.TestCase):
    def test_known_values(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(V.r_squared(y, y), 1.0, places=12)
        self.assertAlmostEqual(V.r_squared(y, np.full(4, y.mean())), 0.0, places=12)
        self.assertLess(V.r_squared(y, y[::-1]), 0.0)

    def test_undefined_cases_return_none(self) -> None:
        self.assertIsNone(V.r_squared([1.0], [1.0]))
        self.assertIsNone(V.r_squared([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))


# --- design matrices ----------------------------------------------------------


class TestDesign(unittest.TestCase):
    def setUp(self) -> None:
        self.rec = make_recording("A", GRID9, 5)

    def test_extra_columns_appear_in_the_requested_order(self) -> None:
        mask = self.rec.rows_accepted()
        names = ("iod_norm", "pnp_pitch", "pitch_a")
        extras = V.extra_matrix(self.rec, mask, names)
        self.assertEqual(extras.shape, (int(mask.sum()), 3))
        np.testing.assert_allclose(
            extras[:, 0], self.rec.head[mask][:, H.HEAD6_NAMES.index("iod_norm")]
        )
        np.testing.assert_allclose(extras[:, 1], self.rec.pnp_deg[mask][:, 1])
        np.testing.assert_allclose(
            extras[:, 2], self.rec.head[mask][:, H.HEAD6_NAMES.index("pitch_a")]
        )

    def test_no_extras_gives_the_bare_embedding(self) -> None:
        mask = self.rec.rows_accepted()
        base = V.design(self.rec, mask, ())
        self.assertEqual(base.shape, (int(mask.sum()), self.rec.feature_dim))
        self.assertEqual(V.extra_matrix(self.rec, mask, ()).shape, (int(mask.sum()), 0))

    def test_unknown_extra_column_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            V.extra_matrix(self.rec, self.rec.rows_accepted(), ("gaze_z",))

    def test_design_width_matches_the_requested_columns(self) -> None:
        mask = self.rec.rows_accepted()
        wide = V.design(self.rec, mask, ("pitch_a", "pnp_pitch"))
        self.assertEqual(wide.shape[1], self.rec.feature_dim + 2)

    def test_usable_rows_drops_invalid_head_and_pnp(self) -> None:
        rec = make_recording("A", GRID9, 4, head_valid_every=2, pnp_valid_every=3)
        accepted = rec.rows_accepted()
        self.assertEqual(V.usable_rows(rec, accepted, ()).sum(), accepted.sum())
        head_only = V.usable_rows(rec, accepted, ("pitch_a",))
        self.assertLess(head_only.sum(), accepted.sum())
        np.testing.assert_array_equal(head_only, accepted & rec.rows_head_valid())
        both = V.usable_rows(rec, accepted, ("pitch_a", "pnp_pitch"))
        self.assertLessEqual(both.sum(), head_only.sum())
        # Every surviving row must actually have finite extras.
        self.assertTrue(np.all(np.isfinite(V.extra_matrix(rec, both, ("pitch_a", "pnp_pitch")))))


class TestCompose(unittest.TestCase):
    def test_takes_x_from_the_first_and_y_from_the_second(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[9.0, 8.0], [7.0, 6.0]])
        np.testing.assert_array_equal(V.compose(a, b), np.array([[1.0, 8.0], [3.0, 6.0]]))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            V.compose(np.zeros((3, 2)), np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            V.compose(np.zeros(3), np.zeros(3))


# --- the selection rule -------------------------------------------------------


class TestSelection(unittest.TestCase):
    def base(self) -> dict:
        return sweep_entry(V.BASELINE, y_slope=0.73, y_median=106.0, x_median=53.0)

    def test_picks_the_lowest_vertical_error_among_survivors(self) -> None:
        sweep = [
            self.base(),
            sweep_entry("good", y_slope=0.90, y_median=70.0, x_median=55.0),
            sweep_entry("also_ok", y_slope=0.95, y_median=80.0, x_median=54.0),
        ]
        self.assertEqual(V.select(sweep)["selected"], "good")

    def test_rejects_a_candidate_that_pays_for_y_with_x(self) -> None:
        sweep = [
            self.base(),
            sweep_entry("y_at_any_price", y_slope=1.0, y_median=40.0, x_median=190.0),
            sweep_entry("balanced", y_slope=0.90, y_median=90.0, x_median=55.0),
        ]
        outcome = V.select(sweep)
        self.assertEqual(outcome["selected"], "balanced")
        self.assertEqual(outcome["rejected"].get("x_regression"), 1)

    def test_rejects_a_lower_error_that_flattens_the_slope_further(self) -> None:
        """A smaller median with a worse slope is not an improvement here."""

        sweep = [
            self.base(),
            sweep_entry("flatter", y_slope=0.50, y_median=60.0, x_median=53.0),
            sweep_entry("steeper", y_slope=0.85, y_median=95.0, x_median=53.0),
        ]
        outcome = V.select(sweep)
        self.assertEqual(outcome["selected"], "steeper")
        self.assertEqual(outcome["rejected"].get("y_slope_regression"), 1)

    def test_rejects_low_coverage(self) -> None:
        sweep = [
            self.base(),
            sweep_entry("lossy", y_slope=1.0, y_median=10.0, x_median=50.0, coverage=0.5),
        ]
        outcome = V.select(sweep)
        self.assertEqual(outcome["selected"], V.BASELINE)
        self.assertIn("coverage_x", outcome["rejected"])

    def test_refuses_when_nothing_passes(self) -> None:
        sweep = [
            sweep_entry(V.BASELINE, y_slope=0.73, y_median=106.0, x_median=53.0, coverage=0.1),
            sweep_entry("other", y_slope=0.2, y_median=400.0, x_median=900.0, coverage=0.1),
        ]
        outcome = V.select(sweep)
        self.assertTrue(outcome["refused"])
        self.assertIsNone(outcome["selected"])

    def test_selection_cannot_see_a_test_protocol(self) -> None:
        """Planting a spectacular T1 result must not move the choice."""

        sweep = [
            self.base(),
            sweep_entry("good", y_slope=0.90, y_median=70.0, x_median=55.0),
            sweep_entry("tempting", y_slope=0.88, y_median=99.0, x_median=55.0),
        ]
        plain = V.select(sweep)["selected"]
        for entry in sweep:
            entry["t1"] = {"x": {"median_abs_px": 0.0}, "y": {"median_abs_px": 0.0, "slope": 1.0}}
        sweep[2]["t1"]["y"]["median_abs_px"] = -1000.0
        self.assertEqual(V.select(sweep)["selected"], plain)

    def test_empty_sweep_is_refused_loudly(self) -> None:
        with self.assertRaises(ValueError):
            V.select([])

    def test_missing_baseline_is_refused_loudly(self) -> None:
        with self.assertRaises(ValueError):
            V.select([sweep_entry("other", y_slope=1.0, y_median=10.0, x_median=10.0)])

    def test_per_axis_picks_each_axis_independently(self) -> None:
        sweep = [
            self.base(),
            sweep_entry("x_specialist", y_slope=0.4, y_median=300.0, x_median=20.0),
            sweep_entry("y_specialist", y_slope=1.0, y_median=60.0, x_median=400.0),
        ]
        outcome = V.select_per_axis(sweep)
        self.assertEqual(outcome["x"], "x_specialist")
        self.assertEqual(outcome["y"], "y_specialist")


# --- cross-validation ---------------------------------------------------------


class TestCrossValidation(unittest.TestCase):
    def test_folds_hold_out_whole_targets_rows_and_columns(self) -> None:
        rec = make_recording("A", GRID9, 3)
        mask = rec.rows_accepted()
        self.assertEqual(len(V.target_folds(rec, mask, None)), 9)
        rows = V.target_folds(rec, mask, "y")
        cols = V.target_folds(rec, mask, "x")
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(cols), 3)
        self.assertTrue(all(len(f) == 3 for f in rows + cols))
        # Every target appears exactly once in each scheme.
        self.assertEqual(sorted(i for f in rows for i in f), list(range(9)))
        self.assertEqual(sorted(i for f in cols for i in f), list(range(9)))

    def test_grouped_cv_predicts_every_row_out_of_fold(self) -> None:
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = X[:, 0] * 0.5
        groups = np.repeat(np.arange(4), 5)
        out = V.grouped_cv(
            X, y, groups, [[0], [1], [2], [3]], lambda a, b, c: V.ridge_predict(a, b, c, 1e-6)
        )
        self.assertTrue(np.all(np.isfinite(out)))

    def test_rows_in_no_fold_stay_nan(self) -> None:
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = X[:, 0] * 0.5
        groups = np.repeat(np.arange(4), 5)
        out = V.grouped_cv(X, y, groups, [[0], [1]], lambda a, b, c: V.ridge_predict(a, b, c, 1e-6))
        self.assertTrue(np.all(np.isfinite(out[:10])))
        self.assertTrue(np.all(np.isnan(out[10:])))

    def test_fold_with_a_constant_training_label_is_skipped(self) -> None:
        X = np.random.default_rng(0).normal(size=(20, 3))
        y = np.repeat([1.0, 1.0, 1.0, 2.0], 5)
        groups = np.repeat(np.arange(4), 5)
        # Holding out group 3 leaves a constant label, so that fold is skipped;
        # holding out group 0 leaves both values, so it is predicted.
        out = V.grouped_cv(X, y, groups, [[3], [0]], lambda a, b, c: V.ridge_predict(a, b, c, 1.0))
        self.assertTrue(np.all(np.isnan(out[15:])))
        self.assertTrue(np.all(np.isfinite(out[:5])))


class TestRegressors(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.X = rng.normal(size=(120, 4))
        self.y = 3.0 * self.X[:, 0] - 2.0 * self.X[:, 1] + 0.5

    def test_ridge_recovers_a_linear_relation(self) -> None:
        pred = V.ridge_predict(self.X[:100], self.y[:100], self.X[100:], 1e-8)
        np.testing.assert_allclose(pred, self.y[100:], rtol=1e-4, atol=1e-4)

    def test_kernel_ridge_and_knn_track_the_relation(self) -> None:
        for name in ("kernel_ridge_g1", "kernel_ridge_g0.1", "knn_k15"):
            pred = V.REGRESSORS[name](self.X[:100], self.y[:100], self.X[100:])
            self.assertEqual(pred.shape, (20,))
            self.assertGreater(V.r_squared(self.y[100:], pred), 0.0, msg=name)

    def test_knn_never_asks_for_more_neighbours_than_it_has(self) -> None:
        pred = V.knn_predict(self.X[:3], self.y[:3], self.X[100:102], 15)
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(np.all(np.isfinite(pred)))


# --- candidate plumbing -------------------------------------------------------


class TestCandidates(unittest.TestCase):
    def test_grid_contains_the_baseline_and_the_asked_for_axes(self) -> None:
        grid = V.build_grid()
        names = {c.name for c in grid}
        self.assertIn(V.BASELINE, names)
        scalings = {(c.feature_scaling, c.label_scaling) for c in grid}
        self.assertEqual(
            scalings,
            {("none", "none"), ("none", "zscore"), ("zscore", "none"), ("zscore", "zscore")},
        )
        extras = {c.extra_names for c in grid}
        for wanted in (("pitch_a",), ("eye_mid_y",), ("iod_norm",), ("pnp_pitch",)):
            self.assertIn(wanted, extras)
        self.assertTrue(any(c.kind == "ridge" for c in grid))
        self.assertTrue({c.P for c in grid} >= {0.001, 0.1})

    def test_fit_and_predict_round_trip_on_a_synthetic_recording(self) -> None:
        rec_a = make_recording("A", GRID9, 12)
        rec_t = make_recording("T1", [(0.4, 0.3), (0.6, 0.7)], 8)
        candidate = V.Candidate(
            "probe", "zscore", "zscore", 10.0, "auto", 0.001, extra_names=("pitch_a",)
        )
        fitted = V.fit_candidate(rec_a, candidate, RIG)
        pred = fitted.predict_norm(rec_t, RIG)
        self.assertEqual(pred.shape, (rec_t.n_rows, 2))
        self.assertTrue(np.all(np.isfinite(pred)))
        reports = V.score_axes(rec_t, pred)
        self.assertIsNone(reports["y"].degenerate)
        self.assertEqual(reports["y"].coverage, 1.0)

    def test_head_using_candidate_loses_rows_without_head(self) -> None:
        rec_a = make_recording("A", GRID9, 12)
        rec_t = make_recording("T1", [(0.4, 0.3), (0.6, 0.7)], 8, head_valid_every=2)
        fitted = V.fit_candidate(rec_a, V.Candidate("h", extra_names=("pitch_a",)), RIG)
        reports = V.score_axes(rec_t, fitted.predict_norm(rec_t, RIG))
        self.assertLess(reports["y"].coverage, 1.0)
        self.assertGreater(reports["y"].coverage, 0.0)

    def test_a_model_refuses_input_of_the_wrong_width(self) -> None:
        rec_a = make_recording("A", GRID9, 12)
        fitted = V.fit_candidate(rec_a, V.Candidate("h", extra_names=("pitch_a",)), RIG)
        with self.assertRaises(ValueError):
            fitted.model.predict_cm(np.zeros((2, rec_a.feature_dim)))


# --- safety and privacy -------------------------------------------------------


class TestBoundaries(unittest.TestCase):
    def test_module_never_imports_the_gazefollower_library(self) -> None:
        source = (Path(__file__).resolve().parent.parent / "gf_vertical.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        offenders = [
            name
            for name in imported
            if name.split(".")[0] in ("gazefollower", "pyautogui", "win32api")
        ]
        self.assertEqual(offenders, [], f"gf_vertical must stay offline, found {offenders}")

    def test_cli_refuses_to_write_inside_recordings(self) -> None:
        target = Path(__file__).resolve().parent.parent / "recordings" / "should_never_appear"
        self.assertEqual(V.main(["--out", str(target)]), 2)
        self.assertFalse(target.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
