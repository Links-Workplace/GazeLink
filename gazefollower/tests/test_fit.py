"""gf_fit on synthetic recordings: models learn a planted relation, save/load
identity, schema refusal, label de-standardisation, selection cannot see T1,
arm A is never head-filtered, and the filter replay skips invalid rows."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_fit as F  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_schema as S  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
GEOM = {"width_px": 4096, "height_px": 1152, "dpi_scale": 1.25, "screen_id": "test", "orientation": "LANDSCAPE"}
DIM = 8
RNG = np.random.default_rng(7)


def _synthetic_features(nx: float, ny: float, n: int, noise: float = 0.02, head_pitch: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Features carry the target linearly in two columns (+ noise); the rest is noise.
    Column 5 is constant. Optional head6 with pitch_a tied to ny."""

    X = RNG.normal(scale=0.3, size=(n, DIM)).astype(np.float32)
    X[:, 0] = 2.0 * nx + RNG.normal(scale=noise, size=n)
    X[:, 1] = -1.5 * ny + RNG.normal(scale=noise, size=n)
    X[:, 5] = 0.7
    head = np.zeros((n, 6))
    head[:, H.HEAD6_NAMES.index("pitch_a")] = (-0.6 + (0.3 * ny if head_pitch is None else head_pitch)) + RNG.normal(scale=0.005, size=n)
    head[:, H.HEAD6_NAMES.index("iod_norm")] = 0.3
    return X, head


def _targets(points: list[tuple[float, float]]) -> list[dict]:
    return [{"index": i, "name": f"P{i}", "screen_position": {"x": x, "y": y}} for i, (x, y) in enumerate(points)]


def make_recording(protocol: str, points: list[tuple[float, float]], per_target: int, *, head_invalid_every: int = 0, fps: float = 30.0, labels_override=None) -> S.Recording:
    meta = {"rig": RIG.to_dict(), "targets": _targets(points), "target_geometry": GEOM, "integrity": {"fps_median": fps, "subscriber_errors": 0, "watchdog_tripped": False}, "session": "test"}
    b = S.RecordingBuilder(protocol, 0, meta)
    seq = 0
    calibration = protocol in S.CALIBRATION_PROTOCOLS
    for tid, (nx, ny) in enumerate(points):
        lx, ly = (nx, ny) if labels_override is None else labels_override[tid]
        X, head = _synthetic_features(nx, ny, per_target)
        for i in range(per_target):
            invalid = head_invalid_every and (seq % head_invalid_every == 0)
            phase = S.PHASE_COLLECT if calibration else S.PHASE_COLLECTING
            b.append(frame_seq=seq, timestamp_ns=seq * 33_000_000, elapsed_ms=1500.0 + i * 33.0, target_id=tid, block=0,
                     phase=phase, target_xy=(lx, ly), label_cm=RIG.norm_to_label_cm(lx, ly), features=X[i],
                     head=None if invalid else head[i], pnp_deg=(0.0, 1.0, 0.0), raw_cm=(0.0, 0.0), openness=(100.0, 100.0),
                     tracking_state="SUCCESS", gaze_status=True, accepted=calibration)
            seq += 1
        if not calibration:
            # one stabilizing row before each target, like a live capture
            b.append(frame_seq=seq, timestamp_ns=seq * 33_000_000, elapsed_ms=100.0, target_id=tid, block=0,
                     phase=S.PHASE_STABILIZING, target_xy=(nx, ny), label_cm=RIG.norm_to_label_cm(nx, ny),
                     features=X[0], head=head[0], pnp_deg=None, raw_cm=(0.0, 0.0), openness=(100.0, 100.0),
                     tracking_state="SUCCESS", gaze_status=True, accepted=False)
            seq += 1
    return b.freeze()


GRID9 = [(x, y) for y in C.GRID_Y for x in C.GRID_X]
TUNE_PTS = [(0.3, 0.3), (0.7, 0.2), (0.6, 0.8), (0.2, 0.6), (0.45, 0.55)]
T1_PTS = [(0.35, 0.25), (0.65, 0.75), (0.15, 0.85), (0.8, 0.4)]


class FitConfigTests(unittest.TestCase):
    def test_grid_has_library_default_first_and_valid_entries(self) -> None:
        grid = F.phase0_grid()
        self.assertEqual(grid[0].name, "lib-default")
        self.assertEqual((grid[0].C, grid[0].gamma, grid[0].P), (1.0, 0.005, 0.001))
        self.assertGreater(len(grid), 30)
        self.assertEqual(len({g.name for g in grid}), len(grid))

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            F.FitConfig(name="x", feature_scaling="minmax")
        with self.assertRaises(ValueError):
            F.FitConfig(name="x", head_names=("nope",))

    def test_auto_gamma_is_one_over_d_after_zscore(self) -> None:
        cfg = F.FitConfig(name="x", feature_scaling="zscore", gamma="auto")
        Z = np.random.default_rng(0).normal(size=(50, 4))
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        self.assertAlmostEqual(F.resolve_gamma(cfg, Z), 0.25, places=6)


class FittedModelTests(unittest.TestCase):
    def _train(self):
        rec = make_recording("A", GRID9, 20)
        mask, dropped = F.training_rows(rec, ())
        X = F.design_matrix(rec, mask, ())
        return rec, X, rec.label_cm[mask], dropped

    def test_svr_zscore_learns_planted_relation(self) -> None:
        rec, X, Y, _ = self._train()
        cfg = F.FitConfig(name="z", feature_scaling="zscore", label_scaling="zscore", C=10.0, gamma="auto")
        model = F.FittedModel.fit(cfg, X, Y, rig=RIG, train_meta={})
        tune = make_recording("TUNE", TUNE_PTS, 15)
        metrics, _, _ = F._score_timed(model, tune, RIG, ())
        self.assertLess(metrics.median_euclid_px, 150.0, metrics.flat())
        self.assertIsNotNone(metrics.y.slope)
        self.assertGreater(metrics.y.slope, 0.6)

    def test_ridge_recovers_linear_relation_exactly(self) -> None:
        rec, X, Y, _ = self._train()
        cfg = F.FitConfig(name="r", feature_scaling="zscore", label_scaling="zscore", kind="ridge", alpha=0.01)
        model = F.FittedModel.fit(cfg, X, Y, rig=RIG, train_meta={})
        tune = make_recording("TUNE", TUNE_PTS, 15)
        metrics, _, _ = F._score_timed(model, tune, RIG, ())
        self.assertLess(metrics.median_euclid_px, 80.0, metrics.flat())
        self.assertAlmostEqual(metrics.x.slope, 1.0, delta=0.1)
        self.assertAlmostEqual(metrics.y.slope, 1.0, delta=0.1)

    def test_label_destandardisation_is_exact(self) -> None:
        rec, X, Y, _ = self._train()
        cfg = F.FitConfig(name="r", feature_scaling="zscore", label_scaling="zscore", kind="ridge", alpha=1e-6)
        model = F.FittedModel.fit(cfg, X, Y, rig=RIG, train_meta={})
        pred = model.predict_cm(X)
        # Training rows predicted back in cm, in the label range, not in z units.
        self.assertGreater(np.median(pred[:, 1]), 25.0)
        self.assertLess(np.median(pred[:, 1]), 70.0)

    def test_save_load_predict_identity_and_refusals(self) -> None:
        rec, X, Y, _ = self._train()
        for cfg in (F.FitConfig(name="s", feature_scaling="zscore", label_scaling="zscore", C=10.0), F.FitConfig(name="r", kind="ridge", alpha=1.0, feature_scaling="zscore")):
            model = F.FittedModel.fit(cfg, X, Y, rig=RIG, train_meta={"protocol": "A"})
            with tempfile.TemporaryDirectory() as tmp:
                model.save(Path(tmp))
                back = F.FittedModel.load(Path(tmp))
                self.assertTrue(np.allclose(back.predict_cm(X[:5]), model.predict_cm(X[:5]), atol=1e-4))
                self.assertEqual(back.schema.columns, model.schema.columns)
                # Tamper: schema says one column fewer than the saved model has.
                sp = Path(tmp) / "schema.json"
                value = json.loads(sp.read_text(encoding="utf-8"))
                value["base_dim"] -= 1
                value["columns"] = value["columns"][:-1]
                value["features"]["mean"] = value["features"]["mean"][:-1]
                value["features"]["std"] = value["features"]["std"][:-1]
                sp.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    F.FittedModel.load(Path(tmp))
            with self.assertRaises(ValueError):
                model.predict_cm(np.zeros((1, DIM + 1)))

    def test_nan_input_rows_give_nan_predictions_not_zero(self) -> None:
        rec, X, Y, _ = self._train()
        model = F.FittedModel.fit(F.FitConfig(name="l"), X, Y, rig=RIG, train_meta={})
        Xn = X[:3].copy()
        Xn[1, 0] = np.nan
        pred = model.predict_cm(Xn)
        self.assertTrue(np.all(np.isnan(pred[1])))
        self.assertTrue(np.all(np.isfinite(pred[[0, 2]])))

    def test_refuses_nan_in_training(self) -> None:
        rec, X, Y, _ = self._train()
        X = X.copy()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            F.FittedModel.fit(F.FitConfig(name="l"), X, Y, rig=RIG, train_meta={})

    def test_head_columns_change_schema_and_require_builder_version(self) -> None:
        rec = make_recording("A", GRID9, 10)
        head = ("pitch_a", "roll_deg")
        mask, dropped = F.training_rows(rec, head)
        X = F.design_matrix(rec, mask, head)
        self.assertEqual(X.shape[1], DIM + 2)
        model = F.FittedModel.fit(F.FitConfig(name="h", head_names=head, feature_scaling="zscore"), X, rec.label_cm[mask], rig=RIG, train_meta={})
        self.assertEqual(model.schema.builder_version, H.BUILDER_VERSION)
        self.assertEqual(model.schema.columns[-2:], head)


class RowSelectionTests(unittest.TestCase):
    def test_arm_a_is_never_head_filtered_but_head_arms_are_and_report_it(self) -> None:
        rec = make_recording("A", GRID9, 10, head_invalid_every=4)
        mask_a, dropped_a = F.training_rows(rec, ())
        self.assertEqual(dropped_a, 0)
        self.assertEqual(int(mask_a.sum()), 90)
        mask_c, dropped_c = F.training_rows(rec, ("pitch_a",))
        self.assertGreater(dropped_c, 0)
        self.assertEqual(int(mask_c.sum()) + dropped_c, 90)
        # B and C must share rows: same mask when both are given the head filter.
        mask_b, _ = F.training_rows(rec, ("pitch_a",))
        self.assertTrue(np.array_equal(mask_b, mask_c))

    def test_scoring_rows_are_collecting_only(self) -> None:
        rec = make_recording("T1", T1_PTS, 5)
        mask, _ = F.scoring_rows(rec, ())
        self.assertEqual(int(mask.sum()), 20)
        self.assertFalse(np.any(mask & (rec.phase == S.PHASE_STABILIZING)))


class EvaluateTests(unittest.TestCase):
    def test_metrics_on_known_offsets(self) -> None:
        targets = np.array([[0.2, 0.2], [0.2, 0.2], [0.8, 0.8], [0.8, 0.8]])
        preds = targets + np.array([[0.01, -0.02]] * 4)  # constant offset
        m = F.evaluate(preds, targets, np.array([0, 0, 1, 1]), 4096, 1152)
        self.assertAlmostEqual(m.x.bias_px, 0.01 * 4095, places=6)
        self.assertAlmostEqual(m.y.bias_px, -0.02 * 1151, places=6)
        self.assertAlmostEqual(m.x.slope, 1.0, places=9)
        self.assertAlmostEqual(m.y.slope, 1.0, places=9)
        self.assertEqual(m.n_lost, 0)

    def test_compressed_predictions_show_low_slope(self) -> None:
        targets = np.array([[0.5, 0.1], [0.5, 0.5], [0.5, 0.9]])
        preds = np.array([[0.5, 0.45], [0.5, 0.5], [0.5, 0.55]])
        m = F.evaluate(preds, targets, np.array([0, 1, 2]), 4096, 1152)
        self.assertAlmostEqual(m.y.slope, 0.125, places=9)
        self.assertIsNone(m.x.slope)  # single distinct x: undefined, not fabricated

    def test_lost_and_off_screen_counted(self) -> None:
        targets = np.array([[0.5, 0.5]] * 3)
        preds = np.array([[np.nan, np.nan], [1.2, 0.5], [0.5, -0.1]])
        m = F.evaluate(preds, targets, np.zeros(3, dtype=int), 4096, 1152)
        self.assertEqual(m.n_lost, 1)
        self.assertEqual(m.off_screen_x, 1)
        self.assertEqual(m.off_screen_y, 1)


class SelectionLeakageTests(unittest.TestCase):
    def test_select_config_uses_tune_metric_only(self) -> None:
        sweep = [
            {"name": "a", "tune": {F.SELECTION_METRIC: 50.0, F.SELECTION_TIEBREAK: 10.0}},
            {"name": "b", "tune": {F.SELECTION_METRIC: 40.0, F.SELECTION_TIEBREAK: 30.0}},
            {"name": "c", "tune": {F.SELECTION_METRIC: 40.0, F.SELECTION_TIEBREAK: 20.0}},
        ]
        self.assertEqual(F.select_config(sweep), "c")

    def test_permuting_t1_labels_does_not_change_the_selection(self) -> None:
        small_grid = [F.library_default_config(), F.FitConfig(name="z", feature_scaling="zscore", label_scaling="zscore", C=10.0, gamma="auto"), F.FitConfig(name="r", kind="ridge", alpha=1.0, feature_scaling="zscore", label_scaling="zscore")]
        rec_a = make_recording("A", GRID9, 8)
        rec_tune = make_recording("TUNE", TUNE_PTS, 6)
        rec_t1 = make_recording("T1", T1_PTS, 6)
        shuffled = list(reversed(T1_PTS))
        rec_t1_shuffled = make_recording("T1", T1_PTS, 6, labels_override=shuffled)
        results = []
        for t1 in (rec_t1, rec_t1_shuffled):
            with tempfile.TemporaryDirectory() as tmp:
                rec_a.save(Path(tmp))
                rec_tune.save(Path(tmp))
                t1.save(Path(tmp))
                report = F.run_phase0(Path(tmp), Path(tmp) / "out", grid=small_grid, replay_filter=False, save_models=False)
                results.append(report)
        self.assertEqual(results[0]["selection"]["selected"], results[1]["selection"]["selected"])
        # ... while T1's own numbers did change, proving T1 was scored, not consulted.
        s = results[0]["selection"]["selected"]
        self.assertNotAlmostEqual(results[0]["t1"][s]["t1"]["median_euclid_px"], results[1]["t1"][s]["t1"]["median_euclid_px"], delta=1.0)


class Phase0ReportTests(unittest.TestCase):
    def test_end_to_end_report_and_predictions_json(self) -> None:
        grid = [F.library_default_config(), F.FitConfig(name="z", feature_scaling="zscore", label_scaling="zscore", C=10.0, gamma="auto")]
        with tempfile.TemporaryDirectory() as tmp:
            make_recording("A", GRID9, 8).save(Path(tmp))
            make_recording("TUNE", TUNE_PTS, 6).save(Path(tmp))
            make_recording("T1", T1_PTS, 6).save(Path(tmp))
            make_recording("T2", [(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)], 6).save(Path(tmp))
            report = F.run_phase0(Path(tmp), Path(tmp) / "out", grid=grid, replay_filter=True, filter_factory=_PassThroughFilter, save_models=True)
            self.assertIn(report["p0_info"]["category"], ("fixed_by_calibration", "partial_improvement", "no_improvement"))
            if report["p0_info"]["category"] == "no_improvement":
                self.assertIn("This is not evidence", report["p0_info"]["statement"])
            self.assertIsNotNone(report["p0_confound"]["corr_pitch_a_target_y"])
            # 8 accepted rows per target in this synthetic set, not the 45 the
            # library's controller would collect.
            self.assertFalse(report["p0_valid"]["A"]["checks"]["every_expected_target_has_45"])
            for name, rep in report["t1"].items():
                payload = json.loads(Path(rep["predictions_json"]).read_text(encoding="utf-8"))
                C.validate_predictions_payload(payload)
                self.assertTrue(all("head_pitch_deg" in s for s in payload["samples"] if s["collection_phase"] == "COLLECTING"))
                self.assertNotIn("freeze_verified", payload["run"])
                self.assertIsNotNone(rep["t1_after_filter"])
            # Models (support vectors = training rows) live under the recording root only.
            self.assertTrue((Path(tmp) / "models" / "lib-default" / "svr_x.xml").exists())
            self.assertFalse(any(p.suffix == ".xml" for p in (Path(tmp) / "out").rglob("*")))
            text = (Path(tmp) / "out" / "phase0_report.md").read_text(encoding="utf-8")
            self.assertIn("G1", text)
            self.assertIn("selected", text)


class _PassThroughFilter:
    def filter_values(self, values):
        return values


class FilterReplayTests(unittest.TestCase):
    def test_replay_skips_invalid_rows_and_keeps_order(self) -> None:
        calls = []

        class Spy:
            def filter_values(self, values):
                calls.append(tuple(values))
                return [values[0] + 1.0, values[1] + 1.0]

        pred = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]])
        out = F.replay_heuristic_filter(pred, np.array([True, False, True]), Spy)
        self.assertEqual(calls, [(1.0, 1.0), (3.0, 3.0)])
        self.assertTrue(np.all(np.isnan(out[1])))
        self.assertTrue(np.allclose(out[[0, 2]], [[2.0, 2.0], [4.0, 4.0]]))

    def test_no_gazefollower_import_in_tests(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
