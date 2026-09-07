"""gf_schema: recordings round-trip with NaN for invalid values; schemas refuse
mismatches; standardiser learns from training rows only."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_schema as S  # noqa: E402

META = {
    "rig": {"camera_x_cm": 60, "camera_y_cm": 63.6, "screen_w_cm": 120, "screen_h_cm": 33.75, "device_w_px": 5120, "device_h_px": 1440},
    "targets": [{"index": 0, "name": "P0", "screen_position": {"x": 0.5, "y": 0.5}}],
    "target_geometry": {"width_px": 4096, "height_px": 1152},
}


def _builder(protocol: str = "A") -> S.RecordingBuilder:
    return S.RecordingBuilder(protocol, round_id=0, meta=META)


class RecordingBuilderTests(unittest.TestCase):
    def test_invalid_rows_are_nan_not_zero(self) -> None:
        b = _builder()
        b.append(frame_seq=0, timestamp_ns=1, elapsed_ms=None, target_id=-1, block=0, phase=S.PHASE_IDLE,
                 target_xy=None, label_cm=None, features=None, head=None, pnp_deg=None, raw_cm=None,
                 openness=(0.0, 0.0), tracking_state="FACE_MISSING", gaze_status=False, accepted=False)
        b.append(frame_seq=1, timestamp_ns=2, elapsed_ms=10.0, target_id=0, block=0, phase=S.PHASE_COLLECT,
                 target_xy=(0.5, 0.5), label_cm=(0.0, 46.7), features=np.ones(4), head=np.arange(6.0),
                 pnp_deg=(1.0, 2.0, 3.0), raw_cm=(0.1, -0.2), openness=(100.0, 90.0),
                 tracking_state="SUCCESS", gaze_status=True, accepted=True)
        rec = b.freeze()
        self.assertEqual(rec.n_rows, 2)
        self.assertEqual(rec.feature_dim, 4)
        self.assertTrue(np.all(np.isnan(rec.features[0])))
        self.assertTrue(np.all(np.isnan(rec.head[0])))
        self.assertTrue(np.all(np.isnan(rec.pnp_deg[0])))
        self.assertTrue(np.all(np.isnan(rec.target_xy[0])))
        self.assertFalse(rec.head_valid[0])
        self.assertTrue(rec.head_valid[1])
        self.assertTrue(np.array_equal(rec.features[1], np.ones(4, dtype=np.float32)))
        self.assertEqual(rec.phase[0], "IDLE")
        self.assertEqual(rec.tracking_state[0], "FACE_MISSING")

    def test_feature_dim_must_not_change(self) -> None:
        b = _builder()
        common = dict(elapsed_ms=0.0, target_id=0, block=0, phase=S.PHASE_COLLECT, target_xy=(0.5, 0.5),
                      label_cm=(0.0, 0.0), head=None, pnp_deg=None, raw_cm=None, openness=(1.0, 1.0),
                      tracking_state="SUCCESS", gaze_status=True, accepted=True)
        b.append(frame_seq=0, timestamp_ns=0, features=np.zeros(3), **common)
        with self.assertRaises(ValueError):
            b.append(frame_seq=1, timestamp_ns=1, features=np.zeros(4), **common)

    def test_accepted_requires_valid_gaze(self) -> None:
        b = _builder()
        with self.assertRaises(ValueError):
            b.append(frame_seq=0, timestamp_ns=0, elapsed_ms=0.0, target_id=0, block=0, phase=S.PHASE_COLLECT,
                     target_xy=(0.5, 0.5), label_cm=(0.0, 0.0), features=None, head=None, pnp_deg=None,
                     raw_cm=None, openness=(1.0, 1.0), tracking_state="FAILURE", gaze_status=False, accepted=True)

    def test_unknown_protocol_rejected(self) -> None:
        with self.assertRaises(ValueError):
            S.RecordingBuilder("X", 0, META)

    def test_save_load_round_trip(self) -> None:
        b = _builder("T1")
        for i in range(5):
            b.append(frame_seq=i, timestamp_ns=i * 33_000_000, elapsed_ms=i * 33.0, target_id=0, block=0,
                     phase=S.PHASE_COLLECTING if i >= 2 else S.PHASE_STABILIZING, target_xy=(0.5, 0.5),
                     label_cm=(0.0, 46.7), features=np.full(4, i, dtype=np.float32), head=np.full(6, 0.1 * i),
                     pnp_deg=None, raw_cm=(0.0, 0.0), openness=(50.0, 50.0), tracking_state="SUCCESS",
                     gaze_status=True, accepted=False)
        rec = b.freeze()
        with tempfile.TemporaryDirectory() as tmp:
            rec.save(Path(tmp))
            back = S.Recording.load(Path(tmp), "T1")
        self.assertEqual(back.n_rows, 5)
        self.assertEqual(back.meta["recording_format"], S.RECORDING_FORMAT_VERSION)
        self.assertEqual(back.meta["feature_dim"], 4)
        self.assertTrue(np.array_equal(back.features, rec.features))
        self.assertTrue(np.array_equal(back.rows_collecting(), np.array([False, False, True, True, True])))
        self.assertEqual(back.meta["targets"][0]["name"], "P0")

    def test_load_refuses_other_format_version(self) -> None:
        b = _builder("T1")
        b.append(frame_seq=0, timestamp_ns=0, elapsed_ms=0.0, target_id=0, block=0, phase=S.PHASE_COLLECTING,
                 target_xy=(0.5, 0.5), label_cm=(0.0, 0.0), features=np.zeros(2), head=None, pnp_deg=None,
                 raw_cm=None, openness=(1.0, 1.0), tracking_state="SUCCESS", gaze_status=True, accepted=False)
        rec = b.freeze()
        with tempfile.TemporaryDirectory() as tmp:
            _, meta_path = rec.save(Path(tmp))
            meta_path.write_text(meta_path.read_text(encoding="utf-8").replace(S.RECORDING_FORMAT_VERSION, "rec-0"), encoding="utf-8")
            with self.assertRaises(ValueError):
                S.Recording.load(Path(tmp), "T1")


class AssembleTests(unittest.TestCase):
    def test_no_head_names_returns_base_unchanged(self) -> None:
        base = np.arange(6.0).reshape(2, 3)
        out = S.assemble(base, None, (), ("a", "b"))
        self.assertTrue(np.array_equal(out, base))

    def test_head_columns_selected_and_ordered(self) -> None:
        base = np.zeros((2, 3))
        head = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = S.assemble(base, head, ("c", "a"), ("a", "b", "c"))
        self.assertEqual(out.shape, (2, 5))
        self.assertTrue(np.array_equal(out[:, 3:], np.array([[3.0, 1.0], [6.0, 4.0]])))

    def test_head_required_when_names_given(self) -> None:
        with self.assertRaises(ValueError):
            S.assemble(np.zeros((2, 3)), None, ("a",), ("a",))
        with self.assertRaises(ValueError):
            S.assemble(np.zeros((2, 3)), np.zeros((3, 1)), ("a",), ("a",))


class VarianceFloorTests(unittest.TestCase):
    """A column that barely moved during the short calibration must not become
    a divisor small enough to turn an ordinary later change into a huge
    z-score. That is what froze the on-screen point: one column with a
    training std of 0.000327 turned a 0.28 shift into z=860 and pushed the
    sample outside the SVR's kernel by itself."""

    def test_a_near_constant_column_cannot_dominate_the_distance(self) -> None:
        rng = np.random.default_rng(20260907)
        wide = rng.normal(0.0, 1.0, size=(40, 1))
        barely = rng.normal(0.0, 3e-4, size=(40, 1))
        z = S.Standardizer.fit(np.hstack([wide, barely]))
        later = np.array([[0.0, 0.28]])
        self.assertLess(abs(z.transform(later)[0, 1]), 50.0)

    def test_the_floor_is_relative_to_the_widest_column(self) -> None:
        rng = np.random.default_rng(7)
        X = np.hstack(
            [rng.normal(0.0, 50.0, size=(40, 1)), rng.normal(0.0, 1e-3, size=(40, 1))]
        )
        z = S.Standardizer.fit(X)
        self.assertGreaterEqual(z.std[1], S.VARIANCE_FLOOR_FRACTION * z.std[0] * 0.99)

    def test_columns_with_healthy_spread_are_left_alone(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.normal(0.0, 1.0, size=(200, 3))
        z = S.Standardizer.fit(X)
        for column in range(3):
            self.assertAlmostEqual(z.std[column], X[:, column].std(), places=9)


class StandardizerTests(unittest.TestCase):
    def test_fit_uses_only_given_rows_and_handles_constant_columns(self) -> None:
        X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
        z = S.Standardizer.fit(X)
        self.assertAlmostEqual(z.mean[0], 3.0)
        self.assertEqual(z.std[1], 1.0)  # constant column -> std 1
        Z = z.transform(X)
        self.assertTrue(np.allclose(Z[:, 1], 0.0))
        self.assertTrue(np.allclose(z.inverse(Z), X))
        # A different (test) row is transformed with the TRAINING statistics.
        self.assertTrue(np.allclose(z.transform(np.array([[3.0, 9.0]])), [[0.0, 4.0]]))

    def test_refuses_nan_and_wrong_width(self) -> None:
        with self.assertRaises(ValueError):
            S.Standardizer.fit(np.array([[1.0, float("nan")], [2.0, 3.0]]))
        z = S.Standardizer.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with self.assertRaises(ValueError):
            z.transform(np.zeros((1, 3)))

    def test_dict_round_trip(self) -> None:
        z = S.Standardizer.fit(np.array([[1.0, 2.0], [3.0, 8.0]]))
        back = S.Standardizer.from_dict(z.to_dict())
        self.assertTrue(np.allclose(back.mean, z.mean))
        self.assertTrue(np.allclose(back.std, z.std))


class FeatureSchemaTests(unittest.TestCase):
    def _schema(self, head=("roll_deg",)) -> S.FeatureSchema:
        return S.FeatureSchema(
            base_dim=3, head_names=tuple(head), feature_scaling="zscore", label_scaling="none",
            features=S.Standardizer.identity(3 + len(head)), labels=S.Standardizer.identity(2),
            svr={"C": 1.0, "gamma": 0.005, "P": 0.001}, builder_version="head-ratios-2", rig=META["rig"],
        )

    def test_columns_and_check(self) -> None:
        schema = self._schema()
        self.assertEqual(schema.columns, ("f000", "f001", "f002", "roll_deg"))
        schema.check_columns(4, "head-ratios-2")
        with self.assertRaises(ValueError):
            schema.check_columns(3, "head-ratios-2")
        with self.assertRaises(ValueError):
            schema.check_columns(4, "head-ratios-1")
        # No head columns: builder version is irrelevant.
        self._schema(head=()).check_columns(3, None)

    def test_json_round_trip_and_version_refusal(self) -> None:
        schema = self._schema()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            schema.save(path)
            back = S.FeatureSchema.load(path)
            self.assertEqual(back.columns, schema.columns)
            self.assertEqual(back.svr, schema.svr)
            text = path.read_text(encoding="utf-8").replace(S.SCHEMA_VERSION, "schema-0")
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                S.FeatureSchema.load(path)

    def test_tampered_column_list_refused(self) -> None:
        value = self._schema().to_dict()
        value["columns"] = value["columns"][:-1]
        with self.assertRaises(ValueError):
            S.FeatureSchema.from_dict(value)


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
