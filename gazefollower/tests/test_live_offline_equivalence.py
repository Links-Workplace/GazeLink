"""Live and offline turn the same features into the same point.

Scope, deliberately narrow: features -> prediction only. The live view feeds
each frame's ``gaze_info.features`` through ``LiveRunner.on_frame``; offline
scoring runs ``FittedModel.predict_norm`` over the saved rows. If those two
disagreed, every offline accuracy number would describe a system nobody
runs. This does NOT test capture, feature extraction, queues or latency.

The real-data test uses ``recordings/resolution/round3/T1`` (hi arm) and the hi
v1 model, which never saw that recording. It skips when either is absent,
unless ``GAZELINK_REQUIRE_REAL_DATA=1`` is set, in which case absence fails:
before a live experiment the check must actually have run, not quietly skipped.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import gf_common as C  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_live as L  # noqa: E402
import gf_presets as PRE  # noqa: E402
import gf_schema as S  # noqa: E402

REAL_RECORDING = HERE / "recordings" / "resolution" / "round3"
REAL_MODEL = HERE / "results" / "hi" / "models" / "svr_fzscore_lnone_C100_g0.0005__hi_pool_r1-3"
CONFIG = "svr_fzscore_lnone_C100_g0.0005"
TOLERANCE_DEVICE_PX = 1.0


def live_points(model: FIT.FittedModel, rig: C.RigGeometry, features: np.ndarray,
                openness: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Each row through the live runner, as the camera thread would deliver it."""

    runner = L.LiveRunner(model, None, rig, None, head_builder=lambda face: None)
    out = np.full((len(features), 2), np.nan)
    for i in range(len(features)):
        gaze = SimpleNamespace(status=bool(status[i]), features=features[i],
                               raw_gaze_coordinates=None, tracking_state=None)
        face = SimpleNamespace(status=True, left_eye_openness=float(openness[i, 0]),
                               right_eye_openness=float(openness[i, 1]))
        runner.on_frame(face, gaze)
        if runner.last_error is not None:
            raise AssertionError(f"live runner raised on row {i}: {runner.last_error}")
        point = runner.state.unfiltered
        if point is not None:
            out[i] = point
    return out


def max_device_px(a: np.ndarray, b: np.ndarray, rig: C.RigGeometry) -> tuple[float, int]:
    both = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    if not both.any():
        return float("nan"), 0
    dx = (a[both, 0] - b[both, 0]) * rig.device_w_px
    dy = (a[both, 1] - b[both, 1]) * rig.device_h_px
    d = np.hypot(dx, dy)
    return float(d.max()), int(both.sum())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticEquivalenceTests(unittest.TestCase):
    def test_live_matches_offline_on_synthetic_features(self) -> None:
        rng = np.random.default_rng(0)
        rig = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
        X = rng.normal(size=(80, 258)).astype(np.float32)
        y = np.column_stack([rng.uniform(-24, 24, 80), rng.uniform(5, 30, 80)])
        config = next(c for c in PRE.sweep_with_presets(()) if c.name == CONFIG)
        model = FIT.FittedModel.fit(config, X, y, rig=rig, train_meta={"fitted_by": "test"})
        test = rng.normal(size=(40, 258)).astype(np.float32)
        live = live_points(model, rig, test, np.full((40, 2), 100.0), np.ones(40, bool))
        offline = model.predict_norm(test, rig)
        err, n = max_device_px(live, offline, rig)
        self.assertEqual(n, 40, "every valid row must produce a live point")
        self.assertLessEqual(err, TOLERANCE_DEVICE_PX)

    def test_a_blink_row_gives_no_live_point(self) -> None:
        rng = np.random.default_rng(1)
        rig = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
        X = rng.normal(size=(40, 258)).astype(np.float32)
        y = np.column_stack([rng.uniform(-24, 24, 40), rng.uniform(5, 30, 40)])
        config = next(c for c in PRE.sweep_with_presets(()) if c.name == CONFIG)
        model = FIT.FittedModel.fit(config, X, y, rig=rig, train_meta={"fitted_by": "test"})
        openness = np.array([[100.0, 100.0], [1.0, 100.0]])
        live = live_points(model, rig, X[:2], openness, np.ones(2, bool))
        self.assertTrue(np.all(np.isfinite(live[0])))
        self.assertTrue(np.all(np.isnan(live[1])))


class RealDataEquivalenceTests(unittest.TestCase):
    def test_live_matches_offline_on_the_held_out_hi_recording(self) -> None:
        present = (REAL_RECORDING / "T1.npz").exists() and (REAL_MODEL / "schema.json").exists()
        if not present:
            if os.environ.get("GAZELINK_REQUIRE_REAL_DATA") == "1":
                self.fail(f"required real data missing: {REAL_RECORDING} or {REAL_MODEL}")
            self.skipTest("real recording or model not present")
        rec = S.Recording.load(REAL_RECORDING, "T1")
        rig = C.RigGeometry.from_dict(rec.meta["rig"])
        model = FIT.FittedModel.load(REAL_MODEL)
        self.assertFalse(model.schema.head_names, "this check assumes a features-only model")
        # The offline scoring path, exactly: eligible rows, float32 as stored.
        rows = FIT.eligible_rows(rec)
        offline = np.full((rec.n_rows, 2), np.nan)
        offline[rows] = model.predict_norm(rec.features[rows], rig)
        live = live_points(model, rig, rec.features, rec.openness, rec.gaze_status)
        err, n = max_device_px(live[rows], offline[rows], rig)
        has_offline = np.all(np.isfinite(offline), axis=1)
        has_live = np.all(np.isfinite(live), axis=1)
        lost = int(np.sum(rows & has_offline & ~has_live))
        print(
            f"\nequivalence: {REAL_RECORDING.relative_to(HERE)}/T1 (hi arm), "
            f"model {REAL_MODEL.name} schema sha256 {sha256(REAL_MODEL / 'schema.json')[:16]}"
            f" svr_x {sha256(REAL_MODEL / 'svr_x.xml')[:16]}"
            f" svr_y {sha256(REAL_MODEL / 'svr_y.xml')[:16]}"
            f"; rows compared {n} of {int(rows.sum())} eligible; offline-only rows {lost}; "
            f"max difference {err:.4f} device px"
        )
        self.assertGreater(n, 0)
        self.assertEqual(lost, 0, "an eligible row the live gate refused is a path difference")
        self.assertLessEqual(err, TOLERANCE_DEVICE_PX)


if __name__ == "__main__":
    unittest.main()
