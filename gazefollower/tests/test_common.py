"""gf_common: the copied formulas must equal the library's, and the JSON must be
what analyze.py accepts. No gazefollower import anywhere in tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402

RIG = C.RigGeometry(
    camera_x_cm=60.0,
    camera_y_cm=63.6,
    screen_w_cm=120.0,
    screen_h_cm=33.75,
    device_w_px=5120,
    device_h_px=1440,
)


class GeometryTests(unittest.TestCase):
    def test_labels_at_grid_match_values_verified_against_the_library(self) -> None:
        # Values computed with gazefollower.misc.px2cm on this rig (plan §1).
        expected_x = {0: -56.875, 1: 0.0, 2: 56.875}
        expected_y = {0: 62.0375, 1: 46.725, 2: 31.4125}
        for ix, gx in enumerate(C.GRID_X):
            for iy, gy in enumerate(C.GRID_Y):
                cm_x, cm_y = RIG.norm_to_label_cm(gx, gy)
                self.assertAlmostEqual(cm_x, expected_x[ix], places=6)
                self.assertAlmostEqual(cm_y, expected_y[iy], places=6)

    def test_camera_is_below_the_screen_on_this_rig(self) -> None:
        self.assertTrue(RIG.camera_below_screen)
        # Screen top is +63.6 cm above the camera, bottom is +29.85: y is UP.
        self.assertAlmostEqual(RIG.norm_to_label_cm(0.5, 0.0)[1], 63.6)
        self.assertAlmostEqual(RIG.norm_to_label_cm(0.5, 1.0)[1], 63.6 - 33.75)

    def test_px2cm_cm2px_round_trip(self) -> None:
        for px in (0.0, 1.0, 2560.0, 5119.0, 5120.0):
            for py in (0.0, 1.0, 720.0, 1439.0, 1440.0):
                cm = RIG.px2cm(px, py)
                back = RIG.cm2px(*cm)
                self.assertAlmostEqual(back[0], px, places=9)
                self.assertAlmostEqual(back[1], py, places=9)

    def test_cm_to_norm_inverts_norm_to_label(self) -> None:
        for nx, ny in ((0.0, 0.0), (0.3, 0.7), (1.0, 1.0)):
            cm = RIG.norm_to_label_cm(nx, ny)
            back = RIG.cm_to_norm(*cm)
            self.assertAlmostEqual(back[0], nx, places=12)
            self.assertAlmostEqual(back[1], ny, places=12)

    def test_geometry_dict_round_trip_and_validation(self) -> None:
        self.assertEqual(C.RigGeometry.from_dict(RIG.to_dict()), RIG)
        with self.assertRaises(ValueError):
            C.RigGeometry(60, 63.6, 0, 33.75, 5120, 1440)
        with self.assertRaises(ValueError):
            C.RigGeometry(60, 63.6, 120, 33.75, 5120, 0)


class ProtocolConstantsTests(unittest.TestCase):
    def test_grid_is_the_library_mesh_with_50px_margins_on_1920x1080(self) -> None:
        self.assertAlmostEqual(C.GRID_X[0], 50 / 1920)
        self.assertAlmostEqual(C.GRID_X[2], 1 - 50 / 1920)
        self.assertAlmostEqual(C.GRID_Y[0], 50 / 1080)
        self.assertAlmostEqual(C.GRID_Y[2], 1 - 50 / 1080)

    def test_nine_point_sequence_shape(self) -> None:
        self.assertEqual(len(C.NINE_POINT_SEQUENCE), 10)
        self.assertEqual(C.NINE_POINT_SEQUENCE[0], (0.5, 0.5))  # warm-up, not stored
        self.assertEqual(C.NINE_POINT_SEQUENCE[-1], (0.5, 0.5))  # stored centre
        self.assertEqual(len(C.NINE_POINT_STORED), 9)
        self.assertEqual(len(set(C.NINE_POINT_STORED)), 9)
        for x, y in C.NINE_POINT_STORED:
            self.assertIn(x, C.GRID_X)
            self.assertIn(y, C.GRID_Y)

    def test_controller_timing_constants(self) -> None:
        self.assertEqual(C.PREPARE_S, 1.5)
        self.assertEqual(C.WAIT_S, 0.5)
        self.assertEqual(C.N_FRAMES_PER_POINT, 45)
        self.assertEqual(C.BLINK_THRESHOLD, 10.0)
        self.assertEqual(C.RESTING_PHASE, "COLLECTING")


class DistanceTests(unittest.TestCase):
    def test_distance_uses_last_pixel_convention(self) -> None:
        # Full diagonal of a 4096 x 1152 logical screen spans (4095, 1151) px.
        d = C.distance_px(0.0, 0.0, 1.0, 1.0, 4096, 1152)
        self.assertAlmostEqual(d, (4095**2 + 1151**2) ** 0.5)

    def test_arrival_times_first_crossing_per_radius(self) -> None:
        rows = [
            (0.0, 0.0, 0.5),  # far
            (100.0, 0.45, 0.5),  # 204.75 px away -> within 400
            (200.0, 0.48, 0.5),  # 81.9 px -> within 100 and 200
            (300.0, 0.50, 0.5),
        ]
        times = C.arrival_times_ms(rows, 0.5, 0.5, 4096, 1152)
        self.assertEqual(times, {"100": 200.0, "200": 200.0, "400": 100.0})

    def test_arrival_times_none_when_never_reached(self) -> None:
        rows = [(0.0, 0.0, 0.0), (50.0, 0.1, 0.1)]
        times = C.arrival_times_ms(rows, 1.0, 1.0, 4096, 1152)
        self.assertEqual(times, {"100": None, "200": None, "400": None})


class PredictionsFileTests(unittest.TestCase):
    GEOMETRY = {
        "dpi_scale": 1.25,
        "height_px": 1152,
        "orientation": "LANDSCAPE",
        "screen_id": "LS49C95xU",
        "width_px": 4096,
    }
    TARGETS = [
        {"index": 0, "name": "TEST_0", "screen_position": {"x": 0.7, "y": 0.3}},
        {"index": 1, "name": "TEST_1", "screen_position": {"x": 0.2, "y": 0.8}},
    ]

    def test_make_sample_required_and_optional_fields(self) -> None:
        row = C.make_sample(
            target_index=1,
            predicted_x=0.25,
            predicted_y=0.75,
            timestamp_ms=1234.5,
            collection_phase=C.RESTING_PHASE,
            fps=30.4,
            head_yaw_deg=1.0,
            head_pitch_deg=-2.0,
            head_roll_deg=0.5,
            raw_x=0.01,
            arm="C",
        )
        for key in ("target_index", "predicted_x", "predicted_y", "timestamp", "accepted"):
            self.assertIn(key, row)
        self.assertEqual(row["collection_phase"], "COLLECTING")
        self.assertEqual(row["head_pitch_deg"], -2.0)
        self.assertEqual(row["arm"], "C")
        self.assertIsInstance(row["accepted"], bool)

    def test_make_sample_omits_pose_unless_all_three_present(self) -> None:
        row = C.make_sample(
            target_index=0, predicted_x=0.5, predicted_y=0.5, timestamp_ms=0.0, head_yaw_deg=1.0
        )
        self.assertNotIn("head_yaw_deg", row)

    def test_make_sample_rejects_bool_index_and_shadowing_extras(self) -> None:
        with self.assertRaises(TypeError):
            C.make_sample(target_index=True, predicted_x=0, predicted_y=0, timestamp_ms=0)
        # The row key is "timestamp" while the parameter is timestamp_ms, so an
        # extra named "timestamp" would silently overwrite the real one.
        with self.assertRaises(ValueError):
            C.make_sample(
                target_index=0, predicted_x=0, predicted_y=0, timestamp_ms=0, timestamp=5.0
            )

    def test_write_validates_and_round_trips(self) -> None:
        rows = [
            C.make_sample(
                target_index=0,
                predicted_x=0.71,
                predicted_y=0.31,
                timestamp_ms=1600.0,
                collection_phase=C.RESTING_PHASE,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = C.write_predictions_json(
                Path(tmp) / "sub" / "p.json",
                screen_geometry=self.GEOMETRY,
                targets=self.TARGETS,
                samples=rows,
                run={"engine": "gazefollower-1.0.2", "arm": "A"},
                timings=[{"target_index": 0, "time_to_target_ms": {"100": 20.0}}],
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
        C.validate_predictions_payload(payload)
        self.assertEqual(payload["samples"][0]["target_index"], 0)
        self.assertEqual(payload["run"]["arm"], "A")
        self.assertNotIn("freeze_verified", payload["run"])

    def test_write_refuses_freeze_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                C.write_predictions_json(
                    Path(tmp) / "p.json",
                    screen_geometry=self.GEOMETRY,
                    targets=self.TARGETS,
                    samples=[],
                    run={"freeze_verified": True},
                )

    def test_validate_rejects_out_of_range_index_and_non_bool_accepted(self) -> None:
        base = {"screen_geometry": self.GEOMETRY, "targets": self.TARGETS}
        bad_index = dict(base, samples=[{"target_index": 2, "predicted_x": 0, "predicted_y": 0, "timestamp": 0, "accepted": True}])
        with self.assertRaises(ValueError):
            C.validate_predictions_payload(bad_index)
        bad_bool = dict(base, samples=[{"target_index": True, "predicted_x": 0, "predicted_y": 0, "timestamp": 0, "accepted": True}])
        with self.assertRaises(ValueError):
            C.validate_predictions_payload(bad_bool)
        bad_acc = dict(base, samples=[{"target_index": 0, "predicted_x": 0, "predicted_y": 0, "timestamp": 0, "accepted": 1}])
        with self.assertRaises(ValueError):
            C.validate_predictions_payload(bad_acc)

    def test_load_targets_reads_export_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            path.write_text(
                json.dumps({"screen_geometry": self.GEOMETRY, "targets": self.TARGETS}),
                encoding="utf-8",
            )
            targets, geometry = C.load_targets(path)
        self.assertEqual(len(targets), 2)
        self.assertEqual(geometry["width_px"], 4096)


if __name__ == "__main__":
    unittest.main()
