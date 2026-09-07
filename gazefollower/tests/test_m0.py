"""M0 modules: setup manifest, target sets and proximity, angular geometry.

No camera, no gazefollower import. The camera probe is exercised only through
its skip path.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_geometry as G  # noqa: E402
import gf_setup as U  # noqa: E402
import gf_targets as T  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
W, H = 4096, 1152


# --- gf_targets --------------------------------------------------------------


class Grid16Tests(unittest.TestCase):
    def test_sixteen_unique_points_on_the_specified_coordinates(self) -> None:
        grid = T.build_grid16(W, H)
        self.assertEqual(len(grid), 16)
        coords = {(t["screen_position"]["x"], t["screen_position"]["y"]) for t in grid}
        self.assertEqual(len(coords), 16)
        for x, y in coords:
            self.assertIn(x, T.GRID16_COORDS)
            self.assertIn(y, T.GRID16_COORDS)

    def test_order_is_shuffled_but_reproducible_from_the_seed(self) -> None:
        a = [t["name"] for t in T.build_grid16(W, H, seed=1)]
        b = [t["name"] for t in T.build_grid16(W, H, seed=1)]
        c = [t["name"] for t in T.build_grid16(W, H, seed=2)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_eight_of_sixteen_are_near_gazefollowers_calibration_grid(self) -> None:
        # The measured conflict: corners sit 98 px from a calibration point,
        # well inside the project's 250 px held-out threshold.
        grid = T.build_grid16(W, H)
        summary = T.proximity_summary(grid)
        self.assertEqual(summary["n_near_calibration"], 8)
        self.assertAlmostEqual(summary["min_px"], 98.2, delta=0.5)
        corners = [t for t in grid if t["screen_position"]["x"] in (0.05, 0.95) and t["screen_position"]["y"] in (0.05, 0.95)]
        self.assertEqual(len(corners), 4)
        for corner in corners:
            self.assertTrue(corner["near_calibration"])
            self.assertLess(corner["distance_to_calibration_px"], 150)

    def test_the_projects_held_out_set_has_none_near_calibration(self) -> None:
        held_out, geometry = C.load_targets(Path(__file__).resolve().parent.parent / "targets.json")
        annotated = T.annotate_targets(held_out, int(geometry["width_px"]), int(geometry["height_px"]))
        summary = T.proximity_summary(annotated)
        self.assertEqual(summary["n_near_calibration"], 0)
        self.assertGreaterEqual(summary["min_px"], T.NEAR_CALIBRATION_PX)

    def test_distance_is_to_the_nearest_calibration_point(self) -> None:
        distance, nearest = T.distance_to_nearest_calibration(0.5, 0.5, W, H)
        self.assertAlmostEqual(distance, 0.0, delta=1.0)  # the centre IS a calibration point
        self.assertAlmostEqual(nearest[0], 0.5)

    def test_zone_map_covers_the_screen_in_sixteen_cells(self) -> None:
        self.assertEqual(T.zone_of(0.0, 0.0), 0)
        self.assertEqual(T.zone_of(0.99, 0.99), 15)
        self.assertEqual(T.zone_of(0.3, 0.0), 1)
        self.assertEqual(T.zone_of(0.0, 0.3), 4)
        zones = {T.zone_of(x, y) for x in (0.05, 0.35, 0.65, 0.95) for y in (0.05, 0.35, 0.65, 0.95)}
        self.assertEqual(zones, set(range(16)))

    def test_written_file_is_loadable_and_carries_the_seed(self) -> None:
        grid = T.build_grid16(W, H, seed=7)
        geometry = {"width_px": W, "height_px": H, "dpi_scale": 1.25, "screen_id": "x", "orientation": "LANDSCAPE"}
        with tempfile.TemporaryDirectory() as tmp:
            path = T.write_targets_file(Path(tmp) / "g.json", grid, geometry, seed=7)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 7)
            targets, geo = C.load_targets(path)
            self.assertEqual(len(targets), 16)
            self.assertEqual(geo["width_px"], W)


# --- gf_setup ----------------------------------------------------------------


class ManifestTests(unittest.TestCase):
    def _manifest(self, rig: C.RigGeometry = RIG) -> U.SetupManifest:
        return U.build_manifest(rig, skip_camera=True, declared_extra={"eye_distance_cm": 60.0})

    def test_manifest_separates_measured_declared_and_derived(self) -> None:
        m = self._manifest()
        self.assertIn("screen", m.measured)
        self.assertIn("machine", m.measured)
        self.assertEqual(m.declared["camera_y_cm"], 63.6)
        self.assertEqual(m.declared["eye_distance_cm"], 60.0)
        self.assertIsNotNone(m.derived["code_sha256"])
        self.assertIn("numpy", m.derived["libraries"])

    def test_code_hash_changes_when_a_module_changes(self) -> None:
        first = U.code_fingerprint(("gf_common.py",))["combined"]
        second = U.code_fingerprint(("gf_common.py", "gf_schema.py"))["combined"]
        self.assertNotEqual(first, second)
        self.assertEqual(first, U.code_fingerprint(("gf_common.py",))["combined"])

    def test_declared_screen_size_is_cross_checked_against_the_os(self) -> None:
        wrong = C.RigGeometry(60.0, 63.6, 60.0, 33.75, 5120, 1440)  # half the real width
        warnings = U.check_consistency(self._manifest(wrong), wrong)
        self.assertTrue(any("screen width" in w for w in warnings), warnings)

    def test_declared_resolution_mismatch_is_flagged(self) -> None:
        wrong = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 1920, 1080)
        warnings = U.check_consistency(self._manifest(wrong), wrong)
        self.assertTrue(any("device resolution" in w for w in warnings), warnings)

    def test_camera_inside_the_screen_area_is_flagged(self) -> None:
        inside = C.RigGeometry(60.0, 10.0, 120.0, 33.75, 5120, 1440)
        warnings = U.check_consistency(self._manifest(inside), inside)
        self.assertTrue(any("within the screen area" in w for w in warnings), warnings)

    def test_correct_rig_produces_no_geometry_warnings(self) -> None:
        warnings = self._manifest().warnings
        self.assertFalse([w for w in warnings if "screen" in w or "resolution" in w], warnings)

    def test_round_trip_and_version_refusal(self) -> None:
        m = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = m.save(Path(tmp) / "setup.json")
            back = U.SetupManifest.load(path)
            self.assertEqual(back.declared, m.declared)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["manifest_version"] = "setup-0"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                U.SetupManifest.load(path)

    def test_moving_the_camera_breaks_comparability(self) -> None:
        before = self._manifest()
        after = U.build_manifest(C.RigGeometry(60.0, 55.0, 120.0, 33.75, 5120, 1440), skip_camera=True)
        comparison = U.compare_manifests(before, after)
        self.assertFalse(comparison["comparable"])
        self.assertIn("declared.camera_y_cm", [c["field"] for c in comparison["critical_changes"]])

    def test_identical_setup_is_comparable(self) -> None:
        self.assertTrue(U.compare_manifests(self._manifest(), self._manifest())["comparable"])

    def test_dotted_lookup(self) -> None:
        m = self._manifest()
        self.assertEqual(m.get("declared.camera_x_cm"), 60.0)
        self.assertIsNone(m.get("declared.nope"))
        self.assertIsNone(m.get("nope.nope"))


# --- gf_geometry -------------------------------------------------------------


def _face_at(distance_mm: float, *, dx_mm: float = 0.0, dy_mm: float = 0.0) -> SimpleNamespace:
    """Project the PnP model face so its EYE MIDPOINT is ``distance_mm`` away.

    The model's eye corners sit 135 mm behind its nose tip, so anchoring on the
    nose would plant the eyes somewhere else and the test would be checking the
    fixture's arithmetic rather than the solver's.
    """

    img_w, img_h = 640, 480
    focal = float(img_w)
    pts = G.H._PNP_MODEL_MM.copy()
    eye_z = (pts[2, 2] + pts[3, 2]) / 2.0
    cam = np.stack([pts[:, 0] + dx_mm, -pts[:, 1] + dy_mm, -(pts[:, 2] - eye_z) + distance_mm], axis=1)
    u = focal * cam[:, 0] / cam[:, 2] + img_w / 2.0
    v = focal * cam[:, 1] / cam[:, 2] + img_h / 2.0
    landmarks = np.zeros((478, 3), dtype=np.float64)
    rng = np.random.default_rng(0)
    landmarks[:, :2] = rng.normal(loc=(img_w / 2, img_h / 2), scale=30.0, size=(478, 2))
    for row, index in enumerate(G.H._PNP_INDICES):
        landmarks[index, :2] = (u[row], v[row])
    return SimpleNamespace(
        status=True, can_gaze_estimation=True, face_landmarks=landmarks, img_w=img_w, img_h=img_h
    )


class ScreenFrameTests(unittest.TestCase):
    def test_target_millimetres_match_the_label_space(self) -> None:
        frame = G.ScreenFrame(RIG)
        top = frame.target_mm(0.5, 0.0)
        bottom = frame.target_mm(0.5, 1.0)
        self.assertAlmostEqual(top[1], 636.0, delta=0.1)  # 63.6 cm above the camera
        self.assertAlmostEqual(bottom[1], 298.5, delta=0.1)
        self.assertAlmostEqual(top[2], 0.0)

    def test_camera_axes_are_flipped_to_screen_axes(self) -> None:
        frame = G.ScreenFrame(RIG)
        # OpenCV +y is down; the screen frame's +y is up.
        out = frame.camera_to_screen_frame(np.array([10.0, 20.0, 600.0]))
        self.assertAlmostEqual(out[1], -20.0)
        self.assertFalse(frame.tilt_corrected)

    def test_tilt_correction_rotates_and_is_recorded(self) -> None:
        frame = G.ScreenFrame(RIG, camera_pitch_deg=20.0)
        self.assertTrue(frame.tilt_corrected)
        out = frame.camera_to_screen_frame(np.array([0.0, 0.0, 600.0]))
        # A point straight ahead of a camera tilted up 20 deg is above the
        # camera once the tilt is undone.
        self.assertGreater(out[1], 100.0)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 600.0, delta=1e-6)


class AngularErrorTests(unittest.TestCase):
    def test_zero_error_when_prediction_equals_target(self) -> None:
        frame = G.ScreenFrame(RIG)
        result = G.angular_error(frame, (0.5, 0.5), (0.5, 0.5), eye_distance_cm=60.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.degrees, 0.0, places=9)
        self.assertTrue(result.estimated)

    def test_known_offset_gives_the_expected_angle(self) -> None:
        # 60 cm away, a 15.7 mm offset subtends about 1.5 degrees.
        frame = G.ScreenFrame(RIG)
        eye = G.nominal_eye_mm(frame, 60.0)
        a = frame.target_mm(0.5, 0.5)
        b = a + np.array([15.7, 0.0, 0.0])
        self.assertAlmostEqual(G.angle_between(eye, a, b), 1.5, delta=0.05)

    def test_angle_is_never_derived_without_an_eye_position(self) -> None:
        frame = G.ScreenFrame(RIG)
        self.assertIsNone(G.angular_error(frame, (0.5, 0.5), (0.6, 0.5)))

    def test_the_same_pixel_error_is_more_degrees_when_closer(self) -> None:
        frame = G.ScreenFrame(RIG)
        near = G.angular_error(frame, (0.4, 0.5), (0.6, 0.5), eye_distance_cm=40.0)
        far = G.angular_error(frame, (0.4, 0.5), (0.6, 0.5), eye_distance_cm=80.0)
        self.assertGreater(near.degrees, far.degrees)

    def test_pnp_recovers_a_planted_distance(self) -> None:
        frame = G.ScreenFrame(RIG)
        eye = G.pnp_eye_mm(_face_at(600.0), frame)
        self.assertIsNotNone(eye)
        self.assertAlmostEqual(abs(float(eye[2])), 600.0, delta=40.0)

    def test_pnp_tracks_head_translation(self) -> None:
        frame = G.ScreenFrame(RIG)
        centre = G.pnp_eye_mm(_face_at(600.0), frame)
        moved = G.pnp_eye_mm(_face_at(600.0, dx_mm=50.0), frame)
        self.assertIsNotNone(centre)
        self.assertIsNotNone(moved)
        self.assertAlmostEqual(float(moved[0] - centre[0]), 50.0, delta=10.0)

    def test_pnp_refuses_implausible_distances_and_invalid_faces(self) -> None:
        frame = G.ScreenFrame(RIG)
        self.assertIsNone(G.pnp_eye_mm(_face_at(3000.0), frame))
        bad = _face_at(600.0)
        bad.status = False
        self.assertIsNone(G.pnp_eye_mm(bad, frame))

    def test_rows_helper_falls_back_to_nominal_and_marks_gaps(self) -> None:
        frame = G.ScreenFrame(RIG)
        targets = np.array([[0.5, 0.5], [0.5, 0.5], [np.nan, np.nan]])
        preds = np.array([[0.5, 0.5], [0.6, 0.5], [0.5, 0.5]])
        degrees = G.angular_errors_for_rows(frame, targets, preds, eye_distance_cm=60.0)
        self.assertAlmostEqual(degrees[0], 0.0, places=9)
        self.assertGreater(degrees[1], 0.0)
        self.assertTrue(math.isnan(degrees[2]))

    def test_summary_is_always_labelled_estimated_and_unverified(self) -> None:
        summary = G.summarise_degrees(np.array([1.0, 2.0, 3.0]), source="pnp", tilt_corrected=False, scale_validated=False)
        self.assertTrue(summary["estimated"])
        self.assertFalse(summary["verified"])
        self.assertIn("NOT VERIFIED", summary["caveat"])
        self.assertAlmostEqual(summary["median_deg"], 2.0)
        verified = G.summarise_degrees(np.array([1.0]), source="pnp", tilt_corrected=True, scale_validated=True)
        self.assertTrue(verified["verified"])
        self.assertTrue(verified["estimated"])  # still an estimate, just a checked one


class ScaleValidationTests(unittest.TestCase):
    def test_consistent_scale_is_detected(self) -> None:
        measured = [500.0, 600.0, 700.0]
        truth = [550.0, 660.0, 770.0]  # a uniform 1.1x
        result = G.validate_scale(measured, truth)
        self.assertAlmostEqual(result["scale_factor"], 1.1, places=6)
        self.assertTrue(result["consistent"])
        self.assertLess(result["max_residual_mm"], 1e-6)

    def test_inconsistent_scale_is_flagged(self) -> None:
        result = G.validate_scale([500.0, 600.0, 700.0], [550.0, 900.0, 700.0])
        self.assertFalse(result["consistent"])

    def test_bad_input_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            G.validate_scale([500.0], [500.0])
        with self.assertRaises(ValueError):
            G.validate_scale([0.0, 600.0], [500.0, 600.0])


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()


class BoundedCameraProbeTests(unittest.TestCase):
    """A blocking camera call must not be able to hang start-up.

    The earlier version checked a deadline BETWEEN frames, which bounds the
    loop but not the call: OpenCV's ``VideoCapture()`` and ``read()`` block
    with no timeout, so a camera another process holds parks one call forever
    and the deadline is never reached to be tested.
    """

    def test_call_with_timeout_returns_while_the_call_is_still_blocked(self) -> None:
        import time as _time

        started = _time.monotonic()
        finished, result = C.call_with_timeout(lambda: _time.sleep(30), timeout_s=0.2)
        self.assertLess(_time.monotonic() - started, 5.0)
        self.assertFalse(finished)
        self.assertIsInstance(result, TimeoutError)

    def test_call_with_timeout_passes_a_value_through(self) -> None:
        finished, result = C.call_with_timeout(lambda: 42, timeout_s=5.0)
        self.assertTrue(finished)
        self.assertEqual(result, 42)

    def test_call_with_timeout_reports_an_exception_rather_than_swallowing_it(self) -> None:
        def boom():
            raise RuntimeError("camera exploded")

        finished, result = C.call_with_timeout(boom, timeout_s=5.0)
        self.assertTrue(finished)
        self.assertIsInstance(result, RuntimeError)

    def test_probe_reports_a_stuck_camera_instead_of_hanging(self) -> None:
        import time as _time

        original = U._probe_camera_blocking
        U._probe_camera_blocking = lambda index, frames, deadline: _time.sleep(30)
        try:
            started = _time.monotonic()
            info = U.probe_camera(timeout_s=0.2)
        finally:
            U._probe_camera_blocking = original
        self.assertLess(_time.monotonic() - started, 8.0)
        self.assertTrue(info["timed_out"])
        self.assertIsNone(info["opened"])
        self.assertIn("holding it", info["note"])
        # A stuck camera must not look like a missing measurement.
        self.assertNotIn("measured_fps", info)

    def test_probe_reports_an_exception_from_the_camera_layer(self) -> None:
        original = U._probe_camera_blocking

        def boom(index, frames, deadline):
            raise OSError("device busy")

        U._probe_camera_blocking = boom
        try:
            info = U.probe_camera(timeout_s=1.0)
        finally:
            U._probe_camera_blocking = original
        self.assertFalse(info["opened"])
        self.assertIn("device busy", info["error"])

    def test_a_slow_camera_still_yields_a_measurement(self) -> None:
        # Merely slow, not stuck: the between-frames deadline handles this and
        # a rate is still reported, so the operator learns the real number.
        original = U._probe_camera_blocking
        U._probe_camera_blocking = lambda index, frames, deadline: {
            "opened": True, "measured_fps": 1.0, "measured_frames": 3,
            "stopped_early": "deadline reached between frames",
        }
        try:
            info = U.probe_camera(timeout_s=1.0)
        finally:
            U._probe_camera_blocking = original
        self.assertAlmostEqual(info["measured_fps"], 1.0)
        self.assertNotIn("timed_out", info)


class CentralBandTests(unittest.TestCase):
    """A narrowed calibration grid, for the central-band experiment.

    The model's raw horizontal output goes flat beyond roughly a third of the
    screen from the centre, so calibrating at the library's edge points trains
    on a signal that is not there. A banded run narrows the grid with it -- and
    is therefore NOT a faithful reproduction of the library's calibration,
    which the recording records.
    """

    def test_default_grid_is_unchanged(self) -> None:
        self.assertEqual(C.nine_point_sequence(), C.NINE_POINT_SEQUENCE)

    def test_narrow_grid_keeps_the_display_order_and_the_warm_up(self) -> None:
        seq = C.nine_point_sequence((0.3, 0.5, 0.7))
        self.assertEqual(len(seq), 10)
        self.assertEqual(seq[0], (0.5, 0.5))          # warm-up, not stored
        self.assertEqual(seq[-1], (0.5, 0.5))         # stored centre, last
        xs = {x for x, _ in seq}
        self.assertEqual(xs, {0.3, 0.5, 0.7})
        self.assertEqual({y for _, y in seq}, set(C.GRID_Y))
        self.assertEqual(len(set(seq[1:])), 9)

    def test_proximity_is_judged_against_the_grid_actually_used(self) -> None:
        # A target at x=0.35 is far from the library's edge grid but close to
        # a 0.3/0.5/0.7 one; judging it against the wrong grid would call it
        # held-out when it is not.
        target = [{"index": 0, "name": "C_0", "screen_position": {"x": 0.33, "y": 0.5}}]
        wide = T.annotate_targets(target, W, H)[0]
        narrow = T.annotate_targets(target, W, H, calibration=T.calibration_points((0.3, 0.5, 0.7)))[0]
        self.assertGreater(wide["distance_to_calibration_px"], narrow["distance_to_calibration_px"])
        self.assertTrue(narrow["near_calibration"])
        self.assertFalse(wide["near_calibration"])

    def test_calibration_points_default_to_the_library_grid(self) -> None:
        self.assertEqual(T.calibration_points(), list(C.NINE_POINT_STORED))

    def test_the_shipped_central_target_files_stay_inside_the_band(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for name, expected in (("targets_central.json", 10), ("targets_tune_central.json", 8)):
            targets, geometry = C.load_targets(root / name)
            self.assertEqual(len(targets), expected, name)
            for t in targets:
                x = float(t["screen_position"]["x"])
                self.assertGreaterEqual(x, 0.3, f"{name} {t['name']}")
                self.assertLessEqual(x, 0.7, f"{name} {t['name']}")
            self.assertEqual(int(geometry["width_px"]), 4096)

    def test_tuning_and_test_targets_do_not_overlap(self) -> None:
        # The configuration is chosen on TUNE and scored once on the test set;
        # a shared point would leak the answer into the choice.
        root = Path(__file__).resolve().parent.parent
        t1, _ = C.load_targets(root / "targets_central.json")
        tune, _ = C.load_targets(root / "targets_tune_central.json")
        for a in tune:
            for b in t1:
                d = C.distance_px(
                    a["screen_position"]["x"], a["screen_position"]["y"],
                    b["screen_position"]["x"], b["screen_position"]["y"], W, H,
                )
                self.assertGreaterEqual(d, T.NEAR_CALIBRATION_PX, f"{a['name']} vs {b['name']}")
