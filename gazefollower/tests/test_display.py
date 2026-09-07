"""Monitor selection, viewing geometry, and full-screen target coverage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_display as D  # noqa: E402

ULTRAWIDE = D.MonitorInfo(0, r"\\.\DISPLAY1", 5120, 1440, 0, 0, True, 1192.0, 336.0)
SECOND = D.MonitorInfo(1, r"\\.\DISPLAY2", 1920, 1080, 5120, 0, False, 531.0, 299.0)


class MonitorSelectionTests(unittest.TestCase):
    def _pick(self, selector, monitors=(ULTRAWIDE, SECOND)):
        original = D.list_monitors
        D.list_monitors = lambda: list(monitors)
        try:
            return D.pick_monitor(selector)
        finally:
            D.list_monitors = original

    def test_none_selects_the_primary(self) -> None:
        self.assertEqual(self._pick(None).index, 0)

    def test_index_selects_exactly(self) -> None:
        self.assertEqual(self._pick(1).index, 1)
        self.assertEqual(self._pick("1").index, 1)

    def test_name_substring_selects(self) -> None:
        self.assertEqual(self._pick("DISPLAY2").index, 1)

    def test_unknown_index_and_name_are_refused_with_the_options(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._pick(7)
        self.assertIn("[0, 1]", str(ctx.exception))
        with self.assertRaises(ValueError):
            self._pick("nosuchmonitor")

    def test_ambiguous_name_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._pick("DISPLAY")
        self.assertIn("several", str(ctx.exception))

    def test_a_second_monitor_carries_a_non_zero_desktop_origin(self) -> None:
        # A window opened at (0,0) would land on the primary instead.
        self.assertEqual(SECOND.origin, (5120, 0))
        self.assertEqual(ULTRAWIDE.origin, (0, 0))

    def test_diagonal_and_aspect_are_derived(self) -> None:
        self.assertAlmostEqual(SECOND.aspect, 16 / 9, places=2)
        self.assertAlmostEqual(SECOND.diagonal_in, 24.0, delta=0.4)
        self.assertAlmostEqual(ULTRAWIDE.diagonal_in, 48.8, delta=0.5)

    def test_physical_size_may_be_unknown(self) -> None:
        bare = D.MonitorInfo(0, "x", 1920, 1080, 0, 0, True)
        self.assertIsNone(bare.diagonal_in)
        self.assertIsNone(bare.to_dict()["diagonal_in"])


class ViewingGeometryTests(unittest.TestCase):
    # 24-inch 16:9 at 73 cm, the planned setup.
    NEW = D.ViewingGeometry(53.1, 29.9, 1920, 1080, 73.0, camera_distance_cm=60.0)
    # The ultrawide, at the distance the earlier rounds were recorded at.
    REF = D.ViewingGeometry(120.0, 33.75, 4096, 1152, 60.0)

    def test_half_angles_match_the_planning_estimate(self) -> None:
        h, v = self.NEW.half_angles_deg()
        self.assertAlmostEqual(h, 20.0, delta=0.3)
        self.assertAlmostEqual(v, 11.6, delta=0.3)

    def test_one_and_a_half_degrees_is_about_seventy_pixels(self) -> None:
        self.assertAlmostEqual(self.NEW.deg_to_px(1.5), 69.0, delta=2.0)

    def test_degrees_per_pixel_shrink_with_distance(self) -> None:
        near = D.ViewingGeometry(53.1, 29.9, 1920, 1080, 50.0)
        self.assertGreater(near.px_to_deg_at_centre()[0], self.NEW.px_to_deg_at_centre()[0])

    def test_the_new_screen_stays_inside_the_angles_already_exercised(self) -> None:
        # The central-band run covered x in [0.3, 0.7] of a 120 cm screen.
        cmp = D.compare_angular_coverage(self.NEW, self.REF, reference_x_range=(0.3, 0.7))
        self.assertTrue(cmp["horizontal_within_reference"])
        self.assertTrue(cmp["vertical_within_reference"])
        self.assertIn("inside the range", cmp["verdict"])

    def test_a_wider_screen_is_flagged_as_asking_for_new_angles(self) -> None:
        wide = D.ViewingGeometry(120.0, 33.75, 4096, 1152, 60.0)
        cmp = D.compare_angular_coverage(wide, self.REF, reference_x_range=(0.3, 0.7))
        self.assertFalse(cmp["horizontal_within_reference"])
        self.assertIn("did not cover", cmp["verdict"])

    def test_horizontal_agreement_alone_does_not_settle_the_vertical(self) -> None:
        # Same horizontal half-angle, taller screen: vertical must fail on its own.
        taller = D.ViewingGeometry(53.1, 60.0, 1920, 1080, 73.0)
        cmp = D.compare_angular_coverage(taller, self.REF, reference_x_range=(0.3, 0.7))
        self.assertTrue(cmp["horizontal_within_reference"])
        self.assertFalse(cmp["vertical_within_reference"])

    def test_camera_distance_is_recorded_separately_from_viewing_distance(self) -> None:
        d = self.NEW.to_dict()
        self.assertEqual(d["viewing_distance_cm"], 73.0)
        self.assertEqual(d["camera_distance_cm"], 60.0)


class CoverageTargetTests(unittest.TestCase):
    def test_every_region_is_represented(self) -> None:
        targets = D.coverage_targets(1920, 1080)
        counts = D.region_summary(targets)
        for region in ("centre", "edge-h", "edge-v", "corner"):
            self.assertGreater(counts.get(region, 0), 0, f"{region} missing: {counts}")
        self.assertEqual(len(targets), len(D.EVAL_COORDS_X) * len(D.EVAL_COORDS_Y))

    def test_all_four_corners_are_covered(self) -> None:
        corners = {
            (round(t["screen_position"]["x"], 2), round(t["screen_position"]["y"], 2))
            for t in D.coverage_targets(1920, 1080)
            if t["region"] == "corner"
        }
        self.assertEqual(corners, {(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)})

    def test_order_is_shuffled_but_reproducible(self) -> None:
        a = [t["name"] for t in D.coverage_targets(1920, 1080, seed=5)]
        b = [t["name"] for t in D.coverage_targets(1920, 1080, seed=5)]
        c = [t["name"] for t in D.coverage_targets(1920, 1080, seed=6)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_proximity_to_the_calibration_grid_rides_along(self) -> None:
        targets = D.coverage_targets(1920, 1080)
        for t in targets:
            self.assertIn("distance_to_calibration_px", t)
            self.assertIn("near_calibration", t)

    def test_region_boundaries(self) -> None:
        self.assertEqual(D.region_of(0.5, 0.5), "centre")
        self.assertEqual(D.region_of(0.05, 0.5), "edge-h")
        self.assertEqual(D.region_of(0.5, 0.05), "edge-v")
        self.assertEqual(D.region_of(0.05, 0.95), "corner")

    def test_geometry_dict_matches_the_analyser_contract(self) -> None:
        g = D.geometry_dict(SECOND, 1.0)
        for key in ("screen_id", "width_px", "height_px", "dpi_scale", "orientation"):
            self.assertIn(key, g)
        self.assertEqual(g["width_px"], 1920)
        self.assertEqual(g["orientation"], "LANDSCAPE")


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
