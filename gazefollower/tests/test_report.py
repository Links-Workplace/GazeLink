"""gf_report: segment weighting, missing cells, proximity split."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_report as RP  # noqa: E402


def _segment(tid: int, x: float, y: float, *, euclid: float, dx: float = 0.0, dy: float = 0.0,
             near: bool = False, condition: str = "static_neutral", n_eligible: int = 40,
             n_valid: int | None = None, deg: float | None = None) -> RP.SegmentResult:
    return RP.SegmentResult(
        target_id=tid, target_name=f"T{tid}", target_x=x, target_y=y, condition=condition,
        near_calibration=near, n_eligible=n_eligible,
        n_valid=n_eligible if n_valid is None else n_valid,
        median_dx_px=dx, median_dy_px=dy, median_euclid_px=euclid, median_deg=deg,
    )


class SegmentWeightingTests(unittest.TestCase):
    def test_a_long_segment_does_not_outvote_a_short_one(self) -> None:
        # 1000 frames at 10 px and 10 frames at 500 px: frame-weighting would
        # report ~15 px, segment-weighting reports the median of {10, 500}.
        segments = [
            _segment(0, 0.2, 0.2, euclid=10.0, n_eligible=1000),
            _segment(1, 0.8, 0.8, euclid=500.0, n_eligible=10),
        ]
        summary = RP.aggregate(segments)
        self.assertEqual(summary["n_segments"], 2)
        self.assertAlmostEqual(summary["euclid_px"]["median"], 255.0)
        self.assertIn("one vote per segment", summary["weighting"])

    def test_coverage_counts_frames_not_segments(self) -> None:
        segments = [_segment(0, 0.2, 0.2, euclid=10.0, n_eligible=100, n_valid=50)]
        summary = RP.aggregate(segments)
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["n_eligible"], 100)

    def test_empty_input_is_reported_not_averaged(self) -> None:
        self.assertEqual(RP.aggregate([])["n_segments"], 0)

    def test_percentiles_present(self) -> None:
        segments = [_segment(i, 0.5, 0.5, euclid=float(i)) for i in range(1, 11)]
        stats = RP.aggregate(segments)["euclid_px"]
        for key in ("mean", "median", "p90", "p95", "max"):
            self.assertIsNotNone(stats[key])
        self.assertAlmostEqual(stats["max"], 10.0)


class ZoneTests(unittest.TestCase):
    def test_a_zone_without_a_segment_is_missing_not_a_pass(self) -> None:
        rows = RP.by_zone([_segment(0, 0.05, 0.05, euclid=10.0)])
        self.assertEqual(len(rows), 16)
        measured = [r for r in rows if r["status"] == "measured"]
        missing = [r for r in rows if r["status"] == RP.MISSING]
        self.assertEqual(len(measured), 1)
        self.assertEqual(len(missing), 15)
        self.assertEqual(measured[0]["zone"], 0)

    def test_every_zone_covered_when_the_grid_is_complete(self) -> None:
        segments = [
            _segment(i, x, y, euclid=10.0)
            for i, (x, y) in enumerate((x, y) for y in (0.05, 0.35, 0.65, 0.95) for x in (0.05, 0.35, 0.65, 0.95))
        ]
        rows = RP.by_zone(segments)
        self.assertFalse([r for r in rows if r["status"] == RP.MISSING])


class MatrixTests(unittest.TestCase):
    META = [{"index": 0, "name": "T0", "screen_position": {"x": 0.2, "y": 0.2}},
            {"index": 1, "name": "T1", "screen_position": {"x": 0.8, "y": 0.8}}]

    def test_missing_cells_are_listed_and_block_completeness(self) -> None:
        segments = [_segment(0, 0.2, 0.2, euclid=10.0, condition="yaw_plus")]
        matrix = RP.target_condition_matrix(segments, ("yaw_plus", "yaw_minus"), self.META)
        self.assertFalse(matrix["complete"])
        self.assertEqual(len(matrix["cells"]), 4)
        self.assertEqual(len(matrix["missing"]), 3)
        measured = [c for c in matrix["cells"] if c["status"] == "measured"]
        self.assertEqual(len(measured), 1)

    def test_complete_matrix(self) -> None:
        segments = [
            _segment(tid, 0.2, 0.2, euclid=10.0, condition=cond)
            for tid in (0, 1)
            for cond in ("yaw_plus", "yaw_minus")
        ]
        matrix = RP.target_condition_matrix(segments, ("yaw_plus", "yaw_minus"), self.META)
        self.assertTrue(matrix["complete"])


class ProximityTests(unittest.TestCase):
    def test_near_and_far_are_reported_separately(self) -> None:
        segments = [
            _segment(0, 0.05, 0.05, euclid=20.0, near=True),
            _segment(1, 0.35, 0.35, euclid=200.0, near=False),
        ]
        split = RP.split_by_calibration_proximity(segments)
        self.assertAlmostEqual(split["near_calibration"]["euclid_px"]["median"], 20.0)
        self.assertAlmostEqual(split["far_from_calibration"]["euclid_px"]["median"], 200.0)
        self.assertIn("no generalisation claim", split["note"])


class PresentationTests(unittest.TestCase):
    def test_single_showing_is_flagged_when_two_are_required(self) -> None:
        segments = [_segment(0, 0.2, 0.2, euclid=10.0), _segment(1, 0.8, 0.8, euclid=10.0), _segment(1, 0.8, 0.8, euclid=12.0)]
        counts = RP.presentation_counts(segments, required=2)
        self.assertFalse(counts["satisfied"])
        self.assertEqual(len(counts["under_required"]), 1)
        self.assertEqual(counts["under_required"][0]["target_id"], 0)


class SegmentsFromRowsTests(unittest.TestCase):
    def test_rows_become_one_segment_per_presentation(self) -> None:
        n = 12
        target_id = np.array([0] * 6 + [1] * 6)
        presentation = np.array([0] * 3 + [1] * 3 + [0] * 3 + [1] * 3)
        eligible = np.ones(n, dtype=bool)
        valid = np.ones(n, dtype=bool)
        valid[0] = False  # one lost frame
        meta = [{"index": 0, "name": "A", "screen_position": {"x": 0.2, "y": 0.2}, "near_calibration": True},
                {"index": 1, "name": "B", "screen_position": {"x": 0.8, "y": 0.8}, "near_calibration": False}]
        segments = RP.segments_from_rows(
            target_id=target_id, target_xy=np.tile([0.5, 0.5], (n, 1)), eligible=eligible, valid=valid,
            dx_px=np.full(n, 10.0), dy_px=np.full(n, -20.0), degrees=np.full(n, 1.2),
            targets_meta=meta, presentation=presentation,
        )
        self.assertEqual(len(segments), 4)
        first = segments[0]
        self.assertEqual(first.n_eligible, 3)
        self.assertEqual(first.n_valid, 2)  # the lost frame is counted, not dropped
        self.assertTrue(first.near_calibration)
        self.assertAlmostEqual(first.median_dy_px, -20.0)
        self.assertAlmostEqual(first.median_deg, 1.2)
        self.assertEqual(first.zone, RP.T.zone_of(0.2, 0.2))

    def test_a_target_with_no_eligible_rows_yields_no_segment(self) -> None:
        segments = RP.segments_from_rows(
            target_id=np.array([0, 0]), target_xy=np.tile([0.5, 0.5], (2, 1)),
            eligible=np.zeros(2, dtype=bool), valid=np.zeros(2, dtype=bool),
            dx_px=np.zeros(2), dy_px=np.zeros(2), degrees=None,
            targets_meta=[{"index": 0, "name": "A", "screen_position": {"x": 0.5, "y": 0.5}}],
        )
        self.assertEqual(segments, [])


class FormatTests(unittest.TestCase):
    def test_report_renders_and_marks_missing_zones(self) -> None:
        meta = [{"index": 0, "name": "T0", "screen_position": {"x": 0.05, "y": 0.05}, "near_calibration": True}]
        report = RP.build_report(
            protocol="GRID16",
            segments=[_segment(0, 0.05, 0.05, euclid=120.0, dx=30.0, dy=-90.0, near=True, deg=1.4)],
            targets_meta=meta,
        )
        text = RP.format_report(report)
        self.assertIn("GRID16", text)
        self.assertIn("MISSING", text)  # 15 zones have no segment
        self.assertIn("ESTIMATED", text)
        self.assertIn("bias", text)

    def test_empty_report_says_so(self) -> None:
        text = RP.format_report(RP.build_report(protocol="T1", segments=[], targets_meta=[]))
        self.assertIn("no segments", text)


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
