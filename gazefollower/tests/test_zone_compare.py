"""The paired, per-zone comparison in gf_recal_compare.

A mistake in either of the two things tested here would make every zone number
meaningless while still looking plausible:

* **pairing** -- a frame one model lost must not be scored on the other side
  only, or a coverage difference is reported as an accuracy difference;
* **presentations** -- a target shown twice is two votes, and two targets are
  never merged into one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_fit as FIT  # noqa: E402
import gf_recal_compare as RC  # noqa: E402

W, H = 4096, 1152


def _rec(target_ids, positions, *, round_id=40, scale=1.25):
    target_ids = np.asarray(target_ids, dtype=np.int32)
    xy = np.array([positions[t] if t >= 0 else (np.nan, np.nan) for t in target_ids], dtype=float)
    return SimpleNamespace(
        protocol="T1",
        round_id=round_id,
        target_id=target_ids,
        target_xy=xy,
        targets=[{"index": t, "name": f"T{t}"} for t in sorted(positions)],
        meta={"target_geometry": {"width_px": W, "height_px": H, "dpi_scale": scale}},
    )


def _pred_with_error(rec, px):
    """Predictions that miss every target by exactly ``px`` logical px on x."""

    pred = rec.target_xy.copy()
    pred[:, 0] += px / (W - 1)
    return pred


class PresentationTests(unittest.TestCase):
    def test_a_target_shown_twice_is_two_segments(self) -> None:
        rec = _rec([0, 0, 0, -1, -1, 0, 0, 0], {0: (0.5, 0.2)})
        rows = rec.target_id >= 0
        segs = RC.paired_segments(_pred_with_error(rec, 10), rec, rows)
        self.assertEqual(len(segs), 2, "two presentations were merged into one vote")

    def test_tracking_dropouts_inside_one_presentation_stay_one_segment(self) -> None:
        """Found by review: gaps in the MASK are not new presentations.

        Rows leave the mask for tracking loss, a blink, or one model not
        predicting. Splitting there turned one showing of one target into
        several segments -- enough to clear ``min_segments`` and mark a zone
        measured on the evidence of a single presentation.
        """

        rec = _rec([0] * 12, {0: (0.5, 0.2)})
        rows = np.ones(12, dtype=bool)
        rows[[3, 4, 8]] = False  # tracking lost twice, mid-presentation
        segs = RC.paired_segments(_pred_with_error(rec, 10), rec, rows)
        self.assertEqual(len(segs), 1, "tracking gaps were counted as new presentations")
        self.assertEqual(segs[0].euclid_px.size, 9)

    def test_idle_rows_between_two_showings_still_separate_them(self) -> None:
        """The other half: the fix must not merge genuine repeats."""

        rec = _rec([0, 0, -1, 0, 0], {0: (0.5, 0.2)})
        segs = RC.paired_segments(_pred_with_error(rec, 10), rec, rec.target_id >= 0)
        self.assertEqual(len(segs), 2)

    def test_consecutive_different_targets_are_not_merged(self) -> None:
        rec = _rec([0, 0, 1, 1], {0: (0.35, 0.2), 1: (0.65, 0.8)})
        segs = RC.paired_segments(_pred_with_error(rec, 10), rec, rec.target_id >= 0)
        self.assertEqual([s.target_id for s in segs], [0, 1])

    def test_the_segment_carries_its_frames_and_position(self) -> None:
        rec = _rec([0, 0, 0], {0: (0.4, 0.9)})
        (seg,) = RC.paired_segments(_pred_with_error(rec, 30), rec, rec.target_id >= 0)
        self.assertEqual(seg.euclid_px.size, 3)
        self.assertAlmostEqual(seg.median_euclid_px, 30.0, places=6)
        self.assertEqual((seg.target_x, seg.target_y), (0.4, 0.9))


class PairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._eligible = FIT.eligible_rows
        FIT.eligible_rows = lambda rec: rec.target_id >= 0

    def tearDown(self) -> None:
        FIT.eligible_rows = self._eligible

    def test_a_frame_one_model_lost_is_scored_on_neither_side(self) -> None:
        rec = _rec([0] * 10, {0: (0.5, 0.2)})
        old = _pred_with_error(rec, 10)
        new = _pred_with_error(rec, 10)
        # The new model loses the first five frames -- where the old one was
        # terrible. Scored unpaired, the old side would carry those frames and
        # look far worse than it is.
        old[:5, 0] += 1000 / (W - 1)
        new[:5] = np.nan
        result = RC.zone_comparison([(old, new, rec)])
        self.assertEqual(result["n_paired"], 5)
        zone = next(r for r in result["before"] if r.get("n_frames"))
        self.assertAlmostEqual(zone["median_px"], 10.0, places=4)

    def test_recordings_are_pooled_into_one_zone_table(self) -> None:
        a = _rec([0] * 30, {0: (0.5, 0.2)}, round_id=30)
        b = _rec([0] * 30, {0: (0.5, 0.2)}, round_id=32)
        runs = [(_pred_with_error(r, 10), _pred_with_error(r, 20), r) for r in (a, b)]
        result = RC.zone_comparison(runs)
        zone = next(r for r in result["before"] if r.get("n_frames"))
        self.assertEqual(zone["n_segments"], 2)
        self.assertEqual(zone["status"], "measured", "two sessions did not verify the zone")

    def test_a_gappy_single_presentation_cannot_verify_a_zone(self) -> None:
        """End to end: the failure the review found, at the gate itself."""

        rec = _rec([0] * 80, {0: (0.5, 0.2)})
        old = _pred_with_error(rec, 10)
        new = _pred_with_error(rec, 5)
        new[[10, 11, 30, 31, 50, 51]] = np.nan  # the new model drops out three times
        result = RC.zone_comparison([(old, new, rec)])
        zone = next(r for r in result["before"] if r.get("n_frames"))
        self.assertEqual(zone["n_segments"], 1)
        self.assertIn(zone["zone"], result["gate"]["unverified"])
        self.assertFalse(result["gate"]["adopt"])

    def test_a_single_session_leaves_the_zone_unverified(self) -> None:
        rec = _rec([0] * 60, {0: (0.5, 0.2)})
        result = RC.zone_comparison([(_pred_with_error(rec, 10), _pred_with_error(rec, 5), rec)])
        self.assertIn(next(r for r in result["before"] if r.get("n_frames"))["zone"],
                      result["gate"]["unverified"])

    def test_recordings_at_different_pixel_scales_are_refused(self) -> None:
        a = _rec([0] * 30, {0: (0.5, 0.2)}, scale=1.25)
        b = _rec([0] * 30, {0: (0.5, 0.2)}, scale=1.0)
        runs = [(_pred_with_error(r, 10), _pred_with_error(r, 10), r) for r in (a, b)]
        with self.assertRaises(ValueError):
            RC.zone_comparison(runs)

    def test_the_operational_limit_is_converted_from_device_to_logical_px(self) -> None:
        rec = _rec([0] * 30, {0: (0.5, 0.2)}, scale=1.25)
        result = RC.zone_comparison(
            [(_pred_with_error(rec, 10), _pred_with_error(rec, 10), rec)],
            target_half_device_px=100.0,
        )
        self.assertAlmostEqual(result["target_half_logical_px"], 80.0)


if __name__ == "__main__":
    unittest.main()
