"""The wink probe's analysis. No camera, no OS input, no thresholds asserted.

The probe exists because every wink threshold so far was reasoned from
summaries and every one was wrong. So the thing under test here is the
grouping and the comparison -- whether the numbers put in front of a decision
are the right numbers -- and not any particular value.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_wink_probe as W  # noqa: E402


def sample(t: float, phase: str, left: float, right: float) -> W.Sample:
    return W.Sample(
        t=t, phase=phase, left=left * 200.0, right=right * 160.0, left_ratio=left, right_ratio=right
    )


def series(phase: str, pattern: list[tuple[float, float]], start: float = 0.0) -> list[W.Sample]:
    return [
        sample(start + i * 0.016, phase, left, right) for i, (left, right) in enumerate(pattern)
    ]


OPEN = (1.0, 1.0)
WINK = (0.5, 0.03)  # right collapses, left narrows with it -- the measured shape
BLINK = (0.05, 0.05)


class ClosureRunTests(unittest.TestCase):
    def test_consecutive_closing_frames_are_one_run(self) -> None:
        samples = series("WINK", [OPEN] * 3 + [WINK] * 10 + [OPEN] * 3)
        runs = W.closure_runs(samples, closing_under=0.7)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].frames, 10)

    def test_two_separated_closures_are_two_runs(self) -> None:
        samples = series("WINK", [OPEN] * 3 + [WINK] * 8 + [OPEN] * 5 + [WINK] * 8 + [OPEN] * 3)
        self.assertEqual(len(W.closure_runs(samples, closing_under=0.7)), 2)

    def test_a_closure_still_running_at_the_end_is_not_lost(self) -> None:
        """A wink held when the phase timer expires is still a wink."""

        samples = series("WINK", [OPEN] * 3 + [WINK] * 8)
        runs = W.closure_runs(samples, closing_under=0.7)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].frames, 8)

    def test_the_grouping_does_not_require_the_other_eye_to_be_open(self) -> None:
        """That assumption is what the probe exists to test, so imposing it
        here would hide the evidence needed to test it."""

        samples = series("BLINK", [OPEN] * 3 + [BLINK] * 8 + [OPEN] * 3)
        runs = W.closure_runs(samples, closing_under=0.7)
        self.assertEqual(len(runs), 1, "a two-eyed closure was dropped before it could be compared")

    def test_never_closing_produces_no_runs(self) -> None:
        self.assertEqual(W.closure_runs(series("WINK", [OPEN] * 20), closing_under=0.7), [])

    def test_the_deepest_frame_is_the_one_reported(self) -> None:
        pattern = [OPEN, (0.6, 0.4), (0.5, 0.03), (0.6, 0.4), OPEN]
        runs = W.closure_runs(series("WINK", pattern), closing_under=0.7)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].deepest_right, 0.03)
        self.assertAlmostEqual(runs[0].deepest_left, 0.5)


class SeparationTests(unittest.TestCase):
    def test_the_gap_tells_a_wink_from_a_blink(self) -> None:
        samples = series("WINK", [OPEN] * 3 + [WINK] * 10 + [OPEN] * 3)
        samples += series("BLINK", [OPEN] * 3 + [BLINK] * 6 + [OPEN] * 3, start=1.0)
        result = W.separation(W.closure_runs(samples, closing_under=0.7))
        wink_gap = result["WINK"]["openness_gap_left_over_right"]["median"]
        blink_gap = result["BLINK"]["openness_gap_left_over_right"]["median"]
        self.assertGreater(wink_gap, blink_gap)
        self.assertAlmostEqual(blink_gap, 1.0, places=1)

    def test_a_phase_with_no_closures_says_so_rather_than_averaging_nothing(self) -> None:
        samples = series("WINK", [OPEN] * 10)
        result = W.separation(W.closure_runs(samples, closing_under=0.7))
        self.assertEqual(result["WINK"], {"n": 0})

    def test_the_report_reads_when_a_phase_is_empty(self) -> None:
        text = W.format_report(
            {
                "frames": 0,
                "frames_per_phase": {"WINK": 0, "BLINK": 0},
                "runs": [],
                "separation": W.separation([]),
            }
        )
        self.assertIn("no closures recorded", text)

    def test_no_threshold_is_proposed_by_the_analysis(self) -> None:
        """A number fitted to the data it came from is not a measurement, and
        every threshold picked that way here has had to be picked again."""

        samples = series("WINK", [OPEN] * 3 + [WINK] * 10 + [OPEN] * 3)
        result = W.separation(W.closure_runs(samples, closing_under=0.7))
        flat = repr(result)
        for word in ("threshold", "recommend", "use_"):
            self.assertNotIn(word, flat)


class PrivacyTests(unittest.TestCase):
    def test_a_sample_row_is_two_numbers_and_where_they_came_from(self) -> None:
        row = sample(1.0, "WINK", 0.5, 0.03).as_row()
        self.assertEqual(len(row), 6)
        self.assertEqual(row[1], "WINK")
        self.assertTrue(all(isinstance(v, (int, float, str)) for v in row))

    def test_the_probe_states_that_it_emits_no_os_input(self) -> None:
        source = (Path(__file__).resolve().parent.parent / "gf_wink_probe.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("SendInput", "MOUSEEVENTF", "SetCursorPos", "CursorAdapter"):
            self.assertNotIn(forbidden, source)


class GapTests(unittest.TestCase):
    """Measured in the first probe run: a two-eyed closure reported a gap of
    inf, because the right ratio was zero and the division blew up. Infinitely
    asymmetric is the exact opposite of what a blink is."""

    def test_a_blink_where_both_eyes_hit_zero_is_not_infinitely_asymmetric(self) -> None:
        self.assertLess(W.openness_gap(0.0, 0.0), 2.0)

    def test_a_wink_where_the_right_hits_zero_is_still_large(self) -> None:
        self.assertGreater(W.openness_gap(0.5, 0.0), 100.0)

    def test_a_blink_sits_near_one(self) -> None:
        self.assertAlmostEqual(W.openness_gap(0.05, 0.05), 1.0, places=6)

    def test_a_left_wink_is_small_not_large(self) -> None:
        self.assertLess(W.openness_gap(0.03, 0.5), 1.0)

    def test_the_gap_never_returns_infinity(self) -> None:
        for left, right in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.03)):
            self.assertTrue(math.isfinite(W.openness_gap(left, right)), (left, right))
