"""Eye-closure probe logic: no camera, no model, no OS input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_gesture_probe as G  # noqa: E402


def _samples(spec: str, *, hz: float = 30.0, face: str | None = None) -> list[G.Sample]:
    """Build frames from a sketch: '.' eyes open, 'x' eyes shut.

    ``face`` is an optional same-length mask where '-' means the face was lost
    on that frame.
    """

    face = face or "+" * len(spec)
    step = 1.0 / hz
    return [
        G.Sample(t=i * step, face=face[i] != "-", shut=spec[i] == "x", left=0.0, right=0.0)
        for i in range(len(spec))
    ]


class ClosureRunTests(unittest.TestCase):
    def test_a_single_run_is_measured_from_first_to_last_shut_frame(self) -> None:
        runs = G.closure_runs(_samples("..xxxx.."))
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0], 3 * (1000.0 / 30.0), places=3)

    def test_separate_closures_are_counted_separately(self) -> None:
        self.assertEqual(len(G.closure_runs(_samples("..xx..xx..xx.."))), 3)

    def test_a_closure_still_running_at_the_end_is_not_lost(self) -> None:
        """The operator may still be holding when the block times out;
        dropping that run would silently discard the longest hold."""

        self.assertEqual(len(G.closure_runs(_samples("....xxxx"))), 1)

    def test_losing_the_face_ends_the_run(self) -> None:
        """'I cannot see you' is not evidence that your eyes are shut.

        Letting a lost face extend a closure is exactly how looking away or
        walking off would be read as a deliberate hold -- a false activation
        with nobody at the desk.
        """

        runs = G.closure_runs(_samples("xxxxxxxx", face="++----++"))
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertLess(run, 4 * (1000.0 / 30.0))

    def test_no_closures_gives_an_empty_list_not_a_zero(self) -> None:
        self.assertEqual(G.closure_runs(_samples("........")), [])


class BridgingTests(unittest.TestCase):
    """A sustained closure arrives fragmented: measured on a real person, a
    ~2000 ms hold came apart into runs of at most 1125 ms because the openness
    estimate jitters back across the threshold mid-hold."""

    def test_a_short_reopening_is_bridged_into_one_hold(self) -> None:
        # 4 shut, 1 open, 4 shut at 30 Hz: the gap is one frame, ~33 ms.
        runs = G.closure_runs(_samples("xxxx.xxxx"), bridge_ms=100.0)
        self.assertEqual(len(runs), 1)
        self.assertGreater(runs[0], 200.0)

    def test_without_bridging_the_same_hold_is_two_fragments(self) -> None:
        self.assertEqual(len(G.closure_runs(_samples("xxxx.xxxx"))), 2)

    def test_a_long_reopening_is_not_bridged(self) -> None:
        """Bridging must not fuse two separate blinks into one apparent hold."""

        runs = G.closure_runs(_samples("xx..........xx"), bridge_ms=100.0)
        self.assertEqual(len(runs), 2)

    def test_losing_the_face_is_never_bridged(self) -> None:
        """Bridging forgives a flickering openness estimate, not absence.

        If absence were bridged, looking away mid-hold -- or leaving the desk
        entirely -- would keep accumulating toward a confirm.
        """

        runs = G.closure_runs(_samples("xxxxxxxx", face="++----++"), bridge_ms=1000.0)
        self.assertEqual(len(runs), 2)


class VerdictTests(unittest.TestCase):
    def _block(self, runs_by_bridge: dict[int, list[float]], *, face_frames: int = 300) -> dict:
        return {
            "frames_with_face": face_frames,
            "by_bridge": {str(int(b)): G.runs_summary(runs) for b, runs in runs_by_bridge.items()},
        }

    def _ladder(self, runs: list[float]) -> dict[int, list[float]]:
        return {int(b): runs for b in G.BRIDGE_LADDER}

    def test_a_clear_gap_is_reported_as_separated(self) -> None:
        ok, message = G.verdict(
            self._block(self._ladder([120.0, 180.0, 150.0])),
            self._block(self._ladder([1800.0, 2100.0])),
        )
        self.assertTrue(ok)
        self.assertIn("bridge 0 ms", message)
        self.assertIn("180", message)

    def test_the_worst_blink_decides_not_the_median(self) -> None:
        """Most blinks being short is no comfort: the one long blink is the
        one that fires the gesture by accident."""

        ok, message = G.verdict(
            self._block(self._ladder([100.0, 110.0, 105.0, 2000.0])),
            self._block(self._ladder([1800.0, 2100.0])),
        )
        self.assertFalse(ok)
        self.assertIn("no bridge", message)

    def test_the_smallest_working_bridge_is_chosen(self) -> None:
        """Bridging is permission to ignore evidence that the eyes reopened,
        so the least of it that works is the right amount."""

        natural = self._block(
            {0: [120.0], 100: [120.0], 150: [120.0], 200: [120.0], 300: [120.0], 400: [900.0]}
        )
        deliberate = self._block(
            {0: [90.0], 100: [90.0], 150: [1800.0], 200: [1900.0], 300: [1900.0], 400: [1900.0]}
        )
        ok, message = G.verdict(natural, deliberate)
        self.assertTrue(ok)
        self.assertIn("bridge 150 ms", message)

    def test_a_signal_that_never_registers_a_hold_is_refused(self) -> None:
        ok, message = G.verdict(
            self._block(self._ladder([120.0])),
            self._block(self._ladder([])),
        )
        self.assertFalse(ok)
        self.assertIn("does not register", message)

    def test_no_face_in_a_block_is_refused_rather_than_scored(self) -> None:
        ok, message = G.verdict(
            self._block(self._ladder([120.0]), face_frames=0),
            self._block(self._ladder([1800.0])),
        )
        self.assertFalse(ok)
        self.assertIn("not tracked", message)

    def test_blinking_that_never_crosses_the_threshold_is_clean_separation(self) -> None:
        ok, message = G.verdict(
            self._block(self._ladder([])),
            self._block(self._ladder([1800.0, 2000.0])),
        )
        self.assertTrue(ok)
        self.assertIn("Clean separation", message)


if __name__ == "__main__":
    unittest.main()
