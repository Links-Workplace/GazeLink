"""The preregistered gates, evaluated as code rather than argued afterwards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_select as SEL  # noqa: E402
import gf_session as SESS  # noqa: E402


def _score(round_id="s1", *, mean=1.2, median=1.1, p90=2.0, avail=0.99, fps=32.0,
           off=0.0, regions=None, protocol="FULL") -> SESS.SessionScore:
    regions = regions or {r: {"median_deg": median, "mean_deg": mean, "n_targets": 4}
                          for r in ("centre", "edge-h", "edge-v", "corner")}
    return SESS.SessionScore(
        round_id=round_id, protocol=protocol, n_eligible=450, n_valid=int(450 * avail),
        availability=avail, fps_median=fps, offscreen_rate=off, mean_deg=mean, median_deg=median,
        p90_deg=p90, p95_deg=p90 * 1.1, p99_deg=p90 * 1.3, max_deg=p90 * 1.6,
        median_px=median * 46, p90_px=p90 * 46, by_region=regions,
    )


def _outcome(refused=False) -> SEL.SelectionOutcome:
    return SEL.SelectionOutcome(
        selected=None if refused else "central-band-svr", mode="preset", reports=[],
        ranking_metric="median_px", thresholds=SEL.ScreeningThresholds(),
        preset_note="no acceptable calibration" if refused else "passed screening",
    )


class Gate1Tests(unittest.TestCase):
    def test_passes_when_every_session_found_a_calibration(self) -> None:
        self.assertEqual(SESS.gate1_calibration_safety([_outcome()] * 3).status, "PASS")

    def test_one_refusal_fails_the_gate(self) -> None:
        g = SESS.gate1_calibration_safety([_outcome(), _outcome(refused=True), _outcome()])
        self.assertEqual(g.status, "FAIL")
        self.assertIn("NO ACCEPTABLE CALIBRATION", g.note)


class Gate2Tests(unittest.TestCase):
    def test_three_consistent_sessions_pass(self) -> None:
        full = [_score(f"s{i}", median=1.1 + 0.1 * i) for i in range(3)]
        g = SESS.gate2_repeatability(full, SESS.gate1_calibration_safety([_outcome()] * 3))
        self.assertEqual(g.status, "PASS", [c for c in g.checks if c["status"] != "PASS"])

    def test_a_wide_spread_between_sessions_fails(self) -> None:
        full = [_score("s1", median=1.0), _score("s2", median=1.2), _score("s3", median=2.9)]
        g = SESS.gate2_repeatability(full, SESS.gate1_calibration_safety([_outcome()] * 3))
        self.assertEqual(g.status, "FAIL")
        spread = next(c for c in g.checks if "spread" in c["check"])
        self.assertEqual(spread["status"], "FAIL")

    def test_low_availability_fails(self) -> None:
        full = [_score("s1", avail=0.80)] + [_score(f"s{i}") for i in (2, 3)]
        g = SESS.gate2_repeatability(full, SESS.gate1_calibration_safety([_outcome()] * 3))
        self.assertEqual(g.status, "FAIL")

    def test_fewer_than_three_sessions_is_unknown_not_pass(self) -> None:
        g = SESS.gate2_repeatability([_score("s1")], SESS.gate1_calibration_safety([_outcome()]))
        self.assertEqual(g.status, SESS.UNKNOWN)


class Gate3Tests(unittest.TestCase):
    def test_a_good_result_is_ready_for_a_supervised_trial(self) -> None:
        full = [_score(f"s{i}", mean=1.6, median=1.5, p90=3.0) for i in range(3)]
        move = [_score(f"s{i}", median=1.9, protocol="MOVE") for i in range(3)]
        self.assertEqual(SESS.gate3_interaction_readiness(full, move).status, "PASS")

    def test_a_dead_corner_fails_even_when_the_average_is_fine(self) -> None:
        regions = {"centre": {"median_deg": 0.8, "mean_deg": 0.8, "n_targets": 3},
                   "edge-h": {"median_deg": 1.0, "mean_deg": 1.0, "n_targets": 2},
                   "edge-v": {"median_deg": 1.0, "mean_deg": 1.0, "n_targets": 6},
                   "corner": {"median_deg": 5.5, "mean_deg": 5.5, "n_targets": 4}}
        full = [_score(f"s{i}", mean=1.2, median=1.1, regions=regions) for i in range(3)]
        g = SESS.gate3_interaction_readiness(full, [])
        self.assertEqual(g.status, "FAIL")
        corner = next(c for c in g.checks if c["check"].startswith("corner"))
        self.assertEqual(corner["status"], "FAIL")

    def test_head_movement_that_degrades_too_much_fails(self) -> None:
        full = [_score(f"s{i}", median=1.2) for i in range(3)]
        move = [_score(f"s{i}", median=2.8, protocol="MOVE") for i in range(3)]
        g = SESS.gate3_interaction_readiness(full, move)
        self.assertEqual(g.status, "FAIL")

    def test_missing_move_block_is_unknown_not_pass(self) -> None:
        full = [_score(f"s{i}", mean=1.5, median=1.4, p90=3.0) for i in range(3)]
        self.assertEqual(SESS.gate3_interaction_readiness(full, []).status, SESS.UNKNOWN)


class Gate4Tests(unittest.TestCase):
    def test_the_specification_target_is_one_and_a_half_degrees(self) -> None:
        self.assertEqual(SESS.GATE4_MAX_MEAN_DEG, 1.5)

    def test_latency_cannot_be_measured_so_the_gate_stays_open(self) -> None:
        full = [_score(f"s{i}", mean=1.0, median=0.9) for i in range(3)]
        g = SESS.gate4_specification(full, latency_p95_ms=None)
        self.assertEqual(g.status, SESS.UNKNOWN)
        self.assertIn("NOT MEASURABLE", g.note)

    def test_it_can_pass_once_latency_is_supplied(self) -> None:
        full = [_score(f"s{i}", mean=1.0, median=0.9) for i in range(3)]
        self.assertEqual(SESS.gate4_specification(full, latency_p95_ms=40.0).status, "PASS")

    def test_missing_the_accuracy_target_fails(self) -> None:
        full = [_score(f"s{i}", mean=2.2) for i in range(3)]
        self.assertEqual(SESS.gate4_specification(full, latency_p95_ms=40.0).status, "FAIL")


class DecisionTests(unittest.TestCase):
    def test_all_gates_passing_proceeds_to_interaction(self) -> None:
        full = [_score(f"s{i}", mean=1.0, median=0.9, p90=2.0) for i in range(3)]
        move = [_score(f"s{i}", median=1.2, protocol="MOVE") for i in range(3)]
        r = SESS.evaluate(full, move, [_outcome()] * 3, latency_p95_ms=40.0)
        self.assertTrue(r["decision"].startswith("PASS"))

    def test_accuracy_short_of_spec_is_partial_not_pass(self) -> None:
        full = [_score(f"s{i}", mean=1.8, median=1.7, p90=3.0) for i in range(3)]
        move = [_score(f"s{i}", median=2.0, protocol="MOVE") for i in range(3)]
        r = SESS.evaluate(full, move, [_outcome()] * 3, latency_p95_ms=40.0)
        self.assertTrue(r["decision"].startswith("PARTIAL"))

    def test_a_refused_calibration_stops_everything(self) -> None:
        full = [_score(f"s{i}") for i in range(3)]
        r = SESS.evaluate(full, [], [_outcome(refused=True)], latency_p95_ms=40.0)
        self.assertIn("NO ACCEPTABLE CALIBRATION", r["decision"])

    def test_no_sessions_is_not_measured_rather_than_a_verdict(self) -> None:
        r = SESS.evaluate([], [], [])
        self.assertIn("NOT MEASURED", r["decision"])

    def test_the_report_renders_gates_and_regions(self) -> None:
        full = [_score(f"s{i}", mean=1.0, median=0.9) for i in range(3)]
        text = SESS.format_report(SESS.evaluate(full, [], [_outcome()] * 3))
        self.assertIn("Decision:", text)
        self.assertIn("Gate 4", text)
        self.assertIn("corner", text)
        self.assertIn("UNKNOWN", text)


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
