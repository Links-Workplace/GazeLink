"""Aggregate filter benchmark tests; no camera or GazeFollower import."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_filter_benchmark as B  # noqa: E402
import gf_gaze_filter as F  # noqa: E402


class FilterBenchmarkTests(unittest.TestCase):
    def test_replay_preserves_missing_rows_and_resets_recovery(self) -> None:
        points = np.array([[0.1, 0.1], [0.2, 0.2], [np.nan, np.nan], [0.9, 0.9]])
        times = np.array([0, 33_000_000, 66_000_000, 99_000_000], dtype=np.int64)
        settings = F.FilterSettings(width_px=1000, height_px=500)
        output, cpu_ms = B.replay(points, times, settings)
        self.assertTrue(np.all(np.isnan(output[2])))
        self.assertEqual(tuple(output[3]), (0.9, 0.9))
        self.assertGreaterEqual(cpu_ms, 0.0)

    def test_one_bad_fixation_cannot_carry_the_per_fixation_summary(self) -> None:
        """Nine steady fixations and one wild one: the pooled P95 follows the
        wild one, the per-fixation median must not."""

        steady = [[0.5, 0.5], [0.501, 0.5], [0.499, 0.5]]
        wild = [[0.1, 0.5], [0.9, 0.5], [0.5, 0.5]]
        points = np.array(steady * 9 + wild)
        target_id = np.repeat(np.arange(10), 3)
        rec = SimpleNamespace(
            target_id=target_id,
            rows_collecting=lambda: np.ones(30, dtype=bool),
        )
        metrics = B.jitter_metrics(points, rec, width=1000, height=1000)
        self.assertEqual(metrics["n_fixations"], 10)
        self.assertGreater(metrics["p95_radius_px"], 100.0)
        self.assertLess(metrics["median_of_fixation_p95_px"], 10.0)
        self.assertGreater(metrics["worst_fixation_p95_px"], 100.0)

    def test_jitter_is_measured_around_each_targets_own_median(self) -> None:
        points = np.array(
            [[0.1, 0.5], [0.11, 0.5], [0.09, 0.5], [0.9, 0.5], [0.91, 0.5], [0.89, 0.5]]
        )
        rec = SimpleNamespace(
            target_id=np.array([0, 0, 0, 1, 1, 1]),
            rows_collecting=lambda: np.ones(6, dtype=bool),
        )
        metrics = B.jitter_metrics(points, rec, width=1000, height=500)
        self.assertEqual(metrics["n"], 6)
        self.assertLess(metrics["p95_radius_px"], 11.0)
        self.assertGreater(metrics["p95_frame_jump_px"], 10.0)

    def test_a_repeated_target_is_two_fixations_not_one(self) -> None:
        """--repeat-first re-shows a target keeping its id.

        The two visits are 30s apart and the signal has drifted between them.
        Pooling them into one "fixation" reports that drift as jitter and
        hides the drift itself, which is the effect those runs exist to
        measure.
        """

        first_visit = [[0.30, 0.5], [0.301, 0.5], [0.299, 0.5]]
        elsewhere = [[0.70, 0.5], [0.701, 0.5], [0.699, 0.5]]
        second_visit = [[0.50, 0.5], [0.501, 0.5], [0.499, 0.5]]
        points = np.array(first_visit + elsewhere + second_visit)
        rec = SimpleNamespace(
            target_id=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
            rows_collecting=lambda: np.ones(9, dtype=bool),
        )
        metrics = B.jitter_metrics(points, rec, width=1000, height=1000)
        self.assertEqual(metrics["n_fixations"], 3)
        # Each visit is steady on its own; only pooling the two visits of
        # target 0 would invent a ~100px spread between them.
        self.assertLess(metrics["worst_fixation_p95_px"], 5.0)

    def test_a_repeated_target_does_not_invent_a_frame_jump(self) -> None:
        """The two visits are not adjacent frames, so no jump spans them."""

        points = np.array(
            [[0.3, 0.5]] * 3 + [[0.7, 0.5]] * 3 + [[0.5, 0.5]] * 3,
            dtype=np.float64,
        )
        rec = SimpleNamespace(
            target_id=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
            rows_collecting=lambda: np.ones(9, dtype=bool),
        )
        metrics = B.jitter_metrics(points, rec, width=1000, height=1000)
        self.assertEqual(metrics["p95_frame_jump_px"], 0.0)

    def test_support_activation_feeds_a_head_model_its_head_columns(self) -> None:
        """A head-aware model rejects an embedding-only design outright.

        Hardcoding "no head columns" here made the function silently usable
        only for embedding-only models, and the caller could not see that
        from the outside -- it surfaced as a schema ValueError deep in the
        model, on a path whose whole job is to report on model coverage.
        """

        seen: dict[str, int] = {}

        class Model:
            schema = SimpleNamespace(
                head_names=("roll_deg", "yaw_ratio"),
                svr={"gamma": 0.0005},
            )
            _svr_x = SimpleNamespace(getSupportVectors=lambda: np.zeros((7, 4)))

            def support_activation(self, design):
                seen["columns"] = design.shape[1]
                return np.ones(design.shape[0])

        rec = SimpleNamespace(
            n_rows=3,
            features=np.zeros((3, 4)),
            head=np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (3, 1)),
        )
        result = B.support_activation(Model(), rec)
        self.assertTrue(result["available"])
        self.assertEqual(seen["columns"], 6)

    def test_selection_never_uses_test_metrics(self) -> None:
        rows = [
            _tune_row("off", 20, 100, 200, 0),
            _tune_row("good", 8, 104, 209, 90),
            _tune_row("inaccurate", 1, 106, 200, 20),
        ]
        selected, verdict = B.choose(rows)
        self.assertEqual(selected["name"], "good")
        self.assertIn("50%", verdict)

    def test_selection_reads_the_worst_step_not_a_convenient_one(self) -> None:
        """A candidate fast on one step size but slow on another is refused."""

        rows = [
            _tune_row("off", 20, 100, 200, 0),
            _tune_row("steady_but_slow_somewhere", 4, 100, 200, 400),
            _tune_row("honest", 12, 100, 200, 80),
        ]
        selected, _ = B.choose(rows)
        self.assertEqual(selected["name"], "honest")

    def test_no_qualifying_candidate_returns_the_baseline_and_says_so(self) -> None:
        rows = [
            _tune_row("off", 20, 100, 200, 0),
            _tune_row("too_slow", 2, 100, 200, 5000),
        ]
        selected, verdict = B.choose(rows)
        self.assertEqual(selected["name"], "off")
        self.assertIn("no filtered candidate", verdict)

    def test_verdict_does_not_claim_the_jitter_goal_when_unmet(self) -> None:
        rows = [
            _tune_row("off", 100, 100, 200, 0),
            _tune_row("mild", 90, 100, 200, 50),
        ]
        _, verdict = B.choose(rows)
        self.assertIn("short of", verdict)
        self.assertNotIn("met the proposed", verdict)


class StepProfileTests(unittest.TestCase):
    def test_profile_covers_every_declared_case_and_reports_the_worst(self) -> None:
        settings = F.FilterSettings(width_px=5120, height_px=1440)
        profile = B.step_response_profile(settings, fps=30.0)
        self.assertEqual(len(profile["per_case_ms"]), len(B.STEP_CASES))
        measured = [v for v in profile["per_case_ms"].values() if v is not None]
        self.assertEqual(profile["worst_ms"], max(measured))

    def test_negative_and_vertical_steps_are_timed_not_skipped(self) -> None:
        settings = F.FilterSettings(width_px=5120, height_px=1440)
        profile = B.step_response_profile(settings, fps=30.0)
        self.assertIsNotNone(profile["per_case_ms"]["x_large_-0.40"])
        self.assertIsNotNone(profile["per_case_ms"]["y_mid_+0.20"])
        self.assertEqual(profile["unreached_cases"], [])

    def test_a_heavier_filter_is_never_reported_as_faster(self) -> None:
        common = {"width_px": 5120, "height_px": 1440}
        quick = F.FilterSettings(
            **common, one_euro_min_cutoff_hz=1.2, one_euro_beta_hz_per_px_s=0.004
        )
        heavy = F.FilterSettings(
            **common, one_euro_min_cutoff_hz=0.4, one_euro_beta_hz_per_px_s=0.0
        )
        self.assertLess(
            B.step_response_profile(quick)["worst_ms"],
            B.step_response_profile(heavy)["worst_ms"],
        )


class RecordedTransitionTests(unittest.TestCase):
    """The recorded transition must key off the signal's own medians, so a
    constant calibration offset cannot masquerade as slow movement."""

    def _recording(self, phases, target_ids, timestamps):
        return SimpleNamespace(
            target_id=np.asarray(target_ids),
            phase=np.asarray(phases),
            timestamp_ns=np.asarray(timestamps, dtype=np.int64),
            rows_collecting=lambda: np.asarray(phases) == "COLLECTING",
        )

    def test_a_constant_offset_does_not_change_the_measured_time(self) -> None:
        phases = ["COLLECTING"] * 4 + ["STABILIZING"] * 2 + ["COLLECTING"] * 4
        ids = [0] * 4 + [1] * 6
        times = [i * 33_000_000 for i in range(10)]
        rec = self._recording(phases, ids, times)
        near = np.array([[0.2, 0.5]] * 4 + [[0.5, 0.5], [0.8, 0.5]] + [[0.8, 0.5]] * 4)
        shifted = near + np.array([0.05, 0.05])
        first = B.recorded_transition_ms(near, rec, 1000, 1000, min_move_px=10.0)
        second = B.recorded_transition_ms(shifted, rec, 1000, 1000, min_move_px=10.0)
        self.assertEqual(first["median_ms"], second["median_ms"])

    def test_the_second_visit_to_a_target_is_timed_from_its_own_anchor(self) -> None:
        """With --repeat-first, keying the medians by target id keeps only the
        last visit, so the first transition into that target gets an anchor
        recorded much later.  Presentation 0 -> 1 must still be timed against
        presentation 0's own median."""

        phases = (
            ["COLLECTING"] * 3
            + ["STABILIZING"] * 2
            + ["COLLECTING"] * 3
            + ["STABILIZING"] * 2
            + ["COLLECTING"] * 3
        )
        ids = [0] * 3 + [1] * 5 + [0] * 5
        times = [i * 33_000_000 for i in range(13)]
        rec = self._recording(phases, ids, times)
        # Target 0 sits at x=0.2 on its first showing and, after drift, at
        # x=0.6 on its second.  Timing 0 -> 1 against the second showing's
        # median would call the move tiny and skip it.
        points = np.array(
            [[0.2, 0.5]] * 3
            + [[0.5, 0.5], [0.8, 0.5]]
            + [[0.8, 0.5]] * 3
            + [[0.7, 0.5], [0.6, 0.5]]
            + [[0.6, 0.5]] * 3
        )
        result = B.recorded_transition_ms(points, rec, 1000, 1000, min_move_px=100.0)
        self.assertEqual(result["n_transitions_measured"], 2)
        self.assertEqual(result["n_transitions_skipped_small_move"], 0)

    def test_a_pair_that_barely_moved_is_skipped_not_timed_as_instant(self) -> None:
        phases = ["COLLECTING"] * 3 + ["STABILIZING"] + ["COLLECTING"] * 3
        ids = [0] * 3 + [1] * 4
        times = [i * 33_000_000 for i in range(7)]
        rec = self._recording(phases, ids, times)
        still = np.array([[0.5, 0.5]] * 7)
        result = B.recorded_transition_ms(still, rec, 1000, 1000, min_move_px=50.0)
        self.assertEqual(result["n_transitions_skipped_small_move"], 1)
        self.assertEqual(result["n_transitions_measured"], 0)


class PresetGuardTests(unittest.TestCase):
    def test_default_one_euro_meets_synthetic_response_guard(self) -> None:
        settings = F.FilterSettings(width_px=5120, height_px=1440)
        response_ms = B.synthetic_step_ms(settings, fps=30.0)
        self.assertIsNotNone(response_ms)
        self.assertLessEqual(response_ms, 100.0)

    def test_responsive_preset_is_the_one_carrying_the_100ms_goal(self) -> None:
        min_cutoff, beta = F.ONE_EURO_PRESETS["responsive"]
        settings = F.FilterSettings(
            width_px=5120,
            height_px=1440,
            one_euro_min_cutoff_hz=min_cutoff,
            one_euro_beta_hz_per_px_s=beta,
        )
        self.assertLessEqual(B.synthetic_step_ms(settings, fps=30.0), 100.0)

    def test_high_stability_preset_is_documented_as_slow_not_as_meeting_the_goal(self) -> None:
        """The heavy preset trades responsiveness away on purpose.

        This asserts the trade is real, so nobody can quietly point the
        default at it and still claim the 100ms experimental goal.
        """

        min_cutoff, beta = F.ONE_EURO_PRESETS["high-stability"]
        settings = F.FilterSettings(
            width_px=5120,
            height_px=1440,
            one_euro_min_cutoff_hz=min_cutoff,
            one_euro_beta_hz_per_px_s=beta,
        )
        response_ms = B.synthetic_step_ms(settings, fps=30.0)
        self.assertGreater(response_ms, 100.0)
        self.assertLessEqual(response_ms, 1200.0)


def _tune_row(name, jitter, median, p95, worst_step):
    return {
        "name": name,
        "synthetic_step_90_ms": worst_step,
        "step_profile": {"worst_ms": worst_step},
        "tune": {
            "jitter": {"p95_radius_px": jitter},
            "accuracy": {"median_euclid_px": median, "p95_euclid_px": p95},
        },
        "test": {"jitter": {"p95_radius_px": -9999}},
    }


if __name__ == "__main__":
    unittest.main()


class _FakeModel:
    """Predicts the feature value straight through, so a test can read it."""

    def __init__(self, head_names: tuple = ()) -> None:
        self.schema = SimpleNamespace(head_names=head_names)

    def predict_norm(self, design, rig):  # noqa: ANN001, ARG002
        col = np.asarray(design, dtype=np.float64)[:, 0]
        return np.stack([col, col], axis=1)


def _recording(n: int = 12, *, collecting: np.ndarray | None = None) -> SimpleNamespace:
    """A recording where only the middle of each window is a scoring row."""

    if collecting is None:
        collecting = np.zeros(n, dtype=bool)
        collecting[2:5] = True
        collecting[8:11] = True
    features = np.linspace(0.1, 0.9, n).reshape(-1, 1)
    return SimpleNamespace(
        n_rows=n,
        features=features,
        head=np.zeros((n, 6)),
        openness=np.full((n, 2), 100.0),
        gaze_status=np.ones(n, dtype=bool),
        timestamp_ns=(np.arange(n) * 33_000_000).astype(np.int64),
        target_id=np.repeat(np.arange(2), n // 2),
        rows_collecting=lambda: collecting,
        rows_eligible=lambda: collecting,
        rows_head_valid=lambda: np.ones(n, dtype=bool),
    )


class LiveParityTests(unittest.TestCase):
    """The filtered number in a report must come from the filter that ran.

    Measured on round34/T1 before this existed: the report predicted 462 of
    905 frames and restarted the filter 10 times, where the live view never
    restarted it once.  Feeding a filter the SCORING window makes it a
    different filter, so the reported number described one nobody ran.
    """

    def test_the_live_frames_are_predicted_not_only_the_scored_ones(self) -> None:
        rec = _recording()
        out = B.predict_live(_FakeModel(), rec, rig=None)
        predicted = np.all(np.isfinite(out), axis=1)
        self.assertEqual(
            int(predicted.sum()),
            rec.n_rows,
            "predicting only the scoring window is what restarts the filter between targets",
        )
        self.assertGreater(
            int((predicted & ~rec.rows_collecting()).sum()),
            0,
            "no frame outside the scoring window was predicted, so the filter will "
            "still be fed one collect window at a time",
        )

    def test_a_blink_is_still_not_a_gaze_sample(self) -> None:
        """Live parity means the LIVE gate, not no gate at all."""

        rec = _recording()
        rec.openness = np.full((rec.n_rows, 2), 100.0)
        rec.openness[6] = (1.0, 100.0)  # one eye shut, below BLINK_THRESHOLD
        out = B.predict_live(_FakeModel(), rec, rig=None)
        self.assertTrue(np.all(np.isnan(out[6])), "predicted through a blink")

    def test_the_scoring_window_no_longer_restarts_the_filter(self) -> None:
        """The second window's first sample must remember the first window."""

        rec = _recording()
        settings = F.FilterSettings(width_px=1000, height_px=1000)
        _, live = B.filtered_like_the_screen(_FakeModel(), rec, None, settings)
        old_style, _ = B.replay(
            np.where(
                rec.rows_collecting()[:, None],
                B.predict_live(_FakeModel(), rec, None),
                np.nan,
            ),
            rec.timestamp_ns,
            settings,
        )
        start_of_second = int(np.where(rec.rows_collecting())[0][3])
        self.assertFalse(
            np.allclose(live[start_of_second], old_style[start_of_second]),
            "the two agree, so the filter was still being restarted by the scoring window",
        )

    def test_arrival_and_hold_together_are_exactly_the_scored_rows(self) -> None:
        rec = _recording()
        arrival, hold = B.split_arrival_from_hold(rec)
        both = np.sort(np.concatenate([arrival, hold]))
        self.assertEqual(both.tolist(), np.where(rec.rows_eligible())[0].tolist())
        self.assertEqual(len(B.fixation_windows(rec)), 2)

    def test_the_measurement_rules_are_named_in_the_output(self) -> None:
        """A corrected number compared against an old one is a wrong comparison."""

        self.assertEqual(B.MEASUREMENT_VERSION, "live-parity-1")
