"""Selection replay: the real engine on logged or synthetic paths. No camera, no OS input.

Synthetic paths test the replay's logic only; they say nothing about people.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import gf_dwell as D  # noqa: E402
import gf_selection_replay as RP  # noqa: E402

BUTTONS = D.layout_zones()
BY_KEY = {b.key: b for b in BUTTONS}
DWELL = 900.0
DEAD = (0.5, 1 / 3)  # between rows, on no button


def on(key: str, dx: float = 0.0) -> tuple[float, float]:
    cx, cy = BY_KEY[key].centre
    return (cx + dx, cy)


def calls(rows: list[tuple[float, tuple[float, float] | None, bool, int | None]]) -> list[list]:
    return [
        [t, None if p is None else p[0], None if p is None else p[1], fresh, uid]
        for t, p, fresh, uid in rows
    ]


def trial(requested: str, engine_rows: list, activated: str | None, at: float | None, **kw) -> dict:
    t = {
        "index": 0,
        "requested": requested,
        "activated": activated,
        "activated_at_ms": at,
        "frames": len(engine_rows),
        "frames_without_point": sum(1 for r in engine_rows if r[1] is None),
        "neutral_reached": True,
        "scored": True,
        "engine_calls": engine_rows,
    }
    t.update(kw)
    return t


def steady(key: str, start: float, end: float, step: float = 16.0, uid0: int = 0) -> list:
    rows, t, uid = [], start, uid0
    while t <= end:
        rows.append((t, on(key), True, uid))
        t += step
        uid += 1
    return rows


class EngineReplay(unittest.TestCase):
    def test_a_full_stay_activates_at_the_dwell_time(self) -> None:
        rows = calls(steady("MID-C", 0.0, 1200.0))
        key, at = RP.replay_calls(BUTTONS, DWELL, RP.calls_for({"engine_calls": rows})[0])
        self.assertEqual(key, "MID-C")
        self.assertGreaterEqual(at, 900.0)
        self.assertLess(at, 900.0 + 16.0 + 1e-9)

    def test_passing_through_a_button_does_not_activate(self) -> None:
        rows = calls(
            steady("UP-C", 0.0, 300.0) + [(316.0, DEAD, True, 99)] + steady("MID-C", 332.0, 600.0)
        )
        key, _ = RP.replay_calls(BUTTONS, DWELL, RP.calls_for({"engine_calls": rows})[0])
        self.assertIsNone(key)

    def test_a_blink_resets_progress_and_never_activates(self) -> None:
        rows = (
            steady("MID-C", 0.0, 600.0)
            + [(616.0, None, True, 40)]
            + steady("MID-C", 632.0, 1700.0, uid0=41)
        )
        key, at = RP.replay_calls(BUTTONS, DWELL, RP.calls_for({"engine_calls": calls(rows)})[0])
        self.assertEqual(key, "MID-C")
        self.assertGreaterEqual(at, 632.0 + 900.0)

    def test_stale_points_never_activate_however_long(self) -> None:
        rows = [(t * 16.0, on("MID-C"), False, t) for t in range(200)]
        key, _ = RP.replay_calls(BUTTONS, DWELL, RP.calls_for({"engine_calls": calls(rows)})[0])
        self.assertIsNone(key)

    def test_missing_points_never_activate(self) -> None:
        rows = [(t * 16.0, None, True, None) for t in range(200)]
        key, _ = RP.replay_calls(BUTTONS, DWELL, RP.calls_for({"engine_calls": calls(rows)})[0])
        self.assertIsNone(key)

    def test_jitter_across_the_border_resets_each_time(self) -> None:
        b = BY_KEY["MID-L"]
        inside = (b.x1 - 0.001, b.centre[1])
        outside = (b.x1 + 0.001, b.centre[1])
        rows = [(i * 16.0, inside if i % 20 else outside, True, i) for i in range(200)]
        key, _ = RP.replay_calls(BUTTONS, DWELL, RP.calls_for({"engine_calls": calls(rows)})[0])
        self.assertIsNone(key)  # 19 frames in (304 ms) never reach 900 ms

    def test_repeated_update_ids_are_replayed_but_counted_once(self) -> None:
        # The display redraws faster than the camera: the same prediction is
        # fed to the engine several times. Time, not samples, decides.
        rows = [(i * 8.0, on("MID-C"), True, i // 4) for i in range(0, 125)]
        t = trial("MID-C", calls(rows), "MID-C", 904.0)
        r = RP.describe(t, BUTTONS, DWELL)
        self.assertEqual(r.calls, 125)
        self.assertEqual(r.unique_updates, 32)
        self.assertEqual(r.replay_activation, "MID-C")
        self.assertTrue(r.agrees)


class Classification(unittest.TestCase):
    def test_requested_target_is_not_given_to_the_engine(self) -> None:
        # Same calls, different requested target: identical replayed activation.
        rows = calls(steady("UP-C", 0.0, 1000.0))
        a = RP.describe(trial("UP-C", rows, "UP-C", 912.0), BUTTONS, DWELL)
        b = RP.describe(
            trial("LOW-L", rows, "UP-C", 912.0), BUTTONS, DWELL, RP.cue_neighbour(BUTTONS)
        )
        self.assertEqual(a.replay_activation, b.replay_activation)
        self.assertEqual(a.classification, "success")
        self.assertEqual(b.classification, "wrong_activation")
        self.assertTrue(b.wrong_on_cue_neighbour)

    def test_reentries_without_completion_are_named_only_with_full_calls(self) -> None:
        rows = []
        t = 0.0
        for _ in range(4):
            rows += steady("MID-R", t, t + 400.0)
            t += 416.0
            rows.append((t, DEAD, True, None))
            t += 16.0
        r = RP.describe(trial("MID-R", calls(rows), None, None), BUTTONS, DWELL)
        self.assertEqual(r.classification, "reentry_no_complete")
        self.assertEqual(r.entries, 4)

    def test_path_log_never_yields_a_reentry_mechanism(self) -> None:
        path = (
            [[t, *on("MID-R")] for t in range(0, 400, 62)]
            + [[430.0, *DEAD]]
            + [[t, *on("MID-R")] for t in range(492, 900, 62)]
        )
        t = {
            "index": 0,
            "requested": "MID-R",
            "activated": None,
            "activated_at_ms": None,
            "frames": 60,
            "frames_without_point": 0,
            "neutral_reached": True,
            "path": path,
        }
        r = RP.describe(t, BUTTONS, DWELL)
        self.assertTrue(r.partial)
        self.assertEqual(r.classification, "entered_short_partial_log")

    def test_unlogged_missing_frames_make_the_mechanism_undetermined(self) -> None:
        path = [[t, *on("MID-R")] for t in range(0, 500, 62)]
        t = {
            "index": 0,
            "requested": "MID-R",
            "activated": None,
            "activated_at_ms": None,
            "frames": 60,
            "frames_without_point": 3,
            "neutral_reached": True,
            "path": path,
        }
        self.assertEqual(RP.describe(t, BUTTONS, DWELL).classification, "undetermined_from_log")

    def test_a_stay_within_log_resolution_of_the_dwell_is_undetermined(self) -> None:
        path = [[t, *on("MID-R")] for t in range(0, 870, 62)]
        t = {
            "index": 0,
            "requested": "MID-R",
            "activated": None,
            "activated_at_ms": None,
            "frames": 60,
            "frames_without_point": 0,
            "neutral_reached": True,
            "path": path,
        }
        self.assertEqual(RP.describe(t, BUTTONS, DWELL).classification, "undetermined_from_log")

    def test_no_neutral_start_is_undetermined(self) -> None:
        rows = calls(steady("MID-C", 0.0, 500.0))
        r = RP.describe(trial("MID-C", rows, None, None, neutral_reached=False), BUTTONS, DWELL)
        self.assertEqual(r.classification, "undetermined_from_log")

    def test_unscored_and_aborted_trials_are_not_classified(self) -> None:
        rows = calls(steady("MID-C", 0.0, 1000.0))
        self.assertEqual(
            RP.describe(
                trial("MID-C", rows, None, None, scored=False), BUTTONS, DWELL
            ).classification,
            "not_scored",
        )
        self.assertEqual(
            RP.describe(
                trial("MID-C", rows, None, None, aborted=True), BUTTONS, DWELL
            ).classification,
            "not_scored",
        )

    def test_wrong_activation_on_the_previous_target_is_flagged(self) -> None:
        rows = calls(steady("LOW-R", 0.0, 1000.0))
        r = RP.describe(trial("UP-C", rows, "LOW-R", 912.0), BUTTONS, DWELL, "UP-C", "LOW-R")
        self.assertTrue(r.wrong_on_previous_target)
        self.assertFalse(r.wrong_on_cue_neighbour)

    def test_wrong_on_both_cue_and_previous_target_is_counted_once_each_way(self) -> None:
        report = {
            "profile": "p",
            "dwell_ms": DWELL,
            "buttons": [
                {"key": b.key, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1} for b in BUTTONS
            ],
            "summary": {},
            "trials": [
                trial("UP-C", calls(steady("UP-C", 0.0, 1000.0)), "UP-C", 912.0),
                dict(trial("UP-R", calls(steady("UP-C", 0.0, 1000.0)), "UP-C", 912.0), index=1),
            ],
        }
        res = RP.replay_report(report, "t")
        self.assertEqual(res["wrong_on_cue_neighbour"], 1)
        self.assertEqual(res["wrong_on_previous_target"], 1)
        self.assertEqual(res["wrong_on_both"], 1)
        self.assertEqual(RP.summarise([res])["p"]["wrong_on_both"], 1)

    def test_cue_neighbour_in_the_zone_grid_is_up_c(self) -> None:
        self.assertEqual(RP.cue_neighbour(BUTTONS), "UP-C")


class ExistingLogRegression(unittest.TestCase):
    LOG = HERE / "results" / "select_compare" / "20260915_073919.json"

    def test_replay_reproduces_the_logged_outcomes_it_can(self) -> None:
        if not self.LOG.exists():
            self.skipTest("comparison log not present")
        doc = json.loads(self.LOG.read_text(encoding="utf-8"))
        results = [RP.replay_report(rep, label) for label, rep in RP.reports_in(doc)]
        of = sum(r["agreement"]["of"] for r in results)
        self.assertEqual(of, 36)
        non = sum(r["agreement"]["non_activations_reproduced"] for r in results)
        non_of = sum(r["agreement"]["non_activations_logged"] for r in results)
        self.assertEqual(non, non_of)  # replay never fires where the live run did not
        for r, (_label, rep) in zip(results, RP.reports_in(doc), strict=True):
            for t, raw in zip(r["trials"], rep["trials"], strict=True):
                if t["replay_lag_ms"] is not None and t["logged_activation_ms"] is not None:
                    # Input is cut at the first logged row later than the
                    # logged activation, so the replay fires ON an actual row
                    # no earlier than the live time and no later than that row.
                    logged = t["logged_activation_ms"]
                    fired = logged + t["replay_lag_ms"]
                    rows = [float(row[0]) for row in raw["path"]]
                    later = [x for x in rows if x > logged + RP.AGREE_EPS_MS]
                    bound = min(later) if later else logged + RP.AGREE_EPS_MS
                    self.assertTrue(any(abs(fired - x) < 1e-6 for x in rows))
                    self.assertLessEqual(fired, bound + 1e-6)
                    if t["agrees"]:
                        self.assertGreaterEqual(t["replay_lag_ms"], -RP.AGREE_EPS_MS)
                if not t["agrees"]:
                    self.assertIsNotNone(t["logged_activation"])
                    self.assertIsNone(t["replay_activation"])

    def test_a_stay_completing_on_the_next_logged_row_agrees(self) -> None:
        # Rows 62 ms apart: the live engine fired at 900 ms, between rows; the
        # replay reaches 900 ms only on the row at 930 ms.
        path = [[float(t), *on("MID-C")] for t in range(0, 1500, 62)]
        t = {
            "index": 0,
            "requested": "MID-C",
            "activated": "MID-C",
            "activated_at_ms": 900.0,
            "frames": 90,
            "frames_without_point": 0,
            "neutral_reached": True,
            "path": path,
        }
        r = RP.describe(t, BUTTONS, DWELL)
        self.assertEqual(r.replay_activation, "MID-C")
        self.assertEqual(r.replay_lag_ms, 30.0)
        self.assertTrue(r.agrees)

    def test_a_replay_firing_earlier_than_the_live_run_does_not_agree(self) -> None:
        rows = calls(steady("MID-C", 0.0, 2000.0))
        r = RP.describe(trial("MID-C", rows, "MID-C", 1500.0), BUTTONS, DWELL)
        self.assertEqual(r.replay_activation, "MID-C")
        self.assertLess(r.replay_lag_ms, 0.0)
        self.assertFalse(r.agrees)

    def test_nothing_long_after_the_logged_activation_can_reproduce_it(self) -> None:
        # A stay that only completes 500 ms after the logged activation must
        # not be counted as reproducing it.
        path = [[float(t), *on("MID-C")] for t in range(0, 1500, 62)]
        t = {
            "index": 0,
            "requested": "MID-C",
            "activated": "MID-C",
            "activated_at_ms": 400.0,
            "frames": 90,
            "frames_without_point": 0,
            "neutral_reached": True,
            "path": path,
        }
        r = RP.describe(t, BUTTONS, DWELL)
        self.assertIsNone(r.replay_activation)
        self.assertFalse(r.agrees)


if __name__ == "__main__":
    unittest.main()
