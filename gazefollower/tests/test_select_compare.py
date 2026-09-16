"""The live two-profile selection comparison: layout, order, scoring, profile.

No camera and no window: everything here is the pure part -- where the nine
buttons sit, which arm runs when, how trials are counted, and that making a
trial profile cannot change the one in use.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_dwell as D  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_select_compare as SC  # noqa: E402
import gf_targets as T  # noqa: E402


class ZoneLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buttons = D.layout_zones()

    def test_one_button_per_band_zone_in_zone_order(self) -> None:
        self.assertEqual(len(self.buttons), 9)
        for i, b in enumerate(self.buttons):
            self.assertEqual(T.band_zone_of(*b.centre), i, b.key)

    def test_the_bottom_row_is_really_at_the_bottom(self) -> None:
        """The failure under test is around y 0.80; a grid that stopped at
        0.70 would leave it out."""

        low = [b for b in self.buttons if b.key.startswith("LOW")]
        self.assertTrue(all(b.centre[1] > 0.80 for b in low))

    def test_size_is_the_fixed_device_size(self) -> None:
        for b in self.buttons:
            self.assertAlmostEqual(b.width * 5120, 400, places=6)
            self.assertAlmostEqual(b.height * 1440, 150, places=6)

    def test_buttons_stay_in_the_band_and_do_not_overlap(self) -> None:
        for b in self.buttons:
            self.assertGreaterEqual(b.x0, 0.30)
            self.assertLessEqual(b.x1, 0.70)
        for i, a in enumerate(self.buttons):
            for b in self.buttons[i + 1 :]:
                overlap = not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)
                self.assertFalse(overlap, f"{a.key} overlaps {b.key}")

    def test_there_is_neutral_space_to_start_a_trial(self) -> None:
        """The practice refuses to score a trial until the gaze is on no target."""

        engine = D.DwellEngine(self.buttons, D.DwellConfig())
        self.assertIsNone(engine.find((0.5, 1 / 3)))

    def test_neighbour_warning_pairs_buttons_side_by_side_not_by_sort_order(self) -> None:
        # reach = half-width 0.039 + gap 0.055 = 0.094
        warnings = D.layout_warnings(self.buttons, bias_x=0.10, bias_y=0.0)
        risky = [w for w in warnings if w.startswith("NEIGHBOUR")]
        # Three rows, each with two side-by-side pairs; never a pair across rows.
        self.assertEqual(len(risky), 6, risky)
        for w in risky:
            left, right = w.split("from ")[1].split(" ")[0], w.split("activate ")[1].split(" ")[0]
            self.assertEqual(left.split("-")[0], right.split("-")[0], w)

    def test_column_layouts_warn_as_before(self) -> None:
        warnings = D.layout_warnings(D.layout_b(), bias_x=0.2, bias_y=0.0)
        self.assertEqual(sum(w.startswith("NEIGHBOUR") for w in warnings), 2)


class OrderTests(unittest.TestCase):
    def test_abba_so_neither_profile_always_goes_first(self) -> None:
        plan = SC.block_plan("baseline", "trial", seed=5)
        self.assertEqual([p["profile"] for p in plan], ["baseline", "trial", "trial", "baseline"])
        seeds = {arm: sorted(p["seed"] for p in plan if p["arm"] == arm) for arm in "AB"}
        self.assertEqual(seeds["A"], seeds["B"], "the arms drew different trial orders")
        self.assertEqual(len(set(seeds["A"])), 2, "each arm should run two different orders")


def _trial(requested, activated=None, scored=True, ms=1200.0):
    return {
        "requested": requested,
        "activated": activated,
        "scored": scored,
        "activated_at_ms": ms if activated else None,
    }


class ScoringTests(unittest.TestCase):
    def test_timeouts_and_wrong_targets_stay_in_the_denominator(self) -> None:
        trials = [
            _trial("LOW-L", "LOW-L"),
            _trial("LOW-L", None),  # timed out
            _trial("LOW-L", "LOW-C"),  # wrong target
        ]
        row = SC.per_zone(trials)[SC.zone_of_key("LOW-L")]
        self.assertEqual(
            (row["trials"], row["correct"], row["wrong"], row["no_selection"]), (3, 1, 1, 1)
        )

    def test_an_unscored_trial_is_counted_apart_not_as_a_success_or_failure(self) -> None:
        row = SC.per_zone([_trial("UP-R", None, scored=False)])[SC.zone_of_key("UP-R")]
        self.assertEqual((row["trials"], row["skipped_no_neutral"]), (0, 1))

    def test_arms_are_pooled_over_their_own_blocks_only(self) -> None:
        blocks = [
            {"arm": "A", "profile": "base", "report": {"trials": [_trial("MID-C", "MID-C")]}},
            {"arm": "B", "profile": "cand", "report": {"trials": [_trial("MID-C", None)]}},
            {"arm": "B", "profile": "cand", "report": {"trials": [_trial("MID-C", None)]}},
            {"arm": "A", "profile": "base", "report": {"aborted": True}},
        ]
        s = SC.summarise(blocks)
        self.assertEqual((s["A"]["correct"], s["A"]["scored_trials"]), (1, 1))
        self.assertEqual((s["B"]["correct"], s["B"]["scored_trials"]), (0, 2))
        self.assertEqual(s["A"]["profile"], "base")
        self.assertIn("Preliminary", SC.format_summary(s))

    def test_a_trial_stopped_by_the_operator_is_not_a_failed_selection(self) -> None:
        aborted = {**_trial("LOW-R", None, scored=False), "aborted": True}
        row = SC.per_zone([aborted])[SC.zone_of_key("LOW-R")]
        self.assertEqual(
            (row["trials"], row["no_selection"], row["skipped_no_neutral"], row["aborted"]),
            (0, 0, 0, 1),
        )

    def test_a_stopped_session_is_reported_as_not_comparable(self) -> None:
        blocks = [
            {"arm": "A", "profile": "base", "report": {"trials": [_trial("MID-C", "MID-C")]}},
            {
                "arm": "B",
                "profile": "cand",
                "report": {"aborted": True, "trials": [_trial("MID-C", None)]},
            },
        ]
        s = SC.summarise(blocks)
        self.assertFalse(s["complete"])
        self.assertIn("NOT comparable", SC.format_summary(s))


class BlockProcessTests(unittest.TestCase):
    def _args(self):
        return SC.build_parser().parse_args(["run", "--candidate", "pooled_trial"])

    def test_a_block_process_that_leaves_no_report_is_a_stop(self) -> None:
        """A crash must stop the session, never skip silently to the next block."""

        from unittest import mock  # noqa: PLC0415

        step = SC.block_plan("baseline", "pooled_trial", seed=1)[1]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(SC.subprocess, "run") as run:
            report = SC.run_block_in_child(step, self._args(), Path(tmp) / "b.json")
        self.assertTrue(report["aborted"])
        command = run.call_args.args[0]
        self.assertIn("run-block", command)
        self.assertEqual(command[command.index("--profile") + 1], "pooled_trial")
        self.assertEqual(command[command.index("--seed") + 1], str(step["seed"]))
        # The child command must parse, or every block would fail.
        SC.build_parser().parse_args(command[2:])

    def test_resume_keeps_only_finished_matching_blocks(self) -> None:
        plan = SC.block_plan("baseline", "pooled_trial", seed=1)
        settings = {"dwell_ms": 900.0, "trial_timeout_s": 8.0, "idle_seconds": 20.0}
        models = {"baseline": "m/base", "pooled_trial": "m/pool"}

        def report(profile, **over):
            r = {
                "trials": [_trial("UP-L", "UP-L")] * 18,
                "dwell_ms": 900.0,
                "trial_timeout_s": 8.0,
                "summary": {"idle_seconds": 20.0},
                "model": models[profile],
            }
            r.update(over)
            return r

        old = {
            "blocks": [
                {**plan[0], "report": report("baseline")},  # the only one to keep
                {**plan[1], "report": report("pooled_trial", aborted=True)},
                {**plan[2], "seed": 99, "report": report("pooled_trial")},
                {**plan[3], "report": report("baseline", trials=[_trial("UP-L", "UP-L")] * 9)},
            ]
        }
        variants = {
            "dwell": report("baseline", dwell_ms=700.0),
            "timeout": report("baseline", trial_timeout_s=10.0),
            "rest": report("baseline", summary={"idle_seconds": 0.0}),
            "model": report("baseline", model="m/other"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            kept = SC.resumable_blocks(path, plan, 18, settings=settings, models=models)
            self.assertEqual(sorted(kept), [1])
            for what, changed in variants.items():
                path.write_text(json.dumps({"blocks": [{**plan[0], "report": changed}]}), "utf-8")
                self.assertEqual(
                    SC.resumable_blocks(path, plan, 18, settings=settings, models=models),
                    {},
                    f"a block with a different {what} was kept",
                )


class PracticeAbortTests(unittest.TestCase):
    def test_score_keeps_aborted_trials_out_of_every_failure_count(self) -> None:
        import gf_dwell_practice as P  # noqa: PLC0415

        done = P.Trial(index=0, requested="UP-L", activated="UP-L", activated_at_ms=900.0)
        stopped = P.Trial(index=1, requested="UP-C", scored=False, aborted=True)
        gate = P.Trial(index=2, requested="UP-R", scored=False, neutral_reached=False)
        s = P.score([done, stopped, gate], unintended=0, idle_seconds=0)
        self.assertEqual(
            (s["scored_trials"], s["no_selection"], s["skipped_no_neutral"], s["aborted_trials"]),
            (1, 0, 1, 1),
        )


class TrialProfileTests(unittest.TestCase):
    def test_only_the_model_changes_and_the_active_profile_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = PROF.Profile(
                name="baseline",
                model_dir="recordings/round18/models/m",
                rig={"k": 1},
                filter={"kind": "one-euro", "one_euro_min_cutoff_hz": 0.4},
                cursor={"smoothing": 0.6},
                gesture={"wink": {"min_ms": 35}},
                x_range=[0.3, 0.7],
            )
            PROF.save(base, root=root)
            PROF.activate("baseline", root=root)
            SC.make_trial_profile(
                base,
                name="pooled_trial",
                model_dir="results/m2",
                note="n",
                root=root,
                require_model=False,
            )
            self.assertEqual(PROF.active_name(root=root), "baseline")
            trial = PROF.load("pooled_trial", root=root)
            self.assertEqual(trial.model_dir, "results/m2")
            for field in ("filter", "cursor", "gesture", "rig", "x_range", "screen"):
                self.assertEqual(getattr(trial, field), getattr(base, field), field)
            before = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
            self.assertEqual(before["model_dir"], "recordings/round18/models/m")

    def test_an_existing_profile_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = PROF.Profile(name="baseline", model_dir="m", rig={}, filter={})
            PROF.save(base, root=root)
            with self.assertRaises(FileExistsError):
                SC.make_trial_profile(
                    base, name="baseline", model_dir="x", note="", root=root, require_model=False
                )

    def test_a_missing_model_is_refused(self) -> None:
        base = PROF.Profile(name="baseline", model_dir="m", rig={}, filter={})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            SC.make_trial_profile(
                base, name="t", model_dir="does/not/exist", note="", root=Path(tmp)
            )


if __name__ == "__main__":
    unittest.main()
