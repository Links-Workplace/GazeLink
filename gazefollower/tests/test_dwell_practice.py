"""Dwell practice bookkeeping: no camera, no model, no OS input.

The scoring is where a bad result can quietly become a good-looking one, so
these check that the three failure kinds stay apart and that the verdict names
the kind rather than only the fraction.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_dwell as D  # noqa: E402
import gf_dwell_practice as P  # noqa: E402


def _trial(index: int, requested: str, activated: str | None, **kw) -> P.Trial:
    return P.Trial(index=index, requested=requested, activated=activated, **kw)


class TrialOrderTests(unittest.TestCase):
    def test_every_target_appears_about_equally_often(self) -> None:
        """Uniform random would leave one target under-tested, and the score
        would then describe whichever targets happened to come up."""

        order = P.trial_order(["A", "B", "C"], 21, seed=3)
        counts = {k: order.count(k) for k in ("A", "B", "C")}
        self.assertEqual(set(counts.values()), {7})

    def test_the_count_is_still_balanced_when_it_does_not_divide(self) -> None:
        order = P.trial_order(["A", "B", "C"], 20, seed=3)
        counts = sorted(order.count(k) for k in ("A", "B", "C"))
        self.assertLessEqual(counts[-1] - counts[0], 1)

    def test_the_same_seed_reproduces_the_same_order(self) -> None:
        """The order is reported so a run can be repeated exactly."""

        self.assertEqual(
            P.trial_order(["A", "B"], 20, seed=7), P.trial_order(["A", "B"], 20, seed=7)
        )

    def test_a_different_seed_gives_a_different_order(self) -> None:
        a = P.trial_order(["A", "B", "C"], 30, seed=1)
        b = P.trial_order(["A", "B", "C"], 30, seed=2)
        self.assertNotEqual(a, b)

    def test_the_requested_number_of_trials_is_produced(self) -> None:
        self.assertEqual(len(P.trial_order(["A", "B"], 20, seed=1)), 20)


class ScoreTests(unittest.TestCase):
    def test_a_wrong_selection_is_not_counted_as_a_miss(self) -> None:
        """One acts against the person's intent, the other only fails to act.
        Pooling them would hide the distinction that decides whether this is
        safe to build on."""

        trials = [
            _trial(0, "LEFT", "LEFT", activated_at_ms=900.0),
            _trial(1, "LEFT", "RIGHT", activated_at_ms=1200.0),
            _trial(2, "RIGHT", None),
        ]
        s = P.score(trials, unintended=0, idle_seconds=60.0)
        self.assertEqual(s["correct_first"], 1)
        self.assertEqual(s["wrong"], 1)
        self.assertEqual(s["no_selection"], 1)
        self.assertEqual(s["wrong_keys"], ["LEFT->RIGHT"])

    def test_time_to_select_covers_correct_trials_only(self) -> None:
        """A fast wrong answer is not a fast selection."""

        trials = [
            _trial(0, "LEFT", "LEFT", activated_at_ms=1000.0),
            _trial(1, "LEFT", "RIGHT", activated_at_ms=50.0),
        ]
        s = P.score(trials, unintended=0, idle_seconds=0.0)
        self.assertEqual(s["time_to_select_ms"]["median"], 1000.0)

    def test_repeat_activations_are_counted_not_discarded(self) -> None:
        trials = [_trial(0, "LEFT", "LEFT", activated_at_ms=900.0, extra_activations=["LEFT"])]
        self.assertEqual(P.score(trials, unintended=0, idle_seconds=0.0)["repeat_activations"], 1)

    def test_tracking_loss_stays_in_the_report(self) -> None:
        trials = [_trial(0, "LEFT", None, frames=100, frames_without_point=40)]
        s = P.score(trials, unintended=0, idle_seconds=0.0)
        self.assertAlmostEqual(s["tracking_lost_fraction"]["max"], 0.4)

    def test_a_trial_with_no_frames_does_not_divide_by_zero(self) -> None:
        self.assertEqual(_trial(0, "LEFT", None).lost_fraction, 0.0)

    def test_an_empty_run_reports_nothing_rather_than_zeroes(self) -> None:
        s = P.score([], unintended=0, idle_seconds=0.0)
        self.assertEqual(s["trials"], 0)
        self.assertIsNone(s["time_to_select_ms"]["median"])


class SkippedTrialTests(unittest.TestCase):
    """A trial that never reached a neutral start must be excluded from the
    score, not merely labelled.

    Running it and counting it is what let the previous target decide the
    answer in the first live run; a trial that starts contaminated cannot be
    repaired by noting that it was.
    """

    def test_a_skipped_trial_is_out_of_the_denominator(self) -> None:
        trials = [
            _trial(0, "LEFT", "LEFT", activated_at_ms=900.0),
            _trial(1, "RIGHT", None, scored=False, neutral_reached=False),
        ]
        s = P.score(trials, unintended=0, idle_seconds=0.0)
        self.assertEqual(s["trials"], 2)
        self.assertEqual(s["scored_trials"], 1)
        self.assertEqual(s["skipped_no_neutral"], 1)
        self.assertEqual(s["correct_first"], 1)
        self.assertEqual(s["no_selection"], 0, "a skipped trial was counted as a miss")

    def test_a_skipped_trial_cannot_be_counted_as_wrong(self) -> None:
        """Whatever it activated happened before the trial legitimately began."""

        trials = [
            _trial(0, "RIGHT", "LEFT", activated_at_ms=901.0, scored=False, neutral_reached=False)
        ]
        s = P.score(trials, unintended=0, idle_seconds=0.0)
        self.assertEqual(s["wrong"], 0)
        self.assertEqual(s["scored_trials"], 0)

    def test_too_few_scored_trials_is_not_reported_as_a_dwell_failure(self) -> None:
        """If the session could not be started cleanly often enough, the
        verdict must say that rather than blame the dwell."""

        summary = {
            "trials": 20,
            "scored_trials": 6,
            "skipped_no_neutral": 14,
            "correct_first": 6,
            "wrong": 0,
            "no_selection": 0,
            "tracking_lost_fraction": {"median": 0.0, "max": 0.0},
            "unintended_activations": 0,
        }
        ok, why = P.verdict(summary, bar=18)
        self.assertFalse(ok)
        self.assertIn("Not a dwell result", why)

    def test_skipped_trials_are_named_in_a_normal_failure_too(self) -> None:
        summary = {
            "trials": 22,
            "scored_trials": 20,
            "skipped_no_neutral": 2,
            "correct_first": 10,
            "wrong": 10,
            "no_selection": 0,
            "tracking_lost_fraction": {"median": 0.0, "max": 0.0},
            "unintended_activations": 0,
        }
        ok, why = P.verdict(summary, bar=18)
        self.assertFalse(ok)
        self.assertIn("neutral start", why)


class VerdictTests(unittest.TestCase):
    def _summary(self, **kw) -> dict:
        base = {
            "trials": 20,
            "correct_first": 0,
            "wrong": 0,
            "no_selection": 0,
            "tracking_lost_fraction": {"median": 0.01, "max": 0.05},
            "unintended_activations": 0,
        }
        base.update(kw)
        return base

    def test_meeting_the_bar_passes(self) -> None:
        ok, why = P.verdict(self._summary(correct_first=18), bar=18)
        self.assertTrue(ok)
        self.assertIn("18/20", why)

    def test_a_diagnostic_layout_has_no_pass_or_fail(self) -> None:
        ok, why = P.verdict(self._summary(correct_first=4), bar=None)
        self.assertIsNone(ok)
        self.assertIn("limit", why)

    def test_a_wrong_target_failure_is_named_as_one(self) -> None:
        """Wrong-target and never-settles lead to different next steps, so the
        verdict has to say which happened."""

        ok, why = P.verdict(self._summary(correct_first=5, wrong=14, no_selection=1), bar=18)
        self.assertFalse(ok)
        self.assertIn("WRONG TARGET", why)

    def test_a_no_selection_failure_is_named_as_one(self) -> None:
        ok, why = P.verdict(self._summary(correct_first=5, wrong=1, no_selection=14), bar=18)
        self.assertFalse(ok)
        self.assertIn("NO SELECTION", why)

    def test_heavy_tracking_loss_is_called_a_capture_problem(self) -> None:
        """Otherwise a camera problem gets read as a dwell problem and the
        next experiment is aimed at the wrong thing."""

        summary = self._summary(
            correct_first=2,
            no_selection=18,
            tracking_lost_fraction={"median": 0.6, "max": 0.9},
        )
        ok, why = P.verdict(summary, bar=18)
        self.assertFalse(ok)
        self.assertIn("capture or seating", why)

    def test_unintended_activations_are_surfaced_even_alongside_other_failures(self) -> None:
        summary = self._summary(correct_first=10, no_selection=10, unintended_activations=3)
        ok, why = P.verdict(summary, bar=18)
        self.assertFalse(ok)
        self.assertIn("unintended", why)

    def test_passing_the_bar_does_not_hide_unintended_activations(self) -> None:
        """A pass with false activations is still worth seeing in the report,
        even though the bar is about first-try correctness."""

        summary = self._summary(correct_first=19, unintended_activations=2)
        ok, _ = P.verdict(summary, bar=18)
        self.assertTrue(ok)
        self.assertEqual(P.score([], unintended=2, idle_seconds=60.0)["unintended_activations"], 2)


class RunnerContractTests(unittest.TestCase):
    """The runner talks to Display and LiveRunner across a module boundary.

    Nothing else here constructs them -- the pure tests deliberately do not --
    so a signature change on the other side lands at runtime, in front of the
    operator, halfway into a session. It did: the first live attempt died on a
    missing keyword-only `headless`. These bind the call shapes without a
    camera, which is the part that can be checked cheaply.
    """

    def test_the_display_call_shape_is_valid(self) -> None:
        import inspect

        import gf_record as R

        inspect.signature(R.Display).bind(5120, 1440, headless=False, origin=(0, 0))

    def test_run_practice_exposes_headless_so_it_can_be_smoke_tested(self) -> None:
        """Without it there is no way to exercise this path except by asking a
        person to sit through a session."""

        import inspect

        self.assertIn("headless", inspect.signature(P.run_practice).parameters)

    def test_the_practice_draw_call_shape_is_valid(self) -> None:
        import inspect

        import gf_record as R

        inspect.signature(R.Display.draw_practice).bind(
            None,
            [],
            (0.5, 0.5),
            hovered=None,
            progress=0.0,
            prompt=[],
            flash=None,
            tracking=True,
        )


class SimulatedSessionTests(unittest.TestCase):
    """The whole trial loop, driven offline by a scripted operator.

    This is the dry run the live session would otherwise be: it proves the
    bookkeeping before any of the operator's time is spent on it, and it turns
    the layout decision into an executable claim.
    """

    HZ = 60.0

    def _run(self, buttons, order, aim, *, dwell_ms=900.0, timeout_s=8.0):
        engine = D.DwellEngine(buttons, D.DwellConfig(dwell_ms=dwell_ms))
        trials = []
        t = 0.0
        for index, requested in enumerate(order):
            engine.reset()
            trial = P.Trial(index=index, requested=requested)
            started = t
            point = aim(requested)
            while t - started < timeout_s:
                t += 1.0 / self.HZ
                trial.frames += 1
                if point is None:
                    trial.frames_without_point += 1
                fired = engine.update(t, point, fresh=point is not None)
                if fired is not None:
                    if trial.activated is None:
                        trial.activated = fired.button
                        trial.activated_at_ms = (t - started) * 1000.0
                        break
                    trial.extra_activations.append(fired.button)
            trials.append(trial)
        return trials

    def _centre_of(self, buttons, key):
        return next(b for b in buttons if b.key == key).centre

    def test_a_perfect_operator_scores_every_trial(self) -> None:
        buttons = D.layout_a()
        order = P.trial_order([b.key for b in buttons], 20, seed=1)
        trials = self._run(buttons, order, lambda k: self._centre_of(buttons, k))
        s = P.score(trials, unintended=0, idle_seconds=0.0)
        self.assertEqual(s["correct_first"], 20)
        self.assertEqual(s["wrong"], 0)
        self.assertEqual(s["no_selection"], 0)
        self.assertTrue(P.verdict(s, bar=18)[0])

    def test_the_measured_sideways_bias_still_passes_the_safe_layout(self) -> None:
        """0.056 is the worst horizontal bias measured on this rig."""

        buttons = D.layout_a()
        order = P.trial_order([b.key for b in buttons], 20, seed=1)

        def aim(key):
            x, y = self._centre_of(buttons, key)
            return (x + 0.056, y)

        s = P.score(self._run(buttons, order, aim), unintended=0, idle_seconds=0.0)
        self.assertEqual(s["correct_first"], 20)

    def test_the_same_bias_fails_the_tight_layout_by_missing_not_by_choosing_wrong(self) -> None:
        """The layout decision as an executable claim: the tight layout is
        expected to stop selecting, NOT to select the neighbour. Missing is
        the tolerable failure; a confident wrong choice is not."""

        buttons = D.layout_b()
        order = P.trial_order([b.key for b in buttons], 21, seed=1)

        def aim(key):
            x, y = self._centre_of(buttons, key)
            return (x + 0.056, y)

        s = P.score(self._run(buttons, order, aim), unintended=0, idle_seconds=0.0)
        self.assertEqual(s["wrong"], 0, "the tight layout chose a neighbour")
        self.assertGreater(s["no_selection"], 0)

    def _run_with_gate(
        self,
        buttons,
        order,
        aim,
        *,
        dwell_ms=900.0,
        timeout_s=8.0,
        neutral=(0.50, 0.5),
        gate_s=5.0,
        reset_each_trial=False,
        gate_point=None,
    ):
        """The corrected protocol: one engine across trials, plus a gate that
        waits for the gaze to leave every target before scoring starts."""

        engine = D.DwellEngine(buttons, D.DwellConfig(dwell_ms=dwell_ms))
        trials, t = [], 0.0
        for index, requested in enumerate(order):
            if reset_each_trial:
                engine.reset()
            trial = P.Trial(index=index, requested=requested)
            gate_started = t
            during_gate = gate_point if gate_point is not None else neutral
            while t - gate_started < gate_s:
                t += 1.0 / self.HZ
                fired = engine.update(t, during_gate, fresh=True)
                if fired is not None:
                    trial.activations_before_start.append(fired.button)
                if engine.find(during_gate) is None:
                    trial.neutral_reached = True
                    break
                trial.neutral_reached = False
            started, point = t, aim(requested)
            while t - started < timeout_s:
                t += 1.0 / self.HZ
                trial.frames += 1
                fired = engine.update(t, point, fresh=True)
                if fired is not None:
                    if trial.activated is None:
                        trial.activated = fired.button
                        trial.activated_at_ms = (t - started) * 1000.0
                        break
                    trial.extra_activations.append(fired.button)
            trials.append(trial)
        return trials

    def test_the_previous_target_can_no_longer_fire_the_next_trial(self) -> None:
        """The measured failure: 8 of 11 wrong activations landed on the
        PREVIOUS trial's target, because resetting the engine each trial
        cleared the latch and a target still under the gaze refired at once.

        Here the operator never moves -- they stare at LEFT the whole session.
        With the reset that scores a LEFT on every trial; without it, LEFT
        fires once and then stays latched.
        """

        buttons = D.layout_a()
        order = ["LEFT", "RIGHT", "LEFT", "RIGHT"]
        stuck = self._centre_of(buttons, "LEFT")

        broken = self._run_with_gate(
            buttons,
            order,
            lambda k: stuck,
            gate_point=stuck,
            gate_s=0.2,
            reset_each_trial=True,
        )
        self.assertGreater(
            sum(1 for t in broken if t.activated == "LEFT"), 1, "the old bug no longer reproduces"
        )

        fixed = self._run_with_gate(buttons, order, lambda k: stuck, gate_point=stuck, gate_s=0.2)
        self.assertEqual(
            [t.activated for t in fixed],
            ["LEFT", None, None, None],
            "a target the gaze never left fired more than once",
        )

    def test_the_gate_records_when_neutral_was_never_reached(self) -> None:
        """It must not hang waiting for something that may be unreachable."""

        buttons = D.layout_a()
        on_a_button = self._centre_of(buttons, "LEFT")
        trials = self._run_with_gate(
            buttons,
            ["RIGHT"],
            lambda k: self._centre_of(buttons, k),
            neutral=on_a_button,
            gate_s=1.0,
        )
        self.assertFalse(trials[0].neutral_reached)
        self.assertEqual(trials[0].requested, "RIGHT")

    def test_leaving_and_returning_still_allows_a_later_trial_to_score(self) -> None:
        """The latch must not become a permanent block: passing through
        neutral is what re-arms it, and the gate guarantees that happens."""

        buttons = D.layout_a()
        order = P.trial_order([b.key for b in buttons], 20, seed=1)
        trials = self._run_with_gate(buttons, order, lambda k: self._centre_of(buttons, k))
        s = P.score(trials, unintended=0, idle_seconds=0.0, buttons=buttons)
        self.assertEqual(s["correct_first"], 20)
        self.assertEqual(s["activations_before_start"], 0)

    def test_the_offset_summary_reports_direction_not_just_size(self) -> None:
        """Signed and per axis, so the direction survives.

        This is context for the selections, NOT a bias measurement: looking
        anywhere inside a target is allowed, so a non-zero value here proves
        nothing about the model. Measuring bias needs a centre marker.
        """

        buttons = D.layout_a()
        centres = {b.key: b.centre for b in buttons}
        trials = [
            P.Trial(index=0, requested="LEFT", median_point=(centres["LEFT"][0] - 0.04, 0.5)),
            P.Trial(index=1, requested="RIGHT", median_point=(centres["RIGHT"][0] - 0.04, 0.5)),
        ]
        off = P.score(trials, unintended=0, idle_seconds=0.0, buttons=buttons)[
            "gaze_offset_from_centre"
        ]
        self.assertAlmostEqual(off["dx_median"], -0.04, places=6)
        self.assertEqual(off["n"], 2)

    def test_an_operator_who_never_tracks_records_the_loss_and_selects_nothing(self) -> None:
        buttons = D.layout_a()
        order = P.trial_order([b.key for b in buttons], 4, seed=1)
        s = P.score(self._run(buttons, order, lambda k: None), unintended=0, idle_seconds=0.0)
        self.assertEqual(s["no_selection"], 4)
        self.assertEqual(s["tracking_lost_fraction"]["median"], 1.0)
        self.assertIn("capture or seating", P.verdict(s, bar=3)[1])


if __name__ == "__main__":
    unittest.main()


class RestBlockDetailTests(unittest.TestCase):
    """A count cannot direct a fix; where and when the target fired can.

    Measured on 9.9: 4 unintended activations in 60 s of rest, on the layout
    that scores 18-20/20 on selection. The count alone does not say whether
    they were spread through the block or bunched at one moment, nor which
    target fired -- and those point at different changes.
    """

    def test_the_detail_survives_into_the_summary(self) -> None:
        records = [
            {"key": "LEFT", "at_s": 3.2, "point": [0.44, 0.5]},
            {"key": "RIGHT", "at_s": 41.9, "point": [0.55, 0.2]},
        ]
        summary = P.score([], unintended=records, idle_seconds=60.0)
        self.assertEqual(summary["unintended_activations"], 2)
        self.assertEqual(summary["unintended_detail"], records)

    def test_a_plain_count_is_still_accepted_and_reports_no_detail(self) -> None:
        """Older reports were written with a count; reading them must not break."""

        summary = P.score([], unintended=4, idle_seconds=60.0)
        self.assertEqual(summary["unintended_activations"], 4)
        self.assertEqual(summary["unintended_detail"], [])

    def test_an_empty_rest_block_is_not_a_missing_one(self) -> None:
        summary = P.score([], unintended=[], idle_seconds=60.0)
        self.assertEqual(summary["unintended_activations"], 0)
        self.assertEqual(summary["idle_seconds"], 60.0)

    def test_the_detail_can_actually_be_written_as_json(self) -> None:
        """The engine returns an Activation, not a key.

        The first version of the rest-block detail stored the Activation
        object itself. ``json.dumps`` cannot take it, so the report would have
        crashed at the end of any run that recorded one -- and it did not,
        only because the run that used it recorded zero. Serialising here is
        the assertion: a summary that cannot be written is not a summary.
        """

        import json  # noqa: PLC0415

        engine = D.DwellEngine(
            [D.Button("LEFT", 0.3, 0.05, 0.46, 0.95)], D.DwellConfig(dwell_ms=1.0)
        )
        fired = None
        now = 0.0
        while fired is None and now < 1.0:
            fired = engine.update(now, (0.38, 0.5), fresh=True)
            now += 0.05
        self.assertIsNotNone(fired, "the engine never fired; this test would prove nothing")
        record = {"key": fired.button, "at_s": 1.0, "point": [0.38, 0.5]}
        summary = P.score([], unintended=[record], idle_seconds=60.0)
        json.dumps(summary)
        self.assertEqual(summary["unintended_detail"][0]["key"], "LEFT")
