"""The practice harness logs every dwell-engine input, and replay reproduces the run from it.

Headless: fake runner, fake display, no camera, no OS input. The fakes are the
ones the schedule tests already use, so this checks the same session loop.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unittest import mock  # noqa: E402

import gf_dwell as D  # noqa: E402
import gf_dwell_practice as P  # noqa: E402
import gf_live as L  # noqa: E402
import gf_selection_replay as RP  # noqa: E402
from test_dwell_practice_schedule import FakeDisplay, FakeRunner, run_fake  # noqa: E402


class GappyRunner(FakeRunner):
    """Every fourth read has no point, as a blink or lost face would -- except
    during rest, so a rest dwell can complete and be counted."""

    gaps = True

    @property
    def state(self) -> L.LiveState:
        s = FakeRunner.state.fget(self)
        if self.gaps and self.frames % 4 == 0:
            return L.LiveState(point=None, unfiltered=None, updated_s=s.updated_s, frames=s.frames)
        return s


class RestOnAButtonDisplay(FakeDisplay):
    """During rest the gaze sits on a button: every fire there is unintended."""

    def wait_for_key(self, lines: list[str]) -> str:
        out = super().wait_for_key(lines)
        if lines and lines[0].startswith("Rest block"):
            self.runner.look = self.runner.centres["MID-C"]
            self.runner.gaps = False
        return out

    def draw_practice(self, buttons, point, *, prompt, **kw: object) -> None:
        if prompt and prompt[0].startswith("REST"):
            self.draws.append((prompt[0], point))
            return
        super().draw_practice(buttons, point, prompt=prompt, **kw)


def run_with_gaps_and_a_rest_on_a_button() -> dict:
    buttons = D.layout_zones()
    runner = GappyRunner(buttons)
    display = RestOnAButtonDisplay(runner)
    profile = mock.MagicMock()
    profile.name = "fake"
    with (
        mock.patch.object(P, "measure_bias", return_value=((0.01, 0.01), "test")),
        mock.patch("gf_fit.FittedModel.load", return_value=None),
        mock.patch.object(L, "build_gaze_follower", return_value=mock.MagicMock()),
        mock.patch.object(L, "LiveRunner", return_value=runner),
        mock.patch.object(P.R, "Display", return_value=display),
        mock.patch.object(P.R, "_sleep_with_escape", return_value=False),
        mock.patch.object(P.R, "shutdown_library"),
    ):
        return P.run_practice(
            profile,
            layout="zones",
            seed=1,
            n_trials=2,
            dwell_ms=100.0,
            idle_seconds=0.6,
            neutral_timeout_s=1.0,
            bar=None,
            show_bias=False,
        )


class EngineCallRowTests(unittest.TestCase):
    def test_row_format_keeps_missing_points_and_the_update_id(self) -> None:
        self.assertEqual(
            P.engine_call_row(12.345678, None, False, 7), [12.3457, None, None, False, 7]
        )
        self.assertEqual(
            P.engine_call_row(1.0, (0.123456789, 0.5), True, None),
            [1.0, 0.12345679, 0.5, True, None],
        )


class LoggedSessionReplays(unittest.TestCase):
    def test_default_session_logs_every_call_and_replays_exactly(self) -> None:
        report, _ = run_fake(n_trials=3)
        res = RP.replay_report(report, "fake")
        for raw, t in zip(report["trials"], res["trials"], strict=True):
            self.assertTrue(raw["engine_calls"])
            # One engine call per scored-window frame on the default path.
            self.assertEqual(len(raw["engine_calls"]), raw["frames"])
            self.assertEqual(t["source"], "engine_calls")
            self.assertFalse(t["partial"])
            self.assertEqual(t["replay_activation"], raw["activated"])
            self.assertTrue(t["agrees"])
            self.assertLessEqual(abs(t["replay_lag_ms"]), RP.AGREE_EPS_MS)
        self.assertEqual(res["agreement"]["agree"], res["agreement"]["of"])

    def test_fixed_window_stops_logging_when_the_engine_stops_being_fed(self) -> None:
        report, _ = run_fake(
            {"wander_after_flash": "MID-R"},
            point_schedule=["shown"],
            targets=["UP-L", "LOW-C"],
            fixed_window_s=1.0,
        )
        for raw in report["trials"]:
            last_call_ms = raw["engine_calls"][-1][0]
            self.assertLessEqual(last_call_ms, raw["activated_at_ms"] + 0.1)
            self.assertGreater(raw["frames"], len(raw["engine_calls"]))
        res = RP.replay_report(report, "fake")
        self.assertEqual(res["agreement"]["agree"], res["agreement"]["of"])

    def test_rest_calls_are_logged_and_replayed(self) -> None:
        report, _ = run_fake(n_trials=2)
        self.assertEqual(len(report["rest_engine_calls"]), 1)
        self.assertTrue(report["rest_engine_calls"][0]["engine_calls"])
        res = RP.replay_report(report, "fake")
        self.assertEqual(
            res["rest"]["replayed_unintended"], report["summary"]["unintended_activations"]
        )

    def test_missing_points_are_logged_as_not_fresh_and_rest_fires_are_replayed(self) -> None:
        report = run_with_gaps_and_a_rest_on_a_button()
        rows = [row for t in report["trials"] for row in t["engine_calls"]]
        missing = [row for row in rows if row[1] is None]
        self.assertTrue(missing)
        self.assertTrue(all(row[3] is False for row in missing))
        self.assertTrue(all(row[3] is True for row in rows if row[1] is not None))
        self.assertGreaterEqual(report["summary"]["unintended_activations"], 1)
        res = RP.replay_report(report, "fake")
        self.assertEqual(
            res["rest"]["replayed_unintended"], report["summary"]["unintended_activations"]
        )
        self.assertEqual(res["agreement"]["agree"], res["agreement"]["of"])

    def test_a_paused_trial_is_undetermined(self) -> None:
        report, _ = run_fake(n_trials=1)
        report["trials"][0]["pauses"] = 1
        report["trials"][0]["activated"] = None
        report["trials"][0]["activated_at_ms"] = None
        res = RP.replay_report(report, "fake")
        self.assertEqual(res["trials"][0]["classification"], "undetermined_from_log")
        self.assertEqual(res["agreement"]["paused_excluded"], 1)
        self.assertEqual(res["agreement"]["of"], 0)


if __name__ == "__main__":
    unittest.main()
