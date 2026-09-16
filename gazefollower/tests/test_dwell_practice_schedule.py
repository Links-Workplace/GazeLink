"""The point-feedback probe: hidden/shown mini-blocks on a fixed window.

What must hold for the comparison to mean anything:
* a hidden block never draws the gaze point, and nothing else changes -- the
  engine still gets the real point, so selection is scored identically;
* every block holds the same targets the same number of times;
* with a fixed window only the first activation counts, the target stays up,
  and recording continues to the end of the window;
* the old command line (no schedule) plans exactly the order it always did;
* callback timing is summarised off the camera thread.

No camera, no library, no OS input: the runner, display and library are fakes.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_dwell as D  # noqa: E402
import gf_dwell_practice as P  # noqa: E402
import gf_live as L  # noqa: E402

KEYS = [b.key for b in D.layout_zones()]


class PlanTests(unittest.TestCase):
    def test_no_schedule_is_the_order_used_so_far(self) -> None:
        plan = P.build_trial_plan(
            KEYS, n_trials=18, seed=3, pattern="shuffled", point_schedule=None, targets=None
        )
        self.assertEqual([k for _, _, k in plan], P.trial_order(KEYS, 18, 3))
        self.assertTrue(all(shown and block == 0 for block, shown, _ in plan))

    def test_every_block_holds_the_same_targets_once(self) -> None:
        targets = ["LOW-L", "LOW-C", "LOW-R", "MID-L", "MID-C", "UP-C"]
        schedule = ["hidden", "shown", "shown", "hidden"]
        plan = P.build_trial_plan(
            KEYS, n_trials=99, seed=4, pattern="shuffled", point_schedule=schedule, targets=targets
        )
        self.assertEqual(len(plan), 24)
        for block, condition in enumerate(schedule):
            rows = [r for r in plan if r[0] == block]
            self.assertEqual(sorted(k for _, _, k in rows), sorted(targets))
            self.assertTrue(all(shown == (condition == "shown") for _, shown, _ in rows))

    def test_bad_input_is_refused_before_anything_opens(self) -> None:
        with self.assertRaises(SystemExit):
            P.parse_point_schedule("hidden,maybe")
        with self.assertRaises(SystemExit):
            P.build_trial_plan(
                KEYS, n_trials=6, seed=1, pattern="shuffled", point_schedule=["shown"], targets=None
            )
        with self.assertRaises(SystemExit):
            P.build_trial_plan(
                KEYS, n_trials=6, seed=1, pattern="shuffled", point_schedule=None, targets=["NOPE"]
            )
        with self.assertRaises(SystemExit):
            P.build_trial_plan(
                KEYS, n_trials=6, seed=1, pattern="shuffled", point_schedule=["shown"],
                targets=["UP-L", "UP-L"],
            )

    def test_timing_summary(self) -> None:
        s = P.timing_summary([(None, 2.0), (30.0, 4.0), (34.0, 6.0)])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["callback_ms"]["median"], 4.0)
        self.assertEqual(s["callback_ms"]["max"], 6.0)
        self.assertEqual(s["interval_ms"]["median"], 32.0)
        self.assertIsNone(P.timing_summary([])["callback_ms"])


class RunnerTimingTests(unittest.TestCase):
    def test_on_frame_records_duration_and_interval_and_drain_clears(self) -> None:
        runner = L.LiveRunner(None, None, mock.MagicMock(), None)
        ticks = iter([1.000, 1.002, 1.033, 1.036])
        runner.perf_clock = lambda: next(ticks)
        runner._on_frame = lambda face, gaze: None  # the timed body is not under test
        runner.on_frame(None, None)
        runner.on_frame(None, None)
        got = runner.drain_timing()
        self.assertEqual(len(got), 2)
        self.assertIsNone(got[0][0])
        self.assertAlmostEqual(got[0][1], 2.0, places=6)
        self.assertAlmostEqual(got[1][0], 33.0, places=6)
        self.assertAlmostEqual(got[1][1], 3.0, places=6)
        self.assertEqual(runner.drain_timing(), [])

    def test_the_buffer_is_bounded(self) -> None:
        runner = L.LiveRunner(None, None, mock.MagicMock(), None)
        runner._on_frame = lambda face, gaze: None
        for _ in range(L.TIMING_BUFFER + 50):
            runner.on_frame(None, None)
        self.assertEqual(len(runner.drain_timing()), L.TIMING_BUFFER)


class FakeRunner:
    """Puts the gaze wherever the fake display last asked the operator to look."""

    def __init__(self, buttons: list[D.Button]) -> None:
        self.centres = {b.key: b.centre for b in buttons}
        self.look: tuple[float, float] = (0.5, 0.33)  # dead space between rows
        self.frames = 0
        self.drains = 0

    @property
    def state(self) -> L.LiveState:
        self.frames += 1
        return L.LiveState(
            point=self.look, unfiltered=self.look, updated_s=time.monotonic(), frames=self.frames
        )

    def drain_gesture_events(self) -> list:
        return []

    def drain_timing(self) -> list:
        # Each drain returns as many samples as drains so far, so a trial's
        # count shows which drains bracketed it.
        self.drains += 1
        return [(31.0, 2.0)] * self.drains


class FakeDisplay:
    def __init__(self, runner: FakeRunner, *, wander_after_flash: str | None = None) -> None:
        self.runner = runner
        self.draws: list[tuple[str, tuple[float, float] | None]] = []
        self.rings: list[tuple[str | None, str | None, float]] = []  # (flash, hovered, progress)
        self.rests = 0
        # After an activation, look at this other button instead: a second
        # activation can then only come from an engine that is still fed.
        self.wander_after_flash = wander_after_flash

    def poll_escape(self) -> bool:
        return False

    def wait_for_key(self, lines: list[str]) -> str:
        if lines and lines[0].startswith("Rest block"):
            self.rests += 1
            self.runner.look = (0.5, 0.33)
        return "ok"

    def draw_message(self, lines: list[str]) -> None:
        pass

    def close(self) -> None:
        pass

    def draw_practice(self, buttons, point, *, prompt, flash=None, hovered=None,
                      progress=0.0, **_: object) -> None:
        first = prompt[0]
        if "LOOK AT:" in first:
            key = first.split("LOOK AT:")[1].strip()
            if flash is not None and self.wander_after_flash is not None:
                self.runner.look = self.runner.centres[self.wander_after_flash]
            else:
                self.runner.look = self.runner.centres[key]
            self.rings.append((flash, hovered, progress))
        elif len(prompt) > 1 and "LOOK AWAY" in prompt[1]:
            self.runner.look = (0.5, 0.33)
        self.draws.append((first, point))


def run_fake(display_kwargs: dict | None = None, **kwargs: object) -> tuple[dict, FakeDisplay]:
    buttons = D.layout_zones()
    runner = FakeRunner(buttons)
    display = FakeDisplay(runner, **(display_kwargs or {}))
    profile = mock.MagicMock()
    profile.name = "fake"
    with mock.patch.object(P, "measure_bias", return_value=((0.01, 0.01), "test")), \
            mock.patch("gf_fit.FittedModel.load", return_value=None), \
            mock.patch.object(L, "build_gaze_follower", return_value=mock.MagicMock()), \
            mock.patch.object(L, "LiveRunner", return_value=runner), \
            mock.patch.object(P.R, "Display", return_value=display), \
            mock.patch.object(P.R, "_sleep_with_escape", return_value=False), \
            mock.patch.object(P.R, "shutdown_library"):
        report = P.run_practice(
            profile, layout="zones", seed=1, dwell_ms=100.0, idle_seconds=0.05,
            neutral_timeout_s=1.0, bar=None, show_bias=False, **kwargs,
        )
    return report, display


class FixedWindowTests(unittest.TestCase):
    def test_the_window_runs_to_its_end_and_nothing_fires_after_the_first_activation(self) -> None:
        report, display = run_fake(
            {"wander_after_flash": "MID-R"},
            point_schedule=["shown"], targets=["UP-L", "LOW-C"], fixed_window_s=1.5,
        )
        for t in report["trials"]:
            self.assertEqual(t["activated"], t["requested"])
            # The gaze moved onto MID-R for over a second: an engine still being
            # fed would have fired it.
            self.assertEqual(t["extra_activations"], [])
            self.assertGreaterEqual(t["window_end_ms"], 1500.0)
            self.assertGreaterEqual(t["raw_path"][-1][0], 1400.0)
        # No ring once selection has ended; the highlight stays.
        after = [r for r in display.rings if r[0] is not None]
        self.assertTrue(after)
        self.assertTrue(all(hovered is None and progress == 0.0 for _, hovered, progress in after))

    def test_each_trial_counts_only_its_own_callbacks(self) -> None:
        report, _ = run_fake(
            point_schedule=["shown", "hidden"], targets=["UP-L", "LOW-C"], fixed_window_s=0.5,
        )
        # Drains alternate start (discarded) / end (kept): 2, 4, 6, 8.
        self.assertEqual([t["callback"]["n"] for t in report["trials"]], [2, 4, 6, 8])

    def test_unintended_rate_uses_the_rest_time_actually_run(self) -> None:
        report, display = run_fake(
            point_schedule=["hidden", "shown", "shown"], targets=["UP-L"], fixed_window_s=0.4,
        )
        self.assertEqual(display.rests, 3)
        self.assertAlmostEqual(report["summary"]["idle_seconds"], 0.15, places=6)

    def test_a_nan_window_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            P.check_fixed_window(float("nan"))
        with self.assertRaises(SystemExit):
            P.check_fixed_window(0.0)
        P.check_fixed_window(None)


class HeadLoggingTests(unittest.TestCase):
    def test_live_state_carries_the_head6_the_runner_built(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        head = (1.5, 0.04, -0.1, 0.51, 0.5, 0.155)
        runner = L.LiveRunner(None, None, mock.MagicMock(), None, head_builder=lambda face: head)
        face = SimpleNamespace(status=True, left_eye_openness=100.0, right_eye_openness=100.0)
        gaze = SimpleNamespace(status=False, features=None, raw_gaze_coordinates=None)
        runner.on_frame(face, gaze)
        self.assertIsNone(runner.last_error)
        self.assertEqual(runner.state.head, head)
        runner.on_frame(SimpleNamespace(status=False, left_eye_openness=0.0,
                                        right_eye_openness=0.0), gaze)
        self.assertIsNone(runner.state.head)  # no face: unknown, never a number

    def test_every_logged_sample_has_a_head_row_at_the_same_instant(self) -> None:
        def head_of(frame: int) -> tuple[float, ...]:
            # Differs every frame, so a row taken from another state is caught.
            return (float(frame), 0.05, -0.12, 0.5, 0.49, 0.16)

        class HeadRunner(FakeRunner):
            @property
            def state(self) -> L.LiveState:
                s = FakeRunner.state.fget(self)
                return dataclasses.replace(s, head=head_of(s.frames))

        buttons = D.layout_zones()
        runner = HeadRunner(buttons)
        display = FakeDisplay(runner)
        with mock.patch.object(P, "measure_bias", return_value=((0.01, 0.01), "test")), \
                mock.patch("gf_fit.FittedModel.load", return_value=None), \
                mock.patch.object(L, "build_gaze_follower", return_value=mock.MagicMock()), \
                mock.patch.object(L, "LiveRunner", return_value=runner), \
                mock.patch.object(P.R, "Display", return_value=display), \
                mock.patch.object(P.R, "_sleep_with_escape", return_value=False), \
                mock.patch.object(P.R, "shutdown_library"):
            profile = mock.MagicMock()
            profile.name = "fake"
            report = P.run_practice(profile, layout="zones", n_trials=2, seed=1, dwell_ms=100.0,
                                    idle_seconds=0.0, neutral_timeout_s=1.0, bar=None,
                                    show_bias=False)
        for t in report["trials"]:
            self.assertEqual(len(t["head_path"]), len(t["raw_path"]))
            for h, r in zip(t["head_path"], t["raw_path"], strict=True):
                self.assertEqual(h[0], r[0])
                self.assertEqual(h[1], r[3])
                self.assertEqual(tuple(h[2:]), head_of(h[1]))


class DefaultPathTests(unittest.TestCase):
    def test_without_the_new_flags_the_session_runs_as_before(self) -> None:
        report, display = run_fake(n_trials=3)
        self.assertEqual(display.rests, 1)
        trials = report["trials"]
        self.assertEqual(len(trials), 3)
        self.assertEqual(report["order"], P.trial_order(KEYS, 3, 1))
        for t in trials:
            self.assertTrue(t["point_shown"])
            self.assertEqual(t["activated"], t["requested"])
            # The 0.4 s post-activation break still ends the trial.
            self.assertLess(t["window_end_ms"], t["activated_at_ms"] + 700.0)
        points = [p for first, p in display.draws if first.startswith("trial ")]
        self.assertTrue(points and all(p is not None for p in points))
        # The ring is not suppressed after activation on the default path.
        self.assertTrue(any(flash is not None and progress > 0.0
                            for flash, _, progress in display.rings))


class ScheduledSessionTests(unittest.TestCase):
    def test_hidden_blocks_hide_only_the_point_and_score_the_same(self) -> None:
        buttons = D.layout_zones()
        runner = FakeRunner(buttons)
        display = FakeDisplay(runner)
        gf = mock.MagicMock()
        profile = mock.MagicMock()
        profile.name = "fake"
        with mock.patch.object(P, "measure_bias", return_value=((0.01, 0.01), "test")), \
                mock.patch("gf_fit.FittedModel.load", return_value=None), \
                mock.patch.object(L, "build_gaze_follower", return_value=gf), \
                mock.patch.object(L, "LiveRunner", return_value=runner), \
                mock.patch.object(P.R, "Display", return_value=display), \
                mock.patch.object(P.R, "_sleep_with_escape", return_value=False), \
                mock.patch.object(P.R, "shutdown_library"):
            report = P.run_practice(
                profile,
                layout="zones",
                seed=1,
                dwell_ms=100.0,
                idle_seconds=0.05,
                neutral_timeout_s=1.0,
                bar=None,
                point_schedule=["hidden", "shown"],
                targets=["UP-L", "LOW-C"],
                fixed_window_s=0.6,
                show_bias=False,
            )
        trials = report["trials"]
        self.assertEqual(len(trials), 4)
        self.assertEqual([t["point_shown"] for t in trials], [False, False, True, True])
        self.assertEqual(display.rests, 2)  # one per mini-block
        for t in trials:
            self.assertEqual(t["activated"], t["requested"])
            self.assertEqual(t["extra_activations"], [])
            # The target stayed up and recording went on past the activation.
            self.assertGreater(t["raw_path"][-1][0], t["activated_at_ms"] + 300.0)
            self.assertEqual(len(t["raw_path"][0]), 5)
            self.assertGreater(t["callback"]["n"], 0)
        def points_of(*numbers: int) -> list:
            tags = tuple(f"trial {n}/" for n in numbers)
            return [p for first, p in display.draws if first.startswith(tags)]

        hidden_points = points_of(1, 2)
        shown_points = points_of(3, 4)
        self.assertTrue(hidden_points and all(p is None for p in hidden_points))
        self.assertTrue(any(p is not None for p in shown_points))
        self.assertEqual(report["summary"]["correct_first"], 4)
        self.assertEqual(report["fixed_window_s"], 0.6)


if __name__ == "__main__":
    unittest.main()
