"""Recalibration decision logic: no camera, no fitting, no OS input."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_recalibrate as RC  # noqa: E402


def _paired(delta: float, fraction: float, n: int = 400) -> dict:
    return {
        "n_paired": n,
        "median_paired_delta_px": delta,
        "fraction_of_frames_new_is_better": fraction,
    }


class AdoptionRuleTests(unittest.TestCase):
    """Replacing a working calibration discards a known quantity for one
    measured over about forty seconds, so "not worse" is the wrong bar."""

    def test_a_marginal_gain_does_not_replace_a_working_calibration(self) -> None:
        """Measured for real: +5.9 px on 54% of frames. Adopting on that would
        churn the active profile for what the next run would reverse."""

        ok, why = RC.worth_adopting(_paired(5.9, 0.54))
        self.assertFalse(ok)
        self.assertIn("under the", why)

    def test_a_big_gain_on_inconsistent_frames_is_refused(self) -> None:
        """A median that improves while barely half the frames improve is a
        difference the next run would reverse."""

        ok, why = RC.worth_adopting(_paired(12.0, 0.55))
        self.assertFalse(ok)
        self.assertIn("consistent", why)

    def test_a_real_improvement_is_adopted(self) -> None:
        ok, why = RC.worth_adopting(_paired(40.0, 0.85))
        self.assertTrue(ok)
        self.assertIn("worth adopting", why)

    def test_a_regression_is_never_adopted(self) -> None:
        for delta, fraction in ((-5.0, 0.45), (-60.0, 0.1), (-200.0, 0.0)):
            ok, _ = RC.worth_adopting(_paired(delta, fraction))
            self.assertFalse(ok, f"{delta} px at {fraction} was adopted")

    def test_nothing_compared_is_not_an_improvement(self) -> None:
        """An empty comparison must not read as success."""

        ok, why = RC.worth_adopting({"n_paired": 0})
        self.assertFalse(ok)
        self.assertIn("nothing was compared", why)

    def test_the_thresholds_are_stated_not_hidden(self) -> None:
        self.assertGreater(RC.MIN_IMPROVEMENT_PX, 0.0)
        self.assertGreater(RC.MIN_FRACTION_BETTER, 0.5)


class UsableCalibrationTests(unittest.TestCase):
    """A camera that produced nothing is a session failure, not a model
    failure, and must say so before anything is fitted."""

    def setUp(self) -> None:
        self._load = RC.S.Recording.load

    def tearDown(self) -> None:
        RC.S.Recording.load = self._load

    def _install(self, accepted: int, targets_seen: int, n_targets: int = 9) -> None:
        import numpy as np

        total = max(accepted, 1)
        ids = np.zeros(total, dtype=int)
        for i in range(min(targets_seen, total)):
            ids[i] = i
        mask = np.zeros(total, dtype=bool)
        mask[:accepted] = True

        def fake(directory, protocol):  # noqa: ARG001
            return SimpleNamespace(
                rows_accepted=lambda: mask,
                target_id=ids,
                targets=[{"index": i} for i in range(n_targets)],
            )

        RC.S.Recording.load = fake

    def test_a_full_calibration_is_accepted(self) -> None:
        self._install(accepted=9 * RC.C.N_FRAMES_PER_POINT, targets_seen=9)
        self.assertEqual(RC.require_usable_calibration(Path(".")), 9 * RC.C.N_FRAMES_PER_POINT)

    def test_a_recording_with_no_rows_is_refused_with_the_real_reason(self) -> None:
        """The fitter would say 'need at least 4 training rows', which reads
        as a problem with the model when the camera produced nothing."""

        self._install(accepted=0, targets_seen=0)
        with self.assertRaises(SystemExit) as caught:
            RC.require_usable_calibration(Path("."))
        message = str(caught.exception)
        self.assertIn("not usable", message)
        self.assertIn("Nothing was changed", message)

    def test_a_partial_calibration_is_refused(self) -> None:
        """Half the points collected is not a calibration; fitting it would
        produce a model confident over a region it never saw."""

        self._install(accepted=4 * RC.C.N_FRAMES_PER_POINT, targets_seen=4)
        with self.assertRaises(SystemExit):
            RC.require_usable_calibration(Path("."))


class RoundIdTests(unittest.TestCase):
    def test_the_series_continues_rather_than_filling_a_gap(self) -> None:
        """Filling a historical gap drops a new recording into the middle of
        the series, where it reads as one of the old experiments."""

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("round0", "round1", "round3"):
                (root / name).mkdir()
            self.assertEqual(RC.next_free_round(root), 4)

    def test_an_empty_root_starts_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(RC.next_free_round(Path(d)), 0)

    def test_a_used_round_is_never_returned(self) -> None:
        """Returning a used id would record over data an earlier session left."""

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(6):
                (root / f"round{i}").mkdir()
            (root / "notes.txt").write_text("x", encoding="utf-8")
            (root / "roundXY").mkdir()
            chosen = RC.next_free_round(root)
            self.assertEqual(chosen, 6)
            self.assertFalse((root / f"round{chosen}").exists())


if __name__ == "__main__":
    unittest.main()


class MonitorIdentityTests(unittest.TestCase):
    """A recalibration must record WHICH display it was made on.

    ``run_session`` writes ``"monitor": None`` when it is not given one, and a
    recording with no monitor block produces a profile with no connector, no
    resolution and no physical millimetres.  ``gf_screen_check`` then refuses
    that profile -- a panel it cannot identify is not a verified ruler -- so
    the calibration is unusable for a cursor no matter how well it scored.
    Measured on round32: 6 of 7 checks passed, "physical size" failed.
    """

    def _profile(self) -> Any:
        import gf_profile as PROF  # noqa: PLC0415

        return PROF.Profile(
            name="t",
            model_dir=str(Path(__file__).resolve().parent.parent / "models" / "cfg"),
            rig={
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            filter={"kind": "one-euro"},
            x_range=[0.3, 0.7],
        )

    def _capture_run_session(self) -> tuple[dict, Any]:
        """Run the real run_sequence far enough to see what it asks for."""

        captured: dict = {}

        def fake_run_session(**kwargs: Any) -> Any:
            captured.update(kwargs)
            # Aborted, so run_sequence stops before fitting anything.
            return SimpleNamespace(
                aborted=True,
                watchdog_tripped=False,
                saved=False,
                failure="stopped by the test",
                recordings={},
            )

        sentinel = SimpleNamespace(name="THE-PANEL", origin=(0, 0), width_px=5120, height_px=1440)
        real_run_session = RC.R.run_session
        real_pick = RC.GD.pick_monitor
        RC.R.run_session = fake_run_session
        RC.GD.pick_monitor = lambda selector=None: sentinel
        try:
            with self.assertRaises(SystemExit):
                RC.run_sequence(self._profile(), round_id=99)
        finally:
            RC.R.run_session = real_run_session
            RC.GD.pick_monitor = real_pick
        return captured, sentinel

    def test_a_recalibration_records_the_display_it_ran_on(self) -> None:
        captured, sentinel = self._capture_run_session()
        self.assertIs(
            captured.get("monitor"),
            sentinel,
            "run_sequence passed no monitor, so the recording will carry "
            '"monitor": null and the new profile will have no ruler to verify',
        )

    def test_naming_a_display_on_the_command_line_reaches_the_session(self) -> None:
        """--monitor was parsed and then dropped, so naming a display did nothing."""

        seen: dict = {}
        chosen = SimpleNamespace(name="SECOND-PANEL")

        def fake_run_sequence(profile: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            raise SystemExit("stopped by the test")

        real_run_sequence = RC.run_sequence
        real_pick = RC.GD.pick_monitor
        real_resolve = RC.L.resolve_profile
        RC.run_sequence = fake_run_sequence
        RC.GD.pick_monitor = lambda selector=None: chosen
        RC.L.resolve_profile = lambda name: self._profile()
        try:
            with self.assertRaises(SystemExit):
                RC.main(["--monitor", "SECOND", "--adopt", "never"])
        finally:
            RC.run_sequence = real_run_sequence
            RC.GD.pick_monitor = real_pick
            RC.L.resolve_profile = real_resolve
        self.assertIs(
            seen.get("monitor"),
            chosen,
            "--monitor was accepted and then ignored, so the session ran on whichever "
            "display happened to be primary",
        )
