"""Recalibration decision logic: no camera, no fitting, no OS input."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_pool as POOL_MODULE  # noqa: E402
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


class CarriedForwardTests(unittest.TestCase):
    """A recalibration replaces the GAZE MODEL. It does not measure the
    person's eyelids and it does not measure what a pointer should feel like,
    so it must not reset either.

    ``adopt`` is the only code path in the project that creates a profile, and
    it built one through ``PROF.from_recording``, which takes no ``gesture``
    and no ``cursor``.  Both blocks therefore defaulted to ``{}`` and the live
    session silently fell back to built-in values: pointer smoothing 0.35 and
    dead zone 12, which the shipped profile records as "felt jumpy" in the
    field explaining why 0.6 / 6 replaced them.
    """

    # Deliberately NOT the built-in defaults, every value. The shipped profile
    # happens to store the defaults, and a fixture copying it made
    # ``test_the_live_session_reads_the_carried_wink_rule`` pass with the carry
    # removed -- ``wink_config()`` returns those same numbers from an empty
    # block. A test that cannot tell the two apart is testing nothing.
    GESTURE = {
        "wink": {"shut_ratio": 0.40, "asymmetry": 3.0, "hold_ms": 35.0, "bridge_ms": 25.0},
        "gate": {"shut_fraction": 0.50, "steady_fraction": 0.70},
        "evidence": {"frames": 12818, "selections": 25},
    }
    CURSOR = {
        "smoothing": 0.6,
        "dead_zone_px": 6,
        "max_step_px": 400,
        "chosen_by": "operator, live: 0.35/12 felt jumpy, 0.6/6 felt right",
    }

    def _old_profile(self, **overrides: Any) -> Any:
        import gf_profile as PROF  # noqa: PLC0415

        fields: dict[str, Any] = {
            "name": "baseline",
            "model_dir": "recordings/round18/models/cfg",
            "rig": {
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            "filter": {"kind": "one-euro"},
            "gesture": dict(self.GESTURE),
            "cursor": dict(self.CURSOR),
        }
        fields.update(overrides)
        return PROF.Profile(**fields)

    def _adopt(self, old: Any) -> Any:
        """Run the real ``adopt``; capture the profile instead of writing it.

        ``PROF.save`` and ``PROF.activate`` are stubbed because ``adopt``
        writes to the real profiles directory, and a test must not activate a
        profile on the machine it runs on.
        """

        import gf_profile as PROF  # noqa: PLC0415

        saved: dict = {}
        fresh = PROF.Profile(
            name="rebuilt-by-from_recording",
            model_dir="recordings/round40/models/cfg",
            rig={},
            filter={"kind": "one-euro"},
        )
        outcome = RC.Outcome(
            recording_dir=Path("recordings/round40"),
            model_dir=Path("recordings/round40/models/cfg"),
            old={
                "filtered": {"median_euclid_px": 180.0},
                "unfiltered": {"median_euclid_px": 190.0},
            },
            new={
                "filtered": {"median_euclid_px": 150.0},
                "unfiltered": {"median_euclid_px": 160.0},
            },
            paired=_paired(30.0, 0.8),
            better=True,
            summary="",
        )
        real = (PROF.from_recording, PROF.save, PROF.activate, RC.PROF.from_recording)
        RC.PROF.from_recording = lambda *a, **k: fresh
        RC.PROF.save = lambda profile, **k: saved.update(profile=profile)
        RC.PROF.activate = lambda name, **k: saved.update(activated=name)
        try:
            RC.adopt(old, outcome)
        finally:
            (PROF.from_recording, PROF.save, PROF.activate, RC.PROF.from_recording) = real
        return saved["profile"]

    def test_the_wink_rule_survives_a_recalibration(self) -> None:
        got = self._adopt(self._old_profile())
        self.assertEqual(
            got.gesture,
            self.GESTURE,
            "the new profile has no wink rule, so the next session judged the person's "
            "eyelids by the built-in defaults instead of by what was measured on them",
        )

    def test_the_pointer_feel_survives_a_recalibration(self) -> None:
        got = self._adopt(self._old_profile())
        self.assertEqual(
            got.cursor,
            self.CURSOR,
            "the new profile has no cursor block, so the pointer reverted to smoothing "
            "0.35 / dead zone 12 -- the setting the old profile records as felt jumpy",
        )

    def test_the_live_session_reads_the_carried_wink_rule(self) -> None:
        """Not just present in the dict: reaching the object the detector uses.

        Asserting on ``gesture`` alone would pass if the block were carried
        under a key ``wink_config`` never looks at.
        """

        import gf_gesture as GEST  # noqa: PLC0415

        default = GEST.WinkConfig()
        self.assertNotAlmostEqual(
            default.hold_ms,
            self.GESTURE["wink"]["hold_ms"],
            msg="the fixture now matches the built-in default, so this test can no longer "
            "tell a carried rule from an empty one -- change the fixture, not the assert",
        )
        got = self._adopt(self._old_profile())
        wink = got.wink_config()
        self.assertAlmostEqual(wink.hold_ms, 35.0)
        self.assertAlmostEqual(wink.bridge_ms, 25.0)
        self.assertAlmostEqual(wink.shut_ratio, 0.40)
        self.assertAlmostEqual(got.gate_config().steady_fraction, 0.70)

    def test_a_carried_block_is_a_copy_not_a_shared_dict(self) -> None:
        """Editing the new profile must not reach back into the old one.

        Both are in memory at once in ``main``, which reports on the old
        profile after building the new one.
        """

        old = self._old_profile()
        got = self._adopt(old)
        self.assertTrue(got.cursor, "nothing was carried, so this proves nothing about sharing")
        got.cursor["smoothing"] = 0.99
        self.assertEqual(
            old.cursor["smoothing"], 0.6, "the two profiles share one dict, so they are one profile"
        )

    def test_what_the_recalibration_does_measure_is_not_carried(self) -> None:
        """The counter-test. A carry that took the model, the scores or the
        calibration pose would keep the OLD calibration under a new name --
        which is the opposite of recalibrating."""

        got = self._adopt(self._old_profile())
        self.assertEqual(
            got.model_dir,
            "recordings/round40/models/cfg",
            "the new profile points at the old model, so nothing was recalibrated",
        )
        self.assertNotIn("chosen_by", got.scores)

    def test_an_old_profile_with_no_settings_stays_empty(self) -> None:
        """Carrying must not invent values. An empty block means "use the
        built-in defaults", and turning that into a stored rule would claim a
        measurement nobody made."""

        got = self._adopt(self._old_profile(gesture={}, cursor={}))
        self.assertEqual(got.gesture, {})
        self.assertEqual(got.cursor, {})

    def test_every_carried_field_exists_on_a_profile(self) -> None:
        """A typo in CARRIED_FORWARD would raise inside ``replace`` only on a
        real adoption, at the end of a forty-second calibration session."""

        import gf_profile as PROF  # noqa: PLC0415

        for field in RC.CARRIED_FORWARD:
            self.assertIn(field, PROF.Profile.__dataclass_fields__, field)

    def test_every_name_in_the_constant_is_actually_carried(self) -> None:
        """The constant and the copy are written in two places, so they can
        drift: a field added to ``CARRIED_FORWARD`` and not to ``adopt`` would
        be documented as preserved and silently dropped anyway."""

        marker = {"carried": "sentinel"}
        for field in RC.CARRIED_FORWARD:
            with self.subTest(field=field):
                old = self._old_profile(**{field: dict(marker)})
                got = self._adopt(old)
                self.assertEqual(
                    getattr(got, field),
                    marker,
                    f"{field!r} is listed in CARRIED_FORWARD but adopt() never copies it",
                )


class PooledFitTests(unittest.TestCase):
    """Recalibration now trains on every compatible past recording rather than
    on the newest session alone.

    Measured, leave-one-session-out over 13 sessions: a single fresh
    calibration scored a median 226.2 px on its own held-out check and the
    pooled fit 117.6 px, better on 11 of 11 folds; still 142.1 px and 10 of 11
    when the held-out session's screen POSITIONS were also removed from
    training. The cause of the old number is in ``gf_pool``: 405 rows at nine
    positions in a 258-dimensional space is memorised, not learned.
    """

    def _profile(self, model_dir: str = "recordings/round18/models/cfg") -> Any:
        import gf_profile as PROF  # noqa: PLC0415

        return PROF.Profile(
            name="baseline",
            model_dir=model_dir,
            rig={
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            filter={"kind": "one-euro"},
        )

    def _capture_pool_build(self, recording_dir: str, config_name: str = "cfg") -> dict:
        """Run the real ``fit_pooled`` far enough to see what it asks the pool
        for, then stop before any model is fitted."""

        seen: dict = {}

        class Stop(Exception):
            pass

        def fake_build(head_names: Any, **kwargs: Any) -> Any:
            seen["head_names"] = head_names
            seen.update(kwargs)
            raise Stop

        real_build = RC.POOL.build
        RC.POOL.build = fake_build
        try:
            with self.assertRaises(Stop):
                RC.fit_pooled(Path(recording_dir), config_name)
        finally:
            RC.POOL.build = real_build
        return seen

    def test_the_check_recording_is_excluded_from_its_own_training_set(self) -> None:
        """The one mistake that would make every number meaningless while
        looking like a large improvement: scoring a model on rows that
        trained it reports memory as accuracy."""

        seen = self._capture_pool_build("recordings/round40", "svr_fzscore_lnone_C100_g0.0005")
        self.assertEqual(
            [(Path("recordings/round40"), "T1")],
            list(seen.get("exclude", [])),
            "the pooled fit did not hold out the recording it is about to be scored on",
        )

    def test_the_pool_is_refused_rather_than_fitted_when_it_is_too_small(self) -> None:
        """A pool below the measured shape has no evidence behind it, and a
        number produced from one would look exactly like a measured one."""

        import gf_pool as POOL  # noqa: PLC0415

        empty = POOL.build((), root=Path(tempfile.mkdtemp()))
        real_build = RC.POOL.build
        RC.POOL.build = lambda *a, **k: empty
        try:
            with self.assertRaises(SystemExit) as caught:
                RC.fit_pooled(Path("recordings/round40"), "svr_fzscore_lnone_C100_g0.0005")
        finally:
            RC.POOL.build = real_build
        self.assertIn("too little to learn from", str(caught.exception))

    def test_an_unknown_configuration_is_refused_before_any_pool_is_read(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            RC.fit_pooled(Path("recordings/round40"), "no-such-configuration")
        self.assertIn("does not know", str(caught.exception))

    def test_recalibrating_a_pooled_profile_does_not_compound_the_suffix(self) -> None:
        """A pooled model is saved under ``<config>__pooled``, and
        ``run_sequence`` takes the configuration name from the profile's model
        directory. Without the strip, a second recalibration would look up
        ``<config>__pooled__pooled`` and exit with "a configuration this build
        does not know" -- at the END of a calibration session, having thrown
        it away."""

        import gf_presets as PRE  # noqa: PLC0415

        pooled_dir = "recordings/round40/models/svr_fzscore_lnone_C100_g0.0005__pooled"
        names = {c.name for c in PRE.sweep_with_presets(())}
        self.assertNotIn(
            Path(pooled_dir).name, names, "the suffixed name is now a real configuration"
        )
        # The real production expression, not a copy of it.
        stripped = Path(self._profile(pooled_dir).model_dir).name.removesuffix("__pooled")
        self.assertIn(stripped, names, "a pooled profile cannot be recalibrated again")

    def test_the_fit_choice_reaches_the_sequence(self) -> None:
        """--fit session must actually select the old single-session path;
        a flag that is parsed and dropped would silently keep the default."""

        seen: dict = {}

        def fake_run_sequence(profile: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            raise SystemExit("stopped by the test")

        real_run_sequence = RC.run_sequence
        real_resolve = RC.L.resolve_profile
        RC.run_sequence = fake_run_sequence
        RC.L.resolve_profile = lambda name: self._profile()
        try:
            with self.assertRaises(SystemExit):
                RC.main(["--fit", "session", "--adopt", "never"])
        finally:
            RC.run_sequence = real_run_sequence
            RC.L.resolve_profile = real_resolve
        self.assertEqual(seen.get("fit"), "session")

    def test_pooled_is_the_default(self) -> None:
        self.assertEqual(RC.build_parser().parse_args([]).fit, "pooled")

    def _adopt_with(self, pool: Any, calibration: dict) -> Any:
        import gf_profile as PROF  # noqa: PLC0415

        saved: dict = {}
        fresh = PROF.Profile(
            name="new",
            model_dir="m",
            rig={},
            filter={"kind": "one-euro"},
            calibration=dict(calibration),
        )
        outcome = RC.Outcome(
            recording_dir=Path("recordings/round40"),
            model_dir=Path("m"),
            old={"filtered": {"median_euclid_px": 180.0}, "unfiltered": {"median_euclid_px": 1.0}},
            new={"filtered": {"median_euclid_px": 150.0}, "unfiltered": {"median_euclid_px": 1.0}},
            paired=_paired(30.0, 0.8),
            better=True,
            summary="",
            pool=pool,
        )
        real = (RC.PROF.from_recording, RC.PROF.save, RC.PROF.activate)
        RC.PROF.from_recording = lambda name, **k: fresh
        RC.PROF.save = lambda profile, **k: saved.update(profile=profile)
        RC.PROF.activate = lambda name, **k: None
        try:
            RC.adopt(self._profile(), outcome)
        finally:
            (RC.PROF.from_recording, RC.PROF.save, RC.PROF.activate) = real
        return saved["profile"]

    def test_the_profile_records_what_trained_it(self) -> None:
        """A model fitted from an accumulating pool is reproducible only if
        the pool is written down; the recording directory names one session
        out of the dozens that trained it."""

        import gf_pool as POOL  # noqa: PLC0415

        pool = POOL.Pool(
            X=np.zeros((3, 2)),
            Y_cm=np.zeros((3, 2)),
            rig=self._profile().rig_geometry(),
            sources=[POOL.Source(Path("recordings/round18"), "A", 18, 405, 9, "s1")],
        )
        got = self._adopt_with(pool, {"round_id": 40}).calibration
        self.assertEqual(got["fitted_on"], "pool")
        self.assertEqual(got["pool"]["recordings"][0]["protocol"], "A")
        self.assertEqual(
            got["round_id"], 40, "the existing calibration block was replaced, not extended"
        )

    def test_a_session_fit_never_claims_a_pool(self) -> None:
        """``--fit session`` produces the old single-session model. A profile
        that recorded a pool it was not fitted from would be a false
        provenance -- worse than none, because it reads as evidence."""

        got = self._adopt_with(None, {"round_id": 40}).calibration
        self.assertNotIn("pool", got)
        self.assertNotIn("fitted_on", got)


class ReviewBlockerTests(unittest.TestCase):
    """The five defects an independent review found in the pooled path.

    Each is a way the mechanism could look like it worked while destroying a
    calibration or measuring memory as accuracy.
    """

    def _profile(self, model_dir: str) -> Any:
        import gf_profile as PROF  # noqa: PLC0415

        return PROF.Profile(
            name="baseline",
            model_dir=model_dir,
            rig={
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            filter={"kind": "one-euro"},
        )

    # --- 1. a refit must never replace a model a profile points at ----------

    def test_the_pooled_model_goes_to_a_new_directory_every_time(self) -> None:
        """It used to be a fixed name written with overwrite=True. Refitting
        against a recording whose pooled model the active profile referenced
        replaced that model IN PLACE, so ``compare_on_check`` then loaded the
        SAME files as both the old and the new side -- and --adopt never still
        destroyed the calibration someone was relying on."""

        import gf_pool as POOL  # noqa: PLC0415

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        recording = root / "round9"
        (recording / "models").mkdir(parents=True)
        config = "svr_fzscore_lnone_C100_g0.0005"
        existing = recording / "models" / f"{config}__pooled_20260101_000000"
        existing.mkdir()
        (existing / "schema.json").write_text('{"train": {}}', encoding="utf-8")

        saved: list[Path] = []

        class FakeModel:
            @staticmethod
            def fit(*a: Any, **k: Any) -> Any:
                return FakeModel()

            def save(self, target: Path) -> Path:
                saved.append(Path(target))
                Path(target).mkdir(parents=True)
                return Path(target)

        pool = POOL.Pool(
            X=np.zeros((900, 2)),
            Y_cm=np.repeat(np.arange(30.0).reshape(-1, 1), 2, axis=1).repeat(30, axis=0),
            rig=self._profile("m").rig_geometry(),
            sources=[POOL.Source(Path("recordings/round1"), "A", 1, 900, 30, "s1")],
        )
        import gf_fit as FIT  # noqa: PLC0415

        real = (RC.POOL.build, RC.POOL.key_of, FIT.FittedModel)
        RC.POOL.build = lambda *a, **k: pool
        RC.POOL.key_of = lambda *a, **k: ("key",)
        FIT.FittedModel = FakeModel  # type: ignore[misc]
        try:
            RC.fit_pooled(recording, config)
            RC.fit_pooled(recording, config)
        finally:
            (RC.POOL.build, RC.POOL.key_of, FIT.FittedModel) = real  # type: ignore[misc]

        self.assertEqual(len(saved), 2)
        self.assertNotEqual(saved[0], saved[1], "two refits wrote to the same directory")
        self.assertNotIn(existing, saved, "a refit overwrote a model already on disk")
        self.assertTrue((existing / "schema.json").exists(), "the pre-existing model was destroyed")

    # --- 2. the check must be held out of BOTH models ----------------------

    def _model_dir_trained_on(self, pairs: list[list[str]] | None) -> Path:
        directory = Path(tempfile.mkdtemp()) / "model"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory.parent, ignore_errors=True)
        train: dict = {} if pairs is None else {"trained_on": pairs}
        (directory / "schema.json").write_text(json.dumps({"train": train}), encoding="utf-8")
        return directory

    def test_a_check_the_old_model_trained_on_is_refused(self) -> None:
        """Once pooled profiles exist, the ACTIVE model may well have trained
        on the recording being used to judge it. Comparing there flatters the
        old side with memorised rows and the adoption decision means nothing."""

        model = self._model_dir_trained_on([["round38", "T1"], ["round18", "A"]])
        with self.assertRaises(RC.CheckNotHeldOut) as caught:
            RC.check_is_held_out(model, Path("recordings/round38"), "T1")
        self.assertIn("memorised", str(caught.exception))

    def test_a_check_the_old_model_did_not_train_on_is_allowed(self) -> None:
        model = self._model_dir_trained_on([["round18", "A"]])
        RC.check_is_held_out(model, Path("recordings/round38"), "T1")

    def test_a_model_that_does_not_say_is_not_assumed_innocent(self) -> None:
        """ "It does not record what trained it" and "it did not train on
        this" are different facts. Only one is safe, and a model saved before
        the pool existed reports neither."""

        self.assertIsNone(
            POOL_MODULE.trained_on(self._model_dir_trained_on(None)),
            "a model with no recorded provenance claimed to have trained on nothing",
        )

    def test_the_existing_recording_path_checks_before_it_fits(self) -> None:
        """Refusing after the fit would burn the work and, worse, would have
        already written a model."""

        model = self._model_dir_trained_on([["round40", "T1"]])
        reached = []
        real = RC.fit_pooled
        RC.fit_pooled = lambda *a, **k: reached.append(1)  # type: ignore[assignment]
        try:
            with self.assertRaises(RC.CheckNotHeldOut):
                RC.refit_from_existing(self._profile(str(model)), Path("recordings/round40"))
        finally:
            RC.fit_pooled = real
        self.assertFalse(reached, "the pool was fitted before the leak was detected")

    # --- 3. compatibility must follow today's setup ------------------------

    def test_the_pool_is_pinned_to_the_recording_being_calibrated(self) -> None:
        """``build`` adopts the FIRST recording's key when none is required.
        After a screen or rig change, old data would form the pool and the new
        calibration -- the only session describing the new setup -- would be
        rejected as incompatible, while the comparison still took its geometry
        from that new session."""

        seen: dict = {}

        class Stop(Exception):
            pass

        def fake_build(head_names: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            raise Stop

        real = (RC.POOL.build, RC.POOL.key_of)
        RC.POOL.build = fake_build
        RC.POOL.key_of = lambda d, p: ("KEY-OF", Path(d).name, p)
        try:
            with self.assertRaises(Stop):
                RC.fit_pooled(Path("recordings/round40"), "svr_fzscore_lnone_C100_g0.0005")
        finally:
            (RC.POOL.build, RC.POOL.key_of) = real
        self.assertEqual(
            seen.get("require"),
            ("KEY-OF", "round40", "T1"),
            "the pool was not pinned to the recording it is about to be scored on",
        )

    # --- 4. equal feature width is not equal meaning -----------------------

    def test_the_same_width_from_a_different_builder_is_not_compatible(self) -> None:
        """Column 17 is a different quantity, and averaging the two trains on
        two measurements presented as one. Nothing about the shapes says so."""

        base = {
            "feature_dim": 258,
            "head_names": [],
            "builder_version": 3,
            "rig": {"camera_x_cm": 60.0},
            "target_geometry": {"screen_id": "P1"},
        }
        other = dict(base, builder_version=4)
        self.assertNotEqual(
            POOL_MODULE.compatibility_key(base), POOL_MODULE.compatibility_key(other)
        )

    def test_a_different_embedding_model_is_not_compatible(self) -> None:
        base = {
            "feature_dim": 258,
            "head_names": [],
            "builder_version": 3,
            "library": {"name": "gazefollower", "model_sha256": "aaa"},
            "rig": {},
            "target_geometry": {},
        }
        other = dict(base, library={"name": "gazefollower", "model_sha256": "bbb"})
        self.assertNotEqual(
            POOL_MODULE.compatibility_key(base), POOL_MODULE.compatibility_key(other)
        )

    def test_a_machine_local_path_does_not_split_the_pool(self) -> None:
        """Temporary files and paths travel with a machine, not a measurement.
        Keying on them would refuse a person's own past recordings."""

        base = {
            "feature_dim": 258,
            "head_names": [],
            "library": {"name": "gazefollower", "version": "1.0", "tmp_dir": "C:/a"},
            "rig": {},
            "target_geometry": {},
        }
        other = dict(base, library={"name": "gazefollower", "version": "1.0", "tmp_dir": "D:/b"})
        self.assertEqual(POOL_MODULE.compatibility_key(base), POOL_MODULE.compatibility_key(other))

    # --- 5. a check-only round cannot become a profile ---------------------

    def test_a_round_with_no_calibration_is_refused_before_any_work(self) -> None:
        """``adopt`` reads the A protocol of this directory for the rig, the
        screen identity and the calibration pose. Rounds on disk hold a check
        and nothing else; one of those used to fit and compare perfectly well
        and then fail at the last step, after all the work."""

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "round7").mkdir(parents=True)
        (root / "round7" / "T1.meta.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            RC.refit_from_existing(self._profile("m"), root / "round7")
        self.assertIn("no A recording", str(caught.exception))

    def test_a_missing_check_is_refused(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "round7").mkdir(parents=True)
        (root / "round7" / "A.meta.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            RC.refit_from_existing(self._profile("m"), root / "round7")
        self.assertIn("no T1 recording", str(caught.exception))

    # --- and the config name, through the real production function ---------

    def test_a_pooled_model_name_resolves_to_a_real_configuration(self) -> None:
        """Reading the directory name straight back would look up a
        configuration no build has, and the failure would land at the END of a
        calibration session with "a configuration this build does not know"."""

        import gf_presets as PRE  # noqa: PLC0415

        names = {c.name for c in PRE.sweep_with_presets(())}
        real = "svr_fzscore_lnone_C100_g0.0005"
        for stored in (
            f"recordings/round40/models/{real}__pooled_20260910_130000",
            f"recordings/round40/models/{real}__pooled_20260910_130000_1",
            f"recordings/round40/models/{real}__pooled",
            f"recordings/round18/models/{real}",
        ):
            with self.subTest(stored=stored):
                self.assertIn(RC.config_name_of(stored), names, stored)
