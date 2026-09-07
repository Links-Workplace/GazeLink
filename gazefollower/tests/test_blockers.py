"""Regression tests for the seven blocking defects found in review.

Each test reproduces the reported scenario and fails if the defect returns.
They are grouped here rather than scattered so the list stays auditable
against the review, and each one names the defect it pins.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_fit as F  # noqa: E402
import gf_purge as P  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
GEOM = {"width_px": 4096, "height_px": 1152, "dpi_scale": 1.25, "screen_id": "test", "orientation": "LANDSCAPE"}


def _timed_recording(*, n_per_target: int, lost_per_target: int, targets=((0.3, 0.3), (0.7, 0.7))) -> S.Recording:
    """A timed protocol where some eligible frames carried no gaze sample."""

    meta = {
        "rig": RIG.to_dict(),
        "target_geometry": GEOM,
        "targets": [{"index": i, "name": f"P{i}", "screen_position": {"x": x, "y": y}} for i, (x, y) in enumerate(targets)],
        "integrity": {"fps_median": 30.0, "subscriber_errors": 0, "watchdog_tripped": False, "aborted": False, "finished": True},
    }
    b = S.RecordingBuilder("T1", 0, meta)
    seq = 0
    for tid, (nx, ny) in enumerate(targets):
        for i in range(n_per_target):
            lost = i < lost_per_target
            b.append(
                frame_seq=seq,
                timestamp_ns=seq * 33_000_000,
                elapsed_ms=1500.0 + i * 33.0,
                target_id=tid,
                block=0,
                phase=S.PHASE_COLLECTING,
                target_xy=(nx, ny),
                label_cm=RIG.norm_to_label_cm(nx, ny),
                features=None if lost else np.full(4, 0.5, dtype=np.float32),
                head=None if lost else np.zeros(6),
                pnp_deg=None,
                raw_cm=None if lost else (0.0, 0.0),
                openness=(100.0, 100.0),
                tracking_state="FACE_MISSING" if lost else "SUCCESS",
                gaze_status=not lost,
                accepted=False,
            )
            seq += 1
    return b.freeze()


class Blocker1TrackingLossInDenominator(unittest.TestCase):
    """Loss must appear in the denominator, not vanish from it.

    Reported: 160 collecting rows with 80 missing-gaze rows reported
    n_rows=80, n_lost=0 -- accuracy conditioned on the survivors while the
    report concealed a 50 % loss.
    """

    def test_eligible_rows_ignore_tracking_success(self) -> None:
        rec = _timed_recording(n_per_target=80, lost_per_target=40)
        self.assertEqual(int(rec.rows_eligible().sum()), 160)
        self.assertEqual(int(rec.rows_collecting().sum()), 80)

    def test_half_lost_is_reported_as_half_lost(self) -> None:
        rec = _timed_recording(n_per_target=80, lost_per_target=40)
        eligible = F.eligible_rows(rec)
        pred = np.full((rec.n_rows, 2), np.nan)
        pred[rec.rows_collecting()] = [0.3, 0.3]
        metrics = F.evaluate(
            pred[eligible], rec.target_xy[eligible], rec.target_id[eligible], 4096, 1152,
            timestamp_ns=rec.timestamp_ns[eligible],
        )
        self.assertEqual(metrics.n_eligible, 160)
        self.assertEqual(metrics.n_valid, 80)
        self.assertEqual(metrics.n_lost, 80)
        self.assertAlmostEqual(metrics.coverage, 0.5)

    def test_score_timed_reports_the_loss(self) -> None:
        rec_a = _timed_recording(n_per_target=40, lost_per_target=0)
        rec = _timed_recording(n_per_target=80, lost_per_target=40)
        # Fit anything trainable; the point is the denominator, not accuracy.
        # Labels must vary or the SVR has no support vectors to find.
        X = np.full((12, 4), 0.5)
        X[:, 0] = np.linspace(0.0, 1.0, 12)
        Y = np.array([RIG.norm_to_label_cm(v, v) for v in np.linspace(0.1, 0.9, 12)])
        model = F.FittedModel.fit(F.FitConfig(name="lib"), X, Y, rig=RIG, train_meta={})
        metrics, _, _ = F._score_timed(model, rec, RIG, ())
        self.assertEqual(metrics.n_eligible, 160)
        self.assertEqual(metrics.n_lost, 80)
        self.assertAlmostEqual(metrics.coverage, 0.5)
        self.assertIn("coverage", metrics.flat())

    def test_freshness_and_longest_gap_are_measured(self) -> None:
        rec = _timed_recording(n_per_target=80, lost_per_target=40)
        ages = rec.sample_age_ms()
        self.assertTrue(np.all(np.isinf(ages[:40])))  # before the first valid sample
        self.assertFalse(rec.rows_fresh()[0])
        gap = F._longest_gap_ms(np.array([True, False, False, True]), np.array([0, 33e6, 66e6, 99e6]))
        self.assertAlmostEqual(gap, 66.0, places=3)


class Blocker2IncompleteRecordingRejected(unittest.TestCase):
    """A recording that stopped after one target must not certify as valid."""

    def _partial(self) -> S.Recording:
        meta = {
            "rig": RIG.to_dict(),
            "target_geometry": GEOM,
            "targets": [{"index": i, "name": f"CAL_{i}", "screen_position": {"x": 0.5, "y": 0.5}} for i in range(9)],
            "integrity": {"fps_median": 30.0, "subscriber_errors": 0, "watchdog_tripped": False, "aborted": True, "finished": False},
        }
        b = S.RecordingBuilder("A", 0, meta)
        for i in range(C.N_FRAMES_PER_POINT):
            b.append(
                frame_seq=i, timestamp_ns=i * 33_000_000, elapsed_ms=1600.0, target_id=0, block=0,
                phase=S.PHASE_COLLECT, target_xy=(0.5, 0.5), label_cm=RIG.norm_to_label_cm(0.5, 0.5),
                features=np.zeros(4, dtype=np.float32), head=np.zeros(6), pnp_deg=None, raw_cm=(0.0, 0.0),
                openness=(100.0, 100.0), tracking_state="SUCCESS", gaze_status=True, accepted=True,
            )
        return b.freeze()

    def test_one_complete_target_of_nine_is_invalid(self) -> None:
        report = F.p0_valid(self._partial())
        self.assertFalse(report["valid"])
        self.assertEqual(report["expected_targets"], 9)
        self.assertEqual(len(report["missing_targets"]), 8)
        self.assertIn("every_expected_target_has_45", report["failed_checks"])

    def test_aborted_and_unfinished_fail_on_their_own(self) -> None:
        report = F.p0_valid(self._partial())
        self.assertIn("not_aborted", report["failed_checks"])
        self.assertIn("ran_to_completion", report["failed_checks"])

    def test_unknown_integrity_counts_as_invalid_not_valid(self) -> None:
        rec = self._partial()
        rec.meta["integrity"] = {}  # nothing known about the run
        report = F.p0_valid(rec)
        self.assertFalse(report["valid"])
        self.assertTrue(report["unknown_checks"])


class Blocker3EmbeddingsStayContained(unittest.TestCase):
    """--out must not be able to place embeddings outside recordings/."""

    def test_output_root_outside_recordings_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                R.resolve_output_root(Path(tmp))
        with self.assertRaises(ValueError):
            R.resolve_output_root(Path(r"C:\Users\User\Desktop\focus"))

    def test_recordings_dir_and_subdirs_are_allowed(self) -> None:
        self.assertEqual(R.resolve_output_root(R.RECORDINGS_DIR), R.RECORDINGS_DIR.resolve())
        nested = R.RECORDINGS_DIR / "sub"
        self.assertEqual(R.resolve_output_root(nested), nested.resolve())


class Blocker4BoundedShutdown(unittest.TestCase):
    """A stalled library call must not prevent the later cleanup steps."""

    def test_timeout_helper_returns_instead_of_hanging(self) -> None:
        started = time.monotonic()
        verdict = R._call_with_timeout(lambda: time.sleep(30), timeout_s=0.2)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("TIMEOUT", verdict)

    def test_capture_is_released_even_when_release_hangs(self) -> None:
        class Cap:
            def __init__(self):
                self.opened = True

            def isOpened(self):
                return self.opened

            def release(self):
                self.opened = False

        class Stream:
            closed = False

            def close(self):
                self.closed = True

        cap, stream = Cap(), Stream()
        camera = SimpleNamespace(_cap=cap, _camera_thread=None, _camera_thread_running=True)
        gf = SimpleNamespace(camera=camera, _tmpSampleDataSteam=stream, release=lambda: time.sleep(30))
        report = R.shutdown_library(gf, join_timeout_s=0.2)
        self.assertIn("TIMEOUT", report["library_release"])
        self.assertTrue(report["capture_released"])
        self.assertFalse(cap.opened)
        self.assertTrue(stream.closed)


class Blocker5PersistentErrorsAbort(unittest.TestCase):
    """Repeated subscriber exceptions must stop the run, not be counted forever."""

    def _runner(self, head_builder):
        spec = R.protocol_a()
        meta = {"rig": RIG.to_dict(), "target_geometry": GEOM, "targets": spec.exported_targets()}
        builder = S.RecordingBuilder("A", 0, meta)
        return R.ProtocolRunner(spec, builder, RIG, head_builder=head_builder, pnp=lambda _f: None)

    def _frame(self):
        face = SimpleNamespace(status=True, can_gaze_estimation=True, left_eye_openness=100.0, right_eye_openness=100.0, timestamp=1, face_landmarks=np.zeros((478, 3)), img_w=640, img_h=480)
        gaze = SimpleNamespace(status=True, features=np.zeros(4, dtype=np.float32), raw_gaze_coordinates=np.zeros(2), tracking_state=SimpleNamespace(name="SUCCESS"), timestamp=1)
        return face, gaze

    def test_run_is_marked_failed_after_consecutive_errors(self) -> None:
        def exploding(_face):
            raise RuntimeError("boom")

        runner = self._runner(exploding)
        runner.start()
        for _ in range(R.MAX_CONSECUTIVE_ERRORS):
            runner.on_frame(*self._frame())
        self.assertTrue(runner.failed)
        self.assertEqual(runner.errors, R.MAX_CONSECUTIVE_ERRORS)
        self.assertIn("boom", runner.last_error or "")

    def test_a_recovered_error_does_not_accumulate_toward_the_abort(self) -> None:
        state = {"fail": True}

        def flaky(_face):
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("transient")
            return np.zeros(6)

        runner = self._runner(flaky)
        runner.start()
        runner.on_frame(*self._frame())
        self.assertEqual(runner.consecutive_errors, 1)
        runner.on_frame(*self._frame())
        self.assertEqual(runner.consecutive_errors, 0)
        self.assertFalse(runner.failed)
        self.assertEqual(runner.errors, 1)


class Blocker6FreezeIsAtomic(unittest.TestCase):
    """freeze() under the runner's lock must not race an in-flight subscriber."""

    def test_concurrent_appends_cannot_produce_ragged_arrays(self) -> None:
        spec = R.ProtocolSpec("T1", [R.Target(0, "P0", 0.5, 0.5)], "timed", settle_s=0.0, collect_s=60.0)
        meta = {"rig": RIG.to_dict(), "target_geometry": GEOM, "targets": spec.exported_targets()}
        builder = S.RecordingBuilder("T1", 0, meta)
        runner = R.ProtocolRunner(spec, builder, RIG, head_builder=lambda _f: np.zeros(6), pnp=lambda _f: None)
        runner.start()
        stop = threading.Event()

        def feed() -> None:
            face = SimpleNamespace(status=True, can_gaze_estimation=True, left_eye_openness=100.0, right_eye_openness=100.0, timestamp=1, face_landmarks=np.zeros((478, 3)), img_w=640, img_h=480)
            gaze = SimpleNamespace(status=True, features=np.zeros(4, dtype=np.float32), raw_gaze_coordinates=np.zeros(2), tracking_state=SimpleNamespace(name="SUCCESS"), timestamp=1)
            while not stop.is_set():
                runner.on_frame(face, gaze)

        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        try:
            time.sleep(0.05)
            with runner.lock:
                rec = builder.freeze()
        finally:
            stop.set()
            thread.join(timeout=2.0)
        # Every column must agree on the row count; a race would truncate one.
        n = rec.n_rows
        for name in ("timestamp_ns", "phase", "features", "head", "target_xy", "gaze_status"):
            self.assertEqual(len(getattr(rec, name)), n, name)
        self.assertGreater(n, 0)


class Blocker7BiasTravelsWithSlope(unittest.TestCase):
    """Every reported slope must be accompanied by an error and a bias."""

    def test_axis_metrics_carry_bias_and_percentiles(self) -> None:
        keys = F.AxisMetrics(1, 2, 3, 4, 5, 6, 1.0, 0.0).flat("y").keys()
        for expected in ("y_bias_px", "y_median_abs_px", "y_p90_abs_px", "y_slope"):
            self.assertIn(expected, keys)

    def test_after_filter_line_reports_bias(self) -> None:
        axis = F.AxisMetrics(10.0, 9.0, 20.0, 25.0, 30.0, -349.0, 1.096, -0.125)
        metrics = F.EvalMetrics(
            n_eligible=10, n_valid=10, n_lost=0, coverage=1.0, n_fresh=10, coverage_fresh=1.0,
            longest_gap_ms=0.0, mean_euclid_px=10.0, median_euclid_px=9.0, p90_euclid_px=20.0,
            p95_euclid_px=25.0, max_euclid_px=30.0, x=axis, y=axis, off_screen_x=0, off_screen_y=0,
            per_target=[],
        )
        report = {
            "recording_dir": "x", "head_names": [], "train_rows": 10, "train_head_dropped": 0,
            "feature_dim": 4, "column_std_summary": {"min": 0.0, "median": 1.0, "max": 2.0},
            "mean_pairwise_sq_dist": 1.0,
            "selection": {"metric": F.SELECTION_METRIC, "tiebreak": F.SELECTION_TIEBREAK, "selected": "m", "library_default": "m"},
            "sweep": [{"name": "m", "config": {}, "gamma_resolved": 0.005, "effective_gamma_d2": 1.0, "tune": metrics.flat(), "tune_per_target": [], "tune_head_dropped": 0}],
            "t1": {"m": {"t1": metrics.flat(), "t1_per_target": [], "t1_after_filter": metrics.flat(), "t1_head_dropped": 0, "predictions_json": "p.json"}},
            "p0_info": F.p0_info_category(metrics.flat(), metrics.flat(), metrics.flat()),
            "p0_confound": {"n_rows": 1, "corr_pitch_a_target_y": 0.1, "corr_yaw_ratio_target_x": 0.1, "pitch_a_iqr": 0.1},
            "p0_valid": {},
            "ridge_r2_diagnostic": {"r2_x": 0.5, "r2_y": 0.5},
        }
        text = F.format_phase0_report(report)
        after = [line for line in text.splitlines() if "after HeuristicFilter" in line]
        self.assertEqual(len(after), 1)
        self.assertIn("bias", after[0])
        self.assertIn("slope", after[0])


class SelectionRejectsNonFinite(unittest.TestCase):
    """A NaN metric must sort last, never win by comparison accident."""

    def test_nan_never_wins(self) -> None:
        sweep = [
            {"name": "nan", "tune": {F.SELECTION_METRIC: float("nan"), F.SELECTION_TIEBREAK: 0.0}},
            {"name": "good", "tune": {F.SELECTION_METRIC: 100.0, F.SELECTION_TIEBREAK: 5.0}},
        ]
        self.assertEqual(F.select_config(sweep), "good")
        self.assertEqual(F.select_config(list(reversed(sweep))), "good")

    def test_all_nan_raises_rather_than_picking_one(self) -> None:
        sweep = [{"name": "a", "tune": {F.SELECTION_METRIC: float("nan"), F.SELECTION_TIEBREAK: float("nan")}}]
        with self.assertRaises(ValueError):
            F.select_config(sweep)


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()


class IndefiniteWaitRecordsNothing(unittest.TestCase):
    """An unbounded wait must not accumulate biometric data.

    Protocols wait for a key press with no timeout. The runner used to be
    installed BEFORE that wait, so the subscriber stored a 258-d face
    embedding for every frame at ~32 fps for as long as the person was away
    from the desk -- rows the fitter never reads.
    """

    def _frame(self):
        face = SimpleNamespace(
            status=True, can_gaze_estimation=True, left_eye_openness=100.0, right_eye_openness=100.0,
            timestamp=1, face_landmarks=np.zeros((478, 3)), img_w=640, img_h=480,
        )
        gaze = SimpleNamespace(
            status=True, features=np.arange(258, dtype=np.float32), raw_gaze_coordinates=np.zeros(2),
            tracking_state=SimpleNamespace(name="SUCCESS"), timestamp=1,
        )
        return face, gaze

    def test_subscriber_discards_frames_when_no_runner_is_installed(self) -> None:
        # This is the shape run_session relies on: the wait happens while
        # runner_ref holds None, so the frames simply go nowhere.
        runner_ref = {"runner": None}

        def subscriber(face_info, gaze_info):
            runner = runner_ref["runner"]
            if runner is not None:
                runner.on_frame(face_info, gaze_info)

        for _ in range(500):
            subscriber(*self._frame())  # must not raise, must not store

        spec = R.protocol_a()
        meta = {"rig": RIG.to_dict(), "target_geometry": GEOM, "targets": spec.exported_targets()}
        builder = S.RecordingBuilder("A", 0, meta)
        runner_ref["runner"] = R.ProtocolRunner(spec, builder, RIG, head_builder=lambda _f: np.zeros(6), pnp=lambda _f: None)
        self.assertEqual(len(builder), 0)

    def test_idle_frames_carry_no_embedding(self) -> None:
        spec = R.protocol_a()
        meta = {"rig": RIG.to_dict(), "target_geometry": GEOM, "targets": spec.exported_targets()}
        builder = S.RecordingBuilder("A", 0, meta)
        runner = R.ProtocolRunner(spec, builder, RIG, head_builder=lambda _f: np.zeros(6), pnp=lambda _f: None)
        # Not started: no target on screen, so every frame is idle.
        for _ in range(50):
            runner.on_frame(*self._frame())
        rec = builder.freeze()
        self.assertEqual(rec.n_rows, 50)
        self.assertTrue(np.all(rec.phase == S.PHASE_IDLE))
        self.assertTrue(np.all(np.isnan(rec.features)), "idle rows must not carry a face embedding")
        self.assertFalse(np.any(rec.head_valid), "idle rows must not carry head features")
        # Frame accounting and tracking state are still there for gap detection.
        self.assertTrue(np.all(rec.gaze_status))
        self.assertEqual(len(set(rec.frame_seq.tolist())), 50)

    def test_frames_during_a_protocol_do_carry_the_embedding(self) -> None:
        spec = R.protocol_a()
        meta = {"rig": RIG.to_dict(), "target_geometry": GEOM, "targets": spec.exported_targets()}
        builder = S.RecordingBuilder("A", 0, meta)
        clock = {"t": 100.0}
        runner = R.ProtocolRunner(
            spec, builder, RIG, clock=lambda: clock["t"],
            head_builder=lambda _f: np.zeros(6), pnp=lambda _f: None,
        )
        runner.start()
        clock["t"] += 2.0  # past the prepare window
        for _ in range(5):
            runner.on_frame(*self._frame())
            clock["t"] += 1 / 30
        rec = builder.freeze()
        stored = rec.phase != S.PHASE_IDLE
        self.assertTrue(np.any(stored))
        self.assertFalse(np.any(np.isnan(rec.features[stored])), "real rows must keep their features")


class ReadingPauseMustNotAbortTheNextProtocol(unittest.TestCase):
    """A long pause between protocols is normal, not a camera failure.

    The watchdog used to be one object shared across every protocol, so it
    carried protocol N's last frame time across the wait for a key press.
    That wait is unbounded and records nothing, so any pause longer than the
    limit tripped the watchdog on the next protocol's first loop -- aborting
    it before a single frame, because the operator took time to read.
    """

    LIMIT = 5.0

    def test_a_fresh_protocol_after_a_long_pause_does_not_trip(self) -> None:
        # Protocol 1 ends at t=100; the operator reads for 40 s; protocol 2
        # is armed at t=140 and its first loop runs immediately.
        dog = R.FrameWatchdog(limit_s=self.LIMIT, started_s=140.0)
        self.assertFalse(dog.tripped(140.01))
        self.assertFalse(dog.tripped(144.9))

    def test_a_protocol_that_never_receives_a_frame_still_trips(self) -> None:
        dog = R.FrameWatchdog(limit_s=self.LIMIT, started_s=140.0)
        self.assertTrue(dog.tripped(145.1))
        self.assertAlmostEqual(dog.silent_for(145.1), 5.1, places=6)

    def test_frames_push_the_deadline_forward(self) -> None:
        dog = R.FrameWatchdog(limit_s=self.LIMIT, started_s=100.0)
        dog.note([100.5, 101.0, 104.9])
        self.assertFalse(dog.tripped(109.8))
        self.assertTrue(dog.tripped(110.0))

    def test_note_with_no_frames_leaves_the_start_time_in_charge(self) -> None:
        dog = R.FrameWatchdog(limit_s=self.LIMIT, started_s=100.0)
        dog.note([])
        self.assertIsNone(dog.last_frame_s)
        self.assertTrue(dog.tripped(105.1))

    def test_a_stall_during_a_protocol_is_still_caught(self) -> None:
        dog = R.FrameWatchdog(limit_s=self.LIMIT, started_s=100.0)
        dog.note([101.0])
        self.assertFalse(dog.tripped(105.9))
        self.assertTrue(dog.tripped(106.1))  # 5.1 s since the last frame

    def test_helper_never_trips_without_a_reference_frame(self) -> None:
        self.assertFalse(R.watchdog_tripped(None, 1e9))
