"""gf_record's protocol logic with a fake clock and duck-typed frames.
No camera, no pygame, no gazefollower import."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_gaze_filter as F  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
META = {"rig": RIG.to_dict(), "target_geometry": {"width_px": 4096, "height_px": 1152}, "targets": []}


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 1 / 30) -> float:
        self.t += dt
        return self.t


def frame(status: bool = True, left: float = 100.0, right: float = 100.0, dim: int = 8, ts: int = 1):
    face = SimpleNamespace(status=status, can_gaze_estimation=status, left_eye_openness=left, right_eye_openness=right, timestamp=ts, face_landmarks=np.zeros((478, 3)), img_w=640, img_h=480)
    gaze = SimpleNamespace(status=status, features=np.arange(dim, dtype=np.float32) if status else None, raw_gaze_coordinates=np.array([0.1, 0.2]) if status else None, tracking_state=SimpleNamespace(name="SUCCESS" if status else "FACE_MISSING"), timestamp=ts)
    return face, gaze


def head_ok(_face):
    return np.array([0.0, 0.0, -0.6, 0.5, 0.4, 0.3])


def head_none(_face):
    return None


def pnp_none(_face):
    return None


class CollectionGateTests(unittest.TestCase):
    def test_prepare_then_exactly_45_accepted_then_wait_then_done(self) -> None:
        clock = FakeClock()
        gate = R.CollectionGate(onset_s=clock.t)
        # During prepare nothing counts.
        for _ in range(10):
            phase, acc = gate.observe(clock.tick(0.1), True, 100, 100)
            self.assertEqual(phase, S.PHASE_PREPARE)
            self.assertFalse(acc)
        self.assertEqual(gate.n_accepted, 0)
        clock.tick(0.6)  # past 1.5 s
        accepted = 0
        for _ in range(44):
            phase, acc = gate.observe(clock.tick(), True, 100, 100)
            self.assertEqual(phase, S.PHASE_COLLECT)
            accepted += acc
        self.assertEqual(accepted, 44)
        self.assertFalse(gate.done)
        phase, acc = gate.observe(clock.tick(), True, 100, 100)
        self.assertTrue(acc)
        self.assertEqual(gate.n_accepted, 45)
        # 46th frame: not stored, wait phase, not done until 0.5 s elapsed.
        phase, acc = gate.observe(clock.tick(), True, 100, 100)
        self.assertEqual(phase, S.PHASE_WAIT)
        self.assertFalse(acc)
        self.assertFalse(gate.done)
        self.assertEqual(gate.n_accepted, 45)
        clock.tick(0.5)
        phase, acc = gate.observe(clock.tick(), True, 100, 100)
        self.assertTrue(gate.done)
        self.assertFalse(acc)

    def test_rejected_frames_are_not_counted_and_threshold_is_strict(self) -> None:
        clock = FakeClock()
        gate = R.CollectionGate(onset_s=clock.t)
        clock.tick(1.6)
        _, acc = gate.observe(clock.tick(), False, 100, 100)  # no gaze
        self.assertFalse(acc)
        _, acc = gate.observe(clock.tick(), True, 10.0, 100)  # equal to threshold: rejected ('>' not '>=')
        self.assertFalse(acc)
        _, acc = gate.observe(clock.tick(), True, 100, 10.0)
        self.assertFalse(acc)
        self.assertEqual(gate.n_accepted, 0)
        _, acc = gate.observe(clock.tick(), True, 10.01, 10.01)
        self.assertTrue(acc)
        self.assertEqual(gate.n_accepted, 1)

    def test_warmup_counts_but_never_stores(self) -> None:
        clock = FakeClock()
        gate = R.CollectionGate(onset_s=clock.t, stored=False)
        clock.tick(1.6)
        for _ in range(45):
            phase, acc = gate.observe(clock.tick(), True, 100, 100)
            self.assertEqual(phase, S.PHASE_WARMUP)
            self.assertFalse(acc)
        self.assertEqual(gate.n_accepted, 45)
        clock.tick(0.6)
        gate.observe(clock.tick(), True, 100, 100)
        self.assertTrue(gate.done)


class TimedGateTests(unittest.TestCase):
    def test_phases_and_completion(self) -> None:
        gate = R.TimedGate(onset_s=0.0, settle_s=1.5, collect_s=1.5)
        self.assertEqual(gate.observe(0.1), S.PHASE_STABILIZING)
        self.assertEqual(gate.observe(1.49), S.PHASE_STABILIZING)
        self.assertEqual(gate.observe(1.5), S.PHASE_COLLECTING)
        self.assertFalse(gate.done)
        self.assertEqual(gate.observe(2.99), S.PHASE_COLLECTING)
        self.assertFalse(gate.done)
        gate.observe(3.0)
        self.assertTrue(gate.done)


class ProtocolRunnerTests(unittest.TestCase):
    def _run_calibration(self, clock: FakeClock, runner: R.ProtocolRunner, frames_status=None) -> None:
        i = 0
        while not runner.finished and i < 5000:
            status = True if frames_status is None else frames_status(i)
            runner.on_frame(*frame(status=status, ts=i))
            clock.tick()
            i += 1

    def test_protocol_a_yields_45_rows_per_stored_point_and_none_for_warmup(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        self._run_calibration(clock, runner)
        rec = builder.freeze()
        self.assertTrue(runner.finished)
        self.assertEqual(runner.errors, 0)
        acc = rec.rows_accepted()
        self.assertEqual(int(acc.sum()), 9 * C.N_FRAMES_PER_POINT)
        for tid in range(9):
            self.assertEqual(int(np.sum(acc & (rec.target_id == tid))), 45, tid)
        self.assertFalse(np.any(acc & (rec.target_id == -1)))
        self.assertTrue(np.any(rec.phase == S.PHASE_WARMUP))
        # Labels are the library's cm labels for the target on screen.
        row = np.flatnonzero(acc & (rec.target_id == 0))[0]
        self.assertTrue(np.allclose(rec.label_cm[row], RIG.norm_to_label_cm(*C.NINE_POINT_STORED[0])))
        self.assertEqual(rec.feature_dim, 8)
        self.assertTrue(np.all(rec.head_valid[acc]))

    def test_face_loss_delays_but_never_corrupts_the_count(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        self._run_calibration(clock, runner, frames_status=lambda i: i % 3 != 0)  # every third frame lost
        rec = builder.freeze()
        acc = rec.rows_accepted()
        self.assertEqual(int(acc.sum()), 9 * 45)
        lost = ~np.asarray(rec.gaze_status, dtype=bool)
        self.assertTrue(np.any(lost))
        self.assertFalse(np.any(acc & lost))
        self.assertTrue(np.all(np.isnan(rec.features[lost])))
        self.assertTrue(np.all(rec.tracking_state[lost] == "FACE_MISSING"))

    def test_invalid_head_is_recorded_as_invalid_not_zero_and_row_still_accepted(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_none, pnp=pnp_none)
        runner.start()
        self._run_calibration(clock, runner)
        rec = builder.freeze()
        self.assertEqual(int(rec.rows_accepted().sum()), 9 * 45)  # arm A is not gated on head
        self.assertFalse(np.any(rec.head_valid))
        self.assertTrue(np.all(np.isnan(rec.head)))

    def test_subscriber_exception_is_counted_not_raised(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))

        def exploding(_face):
            raise RuntimeError("boom")

        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=exploding, pnp=pnp_none)
        runner.start()
        runner.on_frame(*frame())  # must not raise
        self.assertEqual(runner.errors, 1)
        self.assertIn("boom", runner.last_error or "")

    def test_row_target_matches_the_gate_that_judged_it(self) -> None:
        # The frame that completes a target's wait is labelled WAIT for THAT
        # target; the next frame belongs to the next target's PREPARE.
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        self._run_calibration(clock, runner)
        rec = builder.freeze()
        changes = np.flatnonzero(np.diff(rec.target_id) != 0)
        for k in changes:
            self.assertIn(rec.phase[k], (S.PHASE_WAIT, S.PHASE_WARMUP))
            self.assertEqual(rec.phase[k + 1], S.PHASE_PREPARE)
            self.assertLess(rec.elapsed_ms[k + 1], 50.0)

    def test_timed_protocol_phases_and_advance(self) -> None:
        clock = FakeClock()
        spec = R.ProtocolSpec("T1", [R.Target(0, "A", 0.3, 0.3), R.Target(1, "B", 0.7, 0.7)], "timed", settle_s=1.5, collect_s=1.5)
        builder = S.RecordingBuilder("T1", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        i = 0
        while not runner.finished and i < 1000:
            runner.on_frame(*frame(ts=i))
            clock.tick()
            i += 1
        rec = builder.freeze()
        col = rec.rows_collecting()
        self.assertGreaterEqual(int(np.sum(col & (rec.target_id == 0))), 40)
        self.assertGreaterEqual(int(np.sum(col & (rec.target_id == 1))), 40)
        self.assertFalse(np.any(rec.accepted))  # 'accepted' is the calibration gate only
        self.assertTrue(np.any(rec.phase == S.PHASE_STABILIZING))

    def test_speed_shortens_the_waits_but_never_the_accepted_counts(self) -> None:
        # The 45-frame quota is a frame count, so it is identical at any speed;
        # only the prepare/wait seconds shrink, which is the whole point of
        # --speed for dry runs. Compared against speed 1, not a magic number.
        rows = {}
        for speed in (1.0, 10.0):
            clock = FakeClock()
            spec = R.protocol_a()
            builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
            runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, speed=speed, head_builder=head_ok, pnp=pnp_none)
            runner.start()
            self._run_calibration(clock, runner)
            rec = builder.freeze()
            self.assertEqual(int(rec.rows_accepted().sum()), 9 * 45, speed)
            rows[speed] = rec.n_rows
        self.assertLess(rows[10.0], rows[1.0])
        # Overhead beyond the 450 counted frames (9 stored + 1 warm-up) is at
        # least ten times smaller once the 1.5 s/0.5 s waits are scaled down.
        self.assertLess(rows[10.0] - 450, (rows[1.0] - 450) / 10)

    def test_integrity_and_fps(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        self._run_calibration(clock, runner)
        integ = runner.integrity(watchdog_tripped=False, aborted=False)
        self.assertAlmostEqual(integ["fps_median"], 30.0, delta=0.5)
        self.assertEqual(integ["subscriber_errors"], 0)
        self.assertTrue(integ["finished"])
        self.assertEqual(integ["feature_dim"], 8)


class ShutdownAdapterTests(unittest.TestCase):
    def test_releases_open_capture_closes_stream_and_survives_library_errors(self) -> None:
        class Cap:
            def __init__(self):
                self.opened = True

            def isOpened(self):
                return self.opened

            def release(self):
                self.opened = False

        class Thread:
            def __init__(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                self.alive = False

        class Stream:
            closed = False

            def close(self):
                self.closed = True

        camera = SimpleNamespace(_cap=Cap(), _camera_thread=Thread(), _camera_thread_running=True)
        gf = SimpleNamespace(camera=camera, _tmpSampleDataSteam=Stream(), release=lambda: (_ for _ in ()).throw(AttributeError("join on None")))
        report = R.shutdown_library(gf)
        self.assertIn("AttributeError", report["library_release"])
        self.assertTrue(report["thread_joined"])
        self.assertTrue(report["capture_released"])
        self.assertFalse(camera._cap.opened)
        self.assertTrue(report["tmp_stream_closed"])
        self.assertFalse(camera._camera_thread_running)

    def test_never_opened_camera_has_no_thread_and_no_capture_to_release(self) -> None:
        class Cap:
            def isOpened(self):
                return False

            def release(self):
                raise AssertionError("must not release a closed capture")

        camera = SimpleNamespace(_cap=Cap(), _camera_thread=None, _camera_thread_running=None)
        gf = SimpleNamespace(camera=camera, _tmpSampleDataSteam=None, release=lambda: None)
        report = R.shutdown_library(gf)
        self.assertEqual(report["library_release"], "ok")
        self.assertIsNone(report["thread_joined"])
        self.assertFalse(report["capture_released"])
        self.assertFalse(report["tmp_stream_closed"])


class WatchdogTests(unittest.TestCase):
    def test_trips_only_after_limit(self) -> None:
        self.assertFalse(R.watchdog_tripped(None, 100.0))
        self.assertFalse(R.watchdog_tripped(100.0, 104.9))
        self.assertTrue(R.watchdog_tripped(100.0, 105.1))


class ProtocolSpecTests(unittest.TestCase):
    def test_protocol_a_matches_library_sequence(self) -> None:
        spec = R.protocol_a()
        self.assertEqual(len(spec.targets), 10)
        self.assertEqual(spec.targets[0].index, -1)
        self.assertEqual((spec.targets[0].x, spec.targets[0].y), (0.5, 0.5))
        self.assertEqual([(t.x, t.y) for t in spec.targets[1:]], list(C.NINE_POINT_STORED))
        self.assertEqual(len(spec.exported_targets()), 9)

    def test_t2_spec(self) -> None:
        spec = R.protocol_t2()
        self.assertEqual(len(spec.targets), 3)
        self.assertAlmostEqual(spec.settle_s + spec.collect_s, R.T2_SECONDS)

    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()


class CameraVerdictTests(unittest.TestCase):
    """A probe that did not measure a rate must STOP recording, not fall through.

    The earlier guard was `fps is not None and fps < 25`, so a probe that
    timed out -- which reports no rate at all -- skipped the test entirely and
    recording started with the leaked probe thread still holding the camera.
    """

    def test_timeout_is_fatal_and_not_a_prompt(self) -> None:
        verdict = R.camera_verdict({"timed_out": True, "opened": None})
        self.assertTrue(verdict.fatal)
        self.assertFalse(verdict.needs_confirmation)
        self.assertIn("stuck", verdict.reason)

    def test_failure_to_open_is_fatal(self) -> None:
        verdict = R.camera_verdict({"opened": False, "error": "device busy"})
        self.assertTrue(verdict.fatal)
        self.assertIn("device busy", verdict.message)

    def test_opened_but_no_frames_is_fatal(self) -> None:
        verdict = R.camera_verdict({"opened": True, "frame_width": 640})
        self.assertTrue(verdict.fatal)
        self.assertIn("no usable frames", verdict.message)

    def test_slow_camera_asks_rather_than_refusing_outright(self) -> None:
        verdict = R.camera_verdict(
            {"opened": True, "measured_fps": 1.0, "frame_width": 640, "frame_height": 480, "backend": "DSHOW"}
        )
        self.assertFalse(verdict.fatal)
        self.assertTrue(verdict.needs_confirmation)
        self.assertIn("1.0 fps", verdict.message)

    def test_healthy_camera_passes_without_a_question(self) -> None:
        verdict = R.camera_verdict(
            {"opened": True, "measured_fps": 32.3, "frame_width": 640, "frame_height": 480, "backend": "MSMF"}
        )
        self.assertFalse(verdict.fatal)
        self.assertFalse(verdict.needs_confirmation)
        self.assertIn("32.3 fps", verdict.message)

    def test_threshold_boundary(self) -> None:
        base = {"opened": True, "frame_width": 640, "frame_height": 480, "backend": "X"}
        self.assertTrue(R.camera_verdict(dict(base, measured_fps=24.9)).needs_confirmation)
        self.assertFalse(R.camera_verdict(dict(base, measured_fps=25.0)).needs_confirmation)

    def test_no_terminal_means_no_consent(self) -> None:
        class NotATty:
            def isatty(self):
                return False

        original = sys.stdin
        sys.stdin = NotATty()
        try:
            self.assertFalse(R.confirm_low_frame_rate())
        finally:
            sys.stdin = original

    def test_eof_at_the_prompt_means_no(self) -> None:
        class Tty:
            def isatty(self):
                return True

        import builtins

        original_stdin, original_input = sys.stdin, builtins.input
        sys.stdin = Tty()

        def boom(_prompt=""):
            raise EOFError

        builtins.input = boom
        try:
            self.assertFalse(R.confirm_low_frame_rate())
        finally:
            sys.stdin, builtins.input = original_stdin, original_input


class WaitForKeyTests(unittest.TestCase):
    """A protocol must not start until the operator says so.

    A three-second countdown used to do this, which was not long enough to
    read the instruction -- the protocol began before the person knew what it
    asked of them.
    """

    class FakeDisplay(R.Display):
        """Display with a scripted key queue instead of a window."""

        def __init__(self, events):
            self.headless = False
            self.width, self.height = 800, 600
            self.drawn = []
            self._events = list(events)
            self.pg = self._FakePygame()
            self.pg._pending = self._events

        class _FakePygame:
            KEYDOWN, QUIT = 2, 12
            K_ESCAPE, K_SPACE, K_RETURN, K_KP_ENTER = 27, 32, 13, 271

            def __init__(self):
                self._queue = []
                self._pending = []
                self.cleared = 0

            def event_get(self):
                # The scripted keys stand for presses made AFTER the prompt
                # appeared, so they survive the initial drain -- otherwise the
                # fake would clear them and the wait could never end.
                out, self._queue = self._queue, []
                return out

            @property
            def event(self):
                outer = self

                class _E:
                    @staticmethod
                    def get():
                        return outer.event_get()

                    @staticmethod
                    def clear():
                        outer.cleared += 1
                        # Drop what was queued before the prompt; hand the
                        # scripted presses over as if made after it.
                        outer._queue = list(outer._pending)
                        outer._pending = []

                return _E()

        def draw_message(self, lines):
            self.drawn.append(list(lines))

    def _event(self, key=None, kind=None):
        return SimpleNamespace(type=kind if kind is not None else self.FakeDisplay._FakePygame.KEYDOWN, key=key)

    def test_space_starts_the_protocol(self) -> None:
        d = self.FakeDisplay([self._event(key=32)])
        self.assertEqual(d.wait_for_key(["read me"]), "go")

    def test_enter_starts_the_protocol(self) -> None:
        for key in (13, 271):
            d = self.FakeDisplay([self._event(key=key)])
            self.assertEqual(d.wait_for_key(["read me"]), "go")

    def test_escape_aborts(self) -> None:
        d = self.FakeDisplay([self._event(key=27)])
        self.assertEqual(d.wait_for_key(["read me"]), "abort")

    def test_other_keys_do_not_start_it(self) -> None:
        # 'A' then space: only the space counts.
        d = self.FakeDisplay([self._event(key=97), self._event(key=32)])
        self.assertEqual(d.wait_for_key(["read me"]), "go")

    def test_it_keeps_drawing_while_it_waits(self) -> None:
        d = self.FakeDisplay([self._event(key=32)])
        d.wait_for_key(["line one", "line two"])
        self.assertTrue(d.drawn)
        rendered = "\n".join(d.drawn[-1])
        self.assertIn("line one", rendered)
        self.assertIn("SPACE", rendered)
        self.assertIn("Esc", rendered)

    def test_stale_keys_are_dropped_so_one_press_cannot_skip_two_screens(self) -> None:
        d = self.FakeDisplay([self._event(key=32)])
        d.pg._queue = [self._event(key=32), self._event(key=32)]  # left over from before
        d.wait_for_key(["read me"])
        self.assertEqual(d.pg.cleared, 1)
        self.assertEqual(d.pg._queue, [])  # the stale ones never counted

    def test_timeout_is_reported_when_given(self) -> None:
        d = self.FakeDisplay([])
        self.assertEqual(d.wait_for_key(["read me"], timeout_s=0.05), "timeout")

    def test_headless_proceeds_without_a_key(self) -> None:
        d = R.Display(800, 600, headless=True)
        self.assertEqual(d.wait_for_key(["read me"]), "go")


class DurationEstimateTests(unittest.TestCase):
    def test_calibration_estimate_counts_the_warm_up_point(self) -> None:
        spec = R.protocol_a()
        per_point = C.PREPARE_S + C.N_FRAMES_PER_POINT / 30.0 + C.WAIT_S
        self.assertAlmostEqual(R._estimate_seconds(spec), 10 * per_point, places=6)

    def test_timed_estimate_is_targets_times_the_window(self) -> None:
        spec = R.ProtocolSpec("T1", [R.Target(0, "A", 0.3, 0.3), R.Target(1, "B", 0.7, 0.7)], "timed", settle_s=1.5, collect_s=1.5)
        self.assertAlmostEqual(R._estimate_seconds(spec), 6.0)


class OverlayFilterCliTests(unittest.TestCase):
    def test_default_matches_the_configuration_selected_on_tune(self) -> None:
        args = R.build_parser().parse_args(
            [
                "--round",
                "7",
                "--camera-x-cm",
                "60",
                "--camera-y-cm",
                "63.6",
                "--screen-width-cm",
                "120",
                "--screen-height-cm",
                "33.75",
            ]
        )
        self.assertEqual(args.overlay_filter, F.FilterKind.ONE_EURO.value)
        self.assertEqual(args.overlay_one_euro_min_cutoff_hz, 1.2)
        self.assertEqual(args.overlay_one_euro_beta, 0.0005)


class RoundOverwriteGuardTests(unittest.TestCase):
    """Re-running with a used round number replaced round9's good 903-frame
    capture with an aborted 109-frame one. A recording cannot be re-made."""

    def test_an_existing_round_with_data_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round9"
            round_dir.mkdir()
            (round_dir / "T1.npz").write_bytes(b"recorded")
            with self.assertRaises(SystemExit) as caught:
                R.guard_existing_round(round_dir)
            self.assertIn("T1.npz", str(caught.exception))

    def test_an_explicit_overwrite_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round9"
            round_dir.mkdir()
            (round_dir / "T1.npz").write_bytes(b"recorded")
            R.guard_existing_round(round_dir, allow_overwrite=True)

    def test_a_stub_from_an_aborted_no_face_attempt_does_not_block_a_retry(self) -> None:
        """The no-face guard aborts and still writes the protocol file. That
        stub holds nothing, and must not lock the operator out of retrying
        the very round number they just failed on."""

        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round14"
            round_dir.mkdir()
            (round_dir / "A.npz").write_bytes(b"stub")
            (round_dir / "A.meta.json").write_text(
                json.dumps(
                    {"feature_dim": 0, "integrity": {"valid_gaze_frames": 0, "aborted": True}}
                ),
                encoding="utf-8",
            )
            R.guard_existing_round(round_dir)

    def test_a_real_capture_still_blocks_even_beside_a_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round14"
            round_dir.mkdir()
            (round_dir / "A.npz").write_bytes(b"stub")
            (round_dir / "A.meta.json").write_text(
                json.dumps({"feature_dim": 0, "integrity": {"valid_gaze_frames": 0}}),
                encoding="utf-8",
            )
            (round_dir / "T1.npz").write_bytes(b"real")
            (round_dir / "T1.meta.json").write_text(
                json.dumps({"feature_dim": 258, "integrity": {"valid_gaze_frames": 460}}),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as caught:
                R.guard_existing_round(round_dir)
            self.assertIn("T1.npz", str(caught.exception))
            self.assertNotIn("A.npz", str(caught.exception))

    def test_an_unreadable_manifest_is_treated_as_real_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round14"
            round_dir.mkdir()
            (round_dir / "T1.npz").write_bytes(b"real")
            (round_dir / "T1.meta.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(SystemExit):
                R.guard_existing_round(round_dir)

    def test_a_fresh_or_empty_round_is_fine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            R.guard_existing_round(Path(tmp) / "round99")
            empty = Path(tmp) / "round98"
            empty.mkdir()
            (empty / "setup.json").write_text("{}", encoding="utf-8")
            R.guard_existing_round(empty)


class NoFaceStallTests(unittest.TestCase):
    """round12 recorded 903 frames in which the face was never found and still
    reported finished=true. Frames kept arriving and the subscriber never
    raised, so neither the watchdog nor the error counter noticed."""

    def _runner(self):
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(
            spec, builder, RIG, clock=lambda: 0.0, head_builder=head_ok, pnp=pnp_none
        )
        runner.start()
        return runner

    def test_a_recording_with_no_face_at_all_stops_instead_of_finishing(self) -> None:
        runner = self._runner()
        for _ in range(R.MAX_CONSECUTIVE_INVALID_FRAMES):
            runner.on_frame(*frame(status=False))
        self.assertTrue(runner.no_face_stall)
        self.assertEqual(runner.valid_gaze_frames, 0)

    def test_blinks_and_brief_look_aways_do_not_trip_it(self) -> None:
        runner = self._runner()
        for _ in range(20):
            for _ in range(10):
                runner.on_frame(*frame(status=False))
            runner.on_frame(*frame(status=True))
        self.assertFalse(runner.no_face_stall)
        self.assertGreater(runner.valid_gaze_frames, 0)

    def test_integrity_reports_the_valid_fraction_so_it_cannot_hide(self) -> None:
        runner = self._runner()
        for _ in range(10):
            runner.on_frame(*frame(status=False))
        integrity = runner.integrity(watchdog_tripped=False, aborted=False)
        self.assertEqual(integrity["valid_gaze_frames"], 0)
        self.assertEqual(integrity["valid_gaze_fraction"], 0.0)


class LiveOverlayTests(unittest.TestCase):
    """The overlay is the user's explicit ask: see the system's live point,
    not just a number afterwards -- for every protocol, calibration included."""

    def _frame_with_raw(self, raw=(0.05, -0.05), *, status=True, left=100.0, right=100.0):
        face = SimpleNamespace(status=status, can_gaze_estimation=status, left_eye_openness=left,
                               right_eye_openness=right, timestamp=1, face_landmarks=np.zeros((478, 3)),
                               img_w=640, img_h=480)
        gaze = SimpleNamespace(status=status, features=np.arange(8, dtype=np.float32) if status else None,
                               raw_gaze_coordinates=np.array(raw) if status else None,
                               tracking_state=SimpleNamespace(name="SUCCESS" if status else "FACE_MISSING"),
                               timestamp=1)
        return face, gaze

    def _filter_settings(self, kind=F.FilterKind.ONE_EURO):
        return F.FilterSettings(width_px=RIG.device_w_px, height_px=RIG.device_h_px, kind=kind)

    def test_raw_point_is_shown_even_without_an_overlay_model(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertIsNotNone(runner.state.raw_norm)
        self.assertIsNone(runner.state.overlay_norm)  # no model was loaded

    def test_raw_point_is_shown_during_calibration_not_only_test_protocols(self) -> None:
        # The user's explicit ask: no exception for calibration points --
        # including the WARM-UP point, which is not even stored to disk.
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(runner.state.phase, S.PHASE_WARMUP)
        self.assertIsNotNone(runner.state.raw_norm)
        # And once past the warm-up, into a real stored calibration point.
        clock.tick(0.6)
        while runner.state.phase == S.PHASE_WARMUP:
            runner.on_frame(*self._frame_with_raw())
            clock.tick()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(runner.state.phase, S.PHASE_COLLECT)
        self.assertIsNotNone(runner.state.raw_norm)

    def test_idle_frames_show_no_live_point(self) -> None:
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none)
        runner.on_frame(*self._frame_with_raw())  # before start(): idle
        self.assertIsNone(runner.state.overlay_norm)

    def test_overlay_model_prediction_appears_when_a_model_is_loaded(self) -> None:
        class FakeModel:
            class schema:
                head_names = ()

            def predict_norm(self, X, rig):
                return np.array([[0.5, 0.5]])

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none,
                                  overlay_model=FakeModel())
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(runner.state.overlay_raw_norm, (0.5, 0.5))
        self.assertEqual(runner.state.overlay_norm, (0.5, 0.5))

    def test_blue_point_is_filtered_once_per_new_frame(self) -> None:
        class MovingModel:
            class schema:
                head_names = ()

            def __init__(self):
                self.points = iter(((0.2, 0.2), (0.8, 0.8)))

            def predict_norm(self, X, rig):
                return np.array([next(self.points)])

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(
            spec,
            builder,
            RIG,
            clock=clock,
            head_builder=head_ok,
            pnp=pnp_none,
            overlay_model=MovingModel(),
            overlay_filter_settings=self._filter_settings(),
        )
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(runner.state.overlay_norm, (0.2, 0.2))
        clock.tick()
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(runner.state.overlay_raw_norm, (0.8, 0.8))
        self.assertGreater(runner.state.overlay_norm[0], 0.2)
        self.assertLess(runner.state.overlay_norm[0], 0.8)

    def test_blink_hides_blue_point_and_resets_filter_before_recovery(self) -> None:
        class MovingModel:
            class schema:
                head_names = ()

            def __init__(self):
                self.value = 0.1

            def predict_norm(self, X, rig):
                return np.array([[self.value, self.value]])

        model = MovingModel()
        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(
            spec,
            builder,
            RIG,
            clock=clock,
            head_builder=head_ok,
            pnp=pnp_none,
            overlay_model=model,
            overlay_filter_settings=self._filter_settings(),
        )
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        clock.tick()
        runner.on_frame(*self._frame_with_raw(left=C.BLINK_THRESHOLD))
        self.assertIsNone(runner.state.overlay_raw_norm)
        self.assertIsNone(runner.state.overlay_norm)
        model.value = 0.9
        clock.tick()
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(runner.state.overlay_norm, (0.9, 0.9))

    def test_tracking_loss_hides_blue_point(self) -> None:
        class FakeModel:
            class schema:
                head_names = ()

            def predict_norm(self, X, rig):
                return np.array([[0.5, 0.5]])

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(
            spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none,
            overlay_model=FakeModel(), overlay_filter_settings=self._filter_settings()
        )
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        clock.tick()
        runner.on_frame(*self._frame_with_raw(status=False))
        self.assertIsNone(runner.state.overlay_norm)

    def test_stale_blue_point_is_hidden_if_camera_stops(self) -> None:
        state = R.RunnerState(
            protocol="T1", target=None, phase=S.PHASE_COLLECTING, progress=0,
            target_pos=1, target_count=1, frames=1, fps=30.0, head_valid=True,
            finished=False, pitch_a=0.0, overlay_norm=(0.4, 0.6), overlay_updated_s=10.0,
        )
        self.assertEqual(R.visible_overlay_point(state, 10.2), (0.4, 0.6))
        self.assertIsNone(R.visible_overlay_point(state, 10.251))
        self.assertIsNone(R.visible_overlay_point(state, 9.9))

    def test_raw_diagnostic_point_is_also_hidden_if_camera_stops(self) -> None:
        self.assertEqual(R.visible_point((0.2, 0.3), 10.0, 10.2), (0.2, 0.3))
        self.assertIsNone(R.visible_point((0.2, 0.3), 10.0, 10.3))

    def test_overlay_model_with_head_columns_assembles_the_right_design(self) -> None:
        seen = {}

        class FakeModel:
            class schema:
                head_names = ("pitch_a",)

            def predict_norm(self, X, rig):
                seen["shape"] = X.shape
                return np.array([[0.4, 0.6]])

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none,
                                  overlay_model=FakeModel())
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertEqual(seen["shape"], (1, 9))  # 8 base features + 1 head column
        self.assertEqual(runner.state.overlay_norm, (0.4, 0.6))

    def test_overlay_prediction_failure_yields_no_point_not_a_crash(self) -> None:
        class ExplodingModel:
            class schema:
                head_names = ()

            def predict_norm(self, X, rig):
                raise RuntimeError("boom")

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none,
                                  overlay_model=ExplodingModel())
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())  # must not raise
        self.assertIsNone(runner.state.overlay_norm)
        self.assertEqual(runner.errors, 0)  # a bad overlay point is not a recording error

    def test_non_finite_overlay_prediction_is_suppressed(self) -> None:
        class NanModel:
            class schema:
                head_names = ()

            def predict_norm(self, X, rig):
                return np.array([[np.nan, 0.5]])

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none,
                                  overlay_model=NanModel())
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertIsNone(runner.state.overlay_norm)

    def test_overlay_needs_head_but_head_is_invalid_yields_no_point(self) -> None:
        class NeedsHeadModel:
            class schema:
                head_names = ("pitch_a",)

            def predict_norm(self, X, rig):
                raise AssertionError("should not be called when head is missing")

        clock = FakeClock()
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        runner = R.ProtocolRunner(spec, builder, RIG, clock=clock, head_builder=head_none, pnp=pnp_none,
                                  overlay_model=NeedsHeadModel())
        runner.start()
        clock.tick(1.6)
        runner.on_frame(*self._frame_with_raw())
        self.assertIsNone(runner.state.overlay_norm)
