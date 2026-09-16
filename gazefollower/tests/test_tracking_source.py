"""The tracking boundary: observe(), GazeFollowerSource, ReplaySource (ARCH-01 E).

No camera, no gazefollower import, no OS input. The library is a fake object
with ``add_subscriber`` and ``camera.start_sampling``.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import gf_common as C  # noqa: E402
import gf_live as L  # noqa: E402
import gf_schema as S  # noqa: E402
from gazelink_core.domain.clock import FakeClock  # noqa: E402
from gazelink_core.domain.observation import HeadPolicy, Reason  # noqa: E402
from gazelink_core.gaze.pipeline import FramePipeline  # noqa: E402
from gazelink_core.tracking import gazefollower_source as SRC  # noqa: E402
from gazelink_core.tracking.replay_source import ReplaySource, observations  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)


def _face(**kw):
    base = dict(status=True, left_eye_openness=150.0, right_eye_openness=130.0, timestamp=7)
    base.update(kw)
    return SimpleNamespace(**base)


def _gaze(**kw):
    base = dict(
        status=True,
        features=np.arange(4, dtype=np.float32),
        timestamp=9,
        raw_gaze_coordinates=(1.5, 2.5),
        tracking_state=SimpleNamespace(name="SUCCESS"),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class ObserveTests(unittest.TestCase):
    def test_features_are_a_read_only_copy_in_the_library_dtype(self) -> None:
        source = np.arange(4, dtype=np.float32)
        obs = SRC.observe(
            _face(), _gaze(features=source), observed_s=1.0, head_builder=lambda f: None
        )
        self.assertEqual(obs.features.dtype, np.float32)
        source[0] = 99.0
        self.assertEqual(obs.features[0], 0.0, "the observation shares the library's buffer")
        with self.assertRaises(ValueError):
            obs.features[0] = 1.0

    def test_no_gaze_means_no_features_and_no_raw_estimate(self) -> None:
        obs = SRC.observe(_face(), _gaze(status=False), observed_s=1.0, head_builder=lambda f: None)
        self.assertIsNone(obs.features)
        self.assertIsNone(obs.raw_gaze_cm)
        self.assertIn(Reason.NO_GAZE, obs.reasons)

    def test_missing_openness_is_zero_and_flagged_not_invented(self) -> None:
        face = SimpleNamespace(status=True)
        obs = SRC.observe(face, _gaze(), observed_s=1.0, head_builder=lambda f: None)
        self.assertEqual(obs.openness_image, (0.0, 0.0))
        self.assertFalse(obs.openness_available)
        self.assertIn(Reason.OPENNESS_MISSING, obs.reasons)

    def test_head_policy_decides_when_the_pose_is_built(self) -> None:
        calls: list[object] = []
        builder = lambda f: (calls.append(f), np.ones(6))[1]  # noqa: E731
        SRC.observe(
            _face(),
            _gaze(status=False),
            observed_s=0,
            head_builder=builder,
            head_policy=HeadPolicy.GAZE,
        )
        self.assertEqual(calls, [], "the recorder's policy built a pose with no usable gaze")
        obs = SRC.observe(
            _face(),
            _gaze(status=False),
            observed_s=0,
            head_builder=builder,
            head_policy=HeadPolicy.GAZE_OR_FACE,
        )
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(obs.head6)

    def test_tracking_state_renderings(self) -> None:
        named = SRC.observe(_face(), _gaze(), observed_s=0, head_builder=lambda f: None)
        self.assertEqual(named.tracking_label, "SUCCESS")
        absent = SRC.observe(
            _face(), _gaze(tracking_state=None), observed_s=0, head_builder=lambda f: None
        )
        self.assertEqual(absent.tracking_label, "UNKNOWN")
        self.assertEqual(absent.tracking_state_name or absent.tracking_state_text, "None")

    def test_a_malformed_raw_estimate_is_dropped_with_a_reason(self) -> None:
        obs = SRC.observe(
            _face(), _gaze(raw_gaze_coordinates=("x",)), observed_s=0, head_builder=lambda f: None
        )
        self.assertIsNone(obs.raw_gaze_cm)
        self.assertIn(Reason.RAW_GAZE_INVALID, obs.reasons)

    def test_the_library_timestamp_prefers_gaze_then_face(self) -> None:
        obs = SRC.observe(
            _face(timestamp=7), _gaze(timestamp=0), observed_s=0, head_builder=lambda f: None
        )
        self.assertEqual(obs.timestamp_ns, 7)


class _FakeLibrary:
    def __init__(self) -> None:
        self.callbacks: list = []
        self.sampling = False
        self.camera = SimpleNamespace(start_sampling=self._start)

    def _start(self) -> None:
        self.sampling = True

    def add_subscriber(self, fn) -> None:  # noqa: ANN001
        self.callbacks.append(fn)

    def frame(self, face=None, gaze=None) -> None:  # noqa: ANN001
        for fn in self.callbacks:
            fn(face if face is not None else _face(), gaze if gaze is not None else _gaze())


class SourceTests(unittest.TestCase):
    def _source(self, **kw):  # noqa: ANN202
        library = _FakeLibrary()
        shutdowns: list[object] = []
        clock = FakeClock(50.0)
        source = SRC.GazeFollowerSource(
            None,
            clock=clock,
            head_builder=kw.pop("head_builder", lambda f: None),
            open_library=lambda profile: library,
            shutdown_library=lambda gf: (shutdowns.append(gf), "shut")[1],
            **kw,
        ).open()
        return source, library, shutdowns, clock

    def test_publishes_observations_with_sequence_and_injected_time(self) -> None:
        source, library, _s, clock = self._source()
        got = []
        source.subscribe(got.append)
        source.start()
        self.assertTrue(library.sampling)
        library.frame()
        clock.advance(0.5)
        library.frame()
        self.assertEqual([o.frame_seq for o in got], [0, 1])
        self.assertEqual([o.observed_s for o in got], [50.0, 50.5])

    def test_nothing_is_published_after_stop_and_close_is_idempotent(self) -> None:
        source, library, shutdowns, _c = self._source()
        got = []
        source.subscribe(got.append)
        self.assertTrue(source.stop())
        library.frame()
        self.assertEqual(got, [])
        self.assertEqual(source.close(), "shut")
        self.assertEqual(source.close(), "shut")
        self.assertEqual(len(shutdowns), 1, "the library was shut down twice")

    def test_stop_waits_for_a_frame_already_being_delivered(self) -> None:
        source, library, _s, _c = self._source()
        inside, release = threading.Event(), threading.Event()
        order: list[str] = []

        def slow_sink(obs) -> None:  # noqa: ANN001
            inside.set()
            release.wait(5)
            order.append("sink finished")

        source.subscribe(slow_sink)
        camera = threading.Thread(target=library.frame)
        camera.start()
        self.assertTrue(inside.wait(5))
        stopper = threading.Thread(target=lambda: (source.stop(wait_s=5), order.append("stopped")))
        stopper.start()
        time.sleep(0.05)
        self.assertEqual(order, [], "stop returned while a sink was still running")
        release.set()
        stopper.join(5)
        camera.join(5)
        self.assertEqual(order, ["sink finished", "stopped"])

    def test_stop_reports_a_sink_stuck_past_the_bound(self) -> None:
        source, library, _s, _c = self._source()
        inside, release = threading.Event(), threading.Event()
        source.subscribe(lambda obs: (inside.set(), release.wait(5)))
        camera = threading.Thread(target=library.frame)
        camera.start()
        self.assertTrue(inside.wait(5))
        self.assertFalse(source.stop(wait_s=0.05))
        release.set()
        camera.join(5)

    def test_a_translation_failure_is_counted_and_never_raised_into_the_library(self) -> None:
        def broken(face) -> None:  # noqa: ANN001
            raise RuntimeError("landmarks")

        source, library, _s, _c = self._source(head_builder=broken)
        got = []
        source.subscribe(got.append)
        library.frame()
        self.assertEqual((source.errors, got), (1, []))
        self.assertIn("landmarks", source.last_error)


def _synthetic_recording() -> S.Recording:
    rng = np.random.default_rng(3)
    builder = S.RecordingBuilder("T1", 0, {"rig": RIG.to_dict(), "targets": []})
    for i in range(40):
        status = i % 9 != 4
        builder.append(
            frame_seq=i,
            timestamp_ns=1_000_000_000 + i * 33_000_000,
            elapsed_ms=None,
            target_id=-1,
            block=0,
            phase=S.PHASE_IDLE,
            target_xy=None,
            label_cm=None,
            features=rng.normal(size=8).astype(np.float32) if status else None,
            head=None,
            pnp_deg=None,
            raw_cm=(1.0, 2.0) if status else None,
            openness=(150.0, 20.0 if 20 <= i < 24 else 130.0),
            tracking_state="SUCCESS" if status else "FACE_MISSING",
            gaze_status=status,
            accepted=False,
        )
    return builder.freeze()


class ReplayTests(unittest.TestCase):
    def test_replay_drives_the_core_pipeline_exactly_like_the_library_path(self) -> None:
        """The same rows, once as library-shaped callbacks through the adapter
        and once as replayed observations, give identical pipeline states."""

        rec = _synthetic_recording()
        clock = {"t": 0.0}
        via_library = L.LiveRunner(
            None, None, RIG, None, clock=lambda: clock["t"], head_builder=lambda f: None
        )
        replayed = FramePipeline(None, None, RIG, None)
        library_states, replay_states = [], []
        for obs in observations(rec):
            clock["t"] = obs.observed_s
            gaze = SimpleNamespace(
                status=obs.gaze_status,
                features=obs.features,
                raw_gaze_coordinates=obs.raw_gaze_cm,
                tracking_state=None,
            )
            face = SimpleNamespace(
                status=obs.face_present,
                left_eye_openness=obs.openness_image[0],
                right_eye_openness=obs.openness_image[1],
            )
            via_library.on_frame(face, gaze)
            library_states.append(via_library.state)
        source = ReplaySource(rec)
        source.subscribe(replayed.on_observation)
        source.subscribe(lambda obs: replay_states.append(replayed.state))
        self.assertEqual(source.run(), rec.n_rows)
        self.assertEqual(replay_states, library_states)
        self.assertEqual(replayed.errors, 0)
        self.assertEqual(replayed.drain_wink_events(), via_library.drain_wink_events())

    def test_a_stopped_replay_publishes_nothing_more(self) -> None:
        rec = _synthetic_recording()
        source = ReplaySource(rec)
        seen = []
        source.subscribe(lambda obs: (seen.append(obs), source.stop() if len(seen) == 5 else None))
        source.run()
        self.assertEqual(len(seen), 5)
        self.assertEqual(source.close()["published"], 5)


if __name__ == "__main__":
    unittest.main()
