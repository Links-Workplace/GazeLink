"""A whole live session with no hardware (TECHNICAL_SPEC 17, "tests without hardware").

``LiveSession`` runs end to end on: a replayed recording as the tracking source,
the REAL ``FramePipeline`` (gate, model, filter, wink detector), a fitted model,
a recording display, recorded input senders and a fake clock. Nothing opens a
camera or a window and nothing reaches Windows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import gf_presets as PRE  # noqa: E402
import gf_schema as S  # noqa: E402
from gazelink_core.app.live_session import LiveSession  # noqa: E402
from gazelink_core.app.options import LiveOptions  # noqa: E402
from gazelink_core.calibration.model import FittedModel  # noqa: E402
from gazelink_core.domain import common as C  # noqa: E402
from gazelink_core.domain.clock import FakeClock  # noqa: E402
from gazelink_core.gaze.pipeline import FramePipeline  # noqa: E402
from gazelink_core.platform import click as CK  # noqa: E402
from gazelink_core.tracking.replay_source import ReplaySource, observations  # noqa: E402

from fake_live_env import FakeWorld  # noqa: E402
from live_trace_harness import RecordingDisplay, profile  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
FPS = 30.0


def _model() -> FittedModel:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.column_stack([rng.uniform(-10, 10, 60), rng.uniform(10, 25, 60)])
    config = next(
        c for c in PRE.sweep_with_presets(()) if c.name == "svr_fzscore_lnone_C100_g0.0005"
    )
    return FittedModel.fit(config, X, y, rig=RIG, train_meta={"fitted_by": "e2e test"})


def _recording(*, wink: bool, eye: str = "right") -> S.Recording:
    rng = np.random.default_rng(5)
    builder = S.RecordingBuilder("T1", 0, {"rig": RIG.to_dict(), "targets": []})
    base = rng.normal(size=8).astype(np.float32)
    for i in range(240):
        # The person's RIGHT eye is the image-left column: a right wink is a
        # small first value with the other eye open, a LEFT wink a small second.
        shut = wink and 150 <= i < 158
        left_img = 12.0 if (shut and eye == "right") else 150.0
        right_img = 11.0 if (shut and eye == "left") else 130.0
        builder.append(
            frame_seq=i,
            timestamp_ns=1_000_000_000 + int(i * 1e9 / FPS),
            elapsed_ms=None,
            target_id=-1,
            block=0,
            phase=S.PHASE_IDLE,
            target_xy=None,
            label_cm=None,
            features=(base + rng.normal(scale=0.01, size=8)).astype(np.float32),
            head=None,
            pnp_deg=None,
            raw_cm=(1.0, 2.0),
            openness=(left_img, right_img),
            tracking_state="SUCCESS",
            gaze_status=True,
            accepted=False,
        )
    return builder.freeze()


class ClockedReplay(ReplaySource):
    """Publishes each recorded frame when the fake clock reaches its time."""

    def __init__(self, recording: S.Recording, clock: FakeClock) -> None:
        super().__init__(recording)
        self.clock = clock
        self._pending: list = []

    def start(self) -> None:
        super().start()
        self._pending = list(observations(self.recording, start_s=self.clock.now()))
        self.clock.on_advance.append(self._due)

    def _due(self, now: float) -> None:
        while self._pending and self._pending[0].observed_s <= now and not self._stopping.is_set():
            obs = self._pending.pop(0)
            for sink in list(self._sinks):
                sink(obs)
            self.published += 1


def _run(  # noqa: ANN003
    *, wink: bool, eye: str = "right", wink_click: str = "double", **option_overrides
) -> tuple[int, FakeWorld, list, str, ClockedReplay]:
    clock = FakeClock(500.0)
    draws: list = []
    world = FakeWorld(
        runner=None, display=RecordingDisplay(draws), clock=clock, title=lambda hwnd: "window"
    )
    model = _model()
    source = ClockedReplay(_recording(wink=wink, eye=eye), clock)
    env = dataclasses.replace(
        world.environment(),
        open_source=lambda prof: (world._count("open_library"), source)[1],
        make_runner=lambda *a, **kw: FramePipeline(*a, **kw),
        load_model=lambda path: model,
    )
    options = LiveOptions(
        move_cursor=True,
        confirmed=True,
        click_by="wink",
        skip_model_check=True,
        max_seconds=8.5,
        wink_click=wink_click,
        **option_overrides,
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = LiveSession(profile(), options, env).run()
    return code, world, draws, out.getvalue(), source


class EndToEndTests(unittest.TestCase):
    def test_a_replayed_right_wink_right_clicks_through_the_real_pipeline(self) -> None:
        # The eye picks the button: a RIGHT wink is a right click whatever the
        # left-wink setting is (double here, deliberately, to show it is ignored).
        code, world, draws, text, source = _run(wink=True, eye="right", wink_click="double")
        self.assertEqual(code, 0)
        self.assertGreater(source.published, 200, "the replay never reached the pipeline")
        self.assertTrue(world.moves, "the pointer never followed the replayed gaze")
        self.assertEqual(world.sends.count(CK.MOUSEEVENTF_RIGHTDOWN), 1, f"sent {world.sends}")
        self.assertEqual(
            world.sends.count(CK.MOUSEEVENTF_RIGHTUP), 1, "the right button stayed down"
        )
        self.assertNotIn(CK.MOUSEEVENTF_LEFTDOWN, world.sends, "a right wink left clicked")
        self.assertIn("winks detected         : 1", text)
        self.assertTrue(source.closed, "the source was not closed")
        self.assertIn(("close",), draws, "the display was not closed")
        self.assertTrue(any(d[0] == "live" for d in draws))

    def test_a_replayed_left_wink_double_clicks_when_the_toggle_is_on(self) -> None:
        code, world, _draws, text, _source = _run(wink=True, eye="left", wink_click="double")
        self.assertEqual(code, 0)
        downs = world.sends.count(CK.MOUSEEVENTF_LEFTDOWN)
        self.assertEqual(downs, 2, f"expected one double click, sent {world.sends}")
        self.assertEqual(downs, world.sends.count(CK.MOUSEEVENTF_LEFTUP))
        self.assertNotIn(CK.MOUSEEVENTF_RIGHTDOWN, world.sends, "a left wink right clicked")
        self.assertIn("winks detected         : 1", text)

    def test_a_replayed_left_wink_single_clicks_by_default(self) -> None:
        code, world, _draws, _text, _source = _run(wink=True, eye="left", wink_click="single")
        self.assertEqual(code, 0)
        self.assertEqual(world.sends, [CK.MOUSEEVENTF_LEFTDOWN, CK.MOUSEEVENTF_LEFTUP])

    def test_the_same_session_without_a_wink_never_clicks(self) -> None:
        code, world, _draws, text, _source = _run(wink=False)
        self.assertEqual(code, 0)
        self.assertTrue(world.moves)
        self.assertEqual(world.sends, [])
        self.assertIn("winks detected         : 0", text)

    def test_a_session_started_paused_follows_nothing_and_clicks_nothing(self) -> None:
        code, world, _draws, text, _source = _run(wink=True, start_active=False)
        self.assertEqual(code, 0)
        self.assertEqual(world.sends, [])
        self.assertIn("winks while paused     : 1", text)
        # The only pointer placement is the restore on exit.
        self.assertEqual([m for m in world.moves if m != (0, 0)], [])


if __name__ == "__main__":
    unittest.main()
