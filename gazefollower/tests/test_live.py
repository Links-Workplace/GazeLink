"""Free-running live view: no camera, no model fitting, no OS input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_common as C  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as P  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
SETTINGS = GF.FilterSettings(
    width_px=5120,
    height_px=1440,
    kind=GF.FilterKind.ONE_EURO,
    one_euro_min_cutoff_hz=0.4,
    one_euro_beta_hz_per_px_s=0.0,
)
OPEN = C.BLINK_THRESHOLD + 5.0
SHUT = 0.0


class _Model:
    """A model that answers with a fixed point, or explodes on demand."""

    def __init__(self, point=(0.5, 0.5), raises: bool = False) -> None:
        self.schema = SimpleNamespace(head_names=(), columns=range(4))
        self._point = point
        self._raises = raises
        self.calls = 0

    def predict_norm(self, design, rig):
        self.calls += 1
        if self._raises:
            raise RuntimeError("model blew up")
        return np.array([self._point], dtype=np.float64)


def _frame(openness: float = OPEN, status: bool = True):
    face = SimpleNamespace(left_eye_openness=openness, right_eye_openness=openness)
    gaze = SimpleNamespace(
        status=status,
        features=np.zeros(4, dtype=np.float32),
        raw_gaze_coordinates=None,
    )
    return face, gaze


class LiveRunnerTests(unittest.TestCase):
    def _runner(self, model=None, settings=SETTINGS):
        clock = {"t": 0.0}

        def tick():
            clock["t"] += 1.0 / 30.0
            return clock["t"]

        return L.LiveRunner(
            model or _Model(), None, RIG, settings, clock=tick, head_builder=lambda f: None
        )

    def test_an_open_eyed_frame_produces_a_point(self) -> None:
        runner = self._runner()
        runner.on_frame(*_frame())
        self.assertIsNotNone(runner.state.point)
        self.assertTrue(runner.state.tracking)
        self.assertIsNotNone(runner.state.updated_s)

    def test_a_blink_produces_no_point_and_is_not_reported_as_tracking(self) -> None:
        """Predicting through a blink puts the dot where the eye is not."""

        runner = self._runner()
        runner.on_frame(*_frame())
        runner.on_frame(*_frame(openness=SHUT))
        self.assertIsNone(runner.state.point)
        self.assertFalse(runner.state.tracking)

    def test_a_blink_clears_the_filter_so_the_point_cannot_resume_stale(self) -> None:
        """Without the reset the filter would carry pre-blink history across
        the gap and the point would slide out of a blink from where it was,
        rather than from where the eye now is."""

        runner = self._runner()
        for _ in range(10):
            runner.on_frame(*_frame())
        settled = runner.state.point
        self.assertIsNotNone(settled)
        runner.on_frame(*_frame(openness=SHUT))
        self.assertIsNone(runner.state.point)
        # A far-away model output right after the blink must be followed
        # immediately, not damped toward the pre-blink position.
        runner.model = _Model(point=(0.9, 0.9))
        runner.on_frame(*_frame())
        self.assertAlmostEqual(runner.state.point[0], 0.9, places=6)

    def test_a_frame_with_no_gaze_status_yields_nothing(self) -> None:
        runner = self._runner()
        runner.on_frame(*_frame(status=False))
        self.assertIsNone(runner.state.point)
        self.assertFalse(runner.state.tracking)

    def test_a_model_that_raises_never_reaches_the_camera_thread(self) -> None:
        """on_frame runs on the library's thread; an exception there kills
        tracking for the whole session."""

        runner = self._runner(model=_Model(raises=True))
        runner.on_frame(*_frame())  # must not raise
        self.assertIsNone(runner.state.point)

    def test_a_broken_frame_never_reaches_the_camera_thread(self) -> None:
        runner = self._runner()
        runner.on_frame(None, None)  # must not raise
        self.assertGreaterEqual(runner.errors, 0)

    def test_without_a_filter_the_raw_point_is_shown_unchanged(self) -> None:
        runner = self._runner(settings=None)
        runner.on_frame(*_frame())
        self.assertEqual(runner.state.point, (0.5, 0.5))

    def test_the_frame_window_does_not_grow_without_bound(self) -> None:
        """This view is meant to be left running; an unbounded list is a leak."""

        runner = self._runner()
        for _ in range(500):
            runner.on_frame(*_frame())
        self.assertLessEqual(len(runner._frame_times), 60)
        self.assertEqual(runner.state.frames, 500)


class RigCheckTests(unittest.TestCase):
    def _profile(self) -> P.Profile:
        return P.Profile(
            name="baseline",
            model_dir="somewhere",
            rig=RIG.to_dict(),
            filter=P.filter_to_dict(SETTINGS),
        )

    def test_a_matching_rig_is_accepted(self) -> None:
        L.check_rig(self._profile(), RIG, allow_mismatch=False)  # must not raise

    def test_a_different_screen_is_refused(self) -> None:
        """A profile run against another display does not degrade gracefully:
        it produces a confidently wrong point with nothing on screen saying so."""

        other = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 1920, 1080)
        with self.assertRaises(SystemExit) as caught:
            L.check_rig(self._profile(), other, allow_mismatch=False)
        self.assertIn("device_w_px", str(caught.exception))

    def test_the_refusal_can_be_overridden_deliberately(self) -> None:
        other = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 1920, 1080)
        L.check_rig(self._profile(), other, allow_mismatch=True)  # warns, does not raise


if __name__ == "__main__":
    unittest.main()
