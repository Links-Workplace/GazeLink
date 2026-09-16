"""A model only ever runs on the capture pipeline it was trained on.

A 640x480 model fed hi-crop features, or the reverse, does not fail: it puts
the point confidently in the wrong place. Every live tool therefore opens its
camera through ``gf_live.build_gaze_follower(profile)``, which refuses a
mismatch, and real input is refused for a capture whose eyelid rules have not
been re-checked. No camera and no gazefollower import here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_select_compare as SC  # noqa: E402

RIG = {
    "camera_x_cm": 60.0, "camera_y_cm": 63.6, "screen_w_cm": 120.0, "screen_h_cm": 33.75,
    "device_w_px": 5120, "device_h_px": 1440,
}


def _model(root: Path, name: str, pipeline: str | None) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    train = {} if pipeline is None else {"capture_pipeline": pipeline}
    (directory / "schema.json").write_text(json.dumps({"train": train}), encoding="utf-8")
    return directory


def _profile(model_dir: Path, capture: str | None = None) -> PROF.Profile:
    fields = {} if capture is None else {"capture_pipeline": capture}
    return PROF.Profile(name="p", model_dir=str(model_dir), rig=RIG, filter={}, **fields)


class MatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = _model(self.root, "old", None)
        self.lo = _model(self.root, "lo", PROF.CAPTURE_LIBRARY)
        self.hi = _model(self.root, "hi", PROF.CAPTURE_HI)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_matching_pairs_run(self) -> None:
        self.assertEqual(PROF.require_capture_match(_profile(self.lo)), PROF.CAPTURE_LIBRARY)
        self.assertEqual(PROF.require_capture_match(_profile(self.hi, PROF.CAPTURE_HI)), PROF.CAPTURE_HI)

    def test_an_old_model_and_an_old_profile_are_the_library_path(self) -> None:
        self.assertEqual(PROF.require_capture_match(_profile(self.old)), PROF.CAPTURE_LIBRARY)

    def test_a_lo_model_on_a_hi_profile_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            PROF.require_capture_match(_profile(self.lo, PROF.CAPTURE_HI))
        with self.assertRaises(SystemExit):
            PROF.require_capture_match(_profile(self.old, PROF.CAPTURE_HI))

    def test_a_hi_model_on_a_lo_or_old_profile_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            PROF.require_capture_match(_profile(self.hi, PROF.CAPTURE_LIBRARY))
        with self.assertRaises(SystemExit):
            PROF.require_capture_match(_profile(self.hi))  # no field = library

    def test_an_unknown_pipeline_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            PROF.require_capture_match(_profile(self.hi, "4k-something"))

    def test_an_old_saved_profile_without_the_field_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = _profile(self.old).to_dict()
            data.pop("capture_pipeline")
            path = Path(tmp) / "legacy.json"
            data["name"] = "legacy"
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = PROF.load("legacy", root=Path(tmp))
            self.assertEqual(loaded.capture_pipeline, PROF.CAPTURE_LIBRARY)


class BuilderTests(unittest.TestCase):
    """The one camera builder checks the match before touching the library."""

    def test_build_gaze_follower_refuses_a_mismatch_before_importing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hi = _model(Path(tmp), "hi", PROF.CAPTURE_HI)
            with mock.patch.dict(sys.modules, {"gazefollower": mock.MagicMock()}), \
                    self.assertRaises(SystemExit):
                L.build_gaze_follower(_profile(hi))

    def test_every_live_tool_passes_the_profile_not_the_rig(self) -> None:
        here = Path(__file__).resolve().parent.parent
        for name in ("gf_live.py", "gf_click_practice.py", "gf_gesture_probe.py",
                     "gf_wink_probe.py", "gf_dwell_practice.py"):
            source = (here / name).read_text(encoding="utf-8")
            self.assertNotIn("build_gaze_follower(rig)", source, name)
            if name != "gf_live.py":
                self.assertIn("build_gaze_follower(profile)", source, name)

    def test_live_refuses_cursor_or_clicks_for_a_non_library_capture(self) -> None:
        """Behavioural, not a reading of the source: the refusal must come
        before the camera opens, so the fake library must never be opened."""

        from fake_live_env import FakeWorld  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            hi = _model(Path(tmp), "hi", PROF.CAPTURE_HI)
            world = FakeWorld(runner=None, display=None)
            with self.assertRaises(SystemExit) as caught:
                L.run_live(
                    _profile(hi, PROF.CAPTURE_HI), move_cursor=True, confirmed=True,
                    click_by="wink", env=world.environment(),
                )
        self.assertIn("is not validated for this capture yet", str(caught.exception))
        self.assertEqual(world.calls["open_library"], 0, "the camera was opened before refusing")


class TrialProfileTests(unittest.TestCase):
    def test_a_trial_profile_that_cannot_run_is_never_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hi = _model(root, "hi", PROF.CAPTURE_HI)
            base = _profile(_model(root, "lo", PROF.CAPTURE_LIBRARY))
            with self.assertRaises(SystemExit):
                SC.make_trial_profile(base, name="t", model_dir=str(hi), note="", root=root)
            self.assertFalse((root / "t.json").exists())
            SC.make_trial_profile(
                replace(base), name="t", model_dir=str(hi), note="", root=root,
                capture_pipeline=PROF.CAPTURE_HI,
            )
            self.assertEqual(PROF.load("t", root=root).capture_pipeline, PROF.CAPTURE_HI)


if __name__ == "__main__":
    unittest.main()
