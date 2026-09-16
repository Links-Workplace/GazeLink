"""Dual-resolution capture: crop geometry, openness units, paired rows, guards.

No camera and no gazefollower import. The camera class itself needs the
library and a device; what is tested here is everything that decides whether
the two recordings describe the same frames in the same units.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_capture as CAP  # noqa: E402
import gf_common as C  # noqa: E402
import gf_pool as POOL  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402

from tests.test_record_logic import META, RIG, FakeClock, frame, head_ok, pnp_none  # noqa: E402


class CropTests(unittest.TestCase):
    def test_the_crop_is_centred_4x3_and_keeps_the_middle_pixel(self) -> None:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[540, 960] = 255
        crop = CAP.centre_crop_4x3(img)
        self.assertEqual(crop.shape, (1080, 1440, 3))
        self.assertEqual(crop[540, 720, 0], 255, "the crop is not centred")

    def test_a_frame_smaller_than_the_crop_is_refused_not_padded(self) -> None:
        with self.assertRaises(ValueError):
            CAP.centre_crop_4x3(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_lo_is_the_same_region_at_640x480_without_stretching(self) -> None:
        crop = np.zeros((1080, 1440, 3), dtype=np.uint8)
        self.assertEqual(CAP.reduce_to_lo(crop).shape, (480, 640, 3))
        with self.assertRaises(ValueError):
            CAP.reduce_to_lo(np.zeros((1080, 1920, 3), dtype=np.uint8))  # 16:9 -> 4:3 would stretch

    def test_a_square_in_the_crop_stays_square_in_lo(self) -> None:
        """The user's requirement: same proportions in both arms."""

        crop = np.zeros((1080, 1440), dtype=np.uint8)
        crop[405:675, 585:855] = 255  # 270x270 square
        lo = CAP.reduce_to_lo(crop)
        ys, xs = np.nonzero(lo > 127)
        self.assertAlmostEqual((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1), 1.0, delta=0.02)


class OpennessUnitsTests(unittest.TestCase):
    def test_hi_openness_is_reported_in_640x480_area_units(self) -> None:
        inner = SimpleNamespace(
            detect=lambda ts, img: SimpleNamespace(left_eye_openness=506.25, right_eye_openness=0.0)
        )
        wrapped = CAP.ScaledOpennessAlignment(inner, CAP.openness_scale())
        info = wrapped.detect(0, None)
        # (640/1440)^2 * 506.25 = 100: the same eye as a 100 px^2 lo eye.
        self.assertAlmostEqual(info.left_eye_openness, 100.0)
        self.assertEqual(info.right_eye_openness, 0.0)

    def test_the_blink_threshold_means_the_same_eye_in_both_arms(self) -> None:
        """Unscaled, a hi eye just under the lo threshold would pass it ~5x over."""

        hi_area_of_a_closing_eye = 9.0 / CAP.openness_scale()
        self.assertGreater(hi_area_of_a_closing_eye, C.BLINK_THRESHOLD)
        self.assertLessEqual(hi_area_of_a_closing_eye * CAP.openness_scale(), C.BLINK_THRESHOLD)


def shadow_frame(status=True, left=100.0, dim=8, ts=1, offset=1000.0):
    face, gaze = frame(status=status, left=left, right=left, dim=dim, ts=ts)
    if status:
        gaze.features = gaze.features + offset  # distinguishable from the primary
    return SimpleNamespace(face_info=face, gaze_info=gaze)


class PairedRowTests(unittest.TestCase):
    def _run(self, spec, shadow_fn):
        clock = FakeClock()
        meta = dict(META, targets=spec.exported_targets())
        primary = S.RecordingBuilder(spec.name, 0, meta)
        shadow = S.RecordingBuilder(spec.name, 0, meta)
        runner = R.ProtocolRunner(
            spec, primary, RIG, clock=clock, head_builder=head_ok, pnp=pnp_none,
            shadow=shadow_fn, shadow_builder=shadow,
        )
        runner.start()
        i = 0
        while not runner.finished and i < 5000:
            runner.on_frame(*frame(status=True, ts=i + 1))
            clock.tick()
            i += 1
        self.assertEqual(runner.errors, 0, runner.last_error)
        return primary.freeze(), shadow.freeze()

    def test_both_recordings_have_identical_rows_and_differ_only_in_features(self) -> None:
        a, b = self._run(R.protocol_a(), lambda ts: shadow_frame(ts=ts))
        self.assertEqual(len(a.target_id), len(b.target_id))
        for name in ("frame_seq", "timestamp_ns", "target_id", "phase"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        np.testing.assert_array_equal(a.rows_accepted(), b.rows_accepted())
        acc = a.rows_accepted()
        self.assertTrue(np.allclose(b.features[acc] - a.features[acc], 1000.0))

    def test_the_shadow_row_is_looked_up_by_the_frame_timestamp(self) -> None:
        seen = []
        self._run(R.protocol_a(), lambda ts: seen.append(ts) or shadow_frame(ts=ts))
        self.assertEqual(seen[:3], [1, 2, 3], "shadow results were matched to the wrong frames")

    def test_a_frame_the_shadow_lost_is_never_a_shadow_training_row(self) -> None:
        def flaky(ts):
            if ts % 4 == 0:
                return None  # the lo pipeline raised or produced nothing
            return shadow_frame(status=ts % 5 != 0, ts=ts)

        a, b = self._run(R.protocol_a(), flaky)
        lost = np.array([ts % 4 == 0 or ts % 5 == 0 for ts in b.timestamp_ns])
        self.assertTrue(np.any(a.rows_accepted() & lost), "fixture must include lost accepted frames")
        self.assertFalse(np.any(b.rows_accepted() & lost))
        self.assertTrue(np.all(b.tracking_state[b.timestamp_ns % 4 == 0] == "NO_SHADOW_RESULT"))

    def test_shadow_needs_both_parts(self) -> None:
        spec = R.protocol_a()
        builder = S.RecordingBuilder("A", 0, dict(META, targets=spec.exported_targets()))
        with self.assertRaises(ValueError):
            R.ProtocolRunner(spec, builder, RIG, shadow=lambda ts: None)


class GuardTests(unittest.TestCase):
    def test_dual_refuses_dry_run_overlay_and_the_pool_directory(self) -> None:
        ok = dict(dry_run=False, overlay_model_dir=None, out_root=R.RECORDINGS_DIR / "resolution", no_save=False)
        R.check_dual_capture_allowed(**ok)
        for bad in (
            dict(ok, dry_run=True),
            dict(ok, overlay_model_dir=Path("recordings/round18/models/x")),
            dict(ok, out_root=R.RECORDINGS_DIR),
        ):
            with self.assertRaises(SystemExit, msg=str(bad)):
                R.check_dual_capture_allowed(**bad)

    def test_pipelines_never_share_a_pool_key_and_old_recordings_keep_theirs(self) -> None:
        base = {"feature_dim": 258, "rig": RIG.to_dict()}
        old = POOL.compatibility_key(base)
        library = POOL.compatibility_key(dict(base, capture={"pipeline": CAP.LIBRARY_PIPELINE}))
        hi = POOL.compatibility_key(dict(base, capture={"pipeline": CAP.PIPELINE_HI}))
        lo = POOL.compatibility_key(dict(base, capture={"pipeline": CAP.PIPELINE_LO}))
        self.assertEqual(old, library, "recordings without the field stopped pooling")
        self.assertEqual(len({library, hi, lo}), 3)

    def test_the_lo_recording_lands_outside_the_pool_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "resolution" / "round1"
            self.assertEqual((round_dir / R.DUAL_LO_SUBDIR).parent, round_dir)
            self.assertNotEqual(round_dir.parent.name, "recordings")




class HiCaptureGuardTests(unittest.TestCase):
    """Single-pipeline hi recordings: only under recordings/hi, never an overlay."""

    def _ok(self):
        return dict(dry_run=False, overlay_model_dir=None, out_root=R.HI_RECORDINGS_DIR, no_save=False)

    def test_hi_is_allowed_only_under_recordings_hi(self) -> None:
        R.check_hi_capture_allowed(**self._ok())
        R.check_hi_capture_allowed(**dict(self._ok(), out_root=R.HI_RECORDINGS_DIR / "day1"))
        for bad_root in (R.RECORDINGS_DIR, R.RECORDINGS_DIR / "eval", R.RECORDINGS_DIR / "hires"):
            with self.assertRaises(SystemExit, msg=str(bad_root)):
                R.check_hi_capture_allowed(**dict(self._ok(), out_root=bad_root))

    def test_hi_refuses_an_overlay_and_a_dry_run(self) -> None:
        with self.assertRaises(SystemExit):
            R.check_hi_capture_allowed(**dict(self._ok(), overlay_model_dir=Path("m")))
        with self.assertRaises(SystemExit):
            R.check_hi_capture_allowed(**dict(self._ok(), dry_run=True))

    def test_the_hi_pipeline_has_its_own_pool_key(self) -> None:
        base = {"feature_dim": 258, "rig": RIG.to_dict()}
        keys = {
            POOL.compatibility_key(dict(base, capture={"pipeline": p}))
            for p in (CAP.LIBRARY_PIPELINE, CAP.PIPELINE_HI, CAP.PIPELINE_HI_SINGLE)
        }
        self.assertEqual(len(keys), 3, "single-pipeline hi would pool with dual or library data")

    def test_the_cli_accepts_hi(self) -> None:
        self.assertIn("hi", next(a for a in R.build_parser()._actions if a.dest == "capture").choices)


if __name__ == "__main__":
    unittest.main()
