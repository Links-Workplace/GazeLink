"""Profile storage: no camera, no model loading, no OS input."""

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
import gf_gaze_filter as GF  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_profile as P  # noqa: E402

RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
SETTINGS = GF.FilterSettings(
    width_px=5120,
    height_px=1440,
    kind=GF.FilterKind.ONE_EURO,
    one_euro_min_cutoff_hz=0.4,
    one_euro_beta_hz_per_px_s=0.0,
)


def _profile(name: str = "baseline") -> P.Profile:
    return P.Profile(
        name=name,
        model_dir="recordings/round18/models/svr_fzscore_lnone_C100_g0.0005",
        rig=RIG.to_dict(),
        filter=P.filter_to_dict(SETTINGS),
        pose={"yaw_ratio": -0.017, "pitch_a": -0.189},
        created_utc="2026-09-08T09:00:00+00:00",
    )


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_saving_over_an_existing_profile_is_refused_by_default(self) -> None:
        """A profile is the pointer to the setup someone relies on.

        Replacing it silently is how a known-good configuration disappears
        with nothing left to go back to -- the failure this whole module
        exists to prevent.
        """

        P.save(_profile(), root=self.root)
        with self.assertRaises(FileExistsError):
            P.save(_profile(), root=self.root)
        P.save(_profile(), root=self.root, overwrite=True)  # explicit is fine

    def test_activating_a_missing_profile_leaves_no_pointer_behind(self) -> None:
        """A pointer to something unstartable is worse than no pointer."""

        with self.assertRaises(FileNotFoundError):
            P.activate("does-not-exist", root=self.root)
        self.assertIsNone(P.active_name(root=self.root))
        self.assertIsNone(P.load_active(root=self.root))

    def test_round_trip_preserves_the_filter_exactly(self) -> None:
        """JSON has no enums, so the kind returns as a string.

        Rebuilding must still produce settings equal to the ones saved: a
        filter that quietly came back with different parameters would change
        the measured accuracy while the profile still claimed the old one.
        """

        P.save(_profile(), root=self.root)
        back = P.load("baseline", root=self.root)
        self.assertEqual(back.filter_settings(), SETTINGS)

    def test_the_pixel_size_comes_from_the_rig_not_the_stored_filter(self) -> None:
        saved = _profile()
        self.assertNotIn("width_px", saved.filter)
        self.assertNotIn("height_px", saved.filter)
        self.assertEqual(saved.filter_settings().width_px, RIG.device_w_px)

    def test_a_profile_from_a_future_version_is_refused(self) -> None:
        path = P.profile_path("baseline", self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        body = _profile().to_dict()
        body["version"] = "profile-999"
        path.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(ValueError):
            P.load("baseline", root=self.root)

    def test_unknown_fields_are_refused_rather_than_dropped(self) -> None:
        body = _profile().to_dict()
        body["cursor_speed"] = 3
        with self.assertRaises(ValueError):
            P.Profile.from_dict(body)

    def test_a_name_cannot_escape_the_profiles_directory(self) -> None:
        for bad in ("../secrets", "a/b", "a\\b", ".hidden", ""):
            with self.assertRaises(ValueError):
                P.profile_path(bad, self.root)

    def test_listing_ignores_the_active_pointer(self) -> None:
        P.save(_profile("one"), root=self.root)
        P.save(_profile("two"), root=self.root)
        P.activate("one", root=self.root)
        self.assertEqual(P.list_profiles(root=self.root), ["one", "two"])


class RigTests(unittest.TestCase):
    def test_a_resolution_change_is_reported(self) -> None:
        """The model maps through the screen it was fitted for.

        A changed resolution silently maps the prediction through the wrong
        geometry and reads as lost accuracy, so it must be visible.
        """

        mismatch = P.rig_mismatch(_profile(), C.RigGeometry(60.0, 63.6, 120.0, 33.75, 1920, 1080))
        self.assertTrue(any("device_w_px" in m for m in mismatch))
        self.assertTrue(any("device_h_px" in m for m in mismatch))

    def test_an_identical_rig_reports_nothing(self) -> None:
        self.assertEqual(P.rig_mismatch(_profile(), RIG), [])

    def test_a_measurement_rounding_difference_is_not_a_mismatch(self) -> None:
        """Declared centimetres are hand-measured; 0.01 cm is not a change."""

        nearly = C.RigGeometry(60.0, 63.61, 120.0, 33.75, 5120, 1440)
        self.assertEqual(P.rig_mismatch(_profile(), nearly), [])


class PoseTests(unittest.TestCase):
    def _recording(self, head, names=("yaw_ratio", "pitch_a")) -> SimpleNamespace:
        head = np.asarray(head, dtype=np.float64)
        return SimpleNamespace(
            head=head,
            protocol="T1",
            meta={"head_names": list(names)},
            rows_accepted=lambda: np.ones(len(head), dtype=bool),
            rows_collecting=lambda: np.ones(len(head), dtype=bool),
        )

    def test_pose_delta_is_signed_so_the_direction_survives(self) -> None:
        """Reporting only the size would lose which way the head moved, and
        the direction is what tells drift apart from a different seat."""

        rec = self._recording([[0.040, -0.155]] * 5)
        delta = P.pose_delta(_profile(), rec)
        self.assertAlmostEqual(delta["yaw_ratio"], 0.057, places=3)
        self.assertAlmostEqual(delta["pitch_a"], 0.034, places=3)

    def test_no_head_columns_gives_nothing_rather_than_a_fabricated_zero(self) -> None:
        rec = SimpleNamespace(
            head=None,
            protocol="T1",
            meta={"head_names": []},
            rows_accepted=lambda: np.ones(1, dtype=bool),
            rows_collecting=lambda: np.ones(1, dtype=bool),
        )
        self.assertEqual(P.calibration_pose(rec), {})
        self.assertEqual(P.pose_delta(_profile(), rec), {})

    def test_features_the_profile_never_stored_are_skipped(self) -> None:
        rec = self._recording([[0.0, 0.0, 1.0]] * 3, names=("yaw_ratio", "pitch_a", "iod_norm"))
        self.assertNotIn("iod_norm", P.pose_delta(_profile(), rec))


if __name__ == "__main__":
    unittest.main()


class GestureSettingsTests(unittest.TestCase):
    """The eyelid rule belongs to a face and a camera, not to the algorithm.

    Hard-coded, the same numbers are meaningless on another rig and the
    gesture ends up tuned for whoever last ran it.
    """

    def _profile(self, gesture: dict | None = None) -> PROF.Profile:
        return P.Profile(
            name="t",
            model_dir="m",
            rig={
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            filter={"kind": "one-euro"},
            gesture=gesture or {},
        )

    def test_a_profile_with_no_gesture_settings_gets_the_defaults(self) -> None:
        self.assertEqual(self._profile().wink_config(), GEST.WinkConfig())
        self.assertEqual(self._profile().gate_config(), GEST.OpennessGateConfig())

    def test_saved_values_are_used(self) -> None:
        profile = self._profile({"wink": {"shut_ratio": 0.3, "hold_ms": 200.0}})
        self.assertEqual(profile.wink_config().shut_ratio, 0.3)
        self.assertEqual(profile.wink_config().hold_ms, 200.0)

    def test_a_field_left_out_keeps_its_default_rather_than_becoming_zero(self) -> None:
        profile = self._profile({"wink": {"shut_ratio": 0.3}})
        self.assertEqual(profile.wink_config().asymmetry, GEST.WinkConfig().asymmetry)

    def test_a_stale_field_from_an_older_build_does_not_break_loading(self) -> None:
        """The wink rule has already been rewritten once; it will change again."""

        profile = self._profile({"wink": {"shut_ratio": 0.3, "both_eyes_veto_ms": 600.0}})
        self.assertEqual(profile.wink_config().shut_ratio, 0.3)

    def test_saved_nonsense_is_still_refused(self) -> None:
        """Validation must not be skipped just because a value came from disk."""

        with self.assertRaises(ValueError):
            self._profile({"wink": {"asymmetry": 1.0}}).wink_config()

    def test_the_settings_survive_a_round_trip_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = self._profile({"wink": {"shut_ratio": 0.31}, "gate": {"shut_fraction": 0.6}})
            P.save(saved, root=root)
            back = P.load("t", root=root)
            self.assertEqual(back.wink_config().shut_ratio, 0.31)
            self.assertEqual(back.gate_config().shut_fraction, 0.6)

    def test_the_shipped_profile_carries_a_measured_rule(self) -> None:
        """A rule with no evidence beside it is a guess with a home."""

        baseline = P.load("baseline")
        gesture = baseline.gesture
        self.assertTrue(gesture, "the baseline profile has no gesture settings")
        self.assertIn("evidence", gesture)
        self.assertIn("source", gesture["evidence"])
        self.assertIn("person", gesture["eye_frame"])
