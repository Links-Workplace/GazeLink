"""Accessibility settings: adjustable, but never adjustable into a trap.

CLAUDE.md 4.5 requires dwell, control size and cursor size to be tunable, and
forbids copying a fixed value out of a study as though it fitted everyone. The
bounds here are refusals: a settings file cannot make a glance into a click.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.calibration.profile import AccessibilitySettings as S  # noqa: E402
from gazelink_core.calibration.profile import Profile  # noqa: E402

RIG = {
    "camera_x_cm": 60.0, "camera_y_cm": -2.0, "screen_w_cm": 120.0,
    "screen_h_cm": 33.75, "device_w_px": 5120, "device_h_px": 1440,
}


def profile(**kw):
    return Profile(name="t", model_dir=".", rig=RIG, filter={"kind": "one-euro"}, **kw)


class DefaultsTests(unittest.TestCase):
    def test_a_profile_with_no_block_uses_the_defaults(self) -> None:
        self.assertEqual(profile().accessibility_settings(), S())

    def test_the_default_dwell_is_the_value_the_project_already_uses(self) -> None:
        # Not a new number invented here: gf_dwell has used 900 ms since it
        # was written, and it is the one the practice runs were scored on.
        self.assertEqual(S().dwell_ms, 900.0)

    def test_a_missing_field_keeps_its_default_and_the_others_still_apply(self) -> None:
        got = profile(accessibility={"control_scale": 1.3}).accessibility_settings()
        self.assertEqual(got.dwell_ms, S().dwell_ms)
        self.assertEqual(got.control_scale, 1.3)


class BoundsAreRefusalsTests(unittest.TestCase):
    def test_a_dwell_short_enough_to_fire_on_a_blink_is_clamped(self) -> None:
        # The longest natural blink measured on this rig was 297 ms. A dwell
        # under that would make an involuntary pause into an action.
        got = profile(accessibility={"dwell_ms": 50}).accessibility_settings()
        self.assertGreaterEqual(got.dwell_ms, 350.0)
        self.assertGreater(got.dwell_ms, 297.0)

    def test_an_absurdly_long_dwell_is_clamped(self) -> None:
        got = profile(accessibility={"dwell_ms": 999999}).accessibility_settings()
        self.assertLessEqual(got.dwell_ms, S.DWELL_MS_RANGE[1])

    def test_a_control_scale_that_would_shrink_targets_under_the_bias_is_clamped(self) -> None:
        got = profile(accessibility={"control_scale": 0.2}).accessibility_settings()
        self.assertGreaterEqual(got.control_scale, S.CONTROL_SCALE_RANGE[0])

    def test_a_control_scale_wider_than_the_band_is_clamped(self) -> None:
        got = profile(accessibility={"control_scale": 12.0}).accessibility_settings()
        self.assertLessEqual(got.control_scale, S.CONTROL_SCALE_RANGE[1])

    def test_an_invisible_cursor_is_clamped(self) -> None:
        got = profile(accessibility={"cursor_radius_px": 0}).accessibility_settings()
        self.assertGreaterEqual(got.cursor_radius_px, S.CURSOR_RADIUS_RANGE[0])

    def test_every_bound_is_a_real_range(self) -> None:
        for lo, hi in (S.DWELL_MS_RANGE, S.CONTROL_SCALE_RANGE, S.CURSOR_RADIUS_RANGE):
            self.assertLess(lo, hi)


class BadInputNeverStopsASessionTests(unittest.TestCase):
    def test_a_value_of_the_wrong_type_falls_back_to_the_default(self) -> None:
        got = profile(accessibility={"dwell_ms": "soon"}).accessibility_settings()
        self.assertEqual(got.dwell_ms, S().dwell_ms)

    def test_a_null_value_falls_back_to_the_default(self) -> None:
        got = profile(accessibility={"cursor_radius_px": None}).accessibility_settings()
        self.assertEqual(got.cursor_radius_px, S().cursor_radius_px)

    def test_an_unknown_field_inside_the_block_is_ignored(self) -> None:
        # The block is free-form on purpose: a newer build writing a field
        # this one does not know must not make the profile unloadable.
        got = profile(accessibility={"dwell_ms": 1200, "invented": True}).accessibility_settings()
        self.assertEqual(got.dwell_ms, 1200.0)


class RoundTripTests(unittest.TestCase):
    def test_settings_survive_being_written_and_read_back(self) -> None:
        wanted = S(dwell_ms=1500.0, control_scale=1.2, cursor_radius_px=20)
        got = profile(accessibility=wanted.to_dict()).accessibility_settings()
        self.assertEqual(got, wanted)

    def test_a_profile_saved_by_an_older_build_still_loads(self) -> None:
        # No accessibility key at all, which is every profile on disk today.
        raw = profile().to_dict()
        raw.pop("accessibility")
        self.assertEqual(Profile.from_dict(raw).accessibility_settings(), S())

    def test_the_block_is_a_declared_field_and_not_a_stowaway(self) -> None:
        # from_dict rejects unknown keys, so an undeclared block would make a
        # profile that carries it unloadable.
        self.assertIn("accessibility", Profile.__dataclass_fields__)
        self.assertIn("accessibility", profile().to_dict())


class SeparationTests(unittest.TestCase):
    def test_dwell_does_not_live_with_the_cursor_comfort_settings(self) -> None:
        # ``cursor`` is documented as presentation only and safe to change.
        # Dwell changes selection timing and is not in that class.
        self.assertNotIn("dwell_ms", profile().cursor)

    def test_the_filter_is_not_reachable_from_here(self) -> None:
        # The filter belongs to the model and the verified selection task; it
        # must not be retuned to make a cursor feel comfortable.
        self.assertNotIn("filter", S().to_dict())


if __name__ == "__main__":
    unittest.main()
