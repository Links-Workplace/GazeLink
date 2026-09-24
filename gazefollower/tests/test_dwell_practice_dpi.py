"""The --dpi-aware diagnostic on gf_dwell_practice: off by default, and when on,
declared before anything opens a window. No camera, no window, no OS input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_dwell_practice as DP  # noqa: E402


class DpiAwareFlagTests(unittest.TestCase):
    def test_off_by_default_so_the_existing_practice_is_unchanged(self) -> None:
        self.assertFalse(DP.build_parser().parse_args([]).dpi_aware)

    def test_the_flag_parses(self) -> None:
        self.assertTrue(DP.build_parser().parse_args(["--dpi-aware"]).dpi_aware)

    def test_declared_before_the_profile_or_any_window(self) -> None:
        order: list[str] = []
        from gazelink_core.platform import screen_check as SC

        with mock.patch.object(
            SC, "ensure_per_monitor_dpi_aware",
            side_effect=lambda: (order.append("dpi"), (True, "fake"))[1],
        ), mock.patch.object(
            DP.L, "resolve_profile", side_effect=RuntimeError("stop here")
        ) as resolve:
            resolve.side_effect = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError(order.append("profile") or "stop")
            )
            with self.assertRaises(RuntimeError):
                DP.main(["--dpi-aware", "--layout", "zones"])
        self.assertEqual(order, ["dpi", "profile"])

    def test_not_declared_without_the_flag(self) -> None:
        from gazelink_core.platform import screen_check as SC

        with (
            mock.patch.object(SC, "ensure_per_monitor_dpi_aware") as dpi,
            mock.patch.object(DP.L, "resolve_profile", side_effect=RuntimeError("stop")),
            self.assertRaises(RuntimeError),
        ):
            DP.main(["--layout", "zones"])
        dpi.assert_not_called()


if __name__ == "__main__":
    unittest.main()
