"""The real-input interlock is on by default, in any process (ARCH-01 amendment 1).

No test here reaches Windows: the SendInput/SetCursorPos entry points are
replaced at the ctypes layer, BELOW the guard, so what is asserted is exactly
which calls the guard lets through to it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
# ``fake_live_env`` is a sibling helper. Discovery puts this directory on the
# path; running this module on its own does not, so say it here.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_click as CK  # noqa: E402
import gf_cursor as CUR  # noqa: E402
import gf_keys as KEYS  # noqa: E402
from gazelink_core.platform import real_input as RI  # noqa: E402


class _FakeUser32:
    def __init__(self) -> None:
        self.inputs = 0
        self.moves: list[tuple[int, int]] = []

    def SendInput(self, count, pointer, size):  # noqa: ANN001, N802
        self.inputs += 1
        return count

    def SetCursorPos(self, x, y):  # noqa: ANN001, N802
        self.moves.append((x, y))
        return 1


def _fake_windows(test: unittest.TestCase) -> _FakeUser32:
    user32 = _FakeUser32()
    for module in (CK, KEYS, CUR):
        original = module.ctypes
        fake = SimpleNamespace(
            **{k: getattr(original, k) for k in dir(original) if not k.startswith("__")}
        )
        fake.windll = SimpleNamespace(user32=user32)
        module.ctypes = fake
        test.addCleanup(setattr, module, "ctypes", original)
    return user32


class FreshProcessTests(unittest.TestCase):
    def test_a_new_interpreter_refuses_every_real_sender(self) -> None:
        """No test fixture, no conftest, no environment variable: disarmed."""

        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(HERE)!r})
            import gf_click as CK, gf_keys as KEYS, gf_cursor as CUR
            from gazelink_core.platform.real_input import RealInputNotArmed
            refused = 0
            for call in (
                lambda: CK._send(CK.MOUSEEVENTF_LEFTDOWN),
                lambda: CK._send(CK.MOUSEEVENTF_RIGHTDOWN),
                lambda: CK._send_wheel(120),
                lambda: KEYS._send_key(0x41, 0, 0),
                lambda: CUR._set_cursor_pos(5, 5),
            ):
                try:
                    call()
                except RealInputNotArmed:
                    refused += 1
            print(refused)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "5")


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertFalse(RI.is_armed(), "a previous test left real input armed")

    def test_presses_moves_and_wheel_are_refused_before_windows_sees_them(self) -> None:
        user32 = _fake_windows(self)
        for call in (
            lambda: CK._send(CK.MOUSEEVENTF_LEFTDOWN),
            lambda: CK._send_wheel(-120),
            lambda: KEYS._send_key(KEYS.VK_MENU, 0, 0),
            lambda: CUR._set_cursor_pos(1, 1),
        ):
            with self.assertRaises(RI.RealInputNotArmed):
                call()
        self.assertEqual(user32.inputs, 0)
        self.assertEqual(user32.moves, [])

    def test_releases_always_reach_windows(self) -> None:
        """A held button or key must be undoable after the armed scope closed."""

        user32 = _fake_windows(self)
        CK._send(CK.MOUSEEVENTF_LEFTUP)
        CK._send(CK.MOUSEEVENTF_RIGHTUP)
        KEYS._send_key(KEYS.VK_MENU, 0, KEYS.KEYEVENTF_KEYUP)
        self.assertEqual(user32.inputs, 3)

    def test_armed_scope_allows_then_disarms_even_on_error(self) -> None:
        user32 = _fake_windows(self)
        with self.assertRaises(ValueError), RI.armed("test scope"):
            CK._send(CK.MOUSEEVENTF_LEFTDOWN)
            CUR._set_cursor_pos(2, 3)
            self.assertEqual(RI.reason(), "test scope")
            raise ValueError("boom")
        self.assertFalse(RI.is_armed())
        self.assertEqual((user32.inputs, user32.moves), (1, [(2, 3)]))
        with self.assertRaises(RI.RealInputNotArmed):
            CK._send(CK.MOUSEEVENTF_LEFTDOWN)

    def test_nested_scopes_disarm_only_at_the_outermost_exit(self) -> None:
        with RI.armed("outer"):
            with RI.armed("inner"):
                self.assertTrue(RI.is_armed())
            self.assertTrue(RI.is_armed())
            self.assertEqual(RI.reason(), "outer")
        self.assertFalse(RI.is_armed())

    def test_arming_needs_a_reason(self) -> None:
        with self.assertRaises(ValueError), RI.armed(""):
            pass


class ArmingPathTests(unittest.TestCase):
    """Who may arm, and when (ARCH-01 review of stages C-D, B2)."""

    def setUp(self) -> None:
        self.assertFalse(RI.is_armed())

    def _profile(self):  # noqa: ANN202
        import gf_profile as PROF  # noqa: PLC0415

        return PROF.Profile(
            name="t",
            model_dir=str(Path(__file__).resolve().parent),
            rig={
                "camera_x_cm": 60.0,
                "camera_y_cm": 63.6,
                "screen_w_cm": 120.0,
                "screen_h_cm": 33.75,
                "device_w_px": 5120,
                "device_h_px": 1440,
            },
            filter={"kind": "one-euro"},
        )

    def test_run_live_refuses_real_senders_when_nothing_armed(self) -> None:
        import dataclasses  # noqa: PLC0415

        import gf_live as L  # noqa: PLC0415

        from fake_live_env import FakeWorld  # noqa: PLC0415

        world = FakeWorld(runner=None, display=None)
        env = dataclasses.replace(world.environment(), real_input=True)
        with self.assertRaises(SystemExit) as caught:
            L.run_live(
                self._profile(),
                move_cursor=True,
                confirmed=True,
                click_by="wink",
                skip_model_check=True,
                env=env,
            )
        self.assertIn("not armed", str(caught.exception))
        self.assertEqual(world.calls["open_library"], 0, "the library opened before refusing")

    def _live_main(self, argv: list[str]) -> list[bool]:
        import gf_live as L  # noqa: PLC0415

        seen: list[bool] = []
        saved = (L.resolve_profile, L.check_rig, L._run_live_from_args)
        L.resolve_profile = lambda name: self._profile()
        L.check_rig = lambda *a, **kw: None
        L._run_live_from_args = lambda profile, monitor, args: (seen.append(RI.is_armed()), 0)[1]
        try:
            L.main(argv)
        finally:
            L.resolve_profile, L.check_rig, L._run_live_from_args = saved
        return seen

    def test_gf_live_main_arms_only_with_both_flags_and_disarms_after(self) -> None:
        self.assertEqual(self._live_main(["--move-cursor", "--i-mean-it"]), [True])
        self.assertFalse(RI.is_armed(), "arming outlived the run")
        self.assertEqual(self._live_main(["--move-cursor"]), [False])
        self.assertEqual(self._live_main(["--i-mean-it"]), [False])
        self.assertEqual(self._live_main([]), [False])

    def test_click_practice_main_arms_only_with_both_flags(self) -> None:
        import gf_click_practice as P  # noqa: PLC0415

        seen: list[bool] = []
        saved = (P.L.resolve_profile, P._practice_from_args)
        P.L.resolve_profile = lambda name: self._profile()
        P._practice_from_args = lambda profile, args: seen.append(RI.is_armed())
        try:
            P.main(["--click", "--i-mean-it"])
            P.main(["--click"])
            P.main([])
        finally:
            P.L.resolve_profile, P._practice_from_args = saved
        self.assertEqual(seen, [True, False, False])
        self.assertFalse(RI.is_armed())


if __name__ == "__main__":
    unittest.main()
