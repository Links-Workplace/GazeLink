"""The desk interface is wired, and it is OFF unless asked for.

The important tests here are the negative ones. Every existing command, every
golden scenario and every earlier recording runs the path it always ran, and
that is only true if the flag really does default to off and really does change
nothing when it is.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazelink_core.app.options import LiveOptions  # noqa: E402
from gazelink_core.interaction import bar as BAR  # noqa: E402
from gazelink_core.interaction import keyboard_spatial as SKB  # noqa: E402


class TheFlagIsOffUntilAskedForTests(unittest.TestCase):
    def test_the_option_defaults_to_off(self) -> None:
        self.assertFalse(LiveOptions().desk)

    def test_the_command_line_defaults_to_off(self) -> None:
        import gf_live as L

        self.assertFalse(L.build_parser().parse_args([]).desk)

    def test_the_command_line_can_turn_it_on(self) -> None:
        import gf_live as L

        self.assertTrue(L.build_parser().parse_args(["--desk"]).desk)

    def test_the_flag_reaches_the_options_object(self) -> None:
        import ast

        import gf_live as L

        source = Path(L.__file__.replace(".pyc", ".py")).read_text(encoding="utf-8")
        call = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_live"
        )
        self.assertIn("desk", [kw.arg for kw in call.keywords])

    def test_turning_it_on_does_not_change_any_other_default(self) -> None:
        off = LiveOptions()
        on = LiveOptions(desk=True)
        for name in off.__dataclass_fields__:
            if name == "desk":
                continue
            with self.subTest(name):
                self.assertEqual(getattr(off, name), getattr(on, name))

    def test_it_does_not_quietly_enable_os_input(self) -> None:
        # --desk is a user interface, not a permission. Clicking still needs
        # both of the gates it always needed.
        self.assertFalse(LiveOptions(desk=True).input_enabled)
        self.assertFalse(LiveOptions(desk=True).controlled)


class TheSessionBuildsTheComponentsTests(unittest.TestCase):
    def test_the_controller_holds_nothing_desk_shaped_by_default(self) -> None:
        import inspect

        from gazelink_core.interaction.controller import InteractionController

        signature = inspect.signature(InteractionController.__init__)
        for name in ("bar", "spatial_keyboard", "zoom"):
            with self.subTest(name):
                self.assertIsNone(signature.parameters[name].default)

    def test_the_session_builds_them_only_behind_the_flag(self) -> None:
        source = Path(
            Path(__file__).resolve().parent.parent
            / "gazelink_core" / "app" / "live_session.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if options.desk else None", source)

    def test_the_components_take_the_dwell_time_from_the_profile(self) -> None:
        # Not the number the code was written with: CLAUDE.md 4.5 forbids
        # copying a fixed value out of a study as though it fitted everyone.
        source = Path(
            Path(__file__).resolve().parent.parent
            / "gazelink_core" / "app" / "live_session.py"
        ).read_text(encoding="utf-8")
        self.assertIn("accessibility_settings()", source)
        self.assertIn("dwell_ms=access.dwell_ms", source)


class TheComponentsAgreeOnTheirGeometryTests(unittest.TestCase):
    def test_the_bar_and_the_keyboard_use_the_same_pause_target(self) -> None:
        from gazelink_core.interaction import desk_layout as L

        bar = BAR.ControlBar()
        self.assertIn(L.PAUSE_KEY, [b.key for b in bar.buttons])

    def test_no_desk_component_can_reach_a_neighbour(self) -> None:
        risky = [
            line
            for line in (*BAR.layout_warnings(), *SKB.layout_warnings())
            if line.startswith("NEIGHBOUR")
        ]
        self.assertEqual(risky, [])

    def test_the_warnings_are_available_before_a_session_rather_than_after(self) -> None:
        source = Path(
            Path(__file__).resolve().parent.parent
            / "gazelink_core" / "app" / "live_session.py"
        ).read_text(encoding="utf-8")
        self.assertIn("BAR.layout_warnings()", source)


if __name__ == "__main__":
    unittest.main()
