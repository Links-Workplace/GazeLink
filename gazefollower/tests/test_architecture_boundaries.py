"""The structure ARCH-01 builds must stay built (TECHNICAL_SPEC 4.1, 4.4, 17).

Read-only checks over the source tree with ``ast``: comments and docstrings
cannot trip them, code can. Each rule names what it protects.
"""

from __future__ import annotations

import ast
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "gazelink_core"
sys.path.insert(0, str(ROOT))

# Attribute names that exist only on the gazefollower library's frame objects.
LIBRARY_FIELDS = {
    "left_eye_openness",
    "right_eye_openness",
    "face_landmarks",
    "raw_gaze_coordinates",
    "can_gaze_estimation",
    "face_rect",
    "left_rect",
    "right_rect",
}
MOVED_ALIASES = {
    "gf_common": "gazelink_core.domain.common",
    "gf_head_features": "gazelink_core.gaze.head_features",
    "gf_gaze_filter": "gazelink_core.gaze.gaze_filter",
    "gf_schema": "gazelink_core.calibration.schema",
    "gf_targets": "gazelink_core.calibration.targets",
    "gf_profile": "gazelink_core.calibration.profile",
    "gf_gesture": "gazelink_core.interaction.gesture",
    "gf_dwell": "gazelink_core.interaction.dwell",
    "gf_scroll": "gazelink_core.interaction.scroll",
    "gf_menu": "gazelink_core.interaction.menu",
    "gf_keyboard": "gazelink_core.interaction.keyboard",
    "gf_actions": "gazelink_core.interaction.actions",
    "gf_control": "gazelink_core.interaction.control",
    "gf_click": "gazelink_core.platform.click",
    "gf_keys": "gazelink_core.platform.keys",
    "gf_cursor": "gazelink_core.platform.cursor",
    "gf_overlay": "gazelink_core.platform.overlay",
    "gf_display": "gazelink_core.platform.display",
    "gf_screen_check": "gazelink_core.platform.screen_check",
    "gf_capture": "gazelink_core.tracking.capture",
}


def _modules(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _library_field_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in LIBRARY_FIELDS:
            hits.append(f"{path.name}:{node.lineno} .{node.attr}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in LIBRARY_FIELDS
        ):
            hits.append(f"{path.name}:{node.lineno} getattr {node.args[1].value}")
    return hits


def _receiver_name(node: ast.expr) -> str | None:
    """``cursor`` for ``cursor.x``, ``self.cursor.x`` and ``s.executor.cursor.x`` alike."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


class CoreDependsOnNoToolTests(unittest.TestCase):
    def test_the_core_imports_no_gf_tool_module(self) -> None:
        bad = [
            f"{_rel(p)} -> {name}"
            for p in _modules(CORE)
            for name in _imports(p)
            if name.split(".")[0].startswith("gf_")
        ]
        self.assertEqual(bad, [], "the core reaches into a tool")

    def test_the_core_has_no_command_line_and_no_path_hacks(self) -> None:
        bad = []
        for path in _modules(CORE):
            text = path.read_text(encoding="utf-8")
            if "argparse" in _imports(path):
                bad.append(f"{_rel(path)} imports argparse")
            if "sys.path.insert" in text:
                bad.append(f"{_rel(path)} edits sys.path")
        self.assertEqual(bad, [])

    def test_the_live_application_imports_neither_recorder_nor_fitting_tool(self) -> None:
        names = {n.split(".")[0] for n in _imports(ROOT / "gf_live.py")}
        self.assertFalse(names & {"gf_record", "gf_fit", "gf_setup", "gf_report"}, names)


class LayeringTests(unittest.TestCase):
    def test_domain_and_interaction_touch_no_device_window_or_library(self) -> None:
        # PIL joins them: screen capture for the magnifier is a PLATFORM
        # concern, and interaction/ must hold only the port, never the library.
        forbidden_roots = {"pygame", "ctypes", "cv2", "gazefollower", "PIL", "Pillow"}
        forbidden_layers = (
            "gazelink_core.tracking",
            "gazelink_core.ui",
            "gazelink_core.platform",
            "gazelink_core.app",
        )
        bad = []
        for folder in (CORE / "domain", CORE / "interaction"):
            for path in _modules(folder):
                for name in _imports(path):
                    if name.split(".")[0] in forbidden_roots or name.startswith(forbidden_layers):
                        bad.append(f"{_rel(path)} -> {name}")
        self.assertEqual(bad, [])

    def test_gaze_does_not_depend_on_ui_app_or_platform(self) -> None:
        bad = []
        for path in _modules(CORE / "gaze"):
            for name in _imports(path):
                if name.startswith(
                    ("gazelink_core.ui", "gazelink_core.app", "gazelink_core.platform")
                ):
                    bad.append(f"{_rel(path)} -> {name}")
                # One documented exception: head_features resolves its old
                # library-facing names lazily from the adapter, for tools.
                if name.startswith("gazelink_core.tracking") and path.name != "head_features.py":
                    bad.append(f"{_rel(path)} -> {name}")
        self.assertEqual(bad, [])


class LibraryStructuresStayInTheAdapterTests(unittest.TestCase):
    def test_no_library_frame_field_is_read_outside_the_tracking_adapter(self) -> None:
        paths = [p for p in _modules(CORE) if "tracking" not in p.relative_to(CORE).parts]
        # Every gf_* tool too, except the standalone scoring script that runs
        # outside the package by design (ADR-0002).
        paths += [p for p in sorted(ROOT.glob("gf_*.py"))]
        bad = [hit for p in paths for hit in _library_field_uses(p)]
        self.assertEqual(bad, [])


class OwnershipTests(unittest.TestCase):
    def test_the_application_loop_lives_only_in_the_session_and_is_short(self) -> None:
        live = ast.parse((ROOT / "gf_live.py").read_text(encoding="utf-8"))
        self.assertFalse(
            any(isinstance(n, ast.While) for n in ast.walk(live)),
            "gf_live grew a loop of its own again",
        )
        session = ast.parse((CORE / "app" / "live_session.py").read_text(encoding="utf-8"))
        loop = next(
            n for n in ast.walk(session) if isinstance(n, ast.FunctionDef) and n.name == "_loop"
        )
        self.assertLessEqual(
            loop.end_lineno - loop.lineno, 30, "the loop is deciding, not orchestrating"
        )
        run_live = next(
            n for n in live.body if isinstance(n, ast.FunctionDef) and n.name == "run_live"
        )
        self.assertLessEqual(len(run_live.body), 3, "run_live is no longer a thin wrapper")

    def test_only_the_executor_calls_the_input_adapters_in_a_session(self) -> None:
        """Sending methods on click/key/scroll adapters are called from the executor only."""

        senders = {
            "click",
            "double_click",
            "right_click",
            "scroll",
            "type_text",
            "backspace",
            "enter",
            "back",
            "forward",
            "switch_window",
            "escape",
            "jump_to",
            "update",
            # The drag surface. Without these two names here the ownership rule
            # would silently stop covering the one path that can leave a
            # mouse button physically held down.
            "press",
            "drag_release",
        }
        receivers = {"clicker", "scroller", "keys", "cursor"}
        bad = []
        scope = _modules(CORE / "interaction") + _modules(CORE / "app") + _modules(CORE / "ui")
        for path in scope + [ROOT / "gf_live.py"]:
            if path.name == "executor.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in senders
                    and _receiver_name(node.func.value) in receivers
                ):
                    receiver = _receiver_name(node.func.value)
                    bad.append(f"{_rel(path)}:{node.lineno} {receiver}.{node.func.attr}")
        self.assertEqual(bad, [])

    def test_no_constant_armed_true_is_passed_anywhere_in_the_core(self) -> None:
        bad = []
        for path in _modules(CORE):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if (
                            kw.arg == "armed"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            bad.append(f"{_rel(path)}:{node.lineno}")
        self.assertEqual(bad, [])


class CompatibilityAliasTests(unittest.TestCase):
    def test_every_old_name_is_the_same_module_object(self) -> None:
        for old, new in MOVED_ALIASES.items():
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))


if __name__ == "__main__":
    unittest.main()
