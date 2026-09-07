"""gf_purge deletes the biometric-derived files and refuses anything else."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_purge as P  # noqa: E402


class PlanPurgeTests(unittest.TestCase):
    def _make_tree(self, base: Path, tmp_dir: Path) -> Path:
        root = base / "recordings"
        (root / "round0" / "models" / "lib-default").mkdir(parents=True)
        (root / ".gitignore").write_text("*\n", encoding="utf-8")
        (root / "round0" / "A.npz").write_bytes(b"embedding")
        (root / "round0" / "T1.npz").write_bytes(b"embedding")
        (root / "round0" / "models" / "lib-default" / "svr_x.xml").write_text("<svm/>", encoding="utf-8")
        tmp_csv = tmp_dir / "em_my_session_2026_09_07_10_00_00.csv"
        tmp_csv.write_text("t,x,y\n", encoding="utf-8")
        other_csv = tmp_dir / "em_someone_else.csv"
        other_csv.write_text("t,x,y\n", encoding="utf-8")
        (root / "round0" / "A.meta.json").write_text(
            json.dumps({"recording_format": "rec-1", "round_id": 0, "library_tmp_files": [str(tmp_csv)]}),
            encoding="utf-8",
        )
        return root

    def test_lists_recordings_models_and_only_our_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "eval"
            base.mkdir()
            tmp_dir = Path(tmp) / "GazeFollower" / "tmp"
            tmp_dir.mkdir(parents=True)
            root = self._make_tree(base, tmp_dir)
            original_eval = P.PACKAGE_DIR
            P.PACKAGE_DIR = base
            try:
                under_root, tmp_files, warnings = P.plan_purge(root, library_tmp=tmp_dir)
            finally:
                P.PACKAGE_DIR = original_eval
            names = sorted(p.name for p in under_root)
            self.assertIn("A.npz", names)
            self.assertIn("svr_x.xml", names)  # support vectors ARE training rows
            self.assertIn("A.meta.json", names)
            self.assertNotIn(".gitignore", names)
            self.assertEqual([p.name for p in tmp_files], ["em_my_session_2026_09_07_10_00_00.csv"])
            self.assertEqual(warnings, [])

    def test_unreadable_metadata_is_reported_not_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "eval"
            root = base / "recordings" / "round0"
            root.mkdir(parents=True)
            (root / "A.meta.json").write_text("{not json", encoding="utf-8")
            original_eval = P.PACKAGE_DIR
            P.PACKAGE_DIR = base
            try:
                _, _, warnings = P.plan_purge(base / "recordings")
            finally:
                P.PACKAGE_DIR = original_eval
            self.assertEqual(len(warnings), 1)
            self.assertIn("unreadable metadata", warnings[0])

    def test_refuses_root_that_is_not_named_recordings_or_is_outside_eval_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                P.plan_purge(Path(tmp) / "something")
            outside = Path(tmp) / "recordings"
            outside.mkdir()
            with self.assertRaises(ValueError):
                P.plan_purge(outside)

    def test_missing_root_is_not_an_error(self) -> None:
        original_eval = P.PACKAGE_DIR
        with tempfile.TemporaryDirectory() as tmp:
            P.PACKAGE_DIR = Path(tmp)
            try:
                self.assertEqual(P.plan_purge(Path(tmp) / "recordings"), ([], [], []))
            finally:
                P.PACKAGE_DIR = original_eval


class ExecutePurgeTests(unittest.TestCase):
    def test_deletes_listed_files_keeps_gitignore_and_removes_empty_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "eval"
            root = base / "recordings"
            (root / "round0" / "models").mkdir(parents=True)
            (root / ".gitignore").write_text("*\n", encoding="utf-8")
            f1 = root / "round0" / "A.npz"
            f1.write_bytes(b"x")
            f2 = root / "round0" / "models" / "svr_x.xml"
            f2.write_text("<svm/>", encoding="utf-8")
            original_eval = P.PACKAGE_DIR
            P.PACKAGE_DIR = base
            try:
                deleted = P.execute_purge([f1, f2], [], root)
            finally:
                P.PACKAGE_DIR = original_eval
            self.assertEqual(len(deleted), 2)
            self.assertFalse(f1.exists())
            self.assertFalse(f2.exists())
            self.assertTrue((root / ".gitignore").exists())
            self.assertFalse((root / "round0").exists())
            self.assertTrue(root.exists())

    def test_execute_validates_the_root_itself_not_only_the_cli(self) -> None:
        # Codex OPTIONAL #4: execute_purge trusted whatever root it was given.
        with tempfile.TemporaryDirectory() as tmp:
            stray_root = Path(tmp) / "not_recordings"
            stray_root.mkdir()
            victim = stray_root / "important.txt"
            victim.write_text("keep me", encoding="utf-8")
            with self.assertRaises(ValueError):
                P.execute_purge([victim], [], stray_root)
            self.assertTrue(victim.exists())

    def test_refuses_a_path_outside_both_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "eval"
            root = base / "recordings"
            root.mkdir(parents=True)
            stray = Path(tmp) / "important.txt"
            stray.write_text("keep me", encoding="utf-8")
            original_eval = P.PACKAGE_DIR
            P.PACKAGE_DIR = base
            try:
                with self.assertRaises(ValueError):
                    P.execute_purge([stray], [], root)
            finally:
                P.PACKAGE_DIR = original_eval
            self.assertTrue(stray.exists())


class ImportIsolationTests(unittest.TestCase):
    def test_no_gazefollower_import(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


if __name__ == "__main__":
    unittest.main()
