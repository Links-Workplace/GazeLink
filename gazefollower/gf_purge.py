"""Delete the experiment's biometric-derived files (M2-05, plan §7).

What gets deleted, and nothing else:
  * everything under ``recordings/`` except its ``.gitignore`` -- per-frame
    model embeddings, head features, and the fitted models whose support
    vectors are those embeddings;
  * the library's own per-session CSV files under ``~/GazeFollower/tmp/``
    that OUR recording sessions created -- identified by the exact paths
    each session wrote into its ``meta.json`` (``library_tmp_files``), never
    by a wildcard over that directory.

``results/`` (predictions JSON, reports) is kept: it holds predictions and
aggregate numbers, not embeddings.

Refuses any path that does not resolve inside the expected roots. Lists
first, deletes only with ``--yes``; ``--dry-run`` never deletes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_DIR / "recordings"
LIBRARY_TMP = Path.home() / "GazeFollower" / "tmp"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_root(root: Path) -> Path:
    """The only directory this script may delete from.

    Checked in one place and called by BOTH planning and execution, so a
    caller that skips planning cannot hand execution an arbitrary path.
    """

    root = Path(root)
    if root.name != "recordings":
        raise ValueError(f"refusing: purge root must be a directory named 'recordings', got {root}")
    if not _inside(root, PACKAGE_DIR):
        raise ValueError(f"refusing: {root} is outside the experiment directory {PACKAGE_DIR}")
    return root


def plan_purge(root: Path, library_tmp: Path = LIBRARY_TMP) -> tuple[list[Path], list[Path], list[str]]:
    """Files to delete: (under root, library tmp files named by our metas, warnings).

    Unreadable metadata is REPORTED, not swallowed: a meta we cannot parse may
    name a session CSV that would otherwise survive the purge.
    """

    root = validate_root(root)
    warnings: list[str] = []
    if not root.exists():
        return [], [], warnings
    under_root: list[Path] = []
    tmp_files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != ".gitignore":
            under_root.append(path)
        if path.is_file() and path.name.endswith(".meta.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                warnings.append(f"unreadable metadata {path}: {exc!r} -- any session CSV it names will remain")
                continue
            for entry in meta.get("library_tmp_files", []) or []:
                candidate = Path(str(entry))
                if (
                    candidate.name.startswith("em_")
                    and candidate.suffix == ".csv"
                    and _inside(candidate, library_tmp)
                    and candidate.exists()
                    and candidate not in tmp_files
                ):
                    tmp_files.append(candidate)
    return under_root, tmp_files, warnings


def execute_purge(under_root: Sequence[Path], tmp_files: Sequence[Path], root: Path) -> list[Path]:
    """Delete the planned files. Re-validates the root itself."""

    root = validate_root(root)
    deleted: list[Path] = []
    for path in list(under_root) + list(tmp_files):
        if path.name == ".gitignore":
            continue
        if not (_inside(path, root) or _inside(path, LIBRARY_TMP)):
            raise ValueError(f"refusing to delete {path}: outside both roots")
        path.unlink()
        deleted.append(path)
    # Remove now-empty directories under root, keep root and its .gitignore.
    for directory in sorted((p for p in Path(root).rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return deleted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--yes", action="store_true", help="actually delete (otherwise list only)")
    parser.add_argument("--dry-run", action="store_true", help="list only, never delete")
    args = parser.parse_args(argv)
    under_root, tmp_files, warnings = plan_purge(args.root)
    print(f"{len(under_root)} file(s) under {args.root}:")
    for path in under_root:
        print(f"  {path}")
    print(f"{len(tmp_files)} library tmp file(s) written by our sessions:")
    for path in tmp_files:
        print(f"  {path}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if args.dry_run or not args.yes:
        print("nothing deleted (pass --yes to delete)")
        return 0
    deleted = execute_purge(under_root, tmp_files, args.root)
    print(f"deleted {len(deleted)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
