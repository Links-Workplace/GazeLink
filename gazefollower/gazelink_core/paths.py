"""Where the gazefollower package lives on disk.

Modules that moved into gazelink_core still read and write the same
directories they did as ``gf_*.py`` files beside the tools (profiles,
targets, results), so they resolve them from here, not from ``__file__``.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
