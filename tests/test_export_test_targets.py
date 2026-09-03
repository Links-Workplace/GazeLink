"""Deterministic tests for export_test_targets.py: no camera, no Qt.

This script only calls gazelink.test_points.generate_test_targets() and
writes its result to disk, so these tests mostly confirm the CLI wiring and
the output shape -- the target-selection math itself is covered by
tests/test_test_points.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import export_test_targets
import pytest

from gazelink.domain import ScreenGeometry
from gazelink.test_points import DEFAULT_TEST_SEED, generate_test_targets

pytestmark = pytest.mark.unit


def test_main_writes_the_same_targets_generate_test_targets_would_produce(tmp_path: Path) -> None:
    """4096x1152 (the reference display) -- the separation constants in
    test_points.py are tuned for it; a tiny geometry can't fit 10 points."""

    out_path = tmp_path / "targets.json"
    exit_code = export_test_targets.main(
        ["--out", str(out_path), "--width-px", "4096", "--height-px", "1152", "--seed", "42"]
    )
    assert exit_code == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    geometry = ScreenGeometry.from_dict(payload["screen_geometry"])
    expected = generate_test_targets(geometry, seed=42)

    assert len(payload["targets"]) == len(expected)
    for index, target in enumerate(expected):
        entry = payload["targets"][index]
        assert entry["index"] == index
        assert entry["name"] == target.name
        assert entry["screen_position"]["x"] == pytest.approx(target.screen_position.x)
        assert entry["screen_position"]["y"] == pytest.approx(target.screen_position.y)


def test_default_seed_matches_the_gaze_test_default(tmp_path: Path) -> None:
    """No --seed given must reproduce the exact points --gaze-test uses by default."""

    out_path = tmp_path / "targets.json"
    export_test_targets.main(["--out", str(out_path), "--width-px", "4096", "--height-px", "1152"])
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["screen_geometry"]["width_px"] == 4096
    assert payload["screen_geometry"]["height_px"] == 1152

    geometry = ScreenGeometry.from_dict(payload["screen_geometry"])
    expected = generate_test_targets(geometry, seed=DEFAULT_TEST_SEED)
    assert len(payload["targets"]) == len(expected)
    assert payload["targets"][0]["screen_position"]["x"] == pytest.approx(
        expected[0].screen_position.x
    )


def test_count_flag_changes_the_number_of_targets_written(tmp_path: Path) -> None:
    out_path = tmp_path / "targets.json"
    export_test_targets.main(
        ["--out", str(out_path), "--width-px", "4096", "--height-px", "1152", "--count", "5"]
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["targets"]) == 5
