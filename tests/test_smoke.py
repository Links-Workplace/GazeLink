from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gazelink.app import build_parser, cli, run


def test_safe_bootstrap_disables_camera_and_os_input() -> None:
    status = run()

    assert status.mode == "safe-bootstrap"
    assert status.camera_enabled is False
    assert status.os_input_enabled is False


def test_smoke_cli_reports_safe_state(capsys: object) -> None:
    assert cli(["--smoke"]) == 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload == {
        "app": "GAZELINK",
        "camera_enabled": False,
        "mode": "smoke",
        "os_input_enabled": False,
        "version": "0.1.0",
    }


def test_default_and_smoke_paths_never_import_a_camera_or_ui_stack() -> None:
    """Opening a camera must require an explicit flag, never a default run.

    This runs in a subprocess on purpose: other tests in this session import
    MediaPipe and OpenCV, so an in-process check of ``sys.modules`` would pass
    for the wrong reason and could never fail.
    """

    probe = (
        "import sys;"
        "from gazelink.app import cli;"
        "cli(['--smoke']);"
        "cli([]);"
        "leaked=[m for m in ('cv2','mediapipe','PySide6','gazelink.debug_window',"
        "'gazelink.calibration_window')"
        " if m in sys.modules];"
        "sys.exit('imported without --debug-overlay: ' + ','.join(leaked) if leaked else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_debug_overlay_is_opt_in_and_rejects_an_invalid_camera_index() -> None:
    parser = build_parser()

    assert parser.parse_args([]).debug_overlay is False
    assert parser.parse_args(["--debug-overlay"]).debug_overlay is True
    assert parser.parse_args(["--guided-calibration"]).guided_calibration is True
    assert parser.parse_args(["--gaze-check"]).gaze_check is True
    assert parser.parse_args(["--gaze-validation"]).gaze_validation is True
    assert parser.parse_args(["--gaze-feature-check"]).gaze_feature_check is True
    assert parser.parse_args(["--debug-overlay"]).camera_index == 0
    with pytest.raises(SystemExit):
        cli(["--debug-overlay", "--camera-index", "-1"])
