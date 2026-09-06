from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gazelink.app import build_parser, cli, run, selected_screen


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


def test_engine_defaults_to_native_everywhere() -> None:
    """Omitting --engine must never change which engine runs."""

    parser = build_parser()

    assert parser.parse_args([]).engine == "native"
    assert parser.parse_args(["--gaze-check"]).engine == "native"
    assert parser.parse_args(["--debug-overlay"]).engine == "native"
    assert parser.parse_args(["--gaze-check", "--engine", "eyegestures"]).engine == "eyegestures"


def test_unknown_engine_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--gaze-check", "--engine", "tobii"])


@pytest.mark.parametrize(
    "flag",
    # --gaze-test is absent: it now drives an engine itself, so the flag is
    # honoured there rather than rejected.
    ["--debug-overlay", "--guided-calibration", "--gaze-validation"],
)
def test_alternative_engine_fails_loudly_where_it_is_not_wired(flag: str) -> None:
    """An --engine the mode ignores must error, not silently run native."""

    with pytest.raises(SystemExit) as excinfo:
        cli([flag, "--engine", "eyegestures"])
    assert excinfo.value.code == 2


# --- one ordering for the engine guard and the dispatch ---------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--gaze-check"], "gaze_check"),
        (["--gaze-test"], "gaze_test"),
        (["--guided-calibration"], "guided_calibration"),
        (["--smoke"], None),
        ([], None),
        # The dispatch has always preferred --guided-calibration over
        # --gaze-check; selected_screen must report the winner, not the flag
        # the user may have thought would take effect.
        (["--gaze-check", "--guided-calibration"], "guided_calibration"),
        (["--gaze-check", "--gaze-test"], "gaze_test"),
        (["--gaze-check", "--gaze-feature-check"], "gaze_feature_check"),
    ],
)
def test_selected_screen_names_the_screen_that_actually_wins(
    argv: list[str], expected: str | None
) -> None:
    assert selected_screen(build_parser().parse_args(argv)) == expected


@pytest.mark.parametrize(
    "companion",
    # Only screens that OUTRANK --gaze-check in _SCREEN_ORDER can steal the
    # dispatch from it. Pairing with a lower-ranked screen (--gaze-validation)
    # legitimately runs --gaze-check, which does honour --engine, so it is not
    # part of this regression and must not be asserted to error.
    # --gaze-test outranks --gaze-check too, but it is engine-aware, so that
    # pairing legitimately runs it rather than erroring.
    ["--guided-calibration", "--gaze-feature-check"],
)
def test_engine_guard_cannot_be_slipped_past_by_pairing_flags(companion: str) -> None:
    """Regression: this combination used to run NATIVE calibration silently.

    The guard tested only ``not args.gaze_check`` while the dispatch checked
    --guided-calibration first, so passing both satisfied the guard and then
    opened a screen that ignores --engine entirely. The user would have
    believed EyeGestures was driving a calibration it never touched.

    This test asserts on the guard alone: it must reject BEFORE any camera is
    opened, so a failure here can never turn into a live capture session.
    """

    with pytest.raises(SystemExit) as excinfo:
        cli(["--gaze-check", companion, "--engine", "eyegestures"])
    assert excinfo.value.code == 2


# --- the measured screen stays free of smoothing and correction -------------


def test_the_held_out_test_screen_never_pulls_in_smoothing_or_correction() -> None:
    """The measured path must show and record the engine's own output.

    A One-Euro filter or a local correction on this screen would mean the
    number on screen and the number in the file describe something the engine
    never produced. Today neither module is imported; this pins that, because
    an accidental import is a one-line change with no other visible symptom.

    Subprocess for the same reason as the probe above: other tests import these
    modules, so an in-process ``sys.modules`` check could never fail.
    """

    probe = (
        "import sys;"
        "import gazelink.test_window;"
        "leaked=[m for m in ('gazelink.gaze_filter','gazelink.gaze_correction')"
        " if m in sys.modules];"
        "sys.exit('measured screen imported: ' + ','.join(leaked) if leaked else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_no_smoothing_flag_is_off_by_default_and_parses() -> None:
    """Smoothing stays on unless explicitly disabled for a measurement."""

    assert build_parser().parse_args(["--gaze-check"]).no_smoothing is False
    assert build_parser().parse_args(["--gaze-check", "--no-smoothing"]).no_smoothing is True
