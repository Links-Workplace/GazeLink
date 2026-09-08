"""Safe application bootstrap for GAZELINK.

The default entry point still opens nothing. A camera is acquired only when the
operator explicitly asks for a live diagnostic or guided calibration, and no
operating-system input adapter exists yet at all -- that arrives with M3.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from gazelink import __version__
from gazelink.gaze_predictor import ENGINE_CHOICES, NATIVE_ENGINE


@dataclass(frozen=True, slots=True)
class BootstrapStatus:
    """Observable state returned by the safe Foundation bootstrap."""

    app: str = "GAZELINK"
    version: str = __version__
    mode: str = "safe-bootstrap"
    camera_enabled: bool = False
    os_input_enabled: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without creating runtime resources."""

    parser = argparse.ArgumentParser(description="GAZELINK safe Foundation bootstrap")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="print safe bootstrap status and exit",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="open the M1 live debug overlay; this is the only flag that opens a camera",
    )
    parser.add_argument(
        "--guided-calibration",
        action="store_true",
        help=(
            "open the M2 full-screen 9-point calibration UI; this opens a camera but never OS input"
        ),
    )
    parser.add_argument(
        "--gaze-check",
        action="store_true",
        help="open the M2 raw gaze diagnostic; requires a compatible saved calibration model",
    )
    parser.add_argument(
        "--gaze-validation",
        action="store_true",
        help=("open fresh M2 validation for a pending calibration candidate; never emits OS input"),
    )
    parser.add_argument(
        "--gaze-feature-check",
        action="store_true",
        help=(
            "open the M2 five-direction feature diagnostic; does not train a model or emit OS input"
        ),
    )
    parser.add_argument(
        "--gaze-test",
        action="store_true",
        help=(
            "open the held-out test-point screen: randomized targets, disjoint from the 9 "
            "calibration targets, for measuring generalization rather than memorization; "
            "never trains, promotes, or emits OS input"
        ),
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=None,
        help="seed for --gaze-test target placement (default: a fixed built-in seed)",
    )
    parser.add_argument(
        "--test-points",
        type=int,
        default=None,
        help="number of held-out test points for --gaze-test (default: 10)",
    )
    parser.add_argument(
        "--overlay-model",
        type=str,
        default=None,
        help=(
            "path to a calibration model JSON file; when given, --guided-calibration and "
            "--gaze-test draw its live prediction next to the target for visual comparison. "
            "Never affects training, promotion, or the saved dataset"
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="camera index for live debug, calibration, or gaze-check modes (default: 0)",
    )
    parser.add_argument(
        "--no-own-vision",
        action="store_true",
        help=(
            "for --gaze-test --engine eyegestures only: skip GAZELINK's own "
            "face-landmark stage, so the camera frame is processed once instead "
            "of twice. Removes the cost of the second model, and with it head "
            "pose and our confidence signal -- run both ways to measure the trade"
        ),
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help=(
            "disable the One-Euro stability filter on --gaze-check, so the "
            "engine's own output is shown unsmoothed. Smoothing hides jitter "
            "and lag, which is exactly what a measurement needs to see"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=ENGINE_CHOICES,
        default=NATIVE_ENGINE,
        help=(
            "which gaze engine produces the screen point (default: native). "
            "'eyegestures' routes prediction through the optional external "
            "EyeGestures library instead of this project's model; it runs its own "
            "calibration, is supported with --gaze-check and --gaze-test, and needs "
            'the optional extra: pip install -e ".[eyegestures]". EyeGestures is '
            "GPL-3.0 licensed -- see TASKS.md before distributing"
        ),
    )
    return parser


def run(*, smoke: bool = False) -> BootstrapStatus:
    """Start and stop the current safe bootstrap.

    ``smoke`` is retained as an explicit signal for automation. Both modes are
    intentionally equivalent until the M1 runtime is introduced.
    """

    mode = "smoke" if smoke else "safe-bootstrap"
    return BootstrapStatus(mode=mode)


# The one ordering that both the engine guard and the dispatch read from.
# They used to be written out twice, in DIFFERENT orders: the guard tested only
# ``not args.gaze_check`` while dispatch checked --guided-calibration first, so
# ``--gaze-check --guided-calibration --engine eyegestures`` passed the guard
# and then silently ran calibration on the NATIVE engine -- exactly the outcome
# the guard exists to prevent. One tuple cannot disagree with itself.
_SCREEN_ORDER: tuple[str, ...] = (
    "guided_calibration",
    "gaze_test",
    "gaze_feature_check",
    "gaze_check",
    "gaze_validation",
    "debug_overlay",
)
# Screens that actually honour --engine. Both drive a predictor directly, so
# the flag reaching them changes what runs; adding a screen here before it can
# honour the flag would make the CLI promise something the code cannot do.
_ENGINE_AWARE_SCREENS = frozenset({"gaze_check", "gaze_test"})


def selected_screen(args: argparse.Namespace) -> str | None:
    """Which live screen this invocation will actually open, or ``None``.

    Pure and argument-only so the guard can ask the same question the dispatch
    will answer, without opening a camera to find out.
    """

    return next((name for name in _SCREEN_ORDER if getattr(args, name, False)), None)


def cli(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; a camera opens only for an explicit live-mode flag."""

    parser = build_parser()
    args = parser.parse_args(argv)
    screen = selected_screen(args)
    # Fail loudly rather than silently running the native engine under an
    # --engine flag the user believed had taken effect. The error names the
    # screen that actually won, not the flag the user may have expected.
    if args.engine != NATIVE_ENGINE and screen not in _ENGINE_AWARE_SCREENS:
        parser.error(
            f"--engine {args.engine} is supported only with --gaze-check or "
            f"--gaze-test; this invocation would open "
            f"{'no live screen' if screen is None else '--' + screen.replace('_', '-')}"
        )
    if screen is not None:
        if args.camera_index < 0:
            raise SystemExit("--camera-index must be non-negative")
        if screen == "guided_calibration":
            from gazelink.calibration_window import run_guided_calibration

            return run_guided_calibration(
                camera_index=args.camera_index, overlay_model_path=args.overlay_model
            )
        if screen == "gaze_test":
            from gazelink.test_window import run_gaze_test

            return run_gaze_test(
                camera_index=args.camera_index,
                overlay_model_path=args.overlay_model,
                seed=args.test_seed,
                point_count=args.test_points,
                engine=args.engine,
                own_vision=not args.no_own_vision,
            )
        if screen == "gaze_feature_check":
            from gazelink.feature_check_window import run_gaze_feature_check

            return run_gaze_feature_check(camera_index=args.camera_index)
        if screen == "gaze_check":
            from gazelink.gaze_window import RECALIBRATE_REQUESTED_EXIT_CODE, run_gaze_check

            # Captures the display the request was made from: after a drag
            # that is not the primary one, and calibrating the wrong monitor
            # while reporting success is worse than refusing outright.
            requested_screen: list[str | None] = []
            result = run_gaze_check(
                camera_index=args.camera_index,
                engine=args.engine,
                smoothing=not args.no_smoothing,
                on_recalibration_request=requested_screen.append,
            )
            if result != RECALIBRATE_REQUESTED_EXIT_CODE:
                return result
            # The display changed and the user asked, by eye gesture, to
            # calibrate for the screen they are now on. Carrying that out is
            # the whole point of offering it: returning the code and stopping
            # would leave a hands-free user with an option that announces a
            # recalibration and never performs one.
            from gazelink.calibration_window import run_guided_calibration

            print("Starting calibration for the current display...")
            return run_guided_calibration(
                camera_index=args.camera_index,
                overlay_model_path=args.overlay_model,
                screen_name=requested_screen[0] if requested_screen else None,
            )
        if screen == "gaze_validation":
            from gazelink.validation_window import run_gaze_validation

            return run_gaze_validation(camera_index=args.camera_index)
        # Imported here so the default path never loads Qt, OpenCV, or MediaPipe.
        from gazelink.debug_window import run_debug_overlay

        return run_debug_overlay(camera_index=args.camera_index)
    status = run(smoke=args.smoke)
    print(json.dumps(asdict(status), sort_keys=True))
    return 0
