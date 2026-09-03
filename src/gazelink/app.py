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
        "--engine",
        choices=ENGINE_CHOICES,
        default=NATIVE_ENGINE,
        help=(
            "which gaze engine produces the screen point (default: native). "
            "'eyegestures' routes prediction through the optional external "
            "EyeGestures library instead of this project's model; it runs its own "
            "calibration, is currently supported only with --gaze-check, and needs "
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


def cli(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; a camera opens only for an explicit live-mode flag."""

    parser = build_parser()
    args = parser.parse_args(argv)
    # An alternative engine is only wired into --gaze-check so far. Fail loudly
    # rather than silently running the native engine under an --engine flag the
    # user believed had taken effect.
    if args.engine != NATIVE_ENGINE and not args.gaze_check:
        parser.error(f"--engine {args.engine} is currently supported only with --gaze-check")
    if (
        args.debug_overlay
        or args.guided_calibration
        or args.gaze_check
        or args.gaze_validation
        or args.gaze_feature_check
        or args.gaze_test
    ):
        if args.camera_index < 0:
            raise SystemExit("--camera-index must be non-negative")
        if args.guided_calibration:
            from gazelink.calibration_window import run_guided_calibration

            return run_guided_calibration(
                camera_index=args.camera_index, overlay_model_path=args.overlay_model
            )
        if args.gaze_test:
            from gazelink.test_window import run_gaze_test

            return run_gaze_test(
                camera_index=args.camera_index,
                overlay_model_path=args.overlay_model,
                seed=args.test_seed,
                point_count=args.test_points,
            )
        if args.gaze_feature_check:
            from gazelink.feature_check_window import run_gaze_feature_check

            return run_gaze_feature_check(camera_index=args.camera_index)
        if args.gaze_check:
            from gazelink.gaze_window import run_gaze_check

            return run_gaze_check(camera_index=args.camera_index, engine=args.engine)
        if args.gaze_validation:
            from gazelink.validation_window import run_gaze_validation

            return run_gaze_validation(camera_index=args.camera_index)
        # Imported here so the default path never loads Qt, OpenCV, or MediaPipe.
        from gazelink.debug_window import run_debug_overlay

        return run_debug_overlay(camera_index=args.camera_index)
    status = run(smoke=args.smoke)
    print(json.dumps(asdict(status), sort_keys=True))
    return 0
