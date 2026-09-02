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
        "--camera-index",
        type=int,
        default=0,
        help="camera index for live debug, calibration, or gaze-check modes (default: 0)",
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

    args = build_parser().parse_args(argv)
    if (
        args.debug_overlay
        or args.guided_calibration
        or args.gaze_check
        or args.gaze_validation
        or args.gaze_feature_check
    ):
        if args.camera_index < 0:
            raise SystemExit("--camera-index must be non-negative")
        if args.guided_calibration:
            from gazelink.calibration_window import run_guided_calibration

            return run_guided_calibration(camera_index=args.camera_index)
        if args.gaze_feature_check:
            from gazelink.feature_check_window import run_gaze_feature_check

            return run_gaze_feature_check(camera_index=args.camera_index)
        if args.gaze_check:
            from gazelink.gaze_window import run_gaze_check

            return run_gaze_check(camera_index=args.camera_index)
        if args.gaze_validation:
            from gazelink.validation_window import run_gaze_validation

            return run_gaze_validation(camera_index=args.camera_index)
        # Imported here so the default path never loads Qt, OpenCV, or MediaPipe.
        from gazelink.debug_window import run_debug_overlay

        return run_debug_overlay(camera_index=args.camera_index)
    status = run(smoke=args.smoke)
    print(json.dumps(asdict(status), sort_keys=True))
    return 0
