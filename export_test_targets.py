#!/usr/bin/env python
"""Export the seeded held-out test targets as a standalone JSON file.

This exists so a fully separate, external comparison (e.g. scoring another
gaze library against the same held-out points) can use the EXACT same
targets -- same seed, same generator, same exclusion zone around the real
9-point calibration grid -- without that external code importing this
project's package at all. It calls the one real target generator used by
``--gaze-test`` (``gazelink.test_points.generate_test_targets``); it does not
define a second copy of the target-selection logic.

It is read-only except for writing the one output file requested via --out.
It does not calibrate, train, or touch the calibration/mapping/gaze code.

The output JSON's "targets"/"screen_geometry" shape is exactly what
analyze.py's --predictions mode expects a predictions file to also carry
(with "samples" added by whatever captured the external predictions) -- so
this file can be extended in place rather than re-shaped.

Usage:
    python export_test_targets.py --out targets.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze import SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX

from gazelink.domain import ScreenGeometry
from gazelink.test_points import DEFAULT_TEST_SEED, TEST_POINT_COUNT, generate_test_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, type=Path, help="output JSON path")
    parser.add_argument("--screen-id", default="external-comparison")
    parser.add_argument(
        "--width-px",
        type=int,
        default=SCREEN_WIDTH_PX,
        help="must match the actual display the external capture runs on "
        "(defaults to analyze.py's SCREEN_WIDTH_PX)",
    )
    parser.add_argument(
        "--height-px",
        type=int,
        default=SCREEN_HEIGHT_PX,
        help="must match the actual display the external capture runs on "
        "(defaults to analyze.py's SCREEN_HEIGHT_PX)",
    )
    parser.add_argument("--dpi-scale", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TEST_SEED,
        help="must match the seed used for our own --gaze-test run to be comparable",
    )
    parser.add_argument("--count", type=int, default=TEST_POINT_COUNT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    geometry = ScreenGeometry(
        screen_id=args.screen_id,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi_scale=args.dpi_scale,
    )
    targets = generate_test_targets(geometry, seed=args.seed, count=args.count)
    payload = {
        "screen_geometry": geometry.to_dict(),
        "targets": [
            {
                "index": index,
                "name": target.name,
                "screen_position": target.screen_position.to_dict(),
            }
            for index, target in enumerate(targets)
        ],
    }
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(targets)} targets (seed={args.seed}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
