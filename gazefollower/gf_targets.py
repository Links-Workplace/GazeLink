"""Compatibility name and command line for ``gazelink_core.calibration.targets`` (ADR-0002).

Imported, ``gf_targets`` IS the core module (same object). Run as a script, it
is the tool: the command line lives here, not in the core.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gazelink_core.calibration import targets as _module
from gazelink_core.domain import common as C
from gazelink_core.paths import PACKAGE_DIR

__doc__ = _module.__doc__
DEFAULT_GRID16_SEED = _module.DEFAULT_GRID16_SEED
build_grid16 = _module.build_grid16
write_targets_file = _module.write_targets_file
proximity_summary = _module.proximity_summary
annotate_targets = _module.annotate_targets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PACKAGE_DIR / "targets_grid16.json")
    parser.add_argument("--geometry-from", type=Path, default=PACKAGE_DIR / "targets.json", help="take screen_geometry from this file")
    parser.add_argument("--seed", type=int, default=DEFAULT_GRID16_SEED)
    parser.add_argument("--also-annotate", type=Path, default=PACKAGE_DIR / "targets.json", help="report proximity for this set too")
    args = parser.parse_args(argv)

    _, geometry = C.load_targets(args.geometry_from)
    width, height = int(geometry["width_px"]), int(geometry["height_px"])
    grid = build_grid16(width, height, seed=args.seed)
    write_targets_file(args.out, grid, geometry, seed=args.seed)
    print(f"wrote {len(grid)} targets (seed={args.seed}) to {args.out}")
    print(json.dumps(proximity_summary(grid), ensure_ascii=False, indent=2))

    if args.also_annotate and args.also_annotate.exists():
        held_out, geom2 = C.load_targets(args.also_annotate)
        annotated = annotate_targets(held_out, int(geom2["width_px"]), int(geom2["height_px"]))
        print(f"\n{args.also_annotate.name} (held-out set):")
        print(json.dumps(proximity_summary(annotated), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
sys.modules[__name__] = _module
