"""Test-target sets, and how close each target sits to a calibration point.

Two sets are recorded in every session, because they answer different
questions and neither substitutes for the other:

``GRID16``  the 16-point grid the targets document specifies, X and Y from
            {0.05, 0.35, 0.65, 0.95}. It probes the screen corners and edges,
            which the project's own held-out set deliberately avoids.
``T1``      the project's 10 seeded held-out targets (``targets.json``), kept
            for the generalisation claim and for continuity with the four
            earlier GazeFollower runs.

The reason both are needed: GazeFollower calibrates on x in {0.026, 0.5,
0.974} and y in {0.046, 0.5, 0.954}, and against that grid eight of the
sixteen points are closer than the project's 250 px held-out threshold -- the
four corners at only 98 px. Measuring there is interpolation next to training
points and flatters the result. So proximity is COMPUTED and attached to every
target, the near ones are reported apart from the far ones, and a claim about
generalisation rests on the held-out set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent

# The targets document, section 5.
GRID16_COORDS: tuple[float, ...] = (0.05, 0.35, 0.65, 0.95)
# The project's held-out threshold (gazelink.test_points).
NEAR_CALIBRATION_PX = 250.0
DEFAULT_GRID16_SEED = 20260907


def calibration_points(grid_x: tuple[float, float, float] | None = None) -> list[tuple[float, float]]:
    """The nine calibration positions -- the library's, or a narrower band's.

    Proximity must be measured against the grid a run actually calibrated
    on; a central-band run judged against the library's edge points would
    call every target held-out.
    """

    if grid_x is None:
        return list(C.NINE_POINT_STORED)
    return list(C.nine_point_sequence(grid_x)[1:])


def distance_to_nearest_calibration(
    x: float, y: float, width_px: int, height_px: int, points: Sequence[tuple[float, float]] | None = None
) -> tuple[float, tuple[float, float]]:
    grid = list(points) if points is not None else calibration_points()
    best = min(grid, key=lambda p: C.distance_px(x, y, p[0], p[1], width_px, height_px))
    return C.distance_px(x, y, best[0], best[1], width_px, height_px), best


def build_grid16(
    width_px: int,
    height_px: int,
    *,
    seed: int = DEFAULT_GRID16_SEED,
    coords: Sequence[float] = GRID16_COORDS,
    calibration: Sequence[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """The 16-target grid, shuffled with a recorded seed, proximity attached.

    Order is randomised so a systematic drift over the session cannot line up
    with screen position, and seeded so the order is reproducible.
    """

    points = [(x, y) for y in coords for x in coords]
    rng = random.Random(seed)
    rng.shuffle(points)
    targets: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(points):
        distance, nearest = distance_to_nearest_calibration(x, y, width_px, height_px, calibration)
        targets.append(
            {
                "index": index,
                "name": f"G16_{x:.2f}_{y:.2f}".replace("0.", ""),
                "screen_position": {"x": x, "y": y},
                "distance_to_calibration_px": round(distance, 1),
                "nearest_calibration": {"x": nearest[0], "y": nearest[1]},
                "near_calibration": distance < NEAR_CALIBRATION_PX,
            }
        )
    return targets


def annotate_targets(
    targets: Sequence[Mapping[str, Any]], width_px: int, height_px: int, calibration: Sequence[tuple[float, float]] | None = None
) -> list[dict[str, Any]]:
    """Attach proximity to an existing target list (e.g. the project's T1)."""

    out: list[dict[str, Any]] = []
    for entry in targets:
        x = float(entry["screen_position"]["x"])
        y = float(entry["screen_position"]["y"])
        distance, nearest = distance_to_nearest_calibration(x, y, width_px, height_px, calibration)
        out.append(
            dict(
                entry,
                distance_to_calibration_px=round(distance, 1),
                nearest_calibration={"x": nearest[0], "y": nearest[1]},
                near_calibration=distance < NEAR_CALIBRATION_PX,
            )
        )
    return out


def proximity_summary(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distances = [float(t["distance_to_calibration_px"]) for t in targets]
    near = [t for t in targets if t["near_calibration"]]
    return {
        "n": len(targets),
        "n_near_calibration": len(near),
        "near_names": [str(t["name"]) for t in near],
        "min_px": min(distances) if distances else None,
        "median_px": sorted(distances)[len(distances) // 2] if distances else None,
        "max_px": max(distances) if distances else None,
        "threshold_px": NEAR_CALIBRATION_PX,
        "note": (
            "Targets marked near_calibration sit closer to a calibration point than the "
            "project's held-out threshold. Accuracy there measures interpolation beside "
            "training points, not generalisation, and is reported separately."
        ),
    }


def zone_of(x: float, y: float, coords: Sequence[float] = GRID16_COORDS) -> int:
    """Which of the 16 screen zones a normalised point falls in (row-major).

    Zones are the quarters of each axis, so every target set -- including the
    project's held-out one -- can be aggregated on the same map.
    """

    col = min(int(x * len(coords)), len(coords) - 1)
    row = min(int(y * len(coords)), len(coords) - 1)
    return row * len(coords) + col


def write_targets_file(path: Path, targets: Sequence[Mapping[str, Any]], geometry: Mapping[str, Any], *, seed: int | None = None) -> Path:
    payload: dict[str, Any] = {
        "screen_geometry": dict(geometry),
        "targets": [dict(t) for t in targets],
    }
    if seed is not None:
        payload["seed"] = seed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
