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

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from gazelink_core.domain import common as C  # noqa: E402
from gazelink_core.paths import PACKAGE_DIR  # noqa: E402,F401 - re-exported

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


# The band the system is meant to work in. Same numbers as ``gf_dwell.BAND``,
# repeated here rather than imported: this module is the one every report
# already goes through, and a report must not depend on the dwell engine.
WORKING_BAND_X: tuple[float, float] = (0.30, 0.70)
# How the band is divided for a per-region gate. Three by three is the finest
# split that still leaves enough held-out targets per cell to say anything.
BAND_COLS = 3
BAND_ROWS = 3


def band_zone_of(
    x: float,
    y: float,
    *,
    band: tuple[float, float] = WORKING_BAND_X,
    cols: int = BAND_COLS,
    rows: int = BAND_ROWS,
) -> int | None:
    """Which cell of the band-local grid a point falls in. None if outside.

    ``zone_of`` above divides the WHOLE screen into quarters, which is the
    right map for a full-screen protocol and the wrong one here: the working
    band 0.30..0.70 falls entirely inside two of its four columns, so a
    per-region gate built on it would have two horizontal cells for the whole
    usable area and could not see a regression inside one of them.

    The grid is DELIBERATELY ASYMMETRIC, and getting this wrong would defeat
    the purpose:

    * **x is restricted to the band**, because that is where the system is
      meant to work and the only place a gate has anything to say.
    * **y spans the full height**, because the limit is not the band's. The
      vertical error around y 0.80 is the reported failure -- restricting y to
      0.30..0.70 would exclude exactly the region under investigation.

    Returns None for a point outside the band on x. A caller reporting zones
    must treat that as "not covered by this grid", never as a zero.
    """

    if cols < 1 or rows < 1:
        raise ValueError("a grid needs at least one column and one row")
    lo, hi = band
    if not hi > lo:
        raise ValueError(f"band {band} is empty or inverted")
    if not (lo <= x <= hi):
        return None
    col = min(int((x - lo) / (hi - lo) * cols), cols - 1)
    # Clamped rather than rejected: a prediction slightly off the top or
    # bottom of the screen still belongs to the edge row it is nearest, and
    # dropping it would quietly remove the worst samples from the worst cell.
    row = min(max(int(y * rows), 0), rows - 1)
    return row * cols + col


def band_zone_label(
    zone: int,
    *,
    band: tuple[float, float] = WORKING_BAND_X,
    cols: int = BAND_COLS,
    rows: int = BAND_ROWS,
) -> str:
    """A readable name for a band zone, with the range it covers."""

    lo, hi = band
    row, col = divmod(zone, cols)
    width = (hi - lo) / cols
    x0, x1 = lo + col * width, lo + (col + 1) * width
    y0, y1 = row / rows, (row + 1) / rows
    return f"r{row}c{col} x {x0:.2f}-{x1:.2f} y {y0:.2f}-{y1:.2f}"


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
