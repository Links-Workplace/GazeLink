"""Score GazeFollower on GAZELINK's held-out points, in its own environment.

Deliberately OUTSIDE the gazelink package and its virtualenv. GazeFollower is
CC BY-NC-SA 4.0; keeping it in a separate environment means the project's own
dependency closure stays free of it, and the licence question -- which is open
and needs the author's answer before any use beyond evaluation -- is not
quietly decided by an import.

It reads targets.json (written by the project's export_test_targets.py, same
seed and same generator as every other engine was measured with) and writes a
predictions JSON that `analyze.py --predictions` scores unchanged. Nothing here
imports gazelink.

What this script does NOT do: display its own calibration. GazeFollower runs
that itself through its own UI. This script drives only the MEASUREMENT phase,
after calibration has completed, so the two are never mixed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

# Held out from GazeFollower's OWN 9-point calibration grid, not ours. Its
# points sit at 0.026/0.5/0.974 on each axis -- a different layout from the
# project's, so reusing the project's default exclusion set would leave test
# points sitting on top of points the model was trained on.
GAZEFOLLOWER_CALIBRATION_POINTS: tuple[tuple[float, float], ...] = tuple(
    (x, y) for x in (0.026, 0.5, 0.974) for y in (0.046, 0.5, 0.954)
)

# Matches the project's own measurement protocol so the numbers are comparable:
# the eyes need time to arrive before anything counts as resting accuracy, and
# the collecting window has to be long enough to average over.
DEFAULT_SETTLE_MS = 1500.0
DEFAULT_COLLECT_MS = 1500.0
DEFAULT_ARRIVAL_RADII_PX: tuple[float, ...] = (100.0, 200.0, 400.0)

TARGET_GLYPH_PX = 96


def _distance_px(ax: float, ay: float, bx: float, by: float, width: int, height: int) -> float:
    dx = (ax - bx) * (width - 1)
    dy = (ay - by) * (height - 1)
    return math.hypot(dx, dy)


def load_targets(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["targets"], payload["screen_geometry"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GazeFollower on the same held-out points every other engine "
            "was measured on, and write a predictions JSON for analyze.py."
        )
    )
    parser.add_argument("--targets", type=Path, required=True, help="targets.json")
    parser.add_argument("--out", type=Path, required=True, help="predictions JSON to write")
    parser.add_argument("--settle-ms", type=float, default=DEFAULT_SETTLE_MS)
    parser.add_argument("--collect-ms", type=float, default=DEFAULT_COLLECT_MS)
    # Mandatory, not defaulted. The library ships a laptop default of
    # (17.15, -0.68) -- centred, just ABOVE the screen. This rig has the camera
    # BELOW the screen, which is the opposite sign on the axis that has failed
    # in every engine measured so far. Letting that default apply silently
    # would hand the model a geometry that is wrong by the full height of the
    # screen, and the run would look like an engine problem.
    parser.add_argument(
        "--camera-x-cm",
        type=float,
        required=True,
        help="camera centre, cm RIGHT of the screen's top-left corner",
    )
    parser.add_argument(
        "--camera-y-cm",
        type=float,
        required=True,
        help=(
            "camera centre, cm DOWN from the screen's top-left corner. "
            "Negative is above the top edge; a value larger than the screen "
            "height means below the bottom edge"
        ),
    )
    parser.add_argument(
        "--screen-width-cm",
        type=float,
        required=True,
        help="physical width of the display area, cm",
    )
    parser.add_argument(
        "--screen-height-cm",
        type=float,
        required=True,
        help="physical height of the display area, cm",
    )
    parser.add_argument(
        "--calibration-points",
        type=int,
        default=9,
        choices=(5, 9, 13),
        help="GazeFollower calibration mode (default 9, matching its 3x3 grid)",
    )
    args = parser.parse_args()

    targets, geometry = load_targets(args.targets)
    # The TARGET space: Qt logical pixels, what analyze.py scores in.
    width = int(geometry["width_px"])
    height = int(geometry["height_px"])

    import pygame  # noqa: PLC0415 - only needed for the live run

    from gazefollower import GazeFollower  # noqa: PLC0415
    from gazefollower.misc import DefaultConfig  # noqa: PLC0415

    config = DefaultConfig()
    config.cali_mode = args.calibration_points
    # Verified against the library's own px2cm: the origin is the screen's
    # top-left corner, x rightward, y downward.
    config.camera_position = (args.camera_x_cm, args.camera_y_cm)
    config.screen_physical_size = (args.screen_width_cm, args.screen_height_cm)

    below = args.camera_y_cm > args.screen_height_cm
    print(
        f"camera at ({args.camera_x_cm:.1f}, {args.camera_y_cm:.1f}) cm from the "
        f"screen's top-left corner -- "
        f"{'BELOW the bottom edge' if below else 'above or within the screen'}"
    )
    print(f"screen {args.screen_width_cm:.1f} x {args.screen_height_cm:.1f} cm, {width}x{height} px")

    # The LIBRARY's space, from screeninfo: device pixels. On a display with
    # devicePixelRatio 1.25 these differ by exactly that factor, and dividing a
    # library pixel by a Qt logical width inflates every prediction by 1.25
    # while leaving the targets alone. Normalising each by its own width is what
    # puts them back in the same [0, 1] space.
    lib_width = int(config.screen_size[0])
    lib_height = int(config.screen_size[1])
    if (lib_width, lib_height) != (width, height):
        print(
            f"NOTE: library reports {lib_width}x{lib_height} (device px), targets are "
            f"{width}x{height} (Qt logical px). Predictions are normalised by the "
            f"library's size, targets by their own."
        )

    gaze_follower = GazeFollower(config=config)
    samples: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started_at = time.monotonic()

    try:
        gaze_follower.preview()
        # GazeFollower owns this entirely: its own UI, its own points, its own
        # fitting. Nothing is recorded until it returns.
        gaze_follower.calibrate()

        screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        pygame.display.set_caption("GAZELINK - held-out measurement")
        font = pygame.font.Font(None, 48)
        gaze_follower.start_sampling()
        # The library samples on its own thread; give it a moment to fill before
        # the first target so the first point is not scored against an empty
        # buffer.
        time.sleep(0.5)

        aborted = False
        # The library samples on its own thread, so the display loop runs faster
        # than new gaze data arrives. Counting loop iterations would report a
        # rate the tracker never achieved; counting DISTINCT library timestamps
        # reports the rate that actually matters.
        seen_timestamps: set[Any] = set()
        sample_arrival_times: list[float] = []
        for index, target in enumerate(targets):
            if aborted:
                break
            tx = float(target["screen_position"]["x"])
            ty = float(target["screen_position"]["y"])
            arrivals: dict[float, float | None] = dict.fromkeys(DEFAULT_ARRIVAL_RADII_PX)
            shown_at = time.monotonic()

            phase_end = shown_at + (args.settle_ms + args.collect_ms) / 1000.0
            while time.monotonic() < phase_end:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        aborted = True
                if aborted:
                    break

                now = time.monotonic()
                elapsed_ms = (now - shown_at) * 1000.0
                info = gaze_follower.get_gaze_info()

                screen.fill((16, 24, 32))
                _draw_target(screen, tx, ty, width, height)

                if info is not None and getattr(info, "status", False):
                    stamp = getattr(info, "timestamp", None)
                    is_new = stamp is not None and stamp not in seen_timestamps
                    if is_new:
                        seen_timestamps.add(stamp)
                        sample_arrival_times.append(now)
                    point = _pick_point(info)
                    if point is not None:
                        px, py = float(point[0]) / lib_width, float(point[1]) / lib_height
                        _draw_prediction(screen, px, py, width, height)
                        gap = _distance_px(px, py, tx, ty, width, height)
                        label = font.render(f"gap {gap:.0f}px", True, (255, 64, 129))
                        screen.blit(label, (40, height - 80))
                        for radius in DEFAULT_ARRIVAL_RADII_PX:
                            if arrivals[radius] is None and gap <= radius:
                                arrivals[radius] = elapsed_ms
                        # Everything is recorded; the phase label is what lets
                        # the analyser score resting accuracy separately from
                        # the time it took to get there.
                        row = _row(
                            index, px, py, info, elapsed_ms, args.settle_ms, lib_width, lib_height
                        )
                        row["is_new_sample"] = is_new
                        row["fps"] = _recent_fps(sample_arrival_times)
                        samples.append(row)

                progress = font.render(
                    f"point {index + 1}/{len(targets)}   "
                    f"{'SETTLING' if elapsed_ms < args.settle_ms else 'MEASURING'}"
                    "   Esc to stop",
                    True,
                    (255, 215, 64),
                )
                screen.blit(progress, (40, 40))
                pygame.display.flip()

            timings.append(
                {
                    "target_index": index,
                    "shown_at_ms": (shown_at - started_at) * 1000.0,
                    "time_to_target_ms": {
                        str(int(radius)): arrivals[radius] for radius in DEFAULT_ARRIVAL_RADII_PX
                    },
                }
            )

        gaze_follower.stop_sampling()
    finally:
        gaze_follower.release()
        try:
            pygame.quit()
        except Exception:  # noqa: BLE001 - shutting down must never mask a result
            pass

    payload = {
        "screen_geometry": geometry,
        "targets": [
            {
                "index": index,
                "name": target.get("name", f"TEST_{index}"),
                "screen_position": target["screen_position"],
            }
            for index, target in enumerate(targets)
        ],
        "samples": samples,
        "timings": timings,
        "run": {
            "engine": "gazefollower",
            "library_version": _version(),
            "model": "bundled (7M images); the paper's figures used the 32M model",
            "calibration_points_required": args.calibration_points,
            "calibration_owner": "gazefollower",
            # Recorded because it is a geometric input to the model, not a
            # cosmetic setting: a wrong value here is indistinguishable from a
            # bad engine in the resulting numbers.
            "camera_position_cm": [args.camera_x_cm, args.camera_y_cm],
            "screen_physical_size_cm": [args.screen_width_cm, args.screen_height_cm],
            "settle_ms": args.settle_ms,
            "collect_ms": args.collect_ms,
            "units": "qt_logical_px",
            "dpi_scale": geometry.get("dpi_scale"),
            "physical_width_px": round(width * float(geometry.get("dpi_scale", 1.0))),
            "physical_height_px": round(height * float(geometry.get("dpi_scale", 1.0))),
            "samples_recorded": len(samples),
            "library_screen_px": [lib_width, lib_height],
            "target_screen_px": [width, height],
            "distinct_library_samples": len(seen_timestamps),
            "measured_fps_median": _median_fps(sample_arrival_times),
            "aborted": aborted,
            # This engine has no post-calibration training thread to freeze, so
            # the freeze question does not arise. Saying so explicitly keeps the
            # analyser from treating a missing field as an unverified freeze.
            "freeze_verified": True,
            "freeze_detail": "not applicable: calibration is a discrete phase that ends before sampling",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {args.out}")
    return 0


def _recent_fps(arrivals: list[float], window: int = 20) -> float | None:
    """Sample rate over the last few genuinely new samples, or None."""

    if len(arrivals) < 2:
        return None
    recent = arrivals[-window:]
    span = recent[-1] - recent[0]
    return None if span <= 0 else (len(recent) - 1) / span


def _median_fps(arrivals: list[float]) -> float | None:
    if len(arrivals) < 3:
        return None
    gaps = sorted(b - a for a, b in zip(arrivals, arrivals[1:]) if b > a)
    if not gaps:
        return None
    return 1.0 / gaps[len(gaps) // 2]


def _pick_point(info: Any) -> Any:
    """The calibrated point, which is what the library presents as its answer.

    Falls back to the raw one only if calibration produced nothing, and the row
    records which was used so a reader is never guessing.
    """

    for attribute in ("calibrated_gaze_coordinates", "raw_gaze_coordinates"):
        value = getattr(info, attribute, None)
        if value is not None and len(value) >= 2:
            return value
    return None


def _row(
    index: int,
    px: float,
    py: float,
    info: Any,
    elapsed_ms: float,
    settle_ms: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    raw = getattr(info, "raw_gaze_coordinates", None)
    filtered = getattr(info, "filtered_gaze_coordinates", None)
    state = getattr(info, "tracking_state", None)
    return {
        # The five fields analyze.py reads.
        "target_index": index,
        "predicted_x": px,
        "predicted_y": py,
        "timestamp": elapsed_ms,
        "accepted": True,
        # Everything below rides along and its parser ignores it.
        "collection_phase": "COLLECTING" if elapsed_ms >= settle_ms else "STABILIZING",
        "tracking_state": None if state is None else getattr(state, "name", str(state)),
        "left_openness": getattr(info, "left_openness", None),
        "right_openness": getattr(info, "right_openness", None),
        "event": getattr(getattr(info, "event", None), "name", None),
        # Raw and filtered beside the scored value, so "before and after
        # smoothing" is answerable from the file instead of asserted.
        "raw_x": None if raw is None else float(raw[0]) / width,
        "raw_y": None if raw is None else float(raw[1]) / height,
        "filtered_x": None if filtered is None else float(filtered[0]) / width,
        "filtered_y": None if filtered is None else float(filtered[1]) / height,
        "library_pixel_x": px * width,
        "library_pixel_y": py * height,
    }


def _draw_target(surface: Any, x: float, y: float, width: int, height: int) -> None:
    import pygame  # noqa: PLC0415

    cx = round(x * (width - 1))
    cy = round(y * (height - 1))
    pygame.draw.circle(surface, (255, 215, 64), (cx, cy), TARGET_GLYPH_PX // 3)
    pygame.draw.circle(surface, (16, 24, 32), (cx, cy), TARGET_GLYPH_PX // 9)


def _draw_prediction(surface: Any, x: float, y: float, width: int, height: int) -> None:
    import pygame  # noqa: PLC0415

    on_screen = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    colour = (255, 64, 129) if on_screen else (255, 110, 64)
    cx = round(min(1.0, max(0.0, x)) * (width - 1))
    cy = round(min(1.0, max(0.0, y)) * (height - 1))
    arm = TARGET_GLYPH_PX // 3
    pygame.draw.line(surface, colour, (cx - arm, cy - arm), (cx + arm, cy + arm), 6)
    pygame.draw.line(surface, colour, (cx - arm, cy + arm), (cx + arm, cy - arm), 6)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("gazefollower")
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
