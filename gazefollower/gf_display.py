"""Choose a monitor, describe it exactly, and build targets that cover it.

The ultrawide experiments ran on the primary display and could assume it. A
24-inch monitor arriving as a second screen breaks three assumptions at once,
and each one silently corrupts a measurement rather than raising an error:

* **Which display.** ``screeninfo`` returns every monitor; the library takes
  ``get_monitors()[0]``. Calibrating on one screen and scoring against
  another's resolution produces numbers that look plausible and mean nothing.
* **Desktop origin.** A second monitor sits at a non-zero ``(x, y)`` in the
  virtual desktop. A window placed at 0,0 lands on the wrong screen, and a
  target drawn at "half the width" lands half a desktop away.
* **Angles, not pixels.** The finding under test is about visual angle. A
  53 cm screen at 73 cm subtends about +/-20 deg horizontally; the ultrawide
  central band subtended a similar horizontal range but a DIFFERENT vertical
  one, so horizontal agreement alone does not carry over.

Viewing distance (eye to screen) and camera distance (eye to camera) are
recorded separately: on a rig where the camera sits below and in front of the
display they are not the same number, and the specification's 45-70 cm range
refers to the camera.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402

DISPLAY_VERSION = "display-1"

# Full-screen evaluation grid: centre, sides, top, bottom and four corner
# regions. 0.08/0.92 keeps targets a little inside the bezel, where a
# fixation is still comfortable; 0.5 gives the centre row and column.
FULL_COVERAGE_COORDS: tuple[float, ...] = (0.08, 0.5, 0.92)
# A denser evaluation set: the 3x3 coverage grid plus the quarter positions,
# so edges and corners are represented without relying on 16 points alone.
EVAL_COORDS_X: tuple[float, ...] = (0.08, 0.29, 0.5, 0.71, 0.92)
EVAL_COORDS_Y: tuple[float, ...] = (0.08, 0.5, 0.92)


@dataclass(frozen=True)
class MonitorInfo:
    """One display, as the OS reports it, plus what the operator measured."""

    index: int
    name: str
    width_px: int
    height_px: int
    x: int
    y: int
    is_primary: bool
    width_mm: float | None = None
    height_mm: float | None = None

    @property
    def origin(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def aspect(self) -> float:
        return self.width_px / self.height_px

    @property
    def diagonal_in(self) -> float | None:
        if not (self.width_mm and self.height_mm):
            return None
        return math.hypot(self.width_mm, self.height_mm) / 25.4

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aspect"] = round(self.aspect, 4)
        d["diagonal_in"] = None if self.diagonal_in is None else round(self.diagonal_in, 1)
        return d


def list_monitors() -> list[MonitorInfo]:
    from screeninfo import get_monitors  # noqa: PLC0415

    out = []
    for i, m in enumerate(get_monitors()):
        out.append(
            MonitorInfo(
                index=i,
                name=str(m.name),
                width_px=int(m.width),
                height_px=int(m.height),
                x=int(m.x),
                y=int(m.y),
                is_primary=bool(getattr(m, "is_primary", False)),
                width_mm=float(m.width_mm) if getattr(m, "width_mm", None) else None,
                height_mm=float(m.height_mm) if getattr(m, "height_mm", None) else None,
            )
        )
    return out


def pick_monitor(selector: str | int | None = None) -> MonitorInfo:
    """Select by index, by a substring of the name, or fall back to primary.

    Selecting explicitly is the point: on a two-monitor desk the primary is
    usually the wrong one for the experiment.
    """

    monitors = list_monitors()
    if not monitors:
        raise RuntimeError("no monitors reported")
    if selector is None:
        return next((m for m in monitors if m.is_primary), monitors[0])
    if isinstance(selector, int) or (isinstance(selector, str) and selector.isdigit()):
        index = int(selector)
        match = next((m for m in monitors if m.index == index), None)
        if match is None:
            raise ValueError(f"no monitor with index {index}; available: {[m.index for m in monitors]}")
        return match
    matches = [m for m in monitors if str(selector).lower() in m.name.lower()]
    if not matches:
        raise ValueError(f"no monitor whose name contains {selector!r}; available: {[m.name for m in monitors]}")
    if len(matches) > 1:
        raise ValueError(f"{selector!r} matches several monitors: {[m.name for m in matches]}")
    return matches[0]


def describe_monitors() -> str:
    lines = ["Displays reported by the OS:", ""]
    for m in list_monitors():
        size = f"{m.width_mm:.0f}x{m.height_mm:.0f} mm" if m.width_mm else "physical size unknown"
        diag = f", ~{m.diagonal_in:.0f} in" if m.diagonal_in else ""
        lines.append(
            f"  [{m.index}] {m.name}  {m.width_px}x{m.height_px} px at desktop ({m.x}, {m.y})"
            f"  {size}{diag}  aspect {m.aspect:.2f}{'  PRIMARY' if m.is_primary else ''}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ViewingGeometry:
    """Everything needed to turn a pixel error into an angle.

    ``viewing_distance_cm`` is eye to screen centre. ``camera_distance_cm`` is
    eye to camera and may differ; the specification's 45-70 cm range refers to
    the camera, so both are recorded and compared against it separately.
    """

    screen_w_cm: float
    screen_h_cm: float
    width_px: int
    height_px: int
    viewing_distance_cm: float
    camera_distance_cm: float | None = None

    def half_angles_deg(self) -> tuple[float, float]:
        """Half-angles from the centre to the horizontal and vertical edges."""

        h = math.degrees(math.atan2(self.screen_w_cm / 2.0, self.viewing_distance_cm))
        v = math.degrees(math.atan2(self.screen_h_cm / 2.0, self.viewing_distance_cm))
        return h, v

    def px_to_deg_at_centre(self) -> tuple[float, float]:
        """Degrees per pixel near the centre, where the small-angle view holds."""

        cm_per_px_x = self.screen_w_cm / self.width_px
        cm_per_px_y = self.screen_h_cm / self.height_px
        return (
            math.degrees(math.atan2(cm_per_px_x, self.viewing_distance_cm)),
            math.degrees(math.atan2(cm_per_px_y, self.viewing_distance_cm)),
        )

    def deg_to_px(self, degrees: float) -> float:
        """Pixels subtending an angle at the centre, on the horizontal axis."""

        cm = self.viewing_distance_cm * math.tan(math.radians(degrees))
        return cm / (self.screen_w_cm / self.width_px)

    def to_dict(self) -> dict[str, Any]:
        h, v = self.half_angles_deg()
        dpx, dpy = self.px_to_deg_at_centre()
        return {
            **asdict(self),
            "half_angle_h_deg": round(h, 2),
            "half_angle_v_deg": round(v, 2),
            "deg_per_px_x": dpx,
            "deg_per_px_y": dpy,
            "px_per_1.5deg_x": round(self.deg_to_px(1.5), 1),
            "note": (
                "Half-angles are from the centre to the edge, measured from the declared "
                "viewing distance. Degrees per pixel are a small-angle value near the centre "
                "and understate the angle at the edges."
            ),
        }


def compare_angular_coverage(new: ViewingGeometry, reference: ViewingGeometry, reference_x_range: tuple[float, float] | None = None) -> dict[str, Any]:
    """Does the new screen cover the angles the previous experiment covered?

    Horizontal success on a narrow band does not imply full-screen success on
    a 16:9 display, because the vertical extent changes too. This states both
    axes side by side rather than letting the horizontal one speak for both.
    """

    nh, nv = new.half_angles_deg()
    rh, rv = reference.half_angles_deg()
    if reference_x_range is not None:
        lo, hi = reference_x_range
        half_frac = max(abs(0.5 - lo), abs(hi - 0.5))
        rh = math.degrees(math.atan2(reference.screen_w_cm * half_frac, reference.viewing_distance_cm))
    return {
        "new_half_angle_h_deg": round(nh, 2),
        "new_half_angle_v_deg": round(nv, 2),
        "reference_half_angle_h_deg": round(rh, 2),
        "reference_half_angle_v_deg": round(rv, 2),
        "horizontal_within_reference": nh <= rh + 1e-6,
        "vertical_within_reference": nv <= rv + 1e-6,
        "verdict": (
            "both axes inside the range already exercised"
            if (nh <= rh + 1e-6 and nv <= rv + 1e-6)
            else "the new screen asks for angles the previous experiment did not cover"
        ),
        "caveat": (
            "Angular coverage says the eye must travel a comparable distance. It does not "
            "establish that the engine performs equally there; that is what the sessions measure."
        ),
    }


def coverage_targets(
    width_px: int,
    height_px: int,
    *,
    seed: int = 20260908,
    xs: Sequence[float] = EVAL_COORDS_X,
    ys: Sequence[float] = EVAL_COORDS_Y,
    calibration: Sequence[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluation targets covering the whole display, tagged by region.

    Order is randomised with a recorded seed so a drift over the session
    cannot line up with screen position.
    """

    import gf_targets as T  # noqa: PLC0415
    import random  # noqa: PLC0415

    points = [(x, y) for y in ys for x in xs]
    random.Random(seed).shuffle(points)
    out: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(points):
        distance, nearest = T.distance_to_nearest_calibration(x, y, width_px, height_px, calibration)
        out.append(
            {
                "index": index,
                "name": f"F_{round(x * 100):02d}_{round(y * 100):02d}",
                "screen_position": {"x": x, "y": y},
                "region": region_of(x, y),
                "distance_to_calibration_px": round(distance, 1),
                "nearest_calibration": {"x": nearest[0], "y": nearest[1]},
                "near_calibration": distance < T.NEAR_CALIBRATION_PX,
            }
        )
    return out


def region_of(x: float, y: float, *, edge: float = 0.25) -> str:
    """centre / edge-h / edge-v / corner, for reporting by screen region."""

    h_out = x < edge or x > 1.0 - edge
    v_out = y < edge or y > 1.0 - edge
    if h_out and v_out:
        return "corner"
    if h_out:
        return "edge-h"
    if v_out:
        return "edge-v"
    return "centre"


def region_summary(targets: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in targets:
        r = t.get("region") or region_of(t["screen_position"]["x"], t["screen_position"]["y"])
        counts[r] = counts.get(r, 0) + 1
    return counts


def geometry_dict(monitor: MonitorInfo, dpi_scale: float) -> dict[str, Any]:
    """The ``screen_geometry`` block, in the shape analyze.py expects."""

    return {
        "screen_id": monitor.name,
        "width_px": monitor.width_px,
        "height_px": monitor.height_px,
        "dpi_scale": dpi_scale,
        "orientation": "LANDSCAPE" if monitor.width_px >= monitor.height_px else "PORTRAIT",
    }


if __name__ == "__main__":
    print(describe_monitors())
