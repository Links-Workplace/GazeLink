"""Shared, pure helpers for the GazeFollower experiment (GAZELINK task M2-05).

Everything here mirrors a formula or a constant that lives inside the
installed GazeFollower 1.0.2 package, and nothing here imports that package.
The reason is not taste: importing ANY module of ``gazefollower`` -- even
``gazefollower.misc`` -- evaluates the default arguments of
``GazeFollower.__init__``, which constructs a webcam object, a MediaPipe
FaceMesh and the MNN gaze model at import time. Unit tests must run without a
camera and without those models, so the handful of numbers and four-line
formulas this experiment depends on are copied here and cross-checked at
recorder start-up against the library's own functions (``gf_record.py``).

Units, stated once:
  * "device px"   -- the library's pixel space, ``screeninfo`` resolution
                     (5120 x 1440 on this rig).
  * "logical px"  -- Qt's space, what ``analyze.py`` scores in (4096 x 1152).
  * "normalised"  -- [0, 1] from the screen's top-left, the only unit that is
                     the same in both spaces and the one every file carries.
  * "cm"          -- centimetres relative to the camera centre, y UP, the
                     library's calibration label space (``misc.px2cm``).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# --- GazeFollower's calibration protocol -------------------------------------
# calibration/CalibrationController.py, __init__ and add_cali_feature.
PREPARE_S = 1.5  # _prepare_time: seconds after target onset before any frame counts
WAIT_S = 0.5  # _wait_time: seconds after the 45th frame during which nothing is stored
N_FRAMES_PER_POINT = 45  # _n_frame_need_collect: ACCEPTED frames; rejected ones are not counted
BLINK_THRESHOLD = 10.0  # eye_blink_threshold: eye polygon AREA in px^2, per eye, strict '>'

# --- GazeFollower's 9-point grid ---------------------------------------------
# misc.generate_points() lays a 9 x 5 mesh with 50 px margins on a nominal
# 1920 x 1080 canvas; _nine_cali_idx picks these nine, plus a warm-up centre.
GRID_X: tuple[float, float, float] = (50 / 1920, 0.5, 1870 / 1920)
GRID_Y: tuple[float, float, float] = (50 / 1080, 0.5, 1030 / 1080)

# Display order. Index 0 is the warm-up centre: the controller shows it but
# stores none of its frames (``if self._current_index != 0``). The centre is
# shown again at the END and that one IS stored. Nine stored points total.
NINE_POINT_SEQUENCE: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),  # warm-up, NOT stored
    (GRID_X[0], GRID_Y[0]),  # idx 1
    (GRID_X[1], GRID_Y[0]),  # idx 5
    (GRID_X[2], GRID_Y[0]),  # idx 9
    (GRID_X[0], GRID_Y[1]),  # idx 19
    (GRID_X[2], GRID_Y[1]),  # idx 27
    (GRID_X[0], GRID_Y[2]),  # idx 37
    (GRID_X[1], GRID_Y[2]),  # idx 41
    (GRID_X[2], GRID_Y[2]),  # idx 45
    (0.5, 0.5),  # idx 23, stored
)
NINE_POINT_STORED: tuple[tuple[float, float], ...] = NINE_POINT_SEQUENCE[1:]


def nine_point_sequence(
    grid_x: tuple[float, float, float] = GRID_X, grid_y: tuple[float, float, float] = GRID_Y
) -> tuple[tuple[float, float], ...]:
    """The controller's display order on an arbitrary 3 x 3 grid.

    With the default arguments this is exactly ``NINE_POINT_SEQUENCE``. A
    narrower ``grid_x`` exists for the central-band experiment: the model's
    horizontal output saturates beyond about a third of the screen from the
    centre, so a calibration that reaches the edges trains on a flat signal.
    Such a run is NOT a faithful reproduction of the library's calibration
    and is recorded as such.
    """

    x0, x1, x2 = grid_x
    y0, y1, y2 = grid_y
    return (
        (x1, y1),  # warm-up, NOT stored
        (x0, y0),
        (x1, y0),
        (x2, y0),
        (x0, y1),
        (x2, y1),
        (x0, y2),
        (x1, y2),
        (x2, y2),
        (x1, y1),  # stored centre
    )

# --- Measurement constants shared with gazefollower_capture.py ---------------
DEFAULT_SETTLE_MS = 1500.0
DEFAULT_COLLECT_MS = 1500.0
DEFAULT_ARRIVAL_RADII_PX: tuple[float, ...] = (100.0, 200.0, 400.0)
# analyze.py splits resting from transit on this exact string.
RESTING_PHASE = "COLLECTING"
TRANSIT_PHASE = "STABILIZING"


@dataclass(frozen=True)
class RigGeometry:
    """Physical layout the library needs, in the library's own convention.

    ``camera_x_cm`` / ``camera_y_cm`` are the camera centre measured from the
    screen's top-left corner: x to the right, y DOWNWARD. A ``camera_y_cm``
    larger than ``screen_h_cm`` means the camera sits below the bottom edge,
    which is this rig (63.6 cm down on a 33.75 cm tall screen).
    """

    camera_x_cm: float
    camera_y_cm: float
    screen_w_cm: float
    screen_h_cm: float
    device_w_px: int
    device_h_px: int

    def __post_init__(self) -> None:
        for name in ("screen_w_cm", "screen_h_cm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("device_w_px", "device_h_px"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def camera_below_screen(self) -> bool:
        return self.camera_y_cm > self.screen_h_cm

    # The two conversions below reproduce misc.px2cm / misc.cm2px expression
    # by expression (dpi first, then the division) so the self-check in the
    # recorder compares like with like.
    def _dpi(self) -> tuple[float, float]:
        dpi_x = self.device_w_px / (self.screen_w_cm / 2.54)
        dpi_y = self.device_h_px / (self.screen_h_cm / 2.54)
        return dpi_x, dpi_y

    def px2cm(self, px_x: float, px_y: float) -> tuple[float, float]:
        dpi_x, dpi_y = self._dpi()
        cm_x = px_x * 2.54 / dpi_x - self.camera_x_cm
        cm_y = (px_y * 2.54 / dpi_y - self.camera_y_cm) * (-1)
        return cm_x, cm_y

    def cm2px(self, cm_x: float, cm_y: float) -> tuple[float, float]:
        dpi_x, dpi_y = self._dpi()
        px_x = (cm_x + self.camera_x_cm) * dpi_x / 2.54
        px_y = (-cm_y + self.camera_y_cm) * dpi_y / 2.54
        return px_x, px_y

    def norm_to_label_cm(self, nx: float, ny: float) -> tuple[float, float]:
        """The label the controller stores for a target at normalised (nx, ny).

        Mirrors ``px2cm((self.x * screen_size[0], self.y * screen_size[1]))``
        -- note ``* size``, not ``* (size - 1)``.
        """

        return self.px2cm(nx * self.device_w_px, ny * self.device_h_px)

    def cm_to_norm(self, cm_x: float, cm_y: float) -> tuple[float, float]:
        px_x, px_y = self.cm2px(cm_x, cm_y)
        return px_x / self.device_w_px, px_y / self.device_h_px

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RigGeometry":
        return cls(
            camera_x_cm=float(value["camera_x_cm"]),
            camera_y_cm=float(value["camera_y_cm"]),
            screen_w_cm=float(value["screen_w_cm"]),
            screen_h_cm=float(value["screen_h_cm"]),
            device_w_px=int(value["device_w_px"]),
            device_h_px=int(value["device_h_px"]),
        )


def call_with_timeout(func: Callable[[], Any], timeout_s: float) -> tuple[bool, Any]:
    """Run ``func`` on a daemon thread; give up after ``timeout_s``.

    Returns ``(finished, result_or_exception)``.

    This exists because OpenCV's ``VideoCapture`` constructor and ``read()``
    are blocking calls with no timeout of their own. Checking a deadline
    around them bounds the LOOP but not the CALL: a camera another process is
    holding can park a single ``read()`` forever, and the deadline is never
    reached to be tested. The only way to stay responsive is to put the whole
    thing on a thread we can walk away from.

    Walking away leaks that thread, and it keeps holding the camera until the
    process exits -- which is why a timeout here must be treated as a hard
    failure rather than something to continue past.
    """

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        return False, TimeoutError(f"call did not return within {timeout_s}s")
    if "error" in outcome:
        return True, outcome["error"]
    return True, outcome.get("value")


def distance_px(
    ax: float, ay: float, bx: float, by: float, width_px: int, height_px: int
) -> float:
    """Pixel distance between two normalised points, as analyze.py computes it.

    ``(width - 1)`` and ``(height - 1)``: a normalised 1.0 is the LAST pixel.
    """

    dx = (ax - bx) * (width_px - 1)
    dy = (ay - by) * (height_px - 1)
    return math.hypot(dx, dy)


# --- targets.json / predictions.json ----------------------------------------


def load_targets(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a targets file written by the project's export_test_targets.py."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = payload["targets"]
    geometry = payload["screen_geometry"]
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path}: 'targets' must be a non-empty list")
    for entry in targets:
        pos = entry["screen_position"]
        if not (0.0 <= float(pos["x"]) <= 1.0 and 0.0 <= float(pos["y"]) <= 1.0):
            raise ValueError(f"{path}: target {entry.get('name')} is outside [0, 1]")
    return targets, geometry


def make_sample(
    *,
    target_index: int,
    predicted_x: float,
    predicted_y: float,
    timestamp_ms: float,
    accepted: bool = True,
    collection_phase: str | None = None,
    fps: float | None = None,
    head_yaw_deg: float | None = None,
    head_pitch_deg: float | None = None,
    head_roll_deg: float | None = None,
    **extras: Any,
) -> dict[str, Any]:
    """One predictions-file row in the shape analyze.py's PredictionResult reads.

    The five required fields come first; the optional ones use the exact key
    names analyze.py looks for; anything in ``extras`` rides along and is
    ignored by the analyser (raw/filtered coordinates, arm, config...).
    """

    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise TypeError("target_index must be an int (not bool)")
    row: dict[str, Any] = {
        "target_index": target_index,
        "predicted_x": float(predicted_x),
        "predicted_y": float(predicted_y),
        "timestamp": float(timestamp_ms),
        "accepted": bool(accepted),
    }
    if collection_phase is not None:
        row["collection_phase"] = str(collection_phase)
    if fps is not None:
        row["fps"] = float(fps)
    if head_yaw_deg is not None and head_pitch_deg is not None and head_roll_deg is not None:
        row["head_yaw_deg"] = float(head_yaw_deg)
        row["head_pitch_deg"] = float(head_pitch_deg)
        row["head_roll_deg"] = float(head_roll_deg)
    for key, value in extras.items():
        if key in row:
            raise ValueError(f"extra field {key!r} would shadow a required field")
        row[key] = value
    return row


def validate_predictions_payload(payload: Mapping[str, Any]) -> None:
    """Raise if analyze.py's PredictionResult.from_dict would reject this.

    Re-states the analyser's checks rather than importing them: this
    environment must not import the project package.
    """

    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("predictions file must declare a non-empty 'targets' list")
    for entry in targets:
        _ = entry["name"]
        pos = entry["screen_position"]
        float(pos["x"]), float(pos["y"])
    geometry = payload.get("screen_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("predictions file must carry 'screen_geometry'")
    for key in ("width_px", "height_px"):
        if not isinstance(geometry.get(key), int) or geometry[key] <= 0:
            raise ValueError(f"screen_geometry.{key} must be a positive int")
    for sample in payload.get("samples", ()):
        index = sample["target_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(targets):
            raise ValueError(f"sample target_index {index!r} is out of range")
        float(sample["predicted_x"]), float(sample["predicted_y"]), float(sample["timestamp"])
        if not isinstance(sample["accepted"], bool):
            raise ValueError("sample.accepted must be a bool")


def write_predictions_json(
    path: Path,
    *,
    screen_geometry: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]],
    run: Mapping[str, Any],
    timings: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write a file analyze.py --predictions can score. Validates before writing.

    ``run`` deliberately carries NO ``freeze_verified`` field: that claim
    exists for engines that keep learning during use and must prove they
    stopped. GazeFollower's SVR is fitted once and never updated at run time,
    so the honest statement is to make no freeze claim at all.
    """

    if "freeze_verified" in run:
        raise ValueError("do not claim freeze_verified: this engine never learns at run time")
    payload = {
        "screen_geometry": dict(screen_geometry),
        "targets": [
            {
                "index": int(entry["index"]),
                "name": str(entry["name"]),
                "screen_position": {
                    "x": float(entry["screen_position"]["x"]),
                    "y": float(entry["screen_position"]["y"]),
                },
            }
            for entry in targets
        ],
        "samples": [dict(sample) for sample in samples],
        "run": dict(run),
        "timings": [dict(entry) for entry in timings],
    }
    validate_predictions_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def arrival_times_ms(
    rows: Sequence[tuple[float, float, float]],
    target_x: float,
    target_y: float,
    width_px: int,
    height_px: int,
    radii_px: Sequence[float] = DEFAULT_ARRIVAL_RADII_PX,
) -> dict[str, float | None]:
    """First time (ms after onset) a prediction came within each radius.

    ``rows`` are ``(elapsed_ms, predicted_x, predicted_y)`` in onset order.
    Same definition as gazefollower_capture.py, so the offline arms report a
    number that means the same thing as the live capture's.
    """

    result: dict[str, float | None] = {str(int(r)): None for r in radii_px}
    for elapsed_ms, px, py in rows:
        gap = distance_px(px, py, target_x, target_y, width_px, height_px)
        for radius in radii_px:
            key = str(int(radius))
            if result[key] is None and gap <= radius:
                result[key] = float(elapsed_ms)
    return result
