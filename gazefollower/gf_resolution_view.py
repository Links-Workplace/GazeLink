"""Watch the two resolutions side by side, live: which dot follows the eyes?

Visual companion to ``gf_resolution_compare``. Each camera frame goes through
both pipelines of ``gf_capture`` (1440x1080 crop = SHARP, the same crop reduced
to 640x480 = REGULAR), and each pipeline drives its own model, fitted on the
SAME calibration rows of one ``--capture dual`` round. Two dots, same frame,
same filter:

* blue cross  -- SHARP (hi) model
* orange dot  -- REGULAR (lo) model

Nine numbered boxes mark the band zones to look at. Nothing is recorded,
saved or logged from the camera; no cursor moves and no input is sent.

Both models come from one short calibration, so both are less accurate than
the pooled production model. This compares the two pipelines with each
other, not either of them with production.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_capture as CAP  # noqa: E402
from gazelink_core.domain.observation import FrameObservation, HeadPolicy  # noqa: E402
from gazelink_core.gaze import sample_gate as GATE  # noqa: E402
from gazelink_core.tracking import gazefollower_library as LIB  # noqa: E402
from gazelink_core.tracking import gazefollower_source as SRC  # noqa: E402
import gf_common as C  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_live as L  # noqa: E402
import gf_record as R  # noqa: E402
import gf_resolution_compare as RCMP  # noqa: E402
import gf_schema as S  # noqa: E402
import gf_targets as T  # noqa: E402

MODEL_ROOT = Path(__file__).resolve().parent / "results" / "resolution" / "models"


@dataclass(frozen=True)
class Box:
    """A zone marker in the shape Display.draw_live expects (outline only)."""

    key: str
    x0: float
    y0: float
    x1: float
    y1: float


def zone_boxes(size: float = 0.04) -> list[Box]:
    lo, hi = T.WORKING_BAND_X
    boxes = []
    for i in range(T.BAND_COLS * T.BAND_ROWS):
        row, col = divmod(i, T.BAND_COLS)
        cx = lo + (col + 0.5) * (hi - lo) / T.BAND_COLS
        cy = (row + 0.5) / T.BAND_ROWS
        aspect = 5120 / 1440  # keep the box square on this screen
        boxes.append(Box(str(i + 1), cx - size / aspect, cy - size, cx + size / aspect, cy + size))
    return boxes


def fit_pair(round_dir: Path, config: str) -> tuple[FIT.FittedModel, FIT.FittedModel]:
    """(lo, hi) models on identical calibration rows, saved for reuse."""

    cal_hi, cal_lo = RCMP.load_pair(round_dir, "A")
    rig = C.RigGeometry.from_dict(cal_hi.meta["rig"])
    rows = cal_hi.rows_accepted() & cal_lo.rows_accepted() & RCMP.both_valid(cal_hi, cal_lo)
    models = []
    for arm, rec in (("lo", cal_lo), ("hi", cal_hi)):
        target = MODEL_ROOT / f"{round_dir.name}_{arm}_{config}"
        if target.exists():
            models.append(FIT.FittedModel.load(target))
            continue
        model = RCMP.fit_arm(rec, rows, config, rig)
        model.save(target)
        models.append(model)
    return models[0], models[1]


class TwoDots:
    """Latest filtered point per arm, written by the camera thread."""

    def __init__(self, rig: C.RigGeometry, settings: GF.FilterSettings) -> None:
        self.rig = rig
        self.lock = threading.Lock()
        self.filters = {"hi": GF.GazePointFilter(settings), "lo": GF.GazePointFilter(settings)}
        self.points: dict[str, tuple[float, float] | None] = {"hi": None, "lo": None}
        self.updated_s = 0.0
        self.frames = 0

    def update(self, arm: str, model: Any, obs: FrameObservation | None, now_s: float) -> None:
        # This view's own policy, unchanged: a usable gaze with features and
        # both eyes over the blink threshold (a missing face object reads as
        # closed eyes, so it cannot pass); float32 features, no head columns.
        ok = (
            obs is not None
            and GATE.is_gaze_sample(obs, GATE.RECORD_OVERLAY)
            and obs.features is not None
        )
        point = None
        if ok:
            features = GATE.features_for(obs, GATE.RECORD_OVERLAY)
            point = R.predict_one(model, features, None, self.rig)
        with self.lock:
            if point is None:
                # A blink or lost face must not pull the next point toward
                # stale history -- the same rule as the recorder's overlay.
                self.filters[arm].reset()
                self.points[arm] = None
            else:
                self.points[arm] = self.filters[arm].update(point, now_s)
            self.updated_s = now_s

    def snapshot(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None, float]:
        with self.lock:
            return self.points["hi"], self.points["lo"], self.updated_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round-dir", type=Path, default=Path("recordings/resolution/round1"))
    parser.add_argument("--config", default=RCMP.PRODUCTION_CONFIG)
    parser.add_argument("--only", choices=("both", "hi", "lo"), default="both")
    parser.add_argument("--profile", default=None, help="whose filter settings to use (default: active)")
    args = parser.parse_args(argv)

    from gazefollower import GazeFollower  # noqa: PLC0415
    from gazefollower.misc import DefaultConfig  # noqa: PLC0415

    model_lo, model_hi = fit_pair(args.round_dir, args.config)
    rig = C.RigGeometry.from_dict(S.Recording.load(args.round_dir, "A").meta["rig"])
    profile = L.resolve_profile(args.profile)
    dots = TwoDots(rig, profile.filter_settings())

    camera, hi_alignment, hi_estimator = CAP.make_dual_components()
    config = LIB.library_config(rig, DefaultConfig())
    gf = GazeFollower(
        camera=camera,
        face_alignment=hi_alignment,
        gaze_estimator=hi_estimator,
        config=config,
        calibration=R.make_pass_through_calibration(),
    )

    def subscriber(face_info: Any, gaze_info: Any) -> None:
        now = time.monotonic()
        # Both arms go through the tracking adapter; nothing here reads the
        # library's objects.
        hi_obs = SRC.observe(face_info, gaze_info, observed_s=now, head_builder=lambda _f: None,
                             head_policy=HeadPolicy.GAZE)
        shadow = camera.shadow_for(hi_obs.timestamp_ns)
        lo_obs = CAP.observe_shadow(shadow, head_builder=lambda _f: None)
        dots.update("hi", model_hi, hi_obs, now)
        dots.update("lo", model_lo, lo_obs, now)

    gf.add_subscriber(subscriber)
    display = R.Display(rig.device_w_px, rig.device_h_px, headless=False, origin=(0, 0))
    boxes = zone_boxes()
    try:
        camera.start_sampling()
        display.draw_message(["Camera warming up..."])
        if R._sleep_with_escape(display, R.CAMERA_WARMUP_S):
            return 0
        if camera.usable_frames() < R.DUAL_MIN_WARMUP_FRAMES:
            print(f"camera did not deliver 1920x1080 frames: {camera.describe()['stats']}")
            return 1
        legend = {
            "both": "BLUE cross = SHARP (1440x1080)    ORANGE dot = REGULAR (640x480)",
            "hi": "BLUE cross = SHARP (1440x1080) only",
            "lo": "BLUE cross = REGULAR (640x480) only",
        }[args.only]
        fps: float | None = None
        fps_at = 0.0
        while True:
            if display.poll_escape():
                break
            hi, lo, updated = dots.snapshot()
            fresh = time.monotonic() - updated < R.OVERLAY_STALE_S
            if not fresh:
                hi = lo = None
            if args.only == "hi":
                point, second = hi, None
            elif args.only == "lo":
                point, second = lo, None
            else:
                point, second = hi, lo
            if time.monotonic() - fps_at > 1.0:
                # Once a second: the stats copy and summarise every frame so far.
                fps = camera.describe()["stats"]["delivered_fps_median"]
                fps_at = time.monotonic()
            display.draw_live(
                point,
                second,
                None,
                [
                    legend,
                    "Look at the numbered boxes. Esc to quit. Nothing is recorded.",
                    f"camera fps {fps:.1f}" if fps else "camera fps --",
                    f"models: {args.round_dir.name} calibration, {args.config}",
                ],
                tracking=fresh and (point is not None or second is not None),
                zones=boxes,
            )
            time.sleep(1 / 60)
    finally:
        R._call_with_timeout(camera.stop_sampling, R.SHUTDOWN_STEP_TIMEOUT_S)
        R.shutdown_library(gf)
        display.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
