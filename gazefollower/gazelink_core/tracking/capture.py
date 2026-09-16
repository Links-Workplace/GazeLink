"""Dual-resolution capture: one camera frame, two gaze pipelines, paired rows.

The question (TASKS §62): does MGazeNet get better input from the camera's
larger modes? The library's own camera requests 640x480 before the device is
open (ignored), the driver defaults to 640x480, and every frame is resized to
640x480 anyway -- while the camera delivers 1920x1080 at the same frame rate.

Measured with ``gf_camera_probe.py --fov``: the outer inter-ocular distance
grows 1.54x from 640x480 to 1280x720 and 2.13x to 1920x1080, with the eyes at
the same height in every mode. So the 16:9 modes keep the vertical field of
view and add width: 640x480 is a side crop. A 4:3 CENTRE CROP of the large
frame (1440x1080) therefore shows the same region with the same proportions
-- no stretching -- at about 2.25x the linear resolution. (That the 640x480
crop is exactly centred is an assumption; see ``CAPTURE_ASSUMPTIONS``.)

Every frame read in ``SOURCE_MODE`` is centre-cropped to ``HI_SIZE`` and:

* ``hi``  goes to the library exactly as a frame would, through its own,
  freshly built face alignment and gaze estimator;
* ``lo``  is the same crop reduced to ``LO_SIZE`` with ``INTER_AREA`` and run
  through a SECOND, separate face alignment and estimator in this thread.

Both pipelines see the same gaze, head pose and blinks, which is what makes
the comparison paired: the session-to-session drift that swamps an unpaired
comparison is common to both arms. Separate instances because FaceMesh tracks
across frames, and the library's defaults are built once at import and shared.

Eye openness is a polygon AREA in pixels squared, so the same eye reads
``(HI/LO)^2`` times larger in the hi pipeline, and every threshold in this
project (``C.BLINK_THRESHOLD``) is written for 640x480. The hi alignment is
wrapped so its openness is reported in 640x480-equivalent units; both arms are
then gated by the same rule.

Nothing is stored or logged from an image: frames, crops and patches live for
one callback and are dropped. Only the per-frame outputs the recorder already
keeps (features, head ratios, openness) are recorded, for each arm.
"""

from __future__ import annotations

import statistics
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MODE = (1920, 1080)
HI_SIZE = (1440, 1080)
LO_SIZE = (640, 480)
PIPELINE_HI = "dual-hi-1440x1080-crop-of-1920x1080"
PIPELINE_LO = "dual-lo-640x480-area-of-1440x1080-crop"
# One pipeline on the crop, no second arm: the production candidate. Distinct
# from PIPELINE_HI because the dual session ran both arms at ~21 fps.
PIPELINE_HI_SINGLE = "hi-1440x1080-crop-of-1920x1080"
LIBRARY_PIPELINE = "640x480-library"
CAPTURE_ASSUMPTIONS = (
    "the camera's native 640x480 mode is a centred 4:3 crop of its 16:9 modes "
    "(supported by the IOD ratio 1.54 at 1280x720; centring not verified)",
)
# Frames kept for the recorder to look up by timestamp. The lookup happens on
# the same thread immediately after the frame is dispatched, so a handful is
# already generous; bounded so a missed lookup cannot grow memory.
_SHADOW_KEEP = 8


def centre_crop_4x3(frame: np.ndarray, size: tuple[int, int] = HI_SIZE) -> np.ndarray:
    """The centred ``size`` region of ``frame``. Refuses a frame too small."""

    width, height = size
    h, w = frame.shape[:2]
    if w < width or h < height:
        raise ValueError(f"frame {w}x{h} is smaller than the crop {width}x{height}")
    x0 = (w - width) // 2
    y0 = (h - height) // 2
    return frame[y0 : y0 + height, x0 : x0 + width]


def reduce_to_lo(crop: np.ndarray, size: tuple[int, int] = LO_SIZE) -> np.ndarray:
    import cv2  # noqa: PLC0415

    if crop.shape[1] * size[1] != crop.shape[0] * size[0]:
        raise ValueError("crop and target sizes differ in aspect; that would stretch the face")
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def openness_scale(hi: tuple[int, int] = HI_SIZE, lo: tuple[int, int] = LO_SIZE) -> float:
    """Factor that turns a hi-pipeline eye AREA into 640x480-equivalent px^2."""

    return (lo[0] / hi[0]) ** 2


class ScaledOpennessAlignment:
    """A face alignment whose eye openness is reported at another pixel scale.

    Only ``left_eye_openness`` / ``right_eye_openness`` change; landmarks and
    rectangles stay in the frame's own pixels, because the estimator crops
    from that frame.
    """

    def __init__(
        self,
        inner: Any,
        factor: float,
        sink: list[float] | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        self.inner = inner
        self.factor = float(factor)
        self.sink = sink
        self.lock = lock or threading.Lock()

    def detect(self, timestamp: int, image: np.ndarray) -> Any:
        started = time.perf_counter()
        info = self.inner.detect(timestamp, image)
        if self.sink is not None:
            with self.lock:
                self.sink.append((time.perf_counter() - started) * 1000.0)
        for name in ("left_eye_openness", "right_eye_openness"):
            value = getattr(info, name, None)
            if value is not None:
                setattr(info, name, float(value) * self.factor)
        return info

    def release(self) -> None:
        release = getattr(self.inner, "release", None)
        if release is not None:
            release()


class TimedEstimator:
    """Times the hi estimator, so hi and lo processing are the same quantity."""

    def __init__(self, inner: Any, sink: list[float], lock: threading.Lock) -> None:
        self.inner = inner
        self.sink = sink
        self.lock = lock

    def detect(self, image: np.ndarray, face_info: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.inner.detect(image, face_info)
        finally:
            with self.lock:
                self.sink.append((time.perf_counter() - started) * 1000.0)

    def release(self) -> None:
        release = getattr(self.inner, "release", None)
        if release is not None:
            release()


@dataclass
class ShadowResult:
    """What the lo pipeline produced for one frame."""

    face_info: Any
    gaze_info: Any


@dataclass
class CaptureStats:
    frames_read: int = 0
    frames_wrong_shape: int = 0
    read_failures: int = 0
    lo_errors: int = 0
    last_lo_error: str | None = None
    shapes_seen: set[tuple[int, ...]] = field(default_factory=set)
    lo_ms: list[float] = field(default_factory=list)
    # Alignment and estimator timed separately for hi, so hi and lo report
    # the same quantity; the whole library callback is kept apart.
    hi_alignment_ms: list[float] = field(default_factory=list)
    hi_estimator_ms: list[float] = field(default_factory=list)
    hi_callback_ms: list[float] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    _SERIES = ("lo_ms", "hi_alignment_ms", "hi_estimator_ms", "hi_callback_ms", "stamps")

    def reset(self) -> None:
        """A fresh window per protocol, so A and T1 are not reported pooled."""

        with self.lock:
            for name in self._SERIES:
                getattr(self, name).clear()
            self.frames_read = self.frames_wrong_shape = self.read_failures = self.lo_errors = 0
            self.shapes_seen = set()
            self.last_lo_error = None

    def summary(self) -> dict[str, Any]:
        def pct(values: list[float], q: float) -> float | None:
            return float(np.percentile(values, q)) if values else None

        with self.lock:
            snap = {name: list(getattr(self, name)) for name in self._SERIES}
            shapes = sorted(list(s) for s in self.shapes_seen)
            counts = (self.frames_read, self.frames_wrong_shape, self.read_failures, self.lo_errors)
        hi_pipeline = [
            a + e for a, e in zip(snap["hi_alignment_ms"], snap["hi_estimator_ms"], strict=False)
        ]
        gaps = [b - a for a, b in zip(snap["stamps"], snap["stamps"][1:]) if b > a]
        return {
            "frames_read": counts[0],
            "frames_wrong_shape_dropped": counts[1],
            "read_failures": counts[2],
            "lo_errors": counts[3],
            "last_lo_error": self.last_lo_error,
            "shapes_seen": shapes,
            "delivered_fps_median": (1.0 / statistics.median(gaps)) if gaps else None,
            # Both figures: face alignment + estimator, per frame.
            "lo_ms_p50": pct(snap["lo_ms"], 50),
            "lo_ms_p95": pct(snap["lo_ms"], 95),
            "hi_ms_p50": pct(hi_pipeline, 50),
            "hi_ms_p95": pct(hi_pipeline, 95),
            "hi_callback_ms_p95": pct(snap["hi_callback_ms"], 95),
            "window": "since the last reset (per protocol)",
        }


def make_dual_camera_class() -> type:
    """Built lazily: the library base class needs the library imported."""

    from gazefollower.camera import Camera  # noqa: PLC0415
    from gazefollower.misc import CameraRunningState  # noqa: PLC0415

    class DualPipelineCamera(Camera):  # type: ignore[misc]
        """Owns the device: open, set the mode, verify every frame, release.

        Not a WebCamCamera subclass on purpose. That class sets the mode
        before opening (ignored), resizes every frame, and its ``close`` only
        releases a capture that is NOT open.
        """

        def __init__(
            self, lo_alignment: Any | None, lo_estimator: Any | None, index: int = 0
        ) -> None:
            if (lo_alignment is None) != (lo_estimator is None):
                raise ValueError("the lo pipeline needs an alignment and an estimator, or neither")
            super().__init__()
            self.index = index
            self.lo_alignment = lo_alignment
            self.lo_estimator = lo_estimator
            self.stats = CaptureStats()
            self.backend: str | None = None
            self.fourcc: str | None = None
            self._cap: Any = None
            self._thread: threading.Thread | None = None
            self._running = False
            self._shadow: OrderedDict[int, ShadowResult | None] = OrderedDict()
            self._shadow_lock = threading.Lock()
            self._last_timestamp = 0

        # -- device
        def open(self) -> None:
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"camera {self.index} did not open")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, SOURCE_MODE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SOURCE_MODE[1])
            self.backend = cap.getBackendName()
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.fourcc = code.to_bytes(4, "little").decode("ascii", "replace")
            self._cap = cap
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def close(self) -> None:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=3.0)
                self._thread = None
            if self._cap is not None:
                self._cap.release()
                self._cap = None

        def release(self) -> None:
            self.close()
            for component in (self.lo_alignment, self.lo_estimator):
                if component is None:
                    continue
                release = getattr(component, "release", None)
                if release is not None:
                    release()

        def usable_frames(self) -> int:
            with self.stats.lock:
                return self.stats.frames_read - self.stats.frames_wrong_shape

        def reset_stats(self) -> None:
            self.stats.reset()

        # -- per frame
        def _loop(self) -> None:
            import cv2  # noqa: PLC0415

            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    with self.stats.lock:
                        self.stats.read_failures += 1
                    continue
                # Strictly increasing: time_ns ticks at ~15.6 ms on Windows,
                # and the shadow result is looked up by this value.
                timestamp = max(time.time_ns(), self._last_timestamp + 1)
                self._last_timestamp = timestamp
                wrong = frame.shape[:2] != (SOURCE_MODE[1], SOURCE_MODE[0])
                with self.stats.lock:
                    self.stats.frames_read += 1
                    self.stats.shapes_seen.add(tuple(frame.shape))
                    if wrong:
                        # Never resized into shape: a frame in the wrong mode
                        # is a different measurement, not a degraded one.
                        self.stats.frames_wrong_shape += 1
                    else:
                        self.stats.stamps.append(time.monotonic())
                if wrong:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hi = np.ascontiguousarray(centre_crop_4x3(rgb))
                lo = None
                if self.lo_alignment is not None:
                    lo = reduce_to_lo(hi)
                    started = time.perf_counter()
                    shadow: ShadowResult | None = None
                    try:
                        face_lo = self.lo_alignment.detect(timestamp, lo)
                        gaze_lo = self.lo_estimator.detect(lo, face_lo)
                        shadow = ShadowResult(face_lo, gaze_lo)
                    except Exception as exc:  # noqa: BLE001 - one arm failing must not stop capture
                        with self.stats.lock:
                            self.stats.lo_errors += 1
                            self.stats.last_lo_error = repr(exc)
                    elapsed = (time.perf_counter() - started) * 1000.0
                    with self._shadow_lock:
                        # Stored even when it failed, so this frame can never
                        # be paired with a stale result from another one.
                        self._shadow[timestamp] = shadow
                        while len(self._shadow) > _SHADOW_KEEP:
                            self._shadow.popitem(last=False)
                    with self.stats.lock:
                        self.stats.lo_ms.append(elapsed)
                with self.callback_and_param_lock:
                    callback = self.callback_func
                    args, kwargs = self.callback_args, self.callback_kwargs
                if callback is not None and self.camera_running_state != CameraRunningState.CLOSING:
                    started = time.perf_counter()
                    try:
                        callback(self.camera_running_state, timestamp, hi, *args, **(kwargs or {}))
                    except Exception:  # noqa: BLE001 - mirror the library: log, keep capturing
                        traceback.print_exc()
                    with self.stats.lock:
                        self.stats.hi_callback_ms.append((time.perf_counter() - started) * 1000.0)
                del frame, rgb, hi, lo  # no image outlives its frame

        def shadow_for(self, timestamp: int) -> ShadowResult | None:
            with self._shadow_lock:
                return self._shadow.pop(int(timestamp), None)

        def describe(self) -> dict[str, Any]:
            single = self.lo_alignment is None
            return {
                "pipelines": {"hi": PIPELINE_HI_SINGLE} if single else {"hi": PIPELINE_HI, "lo": PIPELINE_LO},
                "source_mode": list(SOURCE_MODE),
                "hi_size": list(HI_SIZE),
                "lo_size": list(LO_SIZE),
                "lo_interpolation": "INTER_AREA",
                "hi_openness_scale": openness_scale(),
                "backend": self.backend,
                "fourcc": self.fourcc,
                "assumptions": list(CAPTURE_ASSUMPTIONS),
                "stats": self.stats.summary(),
            }

    return DualPipelineCamera


def make_dual_components(index: int = 0) -> tuple[Any, Any, Any]:
    """(camera, hi face alignment, hi estimator), every instance fresh."""

    from gazefollower.face_alignment import MediaPipeFaceAlignment  # noqa: PLC0415
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator  # noqa: PLC0415

    camera = make_dual_camera_class()(MediaPipeFaceAlignment(), MGazeNetGazeEstimator(), index)
    stats = camera.stats
    hi_alignment = ScaledOpennessAlignment(
        MediaPipeFaceAlignment(), openness_scale(), stats.hi_alignment_ms, stats.lock
    )
    hi_estimator = TimedEstimator(MGazeNetGazeEstimator(), stats.hi_estimator_ms, stats.lock)
    return camera, hi_alignment, hi_estimator


def make_hi_components(index: int = 0) -> tuple[Any, Any, Any]:
    """(camera, face alignment, estimator) for ONE pipeline on the hi crop.

    Measured with ``gf_camera_probe.py --pipeline-rate``: 32.3 fps delivered
    and alignment + estimator p95 15.7 ms, the same rate as the library path.
    """

    from gazefollower.face_alignment import MediaPipeFaceAlignment  # noqa: PLC0415
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator  # noqa: PLC0415

    camera = make_dual_camera_class()(None, None, index)
    stats = camera.stats
    alignment = ScaledOpennessAlignment(
        MediaPipeFaceAlignment(), openness_scale(), stats.hi_alignment_ms, stats.lock
    )
    estimator = TimedEstimator(MGazeNetGazeEstimator(), stats.hi_estimator_ms, stats.lock)
    return camera, alignment, estimator


def observe_shadow(result, *, head_builder):  # noqa: ANN001, ANN201
    """The dual-capture shadow frame as a FrameObservation (ARCH-01 stage E).

    ``result`` is this module's own shadow result (or None when the shadow
    pipeline produced nothing for the primary's timestamp). Translated under
    the recorder's head policy by the same adapter the primary frame uses. The
    recorder never reads a shadow frame's time, so ``observed_s`` is 0.
    """

    from gazelink_core.domain.observation import HeadPolicy  # noqa: PLC0415
    from gazelink_core.tracking.gazefollower_source import observe  # noqa: PLC0415

    face = None if result is None else result.face_info
    gaze = None if result is None else result.gaze_info
    return observe(
        face, gaze, observed_s=0.0, head_builder=head_builder, head_policy=HeadPolicy.GAZE
    )
