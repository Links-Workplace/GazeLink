"""Fake library components for ``gf_record --dry-run`` (ARCH-01 stage E).

A camera, face aligner and estimator shaped exactly like the gazefollower
library's, so a dry run exercises the REAL ``GazeFollower.process_frame`` with
no camera and no models. They construct the library's own ``FaceInfo`` and
``GazeInfo`` objects, which is why they live beside the adapter that reads
them. Moved verbatim from ``gf_record``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from gazelink_core.gaze import head_features as H

# The model embedding width the fake estimator produces (the library's 258).
EXPECTED_FEATURE_DIM = 258


def make_dry_run_components(fps: float = 30.0) -> tuple[Any, Any, Any]:
    from gazefollower.camera import Camera  # noqa: PLC0415
    from gazefollower.misc import (  # noqa: PLC0415
        CameraRunningState,
        FaceInfo,
        GazeInfo,
        TrackingState,
    )

    class FakeCamera(Camera):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._thread: threading.Thread | None = None
            self._running = False
            self._camera_thread = None
            self._camera_thread_running = None
            self._cap = None

        def open(self) -> None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._camera_thread = self._thread
            self._camera_thread_running = True
            self._thread.start()

        def _loop(self) -> None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            while self._running and self._camera_thread_running:
                with self.callback_and_param_lock:
                    cb = self.callback_func
                if cb is not None and self.camera_running_state != CameraRunningState.CLOSING:
                    cb(
                        self.camera_running_state,
                        time.time_ns(),
                        frame,
                        *self.callback_args,
                        **self.callback_kwargs,
                    )
                time.sleep(1.0 / fps)

        def close(self) -> None:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)

        def release(self) -> None:
            self.close()

    class FakeFaceAlignment:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(1)
            self._t = 0.0

        def detect(self, timestamp: int, image: np.ndarray) -> Any:
            self._t += 0.05
            info = FaceInfo()
            info.timestamp = timestamp
            info.status = True
            info.can_gaze_estimation = True
            info.img_w, info.img_h = image.shape[1], image.shape[0]
            lm = np.zeros((478, 3), dtype=np.float64)
            lm[:, 0] = 320 + self._rng.normal(scale=40, size=478)
            lm[:, 1] = 240 + self._rng.normal(scale=40, size=478)
            nod = 12.0 * math.sin(self._t)
            lm[H.EYE_OUTER_IMAGE_LEFT, :2] = (250, 200)
            lm[H.EYE_OUTER_IMAGE_RIGHT, :2] = (390, 200)
            lm[H.NOSE_TIP, :2] = (320, 260 + nod)
            lm[H.CHIN, :2] = (320, 340 + nod)
            lm[H.MOUTH_IMAGE_LEFT, :2] = (290, 305 + nod)
            lm[H.MOUTH_IMAGE_RIGHT, :2] = (350, 305 + nod)
            info.face_landmarks = np.round(lm).astype(np.int16)
            info.face_rect = np.array([230, 150, 180, 220])
            info.left_rect = np.array([240, 185, 60, 30])
            info.right_rect = np.array([340, 185, 60, 30])
            info.left_eye_openness = 120.0
            info.right_eye_openness = 110.0
            return info

        def release(self) -> None:
            return None

    class FakeEstimator:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(2)

        def detect(self, image: np.ndarray, face_info: Any) -> Any:
            info = GazeInfo()
            info.timestamp = face_info.timestamp
            if not face_info.status:
                info.tracking_state = TrackingState.FACE_MISSING
                return info
            info.features = self._rng.normal(size=EXPECTED_FEATURE_DIM).astype(np.float32)
            info.raw_gaze_coordinates = info.features[:2]
            info.status = True
            info.left_openness = face_info.left_eye_openness
            info.right_openness = face_info.right_eye_openness
            info.tracking_state = TrackingState.SUCCESS
            return info

        def release(self) -> None:
            return None

    return FakeCamera(), FakeFaceAlignment(), FakeEstimator()
