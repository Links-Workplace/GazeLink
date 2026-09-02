from __future__ import annotations

from pathlib import Path

import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
import pytest
from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]
from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = Path(".gazelink/models/face_landmarker.task")


@pytest.mark.integration
def test_face_landmarker_processes_synthetic_no_face_frame() -> None:
    if not MODEL_PATH.exists():
        pytest.skip("run scripts/download_models.py to enable the model integration test")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect_for_video(image, timestamp_ms=0)

    assert result.face_landmarks == []
    assert result.face_blendshapes == []
    assert result.facial_transformation_matrixes == []
