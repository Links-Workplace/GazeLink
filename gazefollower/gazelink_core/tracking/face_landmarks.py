"""Head pose from the library's FaceInfo landmarks (ARCH-01 stage E).

The part of ``gaze.head_features`` that READS the library's face object: it
validates the landmarks and hands plain arrays to the pure builders in
``head_features``. Anything with ``status``, ``can_gaze_estimation``,
``face_landmarks`` (N x 3 pixels), ``img_w`` and ``img_h`` is accepted, so tests
still use a plain namespace. Moved verbatim; the numbers are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gazelink_core.gaze.head_features import (
    _USED_INDICES,
    REQUIRED_LANDMARK_COUNT,
    build_from_landmarks,
    pnp_degrees_from_landmarks,
)


def raw_landmarks(face_info: Any) -> Any:
    """The library's landmark array for a detected face, unvalidated; None otherwise."""

    marks = getattr(face_info, "face_landmarks", None)
    if not getattr(face_info, "status", False) or marks is None:
        return None
    return marks


def landmarks_xy(face_info: Any) -> tuple[np.ndarray, int, int] | None:
    """Validated (N x 2) float64 pixel landmarks, or None."""

    if not getattr(face_info, "status", False):
        return None
    if not getattr(face_info, "can_gaze_estimation", False):
        return None
    img_w = getattr(face_info, "img_w", 0)
    img_h = getattr(face_info, "img_h", 0)
    try:
        img_w = int(img_w)
        img_h = int(img_h)
    except (TypeError, ValueError):
        return None
    if img_w <= 0 or img_h <= 0:
        return None
    raw = getattr(face_info, "face_landmarks", None)
    if raw is None:
        return None
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < REQUIRED_LANDMARK_COUNT or points.shape[1] < 2:
        return None
    xy = points[:, :2]
    if not np.all(np.isfinite(xy)):
        return None
    if not np.any(xy):  # FaceInfo's default is an all-zero array
        return None
    # A landmark we actually use sitting EXACTLY on the image's top-left pixel
    # is a partial failure, not a face; the ratio bounds would not catch it.
    for index in _USED_INDICES:
        if not np.any(xy[index]):
            return None
    return xy, img_w, img_h


def head6_from_face(face_info: Any) -> np.ndarray | None:
    """head6 for this frame, or None (tracking loss). The ONE builder.

    The recorder and the live wrapper must both call this exact function
    object; a test asserts the identity.
    """

    validated = landmarks_xy(face_info)
    if validated is None:
        return None
    xy, img_w, img_h = validated
    return build_from_landmarks(xy, img_w, img_h)


def pnp_from_face(face_info: Any) -> tuple[float, float, float] | None:
    """Approximate (yaw, pitch, roll) degrees via solvePnP. Reporting only."""

    validated = landmarks_xy(face_info)
    if validated is None:
        return None
    xy, img_w, img_h = validated
    return pnp_degrees_from_landmarks(xy, img_w, img_h)
