"""Head-pose features from the face landmarks GazeFollower already computes.

Two products, from ONE input (the ``FaceInfo`` the library hands every
subscriber and every estimator):

``build(face_info)`` -> the vector appended to the calibration features.
    Landmark RATIOS, not angles. For an RBF SVR on standardised columns a
    monotone proxy carries the same information as the angle, and a ratio
    needs no guessed camera intrinsics, no hand-coded 3-D face model, no
    Euler-angle convention and has no solver failure mode. It is also
    trivially testable on synthetic geometry.

``pnp_degrees(face_info)`` -> approximate (yaw, pitch, roll) in degrees.
    For REPORTING only (the ``head_*_deg`` fields analyze.py reads, and the
    "pose range per measurement" rule in docs/MEASUREMENT_DAY.md). Generic
    6-point model, focal length guessed as the image width: expect a
    20-30 % scale error on real faces. Never fed to the SVR.

Both return ``None`` when the input is not trustworthy. A missing or
degenerate head measurement is a tracking loss for that frame, never a zero
and never the previous frame's value: the caller drops the frame from
calibration AND from prediction, so the two paths stay identical.

Sign conventions follow the project's ``vision.py``: pitch > 0 chin up,
yaw > 0 face turned toward the camera's right (nose moves toward image
right in an unmirrored frame), roll > 0 head tipped toward the subject's
right shoulder. For the SVR the sign is irrelevant; it matters for reports.

The camera on this rig sits below the screen and looks up, so every pitch
proxy carries a large constant offset. After per-column standardisation
learned from the calibration rows that offset is invisible to the model --
but a model fitted with one mount must not be used with another, which is
why the schema records the rig geometry (``gf_schema.py``).

No ``gazefollower`` import: anything with ``status``,
``can_gaze_estimation``, ``face_landmarks`` (N x 3, pixels), ``img_w`` and
``img_h`` attributes is accepted, so tests use a plain namespace.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Bump whenever the meaning, order or validity rules of the vector change.
# A saved model whose schema carries a different version must refuse to load.
BUILDER_VERSION = "head-ratios-2"

HEAD6_NAMES: tuple[str, ...] = (
    "roll_deg",  # in-plane tilt of the outer-eye-corner line, degrees
    "yaw_ratio",  # nose-tip horizontal offset from the eye midpoint, / iod
    "pitch_a",  # nose-tip vertical offset (eye_mid.y - nose.y), / iod
    "eye_mid_x",  # eye midpoint, fraction of image width   (translation)
    "eye_mid_y",  # eye midpoint, fraction of image height  (translation)
    "iod_norm",  # inter-ocular distance, fraction of image width (distance)
)
HEAD3_NAMES: tuple[str, ...] = HEAD6_NAMES[:3]
# A chin-based pitch proxy was tried and dropped: under perspective the chin
# moves TOWARD the camera when the chin lifts and its projected offset grows
# faster than its rise, so the ratio is not a monotone pitch proxy even on a
# synthetic face (tests/test_head_features.py history). The nose tip, being
# near the rotation centre in depth, does not have that problem.

# MediaPipe Face Mesh canonical indices. All < 468, so the ten iris points
# that refine_landmarks appends are irrelevant here.
NOSE_TIP = 1
CHIN = 152
EYE_OUTER_IMAGE_LEFT = 33  # subject's RIGHT eye, on the image's LEFT when unmirrored
EYE_OUTER_IMAGE_RIGHT = 263  # subject's LEFT eye
MOUTH_IMAGE_LEFT = 61
MOUTH_IMAGE_RIGHT = 291
REQUIRED_LANDMARK_COUNT = 468

# Validity bounds. Outside these the geometry is not a face we trust.
MIN_IOD_PX = 20.0
MAX_ABS_ROLL_DEG = 45.0
YAW_RATIO_BOUNDS = (-1.5, 1.5)
PITCH_A_BOUNDS = (-2.0, 1.0)
EYE_MID_BOUNDS = (-0.2, 1.2)

# Generic 3-D face model for pnp_degrees, millimetres. Model axes: x toward
# image right, y UP, z toward the camera (nose tip is the closest point).
_PNP_MODEL_MM = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -330.0, -65.0],  # chin
        [-225.0, 170.0, -135.0],  # eye outer corner, image left
        [225.0, 170.0, -135.0],  # eye outer corner, image right
        [-150.0, -150.0, -125.0],  # mouth corner, image left
        [150.0, -150.0, -125.0],  # mouth corner, image right
    ],
    dtype=np.float64,
)
_PNP_INDICES = (
    NOSE_TIP,
    CHIN,
    EYE_OUTER_IMAGE_LEFT,
    EYE_OUTER_IMAGE_RIGHT,
    MOUTH_IMAGE_LEFT,
    MOUTH_IMAGE_RIGHT,
)
# Every landmark either product reads; validity is checked on all of them
# once so the ratio and PnP paths see the same accept/reject decision.
_USED_INDICES = _PNP_INDICES


def _landmarks_xy(face_info: Any) -> tuple[np.ndarray, int, int] | None:
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


def build_from_landmarks(xy: np.ndarray, img_w: int, img_h: int) -> np.ndarray | None:
    """The head6 vector from validated pixel landmarks, or None if degenerate."""

    left = xy[EYE_OUTER_IMAGE_LEFT]
    right = xy[EYE_OUTER_IMAGE_RIGHT]
    nose = xy[NOSE_TIP]

    dx = right[0] - left[0]
    dy = right[1] - left[1]
    iod = math.hypot(dx, dy)
    if iod < MIN_IOD_PX:
        return None

    # Image y points down. A head tipped toward the subject's right shoulder
    # lowers the image-left eye (subject's right) and raises the image-right
    # one, so dy < 0; negate to get roll > 0 for that tilt.
    roll = -math.atan2(dy, dx)
    roll_deg = math.degrees(roll)
    if abs(roll_deg) > MAX_ABS_ROLL_DEG:
        return None

    eye_mid = (left + right) / 2.0
    # De-rotate about the eye midpoint so yaw/pitch proxies do not read roll.
    cos_r, sin_r = math.cos(-roll), math.sin(-roll)

    def derotate(p: np.ndarray) -> tuple[float, float]:
        vx, vy = p[0] - eye_mid[0], p[1] - eye_mid[1]
        # Rotating by +roll undoes the measured tilt (the eye line becomes
        # horizontal); in image coordinates that is a rotation by -roll here.
        return (vx * cos_r + vy * sin_r, -vx * sin_r + vy * cos_r)

    nose_x, nose_y = derotate(nose)

    yaw_ratio = nose_x / iod
    pitch_a = -nose_y / iod  # eye_mid.y - nose.y : increases chin-up

    eye_mid_x = eye_mid[0] / img_w
    eye_mid_y = eye_mid[1] / img_h
    iod_norm = iod / img_w

    vector = np.array(
        [roll_deg, yaw_ratio, pitch_a, eye_mid_x, eye_mid_y, iod_norm],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(vector)):
        return None
    if not (YAW_RATIO_BOUNDS[0] <= yaw_ratio <= YAW_RATIO_BOUNDS[1]):
        return None
    if not (PITCH_A_BOUNDS[0] <= pitch_a <= PITCH_A_BOUNDS[1]):
        return None
    for value in (eye_mid_x, eye_mid_y):
        if not (EYE_MID_BOUNDS[0] <= value <= EYE_MID_BOUNDS[1]):
            return None
    return vector


def build(face_info: Any) -> np.ndarray | None:
    """head6 for this frame, or None (tracking loss). The ONE builder.

    The recorder and the live wrapper must both call this exact function
    object; a test asserts the identity.
    """

    validated = _landmarks_xy(face_info)
    if validated is None:
        return None
    xy, img_w, img_h = validated
    return build_from_landmarks(xy, img_w, img_h)


def select(head6: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Pick named columns out of a head6 vector (e.g. HEAD3_NAMES)."""

    index = {name: i for i, name in enumerate(HEAD6_NAMES)}
    return np.asarray([head6[index[name]] for name in names], dtype=np.float64)


def pnp_degrees(face_info: Any) -> tuple[float, float, float] | None:
    """Approximate (yaw, pitch, roll) degrees via solvePnP. Reporting only."""

    validated = _landmarks_xy(face_info)
    if validated is None:
        return None
    xy, img_w, img_h = validated
    return pnp_degrees_from_landmarks(xy, img_w, img_h)


def pnp_degrees_from_landmarks(
    xy: np.ndarray, img_w: int, img_h: int
) -> tuple[float, float, float] | None:
    import cv2  # local: keeps module import cheap and the dependency explicit

    image_points = np.ascontiguousarray(xy[list(_PNP_INDICES)], dtype=np.float64)
    focal = float(img_w)
    camera = np.array(
        [[focal, 0.0, img_w / 2.0], [0.0, focal, img_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, _tvec = cv2.solvePnP(
            _PNP_MODEL_MM, image_points, camera, dist, flags=cv2.SOLVEPNP_SQPNP
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return _angles_from_rotation(rotation)


def _angles_from_rotation(rotation: np.ndarray) -> tuple[float, float, float] | None:
    """Project-convention angles from a model->camera rotation.

    OpenCV camera axes: x right, y DOWN, z into the scene. The model's
    forward axis (+z, toward the camera) maps to about (0, 0, -1) for a face
    looking straight at the lens; its up axis (+y) maps to about (0, -1, 0).
    """

    forward = rotation @ np.array([0.0, 0.0, 1.0])
    up = rotation @ np.array([0.0, 1.0, 0.0])
    if not (np.all(np.isfinite(forward)) and np.all(np.isfinite(up))):
        return None
    yaw = math.atan2(forward[0], -forward[2])
    pitch = math.atan2(-forward[1], math.hypot(forward[0], forward[2]))
    roll = math.atan2(-up[0], -up[1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
