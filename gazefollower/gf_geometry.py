"""Angular gaze error, and the honest limits of computing it on this rig.

The accuracy goal is stated in degrees, and the measurement contract forbids
turning pixels into degrees using the screen resolution alone: a degree is a
property of the triangle formed by the EYE, the target and the prediction, so
it needs an eye position with a scale somebody has checked.

Two eye-position sources, and they are not equivalent:

``nominal``    the operator's declared eye distance, eye assumed centred on
               the screen. One number, no per-frame variation. Enough for a
               head-still condition, useless for a translation condition --
               where the eye moves is exactly the variable under test.
``pnp``        per-frame, from solvePnP on the face landmarks. It does vary
               with the head, but its scale rests on a guessed focal length
               (the image width) and a generic face model, so the distance can
               be off by tens of per cent.

Neither is a verified measurement, so every angle produced here carries
``estimated=True`` and the source that produced it. Under a translation
condition the accuracy goal stays NOT VERIFIED until ``validate_scale``
below has been run -- see its docstring for the procedure.

A second, separate approximation: the library's coordinate model puts the
screen and the camera in one plane (``px2cm`` returns a 2-D offset from the
camera with no depth), while this rig's camera sits about 30 cm below the
screen and is TILTED UP. The tilt rotates the frame that solvePnP reports the
eye in, relative to the frame the screen coordinates live in. Pass
``camera_pitch_deg`` once it has been measured and the rotation is undone;
leave it None and the tilt stays an unmodelled error, recorded as such.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_head_features as H  # noqa: E402

# Mean adult inter-pupillary distance. Used only to sanity-check a PnP
# distance, never to derive one.
NOMINAL_IPD_MM = 63.0
# A face this far from the camera is not a seated user at a desk.
PLAUSIBLE_DISTANCE_MM = (250.0, 1200.0)


@dataclass(frozen=True)
class AngularError:
    degrees: float
    estimated: bool
    source: str  # "nominal" | "pnp"
    eye_mm: tuple[float, float, float]
    distance_mm: float
    tilt_corrected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "degrees": self.degrees,
            "estimated": self.estimated,
            "source": self.source,
            "distance_mm": self.distance_mm,
            "tilt_corrected": self.tilt_corrected,
        }


@dataclass
class ScreenFrame:
    """The screen in millimetres, in the camera-origin frame the labels use.

    Axes: +x right, +y UP, +z toward the viewer. The camera is the origin and
    the screen lies in the z = 0 plane -- the same simplification the library's
    ``px2cm``/``cm2px`` make. It ignores both the camera's tilt and the fact
    that a screen 30 cm above the camera is not coplanar with it, which is why
    everything here is an estimate.
    """

    rig: C.RigGeometry
    camera_pitch_deg: float | None = None  # >0 = camera tilted UP toward the face

    def target_mm(self, nx: float, ny: float) -> np.ndarray:
        cm_x, cm_y = self.rig.norm_to_label_cm(nx, ny)
        return np.array([cm_x * 10.0, cm_y * 10.0, 0.0], dtype=np.float64)

    @property
    def tilt_corrected(self) -> bool:
        return self.camera_pitch_deg is not None

    def camera_to_screen_frame(self, point_mm: np.ndarray) -> np.ndarray:
        """OpenCV camera axes (+y down, +z into the scene) -> screen axes.

        With a measured ``camera_pitch_deg`` the camera's upward tilt is undone
        first, so a point expressed in the tilted camera frame lands in the
        frame the screen coordinates use.
        """

        x, y, z = float(point_mm[0]), float(point_mm[1]), float(point_mm[2])
        vec = np.array([x, -y, z], dtype=np.float64)  # +y up, +z toward viewer
        if self.camera_pitch_deg is None:
            return vec
        a = math.radians(self.camera_pitch_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        # Rotate about +x by -pitch: undo the camera's upward tilt.
        return np.array([vec[0], cos_a * vec[1] + sin_a * vec[2], -sin_a * vec[1] + cos_a * vec[2]])


def nominal_eye_mm(frame: ScreenFrame, eye_distance_cm: float) -> np.ndarray:
    """Eye assumed centred on the screen, at the declared distance."""

    centre = frame.target_mm(0.5, 0.5)
    return np.array([centre[0], centre[1], float(eye_distance_cm) * 10.0], dtype=np.float64)


def pnp_eye_mm(face_info: Any, frame: ScreenFrame) -> np.ndarray | None:
    """Per-frame eye midpoint in the screen frame, from solvePnP. Estimated.

    Returns None when the pose cannot be solved or lands outside a plausible
    seated distance -- a wrong position must not quietly produce a confident
    angle.
    """

    import cv2  # noqa: PLC0415

    validated = H._landmarks_xy(face_info)
    if validated is None:
        return None
    xy, img_w, img_h = validated
    image_points = np.ascontiguousarray(xy[list(H._PNP_INDICES)], dtype=np.float64)
    focal = float(img_w)
    camera = np.array([[focal, 0.0, img_w / 2.0], [0.0, focal, img_h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            H._PNP_MODEL_MM, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    # Midpoint of the two outer eye corners, in the model, mapped to the camera.
    # Note this is the distance to the EYES, not to the nose tip: the generic
    # model puts the eye corners 135 mm behind the nose, and that depth is a
    # property of the model rather than a measurement of this face -- one more
    # reason the result is an estimate.
    eye_model = (H._PNP_MODEL_MM[2] + H._PNP_MODEL_MM[3]) / 2.0
    eye_cam = (rotation @ eye_model.reshape(3, 1) + tvec).ravel()
    if not np.all(np.isfinite(eye_cam)):
        return None
    distance = float(abs(eye_cam[2]))
    if not (PLAUSIBLE_DISTANCE_MM[0] <= distance <= PLAUSIBLE_DISTANCE_MM[1]):
        return None
    return frame.camera_to_screen_frame(eye_cam)


def angle_between(eye_mm: np.ndarray, a_mm: np.ndarray, b_mm: np.ndarray) -> float | None:
    """Degrees between the rays eye->a and eye->b."""

    va, vb = np.asarray(a_mm) - np.asarray(eye_mm), np.asarray(b_mm) - np.asarray(eye_mm)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na <= 0 or nb <= 0:
        return None
    cos = float(np.dot(va, vb) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def angular_error(
    frame: ScreenFrame,
    target_norm: tuple[float, float],
    predicted_norm: tuple[float, float],
    *,
    eye_mm: np.ndarray | None = None,
    eye_distance_cm: float | None = None,
    source: str = "nominal",
) -> AngularError | None:
    """The angle between where the person was told to look and the prediction.

    Always ``estimated``: no eye-position source available on this rig has a
    verified scale. See :func:`validate_scale`.
    """

    if eye_mm is None:
        if eye_distance_cm is None:
            return None
        eye_mm = nominal_eye_mm(frame, eye_distance_cm)
        source = "nominal"
    target = frame.target_mm(*target_norm)
    predicted = frame.target_mm(*predicted_norm)
    degrees = angle_between(eye_mm, target, predicted)
    if degrees is None:
        return None
    return AngularError(
        degrees=degrees,
        estimated=True,
        source=source,
        eye_mm=(float(eye_mm[0]), float(eye_mm[1]), float(eye_mm[2])),
        distance_mm=float(abs(eye_mm[2])),
        tilt_corrected=frame.tilt_corrected,
    )


def angular_errors_for_rows(
    frame: ScreenFrame,
    target_norm: np.ndarray,
    predicted_norm: np.ndarray,
    *,
    eye_mm_rows: np.ndarray | None = None,
    eye_distance_cm: float | None = None,
) -> np.ndarray:
    """Per-row angular error in degrees; NaN where it cannot be computed."""

    n = len(target_norm)
    out = np.full(n, np.nan)
    fallback = None
    if eye_distance_cm is not None:
        fallback = nominal_eye_mm(frame, eye_distance_cm)
    for i in range(n):
        eye = None
        if eye_mm_rows is not None and np.all(np.isfinite(eye_mm_rows[i])):
            eye = eye_mm_rows[i]
            source = "pnp"
        elif fallback is not None:
            eye = fallback
            source = "nominal"
        if eye is None:
            continue
        if not (np.all(np.isfinite(target_norm[i])) and np.all(np.isfinite(predicted_norm[i]))):
            continue
        result = angular_error(
            frame,
            (float(target_norm[i][0]), float(target_norm[i][1])),
            (float(predicted_norm[i][0]), float(predicted_norm[i][1])),
            eye_mm=eye,
            source=source,
        )
        if result is not None:
            out[i] = result.degrees
    return out


def summarise_degrees(degrees: np.ndarray, *, source: str, tilt_corrected: bool, scale_validated: bool) -> dict[str, Any]:
    """Angle statistics, always labelled with what they rest on.

    ``verified`` stays False until the scale has been validated AND the camera
    tilt modelled, so a reader can never mistake an estimate for a measurement.
    """

    values = np.asarray(degrees, dtype=np.float64)
    values = values[np.isfinite(values)]
    summary: dict[str, Any] = {
        "n": int(values.size),
        "source": source,
        "estimated": True,
        "tilt_corrected": tilt_corrected,
        "scale_validated": scale_validated,
        "verified": bool(scale_validated and tilt_corrected),
        "caveat": (
            "Angles rest on an ESTIMATED eye position. The accuracy goal stays "
            "NOT VERIFIED under translation conditions until validate_scale has "
            "been run and the camera tilt measured."
        ),
    }
    if values.size:
        summary.update(
            {
                "mean_deg": float(np.mean(values)),
                "median_deg": float(np.median(values)),
                "p90_deg": float(np.percentile(values, 90)),
                "p95_deg": float(np.percentile(values, 95)),
                "max_deg": float(np.max(values)),
            }
        )
    return summary


def validate_scale(measured_distances_mm: Sequence[float], true_distances_mm: Sequence[float]) -> dict[str, Any]:
    """Compare PnP distances against a tape measure, on this rig.

    Procedure, once, before any run whose angles are meant to be verified:
      1. Seat the subject at a measured distance from the camera (tape from
         the camera lens to the bridge of the nose). Three distances spanning
         the working range, e.g. 500, 600, 700 mm.
      2. At each, record a few seconds and take the median ``distance_mm``
         that :func:`pnp_eye_mm` reports.
      3. Pass both lists here.

    A scale factor consistently off by more than a few per cent means the
    guessed focal length is wrong; the factor returned can then correct the
    distances, and only then may an angle be called measured rather than
    estimated.
    """

    measured = np.asarray(measured_distances_mm, dtype=np.float64)
    truth = np.asarray(true_distances_mm, dtype=np.float64)
    if measured.shape != truth.shape or measured.size < 2:
        raise ValueError("need at least two paired distances of the same length")
    if np.any(measured <= 0) or np.any(truth <= 0):
        raise ValueError("distances must be positive")
    ratios = truth / measured
    scale = float(np.median(ratios))
    residual = np.abs(measured * scale - truth)
    return {
        "scale_factor": scale,
        "ratios": [float(r) for r in ratios],
        "max_residual_mm": float(np.max(residual)),
        "mean_abs_residual_mm": float(np.mean(residual)),
        "n": int(measured.size),
        "consistent": bool(np.max(np.abs(ratios - scale)) < 0.05 * scale),
        "note": (
            "A consistent scale factor makes PnP distances usable after multiplying "
            "by it. An inconsistent one means the pinhole assumption or the generic "
            "face model does not hold well enough here, and angles stay estimated."
        ),
    }
