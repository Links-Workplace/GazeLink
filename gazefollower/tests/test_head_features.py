"""gf_head_features on synthetic geometry: signs, monotonicity, validity, speed.

A synthetic 3-D face (the same 6-point model the PnP path uses, plus filler
points) is rotated by known angles, projected with a pinhole camera, and
written into the 478 x 3 int16 pixel layout GazeFollower's FaceInfo uses.
Two camera placements are exercised: eye level (the textbook case) and THIS
RIG's placement -- camera ~30 cm below the face, tilted up to centre it.
This proves conventions and plumbing, not accuracy on real faces.
"""

from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_head_features as H  # noqa: E402

IMG_W, IMG_H = 640, 480
FOCAL = float(IMG_W)
DISTANCE_MM = 600.0

# Model -> camera base pose: x stays, y flips (camera y is down), z flips
# (model +z points at the camera, camera +z points into the scene).
_R0 = np.diag([1.0, -1.0, -1.0])


def _rx(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Named head motions in MODEL axes (x right, y up, z toward camera).
def chin_up(deg: float) -> np.ndarray:
    return _rx(-deg)  # tips +z (forward) toward +y (up)


def turn_to_image_right(deg: float) -> np.ndarray:
    return _ry(deg)  # tips +z toward +x


def roll_to_right_shoulder(deg: float) -> np.ndarray:
    return _rz(deg)  # tips +y (up) toward -x (image left = subject's right)


def _project(
    points_mm: np.ndarray,
    head_rotation: np.ndarray,
    *,
    elevation_mm: float = 0.0,
    distance_mm: float = DISTANCE_MM,
) -> np.ndarray:
    """Pinhole projection. ``elevation_mm`` > 0 puts the face ABOVE the camera
    and tilts the camera up to centre it -- this rig's geometry."""

    cam = (_R0 @ head_rotation @ points_mm.T).T + np.array([0.0, -elevation_mm, distance_mm])
    if elevation_mm:
        tilt = math.atan2(elevation_mm, distance_mm)
        cam = (_rx(-math.degrees(tilt)) @ cam.T).T
    u = FOCAL * cam[:, 0] / cam[:, 2] + IMG_W / 2.0
    v = FOCAL * cam[:, 1] / cam[:, 2] + IMG_H / 2.0
    return np.stack([u, v], axis=1)


def synthetic_face_info(
    head_rotation: np.ndarray | None = None,
    *,
    dtype=np.int16,
    status: bool = True,
    elevation_mm: float = 0.0,
) -> SimpleNamespace:
    rotation = np.eye(3) if head_rotation is None else head_rotation
    landmarks = np.zeros((478, 3), dtype=np.float64)
    # Filler so the array is not all-zero and every index has a plausible
    # position: a coarse cloud around the face centre.
    rng = np.random.default_rng(0)
    filler = rng.normal(scale=60.0, size=(478, 3)) + np.array([0.0, -50.0, -100.0])
    landmarks[:, :2] = _project(filler, rotation, elevation_mm=elevation_mm)
    projected = _project(H._PNP_MODEL_MM, rotation, elevation_mm=elevation_mm)
    for row, index in enumerate(H._PNP_INDICES):
        landmarks[index, :2] = projected[row]
    landmarks[:, 2] = 0.0
    if dtype is np.int16:
        landmarks = np.round(landmarks).astype(np.int16)
    return SimpleNamespace(
        status=status,
        can_gaze_estimation=status,
        face_landmarks=landmarks,
        img_w=IMG_W,
        img_h=IMG_H,
    )


def _col(name: str) -> int:
    return H.HEAD6_NAMES.index(name)


def _feature(name: str, rotation: np.ndarray | None = None, **kwargs) -> float:
    vec = H.build(synthetic_face_info(rotation, **kwargs))
    assert vec is not None, "synthetic face should be valid"
    return float(vec[_col(name)])


def _strictly_increasing(values: list[float]) -> bool:
    return all(a < b for a, b in zip(values, values[1:]))


class RatioFeatureTests(unittest.TestCase):
    def test_vector_shape_and_names(self) -> None:
        vec = H.build(synthetic_face_info())
        assert vec is not None
        self.assertEqual(vec.shape, (len(H.HEAD6_NAMES),))
        self.assertEqual(len(H.HEAD6_NAMES), 6)
        self.assertEqual(H.HEAD3_NAMES, ("roll_deg", "yaw_ratio", "pitch_a"))

    def test_neutral_face_has_zero_roll_and_yaw_and_centred_eyes(self) -> None:
        self.assertAlmostEqual(_feature("roll_deg"), 0.0, delta=0.5)
        self.assertAlmostEqual(_feature("yaw_ratio"), 0.0, delta=0.01)
        self.assertAlmostEqual(_feature("eye_mid_x"), 0.5, delta=0.01)
        # Eyes sit above the nose: eye_mid.y - nose.y < 0 in image coordinates.
        self.assertLess(_feature("pitch_a"), 0.0)

    def test_roll_is_recovered_to_half_a_degree(self) -> None:
        for deg in (-30.0, -15.0, -5.0, 5.0, 15.0, 30.0):
            self.assertAlmostEqual(
                _feature("roll_deg", roll_to_right_shoulder(deg)), deg, delta=0.5, msg=f"roll {deg}"
            )

    def test_pitch_a_increases_monotonically_with_chin_up_at_eye_level(self) -> None:
        values = [_feature("pitch_a", chin_up(d)) for d in range(-25, 26, 5)]
        self.assertTrue(_strictly_increasing(values), values)

    def test_pitch_a_increases_monotonically_with_chin_up_on_this_rig(self) -> None:
        # Camera ~30 cm below the face, tilted up to centre it (plan §1).
        values = [_feature("pitch_a", chin_up(d), elevation_mm=300.0) for d in range(-25, 26, 5)]
        self.assertTrue(_strictly_increasing(values), values)

    def test_pitch_a_is_insensitive_to_roll_and_yaw(self) -> None:
        neutral = _feature("pitch_a")
        self.assertAlmostEqual(_feature("pitch_a", roll_to_right_shoulder(20)), neutral, delta=0.02)
        self.assertAlmostEqual(_feature("pitch_a", turn_to_image_right(15)), neutral, delta=0.05)

    def test_yaw_ratio_increases_monotonically_when_turning_to_image_right(self) -> None:
        for elevation in (0.0, 300.0):
            values = [
                _feature("yaw_ratio", turn_to_image_right(d), elevation_mm=elevation)
                for d in range(-25, 26, 5)
            ]
            self.assertTrue(_strictly_increasing(values), (elevation, values))
            self.assertLess(values[0], 0.0)
            self.assertGreater(values[-1], 0.0)

    def test_roll_does_not_leak_into_yaw(self) -> None:
        self.assertAlmostEqual(_feature("yaw_ratio", roll_to_right_shoulder(25)), 0.0, delta=0.02)

    def test_translation_and_distance_columns_respond(self) -> None:
        base = H.build(synthetic_face_info(dtype=np.float64))
        info = synthetic_face_info(dtype=np.float64)
        info.face_landmarks[:, 0] += 100.0
        info.face_landmarks[:, 1] += 40.0
        shifted = H.build(info)
        assert base is not None and shifted is not None
        self.assertAlmostEqual(shifted[_col("eye_mid_x")] - base[_col("eye_mid_x")], 100 / IMG_W, delta=1e-9)
        self.assertAlmostEqual(shifted[_col("eye_mid_y")] - base[_col("eye_mid_y")], 40 / IMG_H, delta=1e-9)
        # Ratios are translation-invariant.
        for name in ("roll_deg", "yaw_ratio", "pitch_a", "iod_norm"):
            self.assertAlmostEqual(shifted[_col(name)], base[_col(name)], delta=1e-9, msg=name)
        # Farther face (half the projected size) -> half the iod_norm.
        info2 = synthetic_face_info(dtype=np.float64)
        centre = np.array([IMG_W / 2.0, IMG_H / 2.0])
        info2.face_landmarks[:, :2] = centre + (info2.face_landmarks[:, :2] - centre) * 0.5
        far = H.build(info2)
        assert far is not None
        self.assertAlmostEqual(far[_col("iod_norm")], base[_col("iod_norm")] * 0.5, delta=1e-9)

    def test_select_picks_named_columns_in_order(self) -> None:
        vec = np.arange(6, dtype=np.float64)
        self.assertTrue(np.array_equal(H.select(vec, H.HEAD3_NAMES), np.array([0.0, 1.0, 2.0])))
        self.assertTrue(np.array_equal(H.select(vec, ("iod_norm", "roll_deg")), np.array([5.0, 0.0])))


class ValidityTests(unittest.TestCase):
    def test_none_when_face_missing_or_out_of_bounds(self) -> None:
        info = synthetic_face_info()
        info.status = False
        self.assertIsNone(H.build(info))
        info = synthetic_face_info()
        info.can_gaze_estimation = False
        self.assertIsNone(H.build(info))

    def test_none_on_default_zero_landmarks(self) -> None:
        info = SimpleNamespace(
            status=True,
            can_gaze_estimation=True,
            face_landmarks=np.zeros((478, 3), dtype=np.int16),
            img_w=IMG_W,
            img_h=IMG_H,
        )
        self.assertIsNone(H.build(info))
        self.assertIsNone(H.pnp_degrees(info))

    def test_none_on_too_few_landmarks_or_bad_image_size_or_missing_array(self) -> None:
        info = synthetic_face_info()
        info.face_landmarks = info.face_landmarks[:100]
        self.assertIsNone(H.build(info))
        info = synthetic_face_info()
        info.img_w = 0
        self.assertIsNone(H.build(info))
        info = synthetic_face_info()
        info.face_landmarks = None
        self.assertIsNone(H.build(info))
        info = synthetic_face_info()
        info.face_landmarks = "not an array"
        self.assertIsNone(H.build(info))

    def test_none_when_a_used_landmark_is_zero_but_others_are_not(self) -> None:
        # Partial failure: the array is not all-zero, but the nose landmark is.
        info = synthetic_face_info(dtype=np.float64)
        info.face_landmarks[H.NOSE_TIP, :2] = 0.0
        vec = H.build(info)
        # A zero nose at the image corner gives an absurd yaw/pitch ratio -> None.
        self.assertIsNone(vec)

    def test_none_on_tiny_inter_ocular_distance(self) -> None:
        info = synthetic_face_info(dtype=np.float64)
        centre = np.array([IMG_W / 2.0, IMG_H / 2.0])
        info.face_landmarks[:, :2] = centre + (info.face_landmarks[:, :2] - centre) * 0.02
        self.assertIsNone(H.build(info))

    def test_none_on_non_finite_values(self) -> None:
        info = synthetic_face_info(dtype=np.float64)
        info.face_landmarks[H.NOSE_TIP, 0] = float("nan")
        self.assertIsNone(H.build(info))
        info = synthetic_face_info(dtype=np.float64)
        info.face_landmarks[H.CHIN, 1] = float("inf")
        self.assertIsNone(H.build(info))

    def test_none_on_extreme_roll(self) -> None:
        self.assertIsNone(H.build(synthetic_face_info(roll_to_right_shoulder(60))))

    def test_none_never_replaced_by_zeros(self) -> None:
        info = synthetic_face_info()
        info.status = False
        self.assertIsNone(H.build(info))
        vec = H.build(synthetic_face_info())
        assert vec is not None
        self.assertTrue(np.any(vec != 0.0))

    def test_pnp_failure_does_not_invalidate_ratio_vector(self) -> None:
        # Ratio path and PnP path are independent products of the same input.
        info = synthetic_face_info(dtype=np.float64)
        info.face_landmarks[H.MOUTH_IMAGE_LEFT, :2] = info.face_landmarks[H.MOUTH_IMAGE_RIGHT, :2]
        info.face_landmarks[H.CHIN, :2] = info.face_landmarks[H.NOSE_TIP, :2]
        self.assertIsNotNone(H.build(info))  # ratios only use eyes + nose


class DeterminismAndSpeedTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        info = synthetic_face_info(chin_up(10))
        a = H.build(info)
        b = H.build(info)
        assert a is not None and b is not None
        self.assertTrue(np.array_equal(a, b))

    def test_builder_is_fast_enough_for_the_camera_thread(self) -> None:
        info = synthetic_face_info(chin_up(5))
        H.build(info)  # warm
        started = time.perf_counter()
        for _ in range(200):
            H.build(info)
        per_call_ms = (time.perf_counter() - started) / 200 * 1000.0
        self.assertLess(per_call_ms, 1.0, f"{per_call_ms:.3f} ms per call")

    def test_module_does_not_import_gazefollower(self) -> None:
        self.assertNotIn("gazefollower", sys.modules)


class PnPReportingTests(unittest.TestCase):
    """Sign conventions match vision.py; magnitudes are exact only because the
    synthetic face IS the solver's model. On real faces: approximate."""

    def test_neutral_is_near_zero(self) -> None:
        angles = H.pnp_degrees(synthetic_face_info())
        assert angles is not None
        for value in angles:
            self.assertAlmostEqual(value, 0.0, delta=1.5)

    def test_signs_and_magnitudes_for_pure_rotations(self) -> None:
        for deg in (-20.0, -10.0, 10.0, 20.0):
            yaw, pitch, _ = H.pnp_degrees(synthetic_face_info(turn_to_image_right(deg)))  # type: ignore[misc]
            self.assertAlmostEqual(yaw, deg, delta=2.0, msg=f"yaw {deg}")
            self.assertAlmostEqual(pitch, 0.0, delta=2.0)
            yaw, pitch, _ = H.pnp_degrees(synthetic_face_info(chin_up(deg)))  # type: ignore[misc]
            self.assertAlmostEqual(pitch, deg, delta=2.0, msg=f"pitch {deg}")
            self.assertAlmostEqual(yaw, 0.0, delta=2.0)
            _, _, roll = H.pnp_degrees(synthetic_face_info(roll_to_right_shoulder(deg)))  # type: ignore[misc]
            self.assertAlmostEqual(roll, deg, delta=2.0, msg=f"roll {deg}")

    def test_pnp_and_ratio_agree_on_pitch_sign(self) -> None:
        neutral = _feature("pitch_a")
        for deg in (-15.0, 15.0):
            angles = H.pnp_degrees(synthetic_face_info(chin_up(deg)))
            assert angles is not None
            ratio_sign = math.copysign(1.0, _feature("pitch_a", chin_up(deg)) - neutral)
            self.assertEqual(ratio_sign, math.copysign(1.0, angles[1]))


if __name__ == "__main__":
    unittest.main()
