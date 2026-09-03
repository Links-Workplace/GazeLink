from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from gazelink.config import ConfidenceThresholds
from gazelink.domain import FramePacket, PixelFormat, ReasonCode, TrackingState
from gazelink.vision import (
    MEDIAPIPE_LEFT_IRIS_INDICES,
    MEDIAPIPE_RIGHT_IRIS_INDICES,
    STRUCTURAL_CONFIDENCE,
    FaceLandmarkerAdapter,
    _frame_to_mediapipe_image,
    head_pose_from_transformation_matrix,
)


@dataclass(frozen=True, slots=True)
class FakeLandmark:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class FakeResult:
    face_landmarks: tuple[tuple[FakeLandmark, ...], ...]
    facial_transformation_matrixes: tuple[tuple[tuple[float, ...], ...], ...] = ()


class FakeLandmarker:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.closed = 0
        self.timestamps: list[int] = []

    def detect_for_video(self, image: object, timestamp_ms: int) -> FakeResult:
        assert image == "synthetic-image"
        self.timestamps.append(timestamp_ms)
        return self.result

    def close(self) -> None:
        self.closed += 1


def _frame(*, frame_id: int = 3) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        captured_at_monotonic_ms=104.7,
        width=2,
        height=2,
        pixel_format=PixelFormat.RGB24,
        image=b"\x00" * 12,
    )


def _face(
    *,
    missing: set[int] | None = None,
    lid_half_gap: float = 0.03,
    iris_offset: float = 0.0,
) -> tuple[FakeLandmark, ...]:
    """Build a synthetic Face Mesh landmark set with a usable eye geometry.

    The subject's right eye sits on the image left (outer corner 33), the left
    eye on the image right (outer corner 263), which is the mirroring a real
    front-facing camera produces.  ``lid_half_gap`` shrinks toward zero to model
    a closing eye; ``iris_offset`` slides the iris toward the inner corner.
    """

    missing = missing or set()
    landmarks = [FakeLandmark(0.4, 0.4) for _ in range(478)]
    landmarks[0] = FakeLandmark(0.2, 0.1)
    landmarks[1] = FakeLandmark(0.8, 0.9)
    # Right eye: outer 33 at image-left, inner 133 toward the nose.
    landmarks[33] = FakeLandmark(0.31, 0.4)
    landmarks[133] = FakeLandmark(0.43, 0.4)
    landmarks[159] = FakeLandmark(0.37, 0.4 - lid_half_gap)
    landmarks[145] = FakeLandmark(0.37, 0.4 + lid_half_gap)
    for index in MEDIAPIPE_RIGHT_IRIS_INDICES:
        landmarks[index] = FakeLandmark(0.37 + iris_offset, 0.4)
    # Left eye: outer 263 at image-right, inner 362 toward the nose.
    landmarks[263] = FakeLandmark(0.69, 0.4)
    landmarks[362] = FakeLandmark(0.57, 0.4)
    landmarks[386] = FakeLandmark(0.63, 0.4 - lid_half_gap)
    landmarks[374] = FakeLandmark(0.63, 0.4 + lid_half_gap)
    for index in MEDIAPIPE_LEFT_IRIS_INDICES:
        landmarks[index] = FakeLandmark(0.63 - iris_offset, 0.4)
    for index in missing:
        landmarks[index] = FakeLandmark(float("nan"), float("nan"))
    return tuple(landmarks)


def _identity_matrix() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_adapter_maps_one_face_without_exposing_provider_types() -> None:
    landmarker = FakeLandmarker(FakeResult((_face(),), (_identity_matrix(),)))
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame())

    assert observation.tracking_state is TrackingState.TRACKED
    assert observation.face_box is not None
    assert observation.face_box.x == pytest.approx(0.2)
    assert observation.face_box.y == pytest.approx(0.1)
    assert observation.face_box.width == pytest.approx(0.6)
    assert observation.face_box.height == pytest.approx(0.8)
    assert observation.left_eye is not None
    assert observation.right_eye is not None
    # iris_center stays in frame coordinates so the overlay can draw it, while
    # iris_in_eye is eye-relative and therefore identical for two centred eyes.
    assert observation.left_eye.iris_center is not None
    assert observation.left_eye.iris_center.x == pytest.approx(0.63)
    assert observation.right_eye.iris_center is not None
    assert observation.right_eye.iris_center.x == pytest.approx(0.37)
    assert observation.left_eye.iris_in_eye is not None
    assert observation.left_eye.iris_in_eye.x == pytest.approx(0.5)
    assert observation.left_eye.iris_in_eye.y == pytest.approx(0.5)
    assert observation.right_eye.iris_in_eye is not None
    assert observation.right_eye.iris_in_eye.x == pytest.approx(0.5)
    assert observation.right_eye.iris_in_eye.y == pytest.approx(0.5)
    # Openness is the lid gap (0.06) over the eye width (0.12).
    assert observation.left_eye.openness == pytest.approx(0.5)
    assert observation.right_eye.openness == pytest.approx(0.5)
    assert observation.head_pose is not None
    assert observation.head_pose.to_dict() == {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    assert observation.overall_confidence == STRUCTURAL_CONFIDENCE
    assert "face_landmarks" not in observation.to_dict()
    assert landmarker.timestamps == [105]


def test_iris_moving_toward_the_nose_raises_eye_relative_x_for_both_eyes() -> None:
    landmarker = FakeLandmarker(
        FakeResult((_face(iris_offset=0.03),), (_identity_matrix(),)),
    )
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame())

    assert observation.left_eye is not None
    assert observation.right_eye is not None
    assert observation.left_eye.iris_in_eye is not None
    assert observation.right_eye.iris_in_eye is not None
    # The eye-relative axis runs outer -> inner, so both eyes agree on
    # direction even though they mirror each other in frame coordinates.
    assert observation.left_eye.iris_in_eye.x == pytest.approx(0.75)
    assert observation.right_eye.iris_in_eye.x == pytest.approx(0.75)
    assert observation.left_eye.iris_center.x == pytest.approx(0.60)  # type: ignore[union-attr]
    assert observation.right_eye.iris_center.x == pytest.approx(0.40)  # type: ignore[union-attr]


def test_adapter_output_can_never_reach_the_control_threshold() -> None:
    """The adapter has no calibrated confidence, so it must not authorize control."""

    thresholds = ConfidenceThresholds()
    landmarker = FakeLandmarker(FakeResult((_face(),), (_identity_matrix(),)))
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame())

    assert observation.tracking_state is TrackingState.TRACKED
    assert observation.overall_confidence >= thresholds.display
    assert observation.overall_confidence < thresholds.calibration
    assert observation.overall_confidence < thresholds.control


def test_closed_eye_is_reported_as_unusable_rather_than_confidently_open() -> None:
    """A shut eye must not yield a gaze-bearing iris position."""

    landmarker = FakeLandmarker(
        FakeResult((_face(lid_half_gap=1e-12),), (_identity_matrix(),)),
    )
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame())

    assert observation.tracking_state is TrackingState.LOW_CONFIDENCE
    assert observation.left_eye is None
    assert observation.right_eye is None
    assert observation.overall_confidence == 0.0


def test_no_face_is_lost_and_cannot_replay_a_previous_observation() -> None:
    landmarker = FakeLandmarker(FakeResult(()))
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame(frame_id=4))

    assert observation.frame_id == 4
    assert observation.tracking_state is TrackingState.LOST
    assert observation.reason_codes == (ReasonCode.FACE_NOT_FOUND,)
    assert observation.face_box is None
    assert observation.left_eye is None


def test_multiple_faces_is_an_explicit_safe_state() -> None:
    landmarker = FakeLandmarker(FakeResult((_face(), _face())))
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame())

    assert observation.tracking_state is TrackingState.MULTIPLE_FACES
    assert observation.reason_codes == (ReasonCode.MULTIPLE_FACES,)
    assert observation.face_box is None


def test_missing_eye_landmarks_produce_low_confidence_not_a_stale_eye() -> None:
    landmarker = FakeLandmarker(FakeResult((_face(missing={473}),), (_identity_matrix(),)))
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    observation = adapter.observe(_frame())

    assert observation.tracking_state is TrackingState.LOW_CONFIDENCE
    assert observation.left_eye is None
    assert observation.right_eye is not None
    assert observation.reason_codes == (ReasonCode.LEFT_EYE_OCCLUDED,)
    assert observation.overall_confidence == 0.0


def test_close_is_idempotent_and_prevents_later_processing() -> None:
    landmarker = FakeLandmarker(FakeResult(()))
    adapter = FaceLandmarkerAdapter(
        landmarker=landmarker, image_factory=lambda _frame: "synthetic-image"
    )

    adapter.close()
    adapter.close()

    assert landmarker.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        adapter.observe(_frame())


def test_bgr_frame_buffer_is_converted_to_short_lived_rgb_image() -> None:
    frame = FramePacket(
        frame_id=1,
        captured_at_monotonic_ms=1.0,
        width=1,
        height=1,
        pixel_format=PixelFormat.BGR24,
        image=bytes((5, 7, 9)),
    )

    image = _frame_to_mediapipe_image(frame)

    assert image.numpy_view().tolist() == [[[9, 7, 5]]]  # type: ignore[attr-defined]


def test_malformed_frame_becomes_fresh_lost_error_without_model_call() -> None:
    landmarker = FakeLandmarker(FakeResult((_face(),), (_identity_matrix(),)))
    adapter = FaceLandmarkerAdapter(landmarker=landmarker)
    malformed = FramePacket(
        frame_id=7,
        captured_at_monotonic_ms=10.0,
        width=2,
        height=2,
        pixel_format=PixelFormat.RGB24,
        image=b"too-short",
    )

    observation = adapter.observe(malformed)

    assert observation.frame_id == 7
    assert observation.tracking_state is TrackingState.LOST
    assert observation.reason_codes == (ReasonCode.ERROR,)
    assert landmarker.timestamps == []


def _face_matrix(
    pitch_up_deg: float, yaw_deg: float, roll_deg: float
) -> tuple[tuple[float, ...], ...]:
    """Build a transform whose columns are the face axes, as MediaPipe supplies.

    Rotations are applied yaw, then pitch, then roll, matching the order the
    angles are read back in.  ``pitch_up_deg`` is positive for chin up, so the
    rotation about X is negated: a positive right-handed rotation about +X
    tips the forward axis DOWN.
    """

    pitch = math.radians(-pitch_up_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)
    yaw_matrix = (
        (math.cos(yaw), 0.0, math.sin(yaw)),
        (0.0, 1.0, 0.0),
        (-math.sin(yaw), 0.0, math.cos(yaw)),
    )
    pitch_matrix = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(pitch), -math.sin(pitch)),
        (0.0, math.sin(pitch), math.cos(pitch)),
    )
    roll_matrix = (
        (math.cos(roll), -math.sin(roll), 0.0),
        (math.sin(roll), math.cos(roll), 0.0),
        (0.0, 0.0, 1.0),
    )

    def multiply(
        left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
            for row in range(3)
        )

    rotation = multiply(multiply(yaw_matrix, pitch_matrix), roll_matrix)
    # Translation mirrors a real capture: the face sits in front of the camera.
    return tuple(
        tuple(rotation[row]) + (translation,) for row, translation in enumerate((0.0, 0.0, -38.0))
    ) + ((0.0, 0.0, 0.0, 1.0),)


def test_pose_reads_zero_for_a_face_aimed_at_the_camera() -> None:
    pose = head_pose_from_transformation_matrix(_identity_matrix())

    assert pose is not None
    assert pose.yaw_deg == 0.0
    assert pose.pitch_deg == 0.0
    assert pose.roll_deg == 0.0


@pytest.mark.parametrize(
    ("pitch_up_deg", "yaw_deg", "roll_deg"),
    [
        (20.0, 0.0, 0.0),
        (-20.0, 0.0, 0.0),
        (45.0, 0.0, 0.0),
        (0.0, 30.0, 0.0),
        (0.0, -30.0, 0.0),
        (0.0, 0.0, 20.0),
        (30.0, 25.0, 15.0),
        (-35.0, -20.0, -10.0),
    ],
)
def test_pose_round_trips_each_axis(pitch_up_deg: float, yaw_deg: float, roll_deg: float) -> None:
    """Every angle must come back as the angle that was put in.

    Reading an angle out of the wrong axis pair still round-trips the identity
    matrix, so a zero-only test cannot tell a correct mapping from a broken
    one.  These cases can.
    """

    pose = head_pose_from_transformation_matrix(_face_matrix(pitch_up_deg, yaw_deg, roll_deg))

    assert pose is not None
    assert pose.pitch_deg == pytest.approx(pitch_up_deg, abs=1e-6)
    assert pose.yaw_deg == pytest.approx(yaw_deg, abs=1e-6)
    assert pose.roll_deg == pytest.approx(roll_deg, abs=1e-6)


def test_pitch_separates_chin_up_from_chin_down() -> None:
    """A nod must move pitch a long way, and in opposite directions.

    This is the property that failed on the live device: chin-up and chin-down
    both landed on the same side of neutral, 1.9 deg apart, so every frame sat
    outside ``HeadPoseLimits.max_abs_pitch_deg`` and the confidence gate
    rejected 100% of frames while the eyes were perfectly visible.  A pitch
    that does not separate a nod is not a pitch, whatever it round-trips to.
    """

    down = head_pose_from_transformation_matrix(_face_matrix(-25.0, 0.0, 0.0))
    level = head_pose_from_transformation_matrix(_face_matrix(0.0, 0.0, 0.0))
    up = head_pose_from_transformation_matrix(_face_matrix(25.0, 0.0, 0.0))

    assert down is not None and level is not None and up is not None
    assert down.pitch_deg < level.pitch_deg < up.pitch_deg
    assert up.pitch_deg - down.pitch_deg == pytest.approx(50.0, abs=1e-6)


def test_roll_is_zero_rather_than_arbitrary_when_looking_straight_up() -> None:
    """Roll is undefined when the forward axis is vertical; it must not invent one."""

    pose = head_pose_from_transformation_matrix(_face_matrix(90.0, 0.0, 0.0))

    assert pose is not None
    assert pose.pitch_deg == pytest.approx(90.0, abs=1e-6)
    assert pose.roll_deg == 0.0
