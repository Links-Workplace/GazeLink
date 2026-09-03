"""Model-isolated face-landmark adapter for the M1 vision pipeline.

The adapter owns MediaPipe objects and maps their transient output directly to
the framework-independent :class:`~gazelink.domain.VisionObservation` contract.
Raw frames and landmark arrays never leave this module or persist on the
adapter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from gazelink.domain import (
    FramePacket,
    HeadPose,
    NormalizedBox,
    NormalizedPoint,
    PixelFormat,
    ReasonCode,
    TrackingState,
    VisionObservation,
)
from gazelink.features import EyeLandmarks, extract_features

# MediaPipe Face Mesh topology, using the subject's anatomical left/right
# convention.  See the canonical Face Mesh/iris topology: right iris is
# 468--472 (centre 468); left iris is 473--477 (centre 473).  The eye-corner
# pairs make the required eye availability explicit without leaking landmarks.
# The first index in each five-point iris set is MediaPipe's iris centre.
MEDIAPIPE_RIGHT_IRIS_INDICES = (468, 469, 470, 471, 472)
MEDIAPIPE_LEFT_IRIS_INDICES = (473, 474, 475, 476, 477)
MEDIAPIPE_RIGHT_IRIS_CENTER_INDEX = MEDIAPIPE_RIGHT_IRIS_INDICES[0]
MEDIAPIPE_LEFT_IRIS_CENTER_INDEX = MEDIAPIPE_LEFT_IRIS_INDICES[0]
# Eye corners are ordered (outer, inner): outer sits toward the temple, inner
# toward the nose.  Eyelids are the canonical Face Mesh mid-lid points, so the
# gap between them is the eye aperture that ``features`` turns into openness.
MEDIAPIPE_RIGHT_EYE_CORNER_INDICES = (33, 133)
MEDIAPIPE_LEFT_EYE_CORNER_INDICES = (263, 362)
MEDIAPIPE_RIGHT_EYELID_INDICES = (159, 145)
MEDIAPIPE_LEFT_EYELID_INDICES = (386, 374)
# The Face Landmarker task exposes landmark positions but no calibrated
# detector confidence.  Every structural confidence this adapter reports is
# pinned to this value, which sits at the display threshold and below the
# calibration and control thresholds, so no adapter output can authorize
# control before M2 defines a measured confidence.
STRUCTURAL_CONFIDENCE = 0.5
DEFAULT_FACE_LANDMARKER_MODEL = Path(".gazelink/models/face_landmarker.task")


class VisionEngine(Protocol):
    """Model-independent port consumed by later pipeline stages."""

    def observe(self, frame: FramePacket) -> VisionObservation:
        """Map one frame to a fresh observation without retaining frame data."""

    def close(self) -> None:
        """Release model resources; repeated calls must be safe."""


class _Landmarker(Protocol):
    def detect_for_video(self, image: object, timestamp_ms: int) -> object: ...

    def close(self) -> None: ...


ImageFactory = Callable[[FramePacket], object]


class FaceLandmarkerAdapter:
    """Map MediaPipe Face Landmarker output to the stable domain contract.

    ``landmarker`` and ``image_factory`` are injectable so unit tests do not
    import MediaPipe, touch a model asset, or need a physical camera.  The
    default model uses VIDEO mode, which matches sequential camera frames.

    The Face Landmarker API exposes landmark locations but no calibrated face
    detector confidence.  This adapter therefore reports the fixed structural
    ``STRUCTURAL_CONFIDENCE`` only when it found exactly one face, both eyes
    with usable geometry, and a valid pose matrix.  That value is diagnostic
    only: it sits below the calibration and control thresholds, so no output of
    this adapter can authorize control until M2 defines a measured confidence.
    """

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_FACE_LANDMARKER_MODEL,
        landmarker: _Landmarker | None = None,
        image_factory: ImageFactory | None = None,
    ) -> None:
        self._model_path = model_path
        self._landmarker = landmarker
        self._image_factory = image_factory or _frame_to_mediapipe_image
        self._closed = False

    def observe(self, frame: FramePacket) -> VisionObservation:
        """Return a new observation for ``frame``; never reuse a prior result."""

        if self._closed:
            raise RuntimeError("VisionEngine is closed")
        if frame.image is None or frame.pixel_format not in {PixelFormat.RGB24, PixelFormat.BGR24}:
            return _lost_observation(frame, ReasonCode.ERROR)

        try:
            image = self._image_factory(frame)
            result = self._get_landmarker().detect_for_video(
                image, timestamp_ms=round(frame.captured_at_monotonic_ms)
            )
        except (RuntimeError, ValueError):
            return _lost_observation(frame, ReasonCode.ERROR)
        return map_face_landmarker_result(result, frame)

    def close(self) -> None:
        """Release the wrapped model exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._landmarker is not None:
            self._landmarker.close()

    def _get_landmarker(self) -> _Landmarker:
        if self._landmarker is None:
            self._landmarker = _create_mediapipe_landmarker(self._model_path)
        return self._landmarker


def map_face_landmarker_result(result: object, frame: FramePacket) -> VisionObservation:
    """Map one MediaPipe result into a fresh, provider-free observation.

    The function only reads the result during this call.  It neither returns
    nor stores a MediaPipe result, landmarks, transformation matrix, or frame.
    """

    faces = _result_sequence(result, "face_landmarks")
    if not faces:
        return _lost_observation(frame, ReasonCode.FACE_NOT_FOUND)
    if len(faces) != 1:
        return VisionObservation(
            frame_id=frame.frame_id,
            observed_at_monotonic_ms=frame.captured_at_monotonic_ms,
            tracking_state=TrackingState.MULTIPLE_FACES,
            face_box=None,
            left_eye=None,
            right_eye=None,
            head_pose=None,
            overall_confidence=0.0,
            reason_codes=(ReasonCode.MULTIPLE_FACES,),
        )

    landmarks = _as_sequence(faces[0])
    if landmarks is None:
        return _lost_observation(frame, ReasonCode.ERROR)
    face_box = _face_box(landmarks)
    head_pose = _head_pose_from_result(result)
    extraction = extract_features(
        _eye_landmarks(
            landmarks,
            iris_indices=MEDIAPIPE_LEFT_IRIS_INDICES,
            eye_corner_indices=MEDIAPIPE_LEFT_EYE_CORNER_INDICES,
            eyelid_indices=MEDIAPIPE_LEFT_EYELID_INDICES,
        ),
        _eye_landmarks(
            landmarks,
            iris_indices=MEDIAPIPE_RIGHT_IRIS_INDICES,
            eye_corner_indices=MEDIAPIPE_RIGHT_EYE_CORNER_INDICES,
            eyelid_indices=MEDIAPIPE_RIGHT_EYELID_INDICES,
        ),
        head_pose=head_pose,
    )
    left_eye = extraction.left_eye.eye
    right_eye = extraction.right_eye.eye

    reason_codes: list[ReasonCode] = list(extraction.reason_codes)
    if face_box is None or head_pose is None:
        reason_codes.append(ReasonCode.LOW_CONFIDENCE)
    if reason_codes:
        return VisionObservation(
            frame_id=frame.frame_id,
            observed_at_monotonic_ms=frame.captured_at_monotonic_ms,
            tracking_state=TrackingState.LOW_CONFIDENCE,
            face_box=face_box,
            left_eye=left_eye,
            right_eye=right_eye,
            head_pose=head_pose,
            overall_confidence=0.0,
            reason_codes=tuple(reason_codes),
        )

    return VisionObservation(
        frame_id=frame.frame_id,
        observed_at_monotonic_ms=frame.captured_at_monotonic_ms,
        tracking_state=TrackingState.TRACKED,
        face_box=face_box,
        left_eye=left_eye,
        right_eye=right_eye,
        head_pose=head_pose,
        overall_confidence=STRUCTURAL_CONFIDENCE,
    )


_AXIS_EPSILON = 1e-6


def head_pose_from_transformation_matrix(matrix: object) -> HeadPose | None:
    """Convert a 4x4 facial transform to pitch/yaw/roll degrees.

    MediaPipe supplies a row-major facial transformation matrix whose
    upper-left 3x3 is a proper rotation -- verified against the live device:
    ``det == 1.000`` and orthonormality error ``0.0000`` over sampled frames.
    Its COLUMNS are the face's own axes expressed in MediaPipe's camera frame
    (X right, Y up, Z toward the camera), so each angle is read straight off an
    axis vector instead of being recovered by a decomposition:

    * ``yaw``   -- azimuth of the face's forward axis about the vertical
    * ``pitch`` -- elevation of that forward axis above the horizon
    * ``roll``  -- rotation of the face's up axis about its own forward axis

    A face aimed straight at the camera is ``(0, 0, 0)`` here, and every angle
    answers one physical question, so a sign can be checked against a held pose
    instead of argued about.

    This replaces a ``Rz(roll) * Ry(yaw) * Rx(pitch)`` decomposition that was
    mathematically correct -- it round-trips synthetic rotations exactly -- but
    assumed an aerospace axis convention this matrix does not use.  It read
    pitch as ``atan2(up_z, forward_z)``, mixing two different axes into a value
    that does not vary monotonically with head pitch.  Measured live over three
    held poses, chin-up and chin-down landed on the SAME side of neutral and
    differed by 1.9 deg where a real nod spans 40-70 deg, while a single held
    pose scattered across 31 deg.  Every frame therefore sat outside
    ``HeadPoseLimits.max_abs_pitch_deg`` and the confidence gate rejected 100%
    of frames -- eyes, iris and openness all present and usable.

    Sign convention: ``pitch`` > 0 chin up, ``yaw`` > 0 face turned toward the
    camera's right, ``roll`` > 0 head tipped toward the face's right shoulder.

    A camera mounted below the screen and angled up leaves a large fixed pitch
    offset even for a level head.  That offset is real geometry, not an error
    in this function, and whether the gate should tolerate it is a separate
    decision that belongs with the threshold, not here.
    """

    rows = _matrix_rows(matrix)
    if rows is None:
        return None
    # Columns of the rotation, i.e. the face's axes in camera coordinates.
    up = (rows[0][1], rows[1][1], rows[2][1])
    forward = (rows[0][2], rows[1][2], rows[2][2])

    horizontal = math.hypot(forward[0], forward[2])
    yaw = math.atan2(forward[0], forward[2])
    pitch = math.atan2(forward[1], horizontal)

    # Roll is the signed angle between the face's up axis and world up, taken
    # about the forward axis.  World up is first made perpendicular to forward
    # so the two vectors share a plane; when the face points almost straight up
    # or down that projection vanishes and roll is genuinely undefined, so it
    # reports 0.0 rather than an arbitrary large angle.
    vertical_component = forward[1]
    reference = (
        -vertical_component * forward[0],
        1.0 - vertical_component * forward[1],
        -vertical_component * forward[2],
    )
    reference_length = math.sqrt(sum(component * component for component in reference))
    if reference_length <= _AXIS_EPSILON:
        roll = 0.0
    else:
        unit = (
            reference[0] / reference_length,
            reference[1] / reference_length,
            reference[2] / reference_length,
        )
        side = (
            (forward[1] * unit[2]) - (forward[2] * unit[1]),
            (forward[2] * unit[0]) - (forward[0] * unit[2]),
            (forward[0] * unit[1]) - (forward[1] * unit[0]),
        )
        roll = math.atan2(
            sum(a * b for a, b in zip(up, side, strict=True)),
            sum(a * b for a, b in zip(up, unit, strict=True)),
        )
    return HeadPose(
        yaw_deg=math.degrees(yaw),
        pitch_deg=math.degrees(pitch),
        roll_deg=math.degrees(roll),
    )


def _lost_observation(frame: FramePacket, reason_code: ReasonCode) -> VisionObservation:
    return VisionObservation(
        frame_id=frame.frame_id,
        observed_at_monotonic_ms=frame.captured_at_monotonic_ms,
        tracking_state=TrackingState.LOST,
        face_box=None,
        left_eye=None,
        right_eye=None,
        head_pose=None,
        overall_confidence=0.0,
        reason_codes=(reason_code,),
    )


def _result_sequence(result: object, attribute: str) -> Sequence[object]:
    value = getattr(result, attribute, ())
    sequence = _as_sequence(value)
    return () if sequence is None else sequence


def _as_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    return value


def _point_at(landmarks: Sequence[object], index: int) -> NormalizedPoint | None:
    if index >= len(landmarks):
        return None
    landmark = landmarks[index]
    x = getattr(landmark, "x", None)
    y = getattr(landmark, "y", None)
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
    ):
        return None
    if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    return NormalizedPoint(x=float(x), y=float(y))


def _face_box(landmarks: Sequence[object]) -> NormalizedBox | None:
    points = [
        point
        for index in range(len(landmarks))
        if (point := _point_at(landmarks, index)) is not None
    ]
    if not points:
        return None
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    if min_x == max_x or min_y == max_y:
        return None
    return NormalizedBox(min_x, min_y, max_x - min_x, max_y - min_y)


def _eye_landmarks(
    landmarks: Sequence[object],
    *,
    iris_indices: tuple[int, ...],
    eye_corner_indices: tuple[int, int],
    eyelid_indices: tuple[int, int],
) -> EyeLandmarks | None:
    """Collect one eye's landmarks, or return ``None`` when any of them is unusable.

    Every point is required.  Averaging whichever subset happened to be valid
    would make the iris centre depend on which landmarks the model dropped, so
    a partial eye is reported as unavailable instead.
    """

    outer_corner = _point_at(landmarks, eye_corner_indices[0])
    inner_corner = _point_at(landmarks, eye_corner_indices[1])
    upper_lid = _point_at(landmarks, eyelid_indices[0])
    lower_lid = _point_at(landmarks, eyelid_indices[1])
    if outer_corner is None or inner_corner is None or upper_lid is None or lower_lid is None:
        return None
    if outer_corner == inner_corner:
        return None
    iris_points: list[NormalizedPoint] = []
    for index in iris_indices:
        point = _point_at(landmarks, index)
        if point is None:
            return None
        iris_points.append(point)
    return EyeLandmarks(
        outer_corner=outer_corner,
        inner_corner=inner_corner,
        upper_lid=upper_lid,
        lower_lid=lower_lid,
        iris_frame_points=tuple(iris_points),
        detector_confidence=STRUCTURAL_CONFIDENCE,
    )


def _head_pose_from_result(result: object) -> HeadPose | None:
    matrices = _result_sequence(result, "facial_transformation_matrixes")
    return None if len(matrices) != 1 else head_pose_from_transformation_matrix(matrices[0])


def _matrix_rows(matrix: object) -> tuple[tuple[float, ...], ...] | None:
    to_list = getattr(matrix, "tolist", None)
    candidate = to_list() if callable(to_list) else matrix
    rows = _as_sequence(candidate)
    if rows is None or len(rows) != 4:
        return None
    converted_rows: list[tuple[float, ...]] = []
    for row in rows:
        values = _as_sequence(row)
        if values is None or len(values) != 4:
            return None
        converted: list[float] = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return None
            converted.append(float(value))
        converted_rows.append(tuple(converted))
    return tuple(converted_rows)


def _frame_to_mediapipe_image(frame: FramePacket) -> object:
    """Build a short-lived SRGB MediaPipe image from immutable RGB/BGR bytes."""

    if frame.image is None:
        raise ValueError("FramePacket.image is required for MediaPipe inference")
    expected_size = frame.width * frame.height * 3
    if len(frame.image) != expected_size:
        raise ValueError("FramePacket.image does not match RGB/BGR frame dimensions")

    import mediapipe as mp  # type: ignore[import-untyped]
    import numpy as np

    pixels = np.frombuffer(frame.image, dtype=np.uint8).reshape((frame.height, frame.width, 3))
    if frame.pixel_format is PixelFormat.BGR24:
        pixels = pixels[:, :, ::-1]
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels.copy())


def _create_mediapipe_landmarker(model_path: Path) -> _Landmarker:
    """Lazy-load MediaPipe so deterministic unit tests do not import it."""

    if not model_path.is_file():
        raise FileNotFoundError("Face Landmarker model is not available")
    from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]
    from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )

    # Load the model as bytes rather than by path. MediaPipe resolves
    # ``model_asset_path`` against a process-wide resource root, and anything
    # that imports the legacy ``mp.solutions`` API repoints that root at
    # MediaPipe's own package -- after which even an ABSOLUTE path here is
    # rewritten to ``site-packages/C:\\Users\\...`` and construction fails.
    # A buffer bypasses that resolution entirely, so this stays correct
    # regardless of the working directory or what else in the process has
    # touched MediaPipe's globals. Same model, same options, same output.
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=RunningMode.VIDEO,
        num_faces=2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return cast(_Landmarker, FaceLandmarker.create_from_options(options))
