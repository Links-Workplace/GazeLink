"""9-point calibration session state machine and target contract for M2.

This module owns the *lifecycle* of a calibration session: which target is
being collected, which samples have landed on it, and when the whole session
is legitimately complete. It stores every submitted :class:`CalibrationSample`
(both accepted and rejected) so a real dataset exists once a session
completes, but it does not decide *whether* a sample is good enough to count
-- that quality policy (confidence gating, outlier/pose/age rejection)
belongs to a caller, which builds the ``CalibrationSample`` and reports its
own ``accepted`` verdict. What this module *does* own, as of M2-02 Stage B,
is the closed set of reasons a verdict may cite: see
:class:`CalibrationReasonCode`. Fixing that set here -- rather than letting
each caller invent its own ad-hoc label -- is what lets a session log, a
serialized dataset, or a later analysis pass trust that ``sample.reason`` is
always one of a known, finite set of outcomes, not an arbitrary string. This
module also does not extract features or train a gaze mapping (M2-03); it
only produces the stored samples and metadata a later training/validation
step needs.

Two safety properties this module is responsible for enforcing:

- A partial session can never be exported as a usable result.
  :meth:`CalibrationSession.result` raises unless the session state is
  ``COMPLETE``, so nothing downstream can mistake an in-progress or abandoned
  session for valid calibration data.
- Calibration never touches control state. This class has no import of, or
  reference to, ``ControlState``/``SystemState``/any OS-input concept --
  running or cancelling a calibration session cannot, by construction, enable
  or disable cursor control.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from gazelink.confidence import Clock, monotonic_ms
from gazelink.domain import (
    ContractValidationError,
    JSONValue,
    NormalizedPoint,
    ScreenGeometry,
)

NUM_CALIBRATION_TARGETS = 9

# Placeholder defaults, not yet tuned against a real user -- matches the same
# "documented placeholder" treatment given to HeadPoseLimits in confidence.py.
# Corners sit slightly inset from the true screen edge (rather than at exactly
# 0.0/1.0) because a target flush against the edge is uncomfortable to look at
# and more prone to eyelid/eyelash occlusion; common practice in eye-tracking
# calibration, not yet validated for this product.
DEFAULT_TARGET_EDGE_INSET = 0.05
ULTRAWIDE_TARGET_HORIZONTAL_INSET = 0.15
ULTRAWIDE_ASPECT_RATIO = 2.40
DEFAULT_MIN_SAMPLES_PER_TARGET = 5

# Bumped whenever features.py's eye-geometry output shape changes, so a
# CalibrationSessionResult can be recognized as incompatible with a newer
# feature extractor rather than silently misinterpreted.
FEATURE_SCHEMA_VERSION = 3


class CalibrationSessionState(StrEnum):
    """Lifecycle of one calibration session; see the module docstring."""

    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class CalibrationReasonCode(StrEnum):
    """Closed set of outcomes a :class:`CalibrationSample`'s ``reason`` may carry.

    Every sample -- accepted or rejected -- is stamped with exactly one of
    these. Closing the set (rather than letting each caller invent its own
    label) is what lets a session log, a serialized dataset, or a later
    analysis pass rely on ``reason`` meaning one specific, known thing
    instead of an arbitrary string nobody downstream can safely branch on.

    The first four values (``ACCEPTED`` through ``FEATURES_UNAVAILABLE``) are
    the pre-Stage-B values already written into session logs and serialized
    ``CalibrationSample`` dicts before this enum existed; they are kept
    byte-identical here on purpose so that existing logs and datasets stay
    readable and comparable across the change. The remaining four are new in
    M2-02 Stage B, giving a name to rejection outcomes the caller's quality
    policy previously had to describe with an unchecked, ad-hoc string.
    """

    ACCEPTED = "accepted"
    TRACKING_NOT_READY = "tracking_not_ready"
    LOW_CONFIDENCE = "low_confidence"
    FEATURES_UNAVAILABLE = "features_unavailable"
    STALE_SAMPLE = "stale_sample"
    EYE_NOT_VISIBLE = "eye_not_visible"
    HEAD_POSE_OUT_OF_RANGE = "head_pose_out_of_range"
    TARGET_OUTLIER = "target_outlier"


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _target_index(value: object, field_name: str = "index") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (0 <= value < NUM_CALIBRATION_TARGETS)
    ):
        raise ContractValidationError(f"{field_name} must be within [0, {NUM_CALIBRATION_TARGETS})")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return value


def _non_negative_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContractValidationError(f"{field_name} must be finite and >= 0")
    return result


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite")
    return result


def _confidence_ratio(value: object, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ContractValidationError(f"{field_name} must be within [0.0, 1.0]")
    return result


def _optional_ratio(value: object | None, field_name: str) -> float | None:
    return None if value is None else _confidence_ratio(value, field_name)


def _optional_angle(value: object | None, field_name: str) -> float | None:
    return None if value is None else _finite_number(value, field_name)


def _optional_point(value: object | None, field_name: str) -> NormalizedPoint | None:
    if value is None:
        return None
    if not isinstance(value, NormalizedPoint):
        raise ContractValidationError(f"{field_name} must be a NormalizedPoint or None")
    return value


def _reason_code(value: object) -> str:
    """Validate ``value`` against the closed :class:`CalibrationReasonCode` set.

    Accepts either a plain string equal to one of the enum's values or an
    actual ``CalibrationReasonCode`` member, and always returns a plain
    ``str`` -- normalizing an enum member down to its ``.value`` here, once,
    is what keeps ``CalibrationSample.reason`` a plain string on the object
    (see the field's own docstring for why that matters for serialization).
    """

    if not isinstance(value, (str, CalibrationReasonCode)):
        raise ContractValidationError(
            f"reason must be a CalibrationReasonCode or its string value, not {value!r}"
        )
    try:
        return CalibrationReasonCode(value).value
    except ValueError as exc:
        raise ContractValidationError(
            f"reason must be one of the closed CalibrationReasonCode values, not {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One of the 9 calibration points.

    ``screen_position`` reuses :class:`~gazelink.domain.NormalizedPoint`'s
    ``[0.0, 1.0]``/top-left-origin contract, but here it names a position on
    the *screen*, not a position within a captured frame -- the same
    deliberate reuse-with-a-different-meaning already used for
    ``EyeFeatures.iris_center`` vs ``iris_in_eye``. Never pass a frame-space
    point here.
    """

    index: int
    screen_position: NormalizedPoint

    def __post_init__(self) -> None:
        _target_index(self.index)
        if not isinstance(self.screen_position, NormalizedPoint):
            raise ContractValidationError("screen_position must be a NormalizedPoint")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"index": self.index, "screen_position": self.screen_position.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> CalibrationTarget:
        data = _mapping(value, "calibration target")
        screen_position = data.get("screen_position")
        if screen_position is None:
            raise ContractValidationError("screen_position is required")
        return cls(
            index=data.get("index"),  # type: ignore[arg-type]
            screen_position=NormalizedPoint.from_dict(screen_position),
        )


def default_nine_point_targets(
    *, edge_inset: float = DEFAULT_TARGET_EDGE_INSET
) -> tuple[CalibrationTarget, ...]:
    """Return the canonical 3x3 grid: corners, mid-edges, and center.

    Targets are numbered row-major from top-left (0) to bottom-right (8);
    index 4 is always the center point.
    """

    if not isinstance(edge_inset, (int, float)) or isinstance(edge_inset, bool):
        raise ContractValidationError("edge_inset must be a number")
    if not 0.0 <= edge_inset < 0.5:
        raise ContractValidationError("edge_inset must be within [0.0, 0.5)")
    low, mid, high = edge_inset, 0.5, 1.0 - edge_inset
    return tuple(
        CalibrationTarget(index=index, screen_position=NormalizedPoint(x, y))
        for index, (y, x) in enumerate((y, x) for y in (low, mid, high) for x in (low, mid, high))
    )


def targets_for_screen_geometry(
    geometry: ScreenGeometry,
) -> tuple[CalibrationTarget, ...]:
    """Return a comfortable 3x3 grid adapted to very wide displays.

    On an ultrawide display the canonical 5% horizontal inset puts the two
    outer columns near the physical extremes.  That encourages large head
    turns and makes the nine targets cover a region wider than the area a
    user will normally operate with their eyes.  Keep the vertical coverage
    unchanged, but use a conservative 15% horizontal inset for aspect ratios
    of 2.40 or wider.  Both values remain documented tuning placeholders.
    """

    if not isinstance(geometry, ScreenGeometry):
        raise ContractValidationError("geometry must be a ScreenGeometry")
    horizontal_inset = (
        ULTRAWIDE_TARGET_HORIZONTAL_INSET
        if geometry.width_px / geometry.height_px >= ULTRAWIDE_ASPECT_RATIO
        else DEFAULT_TARGET_EDGE_INSET
    )
    xs = (horizontal_inset, 0.5, 1.0 - horizontal_inset)
    ys = (DEFAULT_TARGET_EDGE_INSET, 0.5, 1.0 - DEFAULT_TARGET_EDGE_INSET)
    return tuple(
        CalibrationTarget(index=index, screen_position=NormalizedPoint(x, y))
        for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
    )


def _validate_target_order(target_order: Sequence[int]) -> tuple[int, ...]:
    order = tuple(target_order)
    if sorted(order) != list(range(NUM_CALIBRATION_TARGETS)):
        raise ContractValidationError(
            f"target_order must be a permutation of 0..{NUM_CALIBRATION_TARGETS - 1}"
        )
    return order


def _validate_targets(targets: Sequence[CalibrationTarget]) -> tuple[CalibrationTarget, ...]:
    targets_tuple = tuple(targets)
    if len(targets_tuple) != NUM_CALIBRATION_TARGETS:
        raise ContractValidationError(f"exactly {NUM_CALIBRATION_TARGETS} targets are required")
    if any(not isinstance(target, CalibrationTarget) for target in targets_tuple):
        raise ContractValidationError("every target must be a CalibrationTarget")
    if {target.index for target in targets_tuple} != set(range(NUM_CALIBRATION_TARGETS)):
        raise ContractValidationError(
            "target indices must be a complete 0..8 set with no duplicates"
        )
    return targets_tuple


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One real, ingested observation tied to the target the user was looking at.

    This is the actual dataset entry M2-03's gaze mapping will train on --
    unlike ``CalibrationSession``'s per-target *counts*, which only tell you
    how many samples landed, not what they looked like. Both accepted and
    rejected samples are represented (``accepted`` says which); a rejected
    sample is exactly the input a later outlier/quality metric needs to
    explain itself.

    Every field is a small scalar or a screen/eye-relative coordinate --
    never a raw frame, image, or pixel buffer. ``reason`` is declared as
    plain ``str`` here, not :class:`CalibrationReasonCode`, on purpose:
    keeping the field a string means ``to_dict``/``from_dict`` and every
    already-written session log stay byte-identical across this change --
    no migration, no schema bump. What M2-02 Stage B actually changed is
    that the *value* is no longer an unchecked label; it is now validated
    against ``CalibrationReasonCode``'s closed set (``"accepted"``,
    ``"tracking_not_ready"``, ``"low_confidence"``, and five more). A caller
    may pass either the plain string or the enum member -- either way the
    stored value is normalized to the member's plain string, so downstream
    code (including this module's own log rendering) never has to care
    which form it was given.
    """

    source_frame_id: int
    observed_at_monotonic_ms: float
    target_index: int
    left_iris_in_eye: NormalizedPoint | None
    right_iris_in_eye: NormalizedPoint | None
    left_openness: float | None
    right_openness: float | None
    head_yaw_deg: float | None
    head_pitch_deg: float | None
    head_roll_deg: float | None
    confidence: float
    accepted: bool
    reason: str
    left_iris_in_lids_y: float | None = None
    right_iris_in_lids_y: float | None = None

    def __post_init__(self) -> None:
        _non_negative_number(self.source_frame_id, "source_frame_id")
        object.__setattr__(
            self,
            "observed_at_monotonic_ms",
            _non_negative_number(self.observed_at_monotonic_ms, "observed_at_monotonic_ms"),
        )
        _target_index(self.target_index, "target_index")
        object.__setattr__(
            self, "left_iris_in_eye", _optional_point(self.left_iris_in_eye, "left_iris_in_eye")
        )
        object.__setattr__(
            self, "right_iris_in_eye", _optional_point(self.right_iris_in_eye, "right_iris_in_eye")
        )
        object.__setattr__(
            self, "left_openness", _optional_ratio(self.left_openness, "left_openness")
        )
        object.__setattr__(
            self, "right_openness", _optional_ratio(self.right_openness, "right_openness")
        )
        object.__setattr__(
            self,
            "left_iris_in_lids_y",
            _optional_ratio(self.left_iris_in_lids_y, "left_iris_in_lids_y"),
        )
        object.__setattr__(
            self,
            "right_iris_in_lids_y",
            _optional_ratio(self.right_iris_in_lids_y, "right_iris_in_lids_y"),
        )
        object.__setattr__(self, "head_yaw_deg", _optional_angle(self.head_yaw_deg, "head_yaw_deg"))
        object.__setattr__(
            self, "head_pitch_deg", _optional_angle(self.head_pitch_deg, "head_pitch_deg")
        )
        object.__setattr__(
            self, "head_roll_deg", _optional_angle(self.head_roll_deg, "head_roll_deg")
        )
        object.__setattr__(self, "confidence", _confidence_ratio(self.confidence, "confidence"))
        if not isinstance(self.accepted, bool):
            raise ContractValidationError("accepted must be a bool")
        object.__setattr__(self, "reason", _reason_code(self.reason))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "source_frame_id": self.source_frame_id,
            "observed_at_monotonic_ms": self.observed_at_monotonic_ms,
            "target_index": self.target_index,
            "left_iris_in_eye": None
            if self.left_iris_in_eye is None
            else self.left_iris_in_eye.to_dict(),
            "right_iris_in_eye": None
            if self.right_iris_in_eye is None
            else self.right_iris_in_eye.to_dict(),
            "left_openness": self.left_openness,
            "right_openness": self.right_openness,
            "left_iris_in_lids_y": self.left_iris_in_lids_y,
            "right_iris_in_lids_y": self.right_iris_in_lids_y,
            "head_yaw_deg": self.head_yaw_deg,
            "head_pitch_deg": self.head_pitch_deg,
            "head_roll_deg": self.head_roll_deg,
            "confidence": self.confidence,
            "accepted": self.accepted,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationSample:
        data = _mapping(value, "calibration sample")
        left_iris = data.get("left_iris_in_eye")
        right_iris = data.get("right_iris_in_eye")
        return cls(
            source_frame_id=data.get("source_frame_id"),  # type: ignore[arg-type]
            observed_at_monotonic_ms=data.get("observed_at_monotonic_ms"),  # type: ignore[arg-type]
            target_index=data.get("target_index"),  # type: ignore[arg-type]
            left_iris_in_eye=None if left_iris is None else NormalizedPoint.from_dict(left_iris),
            right_iris_in_eye=None if right_iris is None else NormalizedPoint.from_dict(right_iris),
            left_openness=data.get("left_openness"),
            right_openness=data.get("right_openness"),
            left_iris_in_lids_y=data.get("left_iris_in_lids_y"),
            right_iris_in_lids_y=data.get("right_iris_in_lids_y"),
            head_yaw_deg=data.get("head_yaw_deg"),
            head_pitch_deg=data.get("head_pitch_deg"),
            head_roll_deg=data.get("head_roll_deg"),
            confidence=data.get("confidence"),  # type: ignore[arg-type]
            accepted=data.get("accepted"),  # type: ignore[arg-type]
            reason=data.get("reason"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CalibrationSessionResult:
    """Frozen, serializable snapshot of a finished calibration session.

    Only ever produced by :meth:`CalibrationSession.result`, which refuses to
    build one for anything but a genuinely ``COMPLETE`` session -- so this
    type existing at all is itself evidence the session was not partial.
    Consumed independently by a later, separate validation step (M2-T08),
    which never needs the live, mutable session object.

    ``samples`` is the real calibration dataset: every ingested
    :class:`CalibrationSample`, accepted and rejected alike, in ingest order.
    ``sample_counts`` remains as a quick per-target accepted-count summary
    (unchanged from before this dataset existed), so existing callers that
    only need counts don't need to scan ``samples`` themselves.
    """

    target_order: tuple[int, ...]
    targets: tuple[CalibrationTarget, ...]
    sample_counts: tuple[int, ...]
    samples: tuple[CalibrationSample, ...]
    camera_id: str
    screen_geometry: ScreenGeometry
    feature_schema_version: int
    min_samples_per_target: int
    started_at_monotonic_ms: float
    completed_at_monotonic_ms: float

    def __post_init__(self) -> None:
        _validate_target_order(self.target_order)
        object.__setattr__(self, "targets", _validate_targets(self.targets))
        if len(self.sample_counts) != NUM_CALIBRATION_TARGETS:
            raise ContractValidationError(
                f"sample_counts must have {NUM_CALIBRATION_TARGETS} entries"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.sample_counts
        ):
            raise ContractValidationError("sample_counts must be non-negative integers")
        if any(not isinstance(sample, CalibrationSample) for sample in self.samples):
            raise ContractValidationError("samples must contain only CalibrationSample values")
        _nonempty_string(self.camera_id, "camera_id")
        if not isinstance(self.screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be a ScreenGeometry")
        _positive_int(self.feature_schema_version, "feature_schema_version")
        _positive_int(self.min_samples_per_target, "min_samples_per_target")
        if self.completed_at_monotonic_ms < self.started_at_monotonic_ms:
            raise ContractValidationError(
                "completed_at_monotonic_ms must be >= started_at_monotonic_ms"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "target_order": list(self.target_order),
            "targets": [target.to_dict() for target in self.targets],
            "sample_counts": list(self.sample_counts),
            "samples": [sample.to_dict() for sample in self.samples],
            "camera_id": self.camera_id,
            "screen_geometry": self.screen_geometry.to_dict(),
            "feature_schema_version": self.feature_schema_version,
            "min_samples_per_target": self.min_samples_per_target,
            "started_at_monotonic_ms": self.started_at_monotonic_ms,
            "completed_at_monotonic_ms": self.completed_at_monotonic_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationSessionResult:
        data = _mapping(value, "calibration session result")
        screen_geometry = data.get("screen_geometry")
        if screen_geometry is None:
            raise ContractValidationError("screen_geometry is required")
        serialized_targets = data.get("targets")
        targets = (
            default_nine_point_targets()
            if serialized_targets is None
            else tuple(CalibrationTarget.from_dict(target) for target in serialized_targets)
        )
        return cls(
            target_order=tuple(data.get("target_order", ())),
            targets=targets,
            sample_counts=tuple(data.get("sample_counts", ())),
            samples=tuple(
                CalibrationSample.from_dict(sample) for sample in data.get("samples", ())
            ),
            camera_id=data.get("camera_id"),  # type: ignore[arg-type]
            screen_geometry=ScreenGeometry.from_dict(screen_geometry),
            feature_schema_version=data.get("feature_schema_version"),  # type: ignore[arg-type]
            min_samples_per_target=data.get("min_samples_per_target"),  # type: ignore[arg-type]
            started_at_monotonic_ms=data.get("started_at_monotonic_ms"),  # type: ignore[arg-type]
            completed_at_monotonic_ms=data.get("completed_at_monotonic_ms"),  # type: ignore[arg-type]
        )

    def reason_counts(self) -> dict[str, int]:
        """Tally of ``sample.reason`` across every stored sample, accepted or rejected.

        Computed fresh from ``samples`` on every call rather than cached or
        stored as a field -- ``samples`` is the one source of truth for this
        result, and a cached tally could silently drift from it. Only
        reasons that actually occur are included; a reason with zero
        occurrences is simply absent from the dict rather than reported as
        ``0``, so a caller can distinguish "no sample had this reason" from
        "this reason was not worth counting" without special-casing zero.
        """

        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample.reason] = counts.get(sample.reason, 0) + 1
        return counts

    def head_pose_spread(self) -> dict[str, float]:
        """Descriptive head-pose spread over accepted samples, in degrees.

        This is diagnostic only -- it must never gate acceptance. The
        accept/reject verdict for every sample already happened before this
        result existed (that is the caller's quality policy, not this
        module's job), and nothing here may feed back into that decision.
        Only *accepted* samples with all three of ``head_yaw_deg``,
        ``head_pitch_deg`` and ``head_roll_deg`` present are included; a
        rejected sample never contributes regardless of how extreme its
        pose was. That exclusion is the property that makes this metric
        meaningful: a user who briefly turned away and was correctly
        rejected must not appear to have widened the calibration's pose
        spread.

        Returns an empty dict -- not fabricated zeros -- when no sample
        qualifies (no accepted samples, or none with pose data). Zeros would
        misreport "the head was perfectly still" for a session that in fact
        has no pose data to judge from at all.
        """

        yaws: list[float] = []
        pitches: list[float] = []
        rolls: list[float] = []
        for sample in self.samples:
            if not sample.accepted:
                continue
            yaw, pitch, roll = sample.head_yaw_deg, sample.head_pitch_deg, sample.head_roll_deg
            if yaw is None or pitch is None or roll is None:
                continue
            yaws.append(yaw)
            pitches.append(pitch)
            rolls.append(roll)
        if not yaws:
            return {}
        median_yaw = statistics.median(yaws)
        median_pitch = statistics.median(pitches)
        median_roll = statistics.median(rolls)
        return {
            "median_yaw_deg": median_yaw,
            "median_pitch_deg": median_pitch,
            "median_roll_deg": median_roll,
            "max_yaw_deviation_deg": max(abs(yaw - median_yaw) for yaw in yaws),
            "max_pitch_deviation_deg": max(abs(pitch - median_pitch) for pitch in pitches),
            "max_roll_deviation_deg": max(abs(roll - median_roll) for roll in rolls),
        }


class CalibrationSession:
    """Stateful 9-point calibration session; owns no camera, UI, or OS input.

    Stores every submitted :class:`CalibrationSample` (accepted and rejected)
    and gates progression on ``min_samples_per_target`` accepted samples per
    target. The caller decides per-sample validity (confidence today; real
    outlier/pose/age rejection is M2-02 Stage B) and reports it by building a
    ``CalibrationSample`` and passing it to :meth:`record_sample`.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        screen_geometry: ScreenGeometry,
        targets: Sequence[CalibrationTarget] | None = None,
        target_order: Sequence[int] | None = None,
        min_samples_per_target: int = DEFAULT_MIN_SAMPLES_PER_TARGET,
        clock_ms: Clock = monotonic_ms,
    ) -> None:
        self._camera_id = _nonempty_string(camera_id, "camera_id")
        if not isinstance(screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be a ScreenGeometry")
        self._screen_geometry = screen_geometry
        self._targets = _validate_targets(targets or default_nine_point_targets())
        self._target_order = _validate_target_order(
            target_order if target_order is not None else range(NUM_CALIBRATION_TARGETS)
        )
        self._min_samples_per_target = _positive_int(
            min_samples_per_target, "min_samples_per_target"
        )
        self._clock_ms = clock_ms
        self._sample_counts = [0] * NUM_CALIBRATION_TARGETS
        self._samples: list[CalibrationSample] = []
        self._position = 0
        self._state = CalibrationSessionState.COLLECTING
        self._started_at_ms = self._clock_ms()
        self._completed_at_ms: float | None = None

    @property
    def state(self) -> CalibrationSessionState:
        return self._state

    @property
    def current_target(self) -> CalibrationTarget:
        """The target currently being collected.

        Only meaningful while :attr:`state` is ``COLLECTING``; raises
        otherwise so a caller cannot read a stale target from a finished
        session.
        """

        self._require_collecting("read the current target")
        return self._targets[self._target_order[self._position]]

    @property
    def current_target_sample_count(self) -> int:
        self._require_collecting("read the current sample count")
        return self._sample_counts[self.current_target.index]

    @property
    def current_target_accepted_samples(self) -> tuple[CalibrationSample, ...]:
        """Accepted samples collected so far for the target currently on screen.

        Filters ``self._samples`` down to the current target's ``accepted``
        entries, in ingest order -- no separate list is kept, so this can
        never drift from what :meth:`record_sample` actually stored, and
        :meth:`retry_current_target` clearing that target's samples is
        automatically reflected here too. Guarded by
        :meth:`_require_collecting` for the same reason
        :attr:`current_target_sample_count` is: once a session leaves
        ``COLLECTING`` there is no "current target" left to report samples
        for, and a stale read here could misattribute samples after
        :meth:`advance` has already moved the position on.
        """

        self._require_collecting("read the current target's accepted samples")
        index = self.current_target.index
        return tuple(
            sample for sample in self._samples if sample.target_index == index and sample.accepted
        )

    @property
    def accepted_samples(self) -> tuple[CalibrationSample, ...]:
        """Every accepted sample ingested so far, across all targets, in ingest order.

        Unlike :attr:`current_target_accepted_samples`, this is deliberately
        *not* guarded by :meth:`_require_collecting`. A descriptive,
        session-wide metric such as
        :meth:`CalibrationSessionResult.head_pose_spread` needs to be
        readable both mid-session (to give the user live feedback while
        collecting) and after the session has completed; and unlike the
        current-target accessor, there is no "stale target" ambiguity to
        guard against here, because this property never refers to "the
        current target" at all.
        """

        return tuple(sample for sample in self._samples if sample.accepted)

    @property
    def current_target_number(self) -> int:
        """One-based position in the configured target order, while collecting."""

        self._require_collecting("read the current target number")
        return self._position + 1

    @property
    def min_samples_per_target(self) -> int:
        return self._min_samples_per_target

    def record_sample(self, sample: CalibrationSample) -> None:
        """Store one real sample and, if accepted, count it toward the current target.

        ``sample.accepted`` is the caller's own quality decision (confidence
        today; a real outlier/pose/age policy is M2-02 Stage B) -- this
        session stores the sample regardless, so a rejected sample is still
        part of the dataset a later quality metric can explain itself with,
        but only an accepted one counts toward ``min_samples_per_target``.
        Forbidden once the session has left ``COLLECTING``. ``sample`` must
        be for the target currently being collected -- a sample for any other
        target is a caller bug, not a valid outcome to silently accept.
        """

        self._require_collecting("record a sample")
        if not isinstance(sample, CalibrationSample):
            raise ContractValidationError("sample must be a CalibrationSample")
        index = self.current_target.index
        if sample.target_index != index:
            raise ContractValidationError(
                f"sample.target_index ({sample.target_index}) does not match "
                f"the current target ({index})"
            )
        self._samples.append(sample)
        if sample.accepted:
            self._sample_counts[index] += 1

    def advance(self) -> None:
        """Move to the next target, or complete the session if this was the last.

        Raises if the current target has not yet reached
        ``min_samples_per_target`` -- a caller must not advance past a target
        with insufficient data.
        """

        self._require_collecting("advance")
        collected = self.current_target_sample_count
        if collected < self._min_samples_per_target:
            raise ContractValidationError(
                f"target {self.current_target.index} has {collected} of "
                f"{self._min_samples_per_target} required samples"
            )
        if self._position + 1 >= len(self._target_order):
            self._state = CalibrationSessionState.COMPLETE
            self._completed_at_ms = self._clock_ms()
        else:
            self._position += 1

    def cancel(self) -> None:
        """Mark the session cancelled; only valid while still collecting."""

        self._require_collecting("cancel")
        self._state = CalibrationSessionState.CANCELLED

    def fail(self) -> None:
        """Mark the session failed; only valid while still collecting."""

        self._require_collecting("fail")
        self._state = CalibrationSessionState.FAILED

    def restart(self) -> None:
        """Discard all progress and begin collecting again from the first target.

        Allowed from any state, including mid-collection -- this is the
        explicit "start over" a calibration UI needs, not an error path.
        """

        self._sample_counts = [0] * NUM_CALIBRATION_TARGETS
        self._samples = []
        self._position = 0
        self._state = CalibrationSessionState.COLLECTING
        self._started_at_ms = self._clock_ms()
        self._completed_at_ms = None

    def retry_current_target(self) -> None:
        """Discard samples for the displayed target and collect it again.

        This deliberately affects only the target currently on screen.  It is
        the recovery action a guided UI needs when the user was distracted;
        prior targets remain intact and a later target cannot be selected.
        """

        self._require_collecting("retry the current target")
        index = self.current_target.index
        self._sample_counts[index] = 0
        self._samples = [sample for sample in self._samples if sample.target_index != index]

    def result(self) -> CalibrationSessionResult:
        """Return the frozen snapshot; only available once the session is COMPLETE.

        This is the enforcement point for "a partial session is never marked
        valid": calling this on a collecting, cancelled, or failed session
        raises rather than returning something a caller might mistake for
        usable calibration data.
        """

        if self._state is not CalibrationSessionState.COMPLETE:
            raise ContractValidationError(
                f"result is only available for a COMPLETE session, not {self._state.value}"
            )
        assert self._completed_at_ms is not None
        return CalibrationSessionResult(
            target_order=self._target_order,
            targets=self._targets,
            sample_counts=tuple(self._sample_counts),
            samples=tuple(self._samples),
            camera_id=self._camera_id,
            screen_geometry=self._screen_geometry,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            min_samples_per_target=self._min_samples_per_target,
            started_at_monotonic_ms=self._started_at_ms,
            completed_at_monotonic_ms=self._completed_at_ms,
        )

    def _require_collecting(self, action: str) -> None:
        if self._state is not CalibrationSessionState.COLLECTING:
            raise ContractValidationError(f"cannot {action} while session is {self._state.value}")
