"""Per-sample persistence for held-out test points, mirroring the calibration dataset.

The existing live-validation artifact (``validation_logs/validation_*.json``)
stores only per-target aggregates -- a median and a p95.  A memorization check
needs the individual samples, so that the same analysis code can compute a
per-point X/Y breakdown for the calibration file and the test file and compare
them head to head.

This module therefore mirrors ``calibration/dataset_*.json`` exactly.
:class:`TestSample` produces a byte-identical per-sample dict to
:class:`~gazelink.calibration.CalibrationSample`; the only difference is that
``target_index`` is not capped at the 9 calibration targets.  Field validation
is not reimplemented here -- every sample is validated by constructing a real
``CalibrationSample`` carrier, so there is exactly one definition of what a
valid sample looks like.

Nothing in this module trains, maps, corrects, or promotes anything.  It
records what was observed and writes it down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gazelink.calibration import FEATURE_SCHEMA_VERSION, CalibrationSample
from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    JSONValue,
    NormalizedPoint,
    ScreenGeometry,
    VisionObservation,
)
from gazelink.gaze_features import GazeFeatureVector, from_calibration_sample
from gazelink.live_validation import LiveValidationTarget, LiveValidationView

TEST_POINT_DIR = Path(".gazelink") / "test_points"

# The carrier index used purely to satisfy CalibrationSample's 0..8 target
# range while reusing its field validation.  from_calibration_sample() never
# reads target_index, so this value cannot leak into any feature or metric.
_CARRIER_TARGET_INDEX = 0

_CARRIED_FIELDS = (
    "source_frame_id",
    "observed_at_monotonic_ms",
    "left_iris_in_eye",
    "right_iris_in_eye",
    "left_openness",
    "right_openness",
    "left_iris_in_lids_y",
    "right_iris_in_lids_y",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "confidence",
    "accepted",
    "reason",
)


@dataclass(frozen=True, slots=True)
class TestSample:
    """One observation on a held-out test point.

    Field names, types and ``to_dict()`` output are identical to
    :class:`~gazelink.calibration.CalibrationSample`.  ``target_index`` is the
    sole difference: it indexes this file's own test targets, which may number
    more than the 9 the calibration contract allows.
    """

    __test__ = False  # not a pytest test class despite the name

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
        if (
            isinstance(self.target_index, bool)
            or not isinstance(self.target_index, int)
            or self.target_index < 0
        ):
            raise ContractValidationError("target_index must be a non-negative integer")
        # Validate and normalize every other field through the real
        # CalibrationSample contract rather than restating it here.
        carrier = self._build_carrier()
        for name in _CARRIED_FIELDS:
            object.__setattr__(self, name, getattr(carrier, name))

    def _build_carrier(self) -> CalibrationSample:
        return CalibrationSample(
            source_frame_id=self.source_frame_id,
            observed_at_monotonic_ms=self.observed_at_monotonic_ms,
            target_index=_CARRIER_TARGET_INDEX,
            left_iris_in_eye=self.left_iris_in_eye,
            right_iris_in_eye=self.right_iris_in_eye,
            left_openness=self.left_openness,
            right_openness=self.right_openness,
            left_iris_in_lids_y=self.left_iris_in_lids_y,
            right_iris_in_lids_y=self.right_iris_in_lids_y,
            head_yaw_deg=self.head_yaw_deg,
            head_pitch_deg=self.head_pitch_deg,
            head_roll_deg=self.head_roll_deg,
            confidence=self.confidence,
            accepted=self.accepted,
            reason=self.reason,
        )

    def features(self) -> GazeFeatureVector | None:
        """Reuse the one real feature extractor; never a second copy of it."""

        return from_calibration_sample(self._build_carrier())

    def to_dict(self) -> dict[str, JSONValue]:
        payload = self._build_carrier().to_dict()
        payload["target_index"] = self.target_index
        return payload

    @classmethod
    def from_dict(cls, value: object) -> TestSample:
        data = _as_object(value)
        sample = CalibrationSample.from_dict({**data, "target_index": _CARRIER_TARGET_INDEX})
        return cls(
            source_frame_id=sample.source_frame_id,
            observed_at_monotonic_ms=sample.observed_at_monotonic_ms,
            target_index=data.get("target_index"),  # type: ignore[arg-type]
            left_iris_in_eye=sample.left_iris_in_eye,
            right_iris_in_eye=sample.right_iris_in_eye,
            left_openness=sample.left_openness,
            right_openness=sample.right_openness,
            left_iris_in_lids_y=sample.left_iris_in_lids_y,
            right_iris_in_lids_y=sample.right_iris_in_lids_y,
            head_yaw_deg=sample.head_yaw_deg,
            head_pitch_deg=sample.head_pitch_deg,
            head_roll_deg=sample.head_roll_deg,
            confidence=sample.confidence,
            accepted=sample.accepted,
            reason=sample.reason,
        )


def _as_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError("test sample must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class TestPointResult:
    """Frozen snapshot of one held-out test run; the mirror of a calibration dataset."""

    __test__ = False  # not a pytest test class despite the name

    targets: tuple[LiveValidationTarget, ...]
    sample_counts: tuple[int, ...]
    samples: tuple[TestSample, ...]
    camera_id: str
    screen_geometry: ScreenGeometry
    seed: int
    overlay_model_id: str | None
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    min_samples_per_target: int = 5

    def __post_init__(self) -> None:
        if not self.targets:
            raise ContractValidationError("a test result requires at least one target")
        if len(self.sample_counts) != len(self.targets):
            raise ContractValidationError("sample_counts must have one entry per target")
        if any(sample.target_index >= len(self.targets) for sample in self.samples):
            raise ContractValidationError("every sample must reference a declared test target")
        if not isinstance(self.screen_geometry, ScreenGeometry):
            raise ContractValidationError("screen_geometry must be a ScreenGeometry")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ContractValidationError("camera_id must be a non-empty string")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "targets": [
                {
                    "index": index,
                    "name": target.name,
                    "screen_position": target.screen_position.to_dict(),
                }
                for index, target in enumerate(self.targets)
            ],
            "sample_counts": list(self.sample_counts),
            "samples": [sample.to_dict() for sample in self.samples],
            "camera_id": self.camera_id,
            "screen_geometry": self.screen_geometry.to_dict(),
            "feature_schema_version": self.feature_schema_version,
            "min_samples_per_target": self.min_samples_per_target,
            "seed": self.seed,
            "overlay_model_id": self.overlay_model_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> TestPointResult:
        data = _as_object(value)
        targets = tuple(
            LiveValidationTarget(
                str(_as_object(entry)["name"]),
                GazePoint.from_dict(_as_object(entry)["screen_position"]),
            )
            for entry in data.get("targets", ())
        )
        return cls(
            targets=targets,
            sample_counts=tuple(data.get("sample_counts", ())),
            samples=tuple(TestSample.from_dict(item) for item in data.get("samples", ())),
            camera_id=data.get("camera_id"),  # type: ignore[arg-type]
            screen_geometry=ScreenGeometry.from_dict(data.get("screen_geometry")),
            seed=data.get("seed"),  # type: ignore[arg-type]
            overlay_model_id=data.get("overlay_model_id"),
            feature_schema_version=data.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            min_samples_per_target=data.get("min_samples_per_target", 5),
        )


class TestSampleRecorder:
    """Record per-sample rows alongside a :class:`LiveValidationController`.

    This is deliberately a wrapper that infers what happened from the returned
    :class:`LiveValidationView` alone, rather than an edit to
    ``live_validation.py``.  The controller that drives ``--gaze-validation``
    stays byte-for-byte unchanged, so the test run's acceptance semantics are
    provably the same ones already in use.

    The inference is self-checking: when the controller reports a completed
    target it also reports how many samples it counted, and this recorder
    refuses to commit a row set whose size disagrees.  A silent mismatch
    between what the controller measured and what was written to disk would
    make every number in the report untrustworthy.
    """

    __test__ = False  # not a pytest test class despite the name

    def __init__(self) -> None:
        self._pending: list[TestSample] = []
        self._committed: list[TestSample] = []
        self._counts: list[int] = []

    @property
    def samples(self) -> tuple[TestSample, ...]:
        return tuple(self._committed)

    @property
    def sample_counts(self) -> tuple[int, ...]:
        return tuple(self._counts)

    def observe(
        self,
        view: LiveValidationView,
        observation: VisionObservation | None,
    ) -> None:
        """Mirror one controller tick; call immediately after ``controller.ingest``."""

        if not isinstance(view, LiveValidationView):
            raise ContractValidationError("view must be a LiveValidationView")
        completed = len(view.measurements)
        if completed < len(self._counts):
            # The operator restarted the run: drop everything, so rows from
            # two different passes can never be blended into one file.
            self._pending.clear()
            self._committed.clear()
            self._counts.clear()
            return
        if completed > len(self._counts):
            measurement = view.measurements[-1]
            target_index = completed - 1
            if observation is not None and len(self._pending) < measurement.sample_count:
                self._pending.append(_sample_from(observation, target_index))
            if len(self._pending) != measurement.sample_count:
                raise ContractValidationError(
                    f"recorded {len(self._pending)} rows for test target {target_index} "
                    f"but the controller measured {measurement.sample_count}; refusing to "
                    f"write a dataset that does not match the reported measurement"
                )
            self._committed.extend(self._pending)
            self._counts.append(len(self._pending))
            self._pending.clear()
            return
        if view.accepted_samples == 0:
            self._pending.clear()
        elif view.accepted_samples > len(self._pending) and observation is not None:
            self._pending.append(_sample_from(observation, completed))


def _sample_from(observation: VisionObservation, target_index: int) -> TestSample:
    left_eye = observation.left_eye
    right_eye = observation.right_eye
    pose = observation.head_pose
    return TestSample(
        source_frame_id=observation.frame_id,
        observed_at_monotonic_ms=observation.observed_at_monotonic_ms,
        target_index=target_index,
        left_iris_in_eye=None if left_eye is None else left_eye.iris_in_eye,
        right_iris_in_eye=None if right_eye is None else right_eye.iris_in_eye,
        left_openness=None if left_eye is None else left_eye.openness,
        right_openness=None if right_eye is None else right_eye.openness,
        left_iris_in_lids_y=None if left_eye is None else left_eye.iris_in_lids_y,
        right_iris_in_lids_y=None if right_eye is None else right_eye.iris_in_lids_y,
        head_yaw_deg=None if pose is None else pose.yaw_deg,
        head_pitch_deg=None if pose is None else pose.pitch_deg,
        head_roll_deg=None if pose is None else pose.roll_deg,
        confidence=observation.overall_confidence,
        accepted=True,
        reason="accepted",
    )


def format_test_dataset_log(result: TestPointResult, *, generated_at: datetime) -> str:
    """Human-readable mirror of the calibration session log; no image data."""

    accepted = sum(1 for sample in result.samples if sample.accepted)
    lines = [
        "GAZELINK -- Held-out test point session",
        f"Generated: {generated_at.astimezone(UTC).isoformat()}",
        f"Camera: {result.camera_id}",
        f"Screen: {result.screen_geometry.screen_id} {result.screen_geometry.width_px}x"
        f"{result.screen_geometry.height_px}",
        f"Seed: {result.seed}",
        f"Overlay model: {result.overlay_model_id or 'none'}",
        "No OS input was emitted.",
        "",
        "--- Targets ---",
    ]
    for index, target in enumerate(result.targets):
        lines.append(
            f"{index}: {target.name} "
            f"({target.screen_position.x:.4f}, {target.screen_position.y:.4f})"
        )
    lines.append("")
    lines.append(f"sample_counts: {result.sample_counts}")
    lines.append("")
    lines.append(f"--- Dataset ({len(result.samples)} samples, {accepted} accepted) ---")
    for sample in result.samples:
        lines.append(
            f"target={sample.target_index} accepted={sample.accepted} "
            f"reason={sample.reason} confidence={sample.confidence:.2f}"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class TestDatasetPaths:
    __test__ = False  # not a pytest test class despite the name

    json_path: Path
    text_path: Path


def write_test_dataset(
    result: TestPointResult,
    *,
    directory: Path = TEST_POINT_DIR,
    generated_at: datetime | None = None,
) -> TestDatasetPaths:
    """Write the test dataset to its own directory, never beside calibration data."""

    if not isinstance(result, TestPointResult):
        raise ContractValidationError("result must be a TestPointResult")
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"test_dataset_{stamp}.json"
    text_path = directory / f"test_dataset_{stamp}.txt"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    text_path.write_text(format_test_dataset_log(result, generated_at=timestamp), encoding="utf-8")
    return TestDatasetPaths(json_path=json_path, text_path=text_path)
