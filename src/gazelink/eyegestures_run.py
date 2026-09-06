"""Calibrate, freeze, then measure -- and record what actually happened.

A run of the external engine has three phases that must never blur into each
other.  While the library is calibrating it is training on what it sees, so any
"accuracy" read then is the model scoring its own training points.  Between
calibrating and measuring the training has to actually stop, which is not the
same as being told to stop.  Only after that is a number worth writing down.

This module owns the parts of that with no Qt and no camera in them: the phase
machine, the freeze verification, the per-target timing, and the file the run
produces.  The window drives it; everything here can be exercised end to end in
a test with a scripted engine.

Two rules the format depends on, both easy to get backwards:

* ``accepted`` means THE LIBRARY RETURNED A USABLE PREDICTION -- present,
  finite, not stale.  It does not mean our confidence gate approved the frame.
  ``analyze.py`` hides ``accepted=false`` rows by default, so putting our gate
  there would quietly delete most of the library's output from its own
  measurement.  The gate's verdict rides along in ``gate_would_accept``, and a
  second file is written where that verdict *is* the filter, so both pictures
  exist without either being the silent default.
* A prediction off the edge of the screen is a real prediction.  It stays
  ``accepted``, and its coordinates are written unclamped, because clamping it
  to the edge before measuring would shrink the error it deserves.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    GazeSample,
    JSONValue,
    ScreenGeometry,
    TrackingState,
)
from gazelink.live_validation import LiveValidationTarget
from gazelink.screen_mapping import drawn_center

# How close counts as "arrived", for the reaction-time measurement. Arrival
# depends on how big the thing you are trying to hit is, exactly as hit rate
# does, so it is reported at several radii rather than one arbitrary one.
DEFAULT_ARRIVAL_RADII_PX: tuple[float, ...] = (100.0, 200.0, 400.0)

# The library averages its last 20 predictions, so after a saccade it needs
# roughly 20 frames -- about 1.3s at the 66ms external-engine tick -- before it
# has flushed the previous target out of its own buffer. Settling for less than
# that measures the tail of the last target.
DEFAULT_SETTLE_FLOOR_MS = 1500.0

# An upper bound on waiting for the library's training threads to finish. It
# exists so the screen can never look hung; reaching it is not an error, it
# just means the freeze could not be verified and the run is diagnostic.
DEFAULT_FREEZE_TIMEOUT_MS = 10_000.0


class MeasurementPhase(StrEnum):
    """Where a run is. Recorded on every sample, so contamination is visible."""

    ENGINE_CALIBRATING = "ENGINE_CALIBRATING"
    FREEZING = "FREEZING"
    MEASURING = "MEASURING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class FreezeStatus:
    """The result of one freeze-gate tick."""

    ready: bool
    verified: bool
    pending_threads: int | None
    elapsed_ms: float
    timed_out: bool
    detail: str


class FreezeGate:
    """Waits for the library to actually stop training, and says whether it did.

    Not a timer.  ``Calibrator.add`` starts a thread per sample and offers no
    public join, so a fit launched just before the freeze can still land after
    it and move the model.  This waits for those threads to finish and compares
    the model's coefficients across the wait; a fixed sleep would prove neither.

    It is polled, never blocking, so the window stays responsive and ``Esc``
    keeps working.  If the wait exceeds ``timeout_ms`` it gives up and reports
    ``verified=False`` rather than waiting forever -- an unverified measurement
    that says so is more useful than a screen that appears hung.
    """

    def __init__(
        self,
        *,
        settle_floor_ms: float = DEFAULT_SETTLE_FLOOR_MS,
        timeout_ms: float = DEFAULT_FREEZE_TIMEOUT_MS,
    ) -> None:
        if settle_floor_ms < 0.0 or timeout_ms <= 0.0:
            raise ContractValidationError("settle_floor_ms must be >= 0 and timeout_ms > 0")
        self._settle_floor_ms = settle_floor_ms
        self._timeout_ms = timeout_ms
        self._started_at_ms: float | None = None
        self._fingerprint_at_start: tuple[float, ...] | None = None
        self._fingerprint_readable = False

    def start(self, now_ms: float, fingerprint: tuple[float, ...] | None) -> None:
        self._started_at_ms = now_ms
        self._fingerprint_at_start = fingerprint
        self._fingerprint_readable = fingerprint is not None

    def update(
        self, now_ms: float, *, pending_threads: int | None, fingerprint: tuple[float, ...] | None
    ) -> FreezeStatus:
        if self._started_at_ms is None:
            raise ContractValidationError("FreezeGate.update called before start")
        elapsed = now_ms - self._started_at_ms
        timed_out = elapsed >= self._timeout_ms

        threads_quiet = pending_threads == 0
        settled = elapsed >= self._settle_floor_ms
        # Unreadable internals are not evidence of anything. The run may still
        # proceed -- but only once the settle floor has passed, and it is
        # reported as unverified.
        introspectable = pending_threads is not None and self._fingerprint_readable

        if introspectable and threads_quiet and settled:
            # Deliberately NOT comparing against the fingerprint taken at the
            # moment of freezing. A fit launched before the freeze may land
            # during settling; that is expected and harmless, because settling
            # ends only once no thread is running. What must not happen is the
            # model moving DURING the measurement, and that is checked at the
            # end of the run instead -- comparing here rejected runs that were
            # in fact stable by the time anything was recorded.
            return FreezeStatus(
                ready=True,
                verified=True,
                pending_threads=pending_threads,
                elapsed_ms=elapsed,
                timed_out=False,
                detail="no training threads running when measurement began",
            )
        if timed_out:
            return FreezeStatus(
                ready=True,
                verified=False,
                pending_threads=pending_threads,
                elapsed_ms=elapsed,
                timed_out=True,
                detail=(
                    f"gave up waiting after {elapsed / 1000:.1f}s "
                    f"(threads still running: {_thread_text(pending_threads)})"
                ),
            )
        if not introspectable and settled:
            return FreezeStatus(
                ready=True,
                verified=False,
                pending_threads=pending_threads,
                elapsed_ms=elapsed,
                timed_out=False,
                detail="the library's internals could not be read; freeze NOT verified",
            )
        return FreezeStatus(
            ready=False,
            verified=False,
            pending_threads=pending_threads,
            elapsed_ms=elapsed,
            timed_out=False,
            detail=(
                f"settling {elapsed / 1000:.1f}s / {self._settle_floor_ms / 1000:.1f}s — "
                f"threads still running: {_thread_text(pending_threads)}"
            ),
        )


def model_stability(
    at_first_sample: tuple[float, ...] | None,
    at_last_sample: tuple[float, ...] | None,
) -> tuple[bool | None, str]:
    """Did the engine's model stay put across the recorded samples?

    This is the check that matters. A fit landing while settling is expected
    and harmless, because settling ends only once no training thread is
    running. A model that differs between the first and last recorded row is
    not: those rows then describe more than one model, and their median
    describes none of them.

    None means the question could not be answered, which must never be
    reported as a yes.
    """

    if at_first_sample is None or at_last_sample is None:
        return None, (
            "could not read the engine's model, so stability during measurement was NOT verified"
        )
    if at_first_sample == at_last_sample:
        return True, "model identical at the first and last recorded sample"
    return False, (
        "MODEL CHANGED DURING MEASUREMENT: the recorded rows do not all describe the same model"
    )


def _thread_text(pending: int | None) -> str:
    return "unknown" if pending is None else str(pending)


@dataclass(frozen=True, slots=True)
class RunSample:
    """One recorded frame during MEASURING."""

    target_index: int
    predicted: GazePoint
    timestamp_ms: float
    accepted: bool
    gate_would_accept: bool
    phase: MeasurementPhase
    engine_calibration_flag: bool
    source_frame_id: int
    tracking_state: TrackingState | None
    reason_codes: tuple[str, ...]
    confidence: float
    library_pixel: tuple[float, float] | None
    # Which window this row belongs to. A target is displayed, the eyes travel
    # to it, and only then does the controller start collecting. Rows from the
    # travelling part describe how fast the gaze arrives, not how accurately it
    # rests -- averaging the two produces a number that is neither.
    collection_phase: str | None = None
    # The rate actually achieved, not the timer we asked for. The library
    # averages a fixed number of FRAMES, so its lag in seconds depends entirely
    # on this; a nominal interval says nothing about what really happened.
    fps: float | None = None
    latency_ms: float | None = None
    # Diagnostic only, never scored. Head movement is a requirement, so a run
    # has to record how much of it there was, or posture error and mapping
    # error stay indistinguishable in a median.
    head_pose: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        # The first five keys are the only ones analyze.py reads; everything
        # else rides along and is ignored by its parser, which is what lets the
        # diagnostic detail live in the same file as the scored numbers.
        return {
            "target_index": self.target_index,
            "predicted_x": self.predicted.x,
            "predicted_y": self.predicted.y,
            "timestamp": self.timestamp_ms,
            "accepted": self.accepted,
            "gate_would_accept": self.gate_would_accept,
            "phase": str(self.phase.value),
            "engine_calibration_flag": self.engine_calibration_flag,
            "source_frame_id": self.source_frame_id,
            "tracking_state": None
            if self.tracking_state is None
            else str(self.tracking_state.value),
            "reason_codes": list(self.reason_codes),
            "confidence": self.confidence,
            "library_pixel_x": None if self.library_pixel is None else self.library_pixel[0],
            "library_pixel_y": None if self.library_pixel is None else self.library_pixel[1],
            "collection_phase": self.collection_phase,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "head_yaw_deg": None if self.head_pose is None else self.head_pose[0],
            "head_pitch_deg": None if self.head_pose is None else self.head_pose[1],
            "head_roll_deg": None if self.head_pose is None else self.head_pose[2],
        }


@dataclass
class TargetTiming:
    """How long the gaze took to reach one target, per arrival radius.

    Timed from the moment the target was DISPLAYED, including the settling
    window -- measuring only after the settle would hide the very latency this
    is here to expose.  A target never reached records ``None``, never a large
    number, because a large number would silently average as if it had been
    reached late.
    """

    target_index: int
    shown_at_ms: float
    arrivals_ms: dict[float, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "target_index": self.target_index,
            "shown_at_ms": self.shown_at_ms,
            "time_to_target_ms": {
                str(int(radius)): value for radius, value in self.arrivals_ms.items()
            },
        }


class ArrivalTimer:
    """Records the first moment gaze lands within each radius of each target."""

    def __init__(
        self,
        geometry: ScreenGeometry,
        *,
        radii_px: Sequence[float] = DEFAULT_ARRIVAL_RADII_PX,
    ) -> None:
        if not radii_px:
            raise ContractValidationError("radii_px must not be empty")
        self._geometry = geometry
        self._radii = tuple(sorted(float(radius) for radius in radii_px))
        self._timings: dict[int, TargetTiming] = {}

    @property
    def radii_px(self) -> tuple[float, ...]:
        return self._radii

    def target_shown(self, target_index: int, now_ms: float) -> None:
        if target_index in self._timings:
            return
        self._timings[target_index] = TargetTiming(
            target_index=target_index,
            shown_at_ms=now_ms,
            arrivals_ms=dict.fromkeys(self._radii),
        )

    def observe(
        self, target_index: int, predicted: GazePoint, target: GazePoint, now_ms: float
    ) -> None:
        timing = self._timings.get(target_index)
        if timing is None:
            return
        distance = _pixel_distance(predicted, target, self._geometry)
        for radius in self._radii:
            if timing.arrivals_ms[radius] is None and distance <= radius:
                timing.arrivals_ms[radius] = now_ms - timing.shown_at_ms

    def results(self) -> tuple[TargetTiming, ...]:
        return tuple(self._timings[key] for key in sorted(self._timings))


def _pixel_distance(a: GazePoint, b: GazePoint, geometry: ScreenGeometry) -> float:
    dx = (a.x - b.x) * (geometry.width_px - 1)
    dy = (a.y - b.y) * (geometry.height_px - 1)
    return float((dx * dx + dy * dy) ** 0.5)


def sample_from_gaze(
    *,
    target_index: int,
    sample: GazeSample | None,
    phase: MeasurementPhase,
    gate_accepts: bool,
    engine_calibration_flag: bool,
    geometry: ScreenGeometry,
    tracking_state: TrackingState | None,
    now_ms: float,
    frame_id: int,
    collection_phase: str | None = None,
    fps: float | None = None,
    latency_ms: float | None = None,
    head_pose: tuple[float, float, float] | None = None,
) -> RunSample | None:
    """Build one row, or ``None`` when the library produced no prediction.

    ``accepted`` is ``True`` here for every row that exists at all: a row only
    exists when the library returned a usable point.  A frame with no
    prediction is not a rejected measurement, it is an absent one, and inventing
    a row for it would put a coordinate in the file that nothing produced.
    """

    if sample is None:
        return None
    predicted = sample.raw_normalized
    return RunSample(
        target_index=target_index,
        predicted=predicted,
        timestamp_ms=now_ms,
        accepted=True,
        gate_would_accept=gate_accepts,
        phase=phase,
        engine_calibration_flag=engine_calibration_flag,
        source_frame_id=frame_id,
        tracking_state=tracking_state,
        reason_codes=tuple(str(reason.value) for reason in sample.reason_codes),
        confidence=sample.confidence,
        library_pixel=(
            predicted.x * geometry.width_px,
            predicted.y * geometry.height_px,
        ),
        collection_phase=collection_phase,
        fps=fps,
        latency_ms=latency_ms,
        head_pose=head_pose,
    )


def build_predictions_payload(
    *,
    targets: Sequence[LiveValidationTarget],
    geometry: ScreenGeometry,
    samples: Sequence[RunSample],
    timings: Sequence[TargetTiming],
    run_metadata: dict[str, JSONValue],
    gate_filtered: bool = False,
    target_glyph_px: int = 96,
) -> dict[str, JSONValue]:
    """The ``analyze.py --predictions`` file, plus everything else we know.

    ``gate_filtered`` produces the companion file in which ``accepted`` also
    requires our confidence gate to have approved the frame.  Writing both, and
    naming which is which, is how the gate stays visible without becoming the
    silent default.
    """

    target_dicts: list[JSONValue] = []
    for index, target in enumerate(targets):
        point = target.screen_position
        centre = drawn_center(
            _centered(point, geometry, target_glyph_px),
            glyph_width_px=target_glyph_px,
            glyph_height_px=target_glyph_px,
        )
        target_dicts.append(
            {
                "index": index,
                "name": target.name,
                "screen_position": {"x": point.x, "y": point.y},
                # Where the glyph's centre really landed. With this in the file
                # the display/measurement agreement is checkable from the file
                # alone, without re-running anything.
                "drawn_target_center_px": {"x_px": centre.x_px, "y_px": centre.y_px},
            }
        )

    sample_dicts: list[JSONValue] = []
    for run_sample in samples:
        entry = run_sample.to_dict()
        if gate_filtered:
            entry["accepted"] = run_sample.accepted and run_sample.gate_would_accept
        sample_dicts.append(entry)

    metadata = dict(run_metadata)
    metadata["gate_filtered"] = gate_filtered
    metadata["units"] = "qt_logical_px"
    metadata["dpi_scale"] = geometry.dpi_scale
    metadata["physical_width_px"] = round(geometry.width_px * geometry.dpi_scale)
    metadata["physical_height_px"] = round(geometry.height_px * geometry.dpi_scale)

    return {
        "screen_geometry": geometry.to_dict(),
        "targets": target_dicts,
        "samples": sample_dicts,
        "timings": [timing.to_dict() for timing in timings],
        "run": metadata,
    }


def _centered(point: GazePoint, geometry: ScreenGeometry, glyph_px: int) -> tuple[int, int]:
    from gazelink.screen_mapping import centered_top_left  # noqa: PLC0415 - avoids a cycle

    return centered_top_left(point, geometry, glyph_width_px=glyph_px, glyph_height_px=glyph_px)


def write_predictions(path: Path, payload: dict[str, Any]) -> Path:
    """Write one predictions file atomically, creating parents as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


@dataclass(frozen=True, slots=True)
class PhaseView:
    """What the screen should say right now, and whether to record."""

    phase: MeasurementPhase
    banner: str
    accent: str
    recording: bool
    freeze_verified: bool
    freeze_detail: str | None


# Two visually distinct colours so an operator cannot confuse "still training"
# with "measuring" from across the room.
_ACCENT_CALIBRATING = "#FFB300"
_ACCENT_FREEZING = "#FFB300"
_ACCENT_MEASURING = "#00E676"
_ACCENT_DONE = "#B0BEC5"
_ACCENT_ABORTED = "#FF5252"


class PhaseMachine:
    """Calibrating -> freezing -> measuring, and never backwards.

    Kept out of the window so the ordering can be tested without a display.
    The transition that matters is the middle one: nothing may be recorded
    until the library has been told to stop learning AND that has been checked.
    """

    def __init__(self, *, freeze_gate: FreezeGate | None = None, total_points: int) -> None:
        self._gate = freeze_gate or FreezeGate()
        self._total_points = total_points
        self._phase = MeasurementPhase.ENGINE_CALIBRATING
        self._freeze_verified = False
        self._freeze_detail: str | None = None
        self._freeze_started = False

    @property
    def phase(self) -> MeasurementPhase:
        return self._phase

    @property
    def freeze_verified(self) -> bool:
        return self._freeze_verified

    @property
    def freeze_detail(self) -> str | None:
        return self._freeze_detail

    def abort(self, reason: str) -> PhaseView:
        self._phase = MeasurementPhase.ABORTED
        self._freeze_detail = reason
        return self._view(reason)

    def update(
        self,
        now_ms: float,
        *,
        is_calibrated: bool,
        completed_points: int,
        pending_threads: int | None,
        fingerprint: tuple[float, ...] | None,
        measurement_complete: bool,
        on_freeze: Any = None,
    ) -> PhaseView:
        if self._phase in (MeasurementPhase.COMPLETE, MeasurementPhase.ABORTED):
            return self._view()

        if self._phase is MeasurementPhase.ENGINE_CALIBRATING:
            if not is_calibrated:
                return self._view(f"{completed_points}/{self._total_points}")
            # Freeze the moment calibration finishes, then verify it separately;
            # asking and confirming are different acts and are kept apart.
            if on_freeze is not None:
                on_freeze()
            self._gate.start(now_ms, fingerprint)
            self._freeze_started = True
            self._phase = MeasurementPhase.FREEZING
            return self._view("freezing")

        if self._phase is MeasurementPhase.FREEZING:
            status = self._gate.update(
                now_ms, pending_threads=pending_threads, fingerprint=fingerprint
            )
            if not status.ready:
                return self._view(status.detail)
            self._freeze_verified = status.verified
            self._freeze_detail = status.detail
            self._phase = MeasurementPhase.MEASURING
            return self._view()

        if measurement_complete:
            self._phase = MeasurementPhase.COMPLETE
        return self._view()

    def _view(self, detail: str = "") -> PhaseView:
        if self._phase is MeasurementPhase.ENGINE_CALIBRATING:
            return PhaseView(
                self._phase,
                f"EyeGestures is calibrating — {detail} — NOTHING IS BEING MEASURED",
                _ACCENT_CALIBRATING,
                False,
                self._freeze_verified,
                self._freeze_detail,
            )
        if self._phase is MeasurementPhase.FREEZING:
            return PhaseView(
                self._phase,
                f"Calibration frozen. Settling… {detail}  (Esc to cancel)",
                _ACCENT_FREEZING,
                False,
                self._freeze_verified,
                self._freeze_detail,
            )
        if self._phase is MeasurementPhase.MEASURING:
            suffix = "" if self._freeze_verified else "  [FREEZE NOT VERIFIED — diagnostic only]"
            return PhaseView(
                self._phase,
                f"MEASURING — learning is OFF{suffix}",
                _ACCENT_MEASURING,
                True,
                self._freeze_verified,
                self._freeze_detail,
            )
        if self._phase is MeasurementPhase.ABORTED:
            return PhaseView(
                self._phase,
                f"RUN ABORTED — nothing was recorded. {detail}",
                _ACCENT_ABORTED,
                False,
                self._freeze_verified,
                self._freeze_detail,
            )
        return PhaseView(
            self._phase,
            "Measurement complete.",
            _ACCENT_DONE,
            False,
            self._freeze_verified,
            self._freeze_detail,
        )
