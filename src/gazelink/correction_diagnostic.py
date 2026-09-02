"""Pure state machine for the operator-driven M2 correction diagnostic flow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from gazelink.domain import ContractValidationError, GazePoint, VisionObservation
from gazelink.gaze_correction import (
    CorrectionCaptureResult,
    CorrectionSettings,
    CorrectionValidationResult,
    LocalCorrectionEngine,
    accept_candidate,
    capture_correction,
    validate_candidate,
)
from gazelink.gaze_engine import GazeEstimator

DEFAULT_STABILIZATION_MS = 500.0  # UNVALIDATED PLACEHOLDER for diagnostic target settling.
VALIDATION_TARGET_OFFSET = 0.05  # UNVALIDATED PLACEHOLDER, normalized screen distance.


class CorrectionDiagnosticState(StrEnum):
    IDLE = "IDLE"
    CAPTURE_STABILIZING = "CAPTURE_STABILIZING"
    CAPTURE_COLLECTING = "CAPTURE_COLLECTING"
    VALIDATION_STABILIZING = "VALIDATION_STABILIZING"
    VALIDATION_COLLECTING = "VALIDATION_COLLECTING"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class CorrectionDiagnosticView:
    state: CorrectionDiagnosticState
    target: GazePoint | None
    instruction: str
    before_error_normalized: float | None = None
    after_error_normalized: float | None = None
    can_accept: bool = False


class CorrectionDiagnosticSession:
    """Collect, validate, and explicitly commit one correction at a time."""

    def __init__(
        self,
        estimator: GazeEstimator,
        engine: LocalCorrectionEngine,
        *,
        settings: CorrectionSettings | None = None,
        stabilization_ms: float = DEFAULT_STABILIZATION_MS,
    ) -> None:
        if not isinstance(estimator, GazeEstimator):
            raise ContractValidationError("estimator must be GazeEstimator")
        if not isinstance(engine, LocalCorrectionEngine):
            raise ContractValidationError("engine must be LocalCorrectionEngine")
        if engine.base_model_id != estimator.model_id:
            raise ContractValidationError("correction engine must match the gaze estimator model")
        if isinstance(stabilization_ms, bool) or not isinstance(stabilization_ms, (int, float)):
            raise ContractValidationError("stabilization_ms must be a positive number")
        if not math.isfinite(float(stabilization_ms)) or stabilization_ms <= 0:
            raise ContractValidationError("stabilization_ms must be a positive number")
        self._estimator = estimator
        self._engine = engine
        self._settings = settings or engine.settings
        self._stabilization_ms = float(stabilization_ms)
        self._state = CorrectionDiagnosticState.IDLE
        self._phase_started_ms = 0.0
        self._capture_target: GazePoint | None = None
        self._validation_target: GazePoint | None = None
        self._observations: list[VisionObservation] = []
        self._capture: CorrectionCaptureResult | None = None
        self._validation: CorrectionValidationResult | None = None

    @property
    def engine(self) -> LocalCorrectionEngine:
        return self._engine

    def start(self, target: GazePoint, *, now_monotonic_ms: float) -> CorrectionDiagnosticView:
        if (
            not isinstance(target, GazePoint)
            or not 0.0 <= target.x <= 1.0
            or not 0.0 <= target.y <= 1.0
        ):
            raise ContractValidationError("target must be a normalized on-screen GazePoint")
        self._capture_target = target
        self._validation_target = _nearby_validation_target(target)
        self._capture = None
        self._validation = None
        self._observations.clear()
        self._state = CorrectionDiagnosticState.CAPTURE_STABILIZING
        self._phase_started_ms = float(now_monotonic_ms)
        return self.view()

    def ingest(
        self, observation: VisionObservation, *, now_monotonic_ms: float
    ) -> CorrectionDiagnosticView:
        now = float(now_monotonic_ms)
        if self._state in {CorrectionDiagnosticState.IDLE, CorrectionDiagnosticState.REVIEW}:
            return self.view()
        if self._state is CorrectionDiagnosticState.CAPTURE_STABILIZING:
            if now - self._phase_started_ms >= self._stabilization_ms:
                self._state = CorrectionDiagnosticState.CAPTURE_COLLECTING
                self._phase_started_ms = now
                self._observations.clear()
            return self.view()
        if self._state is CorrectionDiagnosticState.VALIDATION_STABILIZING:
            if now - self._phase_started_ms >= self._stabilization_ms:
                self._state = CorrectionDiagnosticState.VALIDATION_COLLECTING
                self._phase_started_ms = now
                self._observations.clear()
            return self.view()
        self._observations.append(observation)
        if now - self._phase_started_ms < self._settings.capture_window_ms:
            return self.view()
        if self._state is CorrectionDiagnosticState.CAPTURE_COLLECTING:
            assert self._capture_target is not None
            self._capture = capture_correction(
                self._observations,
                true_target=self._capture_target,
                estimator=self._estimator,
                now_monotonic_ms=now,
                settings=self._settings,
            )
            if self._capture.candidate is None:
                self._state = CorrectionDiagnosticState.REVIEW
            else:
                self._state = CorrectionDiagnosticState.VALIDATION_STABILIZING
                self._phase_started_ms = now
                self._observations.clear()
            return self.view()
        assert self._state is CorrectionDiagnosticState.VALIDATION_COLLECTING
        assert self._capture is not None and self._capture.candidate is not None
        assert self._validation_target is not None
        self._validation = validate_candidate(
            self._capture.candidate,
            self._observations,
            validation_target=self._validation_target,
            estimator=self._estimator,
            active_engine=self._engine,
            now_monotonic_ms=now,
            settings=self._settings,
        )
        self._state = CorrectionDiagnosticState.REVIEW
        return self.view()

    def accept(self) -> CorrectionDiagnosticView:
        if self._state is not CorrectionDiagnosticState.REVIEW or self._validation is None:
            raise ContractValidationError("there is no reviewed correction to accept")
        self._engine = accept_candidate(self._validation, self._engine)
        self._reset_flow()
        return self.view()

    def reject(self) -> CorrectionDiagnosticView:
        self._reset_flow()
        return self.view()

    def undo(self) -> CorrectionDiagnosticView:
        self._engine = self._engine.undo_last()
        self._reset_flow()
        return self.view()

    def clear(self) -> CorrectionDiagnosticView:
        self._engine = self._engine.clear()
        self._reset_flow()
        return self.view()

    def view(self) -> CorrectionDiagnosticView:
        if self._state is CorrectionDiagnosticState.IDLE:
            return CorrectionDiagnosticView(
                self._state,
                None,
                "Press 1-9 to show a known correction target; U=undo, R=reset, Esc=close.",
            )
        if self._state is CorrectionDiagnosticState.CAPTURE_STABILIZING:
            return CorrectionDiagnosticView(
                self._state, self._capture_target, "Look at the cyan target and hold still."
            )
        if self._state is CorrectionDiagnosticState.CAPTURE_COLLECTING:
            return CorrectionDiagnosticView(
                self._state, self._capture_target, "Capturing stable correction samples…"
            )
        if self._state is CorrectionDiagnosticState.VALIDATION_STABILIZING:
            return CorrectionDiagnosticView(
                self._state,
                self._validation_target,
                "Now look at the nearby green validation target and hold still.",
            )
        if self._state is CorrectionDiagnosticState.VALIDATION_COLLECTING:
            return CorrectionDiagnosticView(
                self._state, self._validation_target, "Capturing held-out validation samples…"
            )
        if self._validation is None:
            reasons = () if self._capture is None else self._capture.reason_codes
            labels = ", ".join(reason.value for reason in reasons) or "capture rejected"
            return CorrectionDiagnosticView(
                self._state, self._capture_target, f"Correction rejected: {labels}. X=retry."
            )
        validation = self._validation
        instruction = (
            "Improved: A=accept, X=reject."
            if validation.accepted
            else "No verified improvement: X=reject/retry."
        )
        return CorrectionDiagnosticView(
            self._state,
            self._validation_target,
            instruction,
            before_error_normalized=validation.before_error_normalized,
            after_error_normalized=validation.after_error_normalized,
            can_accept=validation.accepted,
        )

    def _reset_flow(self) -> None:
        self._state = CorrectionDiagnosticState.IDLE
        self._capture_target = None
        self._validation_target = None
        self._capture = None
        self._validation = None
        self._observations.clear()


def _nearby_validation_target(target: GazePoint) -> GazePoint:
    def move_toward_center(value: float) -> float:
        if value < 0.5:
            return min(1.0, value + VALIDATION_TARGET_OFFSET)
        if value > 0.5:
            return max(0.0, value - VALIDATION_TARGET_OFFSET)
        return value + VALIDATION_TARGET_OFFSET

    return GazePoint(move_toward_center(target.x), move_toward_center(target.y))
