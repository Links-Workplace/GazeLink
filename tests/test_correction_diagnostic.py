"""State-transition tests for the operator-driven correction diagnostic."""

from __future__ import annotations

import pytest

from gazelink.correction_diagnostic import (
    CorrectionDiagnosticSession,
    CorrectionDiagnosticState,
)
from gazelink.domain import (
    ContractValidationError,
    EyeFeatures,
    GazePoint,
    HeadPose,
    NormalizedPoint,
    ScreenGeometry,
    TrackingState,
    VisionObservation,
)
from gazelink.gaze_correction import CorrectionSettings, LocalCorrectionEngine
from gazelink.gaze_engine import CalibrationModel, GazeEstimator
from gazelink.gaze_model import GazeModelKind, RegressionModel


def _geometry() -> ScreenGeometry:
    return ScreenGeometry("primary", 1001, 501, 1.0)


def _estimator() -> GazeEstimator:
    model = CalibrationModel.create(
        regression=RegressionModel(
            GazeModelKind.LINEAR,
            (0.6,) + (0.0,) * 9,
            (0.5,) + (0.0,) * 9,
            0.0,
        ),
        screen_geometry=_geometry(),
        camera_id="camera-0",
        calibration_id="test-calibration",
        trained_at_utc="2026-09-02T00:00:00+00:00",
    )
    return GazeEstimator(model, live_screen_geometry=_geometry())


def _observation(frame_id: int, at_ms: float) -> VisionObservation:
    return VisionObservation(
        frame_id=frame_id,
        observed_at_monotonic_ms=at_ms,
        tracking_state=TrackingState.TRACKED,
        face_box=None,
        left_eye=EyeFeatures(None, 0.5, 0.5, NormalizedPoint(0.4, 0.5)),
        right_eye=EyeFeatures(None, 0.55, 0.5, NormalizedPoint(0.6, 0.5)),
        head_pose=HeadPose(0.0, 0.0, 0.0),
        overall_confidence=0.5,
    )


def _session() -> CorrectionDiagnosticSession:
    estimator = _estimator()
    settings = CorrectionSettings(capture_window_ms=40.0, min_capture_samples=2)
    return CorrectionDiagnosticSession(
        estimator,
        LocalCorrectionEngine(estimator.model_id, _geometry(), settings=settings),
        settings=settings,
        stabilization_ms=10.0,
    )


def _drive_to_review(session: CorrectionDiagnosticSession) -> None:
    session.start(GazePoint(0.8, 0.5), now_monotonic_ms=0.0)
    assert session.ingest(_observation(0, 10.0), now_monotonic_ms=10.0).state is (
        CorrectionDiagnosticState.CAPTURE_COLLECTING
    )
    session.ingest(_observation(1, 20.0), now_monotonic_ms=20.0)
    assert session.ingest(_observation(2, 60.0), now_monotonic_ms=60.0).state is (
        CorrectionDiagnosticState.VALIDATION_STABILIZING
    )
    assert session.ingest(_observation(9, 70.0), now_monotonic_ms=70.0).state is (
        CorrectionDiagnosticState.VALIDATION_COLLECTING
    )
    session.ingest(_observation(10, 80.0), now_monotonic_ms=80.0)
    assert session.ingest(_observation(11, 120.0), now_monotonic_ms=120.0).state is (
        CorrectionDiagnosticState.REVIEW
    )


def test_flow_requires_distinct_capture_and_validation_windows_before_accept() -> None:
    session = _session()
    _drive_to_review(session)
    review = session.view()
    assert review.can_accept
    assert review.before_error_normalized is not None
    assert review.after_error_normalized is not None
    assert review.after_error_normalized < review.before_error_normalized
    accepted = session.accept()
    assert accepted.state is CorrectionDiagnosticState.IDLE
    assert len(session.engine.corrections) == 1


def test_reject_undo_and_clear_are_explicit_recoverable_transitions() -> None:
    session = _session()
    _drive_to_review(session)
    session.reject()
    assert session.engine.corrections == ()
    _drive_to_review(session)
    session.accept()
    session.undo()
    assert session.engine.corrections == ()
    _drive_to_review(session)
    session.accept()
    session.clear()
    assert session.engine.corrections == ()


def test_accept_is_forbidden_before_successful_review() -> None:
    session = _session()
    with pytest.raises(ContractValidationError, match="no reviewed correction"):
        session.accept()
