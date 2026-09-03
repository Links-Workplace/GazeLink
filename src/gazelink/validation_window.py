"""Full-screen M2 live validation for a pending calibration model.

The window is diagnostic only: it never moves the cursor or emits an OS
action.  The candidate is promoted only after the operator reviews fresh
measurements and explicitly accepts it.
"""

from __future__ import annotations

from typing import Any

from gazelink.camera import CameraError
from gazelink.debug_window import DEFAULT_TIMER_INTERVAL_MS, build_runtime
from gazelink.domain import GazePoint, ReasonCode, ScreenGeometry
from gazelink.gaze_engine import CalibrationStore, GazeEstimationResult, GazeEstimator
from gazelink.gaze_features import from_observation
from gazelink.live_validation import (
    LiveValidationCandidate,
    LiveValidationComparisonController,
    LiveValidationComparisonView,
    LiveValidationReportPaths,
    build_calibration_feature_reference,
    format_live_validation_comparison_summary,
    format_live_validation_summary,
    write_live_validation_comparison_report,
    write_live_validation_report,
)
from gazelink.runtime import VisionRuntime

_WINDOW_TITLE = "GAZELINK — M2 calibration validation"


def run_gaze_validation(*, camera_index: int = 0) -> int:
    """Review a pending candidate against freshly captured known targets."""

    runtime = build_runtime(camera_index=camera_index)
    try:
        runtime.start()
    except CameraError as error:
        runtime.close()
        print(f"{error.failure.message} {error.failure.recovery_action.value}")
        return 1
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    application = QApplication.instance() or QApplication([])
    screen = application.primaryScreen()  # type: ignore[attr-defined]
    if screen is None:
        runtime.close()
        print("No primary display is available for validation.")
        return 1
    rectangle = screen.geometry()
    geometry = ScreenGeometry(
        screen_id=screen.name() or "primary",
        width_px=rectangle.width(),
        height_px=rectangle.height(),
        dpi_scale=float(screen.devicePixelRatio()),
    )
    store = CalibrationStore()
    pending = store.load_pending_model(geometry)
    if pending is None:
        runtime.close()
        print("No compatible pending calibration candidate. Run --guided-calibration first.")
        return 1
    estimators = tuple(
        GazeEstimator(model, live_screen_geometry=geometry) for model in pending.candidate_models
    )
    candidates = tuple(
        LiveValidationCandidate(model.model_id, model.regression.label)
        for model in pending.candidate_models
    )
    calibration_dataset = store.load_dataset_for_calibration(pending.model.calibration_id)
    feature_reference = (
        None
        if calibration_dataset is None
        else build_calibration_feature_reference(calibration_dataset)
    )
    window = _LiveValidationWindow(
        runtime,
        estimators,
        LiveValidationComparisonController(
            geometry,
            candidates=candidates,
            feature_reference=feature_reference,
        ),
        store,
    )
    window.show()
    try:
        return int(application.exec())
    finally:
        runtime.close()


class _LiveValidationWindow:  # pragma: no cover - requires display and live camera
    def __init__(
        self,
        runtime: VisionRuntime,
        estimators: tuple[GazeEstimator, ...],
        controller: LiveValidationComparisonController,
        store: CalibrationStore,
    ) -> None:
        from PySide6.QtCore import Qt, QTimer  # noqa: PLC0415
        from PySide6.QtGui import QFont  # noqa: PLC0415
        from PySide6.QtWidgets import (  # noqa: PLC0415
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self._runtime = runtime
        self._estimators = estimators
        self._controller = controller
        self._store = store
        self._closed = False
        self._report_paths: LiveValidationReportPaths | None = None
        self._widget = QWidget()
        self._widget.setWindowTitle(_WINDOW_TITLE)
        self._widget.setStyleSheet("background: #101820; color: white;")
        self._widget.setWindowState(Qt.WindowState.WindowFullScreen)
        self._widget.closeEvent = self._on_close  # type: ignore[method-assign]
        self._widget.keyPressEvent = self._on_key_press  # type: ignore[method-assign]
        self._target = QLabel("●", self._widget)
        font = QFont()
        font.setPointSize(44)
        font.setBold(True)
        self._target.setFont(font)
        self._target.setStyleSheet("color: #FFD740; background: transparent;")
        self._progress = QLabel(self._widget)
        self._feedback = QLabel(self._widget)
        self._summary = QLabel(self._widget)
        for label in (self._progress, self._feedback, self._summary):
            label.setWordWrap(True)
            label.setStyleSheet("background: rgba(16, 24, 32, 210); padding: 8px;")
        controls = QHBoxLayout()
        self._accept = QPushButton("Accept live recommendation (A)")
        self._reject = QPushButton("Reject candidate (X)")
        restart = QPushButton("Restart validation (R)")
        cancel = QPushButton("Close safely (Esc)")
        self._accept.clicked.connect(self._on_accept)
        self._reject.clicked.connect(self._on_reject)
        restart.clicked.connect(self._on_restart)
        cancel.clicked.connect(self._on_cancel)
        for button in (self._accept, self._reject, restart, cancel):
            controls.addWidget(button)
        layout = QVBoxLayout(self._widget)
        layout.addWidget(self._progress)
        layout.addWidget(self._feedback)
        layout.addWidget(self._summary)
        layout.addStretch(1)
        layout.addLayout(controls)
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(DEFAULT_TIMER_INTERVAL_MS)
        self._render(self._controller.view())

    def show(self) -> None:
        from PySide6.QtCore import QTimer  # noqa: PLC0415

        screen = self._widget.screen()
        if screen is not None:
            self._widget.setGeometry(screen.geometry())
        self._widget.show()
        self._widget.raise_()
        self._widget.activateWindow()
        QTimer.singleShot(0, self._show_full_screen)

    def _show_full_screen(self) -> None:
        self._widget.showFullScreen()
        self._widget.raise_()
        self._widget.activateWindow()

    def _on_tick(self) -> None:
        try:
            tick = self._runtime.tick()
        except CameraError as error:
            self._feedback.setText(f"{error.failure.message} {error.failure.recovery_action.value}")
            self._shutdown()
            return
        except Exception:
            self._shutdown()
            raise
        if tick is None:
            return
        now_ms = tick.frame.captured_at_monotonic_ms + (tick.latency_ms or 0.0)
        if tick.accepted_observation is None:
            features = None
            results = {
                estimator.model_id: GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE,))
                for estimator in self._estimators
            }
        else:
            features = from_observation(tick.accepted_observation)
            results = {
                estimator.model_id: estimator.estimate(
                    tick.accepted_observation, now_monotonic_ms=now_ms
                )
                for estimator in self._estimators
            }
        view = self._controller.ingest(results, now_monotonic_ms=now_ms, features=features)
        self._render(view)
        if view.complete:
            self._write_report(view)
            self._timer.stop()

    def _on_key_press(self, event: Any) -> None:
        from PySide6.QtCore import Qt  # noqa: PLC0415

        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
        elif event.key() == Qt.Key.Key_R:
            self._on_restart()
        elif event.key() == Qt.Key.Key_A:
            self._on_accept()
        elif event.key() == Qt.Key.Key_X:
            self._on_reject()
        else:
            event.ignore()

    def _on_restart(self) -> None:
        self._timer.start(DEFAULT_TIMER_INTERVAL_MS)
        self._report_paths = None
        self._render(self._controller.restart())

    def _on_accept(self) -> None:
        view = self._controller.view()
        if not view.complete:
            self._feedback.setText(
                "Complete every validation point before accepting the candidate."
            )
            return
        if view.recommended_model_id is None:
            self._feedback.setText(
                "No model was no-worse at every point; reject this calibration and retry."
            )
            return
        path = self._store.promote_pending_model(
            self._estimators[0].screen_geometry, model_id=view.recommended_model_id
        )
        if path is None:
            self._feedback.setText("Candidate was unavailable; no model was activated.")
            return
        print(format_live_validation_comparison_summary(view))
        print(f"Validation accepted; active model written to: {path}")
        self._feedback.setText("Candidate accepted and now active. No OS input was emitted.")
        self._accept.setEnabled(False)
        self._reject.setEnabled(False)

    def _on_reject(self) -> None:
        archived = self._store.reject_pending_model()
        print(format_live_validation_comparison_summary(self._controller.view()))
        print(f"Validation rejected; pending manifest archived at: {archived}")
        self._feedback.setText("Candidate rejected. No active model was changed.")
        self._accept.setEnabled(False)
        self._reject.setEnabled(False)

    def _write_report(self, view: LiveValidationComparisonView) -> None:
        if self._report_paths is not None:
            return
        if len(view.candidates) == 1:
            self._report_paths = write_live_validation_report(
                view.primary,
                model_id=view.candidates[0].candidate.model_id,
                geometry=self._estimators[0].screen_geometry,
            )
        else:
            self._report_paths = write_live_validation_comparison_report(
                view, geometry=self._estimators[0].screen_geometry
            )
        print(f"Validation text report written to: {self._report_paths.text_path}")
        print(f"Validation JSON report written to: {self._report_paths.json_path}")

    def _on_cancel(self) -> None:
        self._shutdown()
        self._widget.close()

    def _on_close(self, event: Any) -> None:
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._runtime.close()

    def _render(self, view: LiveValidationComparisonView) -> None:
        primary = view.primary
        self._progress.setText(
            f"Validation point {primary.target_number} of {primary.total_targets} — "
            f"{primary.accepted_samples}/{primary.required_samples} samples — {primary.phase.value}"
        )
        self._feedback.setText(primary.feedback)
        self._summary.setText(
            format_live_validation_comparison_summary(view)
            if len(view.candidates) > 1
            else format_live_validation_summary(primary)
        )
        self._target.setVisible(primary.target is not None)
        if primary.target is not None:
            self._place_target(primary.target.screen_position)
        self._accept.setEnabled(view.complete and view.recommended_model_id is not None)
        self._reject.setEnabled(view.complete)

    def _place_target(self, target: GazePoint) -> None:
        self._target.adjustSize()
        self._target.move(
            round(target.x * max(0, self._widget.width() - self._target.width())),
            round(target.y * max(0, self._widget.height() - self._target.height())),
        )
        self._target.raise_()
