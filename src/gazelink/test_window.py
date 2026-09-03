"""Full-screen held-out test-point surface: measures generalization, not memorization.

The 9-point calibration screen and the model comparison in
``validation_window.py`` both show targets a model was trained on, or targets
that partly overlap the training grid.  This window shows targets that are
held out by construction (see ``test_points.py``), collects the same raw
vision features calibration collects, and writes them to their own dataset
file for later offline analysis.  It never trains, maps, corrects, or
promotes a model, and it never emits OS input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gazelink.camera import CameraError
from gazelink.debug_window import DEFAULT_TIMER_INTERVAL_MS, build_runtime
from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    GazeSample,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    VisionObservation,
)
from gazelink.gaze_engine import GazeEstimationResult, GazeEstimator
from gazelink.live_validation import LiveValidationController, LiveValidationView
from gazelink.prediction_overlay import PredictionOverlay, load_overlay_model
from gazelink.runtime import VisionRuntime
from gazelink.test_dataset import TestPointResult, TestSampleRecorder, write_test_dataset
from gazelink.test_points import DEFAULT_TEST_SEED, TEST_POINT_COUNT, generate_test_targets

_WINDOW_TITLE = "GAZELINK — held-out test points"


def _heartbeat_result(observation: VisionObservation) -> GazeEstimationResult:
    """A model-free "tracking is present" signal that drives the controller's clock.

    ``LiveValidationController`` needs *a* :class:`GazeEstimationResult` to
    key its stabilize/collect timing off of, but this window's dataset never
    reads its point -- :class:`~gazelink.test_dataset.TestSampleRecorder`
    builds every recorded row from the raw ``VisionObservation`` instead.
    This placeholder exists only so the held-out test flow can run before any
    model has ever been trained, exactly like calibration itself does.
    """

    point = GazePoint(0.5, 0.5)
    sample = GazeSample(
        source_frame_id=observation.frame_id,
        sampled_at_monotonic_ms=observation.observed_at_monotonic_ms,
        raw_normalized=point,
        corrected_normalized=point,
        filtered_normalized=point,
        screen_position=PixelPoint(0, 0),
        screen_id="heartbeat",
        confidence=observation.overall_confidence,
        valid_for_control=False,
    )
    return GazeEstimationResult(sample, ())


def _driving_result(
    observation: VisionObservation | None,
    overlay_estimator: GazeEstimator | None,
    *,
    now_monotonic_ms: float,
) -> GazeEstimationResult:
    """The result fed to the controller: the real overlay prediction if one
    exists, else the model-free heartbeat above."""

    if observation is None:
        return GazeEstimationResult(None, (ReasonCode.LOW_CONFIDENCE,))
    if overlay_estimator is not None:
        return overlay_estimator.estimate(observation, now_monotonic_ms=now_monotonic_ms)
    return _heartbeat_result(observation)


def run_gaze_test(
    *,
    camera_index: int = 0,
    overlay_model_path: str | None = None,
    seed: int | None = None,
    point_count: int | None = None,
) -> int:
    """Run the held-out test-point screen on the primary display.

    ``overlay_model_path`` is optional and purely a display aid: when given,
    its live prediction is drawn next to each target so an operator can
    cross-check the recorded numbers against what they see. Collection does
    not require a trained model at all, the same way calibration doesn't.
    """

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
        print("No primary display is available for the held-out test.")
        return 1
    rectangle = screen.geometry()
    geometry = ScreenGeometry(
        screen_id=screen.name() or "primary",
        width_px=rectangle.width(),
        height_px=rectangle.height(),
        dpi_scale=float(screen.devicePixelRatio()),
    )
    effective_seed = DEFAULT_TEST_SEED if seed is None else seed
    effective_count = TEST_POINT_COUNT if point_count is None else point_count
    try:
        targets = generate_test_targets(geometry, seed=effective_seed, count=effective_count)
    except ContractValidationError as error:
        runtime.close()
        print(f"Could not place held-out test points: {error}")
        return 1
    overlay_estimator: GazeEstimator | None = None
    overlay_model_id: str | None = None
    if overlay_model_path is not None:
        try:
            overlay_model = load_overlay_model(Path(overlay_model_path))
        except (OSError, ValueError, ContractValidationError) as error:
            runtime.close()
            print(f"--overlay-model could not be loaded: {error}")
            return 1
        overlay_estimator = GazeEstimator(overlay_model, live_screen_geometry=geometry)
        overlay_model_id = overlay_model.model_id
    controller = LiveValidationController(geometry, targets=targets)
    window = _TestPointWindow(
        runtime,
        controller,
        camera_id=f"camera-{camera_index}",
        screen_geometry=geometry,
        seed=effective_seed,
        overlay_estimator=overlay_estimator,
        overlay_model_id=overlay_model_id,
    )
    window.show()
    try:
        return int(application.exec())
    finally:
        runtime.close()


class _TestPointWindow:  # pragma: no cover - requires display and live camera
    def __init__(
        self,
        runtime: VisionRuntime,
        controller: LiveValidationController,
        *,
        camera_id: str,
        screen_geometry: ScreenGeometry,
        seed: int,
        overlay_estimator: GazeEstimator | None,
        overlay_model_id: str | None,
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
        self._controller = controller
        self._camera_id = camera_id
        self._screen_geometry = screen_geometry
        self._seed = seed
        self._overlay_estimator = overlay_estimator
        self._overlay_model_id = overlay_model_id
        self._recorder = TestSampleRecorder()
        self._closed = False
        self._artifacts_written = False
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
        self._target.setAccessibleName("Held-out test target")

        self._prediction_overlay = (
            None if overlay_estimator is None else PredictionOverlay(self._widget)
        )

        self._progress = QLabel(self._widget)
        self._feedback = QLabel(self._widget)
        for label in (self._progress, self._feedback):
            label.setWordWrap(True)
            label.setStyleSheet("background: rgba(16, 24, 32, 210); padding: 8px;")
        controls = QHBoxLayout()
        restart = QPushButton("Restart (R)")
        cancel = QPushButton("Close safely (Esc)")
        restart.clicked.connect(self._on_restart)
        cancel.clicked.connect(self._on_cancel)
        controls.addWidget(restart)
        controls.addWidget(cancel)
        layout = QVBoxLayout(self._widget)
        layout.addWidget(self._progress)
        layout.addWidget(self._feedback)
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
        result = _driving_result(
            tick.accepted_observation, self._overlay_estimator, now_monotonic_ms=now_ms
        )
        view = self._controller.ingest(result, now_monotonic_ms=now_ms)
        self._recorder.observe(view, tick.accepted_observation)
        self._render(view)
        if self._prediction_overlay is not None:
            self._prediction_overlay.update(result if view.target is not None else None)
        if view.complete:
            self._write_dataset(view)
            self._timer.stop()

    def _on_key_press(self, event: Any) -> None:
        from PySide6.QtCore import Qt  # noqa: PLC0415

        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
        elif event.key() == Qt.Key.Key_R:
            self._on_restart()
        else:
            event.ignore()

    def _on_restart(self) -> None:
        self._recorder = TestSampleRecorder()
        self._artifacts_written = False
        self._timer.start(DEFAULT_TIMER_INTERVAL_MS)
        self._render(self._controller.restart())

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

    def _render(self, view: LiveValidationView) -> None:
        self._progress.setText(
            f"Test point {view.target_number} of {view.total_targets} — "
            f"{view.accepted_samples}/{view.required_samples} samples — {view.phase.value}"
        )
        self._feedback.setText(view.feedback)
        self._target.setVisible(view.target is not None)
        if view.target is not None:
            self._place(self._target, view.target.screen_position)
        elif self._prediction_overlay is not None:
            self._prediction_overlay.update(None)

    def _place(self, widget: Any, point: GazePoint) -> None:
        widget.adjustSize()
        widget.move(
            round(point.x * max(0, self._widget.width() - widget.width())),
            round(point.y * max(0, self._widget.height() - widget.height())),
        )
        widget.raise_()

    def _write_dataset(self, view: LiveValidationView) -> None:
        if self._artifacts_written:
            return
        self._artifacts_written = True
        result = TestPointResult(
            targets=tuple(measurement.target for measurement in view.measurements),
            sample_counts=self._recorder.sample_counts,
            samples=self._recorder.samples,
            camera_id=self._camera_id,
            screen_geometry=self._screen_geometry,
            seed=self._seed,
            overlay_model_id=self._overlay_model_id,
        )
        paths = write_test_dataset(result)
        print(f"Held-out test dataset written to: {paths.json_path}")
        print(f"Held-out test session log written to: {paths.text_path}")
        print("No OS input was emitted.")
