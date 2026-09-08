"""Full-screen, guided 9-point calibration surface for M2.

Qt is imported only inside runtime functions and methods.  The ordinary CLI
and deterministic tests therefore remain safe on hosts with no display server.
The window observes camera data and never emits cursor, click, keyboard, or
other operating-system input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gazelink.calibration import (
    CalibrationSession,
    CalibrationSessionResult,
    CalibrationSessionState,
    CalibrationTarget,
    targets_for_screen_geometry,
)
from gazelink.calibration_ui import (
    CALIBRATION_HEAD_POSE_LIMITS,
    CalibrationUiController,
    CalibrationUiView,
)
from gazelink.camera import CameraError
from gazelink.confidence import ConfidencePolicySettings
from gazelink.config import ConfidenceThresholds
from gazelink.debug_window import DEFAULT_TIMER_INTERVAL_MS, build_runtime
from gazelink.display_watch import (
    DisplayGuard,
    DisplayWatcher,
    OutputFreeze,
    describe_screen,
    ensure_high_dpi_policy,
    identify_screen,
    pick_screen,
)
from gazelink.domain import ContractValidationError, NormalizedPoint, ScreenGeometry
from gazelink.gaze_engine import (
    CalibrationEngine,
    CalibrationStore,
    CalibrationTrainingResult,
    GazeEstimator,
    PendingCalibration,
)
from gazelink.overlay import CircleCommand, OverlayCommand, RectangleCommand, TextCommand
from gazelink.prediction_overlay import PredictionOverlay, load_overlay_model
from gazelink.runtime import VisionRuntime

_WINDOW_TITLE = "GAZELINK — Guided calibration"

# Written only when the user explicitly asked for a session log to review;
# never a raw frame, landmark, or image -- only the same kind of aggregate
# diagnostic (target index, confidence, tracking state, feedback text)
# already accepted for the M1 terminal diagnostics this session.
CALIBRATION_LOG_DIR = Path(".gazelink") / "calibration_logs"


@dataclass(frozen=True, slots=True)
class PersistedCalibration:
    dataset_path: Path
    pending_model: PendingCalibration | None
    training: CalibrationTrainingResult


def train_and_persist_calibration(
    result: CalibrationSessionResult,
    *,
    engine: CalibrationEngine | None = None,
    store: CalibrationStore | None = None,
) -> PersistedCalibration:
    """Persist/train a candidate; a separate live review may publish it.

    Offline leave-one-target-out metrics are necessary evidence, but the
    user's current camera posture can still make an apparently better model
    visibly wrong.  Never replace ``latest_model.json`` at this point.
    """

    calibration_store = store or CalibrationStore()
    dataset_path = calibration_store.save_dataset(result)
    training = (engine or CalibrationEngine()).train(result)
    if training.promotable:
        return PersistedCalibration(
            dataset_path,
            calibration_store.save_pending_models(
                training.candidate_models, selected_model_id=training.model.model_id
            ),
            training,
        )
    # A failed new attempt must not erase a previously validated model. The
    # candidate is kept only as its lossless dataset; latest_model.json stays
    # exactly as it was before this attempt.
    return PersistedCalibration(dataset_path, None, training)


def format_training_summary(training: CalibrationTrainingResult) -> str:
    comparison = training.comparison
    selected_metrics = (
        comparison.baseline if comparison.selected_kind.value == "LINEAR" else comparison.advanced
    )
    return (
        f"selected={comparison.selected_kind.value} "
        f"promotable={training.promotable} "
        f"center_median={comparison.center_baseline.median_error_px:.1f}px "
        f"center_p95={comparison.center_baseline.p95_error_px:.1f}px "
        f"baseline_median={comparison.baseline.median_error_px:.1f}px "
        f"baseline_p95={comparison.baseline.p95_error_px:.1f}px "
        f"advanced_median={comparison.advanced.median_error_px:.1f}px "
        f"advanced_p95={comparison.advanced.p95_error_px:.1f}px "
        f"linear_x=(median={comparison.baseline.median_abs_error_x_normalized:.3f},"
        f"p95={comparison.baseline.p95_abs_error_x_normalized:.3f}) "
        f"linear_y=(median={comparison.baseline.median_abs_error_y_normalized:.3f},"
        f"p95={comparison.baseline.p95_abs_error_y_normalized:.3f}) "
        f"advanced_x=(median={comparison.advanced.median_abs_error_x_normalized:.3f},"
        f"p95={comparison.advanced.p95_abs_error_x_normalized:.3f}) "
        f"advanced_y=(median={comparison.advanced.median_abs_error_y_normalized:.3f},"
        f"p95={comparison.advanced.p95_abs_error_y_normalized:.3f}) "
        f"latency={selected_metrics.prediction_latency_ms:.4f}ms"
    )


@dataclass(frozen=True, slots=True)
class CalibrationLogEvent:
    """One ingest outcome; contains no frame, landmark, or image data."""

    target_number: int
    accepted: bool
    tracking_state: str
    confidence: float
    feedback: str


def _format_point(point: NormalizedPoint | None) -> str:
    return "unavailable" if point is None else f"({point.x:.3f}, {point.y:.3f})"


def _format_ratio(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _format_pose(yaw: float | None, pitch: float | None, roll: float | None) -> str:
    if yaw is None or pitch is None or roll is None:
        return "unavailable"
    return f"(yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f})"


def format_calibration_log(
    *,
    camera_id: str,
    screen_geometry: ScreenGeometry,
    min_samples_per_target: int,
    outcome: str,
    events: Sequence[CalibrationLogEvent],
    result: CalibrationSessionResult | None,
    generated_at: datetime,
) -> str:
    """Render a human-readable session log; pure, so it needs no Qt or camera.

    ``generated_at`` is wall-clock and shown for human orientation only, per
    the project's "wall-clock is for human/log metadata, monotonic drives
    logic" convention -- it plays no role in any calibration decision.

    When ``result`` is present, a ``--- Metrics ---`` section sits between
    ``--- Final result ---`` and ``--- Dataset ---``. The per-sample dataset
    dump below it already contains every reason and every head-pose reading,
    but a person checking whether a real session went well should not have to
    tally a hand-count across dozens of dataset lines to answer "why were
    samples rejected, and how much did the head move?". This section answers
    both at a glance: ``result.reason_counts()`` sorted by reason name (so the
    same session always renders identically, whatever order samples were
    ingested in) and ``result.head_pose_spread()``'s six degree values. Either
    metric renders an explicit "none"/"unavailable" line instead of a
    fabricated ``0`` when nothing qualifies -- a `0.0` would misreport a
    session with no head-pose data as one where the head never moved.
    """

    accepted_count = sum(1 for event in events if event.accepted)
    lines = [
        "GAZELINK -- Guided calibration session log",
        f"Generated: {generated_at.astimezone(UTC).isoformat()}",
        f"Camera: {camera_id}",
        f"Screen: {screen_geometry.screen_id} {screen_geometry.width_px}x"
        f"{screen_geometry.height_px} @ {screen_geometry.dpi_scale:.2f}x DPI",
        f"Samples required per target: {min_samples_per_target}",
        f"Outcome: {outcome}",
        "",
        f"--- Events ({len(events)} total, {accepted_count} accepted, "
        f"{len(events) - accepted_count} rejected) ---",
    ]
    for event in events:
        lines.append(
            f"target={event.target_number} accepted={event.accepted} "
            f"tracking={event.tracking_state} confidence={event.confidence:.2f} "
            f'feedback="{event.feedback}"'
        )
    lines.append("")
    if result is not None:
        lines.append("--- Final result ---")
        lines.append(f"sample_counts: {result.sample_counts}")
        lines.append(f"target_order: {result.target_order}")
        lines.append(f"feature_schema_version: {result.feature_schema_version}")
        lines.append(f"started_at_monotonic_ms: {result.started_at_monotonic_ms}")
        lines.append(f"completed_at_monotonic_ms: {result.completed_at_monotonic_ms}")
        lines.append("")
        lines.append("--- Metrics ---")
        reason_counts = result.reason_counts()
        if reason_counts:
            for reason in sorted(reason_counts):
                lines.append(f"reason_count[{reason}]: {reason_counts[reason]}")
        else:
            lines.append("reason_count: none (no samples recorded)")
        pose_spread = result.head_pose_spread()
        if pose_spread:
            lines.append(f"median_yaw_deg: {pose_spread['median_yaw_deg']:.1f}")
            lines.append(f"median_pitch_deg: {pose_spread['median_pitch_deg']:.1f}")
            lines.append(f"median_roll_deg: {pose_spread['median_roll_deg']:.1f}")
            lines.append(f"max_yaw_deviation_deg: {pose_spread['max_yaw_deviation_deg']:.1f}")
            lines.append(f"max_pitch_deviation_deg: {pose_spread['max_pitch_deviation_deg']:.1f}")
            lines.append(f"max_roll_deviation_deg: {pose_spread['max_roll_deviation_deg']:.1f}")
        else:
            lines.append("head_pose_spread: unavailable (no accepted samples with head pose)")
        lines.append("")
        accepted_samples = sum(1 for sample in result.samples if sample.accepted)
        lines.append(
            f"--- Dataset ({len(result.samples)} samples, "
            f"{accepted_samples} accepted, {len(result.samples) - accepted_samples} rejected) ---"
        )
        for sample in result.samples:
            left = _format_point(sample.left_iris_in_eye)
            right = _format_point(sample.right_iris_in_eye)
            pose = _format_pose(sample.head_yaw_deg, sample.head_pitch_deg, sample.head_roll_deg)
            lines.append(
                f"target={sample.target_index} accepted={sample.accepted} "
                f"reason={sample.reason} confidence={sample.confidence:.2f} "
                f"left_iris_in_eye={left} right_iris_in_eye={right} "
                f"left_openness={_format_ratio(sample.left_openness)} "
                f"right_openness={_format_ratio(sample.right_openness)} "
                f"head_pose={pose}"
            )
    else:
        lines.append("--- No final result (session did not reach COMPLETE) ---")
    return "\n".join(lines) + "\n"


def write_calibration_log(content: str, *, directory: Path = CALIBRATION_LOG_DIR) -> Path:
    """Write ``content`` to a fresh timestamped file and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"calibration_session_{stamp}.txt"
    path.write_text(content, encoding="utf-8")
    return path


def run_guided_calibration(
    *,
    camera_index: int = 0,
    target_order: Sequence[int] | None = None,
    overlay_model_path: str | None = None,
    screen_name: str | None = None,
) -> int:
    """Run calibration on the primary display; open camera before Qt on Windows.

    ``overlay_model_path``, when given, is loaded once and used only to draw
    a live prediction marker next to the calibration target for visual
    sanity-checking.  It is never trained, saved, or promoted -- purely a
    display concern layered on top of the unmodified calibration flow.
    """

    runtime = build_runtime(
        camera_index=camera_index,
        confidence_settings=ConfidencePolicySettings(
            thresholds=ConfidenceThresholds(),
            max_sample_age_ms=500.0,
            head_pose_limits=CALIBRATION_HEAD_POSE_LIMITS,
        ),
    )
    try:
        runtime.start()
    except CameraError as error:
        runtime.close()
        print(f"{error.failure.message} {error.failure.recovery_action.value}")
        return 1

    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    ensure_high_dpi_policy()
    application = QApplication.instance() or QApplication([])
    screen = pick_screen(application, screen_name)
    if screen is None:
        runtime.close()
        print("No primary display is available for calibration.")
        return 1
    camera_id = f"camera-{camera_index}"
    screen_geometry = describe_screen(screen)
    display_guard = DisplayGuard(screen_geometry, expected_identity=identify_screen(screen))
    overlay_estimator: GazeEstimator | None = None
    if overlay_model_path is not None:
        try:
            overlay_model = load_overlay_model(Path(overlay_model_path))
        except (OSError, ValueError, ContractValidationError) as error:
            runtime.close()
            print(f"--overlay-model could not be loaded: {error}")
            return 1
        overlay_estimator = GazeEstimator(overlay_model, live_screen_geometry=screen_geometry)
    session = CalibrationSession(
        camera_id=camera_id,
        screen_geometry=screen_geometry,
        targets=targets_for_screen_geometry(screen_geometry),
        target_order=target_order,
    )
    window = _CalibrationWindow(
        display_guard,
        screen,
        runtime,
        CalibrationUiController(session),
        camera_id=camera_id,
        screen_geometry=screen_geometry,
        overlay_estimator=overlay_estimator,
    )
    window.show()
    try:
        return int(application.exec())
    finally:
        runtime.close()


class _CalibrationWindow:  # pragma: no cover - requires a display server and live camera
    """Thin Qt projection of the controller; no calibration decisions live here."""

    def __init__(
        self,
        display_guard: DisplayGuard,
        screen: Any,
        runtime: VisionRuntime,
        controller: CalibrationUiController,
        *,
        camera_id: str,
        screen_geometry: ScreenGeometry,
        overlay_estimator: GazeEstimator | None = None,
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
        self._display_guard = display_guard
        # The SAME QScreen the geometry came from. Reading
        # `self._widget.screen()` in show() instead sized the window by the
        # default display while every target was positioned for another --
        # the exact mismatch test_window already documents.
        self._screen = screen
        self._controller = controller
        self._camera_id = camera_id
        self._screen_geometry = screen_geometry
        self._log_events: list[CalibrationLogEvent] = []
        self._closed = False
        self._artifacts_written = False
        self._overlay_estimator = overlay_estimator
        self._widget = QWidget()
        self._display = DisplayWatcher(display_guard, self._widget.screen, self._on_display_change)
        self._widget.setWindowTitle(_WINDOW_TITLE)
        self._widget.setStyleSheet("background: #101820; color: white;")
        self._widget.setWindowState(Qt.WindowState.WindowFullScreen)
        self._widget.closeEvent = self._on_close  # type: ignore[method-assign]
        self._widget.keyPressEvent = self._on_key_press  # type: ignore[method-assign]

        # The camera view fills the calibration screen but remains transient:
        # no image is retained outside the current paint call or written to
        # disk.  It is lowered below all guidance so no target can be hidden.
        self._preview = QLabel(self._widget)
        self._preview.setStyleSheet("background: #101820;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setAccessibleName("Live camera preview with tracking overlay")
        self._preview.lower()

        self._target = QLabel("●", self._widget)
        target_font = QFont()
        target_font.setPointSize(44)
        target_font.setBold(True)
        self._target.setFont(target_font)
        self._target.setStyleSheet("color: #00E5FF; background: transparent;")
        self._target.setAccessibleName("Calibration target")

        self._prediction_overlay = (
            None if overlay_estimator is None else PredictionOverlay(self._widget)
        )

        self._progress = QLabel(self._widget)
        self._feedback = QLabel(self._widget)
        self._progress.setStyleSheet(
            "background: rgba(16, 24, 32, 210); color: white; padding: 8px; font-weight: bold;"
        )
        self._feedback.setStyleSheet(
            "background: rgba(16, 24, 32, 210); color: white; padding: 8px;"
        )
        self._feedback.setWordWrap(True)
        self._progress.setAccessibleName("Calibration progress")
        self._feedback.setAccessibleName("Calibration feedback")
        controls = QHBoxLayout()
        retry = QPushButton("Retry target (R)")
        restart = QPushButton("Restart all (Ctrl+R)")
        cancel = QPushButton("Cancel (Esc)")
        retry.clicked.connect(self._on_retry)
        restart.clicked.connect(self._on_restart)
        cancel.clicked.connect(self._on_cancel)
        controls.addWidget(retry)
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

        self._freeze = OutputFreeze(
            self._timer.stop,
            self._target.hide,
            lambda: (
                None if self._prediction_overlay is None else self._prediction_overlay.update(None)
            ),
        )

    def show(self) -> None:
        """Show visibly even if Windows initially declines a fullscreen request.

        Qt created a 324x100 window on the reference Windows desktop when
        fullscreen was requested before the event loop had processed its first
        native show event.  Start with a regular visible window, size it to
        the assigned screen, then request fullscreen on the next event-loop
        turn.  The explicit geometry is also a safe readable fallback if a
        window manager declines fullscreen again.
        """

        from PySide6.QtCore import QTimer  # noqa: PLC0415

        if self._screen is not None:
            self._widget.setGeometry(self._screen.geometry())
        self._widget.show()
        self._widget.raise_()
        self._widget.activateWindow()
        QTimer.singleShot(0, self._show_full_screen)

    def _show_full_screen(self) -> None:
        self._widget.showFullScreen()
        self._widget.raise_()
        self._widget.activateWindow()

    def _on_close(self, event: Any) -> None:
        self._shutdown()
        event.accept()

    def _on_key_press(self, event: Any) -> None:
        from PySide6.QtCore import Qt  # noqa: PLC0415

        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
        elif (
            event.key() == Qt.Key.Key_R and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_restart()
        elif event.key() == Qt.Key.Key_R:
            self._on_retry()
        else:
            event.ignore()

    def _on_tick(self) -> None:
        # The ruler these measurements are scored against must still exist.
        # Recording numbers for a screen that has changed is worse than
        # recording nothing, so this aborts rather than warns.
        if self._display.poll() is not None:
            self._freeze_for_display()
            return

        if self._controller.session.state is not CalibrationSessionState.COLLECTING:
            # The session finished (complete, cancelled, or failed): stop
            # driving the camera and MediaPipe pipeline. A live session was
            # left open for minutes here in manual testing, still capturing
            # and running inference on every tick for no purpose once nothing
            # was left to collect. The camera stays acquired (Restart must be
            # able to resume without reopening it), only new capture/inference
            # work stops; `_on_restart` restarts this timer.
            self._timer.stop()
            return
        try:
            tick = self._runtime.tick()
        except CameraError as error:
            self._shutdown()
            self._feedback.setText(f"{error.failure.message} {error.failure.recovery_action.value}")
            return
        except Exception:
            self._shutdown()
            raise
        if tick is not None:
            self._paint_preview(
                tick.frame.width, tick.frame.height, tick.frame.image, tick.view.commands
            )
            # Captured before ingest(): a sample that completes a target
            # advances the session, so the *post*-ingest view can already
            # point at the next target -- the log must attribute the outcome
            # to the target that was actually current when this observation
            # arrived, not whatever target follows it.
            target_number = self._controller.view().target_number
            view = self._controller.ingest(tick.observation)
            if view.sample_evaluated:
                self._log_events.append(
                    CalibrationLogEvent(
                        target_number=target_number,
                        accepted=view.sample_accepted,
                        tracking_state=tick.observation.tracking_state.value,
                        confidence=tick.observation.overall_confidence,
                        feedback=view.feedback,
                    )
                )
            self._render(view)
            if self._prediction_overlay is not None and view.target is not None:
                self._update_prediction_overlay(tick)
            if view.complete:
                # Publish the lossless dataset/model immediately when the
                # final accepted sample completes calibration. The user need
                # not discover that closing an already-complete window was a
                # hidden prerequisite for --gaze-check.
                self._timer.stop()
                self._write_log()

    def _update_prediction_overlay(self, tick: Any) -> None:
        """Draw where ``--overlay-model`` currently predicts, or why it can't.

        Uses ``tick.accepted_observation`` -- the same vetted observation
        ``--gaze-check`` and ``--gaze-validation`` estimate from -- not the
        unvetted ``tick.observation`` the calibration controller ingests.
        This keeps the overlay's own tracking/confidence gating identical to
        every other live-prediction surface in the product.
        """

        assert self._prediction_overlay is not None
        assert self._overlay_estimator is not None
        if tick.accepted_observation is None:
            self._prediction_overlay.update(None)
            return
        now_ms = tick.frame.captured_at_monotonic_ms + (tick.latency_ms or 0.0)
        result = self._overlay_estimator.estimate(
            tick.accepted_observation, now_monotonic_ms=now_ms
        )
        self._prediction_overlay.update(result)

    def _on_retry(self) -> None:
        if self._freeze.engaged:
            # Restarting cannot help: the calibration would be collected
            # against a display this session can no longer describe. Say so
            # again rather than re-arming a timer that immediately stops.
            change = self._display.guard.tripped
            self._feedback.setText(
                f"{change.message if change is not None else 'The display changed.'} "
                "Restart GAZELINK to calibrate for this screen."
            )
            return
        # `CalibrationSession.retry_current_target()` raises once the session
        # has left COLLECTING (there is no "current target" to retry). The
        # Retry button and its "R" shortcut have no state-based disabling, so
        # without this guard, pressing R after "Calibration complete." --
        # entirely plausible, and observed leaving the window open afterward
        # in manual testing -- raised an unhandled ContractValidationError.
        if self._controller.session.state is CalibrationSessionState.COLLECTING:
            self._render(self._controller.retry())

    def _on_restart(self) -> None:
        if self._freeze.engaged:
            # Restarting cannot help: the calibration would be collected
            # against a display this session can no longer describe. Say so
            # again rather than re-arming a timer that immediately stops.
            change = self._display.guard.tripped
            self._feedback.setText(
                f"{change.message if change is not None else 'The display changed.'} "
                "Restart GAZELINK to calibrate for this screen."
            )
            return
        # A restart is a new calibration run, not an extension of the
        # completed run. Give it its own log, dataset, calibration identity,
        # and persisted model when it completes.
        self._log_events.clear()
        self._artifacts_written = False
        self._render(self._controller.restart())
        # restart() always returns the session to COLLECTING, even from a
        # terminal state -- resume ticking if `_on_tick` had stopped it.
        self._timer.start(DEFAULT_TIMER_INTERVAL_MS)

    def _on_cancel(self) -> None:
        if not self._controller.view().complete and not self._controller.view().cancelled:
            self._render(self._controller.cancel())
        self._shutdown()
        self._widget.close()

    def _render(self, view: CalibrationUiView) -> None:
        self._progress.setText(
            f"Target {view.target_number} of {view.total_targets} — "
            f"{view.accepted_samples}/{view.required_samples} quality samples"
        )
        self._feedback.setText(view.feedback)
        self._target.setVisible(view.target is not None)
        if view.target is not None:
            self._place_target(view.target)
        elif self._prediction_overlay is not None:
            # No active target (session complete/cancelled/restarted): a
            # stale prediction mark here would be as misleading as a stale
            # target, so it is cleared through the same code path that
            # decides "no tracking this tick".
            self._prediction_overlay.update(None)

    def _paint_preview(
        self,
        width: int,
        height: int,
        image_bytes: bytes | None,
        commands: Sequence[OverlayCommand],
    ) -> None:
        """Paint this frame's camera image and M1 overlay behind calibration.

        The QImage is detached from the frame buffer before drawing, matching
        the M1 debug window's ownership rule.  The resulting pixmap is used
        only as the current QLabel image and is replaced on the next frame.
        """

        from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: PLC0415
        from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap  # noqa: PLC0415

        if image_bytes is None:
            return
        self._preview.setGeometry(self._widget.rect())
        image = QImage(image_bytes, width, height, width * 3, QImage.Format.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for command in commands:
                pen = QPen(QColor(command.color))
                pen.setWidth(2)
                painter.setPen(pen)
                if isinstance(command, RectangleCommand):
                    painter.drawRect(
                        QRectF(command.x_px, command.y_px, command.width_px, command.height_px)
                    )
                elif isinstance(command, CircleCommand):
                    painter.drawEllipse(
                        QPointF(command.center_x_px, command.center_y_px),
                        command.radius_px,
                        command.radius_px,
                    )
                elif isinstance(command, TextCommand):
                    painter.drawText(QPointF(command.x_px, command.y_px), command.text)
        finally:
            painter.end()
        self._preview.setPixmap(
            pixmap.scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._preview.lower()

    def _place_target(self, target: CalibrationTarget) -> None:
        self._target.adjustSize()
        x = round(target.screen_position.x * max(0, self._widget.width() - self._target.width()))
        y = round(target.screen_position.y * max(0, self._widget.height() - self._target.height()))
        self._target.move(x, y)
        self._target.raise_()

    def _on_display_change(self, change: object) -> None:
        self._feedback.setText(getattr(change, "message", "The display changed."))

    def _freeze_for_display(self) -> None:
        """Abandon this session: its numbers describe a screen that is gone.

        Unlike the live gaze screen, there is nothing to resume here -- every
        sample already collected was scored against the old geometry, so the
        honest outcome is to stop and say so rather than to blend two rulers
        into one dataset.
        """

        self._freeze.engage()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._display.detach()
        self._timer.stop()
        self._runtime.close()
        self._write_log()

    def _write_log(self) -> None:
        """Write each run once: at completion, or at shutdown if incomplete.

        Writing per-event would mean a file grows and is rewritten on every
        tick. A cancelled/failed session's partial event history is still
        useful for diagnosing what went wrong, while a completed run must
        publish its dataset/model without making window closure a prerequisite.
        """

        if self._artifacts_written:
            return
        state = self._controller.session.state
        result = self._controller.result() if state is CalibrationSessionState.COMPLETE else None
        content = format_calibration_log(
            camera_id=self._camera_id,
            screen_geometry=self._screen_geometry,
            min_samples_per_target=self._controller.session.min_samples_per_target,
            outcome=state.value,
            events=self._log_events,
            result=result,
            generated_at=datetime.now(UTC),
        )
        path = write_calibration_log(content)
        self._artifacts_written = True
        print(f"Calibration session log written to: {path}")
        if result is not None:
            try:
                persisted = train_and_persist_calibration(result)
            except (OSError, ValueError) as error:
                # The human-readable log above still survives. A failed model
                # must never be published as latest_model.json or turn window
                # shutdown into a leaked-camera path.
                print(f"Calibration model was not persisted: {error}")
            else:
                print(f"Calibration dataset written to: {persisted.dataset_path}")
                print(f"Gaze model benchmark: {format_training_summary(persisted.training)}")
                if persisted.pending_model is None:
                    reasons = ", ".join(persisted.training.quality_reasons)
                    print(f"Calibration rejected; no candidate available: {reasons}")
                else:
                    print(
                        "Calibration candidate written to: "
                        f"{persisted.pending_model.candidate_path}"
                    )
                    print(
                        "Run --gaze-validation to review fresh live measurements; "
                        "the candidate is not active until accepted."
                    )
