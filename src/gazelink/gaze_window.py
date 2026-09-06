"""Minimal M2 gaze diagnostic: raw/corrected, unfiltered, never OS control."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gazelink.calibration import targets_for_screen_geometry
from gazelink.camera import CameraError
from gazelink.correction_diagnostic import (
    CorrectionDiagnosticSession,
    CorrectionDiagnosticState,
    CorrectionDiagnosticView,
)
from gazelink.debug_window import DEFAULT_TIMER_INTERVAL_MS, build_runtime
from gazelink.domain import ContractValidationError, GazePoint, ScreenGeometry
from gazelink.eyegestures_engine import EXTERNAL_ENGINE_TIMER_INTERVAL_MS
from gazelink.gaze_correction import CorrectionStore
from gazelink.gaze_engine import (
    CalibrationStore,
    GazeEstimationResult,
    GazeEstimator,
    normalized_to_pixel,
)
from gazelink.gaze_filter import GazeStabilityFilter, RollingJitterMonitor
from gazelink.gaze_predictor import (
    ENGINE_CHOICES,
    EYEGESTURES_ENGINE,
    NATIVE_ENGINE,
    GazePredictor,
    NativeGazePredictor,
)
from gazelink.runtime import VisionRuntime
from gazelink.screen_mapping import centered_top_left

_WINDOW_TITLE = "GAZELINK — M2 raw gaze check"
_INFO_PANEL_WIDTH = 1200
_INFO_PANEL_HEIGHT = 174

# Qt is single-threaded: when the timer slot takes longer than the interval,
# paint events never get a turn and the window stops redrawing -- it looks
# frozen even though it is running. The native engine costs ~32 ms per tick
# against the default 16 ms interval, which already overruns; adding an
# external engine's own face mesh (~15 ms) pushes it to ~47 ms and the window
# visibly stops updating. Give that path an interval with real slack instead.


def clamp_panel_position(
    proposed_x: int,
    proposed_y: int,
    *,
    container_width: int,
    container_height: int,
    panel_width: int,
    panel_height: int,
) -> tuple[int, int]:
    """Keep a draggable diagnostic panel reachable inside its parent."""

    max_x = max(0, container_width - panel_width)
    max_y = max(0, container_height - panel_height)
    return min(max(0, proposed_x), max_x), min(max(0, proposed_y), max_y)


def format_gaze_status(
    result: GazeEstimationResult,
    *,
    raw_jitter_p95_px: float | None = None,
    filtered_jitter_p95_px: float | None = None,
) -> str:
    if result.sample is None:
        reasons = ", ".join(reason.value for reason in result.reason_codes)
        return f"NO VETTED GAZE SAMPLE — {reasons} — NOT FOR CONTROL"
    sample = result.sample
    jitter = ""
    if raw_jitter_p95_px is not None and filtered_jitter_p95_px is not None:
        jitter = (
            f" jitter_p95(raw={raw_jitter_p95_px:.0f}px, filtered={filtered_jitter_p95_px:.0f}px)"
        )
    return (
        "RAW / CORRECTED / FILTERED / NOT FOR CONTROL — "
        f"raw=({sample.raw_normalized.x:.3f}, {sample.raw_normalized.y:.3f}) "
        f"corrected=({sample.corrected_normalized.x:.3f}, "
        f"{sample.corrected_normalized.y:.3f}) "
        f"filtered=({sample.filtered_normalized.x:.3f}, "
        f"{sample.filtered_normalized.y:.3f}) "
        f"pixel=({sample.screen_position.x_px}, {sample.screen_position.y_px}){jitter}"
    )


def run_gaze_check(
    *, camera_index: int = 0, engine: str = NATIVE_ENGINE, smoothing: bool = True
) -> int:
    """Display live vetted predictions from the selected engine.

    ``engine="native"`` is the unchanged path: load this project's latest
    compatible model and show its predictions, with local-correction
    diagnostics available.  ``engine="eyegestures"`` instead routes prediction
    through the external library, which owns its own calibration and has no
    model of ours -- so the correction diagnostic is not offered there.

    ``smoothing=False`` removes the One-Euro stability filter from the display
    path. Smoothing hides exactly the behaviour a measurement is trying to
    characterise -- jitter and lag -- so it must be possible to see the engine
    without it. It changes only what this screen draws; nothing is recorded
    here either way.
    """

    if engine not in ENGINE_CHOICES:
        print(f"Unknown engine '{engine}'. Expected one of: {', '.join(ENGINE_CHOICES)}.")
        return 1

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
        print("No primary display is available for the gaze check.")
        return 1
    rectangle = screen.geometry()
    geometry = ScreenGeometry(
        screen_id=screen.name() or "primary",
        width_px=rectangle.width(),
        height_px=rectangle.height(),
        dpi_scale=float(screen.devicePixelRatio()),
    )

    predictor: GazePredictor
    diagnostic: CorrectionDiagnosticSession | None
    correction_store: CorrectionStore | None
    if engine == EYEGESTURES_ENGINE:
        # Say this BEFORE the slow part, not after. Importing EyeGestures pulls
        # in scikit-learn, which takes seconds on a warm cache and much longer
        # on a cold one -- and it happens after the camera is already open but
        # before any window exists, so the screen is blank the whole time. Left
        # unannounced it reads as a hang, and gets interrupted.
        print(
            "Engine: eyegestures (external, GPL-3.0). It runs its own calibration "
            "first; no GAZELINK model or correction is used on this path.",
            flush=True,
        )
        print("Loading EyeGestures (imports scikit-learn) -- please wait...", flush=True)

        from gazelink.eyegestures_engine import EyeGesturesGazePredictor  # noqa: PLC0415

        try:
            predictor = EyeGesturesGazePredictor(screen_geometry=geometry)
        except ContractValidationError as error:
            runtime.close()
            print(str(error))
            return 1
        diagnostic = None
        correction_store = None
        print("EyeGestures ready. Opening the window...", flush=True)
    else:
        model = CalibrationStore().load_latest_model(geometry)
        if model is None:
            runtime.close()
            print("No compatible gaze model. Run --guided-calibration first.")
            return 1
        estimator = GazeEstimator(model, live_screen_geometry=geometry)
        predictor = NativeGazePredictor(estimator)
        correction_store = CorrectionStore()
        diagnostic = CorrectionDiagnosticSession(
            estimator,
            correction_store.load(base_model_id=model.model_id, screen_geometry=geometry),
        )

    window = _GazeCheckWindow(
        runtime,
        predictor,
        diagnostic,
        correction_store,
        smoothing_enabled=smoothing,
        interval_ms=(
            EXTERNAL_ENGINE_TIMER_INTERVAL_MS
            if engine == EYEGESTURES_ENGINE
            else DEFAULT_TIMER_INTERVAL_MS
        ),
    )
    window.show()
    try:
        return int(application.exec())
    finally:
        predictor.close()
        runtime.close()


class _GazeCheckWindow:  # pragma: no cover - requires display and live camera
    def __init__(
        self,
        runtime: VisionRuntime,
        predictor: GazePredictor,
        diagnostic: CorrectionDiagnosticSession | None,
        correction_store: CorrectionStore | None,
        *,
        interval_ms: int = DEFAULT_TIMER_INTERVAL_MS,
        smoothing_enabled: bool = True,
    ) -> None:
        from PySide6.QtCore import Qt, QTimer  # noqa: PLC0415
        from PySide6.QtGui import QFont  # noqa: PLC0415
        from PySide6.QtWidgets import QLabel, QWidget  # noqa: PLC0415

        self._runtime = runtime
        self._predictor = predictor
        # None on the eyegestures path: local correction is defined against
        # our own model, which that engine does not have.
        self._diagnostic = diagnostic
        self._correction_store = correction_store
        self._correction_estimator: GazeEstimator | None = (
            predictor.estimator if isinstance(predictor, NativeGazePredictor) else None
        )
        self._closed = False
        self._last_now_ms: float | None = None
        # None means "draw what the engine produced". The filter is not merely
        # bypassed downstream -- it is never constructed, so no retained state
        # can leak into the drawn point.
        self._stability = GazeStabilityFilter() if smoothing_enabled else None
        self._jitter = RollingJitterMonitor(predictor.screen_geometry)
        self._correction_map_visible = False
        self._widget = QWidget()
        self._widget.setWindowTitle(_WINDOW_TITLE)
        self._widget.setStyleSheet("background: #101820; color: white;")
        self._widget.setWindowState(Qt.WindowState.WindowFullScreen)
        self._widget.closeEvent = self._on_close  # type: ignore[method-assign]
        self._widget.keyPressEvent = self._on_key_press  # type: ignore[method-assign]

        class _DraggableInfoPanel(QWidget):
            """Real Qt subclass so Windows reliably dispatches drag events."""

            def __init__(self, parent: Any) -> None:
                super().__init__(parent)
                self._drag_offset: Any | None = None

            def mousePressEvent(self, event: Any) -> None:  # noqa: N802
                if event.button() != Qt.MouseButton.LeftButton:
                    super().mousePressEvent(event)
                    return
                self._drag_offset = event.position().toPoint()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.raise_()
                event.accept()

            def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
                if self._drag_offset is None or not event.buttons() & Qt.MouseButton.LeftButton:
                    super().mouseMoveEvent(event)
                    return
                parent = self.parentWidget()
                if parent is None:
                    event.ignore()
                    return
                parent_position = parent.mapFromGlobal(event.globalPosition().toPoint())
                proposed = parent_position - self._drag_offset
                x, y = clamp_panel_position(
                    proposed.x(),
                    proposed.y(),
                    container_width=parent.width(),
                    container_height=parent.height(),
                    panel_width=self.width(),
                    panel_height=self.height(),
                )
                self.move(x, y)
                event.accept()

            def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
                if event.button() != Qt.MouseButton.LeftButton:
                    super().mouseReleaseEvent(event)
                    return
                self._drag_offset = None
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                event.accept()

        self._info_panel = _DraggableInfoPanel(self._widget)
        self._info_panel.setGeometry(16, 16, _INFO_PANEL_WIDTH, _INFO_PANEL_HEIGHT)
        self._info_panel.setStyleSheet("background: rgba(0, 0, 0, 180); border: 1px solid #607D8B;")
        self._info_panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._info_panel.setCursor(Qt.CursorShape.OpenHandCursor)
        self._info_panel.setAccessibleName("Draggable gaze diagnostic information panel")

        self._drag_handle = QLabel(
            "↕ Drag this panel anywhere · C = center panel", self._info_panel
        )
        self._drag_handle.setGeometry(0, 0, _INFO_PANEL_WIDTH, 30)
        self._drag_handle.setStyleSheet(
            "padding: 5px 12px; background: #263238; color: white; font-weight: bold;"
        )
        self._drag_handle.setAccessibleName("Drag handle for diagnostic information")
        self._drag_handle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._status = QLabel("Waiting for a vetted gaze sample…", self._info_panel)
        self._status.setWordWrap(True)
        self._status.setStyleSheet("padding: 12px; background: transparent; border: none;")
        self._status.setGeometry(0, 30, _INFO_PANEL_WIDTH, 70)
        self._status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._correction_status = QLabel(self._info_panel)
        self._correction_status.setWordWrap(True)
        self._correction_status.setStyleSheet(
            "padding: 12px; background: transparent; color: #FFD740; border: none;"
        )
        self._correction_status.setGeometry(0, 100, _INFO_PANEL_WIDTH, 72)
        self._correction_status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._raw_dot = QLabel("●", self._widget)
        self._corrected_dot = QLabel("○", self._widget)
        font = QFont()
        font.setPointSize(32)
        font.setBold(True)
        for dot, color, name in (
            (self._raw_dot, "#00E5FF", "Raw unfiltered gaze"),
            (self._corrected_dot, "#76FF03", "Filtered gaze"),
        ):
            dot.setFont(font)
            dot.setStyleSheet(f"color: {color}; background: transparent;")
            dot.setAccessibleName(name)
            dot.adjustSize()
            dot.hide()
        self._target_dot = QLabel("◎", self._widget)
        self._target_dot.setFont(font)
        self._target_dot.setStyleSheet("color: #FFD740; background: transparent;")
        self._target_dot.setAccessibleName("Known correction or validation target")
        self._target_dot.adjustSize()
        self._target_dot.hide()
        self._grid_labels: list[Any] = []
        for number in range(1, 10):
            label = QLabel(str(number), self._widget)
            label.setFont(font)
            label.setStyleSheet(
                "color: #FFD740; background: rgba(16, 24, 32, 210); "
                "border: 2px solid #FFD740; padding: 8px;"
            )
            label.setAccessibleName(f"Correction target {number}")
            label.adjustSize()
            label.hide()
            self._grid_labels.append(label)
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(interval_ms)
        if self._diagnostic is not None:
            self._render_diagnostic(self._diagnostic.view())
        else:
            self._correction_status.setText(
                f"Engine: {self._predictor.engine_name} (external). "
                "Local correction is unavailable on this engine. "
                "Follow its own calibration target first."
            )

    def show(self) -> None:
        """Show reliably on Windows, then center the movable info panel."""

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
        self._center_info_panel()

    def _center_info_panel(self) -> None:
        x, y = clamp_panel_position(
            (self._widget.width() - self._info_panel.width()) // 2,
            (self._widget.height() - self._info_panel.height()) // 2,
            container_width=self._widget.width(),
            container_height=self._widget.height(),
            panel_width=self._info_panel.width(),
            panel_height=self._info_panel.height(),
        )
        self._info_panel.move(x, y)
        self._info_panel.raise_()

    def _on_close(self, event: Any) -> None:
        self._shutdown()
        event.accept()

    def _on_key_press(self, event: Any) -> None:
        from PySide6.QtCore import Qt  # noqa: PLC0415

        if event.key() == Qt.Key.Key_Escape:
            self._shutdown()
            self._widget.close()
            return
        if event.key() == Qt.Key.Key_C:
            self._center_info_panel()
            return
        # Every remaining key drives local correction, which is defined against
        # our own model. On an external engine there is nothing to correct, so
        # those keys stay inert rather than half-working.
        if self._diagnostic is None or self._correction_estimator is None:
            event.ignore()
            return
        if event.key() == Qt.Key.Key_K:
            self._set_correction_map_visible(not self._correction_map_visible)
            return
        text = event.text()
        # ``"" in "123456789"`` is true in Python. Guard the length so a
        # non-text key cannot accidentally enter this branch and reach
        # ``int("")`` after the first live sample.
        if (
            len(text) == 1
            and text in "123456789"
            and self._last_now_ms is not None
            and self._correction_map_visible
        ):
            target = targets_for_screen_geometry(self._predictor.screen_geometry)[
                int(text) - 1
            ].screen_position
            self._set_correction_map_visible(False)
            self._render_diagnostic(
                self._diagnostic.start(
                    GazePoint(target.x, target.y), now_monotonic_ms=self._last_now_ms
                )
            )
            return
        if event.key() == Qt.Key.Key_A and self._diagnostic.view().can_accept:
            self._render_diagnostic(self._diagnostic.accept())
            self._persist_corrections()
            return
        if event.key() == Qt.Key.Key_X:
            self._render_diagnostic(self._diagnostic.reject())
            return
        if event.key() == Qt.Key.Key_U:
            self._render_diagnostic(self._diagnostic.undo())
            self._persist_corrections()
            return
        if event.key() == Qt.Key.Key_R:
            self._render_diagnostic(self._diagnostic.clear())
            self._persist_corrections()
            return
        if event.key() == Qt.Key.Key_D:
            engine = self._diagnostic.engine
            self._diagnostic = CorrectionDiagnosticSession(
                self._correction_estimator,
                engine.enable() if not engine.enabled else engine.disable(),
            )
            self._render_diagnostic(self._diagnostic.view())
            self._persist_corrections()
            return
        event.ignore()

    def _on_tick(self) -> None:
        try:
            tick = self._runtime.tick()
        except CameraError as error:
            self._shutdown()
            self._status.setText(f"{error.failure.message} {error.failure.recovery_action.value}")
            return
        except Exception:
            self._shutdown()
            raise
        if tick is None or tick.accepted_observation is None:
            if self._stability is not None:
                self._stability.reset()
            self._jitter.reset()
            self._raw_dot.hide()
            self._corrected_dot.hide()
            self._status.setText("NO VETTED GAZE SAMPLE — tracking withheld — NOT FOR CONTROL")
            # An external engine still gets this frame, and still shows its own
            # calibration target. Its calibration is what the user must look AT
            # to become trackable at all, and our policy accepts only ~18% of
            # frames -- starving the engine here stretched a calibration pass
            # to about five minutes, which reads as a hung screen. The point it
            # returns is still discarded; only calibration advances.
            if tick is not None and self._diagnostic is None:
                self._predictor.predict(
                    frame=tick.frame,
                    observation=tick.observation,
                    now_monotonic_ms=tick.frame.captured_at_monotonic_ms,
                )
            self._render_engine_calibration()
            return
        now_ms = tick.frame.captured_at_monotonic_ms + (tick.latency_ms or 0.0)
        self._last_now_ms = now_ms
        diagnostic_view: CorrectionDiagnosticView | None = None
        if self._diagnostic is not None:
            diagnostic_view = self._diagnostic.ingest(
                tick.accepted_observation, now_monotonic_ms=now_ms
            )
            self._render_diagnostic(diagnostic_view)
        result = self._predictor.predict(
            frame=tick.frame,
            observation=tick.accepted_observation,
            now_monotonic_ms=now_ms,
        )
        self._render_engine_calibration()
        if result.sample is not None:
            corrected = (
                result.sample
                if self._diagnostic is None
                else self._diagnostic.engine.apply_to_sample(result.sample)
            )
            filtered_point = (
                corrected.corrected_normalized
                if self._stability is None
                else self._stability.update(corrected.corrected_normalized, now_ms)
            )
            filtered = replace(
                corrected,
                filtered_normalized=filtered_point,
                screen_position=normalized_to_pixel(
                    filtered_point, self._predictor.screen_geometry
                ),
            )
            self._jitter.add(now_ms, filtered.raw_normalized, filtered.filtered_normalized)
            result = GazeEstimationResult(filtered, filtered.reason_codes)
        raw_metrics, filtered_metrics = self._jitter.metrics()
        self._status.setText(
            format_gaze_status(
                result,
                raw_jitter_p95_px=None if raw_metrics is None else raw_metrics.p95_radius_px,
                filtered_jitter_p95_px=(
                    None if filtered_metrics is None else filtered_metrics.p95_radius_px
                ),
            )
        )
        if result.sample is None:
            self._raw_dot.hide()
            self._corrected_dot.hide()
            return
        sample = result.sample
        if diagnostic_view is not None and diagnostic_view.state not in {
            CorrectionDiagnosticState.IDLE,
            CorrectionDiagnosticState.REVIEW,
        }:
            self._raw_dot.hide()
            self._corrected_dot.hide()
            return
        self._place(self._raw_dot, sample.raw_normalized.x, sample.raw_normalized.y)
        self._place(
            self._corrected_dot,
            sample.filtered_normalized.x,
            sample.filtered_normalized.y,
        )

    def _set_correction_map_visible(self, visible: bool) -> None:
        if self._diagnostic is None:
            return
        self._correction_map_visible = visible
        targets = targets_for_screen_geometry(self._predictor.screen_geometry)
        for label, target in zip(self._grid_labels, targets, strict=True):
            if visible:
                self._place(label, target.screen_position.x, target.screen_position.y)
            else:
                label.hide()
        self._render_diagnostic(self._diagnostic.view())

    def _place(self, dot: Any, normalized_x: float, normalized_y: float) -> None:
        """Centre ``dot`` on the scored pixel for this normalized point.

        Shared with the measured screen so a dot never lands somewhere the
        error metric would not agree with. See ``screen_mapping``.
        """

        x, y = centered_top_left(
            GazePoint(normalized_x, normalized_y),
            self._predictor.screen_geometry,
            glyph_width_px=dot.width(),
            glyph_height_px=dot.height(),
        )
        dot.move(x, y)
        dot.show()
        dot.raise_()
        self._info_panel.raise_()

    def _render_engine_calibration(self) -> None:
        """Draw an external engine's own calibration target, when it has one.

        The native engine calibrates in its own screen, so this is a no-op
        there and the target dot stays under the correction diagnostic's
        control.
        """

        view = getattr(self._predictor, "calibration_view", None)
        if view is None:
            return
        calibration = view()
        if not calibration.active:
            self._target_dot.hide()
            self._correction_status.setText(
                f"Engine: {self._predictor.engine_name} (external) — calibrated "
                f"({calibration.progress_text}). Local correction unavailable."
            )
            return
        if calibration.target_normalized is None:
            # Still calibrating, but the engine has not published a target yet
            # (it needs at least one processed frame). Say so plainly instead
            # of claiming calibration is finished.
            self._target_dot.hide()
            self._correction_status.setText(
                f"Engine: {self._predictor.engine_name} (external) — waiting for a "
                "stable face before its calibration can start. Sit facing the "
                "camera, centred and well lit."
            )
            return
        self._raw_dot.hide()
        self._corrected_dot.hide()
        self._target_dot.setStyleSheet("color: #FFD740; background: transparent;")
        self._place(
            self._target_dot,
            calibration.target_normalized.x,
            calibration.target_normalized.y,
        )
        self._correction_status.setText(
            f"Engine: {self._predictor.engine_name} (external) — follow the yellow "
            f"target: {calibration.progress_text}. No gaze point is reported until "
            "its calibration completes."
        )

    def _render_diagnostic(self, view: CorrectionDiagnosticView) -> None:
        if self._diagnostic is None:
            return
        details = view.instruction
        if view.before_error_normalized is not None and view.after_error_normalized is not None:
            details += (
                f" Before={view.before_error_normalized:.4f}; "
                f"After={view.after_error_normalized:.4f}."
            )
        details += (
            f" Corrections={len(self._diagnostic.engine.corrections)}; "
            f"enabled={self._diagnostic.engine.enabled}. "
            "K=show numbered target map; D=toggle."
        )
        self._correction_status.setText(details)
        if view.target is None:
            self._target_dot.hide()
            return
        color = (
            "#76FF03"
            if view.state
            in {
                CorrectionDiagnosticState.VALIDATION_STABILIZING,
                CorrectionDiagnosticState.VALIDATION_COLLECTING,
                CorrectionDiagnosticState.REVIEW,
            }
            else "#FFD740"
        )
        self._target_dot.setStyleSheet(f"color: {color}; background: transparent;")
        self._place(self._target_dot, view.target.x, view.target.y)

    def _persist_corrections(self) -> None:
        if self._diagnostic is None or self._correction_store is None:
            return
        try:
            self._correction_store.save(self._diagnostic.engine)
        except OSError as error:
            self._correction_status.setText(f"Correction persistence failed: {error}")

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._runtime.close()
