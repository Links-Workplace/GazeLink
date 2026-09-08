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

from gazelink.calibration import targets_for_screen_geometry
from gazelink.calibration_ui import CalibrationTimingSettings
from gazelink.camera import CameraError
from gazelink.debug_window import DEFAULT_TIMER_INTERVAL_MS, build_runtime
from gazelink.display_watch import (
    DisplayGuard,
    DisplayWatcher,
    OutputFreeze,
    describe_screen,
    ensure_high_dpi_policy,
    identify_screen,
)
from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    GazeSample,
    PixelPoint,
    ReasonCode,
    ScreenGeometry,
    VisionObservation,
)
from gazelink.eyegestures_engine import (
    EXTERNAL_ENGINE_TIMER_INTERVAL_MS,
    FULL_CALIBRATION_POINTS,
    EyeGesturesGazePredictor,
    calibration_grid_points,
    gate_would_accept,
)
from gazelink.eyegestures_run import (
    ArrivalTimer,
    MeasurementPhase,
    PhaseMachine,
    build_predictions_payload,
    model_stability,
    sample_from_gaze,
    write_predictions,
)
from gazelink.gaze_engine import GazeEstimationResult, GazeEstimator
from gazelink.gaze_predictor import EYEGESTURES_ENGINE, NATIVE_ENGINE
from gazelink.live_validation import LiveValidationController, LiveValidationView
from gazelink.prediction_overlay import PredictionOverlay, load_overlay_model
from gazelink.runtime import VisionRuntime
from gazelink.screen_mapping import centered_top_left
from gazelink.test_dataset import TestPointResult, TestSampleRecorder, write_test_dataset
from gazelink.test_points import DEFAULT_TEST_SEED, TEST_POINT_COUNT, generate_test_targets

_WINDOW_TITLE = "GAZELINK — held-out test points"


_TARGET_GLYPH = "●"
_TARGET_COLOR = "#FFD740"
# Deliberately a different shape as well as a different colour: colour alone is
# not enough to tell two states apart at a glance, or for a colour-blind reader.
_CALIBRATION_GLYPH = "◎"
_CALIBRATION_COLOR = "#40C4FF"


def _distance_px(predicted: GazePoint, target: GazePoint, geometry: ScreenGeometry) -> float:
    """The on-screen gap, on the SAME ruler the report uses.

    Shown live so the number beside the two markers is the number that will be
    scored, not a second opinion computed a different way.
    """

    dx = (predicted.x - target.x) * (geometry.width_px - 1)
    dy = (predicted.y - target.y) * (geometry.height_px - 1)
    return float((dx * dx + dy * dy) ** 0.5)


def _pose_tuple(
    observation: VisionObservation | None,
) -> tuple[float, float, float] | None:
    """Yaw, pitch and roll for one frame, or None when no pose was resolved."""

    if observation is None or observation.head_pose is None:
        return None
    pose = observation.head_pose
    return (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)


def _eyegestures_version() -> str:
    """The installed distribution version, not the library's own constant.

    ``eyeGestures.VERSION`` still reports 3.0.0 inside the 3.2.4 distribution,
    so reading it would stamp a wrong version onto every run file.
    """

    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("eyeGestures")
    except PackageNotFoundError:
        return "unknown"


# A fixed box, not `adjustSize()`. Font metrics make a label's box size
# font- and platform-dependent, and the glyph's ink is not necessarily centred
# inside it. Fixing the box and centring the text means the box centre IS the
# intended point by construction, which is what `screen_mapping` then scores.
TARGET_GLYPH_PX = 96


def overlay_result_for_tick(
    tick: Any,
    result: GazeEstimationResult | None,
    view: LiveValidationView | None,
) -> GazeEstimationResult | None:
    """What the prediction overlay should draw. ``None`` means hide it.

    Extracted from the window so the dropped-frame case is testable without
    Qt. A dropped tick used to return from ``_on_tick`` *before* the overlay
    was updated, leaving the previous frame's marker on screen -- a stale
    prediction presented as a live one, which is the one thing a measurement
    display must never do.
    """

    if tick is None or result is None or view is None or view.target is None:
        return None
    return result


def screen_matches_geometry(
    widget_width: int, widget_height: int, geometry: ScreenGeometry
) -> bool:
    """Is the drawn surface the same size as the ruler used to score it?

    If it is not, every recorded pixel error is measured against a screen the
    targets were not drawn on. Refusing to collect is better than recording
    numbers that cannot mean anything.
    """

    return widget_width == geometry.width_px and widget_height == geometry.height_px


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
    engine: str = NATIVE_ENGINE,
    own_vision: bool = True,
) -> int:
    """Run the held-out test-point screen on the primary display.

    ``overlay_model_path`` is optional and purely a display aid: when given,
    its live prediction is drawn next to each target so an operator can
    cross-check the recorded numbers against what they see. Collection does
    not require a trained model at all, the same way calibration doesn't.
    """

    runtime = build_runtime(camera_index=camera_index, own_vision=own_vision)
    try:
        runtime.start()
    except CameraError as error:
        runtime.close()
        print(f"{error.failure.message} {error.failure.recovery_action.value}")
        return 1

    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    ensure_high_dpi_policy()
    application = QApplication.instance() or QApplication([])
    screen = application.primaryScreen()  # type: ignore[attr-defined]
    if screen is None:
        runtime.close()
        print("No primary display is available for the held-out test.")
        return 1
    geometry = describe_screen(screen)
    effective_seed = DEFAULT_TEST_SEED if seed is None else seed
    effective_count = TEST_POINT_COUNT if point_count is None else point_count
    # An external engine calibrates on its OWN grid, so "held out" has to mean
    # held out from that grid, not from ours. Measured on the fixed seed, 4 of
    # the 10 default targets sit within 250px of an EyeGestures calibration
    # point; scoring those could not claim to measure generalisation.
    avoid_points: list[GazePoint] | None = None
    if engine == EYEGESTURES_ENGINE:
        avoid_points = [
            *calibration_grid_points(),
            *(
                GazePoint(target.screen_position.x, target.screen_position.y)
                for target in targets_for_screen_geometry(geometry)
            ),
        ]
    try:
        targets = generate_test_targets(
            geometry, seed=effective_seed, count=effective_count, avoid_points=avoid_points
        )
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
    engine_predictor: EyeGesturesGazePredictor | None = None
    timing = None
    if engine == EYEGESTURES_ENGINE:
        try:
            engine_predictor = EyeGesturesGazePredictor(
                screen_geometry=geometry,
                calibration_points=FULL_CALIBRATION_POINTS,
                # The gate is OUR policy, and this run measures THEIR engine.
                # Its verdict is recorded per sample instead of removing them.
                require_tracked_observation=False,
            )
        except (ImportError, RuntimeError) as error:
            runtime.close()
            print(f"EyeGestures could not be loaded: {error}")
            return 1
        # The library averages its last 20 predictions, so it needs about 1.3s
        # at this tick rate just to flush the previous target out of its own
        # buffer. 750ms would measure the tail of the target before this one.
        timing = CalibrationTimingSettings(stabilization_ms=1500.0, capture_window_ms=1500.0)
    controller = (
        LiveValidationController(geometry, targets=targets)
        if timing is None
        else LiveValidationController(geometry, targets=targets, timing=timing)
    )
    window = _TestPointWindow(
        runtime,
        controller,
        camera_id=f"camera-{camera_index}",
        screen_geometry=geometry,
        seed=effective_seed,
        overlay_estimator=overlay_estimator,
        overlay_model_id=overlay_model_id,
        screen=screen,
        engine=engine,
        engine_predictor=engine_predictor,
        own_vision=own_vision,
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
        screen: Any = None,
        engine: str = NATIVE_ENGINE,
        engine_predictor: EyeGesturesGazePredictor | None = None,
        own_vision: bool = True,
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
        # The display `screen_geometry` was read from. Kept so the window is
        # sized by the same QScreen the errors are scored against, instead of
        # whichever one Qt happens to associate with the widget later.
        self._screen = screen
        self._engine = engine
        self._own_vision = own_vision
        self._engine_predictor = engine_predictor
        self._phases = (
            None if engine_predictor is None else PhaseMachine(total_points=FULL_CALIBRATION_POINTS)
        )
        self._arrivals = None if engine_predictor is None else ArrivalTimer(screen_geometry)
        self._run_rows: list[Any] = []
        self._engine_calibration_flag = True
        # The model as it stood when the FIRST sample was recorded, compared
        # against its state at the last. A fit that landed while settling is
        # harmless -- settling only ends once no thread is running. A model
        # that moves between the first and last recorded sample is not, because
        # then the rows describe more than one model.
        self._model_at_first_sample: tuple[float, ...] | None = None
        self._model_readable = False
        self._seed = seed
        self._overlay_estimator = overlay_estimator
        self._overlay_model_id = overlay_model_id
        self._recorder = TestSampleRecorder()
        self._closed = False
        self._artifacts_written = False
        self._widget = QWidget()
        # Continuous, unlike `_verify_surface_matches_ruler` below, which
        # only fires once on entering fullscreen. A display that changes
        # mid-collection was invisible to that check.
        self._display = DisplayWatcher(
            DisplayGuard(screen_geometry, expected_identity=identify_screen(screen))
            if screen is not None
            else DisplayGuard(screen_geometry),
            lambda: self._widget.screen(),
            self._on_display_change,
        )
        self._widget.setWindowTitle(_WINDOW_TITLE)
        self._widget.setStyleSheet("background: #101820; color: white;")
        self._widget.setWindowState(Qt.WindowState.WindowFullScreen)
        self._widget.closeEvent = self._on_close  # type: ignore[method-assign]
        self._widget.keyPressEvent = self._on_key_press  # type: ignore[method-assign]

        self._target = QLabel(_TARGET_GLYPH, self._widget)
        font = QFont()
        font.setPointSize(44)
        font.setBold(True)
        self._target.setFont(font)
        self._target.setFixedSize(TARGET_GLYPH_PX, TARGET_GLYPH_PX)
        self._target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._target.setStyleSheet(f"color: {_TARGET_COLOR}; background: transparent;")
        self._target.setAccessibleName("Held-out test target")

        # The marker is needed whenever SOMETHING predicts: an overlay model on
        # the native path, or the external engine on its own. Tying it to
        # --overlay-model alone left the EyeGestures run with a target on screen
        # and no visible prediction beside it, so the gap being measured could
        # not be seen at all.
        self._prediction_overlay = (
            PredictionOverlay(self._widget, screen_geometry)
            if (overlay_estimator is not None or engine_predictor is not None)
            else None
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
        # Qt is single-threaded and this path runs TWO face meshes per frame
        # (ours and the library's). At the native 16ms the paint events never
        # get a slot and the window looks hung -- the same reason gaze_window
        # slows this engine down. Calibration is the phase where that matters
        # most, because it is also the longest.
        self._interval_ms = (
            EXTERNAL_ENGINE_TIMER_INTERVAL_MS
            if engine_predictor is not None
            else DEFAULT_TIMER_INTERVAL_MS
        )
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(self._interval_ms)
        self._render(self._controller.view())

        self._freeze = OutputFreeze(
            self._timer.stop,
            self._target.hide,
            lambda: (
                None if self._prediction_overlay is None else self._prediction_overlay.update(None)
            ),
        )

    def show(self) -> None:
        from PySide6.QtCore import QTimer  # noqa: PLC0415

        # The SAME QScreen the recorded geometry came from. Reading
        # `self._widget.screen()` here instead let the window be sized by one
        # display while every error was scored against another.
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
        self._verify_surface_matches_ruler()

    def _verify_surface_matches_ruler(self) -> None:
        """Stop before collecting anything if the surface is not the ruler.

        Every recorded error is a distance on ``self._screen_geometry``. If the
        window is not actually that size, the targets were drawn somewhere else
        and the numbers describe nothing. Recording that is worse than
        recording nothing, so this refuses rather than warns.
        """

        if screen_matches_geometry(
            self._widget.width(), self._widget.height(), self._screen_geometry
        ):
            return
        self._timer.stop()
        self._target.hide()
        if self._prediction_overlay is not None:
            self._prediction_overlay.update(None)
        self._feedback.setStyleSheet("color: #FF5252; background: transparent;")
        self._feedback.setText(
            "REFUSING TO COLLECT — the window is "
            f"{self._widget.width()}x{self._widget.height()} but errors would be "
            f"scored against {self._screen_geometry.width_px}x"
            f"{self._screen_geometry.height_px}. Nothing was recorded. Press Esc."
        )

    def _on_display_change(self, change: object) -> None:
        self._feedback.setStyleSheet("color: #FF5252; background: transparent;")
        self._feedback.setText(
            f"{getattr(change, 'message', 'The display changed.')} "
            "Nothing further was recorded. Press Esc."
        )

    def _refuse_for_display(self) -> None:
        """Stop collecting: the ruler changed underneath the measurements.

        Same reasoning as the surface check below -- numbers scored against a
        screen the targets were not drawn on describe nothing, and recording
        them is worse than recording nothing.
        """

        self._freeze.engage()

    def _on_tick(self) -> None:
        if self._display.poll() is not None:
            self._refuse_for_display()
            return
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
            # A dropped frame must clear the marker, never keep the last one.
            if self._prediction_overlay is not None:
                self._prediction_overlay.update(overlay_result_for_tick(None, None, None))
            return
        now_ms = tick.frame.captured_at_monotonic_ms + (tick.latency_ms or 0.0)
        if self._engine_predictor is not None:
            self._on_engine_tick(tick, now_ms)
            return
        result = _driving_result(
            tick.accepted_observation, self._overlay_estimator, now_monotonic_ms=now_ms
        )
        view = self._controller.ingest(result, now_monotonic_ms=now_ms)
        self._recorder.observe(view, tick.accepted_observation)
        self._render(view)
        if self._prediction_overlay is not None:
            self._prediction_overlay.update(overlay_result_for_tick(tick, result, view))
        if view.complete:
            self._write_dataset(view)
            self._timer.stop()

    def _on_engine_tick(self, tick: Any, now_ms: float) -> None:
        """One frame of an external-engine run: calibrate, freeze, then measure.

        The engine is fed on EVERY frame, including ones our policy rejects --
        it runs its own calibration from what it sees, and gating the feed
        starves it. Nothing is recorded until the phase machine says the
        library has actually stopped learning.
        """

        predictor = self._engine_predictor
        machine = self._phases
        if predictor is None or machine is None or self._arrivals is None:
            return

        observation = tick.observation
        result = predictor.predict(
            frame=tick.frame, observation=observation, now_monotonic_ms=now_ms
        )
        self._engine_calibration_flag = not predictor.is_calibrated and not predictor.is_frozen

        phase = machine.update(
            now_ms,
            is_calibrated=predictor.is_calibrated,
            completed_points=predictor.calibration_view().completed_points,
            pending_threads=predictor.pending_fit_threads(),
            fingerprint=predictor.calibration_fingerprint(),
            measurement_complete=self._controller.view().complete,
            on_freeze=predictor.freeze_calibration,
        )
        self._feedback.setStyleSheet(f"color: {phase.accent}; background: transparent;")
        self._feedback.setText(phase.banner)

        if not phase.recording:
            # Draw the LIBRARY's own calibration target while it is still
            # learning, and nothing at all while freezing, so an operator can
            # never mistake one phase for the other.
            calibrating = phase.phase is MeasurementPhase.ENGINE_CALIBRATING
            self._target.setVisible(calibrating)
            engine_target = predictor.calibration_view().target_normalized
            if calibrating and engine_target is not None:
                # A different glyph and colour from the test target: looking at
                # the engine's calibration point and looking at a measured
                # target are different instructions, and they must not be
                # confused from across the room.
                self._target.setText(_CALIBRATION_GLYPH)
                self._target.setStyleSheet(f"color: {_CALIBRATION_COLOR}; background: transparent;")
                self._place(self._target, engine_target)
            # `update(None)` would print "no tracking this frame", which is
            # false here: tracking is fine, the engine simply refuses to predict
            # until its own calibration finishes (it returns [0, 0] before it
            # is fitted, and that corner looks like a legitimate answer). Say
            # the true reason instead of a misleading one.
            if self._prediction_overlay is not None:
                self._prediction_overlay.hide_with_reason(
                    "no gaze prediction yet — EyeGestures does not predict until its "
                    "own calibration finishes"
                    if calibrating
                    else "prediction withheld while the engine's learning is being stopped"
                )
            self._progress.setText(
                f"{phase.banner}\n"
                f"Look at the CYAN target. Prediction appears once calibration completes."
                if calibrating
                else phase.banner
            )
            return

        # Back to the measured-target look for every phase after calibration.
        self._target.setText(_TARGET_GLYPH)
        self._target.setStyleSheet(f"color: {_TARGET_COLOR}; background: transparent;")

        view = self._controller.ingest(result, now_monotonic_ms=now_ms)
        target_index = view.target_number - 1
        if view.target is not None:
            self._arrivals.target_shown(target_index, now_ms)
            if result.sample is not None:
                self._arrivals.observe(
                    target_index,
                    result.sample.raw_normalized,
                    view.target.screen_position,
                    now_ms,
                )
            row = sample_from_gaze(
                target_index=target_index,
                sample=result.sample,
                phase=phase.phase,
                gate_accepts=gate_would_accept(tick.accepted_observation),
                engine_calibration_flag=self._engine_calibration_flag,
                geometry=self._screen_geometry,
                tracking_state=None if observation is None else observation.tracking_state,
                now_ms=now_ms,
                frame_id=tick.frame.frame_id,
                collection_phase=view.phase.value,
                fps=tick.fps,
                latency_ms=tick.latency_ms,
                head_pose=_pose_tuple(observation),
            )
            if row is not None:
                if not self._run_rows:
                    self._model_at_first_sample = predictor.calibration_fingerprint()
                    self._model_readable = self._model_at_first_sample is not None
                self._run_rows.append(row)
        self._render(view)
        if view.target is not None and result.sample is not None:
            gap_px = _distance_px(
                result.sample.raw_normalized,
                view.target.screen_position,
                self._screen_geometry,
            )
            self._progress.setText(f"{self._progress.text()} — gap now {gap_px:.0f}px")
        if self._prediction_overlay is not None:
            self._prediction_overlay.update(overlay_result_for_tick(tick, result, view))
        if view.complete:
            self._write_engine_run(view, machine)
            self._timer.stop()

    def _write_engine_run(self, view: LiveValidationView, machine: PhaseMachine) -> None:
        """Two files from the same rows: the library's output, and ours gated.

        Writing both is what keeps our confidence gate visible without letting
        it silently become the default filter on someone else's engine.
        """

        if self._artifacts_written:
            return
        self._artifacts_written = True
        targets = tuple(measurement.target for measurement in view.measurements)
        # The decisive check: did the model move while it was being measured?
        model_at_end = (
            self._engine_predictor.calibration_fingerprint()
            if self._engine_predictor is not None
            else None
        )
        model_stable, stability_detail = model_stability(
            self._model_at_first_sample if self._model_readable else None, model_at_end
        )
        metadata: dict[str, Any] = {
            "engine": self._engine,
            "library_version": _eyegestures_version(),
            "calibration_points_required": FULL_CALIBRATION_POINTS,
            "seed": self._seed,
            "timer_interval_ms": self._interval_ms,
            # Both must hold: nothing was training when measurement began, and
            # the model did not move while it ran.
            "freeze_verified": bool(machine.freeze_verified) and model_stable is True,
            "freeze_detail": f"{machine.freeze_detail}; {stability_detail}",
            "model_stable_during_measurement": model_stable,
            "camera_id": self._camera_id,
            # Without this a reader cannot tell an absent head pose from a head
            # pose that was measured and happened to be missing.
            "own_vision": self._own_vision,
            "samples_recorded": len(self._run_rows),
        }
        stamp = str(int(self._run_rows[0].timestamp_ms)) if self._run_rows else "0"
        directory = Path(".gazelink/test_points")
        base = f"predictions_eyegestures_{stamp}"
        for name, gated in ((base, False), (f"{base}_gated", True)):
            write_predictions(
                directory / f"{name}.json",
                build_predictions_payload(
                    targets=targets,
                    geometry=self._screen_geometry,
                    samples=self._run_rows,
                    timings=self._arrivals.results() if self._arrivals else (),
                    run_metadata=metadata,
                    gate_filtered=gated,
                    target_glyph_px=TARGET_GLYPH_PX,
                ),
            )
        self._feedback.setText(
            f"Wrote {len(self._run_rows)} samples to {directory}/{base}.json (+ _gated). "
            "Score with: analyze.py --predictions <file> --note ..."
        )

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
        self._timer.start(self._interval_ms)
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
        self._display.detach()
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
        """Centre ``widget`` on the pixel this point is SCORED at.

        Deliberately measured against ``self._screen_geometry`` -- the same
        ruler ``analyze.py`` uses -- rather than the live widget size, so the
        drawn position and the recorded error cannot describe two different
        surfaces. ``show()`` refuses to collect if the two ever disagree.
        """

        left, top = centered_top_left(
            point,
            self._screen_geometry,
            glyph_width_px=widget.width(),
            glyph_height_px=widget.height(),
        )
        widget.move(left, top)
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
