"""Full-screen five-direction gaze-feature diagnostic; never trains or controls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gazelink.calibration_ui import CALIBRATION_HEAD_POSE_LIMITS, frame_monotonic_ms
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
)
from gazelink.domain import JSONValue, ScreenGeometry
from gazelink.feature_check import (
    FeatureCheckController,
    FeatureCheckPhase,
    FeatureCheckResult,
    FeatureCheckView,
)
from gazelink.runtime import VisionRuntime

FEATURE_CHECK_DIR = Path(".gazelink") / "feature_checks"
_WINDOW_TITLE = "GAZELINK — Gaze feature check"


@dataclass(frozen=True, slots=True)
class FeatureCheckArtifacts:
    json_path: Path
    text_path: Path


def format_feature_check_report(
    result: FeatureCheckResult,
    *,
    camera_id: str,
    screen_geometry: ScreenGeometry,
    generated_at: datetime,
) -> str:
    lines = [
        "GAZELINK -- Five-direction gaze feature check",
        f"Generated: {generated_at.astimezone(UTC).isoformat()}",
        f"Camera: {camera_id}",
        (
            f"Screen: {screen_geometry.screen_id} "
            f"{screen_geometry.width_px}x{screen_geometry.height_px}"
        ),
        "No gaze model was trained and no OS input was emitted.",
        f"Overall diagnostic pass: {result.overall_pass}",
        (
            "Thresholds: "
            f"horizontal_delta={result.settings.min_horizontal_delta:.5f} "
            f"vertical_combined_delta={result.settings.min_vertical_combined_delta:.5f} "
            f"vertical_per_eye_delta={result.settings.min_vertical_per_eye_delta:.5f} "
            f"stationary_p95={result.settings.max_stationary_p95:.5f} "
            f"head_pose_p95_deg={result.settings.max_head_pose_p95_deg:.5f}"
        ),
        "",
        "--- Direction metrics ---",
    ]
    for metric in result.metrics:
        lines.append(
            f"{metric.direction.value}: samples={metric.sample_count} "
            f"horizontal={metric.gaze_horizontal_median:.5f} "
            f"vertical={metric.gaze_vertical_median:.5f} "
            f"vertical_lid={metric.gaze_lid_vertical_median:.5f} "
            f"stationary_p95={metric.stationary_p95:.5f} "
            f"head_pose_p95_deg={metric.head_pose_p95_deg:.5f}"
        )
        lines.append(
            "  "
            + " ".join(
                f"{name}={value:.5f}"
                for name, value in zip(
                    (
                        "left_x",
                        "left_y",
                        "right_x_corrected",
                        "right_y",
                        "left_lid_y",
                        "right_lid_y",
                        "yaw_deg",
                        "pitch_deg",
                        "roll_deg",
                    ),
                    metric.feature_medians,
                    strict=True,
                )
            )
        )
    lines.extend(("", "--- Center-relative direction checks ---"))
    for check in result.direction_checks:
        lines.append(
            f"{check.direction.value}: verdict={check.verdict.value} "
            f"vertical_source={check.vertical_signal_source or '-'} "
            f"combined_delta={check.combined_delta:+.5f} "
            f"left_delta={check.left_eye_delta:+.5f} "
            f"right_delta={check.right_eye_delta:+.5f}"
        )
    stable = ", ".join(direction.value for direction in result.stable_directions) or "none"
    lines.extend(("", f"Stable directions: {stable}"))
    return "\n".join(lines) + "\n"


def write_feature_check_report(
    result: FeatureCheckResult,
    *,
    camera_id: str,
    screen_geometry: ScreenGeometry,
    directory: Path = FEATURE_CHECK_DIR,
    generated_at: datetime | None = None,
) -> FeatureCheckArtifacts:
    generated = generated_at or datetime.now(UTC)
    stamp = generated.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"feature_check_{stamp}.json"
    text_path = directory / f"feature_check_{stamp}.txt"
    payload: dict[str, JSONValue] = {
        "generated_at": generated.astimezone(UTC).isoformat(),
        "camera_id": camera_id,
        "screen_geometry": screen_geometry.to_dict(),
        "result": result.to_dict(),
    }
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    text_path.write_text(
        format_feature_check_report(
            result,
            camera_id=camera_id,
            screen_geometry=screen_geometry,
            generated_at=generated,
        ),
        encoding="utf-8",
    )
    return FeatureCheckArtifacts(json_path, text_path)


def run_gaze_feature_check(*, camera_index: int = 0) -> int:
    """Run the diagnostic using calibration-tolerant tracking limits."""

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
    screen = application.primaryScreen()  # type: ignore[attr-defined]
    if screen is None:
        runtime.close()
        print("No primary display is available for the feature check.")
        return 1
    geometry = describe_screen(screen)
    display_guard = DisplayGuard(geometry, expected_identity=identify_screen(screen))
    camera_id = f"camera-{camera_index}"
    window = _FeatureCheckWindow(
        display_guard,
        runtime,
        FeatureCheckController(clock_ms=frame_monotonic_ms),
        camera_id=camera_id,
        screen_geometry=geometry,
    )
    window.show()
    try:
        return int(application.exec())
    finally:
        runtime.close()


class _FeatureCheckWindow:  # pragma: no cover - display and live camera required
    def __init__(
        self,
        display_guard: DisplayGuard,
        runtime: VisionRuntime,
        controller: FeatureCheckController,
        *,
        camera_id: str,
        screen_geometry: ScreenGeometry,
        interval_ms: int = DEFAULT_TIMER_INTERVAL_MS,
    ) -> None:
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        from PySide6.QtGui import QFont  # noqa: PLC0415
        from PySide6.QtWidgets import QLabel, QWidget  # noqa: PLC0415

        self._runtime = runtime
        self._display_guard = display_guard
        self._controller = controller
        self._camera_id = camera_id
        self._screen_geometry = screen_geometry
        self._closed = False
        self._written = False
        self._widget = QWidget()
        self._display = DisplayWatcher(display_guard, self._widget.screen, self._on_display_change)
        self._widget.setWindowTitle(_WINDOW_TITLE)
        self._widget.setStyleSheet("background: #101820; color: white;")
        self._widget.closeEvent = self._on_close  # type: ignore[method-assign]
        self._widget.keyPressEvent = self._on_key_press  # type: ignore[method-assign]

        self._target = QLabel("●", self._widget)
        target_font = QFont()
        target_font.setPointSize(46)
        target_font.setBold(True)
        self._target.setFont(target_font)
        self._target.setStyleSheet("color: #00E5FF; background: transparent;")
        self._target.setAccessibleName("Known gaze feature-check target")

        self._status = QLabel(self._widget)
        self._status.setGeometry(20, 20, 1180, 125)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            "background: rgba(0, 0, 0, 190); color: white; padding: 14px; "
            "font-size: 18px; border: 1px solid #607D8B;"
        )
        self._status.setAccessibleName("Gaze feature-check progress and instructions")

        self._summary = QLabel(self._widget)
        self._summary.setGeometry(20, 160, 1180, 420)
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(
            "background: rgba(0, 0, 0, 210); color: #FFD740; padding: 16px; font-size: 17px;"
        )
        self._summary.hide()

        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(interval_ms)
        self._render(self._controller.view())

        self._freeze = OutputFreeze(self._timer.stop, self._target.hide)

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
        self._render(self._controller.view())

    def _on_tick(self) -> None:
        # The ruler these measurements are scored against must still exist.
        # Recording numbers for a screen that has changed is worse than
        # recording nothing, so this aborts rather than warns.
        if self._display.poll() is not None:
            self._freeze_for_display()
            return

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
            self._render(self._controller.tracking_lost())
            return
        view = self._controller.ingest(tick.accepted_observation)
        self._render(view)
        if view.phase is FeatureCheckPhase.COMPLETE:
            self._timer.stop()
            self._publish_result()

    def _on_key_press(self, event: Any) -> None:
        from PySide6.QtCore import Qt  # noqa: PLC0415

        if event.key() == Qt.Key.Key_Escape:
            self._controller.cancel()
            self._shutdown()
            self._widget.close()
        elif event.key() == Qt.Key.Key_R:
            self._written = False
            self._summary.hide()
            self._timer.start(DEFAULT_TIMER_INTERVAL_MS)
            self._render(self._controller.restart())
        else:
            event.ignore()

    def _render(self, view: FeatureCheckView) -> None:
        direction = "DONE" if view.target is None else view.target.direction.value
        self._status.setText(
            f"Direction {view.target_number}/{view.total_targets}: {direction} — "
            f"phase={view.phase.value} — samples={view.accepted_samples}/{view.required_samples} — "
            f"{view.remaining_ms / 1000:.1f}s remaining\n{view.feedback}  R=restart, Esc=close"
        )
        self._target.setVisible(view.target is not None)
        if view.target is not None:
            self._place_target(view.target.position.x, view.target.position.y)

    def _place_target(self, normalized_x: float, normalized_y: float) -> None:
        self._target.adjustSize()
        x = round(normalized_x * max(0, self._widget.width() - self._target.width()))
        y = round(normalized_y * max(0, self._widget.height() - self._target.height()))
        self._target.move(x, y)
        self._target.raise_()
        self._status.raise_()

    def _publish_result(self) -> None:
        if self._written:
            return
        result = self._controller.result()
        artifacts = write_feature_check_report(
            result,
            camera_id=self._camera_id,
            screen_geometry=self._screen_geometry,
        )
        self._written = True
        checks = "\n".join(
            f"{check.direction.value}: {check.verdict.value} "
            f"(L={check.left_eye_delta:+.3f}, R={check.right_eye_delta:+.3f})"
            for check in result.direction_checks
        )
        unstable = [
            metric.direction.value
            for metric in result.metrics
            if metric.direction not in result.stable_directions
        ]
        self._summary.setText(
            f"Overall diagnostic pass: {result.overall_pass}\n{checks}\n"
            f"Unstable directions: {', '.join(unstable) or 'none'}\n"
            f"Report: {artifacts.text_path}"
        )
        self._summary.show()
        self._summary.raise_()
        print(f"Feature-check JSON written to: {artifacts.json_path}")
        print(f"Feature-check report written to: {artifacts.text_path}")

    def _on_close(self, event: Any) -> None:
        self._shutdown()
        event.accept()

    def _on_display_change(self, change: object) -> None:
        self._status.setText(getattr(change, "message", "The display changed."))

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
