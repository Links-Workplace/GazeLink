"""The M1 live debug window.

This module is the only place in GAZELINK that talks to Qt, and it is imported
lazily so the default test suite never needs a display server.  It is a
developer diagnostic surface, deliberately separate from the user-facing
experience that later milestones will build.

The window renders the current frame and the overlay commands produced for that
same frame.  It never writes an image, a screenshot, or a landmark to disk, and
closing it releases the camera and the model before the widget goes away.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from gazelink.camera import CameraError, OpenCVCameraSource
from gazelink.confidence import ConfidencePolicySettings, TrackingRecoverySettings
from gazelink.config import ConfidenceThresholds
from gazelink.errors import ErrorCode, UserFacingError, user_facing_error
from gazelink.overlay import CircleCommand, OverlayCommand, RectangleCommand, TextCommand
from gazelink.runtime import RuntimeTick, VisionRuntime
from gazelink.vision import DEFAULT_FACE_LANDMARKER_MODEL, FaceLandmarkerAdapter

DEFAULT_TIMER_INTERVAL_MS = 16
MIN_DIAGNOSTICS_PRINT_INTERVAL_S = 1.0
_WINDOW_TITLE = "GAZELINK — M1 debug overlay"


def build_runtime(
    *,
    camera_index: int = 0,
    confidence_settings: ConfidencePolicySettings | None = None,
) -> VisionRuntime:
    """Assemble the real capture/vision/policy stack for a live session."""

    return VisionRuntime(
        source=OpenCVCameraSource(camera_index),
        engine=FaceLandmarkerAdapter(model_path=DEFAULT_FACE_LANDMARKER_MODEL),
        recovery_settings=TrackingRecoverySettings(
            confidence=confidence_settings
            or ConfidencePolicySettings(thresholds=ConfidenceThresholds())
        ),
    )


def run_debug_overlay(*, camera_index: int = 0) -> int:
    """Open the live debug window and return a process exit code.

    Qt is imported here rather than at module scope so that importing GAZELINK,
    or collecting its tests, never requires a display server.

    The camera is opened *before* ``QApplication`` is constructed. Qt's real
    Windows platform plugin initializes an STA COM apartment as part of its
    native window-system integration; a camera backend's blocking property
    negotiation (``VideoCapture.set()``) can then deadlock waiting on a message
    pump that only starts once ``exec()`` runs later -- which can't happen
    while this call is still blocking. Opening the camera first, in a plain
    Python process with no COM apartment set up yet, avoids that deadlock at
    its root rather than only bounding it with a timeout.
    """

    runtime = build_runtime(camera_index=camera_index)
    try:
        runtime.start()
    except CameraError as error:
        runtime.close()
        _print_failure(error.failure)
        return 1

    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    application = QApplication.instance() or QApplication([])
    window = _DebugWindow(runtime)
    window.show()
    try:
        return int(application.exec())
    finally:
        # exec() also returns when the window is closed by the window manager,
        # so releasing here covers every exit path including an exception.
        runtime.close()


def _print_failure(failure: UserFacingError) -> None:
    print(f"{failure.message} {failure.recovery_action.value}")


def _should_print_diagnostics(
    last_printed_s: float | None, now_s: float, *, min_interval_s: float
) -> bool:
    """Gate periodic terminal diagnostics so a live session stays readable.

    ``None`` means nothing has been printed yet, so the first tick always
    qualifies; after that, printing is allowed once ``min_interval_s`` has
    elapsed since the last successful print.
    """

    if last_printed_s is None:
        return True
    return (now_s - last_printed_s) >= min_interval_s


def _print_diagnostics(tick: RuntimeTick) -> None:
    """Re-emit the same diagnostic text the overlay draws, to the terminal.

    Reuses ``tick.view.commands`` rather than recomputing anything, so the
    printed snapshot can never drift from what the window shows.
    """

    print(f"--- diagnostics @ frame {tick.frame.frame_id} ---")
    for command in tick.view.commands:
        if isinstance(command, TextCommand):
            print(command.text)


class _DebugWindow:  # pragma: no cover - requires a display server
    """Thin Qt shell; all decisions live in ``VisionRuntime``.

    Implemented as a wrapper rather than a ``QWidget`` subclass at module scope
    so that importing this module does not require Qt to be importable.
    """

    def __init__(self, runtime: VisionRuntime, *, interval_ms: int = DEFAULT_TIMER_INTERVAL_MS):
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget  # noqa: PLC0415

        self._runtime = runtime
        self._closed = False
        self._last_diagnostics_print_s: float | None = None
        self._widget = QWidget()
        self._widget.setWindowTitle(_WINDOW_TITLE)
        self._widget.resize(960, 620)
        self._canvas = QLabel()
        self._status = QLabel("Starting…")
        layout = QVBoxLayout(self._widget)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status)
        self._widget.closeEvent = self._on_close  # type: ignore[method-assign]
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(interval_ms)

    def show(self) -> None:
        self._widget.show()

    def _on_close(self, event: Any) -> None:
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._runtime.close()

    def _on_tick(self) -> None:
        try:
            tick = self._runtime.tick()
        except CameraError as error:
            self._shutdown()
            self._show_failure(error.failure)
            return
        except Exception:
            # Any unexpected fault must still release the camera and the model
            # rather than leaving a live capture behind a frozen window.
            self._shutdown()
            self._show_failure(user_facing_error(ErrorCode.INTERNAL_ERROR, state="debug_overlay"))
            raise
        if tick is None:
            return
        self._paint(tick.frame.width, tick.frame.height, tick.frame.image, tick.view.commands)
        self._status.setText(f"{tick.view.status.value} — press the window close button to stop")
        now = time.monotonic()
        if _should_print_diagnostics(
            self._last_diagnostics_print_s, now, min_interval_s=MIN_DIAGNOSTICS_PRINT_INTERVAL_S
        ):
            _print_diagnostics(tick)
            self._last_diagnostics_print_s = now

    def _show_failure(self, failure: UserFacingError) -> None:
        self._status.setText(f"{failure.message} {failure.recovery_action.value}")

    def _paint(
        self,
        width: int,
        height: int,
        image_bytes: bytes | None,
        commands: Sequence[OverlayCommand],
    ) -> None:
        from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: PLC0415
        from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap  # noqa: PLC0415

        if image_bytes is None:
            return
        # copy() detaches the QImage from the frame buffer, so nothing keeps a
        # reference to the captured bytes once this frame has been drawn.
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
                else:
                    painter.drawText(QPointF(command.x_px, command.y_px), command.text)
        finally:
            painter.end()
        self._canvas.setPixmap(
            pixmap.scaled(
                self._canvas.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
