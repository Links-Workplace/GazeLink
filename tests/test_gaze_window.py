"""Pure formatting tests for the M2 diagnostic surface."""

from types import SimpleNamespace

from gazelink.domain import GazePoint, GazeSample, PixelPoint, ReasonCode, ScreenGeometry
from gazelink.gaze_engine import GazeEstimationResult
from gazelink.gaze_window import _GazeCheckWindow, clamp_panel_position, format_gaze_status


def test_status_labels_raw_corrected_and_control_safety_explicitly() -> None:
    sample = GazeSample(
        source_frame_id=1,
        sampled_at_monotonic_ms=100.0,
        raw_normalized=GazePoint(0.4, 0.5),
        corrected_normalized=GazePoint(0.45, 0.52),
        filtered_normalized=GazePoint(0.45, 0.52),
        screen_position=PixelPoint(863, 561),
        screen_id="primary",
        confidence=0.5,
        valid_for_control=False,
    )
    text = format_gaze_status(GazeEstimationResult(sample, ()))
    assert "RAW / CORRECTED / FILTERED / NOT FOR CONTROL" in text
    assert "raw=(0.400, 0.500)" in text
    assert "corrected=(0.450, 0.520)" in text
    assert "filtered=(0.450, 0.520)" in text


def test_rejected_status_has_reason_and_no_coordinate() -> None:
    text = format_gaze_status(GazeEstimationResult(None, (ReasonCode.CALIBRATION_STALE,)))
    assert "NO VETTED GAZE SAMPLE" in text
    assert "CALIBRATION_STALE" in text


def test_draggable_panel_is_clamped_inside_the_diagnostic_screen() -> None:
    size = {
        "container_width": 1920,
        "container_height": 1080,
        "panel_width": 1200,
        "panel_height": 174,
    }
    assert clamp_panel_position(200, 300, **size) == (200, 300)
    assert clamp_panel_position(-50, -20, **size) == (0, 0)
    assert clamp_panel_position(5000, 5000, **size) == (720, 906)


def test_panel_remains_reachable_when_the_screen_is_smaller_than_it() -> None:
    assert clamp_panel_position(
        100,
        100,
        container_width=800,
        container_height=120,
        panel_width=1200,
        panel_height=174,
    ) == (0, 0)


def test_real_qt_panel_centers_and_dispatches_drag_events(monkeypatch: object) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")  # type: ignore[attr-defined]

    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])

    class _Runtime:
        def close(self) -> None:
            return None

    diagnostic = SimpleNamespace(
        engine=SimpleNamespace(corrections=(), enabled=True),
        view=lambda: SimpleNamespace(
            instruction="",
            before_error_normalized=None,
            after_error_normalized=None,
            target=None,
            state=None,
        ),
    )
    window = _GazeCheckWindow(
        _Runtime(),  # type: ignore[arg-type]
        SimpleNamespace(screen_geometry=ScreenGeometry("primary", 1920, 1080, 1.0)),
        diagnostic,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        interval_ms=60_000,
    )
    window._widget.resize(1920, 1080)
    window._center_info_panel()
    panel = window._info_panel
    assert panel.pos() == QPoint(360, 453)

    press_global = panel.mapToGlobal(QPoint(20, 20))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(20, 20),
        QPointF(press_global),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    application.sendEvent(panel, press)

    desired_global = window._widget.mapToGlobal(QPoint(320, 420))
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(20, 20),
        QPointF(desired_global),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    application.sendEvent(panel, move)
    assert panel.pos() == QPoint(300, 400)

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(20, 20),
        QPointF(desired_global),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    application.sendEvent(panel, release)
    window._shutdown()
