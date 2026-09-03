"""Live prediction marker, shown alongside a target -- the visual sanity check.

Neither the calibration screen nor the existing validation screen draws the
point the system currently predicts; both draw only the target.  There is
therefore no way to see with your own eyes whether a reported pixel error
matches what is actually happening on screen -- exactly the cross-check this
measurement day exists to make possible.

The state decision (visible vs. hidden, and why) is a pure function of a
:class:`~gazelink.gaze_engine.GazeEstimationResult`, so it can be tested
without Qt.  :class:`PredictionOverlay` is a thin Qt shell around it.  Nothing
here trains, maps, or corrects gaze -- it only draws a mark that already-computed
prediction, or its absence, produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gazelink.domain import ContractValidationError, GazePoint
from gazelink.gaze_engine import CalibrationModel, GazeEstimationResult

PREDICTION_MARKER_COLOR = "#FF4081"
PREDICTION_MARKER_GLYPH = "✕"  # magenta X, distinct from every target dot


def load_overlay_model(path: Path) -> CalibrationModel:
    """Load an explicitly-named model for live prediction display only.

    Reuses :meth:`CalibrationModel.from_dict`'s own validation (including its
    ``model_id`` integrity check) rather than restating it -- there is
    exactly one definition of what a valid, un-tampered model file looks
    like. Raises :class:`ContractValidationError` on anything invalid, or
    ``OSError`` if the file cannot be read; callers must not treat either as
    "no overlay" and fall silent, since a broken ``--overlay-model`` path is
    an operator mistake, not a normal absence of an overlay.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationModel.from_dict(data)


@dataclass(frozen=True, slots=True)
class PredictionOverlayState:
    """What to show this tick; never carries a point from a prior tick."""

    visible: bool
    point: GazePoint | None
    reason_text: str

    def __post_init__(self) -> None:
        if self.visible and self.point is None:
            raise ContractValidationError("a visible state must carry a point")
        if not self.visible and self.point is not None:
            raise ContractValidationError("a hidden state must not carry a stale point")


_HIDDEN_NO_TRACKING = PredictionOverlayState(
    visible=False, point=None, reason_text="no tracking this frame"
)


def compute_prediction_overlay_state(
    result: GazeEstimationResult | None,
) -> PredictionOverlayState:
    """Decide what the overlay shows for one tick's estimation result.

    ``result`` is ``None`` exactly when the tick produced no observation at
    all (camera loss, tracking withheld before estimation even ran).  A
    rejected estimation (``result.sample is None``) is reported with its own
    reason codes -- including a geometry mismatch between the overlay model
    and the live screen, which must be a visible, readable failure rather
    than a silently blank marker indistinguishable from ordinary tracking
    loss.
    """

    if result is None:
        return _HIDDEN_NO_TRACKING
    if not isinstance(result, GazeEstimationResult):
        raise ContractValidationError("result must be a GazeEstimationResult or None")
    if result.sample is None:
        reasons = ", ".join(reason.value for reason in result.reason_codes) or "unavailable"
        return PredictionOverlayState(visible=False, point=None, reason_text=reasons)
    return PredictionOverlayState(visible=True, point=result.sample.raw_normalized, reason_text="")


class PredictionOverlay:  # pragma: no cover - requires Qt
    """Thin Qt shell around :func:`compute_prediction_overlay_state`.

    Deliberately holds no state of its own beyond the widgets: every call to
    :meth:`update` recomputes visibility from scratch, so a frame with no
    prediction can never leave the previous frame's marker on screen.
    """

    def __init__(self, parent: Any) -> None:
        from PySide6.QtGui import QFont  # noqa: PLC0415
        from PySide6.QtWidgets import QLabel  # noqa: PLC0415

        self._mark = QLabel(PREDICTION_MARKER_GLYPH, parent)
        font = QFont()
        font.setPointSize(40)
        font.setBold(True)
        self._mark.setFont(font)
        self._mark.setStyleSheet(f"color: {PREDICTION_MARKER_COLOR}; background: transparent;")
        self._mark.setAccessibleName("Live gaze prediction")
        self._mark.adjustSize()
        self._mark.hide()
        self._reason = QLabel(parent)
        self._reason.setStyleSheet(
            f"color: {PREDICTION_MARKER_COLOR}; background: rgba(16, 24, 32, 210); "
            "padding: 6px; font-weight: bold;"
        )
        self._reason.setWordWrap(True)
        self._reason.setAccessibleName("Live gaze prediction status")
        self._reason.hide()

    def update(self, result: GazeEstimationResult | None) -> None:
        state = compute_prediction_overlay_state(result)
        if not state.visible:
            self._mark.hide()
            self._reason.setText(f"prediction unavailable: {state.reason_text}")
            self._reason.adjustSize()
            self._reason.show()
            self._reason.raise_()
            return
        self._reason.hide()
        assert state.point is not None
        parent = self._mark.parentWidget()
        width = 0 if parent is None else parent.width()
        height = 0 if parent is None else parent.height()
        x = round(min(1.0, max(0.0, state.point.x)) * max(0, width - self._mark.width()))
        y = round(min(1.0, max(0.0, state.point.y)) * max(0, height - self._mark.height()))
        self._mark.move(x, y)
        self._mark.show()
        self._mark.raise_()

    def place_reason_label(self, x: int, y: int, width: int, height: int) -> None:
        """Position the status text; called once after layout is known."""

        self._reason.setGeometry(x, y, width, height)
