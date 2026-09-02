"""Stable, user-facing failure contracts and recovery guidance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    """Reason codes that are safe to display and emit in structured logs."""

    CAMERA_UNAVAILABLE = "camera_unavailable"
    CAMERA_DISCONNECTED = "camera_disconnected"
    TRACKING_LOST = "tracking_lost"
    LOW_CONFIDENCE = "low_confidence"
    CALIBRATION_REQUIRED = "calibration_required"
    CALIBRATION_INVALID = "calibration_invalid"
    CONTROL_PAUSED = "control_paused"
    INTERNAL_ERROR = "internal_error"


class RecoveryAction(StrEnum):
    """A concise action a person can take without developer terminology."""

    CHECK_CAMERA = "Check that the camera is connected and not in use by another app."
    REPOSITION_FACE = "Move back into view and keep your face well lit."
    RECALIBRATE = "Run calibration again before enabling control."
    RESUME_WHEN_READY = "Resume control when you are ready."
    RESTART_APPLICATION = "Close and reopen GAZELINK."


_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.CAMERA_UNAVAILABLE: "GAZELINK cannot access the camera.",
    ErrorCode.CAMERA_DISCONNECTED: "The camera connection was lost.",
    ErrorCode.TRACKING_LOST: "GAZELINK cannot see your face clearly.",
    ErrorCode.LOW_CONFIDENCE: "GAZELINK needs a steadier view before it can act.",
    ErrorCode.CALIBRATION_REQUIRED: "Calibration is needed before control can start.",
    ErrorCode.CALIBRATION_INVALID: "The saved calibration does not match the current setup.",
    ErrorCode.CONTROL_PAUSED: "Control is paused.",
    ErrorCode.INTERNAL_ERROR: "GAZELINK stopped control to keep you safe.",
}
_RECOVERY_ACTIONS: dict[ErrorCode, RecoveryAction] = {
    ErrorCode.CAMERA_UNAVAILABLE: RecoveryAction.CHECK_CAMERA,
    ErrorCode.CAMERA_DISCONNECTED: RecoveryAction.CHECK_CAMERA,
    ErrorCode.TRACKING_LOST: RecoveryAction.REPOSITION_FACE,
    ErrorCode.LOW_CONFIDENCE: RecoveryAction.REPOSITION_FACE,
    ErrorCode.CALIBRATION_REQUIRED: RecoveryAction.RECALIBRATE,
    ErrorCode.CALIBRATION_INVALID: RecoveryAction.RECALIBRATE,
    ErrorCode.CONTROL_PAUSED: RecoveryAction.RESUME_WHEN_READY,
    ErrorCode.INTERNAL_ERROR: RecoveryAction.RESTART_APPLICATION,
}


@dataclass(frozen=True, slots=True)
class UserFacingError:
    """A display-ready error without raw exception, frame, or biometric data."""

    code: ErrorCode
    state: str
    message: str
    recovery_action: RecoveryAction

    @property
    def reason_code(self) -> str:
        """Return the stable diagnostic reason code shared by UI and logging."""

        return self.code.value

    def to_payload(self) -> dict[str, str]:
        """Return the public JSON-compatible shape for a UI or local API."""

        return {
            "code": self.code.value,
            "state": self.state,
            "message": self.message,
            "recovery_action": self.recovery_action.value,
        }


def user_facing_error(code: ErrorCode, *, state: str) -> UserFacingError:
    """Create the approved plain-language message and recovery action for ``code``."""

    if not state.strip():
        raise ValueError("state must not be empty")
    return UserFacingError(
        code=code,
        state=state,
        message=_MESSAGES[code],
        recovery_action=_RECOVERY_ACTIONS[code],
    )
