from __future__ import annotations

import pytest

from gazelink.errors import ErrorCode, RecoveryAction, user_facing_error


def test_user_facing_error_has_stable_reason_and_recovery_action() -> None:
    error = user_facing_error(ErrorCode.CAMERA_UNAVAILABLE, state="camera_error")

    assert error.reason_code == "camera_unavailable"
    assert error.recovery_action is RecoveryAction.CHECK_CAMERA
    assert error.to_payload() == {
        "code": "camera_unavailable",
        "state": "camera_error",
        "message": "GAZELINK cannot access the camera.",
        "recovery_action": "Check that the camera is connected and not in use by another app.",
    }


def test_user_facing_error_rejects_empty_state() -> None:
    with pytest.raises(ValueError, match="state"):
        user_facing_error(ErrorCode.TRACKING_LOST, state="   ")
