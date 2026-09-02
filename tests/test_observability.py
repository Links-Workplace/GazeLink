from __future__ import annotations

import json
import logging

import pytest

from gazelink.observability import (
    REDACTED,
    UNSERIALIZABLE,
    PrivacyViolation,
    StructuredLogger,
    new_session_correlation_id,
    sanitize_log_fields,
    validate_log_fields,
)


def test_structured_log_has_required_diagnostics_and_no_sensitive_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = StructuredLogger(component="vision", logger=logging.getLogger("gazelink.test"))

    with caplog.at_level(logging.INFO, logger="gazelink.test"):
        logger.info(
            "tracking_lost",
            state="lost",
            reason_code="camera_disconnected",
            frame=b"actual camera bytes",
            screenshot=b"actual screenshot bytes",
            metadata={"landmarks": [[12.0, 13.0]], "frame_id": "f-12"},
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["component"] == "vision"
    assert payload["state"] == "lost"
    assert payload["reason_code"] == "camera_disconnected"
    assert payload["correlation_id"] == logger.correlation_id
    assert payload["frame"] == REDACTED
    assert payload["screenshot"] == REDACTED
    assert payload["metadata"]["landmarks"] == REDACTED
    assert payload["metadata"]["frame_id"] == "f-12"
    assert "actual camera bytes" not in caplog.text
    assert "actual screenshot bytes" not in caplog.text
    assert "12.0" not in caplog.text


def test_sanitization_never_uses_unknown_object_representation() -> None:
    class DangerousRepresentation:
        def __repr__(self) -> str:
            return "private-frame-contents"

    fields = sanitize_log_fields({"model_result": DangerousRepresentation(), "samples": [1, 2]})

    assert fields == {"model_result": UNSERIALIZABLE, "samples": REDACTED}
    assert "private-frame-contents" not in json.dumps(fields)


def test_privacy_validation_rejects_nested_landmarks() -> None:
    with pytest.raises(PrivacyViolation):
        validate_log_fields({"diagnostics": {"face_landmarks": [[0.1, 0.2]]}})


def test_session_correlation_ids_are_opaque_and_distinct() -> None:
    first = new_session_correlation_id()
    second = new_session_correlation_id()

    assert first != second
    assert len(first) == 36
