"""Privacy-preserving structured logging for GAZELINK.

This module intentionally accepts only a small, JSON-safe value set.  Logging
must remain useful for safety and recovery investigation without becoming a
second storage path for camera or biometric data.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Final, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
SafeLogValue: TypeAlias = JsonScalar | dict[str, "SafeLogValue"]

REDACTED: Final = "[REDACTED]"
UNSERIALIZABLE: Final = "[UNSERIALIZABLE]"

_SENSITIVE_EXACT_KEYS: Final = frozenset(
    {
        "frame",
        "frames",
        "raw_frame",
        "raw_frames",
        "image",
        "images",
        "screenshot",
        "screenshots",
        "video",
        "landmark",
        "landmarks",
        "face_landmarks",
        "iris_landmarks",
        "biometric",
        "biometrics",
        "pixel_buffer",
    }
)
_SENSITIVE_KEY_PARTS: Final = (
    "landmark",
    "screenshot",
    "biometric",
    "face_mesh",
    "raw_frame",
    "raw_image",
    "raw_video",
    "pixel_buffer",
)


class PrivacyViolation(ValueError):
    """Raised when a value is not permitted in structured diagnostics."""


def new_session_correlation_id() -> str:
    """Create an opaque correlation ID scoped to the current application session."""

    return str(uuid.uuid4())


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in _SENSITIVE_EXACT_KEYS or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


def _sanitize_value(value: object) -> SafeLogValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else UNSERIALIZABLE
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Enum):
        return _sanitize_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(str(key)) else _sanitize_value(item)
            for key, item in value.items()
        }
    # Lists and tuples can contain landmark vectors.  They have no legitimate
    # role in our logs, so reject them as a class instead of trying to infer
    # whether a numeric sequence is biometric data.
    if isinstance(value, (bytes, bytearray, memoryview, list, tuple, set, frozenset)):
        return REDACTED
    # Never call repr() or str() on an arbitrary object: many vision objects
    # include their underlying frame or complete landmark set in those methods.
    return UNSERIALIZABLE


def sanitize_log_fields(fields: Mapping[str, object]) -> dict[str, SafeLogValue]:
    """Return a JSON-safe, privacy-filtered copy of diagnostic fields.

    Sensitive values are replaced before serialization.  Unknown objects are
    represented by a fixed marker, so custom ``repr`` implementations cannot
    leak their contents.
    """

    return {
        str(key): REDACTED if _is_sensitive_key(str(key)) else _sanitize_value(value)
        for key, value in fields.items()
    }


def validate_log_fields(fields: Mapping[str, object]) -> None:
    """Reject values that would be filtered by :func:`sanitize_log_fields`.

    Use this at subsystem boundaries that construct diagnostic payloads.  The
    logger itself still redacts rather than failing an error path.
    """

    sanitized = sanitize_log_fields(fields)
    if REDACTED in _walk_values(sanitized) or UNSERIALIZABLE in _walk_values(sanitized):
        raise PrivacyViolation("diagnostic fields contain prohibited or unserializable data")


def _walk_values(value: SafeLogValue) -> list[SafeLogValue]:
    if isinstance(value, dict):
        values: list[SafeLogValue] = []
        for nested in value.values():
            values.extend(_walk_values(nested))
        return values
    return [value]


@dataclass(slots=True)
class StructuredLogger:
    """Emit one privacy-filtered JSON record per application event."""

    component: str
    logger: logging.Logger
    correlation_id: str = field(default_factory=new_session_correlation_id)

    def event(
        self,
        level: int,
        event: str,
        *,
        state: str,
        reason_code: str,
        **fields: object,
    ) -> None:
        """Emit a structured event with fields required by the technical spec."""

        payload: dict[str, SafeLogValue] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": logging.getLevelName(level),
            "component": self.component,
            "event": event,
            "state": state,
            "reason_code": reason_code,
            "correlation_id": self.correlation_id,
        }
        payload.update(sanitize_log_fields(fields))
        self.logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def info(self, event: str, *, state: str, reason_code: str, **fields: object) -> None:
        """Emit an informational structured event."""

        self.event(logging.INFO, event, state=state, reason_code=reason_code, **fields)

    def warning(self, event: str, *, state: str, reason_code: str, **fields: object) -> None:
        """Emit a warning structured event."""

        self.event(logging.WARNING, event, state=state, reason_code=reason_code, **fields)

    def error(self, event: str, *, state: str, reason_code: str, **fields: object) -> None:
        """Emit an error structured event without exception representations."""

        self.event(logging.ERROR, event, state=state, reason_code=reason_code, **fields)
