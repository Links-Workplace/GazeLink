"""Validated, serializable configuration grouped by its ownership scope."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gazelink.domain import ContractValidationError, JSONValue, ScreenGeometry


class ConfigurationError(ContractValidationError):
    """Raised when a configuration value is missing, malformed, or unsafe."""


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SmoothingMethod(StrEnum):
    NONE = "NONE"
    EMA = "EMA"
    ONE_EURO = "ONE_EURO"
    KALMAN = "KALMAN"


def _number(
    value: object, field_name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a finite number, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{field_name} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigurationError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigurationError(f"{field_name} must be <= {maximum}")
    return number


def _integer(value: object, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field_name} must be an integer, not {value!r}")
    if value < minimum:
        raise ConfigurationError(f"{field_name} must be >= {minimum}")
    return value


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a bool")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} must be an object")
    return value


def _required(data: Mapping[str, Any], key: str) -> Any:
    """Return a required serialized value so constructor validation owns errors."""

    if key not in data:
        return None
    return data[key]


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """Application-owned settings; no device or profile preferences belong here."""

    schema_version: int = 1
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    local_api_enabled: bool = False

    def __post_init__(self) -> None:
        _integer(self.schema_version, "application.schema_version")
        if not isinstance(self.log_level, LogLevel):
            raise ConfigurationError("application.log_level must be a LogLevel")
        _bool(self.debug_mode, "application.debug_mode")
        _bool(self.local_api_enabled, "application.local_api_enabled")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "log_level": self.log_level.value,
            "debug_mode": self.debug_mode,
            "local_api_enabled": self.local_api_enabled,
        }

    @classmethod
    def from_dict(cls, value: object) -> ApplicationConfig:
        data = _mapping(value, "application config")
        return cls(
            schema_version=data.get("schema_version", 1),
            log_level=LogLevel(data.get("log_level", LogLevel.INFO.value)),
            debug_mode=data.get("debug_mode", False),
            local_api_enabled=data.get("local_api_enabled", False),
        )


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Device-owned settings, including the current single-screen geometry."""

    camera_id: str
    preferred_width_px: int
    preferred_height_px: int
    preferred_fps: float
    screen_geometry: ScreenGeometry

    def __post_init__(self) -> None:
        _string(self.camera_id, "device.camera_id")
        _integer(self.preferred_width_px, "device.preferred_width_px")
        _integer(self.preferred_height_px, "device.preferred_height_px")
        object.__setattr__(
            self, "preferred_fps", _number(self.preferred_fps, "device.preferred_fps", minimum=1.0)
        )
        if not isinstance(self.screen_geometry, ScreenGeometry):
            raise ConfigurationError("device.screen_geometry must be a ScreenGeometry")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "camera_id": self.camera_id,
            "preferred_width_px": self.preferred_width_px,
            "preferred_height_px": self.preferred_height_px,
            "preferred_fps": self.preferred_fps,
            "screen_geometry": self.screen_geometry.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> DeviceConfig:
        data = _mapping(value, "device config")
        return cls(
            camera_id=_required(data, "camera_id"),
            preferred_width_px=_required(data, "preferred_width_px"),
            preferred_height_px=_required(data, "preferred_height_px"),
            preferred_fps=_required(data, "preferred_fps"),
            screen_geometry=ScreenGeometry.from_dict(_required(data, "screen_geometry")),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Minimum confidence ratios for display, calibration, and OS control."""

    display: float = 0.50
    calibration: float = 0.70
    control: float = 0.85

    def __post_init__(self) -> None:
        display = _number(self.display, "profile.confidence.display", minimum=0.0, maximum=1.0)
        calibration = _number(
            self.calibration, "profile.confidence.calibration", minimum=0.0, maximum=1.0
        )
        control = _number(self.control, "profile.confidence.control", minimum=0.0, maximum=1.0)
        if not display <= calibration <= control:
            raise ConfigurationError(
                "confidence thresholds must satisfy display <= calibration <= control"
            )
        object.__setattr__(self, "display", display)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "control", control)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"display": self.display, "calibration": self.calibration, "control": self.control}

    @classmethod
    def from_dict(cls, value: object) -> ConfidenceThresholds:
        data = _mapping(value, "confidence thresholds")
        return cls(
            display=data.get("display", 0.50),
            calibration=data.get("calibration", 0.70),
            control=data.get("control", 0.85),
        )


@dataclass(frozen=True, slots=True)
class GestureTimingConfig:
    """Monotonic-clock durations for future blink/dwell state machines."""

    natural_blink_max_ms: int = 400
    intentional_hold_min_ms: int = 700
    cooldown_ms: int = 800
    dwell_duration_ms: int = 900
    # Held longer than this, a close means "confirm" rather than "next".
    # NOT VALIDATED AGAINST REAL USERS: chosen to sit clear of a natural
    # blink and of the intentional bar above, and to be reachable without
    # strain. It must be measured on real video before it is trusted.
    recovery_confirm_ms: int = 1500

    def __post_init__(self) -> None:
        natural = _integer(self.natural_blink_max_ms, "profile.gesture.natural_blink_max_ms")
        intentional = _integer(
            self.intentional_hold_min_ms, "profile.gesture.intentional_hold_min_ms"
        )
        cooldown = _integer(self.cooldown_ms, "profile.gesture.cooldown_ms")
        dwell = _integer(self.dwell_duration_ms, "profile.gesture.dwell_duration_ms")
        confirm = _integer(self.recovery_confirm_ms, "profile.gesture.recovery_confirm_ms")
        if intentional <= natural:
            raise ConfigurationError("intentional_hold_min_ms must exceed natural_blink_max_ms")
        if confirm <= intentional:
            raise ConfigurationError("recovery_confirm_ms must exceed intentional_hold_min_ms")
        object.__setattr__(self, "natural_blink_max_ms", natural)
        object.__setattr__(self, "intentional_hold_min_ms", intentional)
        object.__setattr__(self, "cooldown_ms", cooldown)
        object.__setattr__(self, "dwell_duration_ms", dwell)
        object.__setattr__(self, "recovery_confirm_ms", confirm)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "natural_blink_max_ms": self.natural_blink_max_ms,
            "intentional_hold_min_ms": self.intentional_hold_min_ms,
            "cooldown_ms": self.cooldown_ms,
            "dwell_duration_ms": self.dwell_duration_ms,
            "recovery_confirm_ms": self.recovery_confirm_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> GestureTimingConfig:
        data = _mapping(value, "gesture timing config")
        return cls(
            natural_blink_max_ms=data.get("natural_blink_max_ms", 400),
            intentional_hold_min_ms=data.get("intentional_hold_min_ms", 700),
            cooldown_ms=data.get("cooldown_ms", 800),
            dwell_duration_ms=data.get("dwell_duration_ms", 900),
            recovery_confirm_ms=data.get("recovery_confirm_ms", 1500),
        )


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """A named smoothing policy plus unitless tuning parameters."""

    method: SmoothingMethod = SmoothingMethod.EMA
    responsiveness: float = 0.35
    jitter_radius_px: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.method, SmoothingMethod):
            raise ConfigurationError("profile.smoothing.method must be a SmoothingMethod")
        object.__setattr__(
            self,
            "responsiveness",
            _number(
                self.responsiveness, "profile.smoothing.responsiveness", minimum=0.0, maximum=1.0
            ),
        )
        object.__setattr__(
            self,
            "jitter_radius_px",
            _number(self.jitter_radius_px, "profile.smoothing.jitter_radius_px", minimum=0.0),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "method": self.method.value,
            "responsiveness": self.responsiveness,
            "jitter_radius_px": self.jitter_radius_px,
        }

    @classmethod
    def from_dict(cls, value: object) -> SmoothingConfig:
        data = _mapping(value, "smoothing config")
        return cls(
            method=SmoothingMethod(data.get("method", SmoothingMethod.EMA.value)),
            responsiveness=data.get("responsiveness", 0.35),
            jitter_radius_px=data.get("jitter_radius_px", 8.0),
        )


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """User-owned interaction preferences; calibration persistence follows in M2/M4."""

    profile_id: str
    display_name: str
    language: str = "he"
    confidence: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    gesture_timing: GestureTimingConfig = field(default_factory=GestureTimingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    sensitivity: float = 1.0

    def __post_init__(self) -> None:
        _string(self.profile_id, "profile.profile_id")
        _string(self.display_name, "profile.display_name")
        if self.language not in {"he", "en"}:
            raise ConfigurationError("profile.language must be one of: he, en")
        if not isinstance(self.confidence, ConfidenceThresholds):
            raise ConfigurationError("profile.confidence must be ConfidenceThresholds")
        if not isinstance(self.gesture_timing, GestureTimingConfig):
            raise ConfigurationError("profile.gesture_timing must be GestureTimingConfig")
        if not isinstance(self.smoothing, SmoothingConfig):
            raise ConfigurationError("profile.smoothing must be SmoothingConfig")
        object.__setattr__(
            self, "sensitivity", _number(self.sensitivity, "profile.sensitivity", minimum=0.01)
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "language": self.language,
            "confidence": self.confidence.to_dict(),
            "gesture_timing": self.gesture_timing.to_dict(),
            "smoothing": self.smoothing.to_dict(),
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProfileConfig:
        data = _mapping(value, "profile config")
        return cls(
            profile_id=_required(data, "profile_id"),
            display_name=_required(data, "display_name"),
            language=data.get("language", "he"),
            confidence=ConfidenceThresholds.from_dict(data.get("confidence", {})),
            gesture_timing=GestureTimingConfig.from_dict(data.get("gesture_timing", {})),
            smoothing=SmoothingConfig.from_dict(data.get("smoothing", {})),
            sensitivity=data.get("sensitivity", 1.0),
        )


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Ephemeral session state; it is serializable but not a profile store."""

    active_profile_id: str | None = None
    active_camera_id: str | None = None
    paused: bool = False
    diagnostics_enabled: bool = False

    def __post_init__(self) -> None:
        if self.active_profile_id is not None:
            _string(self.active_profile_id, "session.active_profile_id")
        if self.active_camera_id is not None:
            _string(self.active_camera_id, "session.active_camera_id")
        _bool(self.paused, "session.paused")
        _bool(self.diagnostics_enabled, "session.diagnostics_enabled")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "active_profile_id": self.active_profile_id,
            "active_camera_id": self.active_camera_id,
            "paused": self.paused,
            "diagnostics_enabled": self.diagnostics_enabled,
        }

    @classmethod
    def from_dict(cls, value: object) -> SessionConfig:
        data = _mapping(value, "session config")
        return cls(
            active_profile_id=data.get("active_profile_id"),
            active_camera_id=data.get("active_camera_id"),
            paused=data.get("paused", False),
            diagnostics_enabled=data.get("diagnostics_enabled", False),
        )


@dataclass(frozen=True, slots=True)
class GazeLinkConfig:
    """Top-level configuration with explicit application, device, profile, and session scopes."""

    application: ApplicationConfig
    device: DeviceConfig
    profile: ProfileConfig
    session: SessionConfig = field(default_factory=SessionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.application, ApplicationConfig):
            raise ConfigurationError("application must be an ApplicationConfig")
        if not isinstance(self.device, DeviceConfig):
            raise ConfigurationError("device must be a DeviceConfig")
        if not isinstance(self.profile, ProfileConfig):
            raise ConfigurationError("profile must be a ProfileConfig")
        if not isinstance(self.session, SessionConfig):
            raise ConfigurationError("session must be a SessionConfig")
        if self.session.active_profile_id not in {None, self.profile.profile_id}:
            raise ConfigurationError(
                "session.active_profile_id must match profile.profile_id when set"
            )
        if self.session.active_camera_id not in {None, self.device.camera_id}:
            raise ConfigurationError(
                "session.active_camera_id must match device.camera_id when set"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "application": self.application.to_dict(),
            "device": self.device.to_dict(),
            "profile": self.profile.to_dict(),
            "session": self.session.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> GazeLinkConfig:
        data = _mapping(value, "GAZELINK config")
        return cls(
            application=ApplicationConfig.from_dict(data.get("application", {})),
            device=DeviceConfig.from_dict(data.get("device")),
            profile=ProfileConfig.from_dict(data.get("profile")),
            session=SessionConfig.from_dict(data.get("session", {})),
        )
