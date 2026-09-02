"""Tests for validated configuration scopes."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from gazelink.config import (
    ApplicationConfig,
    ConfidenceThresholds,
    ConfigurationError,
    DeviceConfig,
    GazeLinkConfig,
    GestureTimingConfig,
    ProfileConfig,
    SessionConfig,
    SmoothingConfig,
)
from gazelink.domain import ScreenGeometry


def _config() -> GazeLinkConfig:
    return GazeLinkConfig(
        application=ApplicationConfig(debug_mode=True),
        device=DeviceConfig(
            camera_id="webcam-0",
            preferred_width_px=1280,
            preferred_height_px=720,
            preferred_fps=30.0,
            screen_geometry=ScreenGeometry("primary", 5120, 1440, 1.25),
        ),
        profile=ProfileConfig(profile_id="ada", display_name="Ada"),
        session=SessionConfig(
            active_profile_id="ada", active_camera_id="webcam-0", diagnostics_enabled=True
        ),
    )


def test_scoped_config_round_trips_through_json() -> None:
    config = _config()

    assert GazeLinkConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: DeviceConfig("webcam", True, 720, 30.0, ScreenGeometry("s", 1, 1, 1.0)), "width"),
        (lambda: ConfidenceThresholds(display=0.8, calibration=0.7, control=0.9), "display"),
        (
            lambda: GestureTimingConfig(natural_blink_max_ms=700, intentional_hold_min_ms=700),
            "exceed",
        ),
        (lambda: SmoothingConfig(responsiveness=True), "responsiveness"),
        (lambda: ProfileConfig(profile_id="p", display_name="P", language="fr"), "language"),
        (
            lambda: GazeLinkConfig(
                application=ApplicationConfig(),
                device=_config().device,
                profile=_config().profile,
                session=SessionConfig(active_profile_id="different"),
            ),
            "match",
        ),
    ],
)
def test_invalid_config_is_rejected_with_clear_errors(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        factory()
