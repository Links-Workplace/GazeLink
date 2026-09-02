"""Manual performance-suite entry point without biometric fixtures."""

from __future__ import annotations

import pytest


@pytest.mark.performance
def test_performance_suite_requires_explicit_environment_opt_in() -> None:
    """The plugin skips this test before any benchmark resource can be used."""

    assert True
