"""Manual hardware-suite entry point.

No test in this file opens a camera yet.  It establishes the opt-in boundary
for future M1 camera validation.
"""

from __future__ import annotations

import pytest


@pytest.mark.hardware
def test_hardware_suite_requires_explicit_environment_opt_in() -> None:
    """The plugin skips this test before any physical resource can be used."""

    assert True
