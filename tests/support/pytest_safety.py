"""Pytest policy for device-free default execution."""

from __future__ import annotations

import os

import pytest

from support.real_input_guard import assert_no_real_input_modules_loaded

_OPT_IN_ENVIRONMENT = {
    "hardware": "GAZELINK_HARDWARE_TESTS",
    "performance": "GAZELINK_PERFORMANCE_TESTS",
}


def _marker_is_requested(config: pytest.Config, marker: str) -> bool:
    expression = config.option.markexpr
    return marker in expression if expression else False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect device and benchmark tests unless their marker is requested."""

    deselected: list[pytest.Item] = []
    selected: list[pytest.Item] = []
    for item in items:
        markers = {marker.name for marker in item.iter_markers()}
        must_opt_in = markers.intersection(_OPT_IN_ENVIRONMENT)
        if must_opt_in and not any(_marker_is_requested(config, marker) for marker in must_opt_in):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Require a second, explicit environment opt-in before device work runs."""

    for marker, environment_variable in _OPT_IN_ENVIRONMENT.items():
        if item.get_closest_marker(marker) and os.getenv(environment_variable) != "1":
            pytest.skip(f"set {environment_variable}=1 to run {marker} tests")


def pytest_runtest_call() -> None:
    """Catch accidental real-input imports made by a test body."""

    assert_no_real_input_modules_loaded()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Make imports made during fixture teardown visible as a test failure."""

    try:
        assert_no_real_input_modules_loaded()
    except AssertionError as error:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal_reporter is not None:
            terminal_reporter.write_line(str(error), red=True)
