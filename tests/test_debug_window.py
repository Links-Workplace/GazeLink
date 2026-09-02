"""Unit tests for the debug window's Qt-free terminal-diagnostics throttle.

Only ``_should_print_diagnostics`` is tested here: it is a pure function with
no Qt dependency, unlike the rest of ``debug_window.py`` which requires a
display server and is verified manually instead (see the M1-T07 completion
report in TASKS.md).
"""

from __future__ import annotations

from gazelink.debug_window import MIN_DIAGNOSTICS_PRINT_INTERVAL_S, _should_print_diagnostics


def test_first_call_with_no_prior_print_always_prints() -> None:
    assert _should_print_diagnostics(None, 100.0, min_interval_s=1.0) is True


def test_call_before_the_interval_elapsed_does_not_print() -> None:
    assert _should_print_diagnostics(100.0, 100.5, min_interval_s=1.0) is False


def test_call_after_the_interval_elapsed_prints_again() -> None:
    assert _should_print_diagnostics(100.0, 101.5, min_interval_s=1.0) is True


def test_call_exactly_at_the_interval_boundary_prints() -> None:
    assert _should_print_diagnostics(100.0, 101.0, min_interval_s=1.0) is True


def test_default_interval_constant_matches_the_one_second_requirement() -> None:
    assert MIN_DIAGNOSTICS_PRINT_INTERVAL_S == 1.0
