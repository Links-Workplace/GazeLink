"""Tests for the deterministic fake layer and automated-test safety policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from support.fakes import FakeCamera, FakeClock, FakeInput, FakeInputEvent
from support.real_input_guard import (
    assert_no_real_input_modules_loaded,
    is_real_input_module_name,
    module_for_guard_test,
)


@pytest.mark.unit
def test_fake_clock_is_monotonic_and_deterministic() -> None:
    clock = FakeClock(initial_seconds=1.5)

    assert clock.now() == 1.5
    assert clock.advance(0.25) == 1.75
    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance(-0.01)


@pytest.mark.unit
def test_fake_camera_is_memory_only_and_requires_open() -> None:
    camera = FakeCamera([{"synthetic": True}, {"synthetic": True}])

    with pytest.raises(RuntimeError, match="must be opened"):
        camera.read()
    camera.open()
    assert camera.read() == {"synthetic": True}
    assert camera.read() == {"synthetic": True}
    assert camera.read() is None
    camera.close()

    assert camera.is_open is False
    assert camera.open_count == 1
    assert camera.close_count == 1


@pytest.mark.unit
def test_fake_input_only_records_immutable_in_memory_events() -> None:
    fake_input = FakeInput()

    fake_input.emit("move", 200, 100)
    fake_input.emit("click", "left")

    assert fake_input.events == (
        FakeInputEvent(name="move", arguments=(200, 100)),
        FakeInputEvent(name="click", arguments=("left",)),
    )
    fake_input.clear()
    assert len(fake_input.events) == 0


@pytest.mark.unit
def test_real_input_guard_identifies_and_rejects_loaded_adapter() -> None:
    module_name = "gazelink.adapters.real_windows_input"
    assert is_real_input_module_name(module_name) is True
    assert is_real_input_module_name("gazelink.adapters.fake_input") is False

    sys.modules[module_name] = module_for_guard_test(module_name)
    try:
        with pytest.raises(AssertionError, match="real OS-input adapter"):
            assert_no_real_input_modules_loaded()
    finally:
        del sys.modules[module_name]


@pytest.mark.unit
def test_current_default_suite_has_not_loaded_real_os_input() -> None:
    assert_no_real_input_modules_loaded()


@pytest.mark.unit
def test_ci_workflow_does_not_upload_test_artifacts() -> None:
    workflow = Path(".github/workflows/quality.yml").read_text(encoding="utf-8")

    assert "upload-artifact" not in workflow
