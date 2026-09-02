# Quality gates

## Default automated gate

The default suite is device-free and uses only synthetic, in-memory data. It
must not open a webcam, store a frame, or send operating-system input.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m build
```

The same local gate can be run as one fail-fast command:

```powershell
.\scripts\run_quality.ps1
```

`tests/support/fakes.py` supplies `FakeClock`, `FakeCamera`, and `FakeInput`.
They keep queues, synthetic frame objects, and input event logs only in memory.
`tests/support/pytest_safety.py` deselects `hardware` and `performance` tests
unless their marker is explicitly requested, and rejects a loaded real
GAZELINK OS-input adapter before and after test execution.

## Manual suites

Hardware and performance/accuracy validation are never part of regular CI.
They require both an explicit marker and an explicit environment opt-in:

```powershell
$env:GAZELINK_HARDWARE_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -m hardware

$env:GAZELINK_PERFORMANCE_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -m performance
```

Run hardware tests only with the tester present and with real OS input disabled
unless the assigned manual protocol says otherwise. Do not upload camera frames,
screenshots, face images, landmarks, calibration samples, or user data as test
artifacts. Record only approved aggregate measurements and non-sensitive logs.

## CI merge gate

The Windows workflow runs the default test suite, Ruff linting, format check,
mypy, and a source/wheel build on a clean checkout. It does not use an
`upload-artifact` step; therefore it cannot publish frames or user data. Build
outputs are only verified in the ephemeral CI workspace.

Merge/release is blocked when any default test, lint/format, type-check, or
build command fails. Hardware, performance, and accuracy evidence remains a
manual milestone gate and must be documented separately after a consent-safe
session.
