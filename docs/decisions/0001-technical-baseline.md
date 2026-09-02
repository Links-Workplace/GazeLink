# ADR-0001 — Technical baseline and reference environment

- **Status:** Accepted for Foundation; camera portion pending hardware
- **Date:** 2026-09-01
- **Tasks:** F-T01, F-T02

## Context

GAZELINK needs a local Windows computer-vision runtime with deterministic algorithm tests, an accessible desktop UI, and a strictly gated Windows-input adapter. At project bootstrap there is no implementation and no connected Webcam was detected.

## Decision

- Primary development/reference OS: Windows 11 Pro x64.
- Core runtime: Python 3.11.x.
- Camera capture baseline: OpenCV behind `CameraSource`.
- Face/eye/iris landmark baseline: MediaPipe Face Landmarker behind an adapter, subject to M1-T01 live spike.
- Numerical baseline: NumPy.
- Desktop UI baseline: PySide6/Qt behind the UI boundary, subject to an overlay/RTL spike.
- Windows input baseline for M3: adapter over Win32 `SendInput`, with fake input as the default in tests.
- Test baseline: pytest, Ruff and mypy.
- Dependency management: `pyproject.toml` for direct dependency ranges and a generated `requirements.lock` for reproducible environments.
- Display scope through M4: one active display. Multi-monitor support is deferred unless the product scope changes.
- Local processing only. No raw video persistence and no real OS input in Foundation/M1/M2.

## Reference environment observed

| Item | Value |
| --- | --- |
| Computer | Lenovo 12JD001AIV |
| OS | Windows 11 Pro x64, build 26200 |
| CPU | Intel Core i7-13700, 16 cores / 24 logical processors |
| Memory | 33,955,532,800 bytes (approximately 31.6 GiB) |
| GPU | Intel UHD Graphics 770, driver 32.0.101.7079 |
| Active display | Samsung device, 5120×1440 |
| DPI | 120 DPI / 125% scaling |
| Python | CPython 3.11.9 |
| Webcam | Not detected at decision time; reference camera remains pending |

Initial camera target after a Webcam is connected: 1280×720 at 30 FPS. M1-T01 must enumerate the device's actual supported modes and record the selected camera model.

## Consequences

- Python is intentionally constrained to 3.11 for the first lock file and CV compatibility spike.
- The ultra-wide 5120×1440 display is a demanding gaze-calibration target; M2 must report region-level error and not only a global average.
- The missing Webcam prevents closing the hardware portion of F-T01 and prevents M1 live validation, but does not block the safe project scaffold.
- Heavy runtime libraries may be installed before they are imported; the Foundation entry point must remain camera-free and input-free.

## Revisit triggers

- MediaPipe or PySide6 fails the M1-T01 spike on the reference machine.
- A target user requires multiple active displays before M5.
- The selected Webcam cannot sustain the M1 performance targets.
- Packaging or accessibility constraints require a different runtime/UI stack.

