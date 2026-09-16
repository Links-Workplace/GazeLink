# ADR-0002 — Modular live architecture over the existing gazefollower path

- **Status:** Accepted (2026-09-15). Supersedes the *Camera capture*, *Face/eye/iris landmark* and *Desktop UI* baselines of ADR-0001 **for the active live path only**. ADR-0001 remains the record for `src/gazelink` and for the reference environment.
- **Tasks:** ARCH-01 (TASKS.md)
- **Spec:** TECHNICAL_SPEC.md v2.0, §3–5, §15–17

## Context

ADR-0001 chose OpenCV behind `CameraSource`, MediaPipe behind an adapter and PySide6 as the UI. Repository reality on 2026-09-15:

- `src/gazelink/` implements that baseline (M1/M2 spike path, its own venv and pytest config). No live control runs through it.
- The path that actually runs calibration, gaze, gestures, dwell, menu, scroll, keyboard and real Windows input is `gazefollower/`. It is executed with the separate `gazefollower_eval` venv (the `gazefollower` library is CC BY-NC-SA and stays outside the repo). Its UI is pygame.
- An architecture review (verified against the code, see TASKS.md ARCH-01) found:
  - `run_live` is a ~1,065-line function whose closures hold all interaction state;
  - `gf_record.py` (3,371 lines) doubles as live infrastructure;
  - frame gating is duplicated;
  - library structures (`face_info`/`gaze_info`) are read everywhere;
  - adapters are called with a constant `armed=True`;
  - cleanup releases input only after ~70 lines of reporting;
  - resources are acquired outside the `try`.

## Decision

1. **Wrap the existing path, do not replace it.** The live engine stays `gazefollower`, and the UI stays pygame. No engine, model, preprocessing, threshold, filter preset, gesture mapping or UI-technology change is part of this refactor.
2. **New core package `gazefollower/gazelink_core/`.** Its sub-packages follow TECHNICAL_SPEC §4.4: `domain`, `tracking`, `gaze`, `calibration`, `interaction`, `platform`, `ui`, `app`.
   - The root is named `gazelink_core`, not `gazelink`. `src/gazelink` is a different, still-tested package, and two importable `gazelink` roots would shadow each other depending on `sys.path`.
   - It lives beside the tools because the tools and the core must run from the same `gazefollower_eval` interpreter.
3. **Tools stay where they are.** The CLI entry points and research scripts (`gf_record`, `gf_fit`, probes, compare, benchmark, practice) remain `gazefollower/gf_*.py`, so every documented command keeps working.
   - Library modules that moved into the core keep their old `gf_*` names as **module aliases**: `sys.modules[__name__]` is bound to the core module. Existing imports and monkeypatch-based tests therefore see one module object, not a copy.
   - Direction rule: tools → core, never core → tools. An import-boundary test enforces this.
4. **Contracts at the adapter boundary.**
   - `GazeFollowerSource` is the only code that touches `face_info`/`gaze_info` or the library's camera. It publishes `FrameObservation`.
   - `ReplaySource` and fake sources publish the same contract.
5. **Existing live/record differences are kept as explicit policies**, not unified:
   - feature dtype: live float64, recorder float32;
   - the recorder's overlay gate also requires a non-idle phase;
   - live preflight checks one model and has no exception guard.

   Unifying any of these is a separate behaviour task.
6. **One owner per state.**
   - `SafetyController`: control mode, face freshness, activation, stopping.
   - `InteractionController`: UI mode, menu, keyboard, scroll, content anchor.
   - `ActionExecutor`: the only caller of the input ports; it passes the real permission, never a constant.
   - `ActionRouter`: stays the single decision policy for requests.
   - `SessionTelemetry`: counters and report.
   - `LiveSession`: composition and lifecycle only.
7. **Lifecycle.** A `CleanupStack` owns ordered, individually guarded, idempotent cleanup. Held input is released first, then the source is stopped, then the display is closed, then the library is shut down, and only then is the report printed. A report failure cannot skip a release.
8. **Seams.** `LiveSession` receives source/display factories, input ports, key reader, screen mapper, model loader and a monotonic `Clock`. Tests inject fakes, with no fallback to real input.

## Consequences

- `gf_setup`'s code fingerprint must include the extracted core modules. Recordings made after this change fingerprint differently by design. Old recordings and models stay readable, because no schema changes.
- `gazefollower_capture.py` is a documented standalone scoring script outside the package and is not migrated.
- `src/gazelink` is untouched. Merging the two trees is future work, not part of ARCH-01.
- ADR-0001 revisit trigger "Packaging or accessibility constraints require a different runtime/UI stack" is recorded as met for the live path: the working stack is gazefollower + pygame.

## Revisit triggers

- A second tracking engine becomes active. It gets its own `TrackingSource` adapter.
- A UI migration to Qt is approved. It replaces the `ui/` display adapter only.
- `src/gazelink` and `gazefollower/` are merged into one distributable package (M6 packaging).
