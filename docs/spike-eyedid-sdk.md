# Eyedid SDK — Alternative gaze provider spike

- **Status:** Research complete (public docs only); no License Key obtained, no code written, no hands-on testing performed
- **Date:** 2026-09-02
- **Reference environment:** ADR-0001 (`docs/decisions/0001-technical-baseline.md`)
- **Trigger:** `TASKS.md` open decision `D-02 — Stack`; ADR-0001's own revisit
  triggers ("MediaPipe... fails the M1-T01 spike," "packaging or
  accessibility constraints require a different runtime/UI stack")
- **Scope:** read-only investigation + benchmark methodology design. No
  change to `src/gazelink/`, `pyproject.toml`, or `requirements.lock`. No
  production architecture touched. No verdict — this produces facts and a
  measurement plan for a future, evidence-based `D-02` decision.

## Summary

Eyedid SDK (VisualCamp, formerly SeeSo) is a webcam-based gaze-tracking
SDK with official Windows/C++ support and a plain-RGB camera requirement —
technically compatible with GAZELINK's existing `OpenCVCameraSource`. It
has **no official Python binding**, requires a **mandatory vendor License
Key with a network authentication call**, and its **pricing/commercial
terms are not published** anywhere in public docs — that last point blocks
any real go/no-go decision until the user contacts VisualCamp directly.

## Checks completed (public documentation only)

Source pages fetched directly: `docs.eyedid.ai` quick-start, API docs,
coordinate-conversion, installation-guide, and privacy-policy pages; plus
web search across VisualCamp/GitHub/pricing sources. Each finding below is
what the docs state, not independently verified against a running SDK.

- **Windows integration is real and C++-based.** CMake-based build
  (CMake ≥3.5), Visual Studio 2017+ recommended, ships as `eyedid_core.dll`
  + third-party DLLs that must sit next to the executable (or be copied via
  a CMake custom command). SDK package itself is downloaded from the
  vendor's "SDK Management Console" (`manage.eyedid.ai`), which requires an
  account — **not a public download**.
- **Camera requirement matches GAZELINK's existing hardware.** Plain RGB
  input, contiguous memory, BGR→RGB conversion shown via OpenCV in the
  official sample — no IR/depth camera needed. `OpenCVCameraSource`
  (`src/gazelink/camera.py`) could in principle feed it without new camera
  hardware.
- **No official Python binding exists.** Confirmed via docs (platform list:
  iOS, Android, Unity, Web/JS, Windows-C++) and via web search (no
  `pypi.org` package, no official `ctypes`/`pybind11` wrapper found).
  Three integration paths, each a real tradeoff:
  1. **`ctypes`/`cffi` direct DLL binding** — lowest overhead, but requires
     hand-written struct/callback marshalling for `EyedidGazeData`,
     `EyedidFaceData`, `EyedidBlinkData`, `EyedidUserStatusData`, and the
     calibration progress/point callbacks.
  2. **`pybind11` C++ extension module** — cleaner Python-side ergonomics,
     but introduces a compiled-extension build step that GAZELINK's
     current pure-Python-wheel `pyproject.toml`/`requirements.lock` model
     does not have today (a real process change, not just a dependency
     add).
  3. **Subprocess/IPC shim** (small C++ process, JSON/socket protocol to
     Python) — most isolated and easiest to sandbox/kill cleanly (fits
     GAZELINK's existing "OS-facing code stays behind an adapter, fake-able
     in tests" convention — `CameraSource`/`VisionEngine` `Protocol`
     pattern in `camera.py`/`vision.py`), at the cost of IPC latency and a
     second process to manage — directly relevant to any future latency
     benchmark.
  No recommendation is made between the three here; that depends on actual
  measured latency (see Benchmark design) and on whether VisualCamp's
  terms even permit redistributing/wrapping the DLL this way.
- **Output coordinates are camera-space millimeters, not normalized or
  pixel space — a real integration mismatch.** `GazeInfo`/`EyedidGazeData`
  `x`/`y` fields are documented as "in millimeters, origin same as the
  center of the camera." Converting to display pixels requires
  `CoordConverterV2`, set up via `makeDefaultCameraToDisplayConverter`,
  which needs the **display's physical size in millimeters**
  (`widthMm`/`heightMm`) in addition to pixel dimensions, and **assumes the
  camera sits at the display's top-center** (a documented, unverified-for-
  our-hardware approximation). This is a concrete gap against GAZELINK's
  own `ScreenGeometry` (`domain.py:353-391`), which today carries
  `width_px`/`height_px`/`dpi_scale` but **no physical millimeter
  dimensions** — a prerequisite that doesn't exist in the codebase yet if
  this conversion were ever wired in.
- **Calibration flow is conceptually parallel but not identical to
  GAZELINK's.** `startCalibration(point_num, criteria, roi_bounds,
  use_previous_calibration)` supports **1 or 5** calibration points (not
  9), an accuracy `criteria` (`default`/`low`/`high`), and
  `setCalibrationData()` to reapply a stored calibration without
  recollecting. Whether calibration point *positions* are settable to
  GAZELINK's own 9-target grid (for a fair side-by-side benchmark) was
  **not confirmed** from the pages fetched — needs the fuller API
  reference before the benchmark can run as designed below.
- **Additional per-frame signals beyond GAZELINK's current
  `VisionObservation`:** `movement_state` (fixation/saccade
  classification), `fixation_x/y`, drowsiness and attention scores. Some
  overlap (face box, yaw/pitch/roll, per-eye blink/openness) with what
  `EyeFeatures`/`HeadPose` already provide.
- **Local processing is real, with one documented exception.** The privacy
  policy page states explicitly: *"we DO NOT store any image and video
  from camera, and any gaze data generated by SDK"* — image processing is
  on-device. **However**, `initialize()`'s License Key authentication call
  does transmit to Eyedid's servers: OS/hardware type, a device UUID,
  browser type/version (web builds), and an authentication call counter —
  stored on the vendor's AWS infrastructure for "troubleshooting, usage
  reports, and product statistics." This is a **narrow but real deviation**
  from ADR-0001's "local processing only" baseline: no video/biometric
  data leaves the device, but a network dependency and external telemetry
  call is introduced that GAZELINK's architecture has never needed before.
  Should be an explicit accept/reject decision if this is ever adopted,
  not silently absorbed.
- **Licensing/pricing: not published anywhere checked.** A License Key is
  mandatory; no tier/pricing/free-quota information was found in public
  docs, the pricing page (JS-rendered, no usable content returned), or web
  search. **This is the single largest open blocker** — cost and
  commercial-use terms are unknown, and only the user can resolve this by
  contacting VisualCamp (`contactus@eyedid.ai`) or checking
  `manage.eyedid.ai` directly.
- **Latency/accuracy/FPS: no numbers published anywhere.** Nothing in the
  fetched docs states throughput or accuracy figures. This can only be
  answered by actually running the SDK — see Benchmark design below.

## Explicit non-goals honored this pass

- No `pyproject.toml`/`requirements.lock` change.
- No code added under `src/gazelink/`.
- No License Key requested or VisualCamp account created on the user's
  behalf.
- No verdict on Eyedid vs. MediaPipe — `D-02` remains open.

## Benchmark design (methodology — not executed; blocked on two external prerequisites)

This benchmark cannot run yet, for two independent reasons that are
outside this spike's control:

1. **No working `GazeEstimator` exists yet.** `M2-03 — Gaze Mapping Engine`
   is `⬜ NOT STARTED` in `TASKS.md` — there is nothing on GAZELINK's side
   to compare against yet.
2. **No License Key.** Eyedid cannot be run at all without one, and terms
   are unknown (see above).

The design below is written so it is ready to execute the moment both
exist, using GAZELINK's own conventions rather than inventing parallel
ones:

- **Targets:** the same 9 normalized positions from
  `default_nine_point_targets()` (`src/gazelink/calibration.py`) — both
  systems calibrate against literally the same points, contingent on
  confirming Eyedid's calibration API can be pointed at arbitrary
  positions (open question above).
- **Screen geometry:** the same `ScreenGeometry` capture path already used
  by GAZELINK's calibration window (`calibration_window.py:188-200`), so
  both systems' output converts into the same real screen's pixel space.
  If Eyedid's `CoordConverterV2` is used, its physical-millimeter
  requirement must be captured alongside `ScreenGeometry` — a small,
  explicit addition, not a silent one.
- **Procedure per session:** calibrate GAZELINK's pipeline on the 9
  targets → calibrate Eyedid on the same 9 targets in the same sitting
  (same head position/lighting/camera) → present a set of **held-out
  validation targets** (distinct from the 9 calibration points) to both
  systems → record each system's predicted point per validation target
  with a timestamp.
- **Metrics — identical yardstick for both systems, reusing M2-03's own
  planned metric set:** median error, P95 error, and per-target/region
  error, each in both normalized and pixel space, plus prediction
  latency (wall-clock per estimate call).
- **Isolation:** a standalone script/module outside `src/gazelink/` (e.g.
  `research/eyedid_spike/benchmark.py`) that imports GAZELINK's domain
  types (`ScreenGeometry`, `CalibrationTarget`, error-metric helpers)
  read-only and treats Eyedid as an external black box behind its own thin
  adapter — deletable without touching `src/gazelink/` at all.
- **Report format:** a table per system (median/P95 error, normalized +
  pixel, per-target breakdown, latency percentiles) plus qualitative notes
  (coordinate-conversion correctness on the real ultra-wide 5120×1440
  reference display, calibration UX, error/crash behavior, License Key
  auth friction).

## Risks / open unknowns

- Licensing cost/terms unknown — potential hard blocker independent of
  technical merit.
- Network call at `initialize()` is a real, narrow deviation from "local
  processing only" — needs an explicit decision, not silent adoption.
- No official Python binding means an ongoing maintenance burden for
  whichever bridge is chosen, regardless of Eyedid's raw accuracy.
- Eyedid's calibration point flexibility (arbitrary positions vs. fixed
  1/5-point patterns) is unconfirmed — affects whether the benchmark above
  can truly be apples-to-apples.
- `CoordConverterV2`'s "camera at top-center of display" assumption is
  unverified against GAZELINK's actual reference machine geometry
  (ADR-0001: Samsung 5120×1440 display) and would need GAZELINK to start
  capturing physical screen dimensions in millimeters, which
  `ScreenGeometry` does not do today.
- This research is based on the public docs pages reachable this session,
  not the complete API reference — flagged as incomplete, not exhaustive.

## Current conclusion

Eyedid SDK is technically plausible on Windows with GAZELINK's existing RGB
webcam, but two external prerequisites — real licensing terms from
VisualCamp, and a working GAZELINK `GazeEstimator` to benchmark against —
block any further action. `D-02 — Stack` stays open. Recommended next
step, when the user is ready: contact VisualCamp for licensing terms and a
trial License Key; only then does building the isolated
`research/eyedid_spike/` proof-of-concept and running the benchmark above
become possible.
