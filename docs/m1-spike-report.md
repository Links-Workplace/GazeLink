# M1-T01 — Spike report

- **Status:** Partial — software/model checks complete; Webcam/live checks unavailable
- **Date:** 2026-09-01
- **Reference environment:** ADR-0001

## Checks completed without a camera

- CPython 3.11.9 environment installed from `requirements.lock`.
- OpenCV 4.11.0, MediaPipe 0.10.35, NumPy 1.26.4 and PySide6 6.11.2 import successfully.
- MediaPipe `FaceLandmarker` exposes VIDEO and LIVE_STREAM modes required by the proposed adapter.
- The official float16 Face Landmarker asset was downloaded to ignored local storage and verified with SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- `FaceLandmarker` initializes with XNNPACK CPU acceleration and processes a synthetic black 640×480 RGB frame.
- The blank-frame result correctly contains zero faces, blendshapes and transformation matrices.
- One observed local run initialized the model in approximately 37ms and processed the blank frame in approximately 5ms. These numbers are diagnostic only and are not a camera or face-performance benchmark.
- Python packages build successfully into both wheel and source distribution.
- Direct dependency licenses were inventoried in `docs/dependency-licenses.md`.

## Licensing/package finding

MediaPipe and OpenCV report Apache-2.0 licensing. NumPy reports BSD-style licensing with bundled component notices. PySide6 reports LGPL-3.0-only or GPL alternatives; proprietary installer compliance requires explicit review before M6. The current approval is for local development and M1 spikes, not commercial distribution.

## Checks still blocked by missing Webcam

- Enumerating and selecting the actual reference camera.
- Verifying supported 1280×720@30FPS mode.
- Two-minute live Face/Eye/Iris tracking run.
- Face-present output and visual landmark alignment.
- Camera capture latency, dropped frames and resource release against a physical device.
- Final Go/No-go for the complete M1 hardware baseline.

## Current conclusion

The software stack and model-loading path are viable enough to continue deterministic contracts, adapters, feature extraction, confidence policy and fake-pipeline work. `M1-T01` cannot be marked complete until a Webcam is connected and the live checks above pass.

