# Dependency license inventory

This inventory records the direct Foundation/M1 dependencies resolved in `requirements.lock`. It is engineering evidence, not legal advice; redistribution and installer obligations require review before M6.

| Package | Locked version | Reported license | Initial engineering note |
| --- | --- | --- | --- |
| MediaPipe | 0.10.35 | Apache-2.0 | Permissive; preserve notices and review bundled model terms. |
| OpenCV Python | 4.11.0.86 | Apache-2.0 | Permissive; installer must preserve required notices. |
| NumPy | 1.26.4 | BSD-3-Clause plus bundled compatible components | Preserve bundled notices. |
| PySide6 | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | Proprietary distribution needs an explicit LGPL compliance and dynamic-linking review before packaging. |

## Current decision

The dependencies are accepted for local development and M1 spikes. This does not approve a commercial installer. M6 packaging has a mandatory license review, with special attention to PySide6/Qt and every transitive binary component.

## Verification command

```powershell
.\.venv\Scripts\python.exe -m pip show mediapipe opencv-python numpy PySide6
```

