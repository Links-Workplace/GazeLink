# Runtime models

Model binaries are not committed to the repository. Download the pinned Face Landmarker asset with:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

The script downloads from Google's official MediaPipe model storage, verifies SHA-256, and writes atomically to `.gazelink/models/face_landmarker.task`.

Pinned asset:

- MediaPipe Face Landmarker float16
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- Observed size: 3,758,596 bytes

If the upstream model changes, do not update the digest blindly. Re-run M1 model validation, record the new asset and licensing evidence, and review the resulting behavior before accepting it.

