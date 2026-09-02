"""Download and verify local MediaPipe model assets.

Model binaries are runtime assets stored below ``.gazelink/models`` and are not
committed. The pinned digest prevents the upstream ``latest`` alias from changing
silently underneath a reproducible development environment.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, expected_sha256: str, destination: Path) -> Path:
    """Atomically replace ``destination`` only after its digest is verified."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)

    try:
        with (
            urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)

        actual_sha256 = sha256_file(temporary_path)
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                f"model digest mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Download verified GAZELINK model assets")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(".gazelink/models/face_landmarker.task"),
    )
    args = parser.parse_args()
    result = download_verified(
        FACE_LANDMARKER_URL,
        FACE_LANDMARKER_SHA256,
        args.destination,
    )
    print(f"verified model: {result} ({sha256_file(result)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
