from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from scripts.download_models import download_verified, sha256_file


def test_sha256_file_streams_expected_digest(tmp_path: Path) -> None:
    source = tmp_path / "model.task"
    source.write_bytes(b"synthetic-model")

    assert sha256_file(source) == hashlib.sha256(b"synthetic-model").hexdigest()


def test_download_verified_rejects_wrong_digest_without_replacing_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.task"
    destination.write_bytes(b"existing")

    class Response:
        consumed = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            del size
            if self.consumed:
                return b""
            self.consumed = True
            return b"different"

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(ValueError, match="model digest mismatch"):
        download_verified("https://example.invalid/model", "0" * 64, destination)

    assert destination.read_bytes() == b"existing"
