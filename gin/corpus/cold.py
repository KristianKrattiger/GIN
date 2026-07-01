"""Cold tier — content-addressed immutable blob store."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .db import cold_path


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_path(digest: str, root: Path | None = None) -> Path:
    base = root or cold_path()
    return base / digest[:2] / digest


def store(data: bytes, root: Path | None = None) -> tuple[str, bool]:
    digest = content_hash(data)
    path = blob_path(digest, root)
    if path.exists():
        return digest, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest, True


def load(digest: str, root: Path | None = None) -> bytes:
    path = blob_path(digest, root)
    if not path.exists():
        raise FileNotFoundError(f"cold tier missing blob: {digest}")
    return path.read_bytes()


def exists(digest: str, root: Path | None = None) -> bool:
    return blob_path(digest, root).exists()
