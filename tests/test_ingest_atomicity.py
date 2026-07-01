from __future__ import annotations

from pathlib import Path

import pytest

from gin.corpus.corpus_manager import CorpusManager, FileStoreBackend
from gin.corpus.manifest import latest_manifest


class FailingBackend(FileStoreBackend):
    def __init__(self, store_root: Path):
        super().__init__(store_root)
        self.calls = 0

    def commit_staged_blob(self, staged_path: Path, content_hash: str) -> tuple[Path, bool]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("injected failure")
        return super().commit_staged_blob(staged_path, content_hash)


def test_ingest_failure_does_not_publish_manifest(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (input_dir / "b.txt").write_text("beta", encoding="utf-8")

    manager = CorpusManager(
        store_root=tmp_path / "store",
        manifests_root=tmp_path / "manifests",
        backend=FailingBackend(tmp_path / "store"),
    )
    with pytest.raises(RuntimeError):
        manager.ingest_directory(input_dir, file_format="txt")

    assert latest_manifest(tmp_path / "manifests") is None


def test_successful_ingest_publishes_single_manifest(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")

    manager = CorpusManager(
        store_root=tmp_path / "store",
        manifests_root=tmp_path / "manifests",
    )
    result = manager.ingest_directory(input_dir, file_format="txt")

    latest = latest_manifest(tmp_path / "manifests")
    assert latest is not None
    assert latest.name == "manifest_v1.json"
    assert result.manifest.info.version == 1
