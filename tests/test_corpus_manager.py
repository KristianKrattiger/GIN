from __future__ import annotations

import json
from pathlib import Path

from gin.corpus.corpus_manager import CorpusManager


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ingest_jsonl_and_txt(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_jsonl(
        input_dir / "docs.jsonl",
        [
            {"title": "A", "authors": ["x"], "source": "wire", "body": "alpha body"},
            {"title": "B", "authors": ["y"], "source": "wire", "body": "beta body"},
        ],
    )
    (input_dir / "note.txt").write_text("plain text body", encoding="utf-8")

    manager = CorpusManager(
        store_root=tmp_path / "store",
        manifests_root=tmp_path / "manifests",
    )
    result = manager.ingest_directory(input_dir, file_format="auto")

    assert result.docs_seen == 3
    assert result.manifest.info.version == 1
    assert result.manifest_path.name == "manifest_v1.json"
    assert len(result.manifest.documents) == 3


def test_doc_id_falls_back_to_content_hash(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_jsonl(
        input_dir / "docs.jsonl",
        [{"body": "same text"}, {"body": "same text"}],
    )

    manager = CorpusManager(
        store_root=tmp_path / "store",
        manifests_root=tmp_path / "manifests",
    )
    result = manager.ingest_directory(input_dir, file_format="jsonl")
    doc_ids = {d.doc_id for d in result.manifest.documents}
    assert len(doc_ids) == 1


def test_load_texts_from_manifest_version(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("first text", encoding="utf-8")

    manager = CorpusManager(
        store_root=tmp_path / "store",
        manifests_root=tmp_path / "manifests",
    )
    manager.ingest_directory(input_dir, file_format="txt")

    docs = manager.load_texts_from_manifest(version=1)
    assert len(docs) == 1
    assert docs[0][1] == "first text"
