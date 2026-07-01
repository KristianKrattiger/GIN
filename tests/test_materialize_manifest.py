from __future__ import annotations

from pathlib import Path

from gin.corpus.corpus_manager import CorpusManager
from gin.corpus.materialize import materialize_all


def _stable_tokenize(data: bytes) -> list[int]:
    words = data.decode("utf-8").split()
    return [sum(ord(c) for c in w) % 10007 for w in words]


def test_materialize_all_from_manifest_version(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha beta", encoding="utf-8")
    (input_dir / "b.txt").write_text("gamma delta", encoding="utf-8")

    manager = CorpusManager(
        store_root=tmp_path / "store",
        manifests_root=tmp_path / "manifests",
    )
    result = manager.ingest_directory(input_dir, file_format="txt")
    assert result.manifest.info.version == 1

    # Make materialize_all use the same manifest root.
    from gin.corpus import materialize as materialize_mod

    original = materialize_mod.CorpusManager
    try:
        materialize_mod.CorpusManager = lambda: manager  # type: ignore[assignment]
        corpus = materialize_all(_stable_tokenize, manifest_version=1)
    finally:
        materialize_mod.CorpusManager = original  # type: ignore[assignment]

    assert len(corpus.doc_names) == 2
