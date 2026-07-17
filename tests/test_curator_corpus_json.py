"""Corpus JSON loading + chunk-id normalization to {doc_id}:{position}."""
import json

import pytest

from gin.curator.corpus_json import load_corpus_chunks

FIXTURE = {
    "node_id": "node_1_institutional",
    "documents": [
        {"doc_id": "n1_doc_005", "chunks": [
            {"chunk_id": "n1_doc_005_c000", "position": 0, "text": "alpha text"},
            {"chunk_id": "n1_doc_005_c002", "position": 2, "text": "gamma text"},
        ]},
        {"doc_id": "n1_doc_008", "chunks": [
            {"chunk_id": "n1_doc_008_c000", "position": 0, "text": "delta text"},
        ]},
    ],
}


def _write(tmp_path, data, name="corpus.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_flattens_and_normalizes_chunk_ids(tmp_path):
    chunks = load_corpus_chunks([_write(tmp_path, FIXTURE)])
    ids = [c.chunk_id for c in chunks]
    assert ids == ["n1_doc_005:0", "n1_doc_005:2", "n1_doc_008:0"]
    assert chunks[1].text == "gamma text"


def test_dedupes_by_chunk_id_first_wins(tmp_path):
    dup = {"documents": [
        {"doc_id": "d", "chunks": [
            {"position": 0, "text": "first"},
            {"position": 0, "text": "second"},
        ]},
    ]}
    chunks = load_corpus_chunks([_write(tmp_path, dup)])
    assert [c.chunk_id for c in chunks] == ["d:0"]
    assert chunks[0].text == "first"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus_chunks([tmp_path / "nope.json"])


def test_missing_position_raises(tmp_path):
    bad = {"documents": [{"doc_id": "d", "chunks": [{"text": "no position"}]}]}
    with pytest.raises(ValueError, match="position"):
        load_corpus_chunks([_write(tmp_path, bad)])


def test_missing_doc_id_raises(tmp_path):
    bad = {"documents": [{"chunks": [{"position": 0, "text": "x"}]}]}
    with pytest.raises(ValueError, match="doc_id"):
        load_corpus_chunks([_write(tmp_path, bad)])


def test_non_dict_top_level_raises(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_corpus_chunks([p])
