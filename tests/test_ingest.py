"""Tests for YAML ingest pipeline."""
from pathlib import Path

import pytest

from gin.corpus import cold, warm
from gin.corpus.db import connect
from gin.corpus.ingest import ingest_path, load_yaml

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "synthetic" / "news_corpus.yaml"


def test_load_yaml_parses_documents_and_edges():
    docs, edges = load_yaml(NEWS)
    assert len(docs) >= 8
    assert any(d.doc_id == "incident_centralwire" for d in docs)
    assert any(e.edge_type.value == "contradicts" for e in edges)


@pytest.mark.integration
def test_ingest_writes_cold_warm_and_is_idempotent(isolated_db, tmp_cold_root):
    stats1 = ingest_path(NEWS, embed=False)
    assert stats1["documents"] >= 1
    assert stats1["chunks"] >= 1
    assert stats1["cold_blobs_written"] >= 1

    with connect() as conn:
        count1 = warm.count_chunks(conn)

    stats2 = ingest_path(NEWS, embed=False)
    assert stats2["cold_blobs_written"] == 0

    with connect() as conn:
        count2 = warm.count_chunks(conn)
        assert count1 == count2
        hits = warm.list_chunks(conn)
        sample = next(h for h in hits if h.chunk_id == "incident_centralwire:0")
    blob = cold.load(sample.content_hash, tmp_cold_root)
    assert sample.text.encode("utf-8") == blob


NODE3_JSON = ROOT / "corpus_node3.json"


def test_load_json_maps_metadata_domain():
    from gin.corpus.ingest import load_json

    docs, _ = load_json(NODE3_JSON)
    assert len(docs) >= 1
    assert all(d.domain == "monetary_policy" for d in docs)


@pytest.mark.integration
def test_ingest_persists_document_domain(isolated_db, tmp_cold_root):
    from gin.corpus.db import connect

    ingest_path(NODE3_JSON, embed=False, ingest_edges=False)
    with connect() as conn:
        row = conn.execute(
            "SELECT DISTINCT domain FROM documents"
        ).fetchall()
    assert row == [("monetary_policy",)]
