"""Tests for hybrid retrieval."""
from pathlib import Path

import pytest

from gin.corpus.ingest import ingest_path
from gin.corpus.retrieve import retrieve

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"


@pytest.mark.integration
def test_retrieve_incident_divergence_pair(isolated_db, tmp_cold_root):
    ingest_path(SYNTHETIC, embed=True)
    hits = retrieve("how many people received treatment at hospitals downtown incident", k=10)
    chunk_ids = {h.chunk_id for h in hits}
    assert "incident_centralwire:0" in chunk_ids
    assert "incident_metrodaily:0" in chunk_ids


@pytest.mark.integration
def test_retrieve_out_of_scope_query_prefers_in_corpus_hits(isolated_db, tmp_cold_root):
    ingest_path(SYNTHETIC, embed=True)
    hits = retrieve("Mars rover sample return launch window", k=5)
    if hits:
        assert hits[0].chunk_id == "out_of_scope_stub:0" or hits[0].eval_layer == "out_of_scope"
