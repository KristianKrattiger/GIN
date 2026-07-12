"""Tests for --no-edges ingest and gold edge loading."""
from pathlib import Path

import yaml

from gin.cartographer.gold_edges import gold_contradicts_keys, load_all_gold_contradicts
from gin.corpus.ingest import ingest_documents, load_yaml
from gin.corpus.models import EdgeDraft, EdgeType

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "disclosure_framing.yaml"


def test_ingest_documents_skips_edges_when_disabled(isolated_db):
    docs, edges = load_yaml(FIXTURE)
    assert edges
    stats = ingest_documents(docs, edges, embed=False, ingest_edges=False)
    assert stats["chunks"] > 0
    assert stats["edges"] == 0


def test_gold_edges_loads_fixture_contradicts():
    keys = gold_contradicts_keys([FIXTURE])
    assert frozenset({"disc_northwind_pr:0", "disc_northwind_complaint:0"}) in keys


def test_gold_edges_includes_corpus_edges_yaml():
    corpus_edges = ROOT / "data" / "corpus_edges.yaml"
    edges = load_all_gold_contradicts([corpus_edges])
    assert len(edges) == 3
    assert all(e.register == "twonode" for e in edges)
