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
    assert len(edges) == 4
    assert all(e.register == "twonode" for e in edges)


def test_default_gold_includes_synthetic_news_corpus_labels():
    """The synthetic news corpus carries four author-labeled contradicts edges
    (incident treatment/arrest counts, election turnout); scan eval must score
    against them — they were previously counted as scan false positives."""
    keys = gold_contradicts_keys()
    assert frozenset({"incident_centralwire:0", "incident_metrodaily:0"}) in keys
    assert frozenset({"incident_centralwire:0", "incident_regionalpost:0"}) in keys
    assert frozenset({"incident_metrodaily:0", "incident_regionalpost:0"}) in keys
    assert frozenset({"election_centralwire:0", "election_metrodaily:0"}) in keys


def test_news_corpus_gold_register_is_news():
    news = ROOT / "data" / "synthetic" / "news_corpus.yaml"
    edges = load_all_gold_contradicts([news])
    assert len(edges) == 4
    assert all(e.register == "news" for e in edges)


def test_gold_edges_carry_relation_class():
    """corpus_edges.yaml pairs are issue_frame (no shared story entities —
    machine-undetectable, 2026-07-12 signal audit); everything unmarked
    defaults to story."""
    corpus_edges = ROOT / "data" / "corpus_edges.yaml"
    edges = load_all_gold_contradicts([corpus_edges])
    assert all(e.relation_class == "issue_frame" for e in edges)

    news = ROOT / "data" / "synthetic" / "news_corpus.yaml"
    assert all(
        e.relation_class == "story" for e in load_all_gold_contradicts([news])
    )


def test_curated_issue_frame_proposals_load_from_yaml():
    """--curated-edges ingests only issue_frame-class contradicts: the story
    class is machine-recoverable and must stay the scan's job."""
    from gin.cartographer.scan import curated_issue_frame_proposals

    corpus_edges = ROOT / "data" / "corpus_edges.yaml"
    fixture = ROOT / "data" / "fixtures" / "disclosure_framing.yaml"
    proposals = curated_issue_frame_proposals([corpus_edges, fixture])
    keys = {frozenset({p.src_chunk_id, p.dst_chunk_id}) for p in proposals}
    assert frozenset({"n1_doc_009:0", "n2_doc_008:2"}) in keys
    assert frozenset({"n1_doc_005:1", "n2_doc_001:1"}) in keys
    assert len(proposals) == 4  # fixture story-class edges are not curated-ingested
    for p in proposals:
        assert p.method == "curated:issue_frame"
        assert p.confidence >= 0.9
