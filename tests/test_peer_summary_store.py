"""PeerSummaryStore (InMemory + Postgres) and build_local_summary."""
from pathlib import Path

import pytest

from gin.federation.peer_summary_store import (
    InMemoryPeerSummaryStore,
    PostgresPeerSummaryStore,
    build_local_summary,
)
from gin.federation.schema import PeerSummaryResponse

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "synthetic" / "news_corpus.yaml"


def _summary(node_id="node_c"):
    return PeerSummaryResponse(
        node_id=node_id, embedding_centroid=[0.1, 0.2], distinctive_terms={"x": 1.0}
    )


def test_in_memory_get_missing_is_none():
    store = InMemoryPeerSummaryStore()
    assert store.get("node_c") is None


def test_in_memory_set_then_get_round_trips():
    store = InMemoryPeerSummaryStore()
    store.set("node_c", _summary())
    got = store.get("node_c")
    assert got.node_id == "node_c"
    assert got.distinctive_terms == {"x": 1.0}


def test_in_memory_set_overwrites():
    store = InMemoryPeerSummaryStore()
    store.set("node_c", _summary())
    store.set("node_c", PeerSummaryResponse(node_id="node_c", embedding_centroid=[9.0], distinctive_terms={}))
    assert store.get("node_c").embedding_centroid == [9.0]


@pytest.mark.integration
def test_postgres_set_then_get_round_trips(isolated_db):
    store = PostgresPeerSummaryStore()
    store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[0.5, 0.25], distinctive_terms={"inflation": 2.0}
    ))
    got = store.get("node_c")
    assert got.node_id == "node_c"
    assert got.embedding_centroid == [0.5, 0.25]
    assert got.distinctive_terms == {"inflation": 2.0}
    # upsert replaces
    store.set("node_c", PeerSummaryResponse(node_id="node_c", embedding_centroid=[1.0], distinctive_terms={}))
    assert store.get("node_c").embedding_centroid == [1.0]


@pytest.mark.integration
def test_build_local_summary_over_ingested_corpus(isolated_db, tmp_cold_root):
    from gin.corpus.ingest import ingest_path

    ingest_path(NEWS, embed=True)
    summary = build_local_summary("node_local", top_n=10)
    assert summary.node_id == "node_local"
    assert len(summary.embedding_centroid) == 384
    # centroid is unit-normalized
    norm = sum(x * x for x in summary.embedding_centroid) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert 0 < len(summary.distinctive_terms) <= 10
