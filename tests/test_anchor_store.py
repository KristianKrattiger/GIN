"""PeerAnchorStore implementations: InMemory (unit) and Postgres (integration,
via the isolated_db fixture — same pattern as every other Postgres-backed
test in this repo)."""
from pathlib import Path

import pytest

from gin.federation.anchor_store import (
    InMemoryPeerAnchorStore,
    PostgresPeerAnchorStore,
    local_anchor_rows,
)
from gin.federation.schema import AnchorLeaf

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "synthetic" / "news_corpus.yaml"


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def test_in_memory_store_replace_bucket_and_read_back():
    store = InMemoryPeerAnchorStore()
    store.replace_bucket("node_b", 3, [_leaf("a"), _leaf("b")])
    assert {r.chunk_id for r in store.bucket_rows("node_b", 3)} == {"a", "b"}
    assert {r.chunk_id for r in store.all_rows("node_b")} == {"a", "b"}


def test_in_memory_store_replace_bucket_drops_stale_rows():
    store = InMemoryPeerAnchorStore()
    store.replace_bucket("node_b", 3, [_leaf("a"), _leaf("b")])
    store.replace_bucket("node_b", 3, [_leaf("b", content_hash="2")])
    rows = store.bucket_rows("node_b", 3)
    assert [r.chunk_id for r in rows] == ["b"]
    assert rows[0].content_hash == "2"


def test_in_memory_store_isolates_peers():
    store = InMemoryPeerAnchorStore()
    store.replace_bucket("node_a", 0, [_leaf("x")])
    assert store.all_rows("node_b") == []


@pytest.mark.integration
def test_postgres_store_replace_bucket_and_read_back(isolated_db):
    store = PostgresPeerAnchorStore()
    store.replace_bucket("node_b", 5, [_leaf("a"), _leaf("b")])
    assert {r.chunk_id for r in store.bucket_rows("node_b", 5)} == {"a", "b"}
    store.replace_bucket("node_b", 5, [_leaf("b", content_hash="2")])
    rows = store.bucket_rows("node_b", 5)
    assert [r.chunk_id for r in rows] == ["b"]
    assert rows[0].content_hash == "2"


@pytest.mark.integration
def test_local_anchor_rows_reflects_ingested_corpus(isolated_db, tmp_cold_root):
    from gin.corpus.ingest import ingest_path

    stats = ingest_path(NEWS, embed=False)
    rows = local_anchor_rows()
    assert len(rows) == stats["chunks"]
    sample = next(r for r in rows if r.chunk_id == "incident_centralwire:0")
    assert sample.content_hash
    assert sample.outlet
