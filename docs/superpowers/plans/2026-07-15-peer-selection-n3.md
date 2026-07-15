# Peer Selection at N>2 Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a federation node can't ground a query, let it pick the right peer among several to delegate to — on the first try — using a dense+sparse RRF-fused routing summary synced from each peer, never their corpus text.

**Architecture:** Extends `gin/federation/` (built across the two prior sub-projects). Each node exposes a `GET /v1/federated/summary` endpoint returning a per-node routing signal (embedding centroid + top-N distinctive IDF terms). Peers' summaries are cached (piggybacking the existing anchor-sync loop's root-mismatch detection). A new pure-logic `peer_selection.py` ranks peers by RRF-fusing dense (query-embedding vs. centroid) and sparse (query-keyword vs. distinctive-terms) rankings, reusing the retrieval stack's own `RRF_K`. The router tries peers in ranked order, falling back to the next on refusal, every request still `hop_count=1`.

**Tech Stack:** Python 3.12, FastAPI/Starlette, httpx, Pydantic, psycopg3/Postgres+pgvector, sentence-transformers (all-MiniLM-L6-v2, 384-dim), pytest, uvicorn.

## Global Constraints

- Chunk **text** never crosses the wire — the summary carries only an embedding centroid (`list[float]`, 384-dim) and a distinctive-terms map (`dict[str, float]`, token→IDF). Same right-to-opacity invariant as the prior two sub-projects.
- Peer selection is **content-similarity only** — no trust weights, no persistent reputation. Ranking is recomputed from the current synced summary every query.
- Loop prevention stays structural: every delegated request carries `hop_count=1`, no node ever re-delegates. Multi-peer fallback means A may make more than one hop-1 request sequentially, nothing more.
- `RRF_K` is imported from `gin.corpus.retrieve` (value 60), never redefined — one source of truth with hybrid retrieval.
- Peers with no cached summary are ranked **last, in `config.peers` order — never excluded**. With zero summaries anywhere, ranking degrades exactly to v1's `config.peers` order.
- All new `create_app` / `run_forever` / `answer_or_delegate` parameters are additive with defaults that reproduce prior behavior exactly. Existing federation + anchor-sync tests must keep passing unmodified in behavior.
- DB-touching tests use the existing `isolated_db` / `require_postgres` fixtures (`tests/conftest.py`), marked `@pytest.mark.integration`.
- Follow the `gin/corpus/db.py` convention: fresh Postgres connection per call (`connect()` / `transaction()`), never held across a background task.

---

### Task 1: Wire schema — PeerSummaryResponse + peers_attempted

**Files:**
- Modify: `gin/federation/schema.py`
- Test: `tests/test_federation_schema.py`

**Interfaces:**
- Consumes: `PROTOCOL_VERSION`, `FederationLayer` (existing, `gin/federation/schema.py`).
- Produces: `PeerSummaryResponse(protocol_version, node_id, embedding_centroid: list[float], distinctive_terms: dict[str, float])`; `FederationLayer` gains `peers_attempted: list[str]` (default empty). Consumed by every later task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_federation_schema.py`:

```python
from gin.federation.schema import PeerSummaryResponse


def test_peer_summary_response_round_trip():
    resp = PeerSummaryResponse(
        node_id="node_c",
        embedding_centroid=[0.1, 0.2, 0.3],
        distinctive_terms={"inflation": 2.1, "reserve": 1.8},
    )
    again = PeerSummaryResponse.model_validate(resp.model_dump())
    assert again == resp
    assert again.protocol_version == PROTOCOL_VERSION


def test_peer_summary_defaults_empty_collections():
    resp = PeerSummaryResponse(node_id="node_c")
    assert resp.embedding_centroid == []
    assert resp.distinctive_terms == {}


def test_federation_layer_peers_attempted_defaults_empty():
    layer = FederationLayer(answered_by="node_b", hop_count=1, request_id="r")
    assert layer.peers_attempted == []


def test_federation_layer_carries_peers_attempted():
    layer = FederationLayer(
        answered_by="node_c", hop_count=1, request_id="r",
        peers_attempted=["node_b", "node_c"],
    )
    again = FederationLayer.model_validate(layer.model_dump())
    assert again.peers_attempted == ["node_b", "node_c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'PeerSummaryResponse'`.

- [ ] **Step 3: Add the model and the field**

In `gin/federation/schema.py`, add `peers_attempted` to `FederationLayer` (after its existing `request_id` field):

```python
class FederationLayer(BaseModel):
    """Provenance extension: how a delegated answer reached the caller."""

    answered_by: str
    hop_count: int
    transport: str = "http"
    peer_url: str = ""
    request_id: str
    # Ordered node_ids A actually contacted for this query (v1: one peer).
    peers_attempted: list[str] = Field(default_factory=list)
```

Append the new response model near the other wire messages (after `FederatedResponse`):

```python
class PeerSummaryResponse(BaseModel):
    """A node's routing signal: an embedding centroid + distinctive IDF terms.
    Chunk text never appears here — only these aggregate statistics."""

    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    embedding_centroid: list[float] = Field(default_factory=list)
    distinctive_terms: dict[str, float] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: PASS, all tests including pre-existing.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/schema.py tests/test_federation_schema.py
git commit -m "Peer summary wire schema + peers_attempted provenance; never chunk text."
```

---

### Task 2: Pure peer-ranking logic

**Files:**
- Create: `gin/federation/peer_selection.py`
- Test: `tests/test_peer_selection.py`

**Interfaces:**
- Consumes: `PeerSummaryResponse` (Task 1); `RRF_K` from `gin.corpus.retrieve`.
- Produces: `cosine(a, b) -> float`; `dense_rank(query_embedding, centroids: dict[str, list[float]]) -> list[str]`; `sparse_rank(query_keywords: set[str], term_maps: dict[str, dict[str, float]]) -> list[str]`; `rank_peers(query_embedding, query_keywords, summaries: dict[str, PeerSummaryResponse], peer_order: list[str]) -> list[str]`. Consumed by Tasks 6, 7, 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_peer_selection.py`:

```python
"""Peer ranking: dense + sparse RRF fusion, no-summary peers sort last,
deterministic and independent of input order."""
from gin.federation.peer_selection import cosine, dense_rank, rank_peers, sparse_rank
from gin.federation.schema import PeerSummaryResponse


def _summary(node_id, centroid, terms):
    return PeerSummaryResponse(
        node_id=node_id, embedding_centroid=centroid, distinctive_terms=terms
    )


def test_cosine_orthogonal_and_parallel():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0  # zero vector safe


def test_dense_rank_orders_by_cosine_desc():
    order = dense_rank(
        [1.0, 0.0],
        {"b": [0.0, 1.0], "c": [0.9, 0.1]},
    )
    assert order == ["c", "b"]


def test_sparse_rank_orders_by_matched_idf_mass_desc():
    order = sparse_rank(
        {"inflation", "reserve"},
        {"b": {"landback": 3.0}, "c": {"inflation": 2.0, "reserve": 1.5}},
    )
    assert order == ["c", "b"]


def test_rank_peers_agreeing_signals():
    summaries = {
        "node_b": _summary("node_b", [0.0, 1.0], {"landback": 3.0, "indigenous": 2.5}),
        "node_c": _summary("node_c", [1.0, 0.0], {"inflation": 2.0, "reserve": 1.8}),
    }
    order = rank_peers(
        [1.0, 0.0], {"inflation", "reserve"}, summaries, ["node_b", "node_c"]
    )
    assert order[0] == "node_c"


def test_rank_peers_no_summary_sorts_last_in_config_order():
    summaries = {
        "node_c": _summary("node_c", [1.0, 0.0], {"inflation": 2.0}),
    }
    # node_b has no summary; must appear after node_c, never dropped.
    order = rank_peers(
        [1.0, 0.0], {"inflation"}, summaries, ["node_b", "node_c"]
    )
    assert order == ["node_c", "node_b"]


def test_rank_peers_empty_summaries_is_config_order():
    order = rank_peers([1.0, 0.0], {"x"}, {}, ["node_b", "node_c"])
    assert order == ["node_b", "node_c"]


def test_rank_peers_deterministic_under_input_reorder():
    summaries = {
        "node_b": _summary("node_b", [0.2, 0.9], {"justice": 2.0}),
        "node_c": _summary("node_c", [0.9, 0.2], {"inflation": 2.0}),
    }
    a = rank_peers([0.9, 0.2], {"inflation"}, summaries, ["node_b", "node_c"])
    b = rank_peers([0.9, 0.2], {"inflation"}, dict(reversed(list(summaries.items()))), ["node_b", "node_c"])
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.peer_selection'`.

- [ ] **Step 3: Write `gin/federation/peer_selection.py`**

```python
"""Pure peer-ranking logic — no I/O, no network, no DB.

Mirrors the retrieval tier's hybrid fusion (gin/corpus/retrieve.py) one level
up: rank peers by dense similarity (query embedding vs. each peer's centroid)
and by sparse overlap (query keywords vs. each peer's distinctive IDF terms),
then RRF-fuse the two rankings with the same RRF_K. Peers without a cached
summary are appended last in config order, never dropped.
"""
from __future__ import annotations

import math

from gin.corpus.retrieve import RRF_K

from .schema import PeerSummaryResponse


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dense_rank(
    query_embedding: list[float], centroids: dict[str, list[float]]
) -> list[str]:
    """Node ids by descending cosine similarity to the query (tiebreak: id)."""
    scored = [
        (cosine(query_embedding, centroid), node_id)
        for node_id, centroid in centroids.items()
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [node_id for _, node_id in scored]


def sparse_rank(
    query_keywords: set[str], term_maps: dict[str, dict[str, float]]
) -> list[str]:
    """Node ids by descending matched-IDF mass (tiebreak: id)."""
    scored = []
    for node_id, terms in term_maps.items():
        mass = sum(terms.get(kw, 0.0) for kw in query_keywords)
        scored.append((mass, node_id))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [node_id for _, node_id in scored]


def rank_peers(
    query_embedding: list[float],
    query_keywords: set[str],
    summaries: dict[str, PeerSummaryResponse],
    peer_order: list[str],
) -> list[str]:
    """Full ranked peer order: RRF-fused for peers with a summary, then any
    remaining peers appended in config order."""
    centroids = {nid: s.embedding_centroid for nid, s in summaries.items()}
    term_maps = {nid: s.distinctive_terms for nid, s in summaries.items()}
    d_rank = {nid: i for i, nid in enumerate(dense_rank(query_embedding, centroids), start=1)}
    s_rank = {nid: i for i, nid in enumerate(sparse_rank(query_keywords, term_maps), start=1)}

    fused = []
    for nid in summaries:
        score = 1.0 / (RRF_K + d_rank[nid]) + 1.0 / (RRF_K + s_rank[nid])
        fused.append((score, nid))
    fused.sort(key=lambda t: (-t[0], t[1]))
    ranked = [nid for _, nid in fused]

    # Peers without a summary: appended in config order, never dropped.
    ranked += [nid for nid in peer_order if nid not in summaries]
    return ranked
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_selection.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/peer_selection.py tests/test_peer_selection.py
git commit -m "Peer ranking: dense+sparse RRF fusion, reusing retrieval's RRF_K; no-summary peers last."
```

---

### Task 3: peer_summaries table + PeerSummaryStore + build_local_summary

**Files:**
- Modify: `docker/init-db.sql`
- Create: `gin/federation/peer_summary_store.py`
- Test: `tests/test_peer_summary_store.py`

**Interfaces:**
- Consumes: `PeerSummaryResponse` (Task 1); `connect`/`transaction` from `gin.corpus.db`; `hot.embed_texts`, `relevance.corpus_idf` (existing); `isolated_db`/`tmp_cold_root` fixtures; `ingest_path` from `gin.corpus.ingest`.
- Produces: `PeerSummaryStore` Protocol (`get(peer_node_id) -> Optional[PeerSummaryResponse]`, `set(peer_node_id, summary) -> None`); `InMemoryPeerSummaryStore`; `PostgresPeerSummaryStore`; `build_local_summary(node_id, top_n=40, conn=None) -> PeerSummaryResponse`. Consumed by Tasks 5, 6, 10.

- [ ] **Step 1: Add the `peer_summaries` table to `docker/init-db.sql`**

Append to the end of `docker/init-db.sql`:

```sql
CREATE TABLE IF NOT EXISTS peer_summaries (
    peer_node_id       TEXT PRIMARY KEY,
    embedding_centroid REAL[] NOT NULL,
    distinctive_terms  JSONB NOT NULL,
    synced_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

This stores THIS node's cached copy of each PEER's routing summary — never
its chunk text.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_peer_summary_store.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_summary_store.py -v -m "not integration"`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.peer_summary_store'`.

- [ ] **Step 4: Write `gin/federation/peer_summary_store.py`**

```python
"""Where each node caches its peers' routing summaries, and how a node builds
its OWN summary to serve.

build_local_summary reads this node's chunk texts, computes a unit-normalized
mean embedding (the centroid) and the top-N distinctive IDF terms. Chunk text
is used only to derive these aggregates — it never leaves the node.
"""
from __future__ import annotations

import json
from typing import Optional, Protocol, runtime_checkable

import psycopg

from gin.corpus.db import connect, transaction
from gin.corpus.hot import EMBEDDING_DIM, embed_texts
from gin.corpus.relevance import corpus_idf

from .schema import PeerSummaryResponse


@runtime_checkable
class PeerSummaryStore(Protocol):
    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]: ...
    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None: ...


class InMemoryPeerSummaryStore:
    def __init__(self) -> None:
        self._data: dict[str, PeerSummaryResponse] = {}

    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]:
        return self._data.get(peer_node_id)

    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None:
        self._data[peer_node_id] = summary


class PostgresPeerSummaryStore:
    """Fresh connection per call, matching the corpus tier's convention."""

    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]:
        with connect() as conn:
            row = conn.execute(
                "SELECT embedding_centroid, distinctive_terms FROM peer_summaries "
                "WHERE peer_node_id = %s",
                (peer_node_id,),
            ).fetchone()
        if row is None:
            return None
        terms = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        return PeerSummaryResponse(
            node_id=peer_node_id,
            embedding_centroid=[float(x) for x in row[0]],
            distinctive_terms=terms,
        )

    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO peer_summaries "
                "(peer_node_id, embedding_centroid, distinctive_terms) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (peer_node_id) DO UPDATE SET "
                "embedding_centroid = EXCLUDED.embedding_centroid, "
                "distinctive_terms = EXCLUDED.distinctive_terms, "
                "synced_at = NOW()",
                (
                    peer_node_id,
                    list(summary.embedding_centroid),
                    json.dumps(summary.distinctive_terms),
                ),
            )


def _unit_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return [0.0] * EMBEDDING_DIM
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    norm = sum(x * x for x in mean) ** 0.5
    if norm == 0.0:
        return mean
    return [x / norm for x in mean]


def build_local_summary(
    node_id: str, top_n: int = 40, conn: Optional[psycopg.Connection] = None
) -> PeerSummaryResponse:
    """This node's routing summary: unit-mean chunk embedding + top-N IDF terms."""
    if conn is None:
        with connect() as conn:
            return build_local_summary(node_id, top_n, conn)
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks").fetchall()]
    centroid = _unit_mean(embed_texts(texts)) if texts else [0.0] * EMBEDDING_DIM
    idf = corpus_idf(texts)
    top = dict(sorted(idf.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n])
    return PeerSummaryResponse(
        node_id=node_id, embedding_centroid=centroid, distinctive_terms=top
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_summary_store.py -v -m "not integration"`
Expected: PASS (3 in-memory tests).

Run (Postgres up in `docker/`): `./venv/Scripts/python.exe -m pytest tests/test_peer_summary_store.py -v`
Expected: PASS (5 tests), or the 2 integration tests SKIP if Postgres isn't running.

- [ ] **Step 6: Commit**

```bash
git add docker/init-db.sql gin/federation/peer_summary_store.py tests/test_peer_summary_store.py
git commit -m "peer_summaries table + PeerSummaryStore + build_local_summary (centroid + top IDF terms)."
```

---

### Task 4: PeerClient.get_summary

**Files:**
- Modify: `gin/federation/client.py`
- Test: `tests/test_federation_client.py`

**Interfaces:**
- Consumes: `PeerSummaryResponse` (Task 1); existing `HttpPeerClient._get`, `PeerConfig`, `PeerUnreachable`.
- Produces: `PeerClient.get_summary(peer) -> PeerSummaryResponse`, raising `PeerUnreachable` on transport failure. Consumed by Tasks 6, 8, 10.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_federation_client.py`:

```python
from gin.federation.schema import PeerSummaryResponse


def test_get_summary_parses_and_hits_summary_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = PeerSummaryResponse(
            node_id="node_c", embedding_centroid=[0.1, 0.2],
            distinctive_terms={"inflation": 2.0},
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient("s3cret", transport=httpx.MockTransport(handler))
    out = client.get_summary(PEER)
    assert out.node_id == "node_c"
    assert out.distinctive_terms == {"inflation": 2.0}
    assert seen["url"] == "http://peer-b/v1/federated/summary"
    assert seen["auth"] == "Bearer s3cret"


def test_get_summary_http_error_maps_to_peer_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.get_summary(PEER)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: FAIL — `AttributeError: 'HttpPeerClient' object has no attribute 'get_summary'`.

- [ ] **Step 3: Extend `gin/federation/client.py`**

Add `PeerSummaryResponse` to the existing schema import block:

```python
from .schema import (
    AnchorBucketsResponse,
    AnchorLeavesResponse,
    AnchorRootResponse,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
)
```

Add to the `PeerClient` Protocol (after the existing anchor method signatures):

```python
    def get_summary(self, peer: PeerConfig) -> PeerSummaryResponse: ...
```

Add to `HttpPeerClient` (after `get_anchor_bucket`):

```python
    def get_summary(self, peer: PeerConfig) -> PeerSummaryResponse:
        return self._get(peer, "/v1/federated/summary", PeerSummaryResponse)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: PASS, all tests including pre-existing.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/client.py tests/test_federation_client.py
git commit -m "PeerClient.get_summary: fetch a peer's routing summary over the wire."
```

---

### Task 5: /v1/federated/summary endpoint

**Files:**
- Modify: `gin/federation/server.py`
- Test: `tests/test_summary_endpoint.py`

**Interfaces:**
- Consumes: `PeerSummaryResponse` (Task 1); existing `create_app` signature, `_check_auth`.
- Produces: `create_app(..., local_summary: Optional[Callable[[], PeerSummaryResponse]] = None)` — new keyword-only param, default `None` (empty-summary callable); new route `GET /v1/federated/summary` behind the existing auth dependency. Consumed by Tasks 6, 8, 10.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_summary_endpoint.py`:

```python
"""The /v1/federated/summary endpoint: auth-gated, injected summary callable."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import PeerSummaryResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_c", host="127.0.0.1", port=8473,
    database_url="postgresql://x/gin_node_c", cold_path="data/cold_node_c",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    shared_secret="s3cret", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)
AUTH = {"Authorization": "Bearer s3cret"}


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _summary() -> PeerSummaryResponse:
    return PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[0.1, 0.2, 0.3],
        distinctive_terms={"inflation": 2.0},
    )


def test_summary_endpoint_returns_injected_summary():
    app = create_app(CFG, answer_fn=_grounded, local_summary=_summary)
    client = TestClient(app)
    r = client.get("/v1/federated/summary", headers=AUTH)
    resp = PeerSummaryResponse.model_validate(r.json())
    assert resp.node_id == "node_c"
    assert resp.distinctive_terms == {"inflation": 2.0}


def test_summary_endpoint_requires_auth():
    app = create_app(CFG, answer_fn=_grounded, local_summary=_summary)
    client = TestClient(app)
    assert client.get("/v1/federated/summary").status_code == 401


def test_summary_endpoint_default_is_empty():
    app = create_app(CFG, answer_fn=_grounded)  # no local_summary injected
    client = TestClient(app)
    r = client.get("/v1/federated/summary", headers=AUTH)
    resp = PeerSummaryResponse.model_validate(r.json())
    assert resp.node_id == "node_c"
    assert resp.embedding_centroid == []
    assert resp.distinctive_terms == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_summary_endpoint.py -v`
Expected: FAIL — `TypeError: create_app() got an unexpected keyword argument 'local_summary'`.

- [ ] **Step 3: Modify `gin/federation/server.py`**

Add `PeerSummaryResponse` to the schema import block:

```python
from .schema import (
    PROTOCOL_VERSION,
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
)
```

Add the `local_summary` parameter to `create_app` (after `peer_anchor_store`):

```python
def create_app(
    config: NodeConfig,
    *,
    answer_fn: AnswerFn,
    peer_client: Optional[PeerClient] = None,
    corpus_fingerprint: Optional[dict] = None,
    local_anchor_rows: Optional[Callable[[], list[AnchorLeaf]]] = None,
    peer_anchor_store: Optional[PeerAnchorStore] = None,
    local_summary: Optional[Callable[[], PeerSummaryResponse]] = None,
) -> FastAPI:
```

Right after the existing `anchor_rows_fn = ...` line, add:

```python
    summary_fn = local_summary or (lambda: PeerSummaryResponse(node_id=config.node_id))
```

Add the route immediately before the final `return app` (after `anchors_sync_stats`):

```python
    @app.get("/v1/federated/summary", response_model=PeerSummaryResponse)
    def federated_summary(_: None = Depends(_check_auth)) -> PeerSummaryResponse:
        return summary_fn()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_summary_endpoint.py -v`
Expected: PASS, all 3 tests.

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py tests/test_anchor_endpoints.py tests/test_federation_loop.py -v`
Expected: PASS, unchanged — confirms additive.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/server.py tests/test_summary_endpoint.py
git commit -m "GET /v1/federated/summary: serve this node's routing summary, same auth as other routes."
```

---

### Task 6: Sync summaries in the background loop + create_app wiring

**Files:**
- Modify: `gin/federation/anchor_sync.py`
- Modify: `gin/federation/server.py`
- Test: `tests/test_anchor_sync.py`, `tests/test_summary_endpoint.py`

**Interfaces:**
- Consumes: `PeerSummaryStore` (Task 3); `sync_once`/`run_forever` (existing); `PeerClient.get_summary` (Task 4).
- Produces: `run_forever(..., summary_store: Optional[PeerSummaryStore] = None)` — when present and a cycle's root did NOT match, refetch and store the peer's summary; `create_app(..., peer_summary_store: Optional[PeerSummaryStore] = None)` — passed into `run_forever` (lifespan) and used to build the query-time peer ranker. Consumed by Tasks 7, 9, 10.

- [ ] **Step 1: Write the failing test (summary sync in run_forever)**

Append to `tests/test_anchor_sync.py`:

```python
import asyncio

from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_sync import run_forever
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import AnchorSyncStats, PeerSummaryResponse


def test_run_forever_fetches_summary_on_root_mismatch():
    rows = _corpus(20)
    client = FakePeerClient(rows)
    client.summary = PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[1.0, 0.0], distinctive_terms={"x": 1.0}
    )
    anchor_store = InMemoryPeerAnchorStore()  # empty -> first cycle mismatches
    summary_store = InMemoryPeerSummaryStore()
    stats = AnchorSyncStats(node_id="node_a", peer_node_id="node_b")

    async def _run():
        task = asyncio.create_task(
            run_forever(PEER, client, anchor_store, 0.02, stats, summary_store=summary_store)
        )
        for _ in range(50):
            if summary_store.get("node_b") is not None:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    got = summary_store.get("node_b")
    assert got is not None
    assert got.distinctive_terms == {"x": 1.0}
```

Add a `get_summary` method and `summary` attribute to the existing `FakePeerClient` in this file (used by the test above):

```python
    # add inside FakePeerClient.__init__:  self.summary = None
    # add as a method:
    def get_summary(self, peer):
        return self.summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_sync.py::test_run_forever_fetches_summary_on_root_mismatch -v`
Expected: FAIL — `TypeError: run_forever() got an unexpected keyword argument 'summary_store'`.

- [ ] **Step 3: Extend `run_forever` in `gin/federation/anchor_sync.py`**

Add the import at the top (with the other local imports):

```python
from .peer_summary_store import PeerSummaryStore
```

Replace the `run_forever` signature and body with:

```python
async def run_forever(
    peer: PeerConfig,
    peer_client: PeerClient,
    store: PeerAnchorStore,
    interval_s: float,
    stats: AnchorSyncStats,
    summary_store: Optional[PeerSummaryStore] = None,
) -> None:
    """One sync_once() per interval, forever, until cancelled. When the peer's
    anchor root changed this cycle, its routing summary is assumed stale too and
    refetched. Any failure is logged and skipped — background maintenance must
    never affect query answering."""
    while True:
        try:
            result = await asyncio.to_thread(sync_once, peer, peer_client, store)
            stats.cycles_run += 1
            stats.last_root_matched = result.root_matched
            stats.last_cycle_buckets_synced = result.buckets_synced
            stats.last_cycle_bytes = result.bytes_transferred
            if summary_store is not None and not result.root_matched:
                summary = await asyncio.to_thread(peer_client.get_summary, peer)
                summary_store.set(peer.node_id, summary)
        except Exception:
            logger.exception("anchor sync with %s failed", peer.node_id)
        await asyncio.sleep(interval_s)
```

Add `Optional` to the typing import at the top of the file if not already present:

```python
from typing import Optional
```

- [ ] **Step 4: Wire `peer_summary_store` into `create_app` (server.py)**

Add the import near the other local imports in `gin/federation/server.py`:

```python
from .peer_summary_store import PeerSummaryStore
```

Add the parameter to `create_app` (after `local_summary`):

```python
    peer_summary_store: Optional[PeerSummaryStore] = None,
```

In the `lifespan` closure, pass `summary_store` into `run_forever`:

```python
            task = asyncio.create_task(
                run_forever(
                    config.peers[0], peer_client, peer_anchor_store,
                    config.anchor_sync_interval_s, sync_stats,
                    summary_store=peer_summary_store,
                )
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_sync.py -v`
Expected: PASS, all tests (new + existing).

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py tests/test_anchor_endpoints.py -v`
Expected: PASS, unchanged.

- [ ] **Step 6: Commit**

```bash
git add gin/federation/anchor_sync.py gin/federation/server.py tests/test_anchor_sync.py
git commit -m "Sync peer routing summaries in the background loop when anchors change."
```

---

### Task 7: Router — multi-peer ranked delegation with fallback

**Files:**
- Modify: `gin/federation/router.py`
- Modify: `gin/federation/server.py`
- Test: `tests/test_federation_router.py`, `tests/test_router_selection.py` (new)

**Interfaces:**
- Consumes: existing `answer_or_delegate`, `RoutedResult`, `FederationLayer.peers_attempted` (Task 1); `rank_peers` (Task 2); `PeerSummaryStore` (Task 3); `query_keywords` from `gin.corpus.relevance`.
- Produces: `answer_or_delegate(..., peer_ranker: Optional[Callable[[str], list[PeerConfig]]] = None)` — tries peers in ranked order, falling back on refusal/unreachable, populating `RoutedResult.peers_attempted` and `FederationLayer.peers_attempted`; `RoutedResult` gains `peers_attempted: list[str]`. In `server.py`, `create_app` builds the ranker closure from `peer_summary_store` + an injected `embed_query_fn`. Consumed by Tasks 8, 10.

- [ ] **Step 1: Write the failing tests (router fallback + attribution)**

Create `tests/test_router_selection.py`:

```python
"""Multi-peer ranked delegation: try peers in ranker order, fall back on
refusal, record the full attempt order — never exceeding hop_count=1."""
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.router import answer_or_delegate
from gin.federation.schema import FederatedAnswer, NodeRefusal, WireClaim

PEER_B = PeerConfig(node_id="node_b", url="http://b")
PEER_C = PeerConfig(node_id="node_c", url="http://c")


def _cfg():
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/a", cold_path="data/cold_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret="s", peer_timeout_s=5.0, peers=(PEER_B, PEER_C),
    )


def _refuse_local(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


class ScriptedPeer:
    """Answers for `answerer` node_id; refuses for everyone else. Records calls."""

    def __init__(self, answerer):
        self.answerer = answerer
        self.calls = []

    def query(self, peer, fq):
        self.calls.append(peer.node_id)
        if peer.node_id == self.answerer:
            return FederatedAnswer(
                request_id=fq.request_id, node_id=peer.node_id,
                answer_text="grounded", claims=[WireClaim(
                    text="grounded", span_type="EXACT", cited_chunk_ids=["c:0"])],
                corpus_fingerprint={"n": 1}, synthesis_mode="convergent",
            )
        return NodeRefusal(request_id=fq.request_id, node_id=peer.node_id,
                           reason="zero_cursors")


def test_ranker_order_tried_first_success_no_fallback():
    peer_client = ScriptedPeer(answerer="node_c")
    # ranker puts C first (correct); B never contacted
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
        peer_ranker=lambda q: [PEER_C, PEER_B],
    )
    assert not result.refused
    assert result.source_node == "node_c"
    assert result.peers_attempted == ["node_c"]
    assert peer_client.calls == ["node_c"]
    assert result.federation.peers_attempted == ["node_c"]


def test_falls_back_to_next_peer_on_refusal():
    peer_client = ScriptedPeer(answerer="node_c")
    # ranker wrongly puts B first; B refuses, C answers
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
        peer_ranker=lambda q: [PEER_B, PEER_C],
    )
    assert not result.refused
    assert result.source_node == "node_c"
    assert result.peers_attempted == ["node_b", "node_c"]


def test_all_peers_refuse_aggregates_reasons():
    peer_client = ScriptedPeer(answerer="none")
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
        peer_ranker=lambda q: [PEER_B, PEER_C],
    )
    assert result.refused
    assert result.refusal_reasons["node_a"] == "retrieval_floor"
    assert result.refusal_reasons["node_b"] == "zero_cursors"
    assert result.refusal_reasons["node_c"] == "zero_cursors"
    assert result.peers_attempted == ["node_b", "node_c"]


def test_default_ranker_is_config_order():
    peer_client = ScriptedPeer(answerer="node_b")
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
    )  # no peer_ranker -> config order [B, C]
    assert result.source_node == "node_b"
    assert result.peers_attempted == ["node_b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_router_selection.py -v`
Expected: FAIL — `TypeError: answer_or_delegate() got an unexpected keyword argument 'peer_ranker'`.

- [ ] **Step 3: Rewrite the delegation section of `gin/federation/router.py`**

Add `peers_attempted` to `RoutedResult` (after `refusal_reasons`):

```python
    refusal_reasons: dict[str, str] = field(default_factory=dict)
    peers_attempted: list[str] = field(default_factory=list)
    request_id: str = ""
```

Add the `Callable`/`PeerConfig` types are already imported (`Callable` from typing; import `PeerConfig`):

```python
from .config import NodeConfig, PeerConfig
```

Replace `answer_or_delegate`'s signature and the block from `reasons = {...}` to the end of the function with:

```python
def answer_or_delegate(
    query: str,
    *,
    config: NodeConfig,
    answer_fn: AnswerFn,
    peer_client: PeerClient,
    request_id: Optional[str] = None,
    peer_ranker: Optional[Callable[[str], list[PeerConfig]]] = None,
) -> RoutedResult:
    rid = request_id or new_request_id()
    local = answer_fn(query)
    if not local.refused:
        return RoutedResult(
            refused=False,
            source_node=config.node_id,
            answer_text=local.raw_text,
            claims=claims_to_wire(local),
            synthesis_mode=local.synthesis_mode or "unknown",
            request_id=rid,
        )

    reasons = {config.node_id: local.refusal_reason or "zero_cursors"}
    if not config.peers:
        return RoutedResult(
            refused=True, source_node=config.node_id,
            refusal_reasons=reasons, request_id=rid,
        )

    peers_to_try = peer_ranker(query) if peer_ranker is not None else list(config.peers)
    attempted: list[str] = []
    for peer in peers_to_try:
        attempted.append(peer.node_id)
        fq = FederatedQuery(
            request_id=rid, query=query, origin_node=config.node_id, hop_count=1
        )
        try:
            outcome = peer_client.query(peer, fq)
        except PeerUnreachable:
            reasons[peer.node_id] = "unreachable"
            continue
        if isinstance(outcome, NodeRefusal):
            reasons[outcome.node_id] = outcome.reason
            continue
        return RoutedResult(
            refused=False,
            source_node=outcome.node_id,
            answer_text=outcome.answer_text,
            claims=list(outcome.claims),
            synthesis_mode=outcome.synthesis_mode,
            corpus_fingerprint=outcome.corpus_fingerprint,
            federation=FederationLayer(
                answered_by=outcome.node_id,
                hop_count=1,
                transport="http",
                peer_url=peer.url,
                request_id=rid,
                peers_attempted=list(attempted),
            ),
            peers_attempted=list(attempted),
            request_id=rid,
        )

    return RoutedResult(
        refused=True, source_node=config.node_id,
        refusal_reasons=reasons, peers_attempted=list(attempted), request_id=rid,
    )
```

- [ ] **Step 4: Build the ranker closure in `create_app` (server.py)**

Add imports near the other local imports:

```python
from gin.corpus.relevance import query_keywords

from .config import NodeConfig, PeerConfig
from .peer_selection import rank_peers
```

(`NodeConfig` is already imported — add `PeerConfig` to that line.)

Add an `embed_query_fn` parameter to `create_app` (after `peer_summary_store`):

```python
    embed_query_fn: Optional[Callable[[str], list[float]]] = None,
```

After `summary_fn = ...`, build the ranker closure:

```python
    def _rank_peers_for_query(query: str) -> list[PeerConfig]:
        if (
            peer_summary_store is None
            or embed_query_fn is None
            or len(config.peers) <= 1
        ):
            return list(config.peers)
        summaries = {}
        for p in config.peers:
            s = peer_summary_store.get(p.node_id)
            if s is not None:
                summaries[p.node_id] = s
        if not summaries:
            return list(config.peers)
        order = rank_peers(
            embed_query_fn(query), query_keywords(query),
            summaries, [p.node_id for p in config.peers],
        )
        by_id = {p.node_id: p for p in config.peers}
        return [by_id[nid] for nid in order]
```

In `federated_query`, pass the ranker into `answer_or_delegate`:

```python
        routed = answer_or_delegate(
            fq.query,
            config=config,
            answer_fn=answer_fn,
            peer_client=peer_client,
            request_id=fq.request_id,
            peer_ranker=_rank_peers_for_query,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_router_selection.py tests/test_federation_router.py -v`
Expected: PASS — new selection tests plus the pre-existing router tests (unchanged behavior at 1 peer / default ranker).

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py tests/test_federation_loop.py -v`
Expected: PASS, unchanged (single-peer configs → ranker returns config order).

- [ ] **Step 6: Commit**

```bash
git add gin/federation/router.py gin/federation/server.py tests/test_router_selection.py
git commit -m "Router: ranked multi-peer delegation with fallback; record peers_attempted."
```

---

### Task 8: Real-socket 3-node integration test

**Files:**
- Create: `tests/test_peer_selection_loop.py`

**Interfaces:**
- Consumes: `create_app` with `peer_summary_store`/`embed_query_fn`/`local_summary` (Tasks 5-7); `HttpPeerClient` (Task 4); `InMemoryPeerSummaryStore` (Task 3); `rank_peers` (Task 2).
- Produces: nothing new — a proof test that A selects the correct peer among two over a real HTTP boundary, model-free (fake embeddings + stub answer fns).

- [ ] **Step 1: Write the test**

Create `tests/test_peer_selection_loop.py`:

```python
"""Three uvicorn nodes over real sockets, no model/DB: node A ranks B vs. C
from injected summaries and delegates to the right one on the first try."""
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import FederatedQuery, FederatedResponse, PeerSummaryResponse
from gin.federation.server import create_app

SECRET = "sel-secret"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _refuse(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


def _grounded(node_id):
    def fn(q):
        return ArmOutput(
            raw_text=f"{node_id} answer",
            claims=[RawClaim(text=f"{node_id} answer", span_type="EXACT",
                             cited_chunk_ids=[f"{node_id}:0"])],
            retrieval_manifest_hash="h", synthesis_mode="convergent",
        )
    return fn


def _cfg(node_id, port, peers):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=10.0, peers=peers,
    )


@pytest.fixture
def three_nodes():
    pa, pb, pc = _free_port(), _free_port(), _free_port()
    peer_client = HttpPeerClient(SECRET, timeout_s=10.0)
    # B answers only "justice"-ish; C answers only "inflation"-ish — via stubs,
    # selection is driven purely by the injected summaries + query embedding.
    cfg_a = _cfg("node_a", pa, (PeerConfig("node_b", f"http://127.0.0.1:{pb}"),
                                PeerConfig("node_c", f"http://127.0.0.1:{pc}")))
    cfg_b = _cfg("node_b", pb, (PeerConfig("node_a", f"http://127.0.0.1:{pa}"),))
    cfg_c = _cfg("node_c", pc, (PeerConfig("node_a", f"http://127.0.0.1:{pa}"),))

    summary_store = InMemoryPeerSummaryStore()
    summary_store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0], distinctive_terms={"justice": 3.0}))
    summary_store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0], distinctive_terms={"inflation": 3.0}))

    # Query embedder: "inflation" -> near C's centroid; else near B's.
    def embed(q):
        return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]

    app_a = create_app(cfg_a, answer_fn=_refuse, peer_client=peer_client,
                       peer_summary_store=summary_store, embed_query_fn=embed)
    app_b = create_app(cfg_b, answer_fn=_grounded("node_b"), peer_client=peer_client)
    app_c = create_app(cfg_c, answer_fn=_grounded("node_c"), peer_client=peer_client)
    sa, sb, sc = _serve(app_a, pa), _serve(app_b, pb), _serve(app_c, pc)
    yield f"http://127.0.0.1:{pa}"
    sa.should_exit = sb.should_exit = sc.should_exit = True
    time.sleep(0.2)


def _ask(url, query):
    fq = FederatedQuery(query=query, origin_node="driver", hop_count=0)
    r = httpx.post(f"{url}/v1/federated/query",
                   headers={"Authorization": f"Bearer {SECRET}"},
                   json=fq.model_dump(), timeout=15.0)
    return FederatedResponse.model_validate(r.json())


def test_selects_c_for_inflation_query_first_try(three_nodes):
    resp = _ask(three_nodes, "what drives inflation")
    assert resp.answer.node_id == "node_c"
    assert resp.federation.peers_attempted == ["node_c"]  # correct peer, first try


def test_selects_b_for_justice_query_first_try(three_nodes):
    resp = _ask(three_nodes, "environmental justice movements")
    assert resp.answer.node_id == "node_b"
    assert resp.federation.peers_attempted == ["node_b"]
```

- [ ] **Step 2: Run the test 3 times to confirm non-flaky**

Run (three separate times): `./venv/Scripts/python.exe -m pytest tests/test_peer_selection_loop.py -v`
Expected: PASS, both tests, all three runs. If timing-flaky, raise `_serve` deadlines only — do not change production code.

- [ ] **Step 3: Commit**

```bash
git add tests/test_peer_selection_loop.py
git commit -m "Prove correct-peer selection over real sockets across three nodes, model-free."
```

---

### Task 9: Third corpus + three-node config + DB setup + queryset

**Files:**
- Create: `corpus_node3.json`
- Create: `config/node_c.yaml`
- Modify: `config/node_a.yaml`, `config/node_b.yaml`
- Modify: `scripts/federation_db_setup.sh`
- Create: `data/eval/queryset_peer_selection.yaml`

**Interfaces:**
- Consumes: the ingest format (`gin/corpus/ingest.py` `load_json`), the node YAML schema (`gin/federation/config.py`).
- Produces: a third topically-distinct corpus (monetary policy), three node configs each listing the other two as peers, a setup script that provisions/ingests all three DBs and re-applies the schema, and a four-class queryset.

- [ ] **Step 1: Create `corpus_node3.json`** (monetary-policy domain, distinct from climate/environmental-justice)

```json
{
  "node_id": "node_3_monetary",
  "documents": [
    {
      "doc_id": "n3_doc_001",
      "source": "Federal Reserve: The Dual Mandate Explained",
      "url": "https://www.federalreserve.gov/dual-mandate",
      "node": "node_3_monetary",
      "metadata": {"domain": "monetary_policy", "type": "institutional_report", "author": "Federal Reserve", "category": "central_banking"},
      "chunks": [
        {"chunk_id": "n3_doc_001_c000", "position": 0, "text": "The Federal Reserve operates under a dual mandate set by Congress: to promote maximum employment and stable prices across the United States economy."},
        {"chunk_id": "n3_doc_001_c001", "position": 1, "text": "The Federal Open Market Committee sets a target range for the federal funds rate to steer monetary policy toward those two goals."}
      ]
    },
    {
      "doc_id": "n3_doc_002",
      "source": "Bureau of Labor Statistics: Understanding the Consumer Price Index",
      "url": "https://www.bls.gov/cpi",
      "node": "node_3_monetary",
      "metadata": {"domain": "monetary_policy", "type": "institutional_report", "author": "BLS", "category": "inflation"},
      "chunks": [
        {"chunk_id": "n3_doc_002_c000", "position": 0, "text": "The Consumer Price Index measures the average change over time in the prices paid by urban consumers for a market basket of goods and services."},
        {"chunk_id": "n3_doc_002_c001", "position": 1, "text": "Core inflation strips out volatile food and energy prices, and central banks commonly target an annual inflation rate near two percent."}
      ]
    },
    {
      "doc_id": "n3_doc_003",
      "source": "Federal Reserve: Quantitative Easing and the Balance Sheet",
      "url": "https://www.federalreserve.gov/quantitative-easing",
      "node": "node_3_monetary",
      "metadata": {"domain": "monetary_policy", "type": "institutional_report", "author": "Federal Reserve", "category": "central_banking"},
      "chunks": [
        {"chunk_id": "n3_doc_003_c000", "position": 0, "text": "Quantitative easing is a monetary policy tool in which a central bank purchases longer-term securities to expand its balance sheet and lower long-term interest rates."},
        {"chunk_id": "n3_doc_003_c001", "position": 1, "text": "The Federal Reserve used large-scale asset purchases during the 2008 financial crisis and again in 2020 to support market functioning and the broader economy."}
      ]
    },
    {
      "doc_id": "n3_doc_004",
      "source": "Bank for International Settlements: The Federal Funds Rate Transmission",
      "url": "https://www.bis.org/funds-rate",
      "node": "node_3_monetary",
      "metadata": {"domain": "monetary_policy", "type": "institutional_report", "author": "BIS", "category": "central_banking"},
      "chunks": [
        {"chunk_id": "n3_doc_004_c000", "position": 0, "text": "When the central bank raises the policy interest rate, borrowing costs rise across mortgages, business loans, and consumer credit, cooling aggregate demand."},
        {"chunk_id": "n3_doc_004_c001", "position": 1, "text": "This transmission mechanism operates with a lag, so rate hikes intended to curb inflation can take many months to fully reach the real economy."}
      ]
    },
    {
      "doc_id": "n3_doc_005",
      "source": "IMF: Why Central Bank Independence Matters",
      "url": "https://www.imf.org/central-bank-independence",
      "node": "node_3_monetary",
      "metadata": {"domain": "monetary_policy", "type": "institutional_report", "author": "IMF", "category": "central_banking"},
      "chunks": [
        {"chunk_id": "n3_doc_005_c000", "position": 0, "text": "Central bank independence insulates monetary policy decisions from short-term political pressures, which helps anchor long-run inflation expectations."},
        {"chunk_id": "n3_doc_005_c001", "position": 1, "text": "Empirical studies find that countries with more independent central banks have historically experienced lower and more stable inflation."}
      ]
    },
    {
      "doc_id": "n3_doc_006",
      "source": "Federal Reserve: Money Supply and Monetary Aggregates",
      "url": "https://www.federalreserve.gov/money-supply",
      "node": "node_3_monetary",
      "metadata": {"domain": "monetary_policy", "type": "institutional_report", "author": "Federal Reserve", "category": "central_banking"},
      "chunks": [
        {"chunk_id": "n3_doc_006_c000", "position": 0, "text": "Monetary aggregates such as M1 and M2 measure the stock of money circulating in the economy, from physical currency to savings deposits."},
        {"chunk_id": "n3_doc_006_c001", "position": 1, "text": "The velocity of money describes how quickly a unit of currency is spent, linking the money supply to nominal economic output."}
      ]
    }
  ]
}
```

- [ ] **Step 2: Create `config/node_c.yaml`**

```yaml
# GIN federation node C (monetary-policy corpus: corpus_node3.json).
# Runs CPU-only (n_gpu_layers: 0) so all three 7B models fit alongside
# node A and node B on a single 12GB GPU.
node_id: node_c
host: 127.0.0.1
port: 8473
database_url: postgresql://gin:gin@localhost:5432/gin_node_c
cold_path: data/cold_node_c
model_path: data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
n_gpu_layers: 0
n_ctx: 4096
shared_secret: dev-federation-secret
peer_timeout_s: 300
anchor_sync_interval_s: 10
peers:
  - node_id: node_a
    url: http://127.0.0.1:8471
  - node_id: node_b
    url: http://127.0.0.1:8472
```

- [ ] **Step 3: Give node A and node B their second peer**

In `config/node_a.yaml`, replace the `peers:` block with:

```yaml
peers:
  - node_id: node_b
    url: http://127.0.0.1:8472
  - node_id: node_c
    url: http://127.0.0.1:8473
```

In `config/node_b.yaml`, replace the `peers:` block with:

```yaml
peers:
  - node_id: node_a
    url: http://127.0.0.1:8471
  - node_id: node_c
    url: http://127.0.0.1:8473
```

- [ ] **Step 4: Extend `scripts/federation_db_setup.sh`** to provision the third node and re-apply the schema to all three

Replace the body of `scripts/federation_db_setup.sh` with:

```bash
#!/usr/bin/env bash
# Create and populate the three per-node federation databases.
# Prereqs: gin-postgres container running; venv installed.
# Idempotent-ish: CREATE DATABASE fails harmlessly if it already exists;
# init-db.sql is all IF NOT EXISTS so re-applying only adds new tables.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./venv/Scripts/python.exe

echo "[1/3] creating databases"
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec gin-postgres psql -U gin -d gin -c "CREATE DATABASE $db OWNER gin;" || true
done

echo "[2/3] applying schema (adds peer_anchors + peer_summaries if missing)"
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec -i gin-postgres psql -U gin -d "$db" < docker/init-db.sql
done

echo "[3/3] ingesting split corpora"
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_a" \
GIN_COLD_PATH="data/cold_node_a" "$PY" scripts/corpus_ingest.py --source corpus_node1.json --no-edges
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_b" \
GIN_COLD_PATH="data/cold_node_b" "$PY" scripts/corpus_ingest.py --source corpus_node2.json --no-edges
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_c" \
GIN_COLD_PATH="data/cold_node_c" "$PY" scripts/corpus_ingest.py --source corpus_node3.json --no-edges

echo "done. verify:"
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec gin-postgres psql -U gin -d "$db" -c "SELECT '$db' AS db, COUNT(*) AS chunks FROM chunks;"
done
```

- [ ] **Step 5: Create `data/eval/queryset_peer_selection.yaml`**

```yaml
# Peer-selection bar queryset (three nodes).
# Spec: docs/superpowers/specs/2026-07-15-peer-selection-n3-design.md
# Node A: corpus_node1 (institutional/climate). Node B: corpus_node2
# (grassroots/environmental-justice). Node C: corpus_node3 (monetary policy).
# Classes: a_answerable -> A local (routing is a false positive);
# b_only -> route, B grounds, correct peer node_b; c_only -> route, C grounds,
# correct peer node_c; neither -> all refuse honestly.
queries:
  - id: sel_a_2023_anomaly
    query: How much warmer was 2023 than the twentieth-century average?
    federation_class: a_answerable
  - id: sel_a_ocean_acidification
    query: What is ocean acidification and what causes it?
    federation_class: a_answerable
  - id: sel_b_landback
    query: What is LANDBACK and what does it seek to reclaim?
    federation_class: b_only
  - id: sel_b_weact_founding
    query: How was WE ACT for Environmental Justice founded?
    federation_class: b_only
  - id: sel_c_dual_mandate
    query: What is the Federal Reserve's dual mandate?
    federation_class: c_only
  - id: sel_c_quantitative_easing
    query: What is quantitative easing and when was it used?
    federation_class: c_only
  - id: sel_neither_sports
    query: Who won the national football championship final this year?
    federation_class: neither
  - id: sel_neither_sourdough
    query: What temperature should sourdough bread be baked at?
    federation_class: neither
```

- [ ] **Step 6: Provision and verify all three databases**

Run: `bash scripts/federation_db_setup.sh`
Expected: three CREATE DATABASE (or already-exists), schema applied to all three (peer_anchors + peer_summaries created where missing), three ingests report chunk counts, final SELECTs show nonzero chunks in each (node A ~55, node B ~50, node C ~12).

- [ ] **Step 7: Commit**

```bash
git add corpus_node3.json config/node_c.yaml config/node_a.yaml config/node_b.yaml \
        scripts/federation_db_setup.sh data/eval/queryset_peer_selection.yaml
git commit -m "Third node (monetary corpus) + three-node configs + setup + selection queryset."
```

---

### Task 10: Live eval driver + metrics module + docs + final validation

**Files:**
- Create: `gin/federation/selection_eval.py`
- Create: `scripts/eval_peer_selection.py`
- Test: `tests/test_selection_eval.py`
- Modify: `architecture.md`, `README.md`, `docs/GIN_Node_Architecture_v1.md`

**Interfaces:**
- Consumes: `verify_claims_in_db`, `claims_verify` from `gin.federation.eval` (existing); `FederatedResponse` (existing, now with `federation.peers_attempted`).
- Produces: `selection_eval.load_selection_queryset`, `SelectionOutcome`, `compute_selection_metrics`; a live driver writing `data/eval_runs/<ts>/peer_selection_metrics.json`.

- [ ] **Step 1: Write the failing metrics tests**

Create `tests/test_selection_eval.py`:

```python
"""Selection metrics: precision@1 and avg peers tried over routed queries."""
from gin.federation.selection_eval import SelectionOutcome, compute_selection_metrics


def _routed(id, cls, source, attempted, verified=True):
    return SelectionOutcome(id=id, federation_class=cls, refused=False, routed=True,
                            source_node=source, peers_attempted=attempted,
                            attribution_verified=verified)


def test_precision_at_1_all_correct():
    outcomes = [
        _routed("b1", "b_only", "node_b", ["node_b"]),
        _routed("c1", "c_only", "node_c", ["node_c"]),
    ]
    m = compute_selection_metrics(outcomes)
    assert m["selection_precision_at_1"] == 1.0
    assert m["avg_peers_tried"] == 1.0
    assert m["routed_fabrication_rate"] == 0.0


def test_precision_at_1_penalizes_wrong_first_pick():
    outcomes = [
        _routed("b1", "b_only", "node_b", ["node_b"]),
        _routed("c1", "c_only", "node_c", ["node_b", "node_c"]),  # wrong first
    ]
    m = compute_selection_metrics(outcomes)
    assert m["selection_precision_at_1"] == 0.5
    assert m["avg_peers_tried"] == 1.5


def test_routing_false_positive_and_honest_refusal():
    outcomes = [
        SelectionOutcome(id="a1", federation_class="a_answerable", refused=False,
                         routed=False, source_node="node_a"),
        SelectionOutcome(id="n1", federation_class="neither", refused=True,
                         routed=True, peers_attempted=["node_b", "node_c"]),
    ]
    m = compute_selection_metrics(outcomes)
    assert m["routing_false_positives"] == 0
    assert m["honest_refusal_rate"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_selection_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.selection_eval'`.

- [ ] **Step 3: Write `gin/federation/selection_eval.py`**

```python
"""Peer-selection bar: four-class queryset, outcomes, and metrics.

The correct peer for a b_only/c_only query is implied by its class label
(b_only -> node_b, c_only -> node_c). Selection precision@1 asks whether A's
FIRST contacted peer was that correct one; avg peers tried asks how far down
the ranked list A had to go. Attribution reuses gin.federation.eval's verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

VALID_CLASSES = {"a_answerable", "b_only", "c_only", "neither"}
CLASS_TO_PEER = {"b_only": "node_b", "c_only": "node_c"}


@dataclass(frozen=True)
class SelectionQuery:
    id: str
    query: str
    federation_class: str


def load_selection_queryset(path: str | Path) -> list[SelectionQuery]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: list[SelectionQuery] = []
    for q in raw["queries"]:
        cls = q["federation_class"]
        if cls not in VALID_CLASSES:
            raise ValueError(f"query {q['id']!r}: class {cls!r} not in {sorted(VALID_CLASSES)}")
        out.append(SelectionQuery(id=str(q["id"]), query=str(q["query"]), federation_class=cls))
    return out


@dataclass
class SelectionOutcome:
    id: str
    federation_class: str
    refused: bool
    routed: bool
    source_node: str = ""
    peers_attempted: list[str] = field(default_factory=list)
    attribution_verified: Optional[bool] = None
    refusal_reasons: dict = field(default_factory=dict)


def compute_selection_metrics(outcomes: list[SelectionOutcome]) -> dict:
    a = [o for o in outcomes if o.federation_class == "a_answerable"]
    neither = [o for o in outcomes if o.federation_class == "neither"]
    routed_grounded = [
        o for o in outcomes
        if o.federation_class in CLASS_TO_PEER and o.routed and not o.refused
    ]
    correct_first = [
        o for o in routed_grounded
        if o.peers_attempted[:1] == [CLASS_TO_PEER[o.federation_class]]
    ]
    verified = [o for o in routed_grounded if o.attribution_verified]
    tried_counts = [len(o.peers_attempted) for o in routed_grounded]
    return {
        "n_queries": len(outcomes),
        # Bar: 1.0 — the correct peer is contacted first.
        "selection_precision_at_1": (len(correct_first) / len(routed_grounded)) if routed_grounded else None,
        # Bar: ~1.0 — selection beats blind sequential fan-out.
        "avg_peers_tried": (sum(tried_counts) / len(tried_counts)) if tried_counts else None,
        # Bar: 0 — A-answerable queries never route.
        "routing_false_positives": sum(1 for o in a if o.routed),
        # Bar: 1.0 / 0.0 — routed answers verify against the answering node's corpus.
        "routed_attribution_verified": (len(verified) / len(routed_grounded)) if routed_grounded else None,
        "routed_fabrication_rate": (1.0 - len(verified) / len(routed_grounded)) if routed_grounded else None,
        # Bar: 1.0 — neither-class queries end in refusal.
        "honest_refusal_rate": (sum(1 for o in neither if o.refused) / len(neither)) if neither else None,
        "per_query": [o.__dict__ for o in outcomes],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_selection_eval.py -v`
Expected: PASS, all 3 tests.

- [ ] **Step 5: Write `scripts/eval_peer_selection.py`**

```python
"""Measure the peer-selection bar against three live node processes.

Prereqs (four terminals):
    bash scripts/federation_db_setup.sh                        # once
    python scripts/node_serve.py --config config/node_b.yaml    # terminal 1
    python scripts/node_serve.py --config config/node_c.yaml    # terminal 2
    python scripts/node_serve.py --config config/node_a.yaml    # terminal 3
    python scripts/eval_peer_selection.py                       # terminal 4

Node A must have completed at least one anchor-sync cycle with each peer so its
summary cache is populated (the driver polls A's sync_stats and sleeps first).

Bar (spec): selection_precision_at_1 1.0; avg_peers_tried ~1.0;
routing_false_positives 0; routed_fabrication_rate 0.0; honest_refusal_rate 1.0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from gin.federation.eval import verify_claims_in_db
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.selection_eval import (
    SelectionOutcome,
    compute_selection_metrics,
    load_selection_queryset,
)

DEFAULT_OUT = ROOT / "data" / "eval_runs"
DB_FOR_NODE = {
    "node_b": "postgresql://gin:gin@localhost:5432/gin_node_b",
    "node_c": "postgresql://gin:gin@localhost:5432/gin_node_c",
}


def _await_summaries(node_a_url: str, headers: dict, timeout_s: float) -> None:
    """Wait until A has run enough sync cycles to have cached peer summaries."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = httpx.get(f"{node_a_url}/v1/federated/anchors/sync_stats", headers=headers, timeout=10.0)
        if r.status_code == 200 and r.json().get("cycles_run", 0) >= 1:
            time.sleep(2.0)  # let the summary fetch that follows the mismatch land
            return
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-url", default="http://127.0.0.1:8471")
    parser.add_argument("--queryset", default=str(ROOT / "data" / "eval" / "queryset_peer_selection.yaml"))
    parser.add_argument("--secret", default="dev-federation-secret")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--summary-wait", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    queries = load_selection_queryset(args.queryset)
    headers = {"Authorization": f"Bearer {args.secret}"}
    _await_summaries(args.node_a_url, headers, args.summary_wait)

    outcomes: list[SelectionOutcome] = []
    with httpx.Client(timeout=args.timeout) as client:
        for q in queries:
            fq = FederatedQuery(query=q.query, origin_node="eval_driver", hop_count=0)
            r = client.post(f"{args.node_a_url}/v1/federated/query", headers=headers, json=fq.model_dump())
            r.raise_for_status()
            resp = FederatedResponse.model_validate(r.json())

            if resp.refusal is not None:
                attempted = list(resp.refusal.peer_reasons.keys())
                outcomes.append(SelectionOutcome(
                    id=q.id, federation_class=q.federation_class, refused=True,
                    routed=bool(resp.refusal.peer_reasons), peers_attempted=attempted,
                    refusal_reasons={resp.refusal.node_id: resp.refusal.reason, **resp.refusal.peer_reasons},
                ))
            else:
                routed = resp.federation is not None
                attempted = list(resp.federation.peers_attempted) if routed else []
                verified = (
                    verify_claims_in_db(resp.answer.claims, DB_FOR_NODE[resp.answer.node_id])
                    if routed and resp.answer.node_id in DB_FOR_NODE else None
                )
                outcomes.append(SelectionOutcome(
                    id=q.id, federation_class=q.federation_class, refused=False,
                    routed=routed, source_node=resp.answer.node_id,
                    peers_attempted=attempted, attribution_verified=verified,
                ))
            o = outcomes[-1]
            print(f"[{q.id}] class={q.federation_class} routed={o.routed} "
                  f"source={o.source_node!r} attempted={o.peers_attempted} verified={o.attribution_verified}")

    metrics = compute_selection_metrics(outcomes)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "peer_selection_metrics.json"
    artifact.write_text(json.dumps(
        {"run_id": ts, "generated_at": datetime.now(timezone.utc).isoformat(),
         "node_a_url": args.node_a_url, "queryset": str(args.queryset), **metrics},
        indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_query"}, indent=2))
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Wire the real summary + embedder into `scripts/node_serve.py`**

Add to the deferred import block inside `main()` (with the other federation imports):

```python
    from gin.corpus.hot import embed_query
    from gin.federation.peer_summary_store import (
        PostgresPeerSummaryStore, build_local_summary,
    )
```

Extend the `create_app(...)` call with the new arguments:

```python
    app = create_app(
        config,
        answer_fn=lambda q: answer_query(q, llm, arm_cfg),
        peer_client=HttpPeerClient(config.shared_secret, config.peer_timeout_s),
        corpus_fingerprint=fingerprint,
        local_anchor_rows=local_anchor_rows,
        peer_anchor_store=PostgresPeerAnchorStore(),
        local_summary=lambda: build_local_summary(config.node_id),
        peer_summary_store=PostgresPeerSummaryStore(),
        embed_query_fn=embed_query,
    )
```

- [ ] **Step 7: Run the live eval** (four terminals per the driver docstring; node C is CPU-only per its config)

Start Postgres if needed (`docker compose up -d` in `docker/`), run `bash scripts/federation_db_setup.sh` once, then start nodes B, C, A and run:

Run: `./venv/Scripts/python.exe scripts/eval_peer_selection.py`
Expected: `selection_precision_at_1: 1.0`, `avg_peers_tried: 1.0`, `routing_false_positives: 0`, `routed_fabrication_rate: 0.0`, `honest_refusal_rate: 1.0`. Note the run timestamp. **Shut down all three node processes afterward** (they hold model instances).

- [ ] **Step 8: Update `architecture.md` Phase 3 checklist**

Replace the `🔲 Trust weights, PKI/mTLS, peer selection` line (and/or add after the Merkle sync line) with:

```markdown
- ✅ Peer selection at N>2 (spec #3) — third node added; A ranks peers by
  dense+sparse RRF fusion (reusing the retrieval stack's `RRF_K`) over routing
  summaries synced alongside anchors, and delegates to the best peer first,
  falling back through the ranked list. Measured:
  `data/eval_runs/<ts>/peer_selection_metrics.json` (selection precision@1 1.0,
  avg peers tried ~1.0, fabrication 0, honest refusal 1.0).
  Spec: docs/superpowers/specs/2026-07-15-peer-selection-n3-design.md
- 🔲 Trust weights (per-domain asymmetric), gRPC/QUIC wire, PKI/mTLS
```

(Substitute the real run timestamp for `<ts>`.)

- [ ] **Step 9: Update `README.md`** — add a three-node subsection after the Merkle anchor sync section (before its closing `---`), and update the federation status row

Subsection:

```markdown
### Peer selection (three nodes)

With a third node (monetary-policy corpus) added, a node that can't ground a
query ranks its peers by dense+sparse RRF fusion over routing summaries synced
alongside anchors, then delegates to the best-matching peer first:

```bash
bash scripts/federation_db_setup.sh   # provisions gin_node_a/b/c
# serve three nodes (node C is CPU-only so all three 7B models fit on one GPU)
python scripts/node_serve.py --config config/node_b.yaml
python scripts/node_serve.py --config config/node_c.yaml
python scripts/node_serve.py --config config/node_a.yaml
python scripts/eval_peer_selection.py
```

Selection is content-similarity only; trust weights remain a later mechanism.
Bar and scope: docs/superpowers/specs/2026-07-15-peer-selection-n3-design.md.
```

Update the federation status-table row to append: `✅ peer selection at N=3 measured (run <ts>: selection precision@1 1.0, avg peers tried ~1.0); trust weights + gRPC/QUIC + mTLS deferred`.

- [ ] **Step 10: Add a note to `docs/GIN_Node_Architecture_v1.md`** after the summary/sync note added in the prior sub-project:

```markdown
> **v1 peer selection (2026-07):** when a node routes a query it ranks peers
> by content similarity only — dense (query embedding vs. each peer's synced
> centroid) and sparse (query keywords vs. each peer's distinctive IDF terms),
> RRF-fused with the same constant as hybrid retrieval. Trust weights (the
> per-domain asymmetric weighting described above) remain a separate, later
> mechanism layered on top of this similarity signal.
```

- [ ] **Step 11: Full suite + final validation**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all green (integration tests run with Postgres up, else skip).

Run: `git status --short`
Expected: only intended files; no stray artifacts beyond the eval metrics JSON.

- [ ] **Step 12: Commit** (hold the push for the final whole-branch review, matching the prior two sub-projects)

```bash
git add gin/federation/selection_eval.py scripts/eval_peer_selection.py \
        scripts/node_serve.py tests/test_selection_eval.py \
        data/eval_runs/<ts>/peer_selection_metrics.json \
        architecture.md README.md docs/GIN_Node_Architecture_v1.md
git commit -m "Peer selection measured at N=3: eval driver + metrics + docs. Precision@1 1.0.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

(Substitute the real run timestamp for `<ts>`. Do NOT push — the controlling session pushes after the final whole-branch review.)

---

## Self-review notes (already applied)

- **Spec coverage:** falsifiable claim (precision@1, avg peers tried, carried-forward v1 bars) → Task 10 metrics + eval; dense+sparse RRF signal → Task 2; per-node summary (centroid + IDF terms) never carrying text → Tasks 1, 3, 5; piggyback summary sync on anchor root-mismatch → Task 6; sequential ranked fallback, hop_count=1 → Task 7; peers-with-no-summary-last → Task 2 `rank_peers`; three-node deployment + CPU-only node C → Task 9; three-tier testing (unit → Tasks 2/3/7, integration → Task 8, live → Task 10) → present; docs → Task 10.
- **Placeholder scan:** no TBD/TODO; every code step is complete. `<ts>` in Task 10 docs steps is an explicit "substitute the real run timestamp" instruction, not a code placeholder.
- **Type consistency:** `PeerSummaryResponse` (fields node_id/embedding_centroid/distinctive_terms), `PeerSummaryStore` (get/set), `rank_peers`/`dense_rank`/`sparse_rank` signatures, `answer_or_delegate`'s `peer_ranker`, `RoutedResult.peers_attempted`, `FederationLayer.peers_attempted`, `SelectionOutcome` fields — all verified identical across every task that references them (schema Task 1 → selection Task 2 → store Task 3 → client Task 4 → server Tasks 5-7 → router Task 7 → eval Task 10).
- **Non-breaking check:** every new `create_app` parameter (`local_summary`, `peer_summary_store`, `embed_query_fn`) and `run_forever`/`answer_or_delegate` parameter is keyword-only with a default reproducing prior behavior (empty summary, no summary sync, config-order ranking). Single-peer configs and the existing two-node tests exercise the unchanged path. `RRF_K` reused via import, not redefined.
