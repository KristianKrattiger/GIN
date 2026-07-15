# Merkle Anchor Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each federation node a background-synced local cache of its peer's anchor metadata (`chunk_id`, `content_hash`, `outlet`, `title` — never chunk text), using a 16-bucket Merkle tree so unchanged data never crosses the wire.

**Architecture:** Extends `gin/federation/` (not a new package) with three new modules — `anchor_tree.py` (pure bucket/hash logic, no I/O), `anchor_store.py` (the `peer_anchors` cache table + reading a node's own chunks as anchors), `anchor_sync.py` (one sync cycle + the forever loop). Three new `GET` endpoints on the existing FastAPI app expose a node's own tree; a background `asyncio` task per node process pulls its peer's tree on an interval and updates only the buckets that changed.

**Tech Stack:** Python 3.12, FastAPI/Starlette (lifespan), httpx, Pydantic, psycopg3/Postgres, pytest, uvicorn (real-socket tests).

## Global Constraints

- Chunk **text** never crosses the anchor-sync wire — only `chunk_id`, `content_hash`, `outlet`, `title`. This is the right-to-opacity constraint from federation v1, unchanged.
- `NUM_BUCKETS = 16` is a fixed protocol constant for v1 (documented next to `PROTOCOL_VERSION` in `schema.py`), not configurable per node.
- Background sync must never block or fail a live federated query — any exception in a sync cycle is logged and the loop continues on the next interval.
- Existing federation v1 tests (`test_federation_server.py`, `test_federation_loop.py`, `test_federation_client.py`) must keep passing unmodified in behavior — new `create_app`/`PeerClient` parameters are additive and default to "anchor sync disabled."
- Follow the existing `gin/corpus/db.py` convention: open a fresh Postgres connection per call (`connect()`/`transaction()`), never hold one across the life of a background task.
- DB-touching tests use the existing `isolated_db` / `require_postgres` fixtures (`tests/conftest.py`) and are marked `@pytest.mark.integration`, matching every other Postgres-backed test in this repo.

---

### Task 1: Wire schema additions

**Files:**
- Modify: `gin/federation/schema.py`
- Test: `tests/test_federation_schema.py`

**Interfaces:**
- Consumes: `PROTOCOL_VERSION` (existing, `gin/federation/schema.py:15`).
- Produces: `AnchorLeaf(chunk_id: str, content_hash: str, outlet: str, title: str)`; `AnchorRootResponse(protocol_version, node_id, root_hash, leaf_count)`; `AnchorBucketsResponse(protocol_version, node_id, bucket_hashes: list[str])`; `AnchorLeavesResponse(protocol_version, node_id, bucket_index: int, leaves: list[AnchorLeaf])`; `AnchorSyncStats(node_id, peer_node_id, cycles_run: int, last_root_matched: bool, last_cycle_buckets_synced: int, last_cycle_bytes: int)`. All consumed by every later task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_federation_schema.py`:

```python
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
)


def test_anchor_root_response_round_trip():
    resp = AnchorRootResponse(node_id="node_a", root_hash="abc123", leaf_count=55)
    again = AnchorRootResponse.model_validate(resp.model_dump())
    assert again == resp
    assert again.protocol_version == PROTOCOL_VERSION


def test_anchor_buckets_response_round_trip():
    resp = AnchorBucketsResponse(node_id="node_a", bucket_hashes=["h"] * 16)
    again = AnchorBucketsResponse.model_validate(resp.model_dump())
    assert len(again.bucket_hashes) == 16


def test_anchor_leaves_response_round_trip():
    resp = AnchorLeavesResponse(
        node_id="node_b",
        bucket_index=3,
        leaves=[
            AnchorLeaf(
                chunk_id="n2_doc_001:0", content_hash="h1",
                outlet="node_2_grassroots", title="WE ACT",
            )
        ],
    )
    again = AnchorLeavesResponse.model_validate(resp.model_dump())
    assert again.leaves[0].chunk_id == "n2_doc_001:0"


def test_anchor_sync_stats_defaults():
    stats = AnchorSyncStats(node_id="node_a", peer_node_id="node_b")
    assert stats.cycles_run == 0
    assert stats.last_root_matched is False
    assert stats.last_cycle_buckets_synced == 0
    assert stats.last_cycle_bytes == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'AnchorRootResponse'`.

- [ ] **Step 3: Add the models to `schema.py`**

Append to `gin/federation/schema.py`, after `FederatedResponse`:

```python
# --- Anchor sync wire messages -------------------------------------------
# A 2-level Merkle tree over (chunk_id, content_hash, outlet, title) tuples,
# bucketed by sha256(chunk_id)[0] into NUM_BUCKETS (gin/federation/anchor_tree.py)
# fixed buckets. Right-to-opacity applies here too: chunk TEXT never appears
# on this wire, only these four fields.

NUM_BUCKETS = 16


class AnchorLeaf(BaseModel):
    chunk_id: str
    content_hash: str
    outlet: str
    title: str


class AnchorRootResponse(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    root_hash: str
    leaf_count: int


class AnchorBucketsResponse(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    bucket_hashes: list[str]


class AnchorLeavesResponse(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    bucket_index: int
    leaves: list[AnchorLeaf] = Field(default_factory=list)


class AnchorSyncStats(BaseModel):
    node_id: str
    peer_node_id: str
    cycles_run: int = 0
    last_root_matched: bool = False
    last_cycle_buckets_synced: int = 0
    last_cycle_bytes: int = 0
```

Note: `NUM_BUCKETS` is defined here (next to `PROTOCOL_VERSION`, the other
protocol constant) and re-exported by `gin/federation/anchor_tree.py` in
Task 2 rather than redefined.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: PASS, all tests including the pre-existing ones in this file.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/schema.py tests/test_federation_schema.py
git commit -m "Anchor sync wire schema: root/buckets/leaves responses, never chunk text."
```

---

### Task 2: Pure Merkle bucket-tree logic

**Files:**
- Create: `gin/federation/anchor_tree.py`
- Test: `tests/test_anchor_tree.py`

**Interfaces:**
- Consumes: `AnchorLeaf` (Task 1).
- Produces: `NUM_BUCKETS` (re-exported from `schema.py`); `bucket_index(chunk_id: str) -> int`; `build_buckets(rows: list[AnchorLeaf]) -> dict[int, list[AnchorLeaf]]`; `bucket_hash(rows: list[AnchorLeaf]) -> str`; `all_bucket_hashes(rows: list[AnchorLeaf]) -> list[str]`; `root_hash(bucket_hashes: list[str]) -> str`; `AnchorDiff(added, changed, removed_chunk_ids)`; `diff_leaves(local: list[AnchorLeaf], remote: list[AnchorLeaf]) -> AnchorDiff`. Consumed by Tasks 5, 6, 8, 10.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anchor_tree.py`:

```python
"""Merkle bucket-tree logic: deterministic, order-independent, and a single
mutation perturbs exactly one of 16 buckets — the property that motivates
bucketing over a plain sorted-array tree."""
import pytest

from gin.federation.anchor_tree import (
    NUM_BUCKETS,
    all_bucket_hashes,
    bucket_index,
    build_buckets,
    diff_leaves,
    root_hash,
)
from gin.federation.schema import AnchorLeaf


def _leaf(chunk_id: str, content_hash: str = "h", outlet: str = "o", title: str = "t") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet=outlet, title=title)


def _corpus(n: int) -> list[AnchorLeaf]:
    return [_leaf(f"doc_{i}:{i % 3}", content_hash=f"h{i}") for i in range(n)]


def test_bucket_assignment_in_range():
    for i in range(50):
        assert 0 <= bucket_index(f"doc_{i}:0") < NUM_BUCKETS


def test_all_bucket_hashes_deterministic_and_order_independent():
    rows = _corpus(30)
    h1 = all_bucket_hashes(rows)
    h2 = all_bucket_hashes(list(reversed(rows)))
    assert h1 == h2
    assert len(h1) == NUM_BUCKETS


def test_root_hash_requires_exactly_num_buckets():
    with pytest.raises(ValueError):
        root_hash(["a", "b"])


def test_single_content_change_perturbs_exactly_one_bucket():
    rows = _corpus(40)
    before = all_bucket_hashes(rows)
    mutated = list(rows)
    mutated[5] = _leaf(mutated[5].chunk_id, content_hash="CHANGED")
    after = all_bucket_hashes(mutated)
    changed_indices = [i for i in range(NUM_BUCKETS) if before[i] != after[i]]
    assert len(changed_indices) == 1
    assert root_hash(before) != root_hash(after)


def test_single_insert_perturbs_exactly_one_bucket():
    rows = _corpus(40)
    before = all_bucket_hashes(rows)
    inserted = rows + [_leaf("brand_new_doc:0", content_hash="new")]
    after = all_bucket_hashes(inserted)
    changed_indices = [i for i in range(NUM_BUCKETS) if before[i] != after[i]]
    assert len(changed_indices) == 1


def test_single_delete_perturbs_exactly_one_bucket():
    rows = _corpus(40)
    before = all_bucket_hashes(rows)
    removed = rows[:10] + rows[11:]
    after = all_bucket_hashes(removed)
    changed_indices = [i for i in range(NUM_BUCKETS) if before[i] != after[i]]
    assert len(changed_indices) == 1


def test_unrelated_insert_leaves_other_buckets_contents_identical():
    rows = _corpus(40)
    before = build_buckets(rows)
    inserted = rows + [_leaf("brand_new_doc:0", content_hash="new")]
    after = build_buckets(inserted)
    new_bucket = bucket_index("brand_new_doc:0")
    for i in range(NUM_BUCKETS):
        if i == new_bucket:
            continue
        assert before[i] == after[i]


def test_diff_leaves_added_changed_removed():
    local = [_leaf("a", content_hash="1"), _leaf("b", content_hash="1")]
    remote = [_leaf("b", content_hash="2"), _leaf("c", content_hash="1")]
    diff = diff_leaves(local, remote)
    assert [r.chunk_id for r in diff.added] == ["c"]
    assert [r.chunk_id for r in diff.changed] == ["b"]
    assert diff.removed_chunk_ids == ["a"]


def test_build_buckets_sorts_within_bucket():
    rows = _corpus(10)
    buckets = build_buckets(rows)
    for bucket_rows in buckets.values():
        ids = [r.chunk_id for r in bucket_rows]
        assert ids == sorted(ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_tree.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.anchor_tree'`.

- [ ] **Step 3: Write `gin/federation/anchor_tree.py`**

```python
"""Pure Merkle-tree logic over anchor metadata — no I/O, no network, no DB.

16 fixed buckets keyed by the first hex digit of sha256(chunk_id) rather than
a plain sorted-array tree: a chunk's bucket membership is stable regardless
of what else is inserted or removed elsewhere in the set, so a single change
perturbs exactly one bucket hash (and the root), not everything after it in
sort order. See docs/superpowers/specs/2026-07-14-merkle-anchor-sync-design.md
for the full rationale.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .schema import NUM_BUCKETS, AnchorLeaf

_EMPTY_BUCKET_SENTINEL = hashlib.sha256(b"empty").hexdigest()


def bucket_index(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[0], 16)


def build_buckets(rows: list[AnchorLeaf]) -> dict[int, list[AnchorLeaf]]:
    buckets: dict[int, list[AnchorLeaf]] = {i: [] for i in range(NUM_BUCKETS)}
    for row in rows:
        buckets[bucket_index(row.chunk_id)].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda r: r.chunk_id)
    return buckets


def bucket_hash(rows: list[AnchorLeaf]) -> str:
    if not rows:
        return _EMPTY_BUCKET_SENTINEL
    payload = "|".join(
        f"{r.chunk_id}:{r.content_hash}:{r.outlet}:{r.title}" for r in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def all_bucket_hashes(rows: list[AnchorLeaf]) -> list[str]:
    buckets = build_buckets(rows)
    return [bucket_hash(buckets[i]) for i in range(NUM_BUCKETS)]


def root_hash(bucket_hashes: list[str]) -> str:
    if len(bucket_hashes) != NUM_BUCKETS:
        raise ValueError(f"expected {NUM_BUCKETS} bucket hashes, got {len(bucket_hashes)}")
    return hashlib.sha256("|".join(bucket_hashes).encode("utf-8")).hexdigest()


@dataclass
class AnchorDiff:
    added: list[AnchorLeaf] = field(default_factory=list)
    changed: list[AnchorLeaf] = field(default_factory=list)
    removed_chunk_ids: list[str] = field(default_factory=list)


def diff_leaves(local: list[AnchorLeaf], remote: list[AnchorLeaf]) -> AnchorDiff:
    """What changes if ``local`` becomes ``remote`` — added/changed/removed by chunk_id."""
    local_by_id = {r.chunk_id: r for r in local}
    remote_by_id = {r.chunk_id: r for r in remote}
    diff = AnchorDiff()
    for chunk_id, remote_row in remote_by_id.items():
        local_row = local_by_id.get(chunk_id)
        if local_row is None:
            diff.added.append(remote_row)
        elif local_row != remote_row:
            diff.changed.append(remote_row)
    for chunk_id in local_by_id:
        if chunk_id not in remote_by_id:
            diff.removed_chunk_ids.append(chunk_id)
    return diff
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_tree.py -v`
Expected: PASS, all 9 tests.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/anchor_tree.py tests/test_anchor_tree.py
git commit -m "Anchor tree: 16-bucket Merkle hashing, pure logic, no I/O."
```

---

### Task 3: `peer_anchors` table + anchor storage

**Files:**
- Modify: `docker/init-db.sql`
- Create: `gin/federation/anchor_store.py`
- Test: `tests/test_anchor_store.py`

**Interfaces:**
- Consumes: `AnchorLeaf` (Task 1); `connect()`/`transaction()` from `gin.corpus.db` (existing); `isolated_db`/`tmp_cold_root` fixtures from `tests/conftest.py` (existing); `ingest_path` from `gin.corpus.ingest` (existing).
- Produces: `PeerAnchorStore` Protocol with `all_rows(peer_node_id) -> list[AnchorLeaf]`, `bucket_rows(peer_node_id, bucket_index) -> list[AnchorLeaf]`, `replace_bucket(peer_node_id, bucket_index, rows) -> None`; `InMemoryPeerAnchorStore`; `PostgresPeerAnchorStore`; `local_anchor_rows(conn=None) -> list[AnchorLeaf]`. Consumed by Tasks 5, 6, 7, 8, 9.

- [ ] **Step 1: Add the `peer_anchors` table to `docker/init-db.sql`**

Append to the end of `docker/init-db.sql`:

```sql
CREATE TABLE IF NOT EXISTS peer_anchors (
    peer_node_id  TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    outlet        TEXT NOT NULL,
    title         TEXT NOT NULL,
    bucket_index  SMALLINT NOT NULL,
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (peer_node_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_peer_anchors_bucket
    ON peer_anchors(peer_node_id, bucket_index);
```

This table stores THIS node's cached copy of a PEER's anchor set — never
this node's own chunks (those already live in `chunks`/`documents`).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_anchor_store.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.anchor_store'`.

- [ ] **Step 4: Write `gin/federation/anchor_store.py`**

```python
"""Where each node's cached copy of a peer's anchor set lives, and how a
node reads its OWN chunks as anchors to serve to a peer.

InMemoryPeerAnchorStore backs unit/integration tests; PostgresPeerAnchorStore
is the production cache, backed by the peer_anchors table. local_anchor_rows
reads THIS node's own chunks/documents — the set a peer's sync loop pulls
from. Never chunk text, in either direction.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import psycopg

from gin.corpus.db import connect, transaction

from .schema import AnchorLeaf


@runtime_checkable
class PeerAnchorStore(Protocol):
    def all_rows(self, peer_node_id: str) -> list[AnchorLeaf]: ...
    def bucket_rows(self, peer_node_id: str, bucket_index: int) -> list[AnchorLeaf]: ...
    def replace_bucket(
        self, peer_node_id: str, bucket_index: int, rows: list[AnchorLeaf]
    ) -> None: ...


class InMemoryPeerAnchorStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[int, list[AnchorLeaf]]] = {}

    def all_rows(self, peer_node_id: str) -> list[AnchorLeaf]:
        buckets = self._data.get(peer_node_id, {})
        return [row for rows in buckets.values() for row in rows]

    def bucket_rows(self, peer_node_id: str, bucket_index: int) -> list[AnchorLeaf]:
        return list(self._data.get(peer_node_id, {}).get(bucket_index, []))

    def replace_bucket(
        self, peer_node_id: str, bucket_index: int, rows: list[AnchorLeaf]
    ) -> None:
        self._data.setdefault(peer_node_id, {})[bucket_index] = list(rows)


class PostgresPeerAnchorStore:
    """Opens a fresh connection per call — matches the corpus tier's
    connect()-per-call convention (gin/corpus/fingerprint.py)."""

    def all_rows(self, peer_node_id: str) -> list[AnchorLeaf]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, content_hash, outlet, title FROM peer_anchors "
                "WHERE peer_node_id = %s",
                (peer_node_id,),
            ).fetchall()
        return [AnchorLeaf(chunk_id=r[0], content_hash=r[1], outlet=r[2], title=r[3]) for r in rows]

    def bucket_rows(self, peer_node_id: str, bucket_index: int) -> list[AnchorLeaf]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, content_hash, outlet, title FROM peer_anchors "
                "WHERE peer_node_id = %s AND bucket_index = %s",
                (peer_node_id, bucket_index),
            ).fetchall()
        return [AnchorLeaf(chunk_id=r[0], content_hash=r[1], outlet=r[2], title=r[3]) for r in rows]

    def replace_bucket(
        self, peer_node_id: str, bucket_index: int, rows: list[AnchorLeaf]
    ) -> None:
        with transaction() as conn:
            conn.execute(
                "DELETE FROM peer_anchors WHERE peer_node_id = %s AND bucket_index = %s",
                (peer_node_id, bucket_index),
            )
            for row in rows:
                conn.execute(
                    "INSERT INTO peer_anchors "
                    "(peer_node_id, chunk_id, content_hash, outlet, title, bucket_index) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (peer_node_id, row.chunk_id, row.content_hash, row.outlet,
                     row.title, bucket_index),
                )


def local_anchor_rows(conn: Optional[psycopg.Connection] = None) -> list[AnchorLeaf]:
    """This node's own chunks as anchors — what a peer's sync loop may read."""
    if conn is None:
        with connect() as conn:
            return local_anchor_rows(conn)
    rows = conn.execute(
        "SELECT c.chunk_id, c.content_hash, d.outlet, d.title "
        "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
    ).fetchall()
    return [AnchorLeaf(chunk_id=r[0], content_hash=r[1], outlet=r[2], title=r[3]) for r in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_store.py -v -m "not integration"`
Expected: PASS (3 in-memory tests). The 2 `integration`-marked tests need
Postgres; run them too if `docker compose up -d` is running in `docker/`:

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_store.py -v`
Expected: PASS (5 tests), or the 2 integration tests SKIP with "Postgres not
available" if the container isn't running.

- [ ] **Step 6: Commit**

```bash
git add docker/init-db.sql gin/federation/anchor_store.py tests/test_anchor_store.py
git commit -m "peer_anchors table + PeerAnchorStore: cache a peer's anchor set, never its text."
```

---

### Task 4: `PeerClient` anchor-fetch methods

**Files:**
- Modify: `gin/federation/client.py`
- Test: `tests/test_federation_client.py`

**Interfaces:**
- Consumes: `AnchorRootResponse`, `AnchorBucketsResponse`, `AnchorLeavesResponse` (Task 1); existing `PeerConfig`, `HttpPeerClient.__init__`, `PeerUnreachable`.
- Produces: `PeerClient.get_anchor_root(peer) -> AnchorRootResponse`; `PeerClient.get_anchor_buckets(peer) -> AnchorBucketsResponse`; `PeerClient.get_anchor_bucket(peer, index: int) -> AnchorLeavesResponse`, all raising `PeerUnreachable` on transport failure. Consumed by Task 5 (`sync_once`) and Task 8.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_federation_client.py`:

```python
from gin.federation.schema import AnchorBucketsResponse, AnchorLeaf, AnchorLeavesResponse, AnchorRootResponse


def test_get_anchor_root_parses_and_sends_bearer():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = AnchorRootResponse(node_id="node_b", root_hash="abc", leaf_count=50)
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient("s3cret", transport=httpx.MockTransport(handler))
    out = client.get_anchor_root(PEER)
    assert out.root_hash == "abc"
    assert seen["url"] == "http://peer-b/v1/federated/anchors/root"
    assert seen["auth"] == "Bearer s3cret"


def test_get_anchor_buckets_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        body = AnchorBucketsResponse(node_id="node_b", bucket_hashes=["h"] * 16)
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    out = client.get_anchor_buckets(PEER)
    assert len(out.bucket_hashes) == 16


def test_get_anchor_bucket_hits_indexed_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        body = AnchorLeavesResponse(
            node_id="node_b", bucket_index=7,
            leaves=[AnchorLeaf(chunk_id="c", content_hash="h", outlet="o", title="t")],
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    out = client.get_anchor_bucket(PEER, 7)
    assert seen["url"] == "http://peer-b/v1/federated/anchors/bucket/7"
    assert out.leaves[0].chunk_id == "c"


def test_anchor_endpoint_http_error_maps_to_peer_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.get_anchor_root(PEER)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: FAIL — `AttributeError: 'HttpPeerClient' object has no attribute 'get_anchor_root'`.

- [ ] **Step 3: Extend `gin/federation/client.py`**

Modify the import line to add the anchor response types:

```python
from .schema import (
    AnchorBucketsResponse,
    AnchorLeavesResponse,
    AnchorRootResponse,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
)
```

Extend the `PeerClient` Protocol (add these lines inside the existing `class PeerClient(Protocol):` block, after `query`):

```python
    def get_anchor_root(self, peer: PeerConfig) -> AnchorRootResponse: ...
    def get_anchor_buckets(self, peer: PeerConfig) -> AnchorBucketsResponse: ...
    def get_anchor_bucket(self, peer: PeerConfig, index: int) -> AnchorLeavesResponse: ...
```

Add these methods to `HttpPeerClient` (after the existing `query` method):

```python
    def get_anchor_root(self, peer: PeerConfig) -> AnchorRootResponse:
        return self._get(peer, "/v1/federated/anchors/root", AnchorRootResponse)

    def get_anchor_buckets(self, peer: PeerConfig) -> AnchorBucketsResponse:
        return self._get(peer, "/v1/federated/anchors/buckets", AnchorBucketsResponse)

    def get_anchor_bucket(self, peer: PeerConfig, index: int) -> AnchorLeavesResponse:
        return self._get(peer, f"/v1/federated/anchors/bucket/{index}", AnchorLeavesResponse)

    def _get(self, peer: PeerConfig, path: str, model_cls):
        try:
            with httpx.Client(
                transport=self._transport, timeout=self._timeout
            ) as client:
                r = client.get(f"{peer.url}{path}", headers=self._headers)
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise PeerUnreachable(peer, exc) from exc
        return model_cls.model_validate(r.json())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: PASS, all tests including the pre-existing ones in this file.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/client.py tests/test_federation_client.py
git commit -m "PeerClient: anchor root/buckets/bucket GET methods alongside the existing query POST."
```

---

### Task 5: `sync_once` — one anchor-sync cycle

**Files:**
- Create: `gin/federation/anchor_sync.py`
- Test: `tests/test_anchor_sync.py`

**Interfaces:**
- Consumes: `PeerAnchorStore`, `InMemoryPeerAnchorStore` (Task 3); `NUM_BUCKETS`, `all_bucket_hashes`, `root_hash` (Task 2); `PeerClient` Protocol, `PeerConfig` (Task 4 / existing).
- Produces: `SyncStats(root_matched: bool, buckets_synced: int, bytes_transferred: int)`; `sync_once(peer: PeerConfig, peer_client: PeerClient, store: PeerAnchorStore) -> SyncStats`. Consumed by Task 7 (`run_forever`) and Task 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anchor_sync.py`:

```python
"""sync_once(): root-match short-circuit, bucket-level drill-down, and the
bandwidth property — matched buckets are never fetched."""
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_sync import sync_once
from gin.federation.anchor_tree import all_bucket_hashes, bucket_index, root_hash
from gin.federation.config import PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
)

PEER = PeerConfig(node_id="node_b", url="http://peer-b")


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _corpus(n: int) -> list[AnchorLeaf]:
    return [_leaf(f"doc_{i}:0", content_hash=f"h{i}") for i in range(n)]


class FakePeerClient:
    """Serves a fixed peer corpus over the anchor endpoints; counts bucket fetches."""

    def __init__(self, rows: list[AnchorLeaf]) -> None:
        self.rows = rows
        self.bucket_fetch_calls: list[int] = []

    def get_anchor_root(self, peer):
        return AnchorRootResponse(
            node_id=peer.node_id, root_hash=root_hash(all_bucket_hashes(self.rows)),
            leaf_count=len(self.rows),
        )

    def get_anchor_buckets(self, peer):
        return AnchorBucketsResponse(node_id=peer.node_id, bucket_hashes=all_bucket_hashes(self.rows))

    def get_anchor_bucket(self, peer, index):
        self.bucket_fetch_calls.append(index)
        buckets: dict[int, list[AnchorLeaf]] = {}
        for row in self.rows:
            buckets.setdefault(bucket_index(row.chunk_id), []).append(row)
        return AnchorLeavesResponse(node_id=peer.node_id, bucket_index=index, leaves=buckets.get(index, []))


def test_first_sync_bootstraps_full_cache():
    rows = _corpus(40)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    stats = sync_once(PEER, client, store)
    assert stats.root_matched is False
    assert {r.chunk_id for r in store.all_rows("node_b")} == {r.chunk_id for r in rows}
    assert stats.buckets_synced == len(set(client.bucket_fetch_calls))


def test_no_op_sync_after_convergence_fetches_no_buckets():
    rows = _corpus(40)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    sync_once(PEER, client, store)  # bootstrap
    client.bucket_fetch_calls.clear()
    stats = sync_once(PEER, client, store)
    assert stats.root_matched is True
    assert stats.buckets_synced == 0
    assert client.bucket_fetch_calls == []


def test_single_chunk_change_syncs_exactly_one_bucket():
    rows = _corpus(40)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    sync_once(PEER, client, store)  # bootstrap
    client.bucket_fetch_calls.clear()

    client.rows = list(rows)
    client.rows[5] = _leaf(client.rows[5].chunk_id, content_hash="CHANGED")
    stats = sync_once(PEER, client, store)

    assert stats.root_matched is False
    assert stats.buckets_synced == 1
    assert len(set(client.bucket_fetch_calls)) == 1
    changed_row = next(
        r for r in store.all_rows("node_b") if r.chunk_id == client.rows[5].chunk_id
    )
    assert changed_row.content_hash == "CHANGED"


def test_no_op_cycle_transfers_far_fewer_bytes_than_bootstrap():
    rows = _corpus(200)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    bootstrap_stats = sync_once(PEER, client, store)
    noop_stats = sync_once(PEER, client, store)
    assert noop_stats.bytes_transferred < bootstrap_stats.bytes_transferred / 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.anchor_sync'`.

- [ ] **Step 3: Write `gin/federation/anchor_sync.py`**

```python
"""One anchor-sync cycle: compare cached root to the peer's, and if they
differ, drill down to just the mismatched buckets. Bandwidth is the point —
matched buckets are never fetched.
"""
from __future__ import annotations

from dataclasses import dataclass

from .anchor_store import PeerAnchorStore
from .anchor_tree import NUM_BUCKETS, all_bucket_hashes, root_hash
from .client import PeerClient
from .config import PeerConfig


@dataclass
class SyncStats:
    root_matched: bool
    buckets_synced: int
    bytes_transferred: int


def _response_bytes(model) -> int:
    return len(model.model_dump_json().encode("utf-8"))


def sync_once(
    peer: PeerConfig, peer_client: PeerClient, store: PeerAnchorStore
) -> SyncStats:
    root_resp = peer_client.get_anchor_root(peer)
    bytes_transferred = _response_bytes(root_resp)
    local_rows = store.all_rows(peer.node_id)
    local_root = root_hash(all_bucket_hashes(local_rows))

    if local_root == root_resp.root_hash:
        return SyncStats(root_matched=True, buckets_synced=0, bytes_transferred=bytes_transferred)

    buckets_resp = peer_client.get_anchor_buckets(peer)
    bytes_transferred += _response_bytes(buckets_resp)
    local_bucket_hashes = all_bucket_hashes(local_rows)
    mismatched = [
        i for i in range(NUM_BUCKETS)
        if local_bucket_hashes[i] != buckets_resp.bucket_hashes[i]
    ]

    for i in mismatched:
        leaves_resp = peer_client.get_anchor_bucket(peer, i)
        bytes_transferred += _response_bytes(leaves_resp)
        store.replace_bucket(peer.node_id, i, leaves_resp.leaves)

    return SyncStats(
        root_matched=False, buckets_synced=len(mismatched), bytes_transferred=bytes_transferred
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_sync.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/anchor_sync.py tests/test_anchor_sync.py
git commit -m "sync_once: root-match short-circuit, then fetch only mismatched buckets."
```

---

### Task 6: Anchor read endpoints on the node server

**Files:**
- Modify: `gin/federation/server.py`
- Test: `tests/test_anchor_endpoints.py` (new)

**Interfaces:**
- Consumes: `build_buckets`, `all_bucket_hashes`, `root_hash` (Task 2); `AnchorRootResponse`, `AnchorBucketsResponse`, `AnchorLeavesResponse`, `AnchorSyncStats` (Task 1); existing `create_app` signature.
- Produces: `create_app(..., local_anchor_rows: Optional[Callable[[], list[AnchorLeaf]]] = None)` — new keyword-only parameter, default `None` (existing callers unaffected); four new routes: `GET /v1/federated/anchors/root`, `GET /v1/federated/anchors/buckets`, `GET /v1/federated/anchors/bucket/{index}`, `GET /v1/federated/anchors/sync_stats`, all behind the existing bearer-auth dependency. Consumed by Task 7 (adds the background loop that populates `sync_stats`) and Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anchor_endpoints.py`:

```python
"""Read-only anchor endpoints: root/buckets/bucket/sync_stats, auth-gated
like the query endpoint, backed by an injected local_anchor_rows callable."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.anchor_tree import NUM_BUCKETS, all_bucket_hashes, bucket_index, root_hash
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
)
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    shared_secret="s3cret", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_b", url="http://peer-b"),),
)
AUTH = {"Authorization": "Bearer s3cret"}


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _rows() -> list[AnchorLeaf]:
    return [
        AnchorLeaf(chunk_id="n1_doc_001:0", content_hash="h1", outlet="node_1", title="t1"),
        AnchorLeaf(chunk_id="n1_doc_002:0", content_hash="h2", outlet="node_1", title="t2"),
    ]


def test_anchors_root_matches_pure_computation():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root", headers=AUTH)
    resp = AnchorRootResponse.model_validate(r.json())
    assert resp.root_hash == root_hash(all_bucket_hashes(_rows()))
    assert resp.leaf_count == 2


def test_anchors_root_requires_auth():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root")
    assert r.status_code == 401


def test_anchors_buckets_has_16_entries():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/buckets", headers=AUTH)
    resp = AnchorBucketsResponse.model_validate(r.json())
    assert len(resp.bucket_hashes) == NUM_BUCKETS


def test_anchors_bucket_returns_only_that_bucket():
    rows = _rows()
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=lambda: rows)
    client = TestClient(app)
    idx = bucket_index(rows[0].chunk_id)
    r = client.get(f"/v1/federated/anchors/bucket/{idx}", headers=AUTH)
    resp = AnchorLeavesResponse.model_validate(r.json())
    assert rows[0].chunk_id in {leaf.chunk_id for leaf in resp.leaves}
    assert resp.bucket_index == idx


def test_anchors_default_empty_when_not_configured():
    app = create_app(CFG, answer_fn=_grounded)  # no local_anchor_rows injected
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root", headers=AUTH)
    resp = AnchorRootResponse.model_validate(r.json())
    assert resp.leaf_count == 0


def test_sync_stats_defaults_before_any_cycle():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/sync_stats", headers=AUTH)
    resp = AnchorSyncStats.model_validate(r.json())
    assert resp.node_id == "node_a"
    assert resp.peer_node_id == "node_b"
    assert resp.cycles_run == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_endpoints.py -v`
Expected: FAIL — `TypeError: create_app() got an unexpected keyword argument 'local_anchor_rows'`.

- [ ] **Step 3: Modify `gin/federation/server.py`**

Update the imports at the top of the file:

```python
from typing import Callable, Optional

from .anchor_tree import all_bucket_hashes, build_buckets, root_hash
from .client import PeerClient
from .config import NodeConfig
from .router import AnswerFn, answer_or_delegate
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
)
from .service import claims_to_wire
```

Update the `create_app` signature and body — add `local_anchor_rows` as a new
keyword-only parameter with a default, compute `anchor_rows_fn` and
`sync_stats` right after `fingerprint`, and add the four new routes just
before the existing `return app`:

```python
def create_app(
    config: NodeConfig,
    *,
    answer_fn: AnswerFn,
    peer_client: Optional[PeerClient] = None,
    corpus_fingerprint: Optional[dict] = None,
    local_anchor_rows: Optional[Callable[[], list[AnchorLeaf]]] = None,
) -> FastAPI:
    app = FastAPI(title=f"GIN federation node {config.node_id}")
    fingerprint = corpus_fingerprint or {}
    anchor_rows_fn = local_anchor_rows or (lambda: [])
    sync_stats = AnchorSyncStats(
        node_id=config.node_id,
        peer_node_id=config.peers[0].node_id if config.peers else "",
    )
```

(The rest of the existing body — `_check_auth`, `_refusal`,
`federated_query` — is unchanged.) Add these four routes immediately before
the final `return app`:

```python
    @app.get("/v1/federated/anchors/root", response_model=AnchorRootResponse)
    def anchors_root(_: None = Depends(_check_auth)) -> AnchorRootResponse:
        rows = anchor_rows_fn()
        return AnchorRootResponse(
            node_id=config.node_id,
            root_hash=root_hash(all_bucket_hashes(rows)),
            leaf_count=len(rows),
        )

    @app.get("/v1/federated/anchors/buckets", response_model=AnchorBucketsResponse)
    def anchors_buckets(_: None = Depends(_check_auth)) -> AnchorBucketsResponse:
        rows = anchor_rows_fn()
        return AnchorBucketsResponse(node_id=config.node_id, bucket_hashes=all_bucket_hashes(rows))

    @app.get(
        "/v1/federated/anchors/bucket/{index}", response_model=AnchorLeavesResponse
    )
    def anchors_bucket(index: int, _: None = Depends(_check_auth)) -> AnchorLeavesResponse:
        rows = anchor_rows_fn()
        buckets = build_buckets(rows)
        return AnchorLeavesResponse(
            node_id=config.node_id, bucket_index=index, leaves=buckets.get(index, [])
        )

    @app.get("/v1/federated/anchors/sync_stats", response_model=AnchorSyncStats)
    def anchors_sync_stats(_: None = Depends(_check_auth)) -> AnchorSyncStats:
        return sync_stats

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_endpoints.py -v`
Expected: PASS, all 6 tests.

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py tests/test_federation_loop.py -v`
Expected: PASS, unchanged — confirms the new parameter is additive.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/server.py tests/test_anchor_endpoints.py
git commit -m "Anchor read endpoints: root/buckets/bucket/sync_stats, same auth as the query route."
```

---

### Task 7: Background sync loop + config interval

**Files:**
- Modify: `gin/federation/config.py`
- Modify: `gin/federation/anchor_sync.py`
- Modify: `gin/federation/server.py`
- Test: `tests/test_federation_config.py`, `tests/test_anchor_endpoints.py`

**Interfaces:**
- Consumes: `sync_once` (Task 5); `AnchorSyncStats` (Task 1); existing `NodeConfig`, `load_node_config`.
- Produces: `NodeConfig.anchor_sync_interval_s: float` (default `30.0`, YAML key `anchor_sync_interval_s`); `run_forever(peer, peer_client, store, interval_s, stats) -> None` (async, runs forever until cancelled); `create_app(..., peer_anchor_store: Optional[PeerAnchorStore] = None)` — when provided (and `peer_client` set and `config.peers` non-empty), the FastAPI lifespan starts `run_forever` as a background task on startup and cancels it on shutdown. Consumed by Tasks 8, 9.

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_federation_config.py`:

```python
def test_anchor_sync_interval_default(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.anchor_sync_interval_s == 30.0


def test_anchor_sync_interval_override(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML + "anchor_sync_interval_s: 5\n", encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.anchor_sync_interval_s == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: FAIL — `AttributeError: 'NodeConfig' object has no attribute 'anchor_sync_interval_s'`.

- [ ] **Step 3: Add the field to `gin/federation/config.py`**

Add `anchor_sync_interval_s: float = 30.0` as the last field of `NodeConfig`
(after `chat_template`):

```python
@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    host: str
    port: int
    database_url: str
    cold_path: str
    model_path: str
    n_gpu_layers: int
    n_ctx: int
    shared_secret: str
    peer_timeout_s: float
    peers: tuple[PeerConfig, ...]
    chat_template: str = "mistral"
    anchor_sync_interval_s: float = 30.0
```

Add the parse line inside `load_node_config`, right after `chat_template=`:

```python
        chat_template=raw.get("chat_template", "mistral"),
        anchor_sync_interval_s=float(raw.get("anchor_sync_interval_s", 30.0)),
```

- [ ] **Step 4: Run config tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: PASS, all tests including pre-existing ones.

- [ ] **Step 5: Add `run_forever` to `gin/federation/anchor_sync.py`**

Add these imports at the top of the file (alongside the existing ones):

```python
import asyncio
import logging

from .schema import AnchorSyncStats

logger = logging.getLogger(__name__)
```

Append `run_forever` at the end of the file:

```python
async def run_forever(
    peer: PeerConfig,
    peer_client: PeerClient,
    store: PeerAnchorStore,
    interval_s: float,
    stats: AnchorSyncStats,
) -> None:
    """One sync_once() per interval, forever, until the task is cancelled.
    Any failure (unreachable peer, transport error) is logged and skipped —
    this is background maintenance and must never affect query answering."""
    while True:
        try:
            result = await asyncio.to_thread(sync_once, peer, peer_client, store)
            stats.cycles_run += 1
            stats.last_root_matched = result.root_matched
            stats.last_cycle_buckets_synced = result.buckets_synced
            stats.last_cycle_bytes = result.bytes_transferred
        except Exception:
            logger.exception("anchor sync with %s failed", peer.node_id)
        await asyncio.sleep(interval_s)
```

- [ ] **Step 6: Write the failing lifespan test**

Append to `tests/test_anchor_endpoints.py`:

```python
import asyncio

from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_tree import all_bucket_hashes, root_hash
from gin.federation.schema import AnchorBucketsResponse


class StubPeerClient:
    """Serves one fixed peer row set over the anchor GET methods only."""

    def __init__(self, rows):
        self.rows = rows

    def get_anchor_root(self, peer):
        return AnchorRootResponse(
            node_id=peer.node_id, root_hash=root_hash(all_bucket_hashes(self.rows)),
            leaf_count=len(self.rows),
        )

    def get_anchor_buckets(self, peer):
        return AnchorBucketsResponse(node_id=peer.node_id, bucket_hashes=all_bucket_hashes(self.rows))

    def get_anchor_bucket(self, peer, index):
        from gin.federation.anchor_tree import bucket_index as _bi
        matches = [r for r in self.rows if _bi(r.chunk_id) == index]
        return AnchorLeavesResponse(node_id=peer.node_id, bucket_index=index, leaves=matches)


def test_lifespan_starts_and_stops_background_sync():
    peer_rows = [AnchorLeaf(chunk_id="p:0", content_hash="h", outlet="o", title="t")]
    store = InMemoryPeerAnchorStore()
    app = create_app(
        CFG, answer_fn=_grounded, peer_client=StubPeerClient(peer_rows),
        peer_anchor_store=store,
    )

    async def _run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.2)

    asyncio.run(_run())
    assert {r.chunk_id for r in store.all_rows("node_b")} == {"p:0"}


def test_no_background_task_without_peer_anchor_store():
    # Existing callers (no peer_anchor_store) must see no background activity.
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)

    async def _run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.1)

    asyncio.run(_run())  # must not raise, must not hang
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_endpoints.py -v`
Expected: FAIL — `TypeError: create_app() got an unexpected keyword argument 'peer_anchor_store'`.

- [ ] **Step 8: Wire the lifespan into `gin/federation/server.py`**

Add imports at the top:

```python
import asyncio
import contextlib
from contextlib import asynccontextmanager

from .anchor_store import PeerAnchorStore
from .anchor_sync import run_forever
```

Replace the `create_app` signature and its first few lines (the
`app = FastAPI(...)` line moves down, after `lifespan` is defined):

```python
def create_app(
    config: NodeConfig,
    *,
    answer_fn: AnswerFn,
    peer_client: Optional[PeerClient] = None,
    corpus_fingerprint: Optional[dict] = None,
    local_anchor_rows: Optional[Callable[[], list[AnchorLeaf]]] = None,
    peer_anchor_store: Optional[PeerAnchorStore] = None,
) -> FastAPI:
    fingerprint = corpus_fingerprint or {}
    anchor_rows_fn = local_anchor_rows or (lambda: [])
    sync_stats = AnchorSyncStats(
        node_id=config.node_id,
        peer_node_id=config.peers[0].node_id if config.peers else "",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if peer_anchor_store is not None and peer_client is not None and config.peers:
            task = asyncio.create_task(
                run_forever(
                    config.peers[0], peer_client, peer_anchor_store,
                    config.anchor_sync_interval_s, sync_stats,
                )
            )
        yield
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title=f"GIN federation node {config.node_id}", lifespan=lifespan)
```

(Everything below this — `_check_auth`, `_refusal`, `federated_query`, the
anchor GET routes from Task 6, `return app` — is unchanged.)

- [ ] **Step 9: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_endpoints.py -v`
Expected: PASS, all 8 tests.

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py tests/test_federation_loop.py -v`
Expected: PASS, unchanged.

- [ ] **Step 10: Commit**

```bash
git add gin/federation/config.py gin/federation/anchor_sync.py gin/federation/server.py \
        tests/test_federation_config.py tests/test_anchor_endpoints.py
git commit -m "Background anchor-sync loop: FastAPI lifespan runs run_forever when a store is configured."
```

---

### Task 8: Real-socket sync loop test (two uvicorn nodes, no model, no DB)

**Files:**
- Create: `tests/test_anchor_sync_loop.py`

**Interfaces:**
- Consumes: `create_app` with `peer_anchor_store` (Task 7); `HttpPeerClient` (Task 4); `InMemoryPeerAnchorStore` (Task 3); `NodeConfig.anchor_sync_interval_s` (Task 7).
- Produces: nothing new — this is a proof test that the full loop converges over a real HTTP boundary, mirroring `tests/test_federation_loop.py`'s role for the query path.

- [ ] **Step 1: Write the test**

Create `tests/test_anchor_sync_loop.py`:

```python
"""Real-socket anchor sync loop: two uvicorn nodes, in-memory stores, no DB,
no model — the background task actually runs and converges the cache."""
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import AnchorLeaf
from gin.federation.server import create_app

SECRET = "anchor-loop-secret"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _config(node_id: str, port: int, peer: PeerConfig, interval_s: float = 0.05) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=10.0, peers=(peer,),
        anchor_sync_interval_s=interval_s,
    )


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


@pytest.fixture
def two_nodes():
    port_a, port_b = _free_port(), _free_port()
    peer_client = HttpPeerClient(SECRET, timeout_s=10.0)
    b_rows = [_leaf(f"doc_{i}:0", content_hash=f"h{i}") for i in range(20)]
    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"http://127.0.0.1:{port_b}"))
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"http://127.0.0.1:{port_a}"))
    store_a = InMemoryPeerAnchorStore()  # A's cache of B
    app_a = create_app(
        cfg_a, answer_fn=_grounded, peer_client=peer_client,
        local_anchor_rows=lambda: [], peer_anchor_store=store_a,
    )
    app_b = create_app(
        cfg_b, answer_fn=_grounded, peer_client=peer_client,
        local_anchor_rows=lambda: b_rows,
    )
    server_a = _serve(app_a, port_a)
    server_b = _serve(app_b, port_b)
    yield store_a, b_rows, f"http://127.0.0.1:{port_a}"
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def test_background_loop_converges_cache_to_peer_ground_truth(two_nodes):
    store_a, b_rows, _ = two_nodes
    deadline = time.monotonic() + 5
    expected = {r.chunk_id for r in b_rows}
    while time.monotonic() < deadline:
        if {r.chunk_id for r in store_a.all_rows("node_b")} == expected:
            break
        time.sleep(0.05)
    assert {r.chunk_id for r in store_a.all_rows("node_b")} == expected


def test_sync_stats_endpoint_reflects_cycles(two_nodes):
    _, _, url_a = two_nodes
    time.sleep(0.5)
    r = httpx.get(
        f"{url_a}/v1/federated/anchors/sync_stats",
        headers={"Authorization": f"Bearer {SECRET}"}, timeout=5.0,
    )
    assert r.json()["cycles_run"] >= 1
```

- [ ] **Step 2: Run the test**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_sync_loop.py -v`
Expected: PASS, both tests (the loop converges within the 5s deadline at a
0.05s interval).

- [ ] **Step 3: Commit**

```bash
git add tests/test_anchor_sync_loop.py
git commit -m "Prove the anchor sync loop over real localhost sockets, model-free."
```

---

### Task 9: Node entrypoint wiring + per-node config

**Files:**
- Modify: `scripts/node_serve.py`
- Modify: `config/node_a.yaml`
- Modify: `config/node_b.yaml`

**Interfaces:**
- Consumes: `PostgresPeerAnchorStore`, `local_anchor_rows` (Task 3); `create_app(..., local_anchor_rows=, peer_anchor_store=)` (Tasks 6–7).
- Produces: `python scripts/node_serve.py --config config/node_a.yaml` now also serves the anchor endpoints and runs the background sync loop against its configured peer.

- [ ] **Step 1: Wire `scripts/node_serve.py`**

Add to the import block inside `main()` (alongside the existing federation
imports):

```python
    from gin.federation.anchor_store import PostgresPeerAnchorStore, local_anchor_rows
```

Update the `create_app(...)` call to add the two new arguments:

```python
    app = create_app(
        config,
        answer_fn=lambda q: answer_query(q, llm, arm_cfg),
        peer_client=HttpPeerClient(config.shared_secret, config.peer_timeout_s),
        corpus_fingerprint=fingerprint,
        local_anchor_rows=local_anchor_rows,
        peer_anchor_store=PostgresPeerAnchorStore(),
    )
```

- [ ] **Step 2: Smoke-test the entrypoint**

Run: `./venv/Scripts/python.exe scripts/node_serve.py --help`
Expected: usage text, exit 0 (no model/DB touched — the new imports are
inside `main()`, same as the existing federation imports).

- [ ] **Step 3: Add `anchor_sync_interval_s` to both node configs**

Append to `config/node_a.yaml`:

```yaml
anchor_sync_interval_s: 10
```

Append to `config/node_b.yaml`:

```yaml
anchor_sync_interval_s: 10
```

(10s keeps the live eval in Task 10 fast without hammering Postgres; the
`NodeConfig` default of 30s is for the un-configured/production-default
case.)

- [ ] **Step 4: Apply the schema update to the already-provisioned databases**

The `peer_anchors` table was added to `docker/init-db.sql` in Task 3;
`gin_node_a`/`gin_node_b` were created before that change, so re-apply the
(idempotent, `IF NOT EXISTS`) schema:

Run:
```bash
docker exec -i gin-postgres psql -U gin -d gin_node_a < docker/init-db.sql
docker exec -i gin-postgres psql -U gin -d gin_node_b < docker/init-db.sql
```
Expected: `CREATE TABLE`/`CREATE INDEX` (or harmless `NOTICE: ... already
exists, skipping` for the tables/columns/indexes that were already there).

- [ ] **Step 5: Commit**

```bash
git add scripts/node_serve.py config/node_a.yaml config/node_b.yaml
git commit -m "Wire anchor sync into node_serve.py: PostgresPeerAnchorStore + local_anchor_rows."
```

---

### Task 10: Live eval driver + documentation + final validation

**Files:**
- Create: `scripts/eval_anchor_sync.py`
- Modify: `architecture.md`
- Modify: `README.md`
- Modify: `docs/GIN_Node_Architecture_v1.md`

**Interfaces:**
- Consumes: node A's `/v1/federated/anchors/sync_stats` endpoint (Task 6/7, live over HTTP); direct Postgres access to `gin_node_a.peer_anchors` and `gin_node_b.chunks`/`documents` (same "driver legitimately holds both DBs" pattern as `scripts/eval_federation.py`); `AnchorLeaf`, `AnchorLeavesResponse` (Task 1).
- Produces: `data/eval_runs/<ts>/anchor_sync_metrics.json` with `correctness_pass`, `correctness_pass_after_mutation`, `no_op_cycle_bytes`, `mutation_cycle_bytes`, `bandwidth_pass`.

- [ ] **Step 1: Write `scripts/eval_anchor_sync.py`**

```python
"""Measure the Merkle anchor sync bar against two live node processes.

Prereqs (three terminals):
    bash scripts/federation_db_setup.sh                        # once
    python scripts/node_serve.py --config config/node_b.yaml    # terminal 1
    python scripts/node_serve.py --config config/node_a.yaml    # terminal 2
    python scripts/eval_anchor_sync.py                          # terminal 3

Bar (spec): 0 diff between node A's cache of B and B's ground truth after
convergence; a no-op cycle transfers O(1) bytes; a cycle following a
single-chunk mutation transfers far less than a full-corpus resync would.
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
import psycopg

from gin.federation.schema import AnchorLeaf, AnchorLeavesResponse, AnchorSyncStats

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def _ground_truth(database_url: str) -> dict[str, dict]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT c.chunk_id, c.content_hash, d.outlet, d.title "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
        ).fetchall()
    return {r[0]: {"content_hash": r[1], "outlet": r[2], "title": r[3]} for r in rows}


def _peer_anchors(database_url: str, peer_node_id: str) -> dict[str, dict]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT chunk_id, content_hash, outlet, title FROM peer_anchors "
            "WHERE peer_node_id = %s", (peer_node_id,),
        ).fetchall()
    return {r[0]: {"content_hash": r[1], "outlet": r[2], "title": r[3]} for r in rows}


def _diff(local: dict, remote: dict) -> dict:
    return {
        "added": [cid for cid in remote if cid not in local],
        "changed": [cid for cid in remote if cid in local and local[cid] != remote[cid]],
        "removed": [cid for cid in local if cid not in remote],
    }


def _diff_is_empty(diff: dict) -> bool:
    return not diff["added"] and not diff["changed"] and not diff["removed"]


def _wait_for_convergence(node_a_db: str, node_b_db: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    diff = {"added": ["<not yet checked>"], "changed": [], "removed": []}
    while time.monotonic() < deadline:
        diff = _diff(_peer_anchors(node_a_db, "node_b"), _ground_truth(node_b_db))
        if _diff_is_empty(diff):
            return diff
        time.sleep(1.0)
    return diff


def _sync_stats(node_a_url: str, secret: str) -> AnchorSyncStats:
    r = httpx.get(
        f"{node_a_url}/v1/federated/anchors/sync_stats",
        headers={"Authorization": f"Bearer {secret}"}, timeout=10.0,
    )
    r.raise_for_status()
    return AnchorSyncStats.model_validate(r.json())


def _full_corpus_bytes(truth: dict[str, dict]) -> int:
    leaves = [AnchorLeaf(chunk_id=cid, **fields) for cid, fields in truth.items()]
    resp = AnchorLeavesResponse(node_id="node_b", bucket_index=-1, leaves=leaves)
    return len(resp.model_dump_json().encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-url", default="http://127.0.0.1:8471")
    parser.add_argument("--node-a-db", default="postgresql://gin:gin@localhost:5432/gin_node_a")
    parser.add_argument("--node-b-db", default="postgresql://gin:gin@localhost:5432/gin_node_b")
    parser.add_argument("--secret", default="dev-federation-secret")
    parser.add_argument("--converge-timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("[1/4] waiting for initial convergence (node A's cache of B vs. B's ground truth)")
    initial_diff = _wait_for_convergence(args.node_a_db, args.node_b_db, args.converge_timeout)
    correctness_pass = _diff_is_empty(initial_diff)
    print(f"    diff: {initial_diff}")

    print("[2/4] recording a no-op cycle's byte count")
    no_op_bytes = None
    deadline = time.monotonic() + args.converge_timeout
    while time.monotonic() < deadline:
        stats = _sync_stats(args.node_a_url, args.secret)
        if stats.last_root_matched:
            no_op_bytes = stats.last_cycle_bytes
            break
        time.sleep(1.0)
    print(f"    no-op cycle bytes: {no_op_bytes}")

    print("[3/4] mutating one chunk in node B directly, then waiting for the next sync")
    with psycopg.connect(args.node_b_db) as conn:
        mutated_chunk_id, original_hash = conn.execute(
            "SELECT chunk_id, content_hash FROM chunks LIMIT 1"
        ).fetchone()
        conn.execute(
            "UPDATE chunks SET content_hash = %s WHERE chunk_id = %s",
            (original_hash + "-eval-mutated", mutated_chunk_id),
        )
        conn.commit()

    try:
        mutation_bytes = None
        cycles_before = _sync_stats(args.node_a_url, args.secret).cycles_run
        deadline = time.monotonic() + args.converge_timeout
        while time.monotonic() < deadline:
            stats = _sync_stats(args.node_a_url, args.secret)
            if stats.cycles_run > cycles_before and not stats.last_root_matched:
                mutation_bytes = stats.last_cycle_bytes
                break
            time.sleep(1.0)
        print(f"    mutation cycle bytes: {mutation_bytes}")

        post_mutation_diff = _wait_for_convergence(
            args.node_a_db, args.node_b_db, args.converge_timeout
        )
        correctness_pass_after_mutation = _diff_is_empty(post_mutation_diff)
        full_corpus_bytes = _full_corpus_bytes(_ground_truth(args.node_b_db))
    finally:
        print("[4/4] reverting the mutation")
        with psycopg.connect(args.node_b_db) as conn:
            conn.execute(
                "UPDATE chunks SET content_hash = %s WHERE chunk_id = %s",
                (original_hash, mutated_chunk_id),
            )
            conn.commit()

    bandwidth_pass = (
        no_op_bytes is not None and no_op_bytes < 1000
        and mutation_bytes is not None and mutation_bytes < full_corpus_bytes / 4
    )

    metrics = {
        "initial_convergence_diff": initial_diff,
        "correctness_pass": correctness_pass,
        "no_op_cycle_bytes": no_op_bytes,
        "mutation_cycle_bytes": mutation_bytes,
        "full_corpus_bytes_reference": full_corpus_bytes,
        "post_mutation_diff": post_mutation_diff,
        "correctness_pass_after_mutation": correctness_pass_after_mutation,
        "bandwidth_pass": bandwidth_pass,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "anchor_sync_metrics.json"
    artifact.write_text(
        json.dumps(
            {"run_id": ts, "generated_at": datetime.now(timezone.utc).isoformat(), **metrics},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the live eval**

Prereqs: `docker compose up -d` in `docker/`; both node servers running per
the docstring above (Task 9 must be complete, including the schema
re-apply step).

Run: `./venv/Scripts/python.exe scripts/eval_anchor_sync.py`
Expected: `correctness_pass: true`, `correctness_pass_after_mutation: true`,
`bandwidth_pass: true`, `no_op_cycle_bytes` a few hundred bytes,
`mutation_cycle_bytes` well under a quarter of `full_corpus_bytes_reference`.
Note the run timestamp for the doc updates below.

- [ ] **Step 3: Update `architecture.md` Phase 3 checklist**

In the `### Phase 3 — Federation` section, replace:

```markdown
- 🔲 Merkle diff sync of anchor metadata (spec #2 — load-bearing at N>2)
```

with:

```markdown
- ✅ Merkle diff sync of anchor metadata (spec #2) — 16-bucket Merkle tree
  over (chunk_id, content_hash, outlet, title); background asyncio loop per
  node pulls its peer's root, drills into mismatched buckets only. Measured:
  `data/eval_runs/<ts>/anchor_sync_metrics.json` (0 diff vs. peer ground
  truth; no-op cycle O(1) bytes; single-chunk-change cycle « full corpus).
  Not load-bearing at N=2 (built as the primitive for N>2 peer selection).
  Spec: docs/superpowers/specs/2026-07-14-merkle-anchor-sync-design.md
```

(Substitute the real run timestamp for `<ts>`.)

- [ ] **Step 4: Add README quick-start subsection + update status row**

Insert after the existing `### Federation v1 — sovereign delegation` section
(after its closing `scope: docs/...` line, before the `---`):

```markdown
### Merkle anchor sync (background, both nodes)

Once both node servers are running (see above), each polls its peer's
anchor-metadata root on `anchor_sync_interval_s` (config, default 10s in
`config/node_*.yaml`) and drills into mismatched buckets only — never chunk
text. Inspect live convergence:

```bash
curl -H "Authorization: Bearer dev-federation-secret" \
  http://127.0.0.1:8471/v1/federated/anchors/sync_stats

# measure the bar (correctness + bandwidth)
python scripts/eval_anchor_sync.py
```

Scope and bar: docs/superpowers/specs/2026-07-14-merkle-anchor-sync-design.md.
```

Update the federation status table row (currently ending in `deferred to
spec #2+`) to:

```markdown
| Federation routing with sync metadata (Phase 3) | ✅ v1 sovereign delegation loop measured (run `20260714T175645Z`: routing FP 0, recall 1.0, routed fabrication 0.0, honest refusal 1.0); ✅ Merkle anchor sync measured (run `<ts>`: 0 diff vs. ground truth, no-op O(1) bytes, single-change cycle « full corpus); trust weights + peer selection deferred to N>2 |
```

(Substitute the real run timestamp for `<ts>`.)

- [ ] **Step 5: Add implementation note to `docs/GIN_Node_Architecture_v1.md`**

Immediately after the `2. **Corpus-Diff Sync Endpoint**...` bullet, add:

```markdown
> **v1 implementation note (2026-07):** the shipped sync endpoint is a
> 2-level, 16-bucket prefix tree (`gin/federation/anchor_tree.py`), not a
> full Merkle trie — sufficient at corpus sizes in the tens to low hundreds
> of chunks per node. A deeper tree is a later revision if bucket sizes grow
> enough that a single changed bucket still means transferring hundreds of
> leaves.
```

- [ ] **Step 6: Full suite + final validation**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all green (Postgres-backed `integration`-marked tests pass if
`docker compose up -d` is running, else skip).

Run: `git status --short`
Expected: only the intended doc modifications; nothing unexpected.

- [ ] **Step 7: Commit and push**

```bash
git add scripts/eval_anchor_sync.py README.md architecture.md docs/GIN_Node_Architecture_v1.md
git commit -m "Merkle anchor sync measured: eval driver + docs. Correctness and bandwidth bar both green.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
git push origin main
```

---

## Self-review notes (already applied)

- **Spec coverage:** falsifiable claim → Task 10 eval; 16-bucket prefix
  tree rationale → Task 2; metadata fields (chunk_id/content_hash/outlet/
  title, never text) → Tasks 1–3; background asyncio loop → Task 7;
  bidirectional-capable design (each node runs its own loop against its one
  configured peer) → Task 9 wires both configs identically; storage → Task
  3; three-tier testing (unit/integration/live) → Tasks 2&5 (unit),
  Task 8 (integration), Task 10 (live); docs updates → Task 10.
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `AnchorLeaf`, `PeerAnchorStore`, `SyncStats`,
  `AnchorSyncStats` field names verified identical across every task that
  references them (schema Task 1 → tree Task 2 → store Task 3 → sync
  Task 5 → server Tasks 6–7 → eval Task 10).
- **Non-breaking check:** `create_app`'s three new parameters
  (`local_anchor_rows`, `peer_anchor_store`) are keyword-only with defaults
  that reproduce prior behavior exactly (empty anchor rows, no background
  task); `PeerClient`'s three new Protocol methods are additive, so existing
  fakes implementing only `.query()` remain valid wherever they're never
  asked for anchor data.
