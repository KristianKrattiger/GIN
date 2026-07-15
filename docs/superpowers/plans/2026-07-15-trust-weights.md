# Trust Weights: Per-Domain Peer Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A node gates (excludes) a peer from its ranked delegation candidates when a configured trust weight for that peer, in a domain the peer serves, falls below a threshold — layered after sub-project 3's existing dense+sparse RRF ranking, not replacing it.

**Architecture:** Extends `gin/federation/` (built across three prior sub-projects) and the corpus-ingest pipeline. A new `documents.domain` column (currently dropped at ingest) is populated from `metadata.domain`; a node's own domain coverage is computed in `build_local_summary` and synced to peers as an additive field on the existing `PeerSummaryResponse`/`peer_summaries` machinery. A new pure-logic `trust_gate.py` filters a node's already-ranked peer list against per-`(peer, domain)` trust weights read from that node's own YAML config, before the filtered list reaches the router's existing sequential-fallback loop.

**Tech Stack:** Python 3.12, FastAPI/Starlette, Pydantic, psycopg3/Postgres, pytest.

## Global Constraints

- A trust weight below threshold **gates a peer out entirely** for that
  query — it is never contacted. No blending into the RRF score in this
  sub-project.
- **No query-time domain classification.** Gating keys off which domain(s) a
  peer's own corpus covers (synced), never what the live query is about.
- **Conservative gating policy:** a peer is eligible only if *every* one of
  its known domains clears the configured weight (default `1.0` when a
  `(peer, domain)` pair isn't explicitly configured). A peer with **no**
  known domains (no synced summary, or a summary whose corpus has no tagged
  documents) is **never** gated — absence of information never gates,
  matching sub-project 3's existing "no-summary peers ranked last, never
  dropped" invariant.
- `trust_gate_threshold` defaults to `0.5` per node. Since the implicit
  per-domain weight is `1.0` when unconfigured, `1.0 >= 0.5` always holds —
  an empty/absent `trust_weights` config reproduces sub-project 3's ungated
  behavior exactly, regardless of the configured threshold.
- All new fields/parameters are **additive with defaults reproducing prior
  behavior**: `documents.domain` defaults to `''` (never matches a real
  configured domain, so untagged corpora are never gated); `PeerSummaryResponse.domains`
  defaults to `[]`; `NodeConfig.trust_weights` defaults to `{}`. Existing
  federation, anchor-sync, peer-selection, and corpus-ingest tests must keep
  passing unmodified.
- Gating touches nothing downstream of ranking: `answer_or_delegate`, the
  router's `hop_count=1` sequential-fallback loop, and `peer_selection.py`'s
  RRF logic are untouched. A gated peer is simply absent from the list the
  router iterates — the same code path as "this peer doesn't exist."
- DB-touching tests use the existing `isolated_db` / `tmp_cold_root` fixtures
  (`tests/conftest.py`), marked `@pytest.mark.integration`.
- Follow the `gin/corpus/db.py` convention: fresh Postgres connection per
  call (`connect()` / `transaction()`), never held across a background task.
- Schema changes to already-running databases use `ALTER TABLE ... ADD
  COLUMN IF NOT EXISTS` alongside the `CREATE TABLE IF NOT EXISTS` for fresh
  installs, so re-applying `docker/init-db.sql` against the live
  `gin_node_a/b/c` databases (already provisioned in sub-project 3) picks up
  the new columns without a full re-ingest.

---

### Task 1: Wire schema — PeerSummaryResponse.domains

**Files:**
- Modify: `gin/federation/schema.py`
- Test: `tests/test_federation_schema.py`

**Interfaces:**
- Consumes: `PeerSummaryResponse` (existing, `gin/federation/schema.py`).
- Produces: `PeerSummaryResponse.domains: list[str]` (default empty).
  Consumed by Tasks 5, 6.

- [ ] **Step 1: Write the failing tests**

Modify the existing `test_peer_summary_response_round_trip` and
`test_peer_summary_defaults_empty_collections` in
`tests/test_federation_schema.py`, and add one new test:

```python
def test_peer_summary_response_round_trip():
    resp = PeerSummaryResponse(
        node_id="node_c",
        embedding_centroid=[0.1, 0.2, 0.3],
        distinctive_terms={"inflation": 2.1, "reserve": 1.8},
        domains=["monetary_policy"],
    )
    again = PeerSummaryResponse.model_validate(resp.model_dump())
    assert again == resp
    assert again.protocol_version == PROTOCOL_VERSION


def test_peer_summary_defaults_empty_collections():
    resp = PeerSummaryResponse(node_id="node_c")
    assert resp.embedding_centroid == []
    assert resp.distinctive_terms == {}
    assert resp.domains == []


def test_peer_summary_domains_round_trips_multiple():
    resp = PeerSummaryResponse(
        node_id="node_a", domains=["environmental_measurement", "monetary_policy"],
    )
    again = PeerSummaryResponse.model_validate(resp.model_dump())
    assert again.domains == ["environmental_measurement", "monetary_policy"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: FAIL — `test_peer_summary_response_round_trip` and the new test
fail because `PeerSummaryResponse` rejects/ignores the `domains` kwarg
inconsistently with the assertions (Pydantic silently drops unknown kwargs
by default only if configured to; here it will raise `ValidationError:
Extra inputs are not permitted` since the model has no `domains` field —
confirm this is the actual error you see).

- [ ] **Step 3: Add the field**

In `gin/federation/schema.py`, add `domains` to `PeerSummaryResponse` (after
`distinctive_terms`):

```python
class PeerSummaryResponse(BaseModel):
    """A node's routing signal: an embedding centroid + distinctive IDF terms
    + the distinct domains this node's corpus covers. Chunk text never
    appears here — only these aggregate statistics."""

    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    embedding_centroid: list[float] = Field(default_factory=list)
    distinctive_terms: dict[str, float] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: PASS, all tests including pre-existing.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/schema.py tests/test_federation_schema.py
git commit -m "PeerSummaryResponse gains domains: list[str] (additive, still never chunk text)."
```

---

### Task 2: Pure trust-gate logic

**Files:**
- Create: `gin/federation/trust_gate.py`
- Test: `tests/test_trust_gate.py`

**Interfaces:**
- Consumes: nothing beyond stdlib types — deliberately decoupled from
  `PeerSummaryResponse`/`NodeConfig` so it's testable in total isolation.
- Produces: `is_trusted(peer_domains: list[str], peer_weights: dict[str, float], threshold: float) -> bool`;
  `filter_trusted(ranked_peer_ids: list[str], domains_by_peer: dict[str, list[str]], trust_weights: dict[str, dict[str, float]], threshold: float) -> list[str]`.
  Consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trust_gate.py`:

```python
"""Trust gating: a peer is excluded only if every domain it's known to
serve falls below the configured weight; absence of domain information
never gates; order is preserved for peers that pass."""
from gin.federation.trust_gate import filter_trusted, is_trusted


def test_is_trusted_true_when_all_domains_clear_threshold():
    assert is_trusted(["monetary_policy"], {"monetary_policy": 0.9}, 0.5) is True


def test_is_trusted_false_when_any_domain_below_threshold():
    assert is_trusted(
        ["monetary_policy", "inflation"],
        {"monetary_policy": 0.9, "inflation": 0.1},
        0.5,
    ) is False


def test_is_trusted_true_for_unconfigured_domain_default_full_trust():
    # No entry for this domain -> implicit weight 1.0, clears any threshold <= 1.0.
    assert is_trusted(["monetary_policy"], {}, 0.5) is True


def test_is_trusted_true_for_no_known_domains():
    # Absence of domain information never gates.
    assert is_trusted([], {"monetary_policy": 0.0}, 0.5) is True


def test_filter_trusted_excludes_gated_peer_preserves_order():
    order = filter_trusted(
        ["node_c", "node_b"],
        {"node_c": ["monetary_policy"], "node_b": ["environmental_impact"]},
        {"node_c": {"monetary_policy": 0.1}},
        0.5,
    )
    assert order == ["node_b"]


def test_filter_trusted_keeps_peer_with_no_domain_entry():
    # node_b has no entry in domains_by_peer at all (e.g. no synced summary).
    order = filter_trusted(
        ["node_c", "node_b"],
        {"node_c": ["monetary_policy"]},
        {"node_c": {"monetary_policy": 0.1}},
        0.5,
    )
    assert order == ["node_b"]


def test_filter_trusted_empty_trust_weights_keeps_everyone():
    order = filter_trusted(
        ["node_c", "node_b"],
        {"node_c": ["monetary_policy"], "node_b": ["environmental_impact"]},
        {},
        0.5,
    )
    assert order == ["node_c", "node_b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_trust_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.trust_gate'`.

- [ ] **Step 3: Write `gin/federation/trust_gate.py`**

```python
"""Pure trust-gate logic — no I/O, no network, no DB.

A peer is gated out of consideration only if a domain it's known to serve
falls below the configured trust weight; absence of domain information
(no synced summary, or a summary with no tagged domains) never gates a
peer, matching peer_selection.py's "no-summary peers never dropped"
invariant. This is a filter applied AFTER ranking, not a re-ranking: order
is preserved for every peer that passes.
"""
from __future__ import annotations


def is_trusted(
    peer_domains: list[str], peer_weights: dict[str, float], threshold: float
) -> bool:
    """True unless some known domain of this peer falls below threshold.
    An unconfigured domain defaults to full trust (1.0); a peer with no
    known domains passes vacuously."""
    return all(peer_weights.get(d, 1.0) >= threshold for d in peer_domains)


def filter_trusted(
    ranked_peer_ids: list[str],
    domains_by_peer: dict[str, list[str]],
    trust_weights: dict[str, dict[str, float]],
    threshold: float,
) -> list[str]:
    """Ranked peer ids with any gated peer removed; relative order preserved."""
    return [
        nid
        for nid in ranked_peer_ids
        if is_trusted(domains_by_peer.get(nid, []), trust_weights.get(nid, {}), threshold)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_trust_gate.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/trust_gate.py tests/test_trust_gate.py
git commit -m "Trust gate: pure filter on a ranked peer list, absence of domain info never gates."
```

---

### Task 3: NodeConfig trust_weights + trust_gate_threshold

**Files:**
- Modify: `gin/federation/config.py`
- Test: `tests/test_federation_config.py`

**Interfaces:**
- Consumes: existing `NodeConfig`, `load_node_config` (`gin/federation/config.py`).
- Produces: `NodeConfig.trust_weights: dict[str, dict[str, float]]` (default
  `{}`); `NodeConfig.trust_gate_threshold: float` (default `0.5`); both
  parsed by `load_node_config`. Consumed by Task 6, Task 8's eval config.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_federation_config.py`:

```python
def test_trust_weights_default_empty(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.trust_weights == {}
    assert cfg.trust_gate_threshold == 0.5


def test_trust_weights_parsed_from_yaml(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(
        _YAML + "trust_weights:\n  node_c:\n    monetary_policy: 0.1\n"
        "trust_gate_threshold: 0.6\n",
        encoding="utf-8",
    )
    cfg = load_node_config(p)
    assert cfg.trust_weights == {"node_c": {"monetary_policy": 0.1}}
    assert cfg.trust_gate_threshold == 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: FAIL — `AttributeError: 'NodeConfig' object has no attribute 'trust_weights'`.

- [ ] **Step 3: Extend `gin/federation/config.py`**

Add `field` to the dataclasses import:

```python
from dataclasses import dataclass, field
```

Add the two new fields to `NodeConfig` (after `anchor_sync_interval_s`):

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
    trust_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    trust_gate_threshold: float = 0.5
```

Add parsing to `load_node_config` (inside the returned `NodeConfig(...)`, after
`anchor_sync_interval_s=...`):

```python
        anchor_sync_interval_s=float(raw.get("anchor_sync_interval_s", 30.0)),
        trust_weights=raw.get("trust_weights", {}),
        trust_gate_threshold=float(raw.get("trust_gate_threshold", 0.5)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: PASS, all tests including pre-existing.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/config.py tests/test_federation_config.py
git commit -m "NodeConfig: trust_weights + trust_gate_threshold (default 0.5, additive)."
```

---

### Task 4: documents.domain column + ingest plumbing

**Files:**
- Modify: `docker/init-db.sql`
- Modify: `gin/corpus/models.py`
- Modify: `gin/corpus/warm.py`
- Modify: `gin/corpus/ingest.py`
- Test: `tests/test_ingest.py`, `tests/test_bookkeeper_persist.py` (regression only, unmodified)

**Interfaces:**
- Consumes: existing `DocumentDraft`, `warm.upsert_document`,
  `ingest.load_json`/`ingest_documents` (`gin/corpus/`).
- Produces: `DocumentDraft.domain: str` (default `""`);
  `warm.upsert_document(..., domain: str = "")`; `load_json` maps
  `metadata.domain` → `DocumentDraft.domain`; `ingest_documents` passes
  `doc.domain` through to `upsert_document`. Consumed by Task 5, Task 7.

- [ ] **Step 1: Add the `domain` column to `docker/init-db.sql`**

In `docker/init-db.sql`, add `domain` to the `documents` table definition:

```sql
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    source_uri TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'synthetic',
    outlet TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Immediately after that `CREATE TABLE` statement, add a standalone
idempotent migration for already-provisioned databases (the live
`gin_node_a/b/c` databases from sub-project 3 already have this table
without the column — `CREATE TABLE IF NOT EXISTS` alone would skip them):

```sql
ALTER TABLE documents ADD COLUMN IF NOT EXISTS domain TEXT NOT NULL DEFAULT '';
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
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
```

`ROOT`, `Path`, and `pytest` are already imported/defined at the top of
`tests/test_ingest.py` (`ROOT = Path(__file__).resolve().parents[1]`) —
reuse that existing constant, don't redefine it.

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_ingest.py -v -m "not integration"`
Expected: FAIL — `AttributeError: 'DocumentDraft' object has no attribute 'domain'`
(the plain unit test fails first; the integration test would fail the same
way once run).

- [ ] **Step 4: Add `domain` to `DocumentDraft` (`gin/corpus/models.py`)**

Add the field after `eval_tag`:

```python
@dataclass
class DocumentDraft:
    doc_id: str
    outlet: str
    title: str
    eval_layer: EvalLayer
    source_uri: str = ""
    source_type: str = "synthetic"
    chunks: list[str] = field(default_factory=list)
    eval_tag: Optional[str] = None
    domain: str = ""
```

- [ ] **Step 5: Map `metadata.domain` in `load_json` (`gin/corpus/ingest.py`)**

Update `load_json`'s docstring and the `DocumentDraft(...)` construction:

```python
def load_json(path: Path) -> tuple[list[DocumentDraft], list[EdgeDraft]]:
    """Load a node corpus manifest (corpus_node*.json) into DocumentDrafts.

    Maps the fetched-corpus schema to the ingest model:
      source   -> title           node              -> outlet (federation node)
      url      -> source_uri       metadata.type     -> source_type
      metadata.category -> eval_tag   metadata.domain -> domain
    ``outlet`` is the document's node id (node_1_institutional / node_2_grassroots)
    so the eval's chunk->outlet map is the federation boundary; the per-source
    name is preserved in ``title``.
    Chunk objects are ordered by their ``position`` field; chunk ``text`` becomes
    the ingest chunk body (warm-tier chunk ids remain ``<doc_id>:<index>``).
    JSON manifests carry no edges, so an empty edge list is returned.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    documents: list[DocumentDraft] = []
    for item in data.get("documents", []):
        raw_chunks = item.get("chunks")
        if not raw_chunks:
            raise ValueError(f"document {item.get('doc_id')} has no chunks")
        ordered = sorted(raw_chunks, key=lambda c: c.get("position", 0))
        texts = [c["text"].strip() for c in ordered]
        meta = item.get("metadata", {})
        documents.append(
            DocumentDraft(
                doc_id=item["doc_id"],
                outlet=item.get("node") or meta.get("author", ""),
                title=item.get("source", item["doc_id"]),
                eval_layer=_parse_eval_layer(item.get("eval_layer", "realism")),
                source_uri=item.get("url", str(path)),
                source_type=meta.get("type", "curated"),
                chunks=texts,
                eval_tag=meta.get("category"),
                domain=meta.get("domain", ""),
            )
        )
    return documents, []
```

- [ ] **Step 6: Thread `domain` through `warm.upsert_document` (`gin/corpus/warm.py`)**

```python
def upsert_document(
    conn: psycopg.Connection,
    *,
    doc_id: str,
    content_hash: str,
    outlet: str,
    title: str,
    source_uri: str = "",
    source_type: str = "synthetic",
    domain: str = "",
) -> UUID:
    uid = _doc_uuid(doc_id)
    conn.execute(
        """
        INSERT INTO documents (doc_id, content_hash, source_uri, source_type, outlet, title, domain)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_hash) DO UPDATE SET
            outlet = EXCLUDED.outlet,
            title = EXCLUDED.title,
            source_uri = EXCLUDED.source_uri,
            source_type = EXCLUDED.source_type,
            domain = EXCLUDED.domain
        RETURNING doc_id
        """,
        (uid, content_hash, source_uri, source_type, outlet, title, domain),
    )
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE content_hash = %s",
        (content_hash,),
    ).fetchone()
    return row[0]
```

- [ ] **Step 7: Pass `doc.domain` through in `ingest_documents` (`gin/corpus/ingest.py`)**

Update the `warm.upsert_document(...)` call inside `ingest_documents`:

```python
                doc_uuid = warm.upsert_document(
                    conn,
                    doc_id=doc.doc_id,
                    content_hash=doc_hash,
                    outlet=doc.outlet,
                    title=doc.title,
                    source_uri=doc.source_uri,
                    source_type=doc.source_type,
                    domain=doc.domain,
                )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_ingest.py -v`
Expected: PASS (integration test runs since Postgres is up; otherwise skips).

Run the regression set to confirm `domain` being keyword-only-with-default
doesn't break existing callers of `upsert_document`:
Run: `./venv/Scripts/python.exe -m pytest tests/test_bookkeeper_persist.py tests/test_cartographer_scan_e2e.py -v`
Expected: PASS, unchanged.

- [ ] **Step 9: Commit**

```bash
git add docker/init-db.sql gin/corpus/models.py gin/corpus/warm.py gin/corpus/ingest.py tests/test_ingest.py
git commit -m "documents.domain column + ingest plumbing: metadata.domain now persisted, not dropped."
```

---

### Task 5: peer_summaries.domains column + store + build_local_summary

**Files:**
- Modify: `docker/init-db.sql`
- Modify: `gin/federation/peer_summary_store.py`
- Test: `tests/test_peer_summary_store.py`

**Interfaces:**
- Consumes: `PeerSummaryResponse.domains` (Task 1); `documents.domain`
  (Task 4); existing `PostgresPeerSummaryStore`, `build_local_summary`.
- Produces: `PostgresPeerSummaryStore.get`/`.set` round-trip `domains`;
  `build_local_summary` populates `domains` from the local `documents` table.
  Consumed by Task 6, Task 8's live eval.

- [ ] **Step 1: Add the `domains` column to `docker/init-db.sql`**

Add `domains` to the `peer_summaries` table definition, plus the same
idempotent-migration pattern as Task 4:

```sql
CREATE TABLE IF NOT EXISTS peer_summaries (
    peer_node_id       TEXT PRIMARY KEY,
    embedding_centroid REAL[] NOT NULL,
    distinctive_terms  JSONB NOT NULL,
    domains            JSONB NOT NULL DEFAULT '[]'::jsonb,
    synced_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE peer_summaries ADD COLUMN IF NOT EXISTS domains JSONB NOT NULL DEFAULT '[]'::jsonb;
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_peer_summary_store.py`:

```python
@pytest.mark.integration
def test_postgres_summary_store_round_trips_domains(isolated_db):
    store = PostgresPeerSummaryStore()
    store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[0.5], distinctive_terms={},
        domains=["monetary_policy"],
    ))
    got = store.get("node_c")
    assert got.domains == ["monetary_policy"]


@pytest.mark.integration
def test_build_local_summary_includes_domains(isolated_db, tmp_cold_root):
    from gin.corpus.ingest import ingest_path

    ingest_path(ROOT / "corpus_node3.json", embed=True, ingest_edges=False)
    summary = build_local_summary("node_c")
    assert summary.domains == ["monetary_policy"]


@pytest.mark.integration
def test_build_local_summary_domains_empty_when_untagged(isolated_db, tmp_cold_root):
    ingest_path(NEWS, embed=True)
    summary = build_local_summary("node_local")
    assert summary.domains == []
```

`ROOT` and `NEWS` are already defined at the top of
`tests/test_peer_summary_store.py` (`ROOT = Path(__file__).resolve().parents[1]`,
`NEWS = ROOT / "data" / "synthetic" / "news_corpus.yaml"`) — reuse them,
don't redefine.

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_summary_store.py -v`
Expected: FAIL — `test_postgres_summary_store_round_trips_domains` fails with
a `TypeError`/`ValidationError` about the unexpected `domains` kwarg on
`PeerSummaryResponse` being silently dropped by the store's `get` (it won't
round-trip since `set`/`get` don't touch the column yet); the two
`build_local_summary` tests fail because `summary.domains` doesn't exist yet
(`AttributeError`, since Task 1 already added the field with a default `[]`
— confirm the actual failure is the assertion `== ["monetary_policy"]`
comparing against the default `[]`, not a missing attribute).

- [ ] **Step 4: Update `PostgresPeerSummaryStore` and `build_local_summary`**

In `gin/federation/peer_summary_store.py`, update `get`:

```python
    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]:
        with connect() as conn:
            row = conn.execute(
                "SELECT embedding_centroid, distinctive_terms, domains FROM peer_summaries "
                "WHERE peer_node_id = %s",
                (peer_node_id,),
            ).fetchone()
        if row is None:
            return None
        terms = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        domains = row[2] if isinstance(row[2], list) else json.loads(row[2])
        return PeerSummaryResponse(
            node_id=peer_node_id,
            embedding_centroid=[float(x) for x in row[0]],
            distinctive_terms=terms,
            domains=domains,
        )
```

Update `set`:

```python
    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO peer_summaries "
                "(peer_node_id, embedding_centroid, distinctive_terms, domains) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (peer_node_id) DO UPDATE SET "
                "embedding_centroid = EXCLUDED.embedding_centroid, "
                "distinctive_terms = EXCLUDED.distinctive_terms, "
                "domains = EXCLUDED.domains, "
                "synced_at = NOW()",
                (
                    peer_node_id,
                    list(summary.embedding_centroid),
                    json.dumps(summary.distinctive_terms),
                    json.dumps(summary.domains),
                ),
            )
```

Update `build_local_summary` to also query domains:

```python
def build_local_summary(
    node_id: str, top_n: int = 40, conn: Optional[psycopg.Connection] = None
) -> PeerSummaryResponse:
    """This node's routing summary: unit-mean chunk embedding + top-N IDF
    terms + the distinct non-empty domains this node's corpus covers."""
    if conn is None:
        with connect() as conn:
            return build_local_summary(node_id, top_n, conn)
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks").fetchall()]
    centroid = _unit_mean(embed_texts(texts)) if texts else [0.0] * EMBEDDING_DIM
    idf = corpus_idf(texts)
    top = dict(sorted(idf.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n])
    domains = sorted({
        r[0] for r in conn.execute(
            "SELECT DISTINCT domain FROM documents WHERE domain != ''"
        ).fetchall()
    })
    return PeerSummaryResponse(
        node_id=node_id, embedding_centroid=centroid, distinctive_terms=top,
        domains=domains,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_summary_store.py -v`
Expected: PASS (all tests; integration tests run since Postgres is up).

- [ ] **Step 6: Commit**

```bash
git add docker/init-db.sql gin/federation/peer_summary_store.py tests/test_peer_summary_store.py
git commit -m "peer_summaries.domains column + store round-trip + build_local_summary domain query."
```

---

### Task 6: Wire the trust gate into server.py's peer ranker

**Files:**
- Modify: `gin/federation/server.py`
- Test: `tests/test_trust_gate_wiring.py`

**Interfaces:**
- Consumes: `filter_trusted` (Task 2); `PeerSummaryResponse.domains` (Task
  1); `NodeConfig.trust_weights`/`trust_gate_threshold` (Task 3); existing
  `create_app`, `_rank_peers_for_query`.
- Produces: no new `create_app` parameter — the existing
  `_rank_peers_for_query` closure additionally filters its output through
  the trust gate before returning. Consumed by the live eval (Task 9).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trust_gate_wiring.py`:

```python
"""Trust gate wired into create_app: a peer below the configured trust
threshold for a domain it serves is never contacted, even when it ranks
first on similarity. An unconfigured (empty) trust_weights config must
reproduce sub-project 3's ungated behavior exactly."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import (
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
    WireClaim,
)
from gin.federation.server import create_app

SECRET = "trust-secret"
AUTH = {"Authorization": f"Bearer {SECRET}"}


def _cfg(trust_weights=None, trust_gate_threshold=0.5):
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/a", cold_path="data/cold_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=5.0,
        peers=(PeerConfig("node_b", "http://b"), PeerConfig("node_c", "http://c")),
        trust_weights=trust_weights or {},
        trust_gate_threshold=trust_gate_threshold,
    )


def _refuse(q):
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
        return NodeRefusal(request_id=fq.request_id, node_id=peer.node_id, reason="zero_cursors")


def _summaries():
    store = InMemoryPeerSummaryStore()
    # node_c ranks first on similarity (matches the fake embedder below),
    # but its only domain, monetary_policy, will be untrusted in the gated test.
    store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0],
        distinctive_terms={"inflation": 3.0}, domains=["monetary_policy"],
    ))
    store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0],
        distinctive_terms={"justice": 3.0}, domains=["environmental_impact"],
    ))
    return store


def _embed(q):
    return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]


def test_gated_peer_never_contacted_falls_back_to_refusal():
    peer_client = ScriptedPeer(answerer="node_c")  # only node_c could ground it
    app = create_app(
        _cfg(trust_weights={"node_c": {"monetary_policy": 0.1}}),
        answer_fn=_refuse, peer_client=peer_client,
        peer_summary_store=_summaries(), embed_query_fn=_embed,
    )
    client = TestClient(app)
    fq = FederatedQuery(query="what drives inflation", origin_node="d", hop_count=0)
    r = client.post("/v1/federated/query", headers=AUTH, json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal is not None
    assert peer_client.calls == ["node_b"]  # node_c never contacted
    assert "node_c" not in (resp.refusal.peer_reasons or {})


def test_ungated_query_still_reaches_correct_peer():
    peer_client = ScriptedPeer(answerer="node_c")
    app = create_app(
        _cfg(),  # no trust_weights configured -> fully trusted, sub-project 3 baseline
        answer_fn=_refuse, peer_client=peer_client,
        peer_summary_store=_summaries(), embed_query_fn=_embed,
    )
    client = TestClient(app)
    fq = FederatedQuery(query="what drives inflation", origin_node="d", hop_count=0)
    r = client.post("/v1/federated/query", headers=AUTH, json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_c"
    assert peer_client.calls == ["node_c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_trust_gate_wiring.py -v`
Expected: FAIL — `test_gated_peer_never_contacted_falls_back_to_refusal` fails
because node_c is still contacted (`peer_client.calls == ["node_c"]`, not
`["node_b"]`) since the gate isn't wired yet.
`test_ungated_query_still_reaches_correct_peer` passes already (this is the
regression case) — confirm it passes even before your change.

- [ ] **Step 3: Wire the gate into `gin/federation/server.py`**

Add the import (with the other local imports):

```python
from .trust_gate import filter_trusted
```

Modify `_rank_peers_for_query` — after computing `order` via `rank_peers`,
filter it before mapping back to `PeerConfig`:

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
        domains_by_peer = {nid: s.domains for nid, s in summaries.items()}
        order = filter_trusted(
            order, domains_by_peer, config.trust_weights, config.trust_gate_threshold
        )
        by_id = {p.node_id: p for p in config.peers}
        return [by_id[nid] for nid in order]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_trust_gate_wiring.py -v`
Expected: PASS, both tests.

Run the regression set to confirm sub-project 3's behavior is unchanged for
configs without `trust_weights`:
Run: `./venv/Scripts/python.exe -m pytest tests/test_router_selection.py tests/test_peer_selection_loop.py tests/test_federation_server.py tests/test_federation_loop.py -v`
Expected: PASS, unchanged.

- [ ] **Step 5: Commit**

```bash
git add gin/federation/server.py tests/test_trust_gate_wiring.py
git commit -m "Wire trust gate into the peer ranker: gated peers are never contacted."
```

---

### Task 7: Migrate + backfill the live 3-node deployment

**Files:** none (operational step against the already-provisioned
`gin_node_a`/`gin_node_b`/`gin_node_c` Postgres databases from sub-project 3)

**Interfaces:**
- Consumes: `docker/init-db.sql` (Tasks 4, 5); `scripts/corpus_ingest.py`
  (existing CLI, unmodified).
- Produces: `documents.domain` populated for all three live databases,
  without a full re-embed. Consumed by Task 9's live eval.

- [ ] **Step 1: Confirm Postgres is up**

Run: `docker ps --format "{{.Names}}\t{{.Status}}"` — confirm `gin-postgres`
shows `Up ... (healthy)`. If not running, start it (`docker compose up -d`
in `docker/`) before continuing.

- [ ] **Step 2: Re-apply the schema to all three live databases**

The `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements from Tasks 4 and 5
make this idempotent — existing tables gain the new columns, nothing else
changes:

```bash
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec -i gin-postgres psql -U gin -d "$db" < docker/init-db.sql
done
```

Expected: no errors; `ALTER TABLE` lines report success (or are silently
skipped if already applied).

- [ ] **Step 3: Backfill `documents.domain` for already-ingested corpora**

Re-run ingestion with `--no-embed` (chunks/embeddings are unchanged and
already ingested; this only needs to re-walk `metadata.domain` through the
`ON CONFLICT (content_hash) DO UPDATE` path added in Task 4):

```bash
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_a" \
GIN_COLD_PATH="data/cold_node_a" ./venv/Scripts/python.exe scripts/corpus_ingest.py \
  --source corpus_node1.json --no-edges --no-embed
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_b" \
GIN_COLD_PATH="data/cold_node_b" ./venv/Scripts/python.exe scripts/corpus_ingest.py \
  --source corpus_node2.json --no-edges --no-embed
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_c" \
GIN_COLD_PATH="data/cold_node_c" ./venv/Scripts/python.exe scripts/corpus_ingest.py \
  --source corpus_node3.json --no-edges --no-embed
```

- [ ] **Step 4: Verify domain backfill**

```bash
for db in gin_node_a gin_node_b gin_node_c; do
  echo "=== $db ==="
  docker exec gin-postgres psql -U gin -d "$db" -tAc \
    "SELECT DISTINCT domain FROM documents WHERE domain != '';"
done
```

Expected: `gin_node_a` → `environmental_measurement`; `gin_node_b` →
`environmental_impact`; `gin_node_c` → `monetary_policy`. All three
non-empty.

- [ ] **Step 5: Clear stale cached peer summaries**

Existing `peer_summaries` rows in `gin_node_a` (from sub-project 3's live
eval) were cached before the `domains` column existed and will not be
refreshed automatically — the summary-sync loop only refetches on anchor
change or when nothing is cached (sub-project 3's live-eval robustness
fix), and neither condition is true for an already-cached, unchanged peer.
Clear the cache once so the next sync cycle repopulates it with domains:

```bash
for db in gin_node_a gin_node_b gin_node_c; do
  docker exec gin-postgres psql -U gin -d "$db" -c "DELETE FROM peer_summaries;"
done
```

No commit for this task — it's a live-environment operational step with no
code change. Record the outcome in the SDD progress ledger instead.

---

### Task 8: Gated eval config + selection_eval.py gated-peer metric

**Files:**
- Create: `config/node_a_trust_gated.yaml`
- Modify: `gin/federation/selection_eval.py`
- Modify: `scripts/eval_peer_selection.py`
- Test: `tests/test_selection_eval.py`

**Interfaces:**
- Consumes: existing `compute_selection_metrics`, `SelectionOutcome`
  (`gin/federation/selection_eval.py`); `config/node_a.yaml` (existing,
  left untouched as the ungated baseline).
- Produces: `compute_selection_metrics(outcomes, gated_peer=None)` — new
  optional parameter; when given, adds a `gated_peer_contacted` key to the
  returned metrics (count of outcomes whose `peers_attempted` includes
  `gated_peer`); `scripts/eval_peer_selection.py --gated-peer` CLI flag
  passing through to `compute_selection_metrics`. Consumed by Task 9's live
  eval.

- [ ] **Step 1: Create `config/node_a_trust_gated.yaml`**

A full copy of `config/node_a.yaml`'s current content (below), with the
`trust_weights`/`trust_gate_threshold` block appended — do not modify
`config/node_a.yaml` itself, so the default (ungated) baseline config used
by every prior sub-project's eval stays exactly as it is:

```yaml
# GIN federation node A (institutional corpus: corpus_node1.json).
# shared_secret here is a NON-secret dev default for localhost;
# set GIN_FED_SECRET in the environment to override off-localhost.
# This variant additionally trust-gates node_c out of consideration for its
# monetary_policy domain, for the trust-weights live eval's gated run.
node_id: node_a
host: 127.0.0.1
port: 8471
database_url: postgresql://gin:gin@localhost:5432/gin_node_a
cold_path: data/cold_node_a
model_path: data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
n_gpu_layers: -1
n_ctx: 4096
shared_secret: dev-federation-secret
peer_timeout_s: 300
peers:
  - node_id: node_b
    url: http://127.0.0.1:8472
  - node_id: node_c
    url: http://127.0.0.1:8473
anchor_sync_interval_s: 10
trust_weights:
  node_c:
    monetary_policy: 0.1
trust_gate_threshold: 0.5
```

If `config/node_a.yaml` has changed since this plan was written (check with
`git diff` against the content above before creating the new file), copy
its actual current content instead and append the same
`trust_weights`/`trust_gate_threshold` block — the point is an exact copy
plus that one addition, never a re-typed guess.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_selection_eval.py`:

```python
def test_gated_peer_contacted_counts_outcomes_that_reach_it():
    outcomes = [
        _routed("c1", "c_only", "", ["node_b", "node_c"], verified=None),
        _routed("c2", "c_only", "", ["node_b"], verified=None),
    ]
    m = compute_selection_metrics(outcomes, gated_peer="node_c")
    assert m["gated_peer_contacted"] == 1


def test_gated_peer_contacted_absent_when_not_requested():
    outcomes = [_routed("c1", "c_only", "node_c", ["node_c"])]
    m = compute_selection_metrics(outcomes)
    assert "gated_peer_contacted" not in m
```

(`_routed` is the existing helper already defined in this test file; the
first test's outcomes model a `neither`-style refusal shape by leaving
`source_node` blank — reuse `_routed`'s existing signature as-is, since it
already accepts `verified` as a keyword arg with a default.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_selection_eval.py -v`
Expected: FAIL — `TypeError: compute_selection_metrics() got an unexpected
keyword argument 'gated_peer'`.

- [ ] **Step 4: Extend `compute_selection_metrics` (`gin/federation/selection_eval.py`)**

Add the `gated_peer` parameter and the new metric — insert the parameter
into the signature and compute the new key inside the returned dict (after
`"honest_refusal_rate"`, before `"per_query"`):

```python
def compute_selection_metrics(
    outcomes: list[SelectionOutcome], gated_peer: Optional[str] = None
) -> dict:
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
    result = {
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
    }
    if gated_peer is not None:
        # Bar: 0 — a gated peer is never contacted at all.
        result["gated_peer_contacted"] = sum(
            1 for o in outcomes if gated_peer in o.peers_attempted
        )
    result["per_query"] = [o.__dict__ for o in outcomes]
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_selection_eval.py -v`
Expected: PASS, all tests including pre-existing.

- [ ] **Step 6: Add the `--gated-peer` CLI flag to `scripts/eval_peer_selection.py`**

Add the argument (with the other `parser.add_argument` calls):

```python
    parser.add_argument(
        "--gated-peer", default=None,
        help="If set, report how many outcomes still reached this peer_node_id "
             "(expected 0 when that peer is trust-gated in the target node's config)",
    )
```

Update the `compute_selection_metrics(...)` call to pass it through:

```python
    metrics = compute_selection_metrics(outcomes, gated_peer=args.gated_peer)
```

- [ ] **Step 7: Sanity-check the driver parses cleanly**

Run: `./venv/Scripts/python.exe -c "import ast; ast.parse(open('scripts/eval_peer_selection.py').read()); print('parse ok')"`
Expected: `parse ok`. Do NOT run the driver itself here — it needs live
node processes (Task 9).

- [ ] **Step 8: Commit**

```bash
git add config/node_a_trust_gated.yaml gin/federation/selection_eval.py scripts/eval_peer_selection.py tests/test_selection_eval.py
git commit -m "Gated eval config variant + gated_peer_contacted metric (pre-eval)."
```

---

### Task 9: Live eval (gated + ungated regression) + docs + final validation

**Files:**
- Modify: `architecture.md`, `README.md`, `docs/GIN_Node_Architecture_v1.md`

**Interfaces:**
- Consumes: everything from Tasks 1–8; the already-provisioned 3-node
  deployment (Task 7).
- Produces: two eval artifacts under `data/eval_runs/<ts>/`; doc updates
  with real run timestamps and numbers.

- [ ] **Step 1: Start the three nodes**

In three terminals (or background processes), in this order (B and C
first, so A's background sync loop has peers to reach immediately on
startup):

```bash
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_b.yaml
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_c.yaml
```

Then, for the **gated run**, start node A with the trust-gated config:

```bash
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_a_trust_gated.yaml
```

Wait for all three to log `Uvicorn running`.

- [ ] **Step 2: Confirm node A's peer summaries include domains before querying**

```bash
docker exec gin-postgres psql -U gin -d gin_node_a -tAc \
  "SELECT peer_node_id, domains FROM peer_summaries ORDER BY peer_node_id;"
```

Expected: `node_b | ["environmental_impact"]` and
`node_c | ["monetary_policy"]` (order of the JSON array may differ if a
peer's corpus has multiple domains; each of these three corpora currently
has exactly one). If empty, wait for at least one sync cycle
(`anchor_sync_interval_s`, 10s in these configs) and re-check.

- [ ] **Step 3: Run the gated eval**

```bash
./venv/Scripts/python.exe scripts/eval_peer_selection.py --gated-peer node_c
```

Expected: `gated_peer_contacted: 0`; `honest_refusal_rate: 1.0` for the
`c_only` queries specifically (check `per_query` in the written artifact —
`sel_c_dual_mandate` and `sel_c_quantitative_easing` should show
`refused=True`, empty or `node_b`-only `peers_attempted`, and `source=""`);
`b_only`/`a_answerable`/`neither` classes unaffected (same shape as
sub-project 3). Note the run timestamp from the printed `artifact:` path.

- [ ] **Step 4: Stop node A, restart with the ungated (default) config, re-run**

```bash
# stop node A's process (Ctrl-C or kill its PID)
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_a.yaml
```

Wait for `Uvicorn running`, then confirm sync cycles have run (repeat Step
2's query against `config/node_a.yaml`'s port if different, or simply wait
one `anchor_sync_interval_s`), then:

```bash
./venv/Scripts/python.exe scripts/eval_peer_selection.py
```

Expected: reproduces sub-project 3's exact result set — `selection_precision_at_1: 1.0`,
`avg_peers_tried: 1.0`, `routing_false_positives: 0`,
`routed_fabrication_rate: 0.0`, `honest_refusal_rate: 1.0`. This is the
regression proof that trust gating is opt-in. Note this run's timestamp too.

- [ ] **Step 5: Shut down all three node processes**

Confirm no `node_serve.py` processes remain running (they hold model
instances / GPU memory).

- [ ] **Step 6: Update `architecture.md` Phase 3 checklist**

Replace the `🔲 Trust weights (per-domain asymmetric), PKI/mTLS` line with:

```markdown
- ✅ Trust weights (per-domain peer gating, spec #4) — a node excludes a
  peer from ranked delegation candidates when a configured
  `(peer, domain)` trust weight falls below threshold; domain coverage
  synced automatically alongside the routing summary, gating applied after
  RRF ranking and before delegation. Measured on the live 3-node
  deployment: gated run `data/eval_runs/<gated-ts>/peer_selection_metrics.json`
  (gated_peer_contacted 0, honest_refusal_rate 1.0 for the affected
  queries); ungated regression run
  `data/eval_runs/<ungated-ts>/peer_selection_metrics.json` reproduces
  sub-project 3's exact bar (precision@1 1.0, avg peers tried 1.0).
  Spec: docs/superpowers/specs/2026-07-15-trust-weights-design.md
- 🔲 gRPC/QUIC wire, PKI/mTLS
```

(Substitute the two real run timestamps for `<gated-ts>`/`<ungated-ts>`.)

- [ ] **Step 7: Update `README.md`**

Add a "Trust weights" subsection after the existing "Peer selection (three
nodes)" subsection (before its closing `---`):

```markdown
### Trust weights (per-domain peer gating)

A node can exclude a specific peer from delegation for a domain it serves,
via its own config — no runtime API, no automated inference. `config/node_a_trust_gated.yaml`
is a variant of `config/node_a.yaml` carrying a below-threshold weight for
`node_c`'s `monetary_policy` domain:

```bash
# gated run (node_c excluded for monetary_policy)
python scripts/node_serve.py --config config/node_a_trust_gated.yaml
python scripts/eval_peer_selection.py --gated-peer node_c

# ungated regression (default config, reproduces sub-project 3 exactly)
python scripts/node_serve.py --config config/node_a.yaml
python scripts/eval_peer_selection.py
```

Domain coverage syncs automatically (no query-time classification); a peer
with no known domains is never gated. Bar and scope:
docs/superpowers/specs/2026-07-15-trust-weights-design.md.
```

Update the federation status-table row to append: `; ✅ trust weights measured
(gated run <gated-ts>: gated_peer_contacted 0; ungated regression <ungated-ts>
reproduces N=3 bar exactly); gRPC/QUIC + mTLS deferred`.

- [ ] **Step 8: Update `docs/GIN_Node_Architecture_v1.md`**

Add a v1 implementation note after the existing peer-selection note (which
already sits after the trust-weights bullet):

```markdown
> **v1 trust weights (2026-07):** gating only, not yet blended into the
> ranking score — a peer is excluded entirely if any domain it's known to
> serve (synced automatically, never query-classified) falls below a
> configured `(peer, domain)` weight. Configured statically by a human in
> each node's own YAML, standing in for the not-yet-built Epistemic
> Council; dynamic weight-setting and blending trust into the fused RRF
> score remain future work.
```

- [ ] **Step 9: Full suite + final validation**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all green (integration tests run with Postgres up).

Run: `git status --short`
Expected: only intended files; no stray artifacts beyond the two eval
metrics JSON files.

- [ ] **Step 10: Commit** (hold the push for the final whole-branch review,
matching every prior sub-project)

```bash
git add data/eval_runs/<gated-ts>/peer_selection_metrics.json \
        data/eval_runs/<ungated-ts>/peer_selection_metrics.json \
        architecture.md README.md docs/GIN_Node_Architecture_v1.md
git commit -m "Trust weights measured on live 3-node deployment: gated 0-contact, ungated regression exact.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

(Substitute the real run timestamps for `<gated-ts>`/`<ungated-ts>`. Do NOT
push — the controlling session pushes after the final whole-branch review.)

---

## Self-review notes (already applied)

- **Spec coverage:** falsifiable claim (gated_peer_contacted 0, honest
  refusal 1.0, ungated regression exact) → Task 9; gate-not-blend, static
  config, domain-not-query-classification, conservative any-domain-gates
  policy, synced-not-declared domain coverage, absence-never-gates → Tasks
  1–2, 6; `documents.domain` persistence gap found during spec review →
  Task 4; `peer_summaries.domains` persistence (the actual query-time cache)
  → Task 5; gate sits between ranking and delegation, untouched router/RRF
  → Task 6; three-tier testing (unit → Tasks 2/3, integration → Task 6,
  live → Task 9) → present; docs → Task 9.
- **Placeholder scan:** no TBD/TODO. `<ts>`/`<gated-ts>`/`<ungated-ts>` in
  Task 9's doc/commit steps are explicit "substitute the real run
  timestamp" instructions, not code placeholders.
- **Type consistency:** `PeerSummaryResponse.domains: list[str]`,
  `is_trusted`/`filter_trusted` signatures, `NodeConfig.trust_weights: dict[str, dict[str, float]]`
  / `trust_gate_threshold: float`, `DocumentDraft.domain: str`,
  `warm.upsert_document(..., domain: str = "")`,
  `compute_selection_metrics(outcomes, gated_peer: Optional[str] = None)` —
  verified identical across every task that references them (schema Task 1
  → trust_gate Task 2 → config Task 3 → ingest Task 4 → store Task 5 →
  server Task 6 → eval Task 8).
- **Non-breaking check:** every new field/parameter (`documents.domain`,
  `PeerSummaryResponse.domains`, `NodeConfig.trust_weights`/
  `trust_gate_threshold`, `warm.upsert_document`'s `domain` kwarg) has a
  default reproducing prior behavior (`''`, `[]`, `{}`/`0.5`, `""`
  respectively); `_rank_peers_for_query`'s new filter step is a pure
  pass-through no-op when `trust_weights` is empty (`filter_trusted` keeps
  every peer whose weights dict is `{}`, since `is_trusted` defaults every
  domain to `1.0 >= 0.5`); `config/node_a.yaml` is never modified, so every
  existing test and the ungated eval run reproduce sub-project 3 exactly.
