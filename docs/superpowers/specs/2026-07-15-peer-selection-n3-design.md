# Peer Selection at N>2 Nodes (Design)

**Date:** 2026-07-15
**Status:** approved design, pre-implementation
**Phase:** GIN Phase 3 (federation), third sub-project

---

## Falsifiable claim

Given three nodes (A, B, C), when Node A cannot ground a query locally, it
picks the correct peer to delegate to **on the first try** — not by guessing,
and not by trying every peer in sequence — using only data already synced
from peers, never their corpus text.

| Metric | Bar |
|---|---|
| Selection precision@1 (b_only/c_only queries, correct peer tried first) | 1.0 |
| Average peers tried per routed query | ≈ 1.0 (not N-1) |
| Routing false-positive rate (a_answerable queries that route) | 0 |
| Fabrication rate on routed answers | 0.000 |
| Attribution verification (answering peer's spans verify against its corpus) | 1.0 |
| Honest refusal rate (neither-node-answerable queries) | 1.0 |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-15, with rationale)

1. **Three nodes, not more.** Two peers is the minimum at which "selection"
   is a real decision rather than "ask your only peer" (the exact reasoning
   that made Merkle sync not load-bearing at N=2). A fourth corpus/node adds
   deployment complexity without proving anything a third node doesn't
   already prove.
2. **Selection signal: dense + sparse, RRF-fused — reusing the retrieval
   stack's own fusion, not inventing a new one.** `gin/corpus/retrieve.py`
   already fuses dense pgvector search and sparse tsvector search via
   reciprocal rank fusion (`RRF_K=60`). Peer selection mirrors this exactly
   at the node level: rank peers by cosine similarity of the query embedding
   against each peer's cached embedding centroid (dense), rank peers by
   IDF-weighted term overlap against each peer's cached distinctive-terms
   set (sparse), then RRF-fuse the two rankings with the same constant.
3. **The signal is a per-node aggregate, not per-chunk, and is NOT
   Merkle-tree-diffed.** Unlike anchor metadata (one row per chunk, tree-diffed
   because there can be tens to hundreds of them), a node's routing summary
   is one small object: an embedding centroid (384 floats) and a top-N
   distinctive-terms map. It's cheap enough to fetch whole; the only
   bandwidth discipline worth keeping is *not* re-fetching it every cycle —
   so it piggybacks on the anchor sync loop's existing root-mismatch
   detection (already built) rather than adding a second independent poll.
4. **Sequential fallback through the ranked list, not parallel fan-out.**
   If the top-ranked peer refuses, A tries the next-ranked peer, and so on.
   Every delegated request is still `hop_count=1` and no peer ever
   re-delegates — loop prevention stays exactly as structural as it was at
   N=2; A may now simply make more than one hop-1 request per query,
   sequentially, stopping at the first success.
5. **No trust weights, no persistent reputation.** Selection is pure
   content-similarity, recomputed from the current synced summary every
   time. Per-domain asymmetric trust (the Council-governance model in
   `GIN_Node_Architecture_v1.md`) is a later sub-project, deliberately kept
   separate — trust is a governance/social concern, similarity is a
   retrieval concern, and conflating them would make either one harder to
   reason about or test in isolation.
6. **Peers with no cached summary yet are tried last, never excluded.** A
   fresh deployment (or a peer whose first sync hasn't landed) shouldn't be
   silently dropped from consideration — it's just deprioritized behind
   every peer for which A actually has a signal. With zero summaries synced
   anywhere, this degrades exactly to v1's plain sequential order.

## Architecture

Extends the existing 3-process-capable federation deployment (was 2) with a
third node: `config/node_c.yaml`, database `gin_node_c`, and a new
`corpus_node3.json` — a synthetic corpus in a topical domain distinct from
node1 (institutional/climate) and node2 (grassroots/environmental-justice),
so `b_only`/`c_only` query classes are genuinely separable.

| Module | Change |
|---|---|
| `gin/federation/schema.py` | New `PeerSummaryResponse` wire message. |
| `gin/federation/peer_selection.py` (new) | Pure logic, no I/O: `dense_rank`, `sparse_rank`, `rank_peers` (RRF fusion, reusing `RRF_K` from `gin.corpus.retrieve`). Same "pure logic, unit-testable without a model or DB" pattern as `anchor_tree.py`. |
| `gin/federation/peer_summary_store.py` (new) | `PeerSummaryStore` Protocol + `InMemoryPeerSummaryStore` + `PostgresPeerSummaryStore`, same split as `PeerAnchorStore`. Backed by a new `peer_summaries` table. |
| `gin/federation/client.py` | New `PeerClient.get_summary(peer) -> PeerSummaryResponse` method, same pattern as the anchor GET methods. |
| `gin/federation/anchor_sync.py` | `sync_once` additionally triggers a summary refetch when the peer's anchor root doesn't match the cached one (summary is assumed stale exactly when anchors are). |
| `gin/federation/server.py` | New `GET /v1/federated/summary` endpoint, same auth dependency as every other route; computed live from local chunks via `hot.embed_texts` + `relevance.corpus_idf`. |
| `gin/federation/router.py` | `answer_or_delegate`: when `len(config.peers) > 1`, rank peers via cached summaries before trying them in order; `len(config.peers) == 1` keeps the exact v1 code path unchanged (no ranking overhead where there's nothing to rank). |

**New table `peer_summaries`:**

```sql
CREATE TABLE peer_summaries (
    peer_node_id       TEXT PRIMARY KEY,
    embedding_centroid REAL[] NOT NULL,
    distinctive_terms  JSONB NOT NULL,
    synced_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## The selection signal

- **Embedding centroid:** mean of `hot.embed_texts(all chunk texts)` for the
  node's own corpus, normalized. Computed live per request on
  `/v1/federated/summary` (corpora are small — tens to ~100 chunks — so this
  is cheap; no need to cache server-side beyond what the requester caches).
- **Distinctive terms:** `relevance.corpus_idf(all chunk texts)`, keeping the
  top 40 tokens by IDF weight. Reuses the existing normalization
  (singular/plural fold, stopword filtering) already validated by the
  divergence-gate work.
- **Dense rank:** cosine similarity between `hot.embed_query(query)` and each
  cached peer centroid, descending.
- **Sparse rank:** for each peer, sum the IDF weights (from that peer's
  cached distinctive-terms map) of every query keyword
  (`relevance.query_keywords(query)`) that appears in it; descending.
- **Fusion:** `score[peer] = 1/(RRF_K + dense_rank[peer]) + 1/(RRF_K + sparse_rank[peer])`,
  same formula and `RRF_K=60` as `gin/corpus/retrieve.py`'s hybrid retrieval —
  imported, not reimplemented.
- **No-signal peers:** appended after every ranked peer, in `config.peers`
  order, never dropped.

## Data flow

1. A's existing background anchor-sync loop (per peer) notices the peer's
   root hash differs from its last cached value → also calls
   `GET /v1/federated/summary` on that peer and upserts `peer_summaries`.
   (An empty local cache — first sync ever — trivially counts as "differs.")
2. Caller queries A; A's local answer fails to ground (pre-commitment
   failure, same trigger as v1).
3. Router loads cached summaries for all configured peers, embeds the query
   and extracts its keywords, ranks peers via `peer_selection.rank_peers`.
4. Router calls peers via the existing `HttpPeerClient.query` (`hop_count=1`,
   unchanged wire) in ranked order. First grounded answer wins and is
   relayed with the existing federation provenance layer. If a peer refuses
   or is unreachable, try the next; if every peer is exhausted, return an
   honest refusal aggregating every peer's reason (or `"unreachable"`).

## Error handling

- **Peer unreachable during summary fetch:** logged and skipped by the
  background loop, exactly like a failed anchor-sync cycle — never blocks
  or fails a live query. The stale (or absent) cached summary is used as-is
  on the next query; a peer with no summary at all falls to the back of the
  ranking, not out of consideration.
- **Peer unreachable during the actual delegated query:** recorded as
  `"unreachable"` for that peer, router moves to the next-ranked peer —
  identical to v1's single-peer unreachable handling, just iterated.
- **Zero peers have any cached summary** (cold start before any sync cycle
  has completed): ranking degrades to `config.peers` order — identical to
  v1's behavior before this sub-project existed.
- **Every peer refuses:** aggregated honest refusal, same shape as v1's
  two-peer case, generalized to N-1 reasons.

## Infrastructure constraint (flagged now, not discovered at eval time)

Three simultaneous Mistral-7B-Instruct-v0.3-Q4_K_M instances at ~5GB VRAM
each would need ~15GB against the RTX 4070's 12GB. The live eval will run
one of the three nodes with `n_gpu_layers: 0` (CPU decode) — the escape
hatch already documented as a comment in `config/node_b.yaml` from
Federation v1, here actually exercised rather than hedged against.

## Testing — three tiers

1. **Unit (no I/O, no model, no DB):** `dense_rank`/`sparse_rank` correctness
   on synthetic centroids/term-maps; `rank_peers` RRF fusion — a peer strong
   on both signals outranks one strong on only one; a peer with no summary
   sorts after every peer that has one, never excluded; fusion is
   deterministic and independent of input peer order.
2. **Integration (no GGUF, CI-safe):** three in-process FastAPI apps (extends
   `test_federation_loop.py`'s two-node real-socket pattern to three),
   summaries wired directly into fake stores, proving the router tries
   peers in ranked order and falls back correctly on refusal without ever
   exceeding `hop_count=1`.
3. **Live eval:** three real nodes (`gin_node_a/b/c`, one CPU-only per the
   constraint above), a new class-labeled queryset extending
   `queryset_federation.yaml`'s pattern to four classes (`a_answerable`,
   `b_only`, `c_only`, `neither`). New driver
   `scripts/eval_peer_selection.py` measures selection precision@1, average
   peers-tried-per-routed-query, and the carried-forward v1 bars
   (fabrication, attribution, honest refusal), writing
   `data/eval_runs/<ts>/peer_selection_metrics.json`.

## Out of scope (later, in likely order — unchanged from the original roadmap)

1. Trust weights / per-domain asymmetric trust, Council governance hooks
2. gRPC/QUIC wire (swap inside `PeerClient`, orthogonal to this work)
3. PKI/mTLS
4. Dynamic peer discovery (still a static `config.peers` list, just longer)
5. Cross-node joint divergent synthesis (chunk transfer)

## Documentation updates shipped with implementation

- `architecture.md` Phase 3 checklist: mark peer selection in progress/done,
  note trust weights as the next item.
- `README.md`: three-node quick-start (start three nodes, run the
  peer-selection eval) and status-table row update.
- `docs/GIN_Node_Architecture_v1.md`: note that v1 peer selection is
  content-similarity only (dense+sparse RRF over synced summaries); trust
  weights remain a distinct, later mechanism.

## New dependencies

None — reuses `sentence-transformers` (already a dependency via
`gin/corpus/hot.py`) and the existing `httpx`/`fastapi`/`psycopg` stack.
