# Merkle Anchor Sync (Design)

**Date:** 2026-07-14
**Status:** approved design, pre-implementation
**Phase:** GIN Phase 3 (federation), second sub-project

---

## Falsifiable claim

Each node maintains an eventually-consistent local cache of its peer's anchor
set (`chunk_id`, `content_hash`, `outlet`, `title` — never chunk text) via
periodic background sync, and does so with bandwidth that scales with what
changed, not with corpus size.

| Metric | Bar |
|---|---|
| Cache correctness vs. peer's ground-truth anchor set, after convergence | 0 false adds / removes / changes |
| Bytes transferred on a no-op sync cycle (root already matches) | O(1) — root hash only |
| Bytes transferred on a cycle following a single-chunk mutation | O(bucket), not O(corpus) |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-14, with rationale)

1. **Not load-bearing at N=2, built anyway as the primitive.** With two
   nodes, "sync with peer" degenerates to "sync with your only peer" — same
   status the federation v1 spec gave routing at N=2. It becomes load-bearing
   at N>2 for peer selection (ask a specific node only if its anchor cache
   suggests it might have relevant material) and for reducing repeated full
   re-fingerprinting.
2. **Metadata only, chosen deliberately over the tighter alternative.**
   `chunk_id` + `content_hash` alone (matching today's `corpus_fingerprint`
   tuple) would be the strictest reading of right-to-opacity. This design
   also syncs doc-level `outlet` + `title` — enough for future peer-selection
   heuristics ("does this peer likely cover topic Y") at the cost of leaking
   topical/source signal about a peer's corpus even on cycles where no query
   ever routes there. Chunk text itself never leaves its owning node under
   any configuration.
3. **Prefix-bucketed tree, not a plain sorted-array Merkle tree.** A sorted
   binary tree over positions works for detecting *modifications in place*
   but degrades badly on insertions/deletions — a single insert shifts every
   leaf after it, so the diff looks like "everything past this point
   changed." Partitioning chunks into a fixed 16 buckets by
   `sha256(chunk_id)[0]` makes each chunk's bucket membership stable
   regardless of what else is inserted or deleted elsewhere in the set, so a
   single-chunk change perturbs exactly one bucket hash and the root — the
   actual property the bandwidth bar measures.
4. **Background periodic sync inside each node process, not a separate
   script or a one-shot pull.** Matches the "sovereign, always-running node"
   framing already established by federation v1's server; no new process to
   deploy or keep alive.
5. **Bidirectional, symmetric pull.** Each node polls its peer's root on its
   own interval; no node is a designated authority for anchor state. This
   matches the sovereign-peer model — routing already treats A and B as
   equals with different corpora, not client/server roles.

## Architecture

Extends `gin/federation/` rather than opening a new package — this is
another wire-level concern between the same two node processes.

| Module | Change |
|---|---|
| `gin/federation/schema.py` | New Pydantic messages: `AnchorRootResponse`, `AnchorBucketsResponse`, `AnchorLeaf`, `AnchorLeavesResponse`. |
| `gin/federation/anchor_tree.py` (new) | Pure functions: build the 16-bucket tree from `(chunk_id, content_hash, outlet, title)` rows; compute bucket hashes and root hash; diff two bucket-leaf-lists into added/removed/changed. No I/O — same "pure logic, tested without a model or DB" pattern as `router.py`'s trigger logic. |
| `gin/federation/anchor_sync.py` (new) | The sync loop: fetch peer root via `PeerClient` → compare to root recomputed from local `peer_anchors` cache → on mismatch, fetch peer buckets → recompute local bucket hashes → fetch only mismatched buckets' leaves → upsert/delete `peer_anchors` rows for those buckets only. |
| `gin/federation/server.py` | Three new `GET` endpoints (same bearer auth dependency as the existing federated-query route): `/v1/federated/anchors/root`, `/v1/federated/anchors/buckets`, `/v1/federated/anchors/bucket/{i}`. FastAPI lifespan starts the sync loop as a background `asyncio.Task` on startup, cancels it on shutdown. |
| `gin/federation/config.py` | Add `anchor_sync_interval_s` (default 30) to `NodeConfig`. |
| `gin/corpus/db.py` (migration) | New table `peer_anchors`. |

**New table `peer_anchors`:**

```sql
CREATE TABLE peer_anchors (
    peer_node_id  TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    outlet        TEXT NOT NULL,
    title         TEXT NOT NULL,
    bucket_index  SMALLINT NOT NULL,
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (peer_node_id, chunk_id)
);
```

No separate hash-cache table: bucket and root hashes are recomputed from
`peer_anchors` each cycle via the same pure function used to build the
authoritative tree, so an empty cache naturally bootstraps as "all 16 buckets
differ" with no special-cased first-sync path.

## The tree

- **Leaves:** one per chunk, assigned to bucket `sha256(chunk_id)[0]` (first
  hex digit → 16 buckets, fixed protocol constant for v1, documented next to
  `protocol_version` in `schema.py`).
- **Bucket hash:** `sha256` over the bucket's `(chunk_id, content_hash,
  outlet, title)` tuples, sorted by `chunk_id`, joined deterministically —
  same construction style as `corpus_fingerprint`'s flat hash.
- **Root hash:** `sha256` over the 16 bucket hashes in fixed index order.
- Empty buckets hash a fixed empty-string sentinel so root computation never
  branches on bucket occupancy.

This is deliberately a 2-level tree (root → 16 buckets → leaves), not a full
binary tree down to individual chunks. At corpus sizes in the tens to low
hundreds of chunks, drilling further than "which of 16 buckets changed"
buys no additional bandwidth savings and adds real implementation
complexity (rebalancing on insert/delete). Revisit bucket count or add a
level only if corpus sizes grow enough that a "changed bucket" still means
transferring hundreds of leaves.

## Sync loop (per node, per peer, per cycle)

1. `GET {peer}/v1/federated/anchors/root` → `AnchorRootResponse`.
2. Recompute local root from current `peer_anchors` rows for this peer.
3. If hashes match: done. (1 round trip, no leaf data transferred.)
4. Else: `GET {peer}/v1/federated/anchors/buckets` → 16 bucket hashes.
5. Recompute local bucket hashes from `peer_anchors`; diff index-by-index
   against the fetched hashes.
6. For each mismatched bucket index *i*: `GET {peer}/v1/federated/anchors/bucket/{i}`
   → full leaf list for that bucket. Diff against locally cached rows for
   that `(peer_node_id, bucket_index)`; upsert changed/added rows, delete
   rows no longer present.
7. Matched buckets are never fetched — this is the bandwidth property under
   test.

Peer unreachable or auth failure: log and skip the cycle, retry next
interval. This is background maintenance, not on the query-answering path —
it must never block or fail a live federated query. No exponential backoff
in v1 (fixed interval is enough at N=2; add backoff if this becomes
noisy at N>2).

## Wire protocol additions (`schema.py`)

**`AnchorRootResponse`** — `node_id`, `root_hash`, `leaf_count`.

**`AnchorBucketsResponse`** — `node_id`, `bucket_hashes: list[str]` (length
16, fixed index order).

**`AnchorLeaf`** — `chunk_id`, `content_hash`, `outlet`, `title`.

**`AnchorLeavesResponse`** — `node_id`, `bucket_index`, `leaves: list[AnchorLeaf]`.

All reuse the existing `protocol_version` field convention; a version
mismatch on any anchor endpoint returns the same typed refusal path as
federated queries.

## Testing — three tiers

1. **Unit (no model, no DB):** tree-building is deterministic and
   input-order-independent; a single insert, delete, or content-hash change
   in a synthetic chunk set perturbs exactly one of 16 bucket hashes (and
   the root) while the other 15 are byte-identical to the pre-change tree;
   bucket assignment is stable under insertions elsewhere in the set (the
   property that motivates bucketing over a plain sorted-array tree).
2. **Integration (no GGUF, CI-safe):** two in-process FastAPI apps over a
   test HTTP transport (same harness as federation v1's integration tier).
   Seed each with a small synthetic anchor set, run the sync loop for N
   cycles, assert the cache converges to ground truth; mutate one chunk
   between cycles and assert only its bucket round-trips.
3. **Live eval (`scripts/eval_anchor_sync.py`):** against the real two-node
   deployment (`gin_node_a`, `gin_node_b`) with corpora already ingested.
   Run sync to convergence; the driver — which, like
   `scripts/eval_federation.py`, legitimately holds both DB connections —
   diffs node A's `peer_anchors` cache of B against B's actual `chunks` JOIN
   `documents`, asserting the correctness bar. Then mutate one chunk's
   `content_hash` directly in `gin_node_b`, run one more sync cycle, and
   compare the cycle's transferred bytes (sum of response body sizes) against
   the no-op baseline cycle to confirm O(bucket) vs. O(corpus) scaling.
   Writes `data/eval_runs/<ts>/anchor_sync_metrics.json`.

## Out of scope (later, in likely order)

1. Peer selection using the synced anchor cache (>2 nodes)
2. Trust weights / per-domain asymmetric trust (per `GIN_Node_Architecture_v1`)
3. Exponential backoff / adaptive sync interval
4. Deeper tree (>2 levels) if bucket sizes grow large enough to matter
5. gRPC/QUIC wire (same deferred item as federation v1 — orthogonal)
6. Conflict resolution beyond last-write-wins (architecture doc's stated v1
   posture; no conflicting-write scenario exists yet since each node only
   ever writes its own corpus)

## Documentation updates shipped with implementation

- `architecture.md` Phase 3 checklist: mark Merkle anchor sync in progress / done.
- `README.md`: note the anchor-sync background loop in the federation
  quick-start section.
- `docs/GIN_Node_Architecture_v1.md`: note that the Corpus-Diff Sync Endpoint
  described there is implemented as a 2-level prefix-bucketed tree, not a
  full Merkle trie, for the current corpus-size regime.

## New dependencies

None — reuses `httpx`/`fastapi` from federation v1 and the stdlib `hashlib`.
