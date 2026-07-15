# Trust Weights: Per-Domain Peer Gating (Design)

**Date:** 2026-07-15
**Status:** approved design, pre-implementation
**Phase:** GIN Phase 3 (federation), fourth sub-project

---

## Falsifiable claim

Given the existing 3-node deployment, when node A's operator configures a
below-threshold trust weight for node_c in the `monetary_policy` domain (the
one domain node_c serves), `c_only` queries — which today route to node_c with
precision@1 1.0 (sub-project 3) — must **never contact node_c**, and since no
other peer covers that domain, must resolve to **honest refusal**. Every other
query class, and every query when `trust_weights` is unconfigured, must
reproduce sub-project 3's baseline exactly.

| Metric | Bar |
|---|---|
| Gated peer contacted (`c_only` queries, node_c gated) | 0 |
| Honest refusal rate (`c_only` queries, node_c gated) | 1.0 |
| Regression: `a_answerable`/`b_only`/`neither` classes (unaffected by the gate) | unchanged from sub-project 3 (routing FP 0, `b_only` precision@1 1.0, honest refusal 1.0 for `neither`) |
| Regression: full queryset with `trust_weights` empty/absent | reproduces sub-project 3's exact result set (precision@1 1.0, avg peers tried 1.0, fabrication 0.0, attribution 1.0, honest refusal 1.0) |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-15, with rationale)

1. **Gate, not blend.** A trust weight below threshold removes a peer from
   the ranked candidate list entirely — it is never contacted for that
   query — rather than nudging its RRF score. This is the simplest correct
   behavior for a first increment and matches the architecture doc's
   "quarantine" framing; blending trust into the fused score is a later
   refinement once there's a reason to need graduated behavior instead of
   hard exclusion.
2. **Domain, not query classification.** No query-time domain classifier is
   introduced. Gating instead keys off which domain(s) a *peer's own corpus*
   covers — information a node can know about its peers without ever
   inspecting what a live query is about. This trades some precision (a
   peer covering multiple domains, one trusted and one not, is gated
   wholesale — see decision 4) for avoiding an entirely new classification
   subsystem this sub-project doesn't need.
3. **Static human-set config, not a runtime API or automated inference.**
   Trust weights are declared in each node's own YAML, the same way peers
   and secrets already are. No admin endpoint, no Postgres-backed mutable
   store, no derivation from attribution-verification history. A human
   stands in for the not-yet-built Epistemic Council, exactly as the
   architecture doc anticipates ("new node starts with default weights;
   Epistemic Council reviews and sets final weights").
4. **Conservative gating policy: any distrusted domain gates the whole
   peer.** A peer is eligible only if *every* domain it's known to serve
   clears the node's configured weight (default `1.0`, i.e. fully trusted,
   when unconfigured). Since queries aren't domain-classified, there's no
   way to know whether a given query needs the peer's trusted domain or its
   distrusted one — so the conservative choice is exclusion, not partial
   inclusion. A finer per-query policy is future work, same as query
   classification itself.
5. **Domain must actually be persisted to be queryable — it isn't today.**
   `metadata.domain` from the corpus JSON is currently dropped at ingest
   time; only `metadata.category` survives, mapped to `chunks.eval_tag`.
   This sub-project adds a `documents.domain` column and the corresponding
   one-line ingestion mapping (parallel to the existing `category` →
   `eval_tag` mapping) so `metadata.domain` becomes real, queryable data —
   not a repurposing of `eval_tag`/category, which is a narrower, different
   field (`central_banking`, `inflation`) than domain
   (`monetary_policy`).
6. **Domain coverage rides the existing summary-sync loop.** No new sync
   mechanism, no new wire message. `PeerSummaryResponse` gains one additive
   field (`domains: list[str]`); it's computed in `build_local_summary`
   (which already reads this node's own chunk rows) and served over the
   already-existing `/v1/federated/summary` endpoint, synced by the
   already-existing per-peer background loop (sub-project 3, further
   hardened in the live-eval fixes).
7. **The gate sits between ranking and delegation, touching nothing
   downstream.** It's a filter applied to the ranked list `rank_peers`
   already produces, before that list reaches the router's fallback loop.
   `answer_or_delegate`, `peer_selection.py`'s ranking logic, and the
   router's sequential-fallback/hop_count=1 loop-prevention are untouched —
   a gated peer is simply never in the list the router iterates, which is
   the same code path as "this peer doesn't exist."
8. **Absence of information never gates.** A peer with no synced summary
   yet has an empty `domains` list; `all(...)` over an empty list is
   vacuously true, so it is *not* gated — it falls through to sub-project
   3's existing "no-summary peers ranked last, never dropped" behavior.
   Trust gating only ever acts on positive domain information, symmetric
   with that existing invariant.

## Architecture

Extends `gin/federation/` (built across the three prior sub-projects) and the
ingestion pipeline (`docker/init-db.sql`, `scripts/corpus_ingest.py`) that
feeds it.

| Module | Change |
|---|---|
| `docker/init-db.sql` | `documents` gains a `domain TEXT NOT NULL DEFAULT ''` column. |
| `scripts/corpus_ingest.py` | Map `metadata.domain` → `documents.domain` at ingest time, parallel to the existing `metadata.category` → `chunks.eval_tag` mapping. |
| `gin/federation/schema.py` | `PeerSummaryResponse` gains `domains: list[str]` (default empty — additive, non-breaking). |
| `gin/federation/peer_summary_store.py` | `build_local_summary` additionally queries the distinct `documents.domain` values for this node's corpus and populates the new field. |
| `gin/federation/config.py` | `NodeConfig` gains `trust_weights: dict[str, dict[str, float]]` (peer_node_id → {domain: weight}, default empty) and `trust_gate_threshold: float` (default `0.5`). |
| `gin/federation/trust_gate.py` (new) | Pure logic, no I/O: `is_trusted(peer_domains: list[str], peer_weights: dict[str, float], threshold: float) -> bool`; `filter_trusted(ranked_peer_ids: list[str], domains_by_peer: dict[str, list[str]], trust_weights: dict[str, dict[str, float]], threshold: float) -> list[str]` — `domains_by_peer` maps a peer_id to its synced `domains` list, or is simply absent/empty for a peer with no cached summary, which `is_trusted` then passes vacuously (decision 8). Same "pure, unit-testable without a model or DB" pattern as `peer_selection.py` and `anchor_tree.py`. |
| `gin/federation/server.py` | `_rank_peers_for_query`'s ranked output is passed through the new trust filter before being returned as the `peer_ranker` the router consumes. No change to `answer_or_delegate`'s signature or the router's loop. |
| `config/node_a.yaml` (eval scenario) | Adds `trust_weights: {node_c: {monetary_policy: 0.1}}` for the live-eval gated run; a second run uses the unmodified config (no `trust_weights`) to prove the regression bar. |

## Data flow

1. At ingest, `corpus_ingest.py` now writes `metadata.domain` into
   `documents.domain` for every document (in addition to the existing
   `category` → `eval_tag` mapping). Purely additive to the schema and the
   script; no existing behavior changes for corpora/tools that don't set
   `metadata.domain` (defaults to `''`, which never matches a configured
   trust-weight domain key, so untagged corpora are simply never gated).
2. `build_local_summary` queries `SELECT DISTINCT domain FROM documents
   WHERE domain != ''` alongside its existing centroid/IDF computation, and
   includes the result in the `PeerSummaryResponse` it serves.
3. The existing per-peer background sync loop (sub-project 3) caches this
   field in `PeerSummaryStore` exactly as it already caches
   `embedding_centroid`/`distinctive_terms` — no new sync trigger, no new
   staleness logic.
4. When node A ranks peers for a query, `_rank_peers_for_query` first
   computes the RRF-fused order (unchanged), then filters it: a peer is kept
   only if, for every domain in its cached `domains`, `trust_weights.get(peer,
   {}).get(domain, 1.0) >= trust_gate_threshold`. Peers with no cached
   summary (empty `domains`) are never filtered out by this step.
5. The filtered, still-ranked list is what the router's existing
   sequential-fallback loop consumes — a gated peer is simply absent, so if
   it was the only peer covering the query's actual domain, every remaining
   candidate refuses and the router returns its existing aggregated honest
   refusal, unchanged.

## Error handling

- **Peer with no synced summary:** not gated (decision 8) — same as
  sub-project 3's existing "ranked last, never dropped" handling.
- **Untagged corpus (`domain = ''`):** never matches a configured
  trust-weight key (which are always real domain strings), so such peers
  are never gated. This is the default for any corpus that doesn't set
  `metadata.domain` — fully backward compatible.
- **Unconfigured `trust_weights` (empty dict, the default):** every domain's
  implicit weight is `1.0` (full trust), which clears any `trust_gate_threshold`
  ≤ `1.0` — including the default `0.5` — so every peer passes the gate
  trivially, reproducing sub-project 3's ungated behavior exactly regardless
  of the configured threshold.
- **All candidates gated or refusing:** identical to today's "every peer
  exhausted" path — an aggregated honest refusal, no new refusal reason
  needed (the router doesn't distinguish "refused by the peer" from "never
  contacted because gated" in `refusal_reasons`, since gating happens before
  the router's loop even sees that peer as a candidate — this is
  intentional: from the router's perspective a gated peer was never a
  candidate, exactly as if it didn't exist).

## Testing — three tiers

1. **Unit (no I/O, no model, no DB):** `trust_gate.py`'s `is_trusted`/
   `filter_trusted` — a peer with all-domain weights ≥ threshold passes; a
   peer with any domain below threshold is excluded; a peer with no synced
   domains is never excluded (vacuous pass); an empty `trust_weights` dict
   defaults every peer to trusted (this is the regression proof at the pure-logic
   level); domain-level asymmetry (peer P trusted by A but not by B, or vice
   versa) is representable and independent per node config.
2. **Integration (in-process, no GPU/DB):** extends `test_peer_selection_loop.py`'s
   three-real-socket pattern — inject a summary with `domains=["monetary_policy"]`
   for node_c, configure a below-threshold weight in node A's `NodeConfig`,
   and confirm a query that would otherwise route to node_c never contacts
   it and resolves to refusal, while an ungated query is unaffected.
3. **Live eval:** two runs of the existing `scripts/eval_peer_selection.py`
   (unmodified) against the existing 3-node deployment and queryset:
   - **Gated run:** node_a's config carries `trust_weights: {node_c:
     {monetary_policy: 0.1}}`. New metric in `gin/federation/selection_eval.py`:
     `gated_peer_contacted` — must be `0` across the `c_only` queries;
     `honest_refusal_rate` for those same queries must be `1.0` (node_b
     doesn't cover `monetary_policy` either).
   - **Ungated run:** node_a's config unchanged (no `trust_weights`) — must
     reproduce sub-project 3's exact result set, proving the mechanism is
     opt-in and non-breaking.

## Out of scope (later, in likely order — unchanged from the original roadmap)

1. Blending trust into the RRF-fused score instead of hard-gating
2. Query-time domain classification (finer-grained than "gate the whole peer")
3. Runtime/API-driven weight updates, Postgres-backed mutable trust store
4. Epistemic Council automation (dynamic weight-setting, probationary review)
5. gRPC/QUIC wire
6. PKI/mTLS

## Documentation updates shipped with implementation

- `architecture.md` Phase 3 checklist: mark trust weights done/measured with
  the live-eval numbers; update the remaining-work line.
- `README.md`: a short "trust weights" subsection alongside the existing
  peer-selection one, showing the gated-run config snippet and the eval
  command.
- `docs/GIN_Node_Architecture_v1.md`: a v1 implementation note (matching the
  existing peer-selection note's pattern) clarifying that weights are
  per-`(peer, domain)`, gate-only (not yet blended into ranking), configured
  statically by a human, with domain coverage synced automatically — and
  that Epistemic Council-driven dynamic weight-setting remains future work.

## New dependencies

None — reuses the existing Postgres/psycopg stack, the existing summary-sync
loop, and the existing `NodeConfig`/YAML loading machinery.
