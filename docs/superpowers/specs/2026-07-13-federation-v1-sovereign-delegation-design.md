# Federation v1 — Sovereign Delegation Loop (Design)

**Date:** 2026-07-13
**Status:** approved design, pre-implementation
**Phase:** GIN Phase 3 (federation), first sub-project

---

## Falsifiable claim

When Node A cannot ground a query, it routes the query to Node B across a real
network boundary and returns B's answer with attribution intact and explicitly
marked as B's synthesis — with:

| Metric | Bar |
|---|---|
| Fabrication rate on routed queries | 0.000 |
| Routing false-positive rate (A-answerable queries that route) | 0 |
| Routing recall (B-only queries that route) | 1.0 |
| Attribution verification (B's spans verify against B's corpus) | 1.0 |
| Honest refusal on neither-node-answerable queries | 1.0 |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-13, with rationale)

1. **First target: zero-cursor routing loop.** Merkle anchor sync is deferred
   to spec #2 — with two nodes, "route to peer" degenerates to "ask your only
   peer," so sync is not load-bearing at N=2. It becomes load-bearing at N>2
   (peer selection) or for bandwidth.
2. **Deployment: two processes, one machine.** Two Postgres databases, two
   node processes on different localhost ports. A real socket boundary
   (serialization, timeouts, auth) without second hardware. Nothing may
   assume localhost — peers are addressed by URL in config.
3. **Return type: B's finished answer.** B runs its own retrieve → SEAR
   constrained decode and returns generated text + its full attribution
   record. B's corpus text never leaves B except as the exact spans B chose
   to emit. This is the *right to opacity* made mechanical. Joint A+B
   divergent synthesis (chunk transfer) is explicitly a later spec.
4. **Transport: HTTP + JSON now, schema-first.** Pydantic message schemas are
   the versioned protocol contract; the wire is incidental and hidden behind
   a small client interface so gRPC/QUIC (the documented institutional
   target) can replace it without touching routing logic.
5. **Route trigger: both pre-commitment signals.** Delegate on
   `RetrievalConfidenceError` OR zero-cursors-at-first-decode-step — i.e.,
   whenever A fails before emitting any content. Mid-generation cursor death
   keeps its current behavior (graceful local termination); delegating a
   half-emitted answer raises splicing questions that don't belong in v1.

## Architecture

Two sovereign node processes. Each has its own Postgres database
(`gin_node_a` / `gin_node_b` in the existing docker Postgres), its own
ingested corpus, its own llama.cpp model instance, and its own HTTP server.

New package `gin/federation/`:

| Module | Single purpose |
|---|---|
| `schema.py` | Pydantic wire messages (below). This is the protocol contract. |
| `server.py` | FastAPI app factory: `POST /v1/federated/query` wrapping the local answer path. |
| `client.py` | `PeerClient` interface + HTTP implementation: `query(peer, FederatedQuery) → FederatedAnswer \| NodeRefusal`. Injectable, same pattern as the frame judge, so the router is testable with a fake peer and the wire is swappable. |
| `router.py` | Delegation logic: catch pre-commitment grounding failures → try the configured peer → wrap the result with the federation provenance layer. |
| `config.py` | `NodeConfig`: `node_id`, `port`, `database_url`, `model_path`, `n_gpu_layers`, `peers` (v1: exactly one), `shared_secret`, `peer_timeout_s`. Loaded from a per-node YAML. |

New entrypoint: `scripts/node_serve.py --config config/node_a.yaml` — starts
one node process.

**Required refactor (one seam):** the query → retrieve → floor → materialize →
decode path currently driven by `scripts/corpus_generate.py` is extracted
into a callable service function `answer_query(query, ...) → LocalAnswer`,
used by the CLI, the federation server, and the router identically. The
`decode_bundle()` seam extracted during the edge-degradation work already
covers the decode half. `LocalAnswer` carries a structured grounding-failure
signal: which signal fired (`retrieval_floor` / `zero_cursors`) and whether
any content token had been emitted.

## Wire protocol (`schema.py`)

All messages carry `protocol_version: int` (v1 = 1) and `request_id: str`
(UUID, for tracing). Version mismatch → typed refusal, never best-effort
parsing.

**`FederatedQuery`** — `query: str`, `origin_node: str`, `hop_count: int`.

**`FederatedAnswer`** — `node_id` (answering node), `answer_text`,
`attribution_record` (the existing span structure serialized: doc ids,
positions, EXACT/AMBIGUOUS tags, steering tags), `corpus_fingerprint` (B's),
`synthesis_mode`, `timing_s`.

**`NodeRefusal`** — `node_id`, `reason` (enum: `retrieval_floor`,
`zero_cursors`, `hop_limit`, `version_mismatch`), `detail: str`.

HTTP semantics: `401` bad/missing bearer token, `422` malformed body;
everything else returns `200` with `FederatedAnswer | NodeRefusal` — a
refusal is a first-class epistemic outcome, not a transport error.

## Data flow

1. Caller queries Node A (CLI or eval driver, `hop_count = 0` internally).
2. A runs `answer_query`: retrieve → confidence floor → materialize → decode.
3. Pre-commitment failure → `router` POSTs
   `FederatedQuery{query, origin_node: A, hop_count: 1}` to the peer.
4. B authenticates the request, checks `protocol_version` and `hop_count`,
   then runs its own complete `answer_query`. Returns `FederatedAnswer` or
   `NodeRefusal`.
5. A returns the answer to the caller with the layered provenance record
   extended by a **federation layer**:
   `{answered_by, hop_count, transport: "http", peer_url, request_id}`.
6. If B refuses (or is unreachable): A returns an honest refusal aggregating
   both nodes' failure reasons. No fabrication path exists in the loop.

**Loop prevention is structural:** a node never re-delegates an incoming
federated request. A request with `hop_count = 1` is answered from local
corpus or refused with the local failure reason; a request with
`hop_count > 1` is refused outright with `hop_limit` (it should not exist in
a two-node deployment). No routing tables, no discovery; the peer is config.

## Sovereignty and provenance semantics

A relays B's synthesis **without re-verifying it** — A cannot verify spans
against text it does not hold, and that is correct behavior, not a gap. The
provenance record states: "this answer is Node B's synthesis, attributed to
B's corpus at fingerprint X, received at hop 1 over authenticated transport."
Verification stays where the corpus is; trust is legible, not laundered.
Trust *weights* (per-domain, asymmetric — per GIN_Node_Architecture_v1) are a
later phase; v1 trust is binary: the configured peer.

## Auth and failure handling

- **Auth:** shared-secret bearer header; both node configs hold the same
  secret. PKI/mTLS is the institutional-tier posture, deferred. Tailscale
  provides authenticated transport when this leaves localhost.
- **Peer unreachable / timeout** (default 300 s, configurable — decode takes
  tens of seconds): A returns an honest refusal noting the transport failure.
  A never degrades into answering ungrounded.
- **Version mismatch:** typed `NodeRefusal{version_mismatch}`.
- **Concurrency:** v1 is serial; no queueing or backpressure.

## Testing — three tiers

1. **Unit (no model, no DB):** schema round-trips; router trigger logic
   against a fake `PeerClient` (delegates on both signals, never on success,
   never on mid-decode failure); hop-count rejection; auth rejection.
2. **Integration (no GGUF, CI-safe):** two in-process FastAPI apps over a
   test HTTP transport, decode via the deterministic `GreedyMaskDecoder`
   path the edge-degradation harness already uses — the full loop without a
   model or GPU.
3. **Live eval (the falsifiable bar):** `corpus_node1.json` ingested into
   `gin_node_a` **only**; `corpus_node2.json` into `gin_node_b` **only** —
   the two corpora finally live apart. New driver
   `scripts/eval_federation.py` runs
   `data/eval/queryset_federation.yaml` (derived from the twonode queryset,
   each query labeled `a_answerable` / `b_only` / `neither`) against a live
   two-node deployment and writes
   `data/eval_runs/<ts>/federation_metrics.json` with the bar metrics from
   the top of this spec. The driver — which, unlike Node A, legitimately has
   access to both databases — performs the attribution verification of B's
   spans against B's corpus.

## Out of scope (later specs, in likely order)

1. Merkle anchor-metadata sync (load-bearing at N>2; enables peer selection)
2. \>2 nodes, peer selection, trust weights
3. gRPC/QUIC wire (swap inside `PeerClient`)
4. PKI / mTLS
5. Mid-decode delegation and answer splicing
6. Cross-node joint divergent synthesis (chunk transfer — weakens opacity,
   needs its own trust design)
7. MOCAP / DTN transports

## Documentation updates shipped with implementation

- `architecture.md` Phase 3 checklist: mark the routing loop in progress /
  done; note Merkle sync as spec #2.
- `README.md`: federation quick-start (start two nodes, run the federated
  eval) and status-table row.
- `docs/GIN_Node_Architecture_v1.md`: note that v1 transport is HTTP+JSON
  behind the `PeerClient` seam; gRPC/QUIC remains the institutional target.

## New dependencies

`fastapi`, `uvicorn`, `httpx` (client + test transport). All pure-Python
installs on Windows; no toolchain friction.
