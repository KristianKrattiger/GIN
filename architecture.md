# GIN Architecture

Technical architecture for the Grounded Information Network — what is implemented in this repository, how the pieces connect, and where the design is headed.

For node-tier deployment specs and federation protocol detail, see [docs/GIN_Node_Architecture_v1.md](docs/GIN_Node_Architecture_v1.md). Operational commands live in [README.md](README.md).

---

## Design principles

These constraints govern every component decision:

1. **Complexity earns its place** — no layer exists for organizational convenience alone.
2. **Plurality is the mechanism** — independent nodes produce genuinely different grounded outputs; avoid covert homogenization.
3. **Honest by architecture** — SEAR constraints make hallucination structurally difficult; failures stay visible.
4. **Minimalism as discipline** — prove the core before building governance UI or federation polish.
5. **Provenance is first-class** — every claim traces to a content-addressed anchor.

---

## System overview

GIN separates **what can be said** (corpus + graph), **what is admitted** (Bookkeeper), and **how answers are produced** (SEAR reasoning). This repository implements the corpus tier, SEAR constrained decoding, an automated Cartographer relation detector, the Bookkeeper admission gate, measured federation through mTLS (`gin/federation/`), and a curator labeling substrate for the `issue_frame` residue (`gin/curator/`).

The diagrams below are one concern each. Start with the overview, then open the layer you care about.

### Overview — how packages connect

Entrypoints reach federation, eval, cartographer, or curator. Answers flow eval → corpus → SEAR. Graph writes go cartographer → bookkeeper → Postgres. Curator consumes cartographer signals only; nothing imports it.

```mermaid
flowchart TB
    CLI["scripts/* entrypoints"]
    FED["gin.federation\nwire + router + mTLS"]
    EVAL["gin.eval\nNoContinuationArm"]
    CORP["gin.corpus\nretrieve + materialize"]
    SEAR["sear\nExtractiveCopyConstraint"]
    CART["gin.cartographer\npropose edges"]
    BK["gin.bookkeeper\nadmit + persist"]
    CUR["gin.curator\nlabel store"]
    PG[("Postgres warm/hot")]

    CLI --> FED
    CLI --> EVAL
    CLI --> CART
    CLI --> CUR
    FED -->|"answer_fn"| EVAL
    EVAL --> CORP
    CORP --> SEAR
    CORP --> PG
    CART -->|"EdgeProposal"| BK
    BK -->|"edges"| PG
    CUR -.->|"signals only"| CART
```

### Corpus ingest

YAML or local files land in three storage tiers. Bookkeeper-admitted edges later sit in the warm tier.

```mermaid
flowchart LR
    SRC["YAML / local corpus"] --> COLD["Cold tier\nSHA-256 blobs"]
    SRC --> WARM["Warm tier\nPostgres metadata"]
    SRC --> HOT["Hot tier\npgvector embeddings"]
    BK["Bookkeeper persist"] --> EDGES["Warm edges table"]
    EDGES --> WARM
```

### Retrieval + SEAR

Local query path from hybrid retrieve through extractive decode and post-decode gates.

```mermaid
flowchart LR
    Q[query] --> RFS["retrieve_for_synthesis"]
    RFS --> RET["retrieve RRF"]
    RFS --> EDGES["warm.fetch_edges_among"]
    RFS --> BUN["SynthesisBundle\nconvergent or divergent"]
    BUN --> MAT["materialize_from_synthesis"]
    MAT --> DEC["decode_bundle"]
    DEC --> ECC["ExtractiveCopyConstraint"]
    ECC --> SEG[segments]
    SEG --> ARM["NoContinuationArm\nrelevance gates"]
    ARM --> OUT[ArmOutput]
```

### Cartographer + Bookkeeper

Cartographer proposes; Bookkeeper alone admits and persists canonical edges.

```mermaid
flowchart TB
    CH[chunks] --> PRUNE["relatedness prune"]
    PRUNE --> PROP["CombinedRelationProposer"]
    PROP --> ESC["optional frame escalation"]
    ESC --> EP[EdgeProposal]
    PROP --> EP
    EP --> ADMIT["Bookkeeper.admit"]
    ADMIT -->|ADMITTED| GS[GraphState]
    ADMIT -->|DENIED| D[AdmissionCode]
    GS --> SYNC["persist.sync_admissions"]
    SYNC --> PG[("edges table")]
```

### Federation query

Hop-0 callers may delegate on local refusal; hop≥1 answers locally only. Peer contact is mTLS with pinned certs.

```mermaid
flowchart TB
    IN["POST /v1/federated/query"] --> GUARD["version + hop_limit"]
    GUARD --> HOP{"hop_count ge 1\nor no peers?"}
    HOP -->|yes| LOCAL["answer_fn local"]
    HOP -->|no| RTR["answer_or_delegate"]
    RTR --> LOCAL2["answer_fn local"]
    LOCAL2 -->|ok| OK[FederatedAnswer]
    LOCAL2 -->|refuse| RANK["rank_peers + filter_trusted"]
    RANK --> PC["HttpPeerClient.query\nhop=1 mTLS"]
    PC --> PEER[peer local answer]
    PEER --> OK
    PC -->|all fail| REF[NodeRefusal]
```

`POST /v1/federated/query/stream` uses the same path and emits NDJSON trace events (`retrieval_settled`, `claim_admitted`, `synthesis_complete`) before the terminal response.

### Federation control plane

Background lifespan syncs Merkle anchors and routing summaries; selection ranks peers then applies the trust gate.

```mermaid
flowchart LR
    LIFE[lifespan tasks] --> SYNC["anchor_sync.run_forever"]
    SYNC --> ROOT["GET anchors/root"]
    ROOT -->|mismatch| BUCK["GET buckets + leaves"]
    BUCK --> STORE[PeerAnchorStore]
    SYNC --> SUM["GET summary"]
    SUM --> PSS[PeerSummaryStore]
    Q[query] --> EMB["embed_query"]
    EMB --> RANK["rank_peers RRF"]
    PSS --> RANK
    RANK --> TRUST["filter_trusted"]
    TRUST --> ORDER[ordered PeerConfig]
```

### Curator labeling

Offline FastAPI UI over an append-only JSONL store. Labels do not write Bookkeeper edges.

```mermaid
flowchart TB
    SRC["CandidateSource\nlabeled-set or residue"] --> NEXT["GET /curator/next"]
    PROP["CombinedRelationProposer"] --> SIG["pair_signals"]
    SIG --> NEXT
    NEXT --> UI[curator HTML]
    UI --> LAB["POST /curator/label"]
    LAB --> ST["Store.append JSONL"]
    ST --> READY["GET /curator/readiness"]
```

The **node4** issue_frame corpus (`corpus_node4.json`, built via `scripts/build_node4.py`, gated by `scripts/verify_node4_surfacing.py`) is the contested-policy polar substrate for residue labeling. Spec: [docs/superpowers/specs/2026-07-20-issue-frame-corpus-node4-design.md](docs/superpowers/specs/2026-07-20-issue-frame-corpus-node4-design.md).

---

## Three-layer epistemic model

The full GIN design assigns distinct responsibilities so no single component can inflate its own grounding record.

| Layer | Responsibility | Writes canonical graph? | In this repo |
|-------|----------------|-------------------------|--------------|
| **Cartographer** | Proposes typed edges (`cites`, `contradicts`, `supersedes`, `translated_from`) | No | `gin/cartographer/` — relatedness gate + combined register-robust relation detector (recall 1.0 / precision 0.875 on a 13-pair labeled set); measured on its own edge precision/recall axis |
| **Bookkeeper** | Verifies anchors, enforces DAG invariants, stamps provenance | Yes (sole writer) | `gin/bookkeeper/` — uniform admission gate (confidence, endpoint/anchor integrity, dedup, DAG acyclicity), provenance stamp; sole writer of `GraphState` |
| **Reasoning (SEAR)** | Read-only synthesis with exact span attribution | No | `sear/processor.py`, `scripts/corpus_generate.py` |

The Reasoning layer may feed **proposals** back to discovery, but never writes canonical edges directly. The three layers are independently falsifiable — Cartographer on edge precision/recall (`gin/cartographer/evaluation.py`), Bookkeeper on invariant maintenance (`tests/test_bookkeeper.py`), Reasoning on SEAR grounding rate — so no layer can inflate its own record. See **[docs/nc_cartographer_design.plan.md](docs/nc_cartographer_design.plan.md)** and **[docs/nc_reasoning_robustness_noisy_edges.plan.md](docs/nc_reasoning_robustness_noisy_edges.plan.md)**.

**Curator tier (`gin/curator/`, framing-corpus spine).** The Cartographer's cheap pipeline types every same-story conflict, but the `issue_frame` residue — same issue, opposing frames, no shared story entities — is *curation-only by nature*: a model sweep found no off-the-shelf judge (7B → Opus 4.8 frontier) reproduces the curatorial stance, and recall does not rise with capability. The curator tier is the honest home for that class: a local FastAPI labeling tool over an append-only JSONL label store (`data/curator/labels.jsonl`, latest-wins fold, git-diffable, mirrors the Bookkeeper's provenance ethos). It is strictly **additive** — it consumes `gin/cartographer` (`Relation`, `CombinedRelationProposer`, `sentence_anchor`, the gold loaders) and is imported by nothing. Its labels are the shared substrate for two still-open forward layers: a bi-encoder frame detector (learns the curatorial frame as a discriminative embedding geometry, not a generative judge) and a larger-set recalibration of the cheap pipeline's thresholds. Spec: **[docs/superpowers/specs/2026-07-17-curator-ui-label-store-design.md](docs/superpowers/specs/2026-07-17-curator-ui-label-store-design.md)**. The bi-encoder is **gated on labels** — the 4 issue_frame pairs *are* the escalation eval set — so sub-project B0 (`gin/curator/residue.py` surfaces the escalation residue over `corpus_node*.json` for labeling; `gin/curator/readiness.py` reports a no-model class-count readiness gauge via `GET /curator/readiness` / `scripts/curator_readiness.py`, excluding the fixed bar pairs) makes the residue labelable and the gate measurable; B unblocks when the gauge is green.

**Frames tier (`gin/frames/`, learned frame detector — sub-project B, measured 2026-07-25).** The readiness gauge went green on 2026-07-24, so B was built: frozen `all-MiniLM-L6-v2` embeddings → order-invariant pair features `[|a-b|, a*b, (a+b)/2]` → a small scikit-learn head → a drop-in `FrameJudge`. It composes **above** both tiers (`gin/frames/` imports `gin.curator` and `gin.cartographer`; neither imports it, preserving the invariant that `gin.cartographer` never imports `gin.curator`). Runs DB-free — `default_text_index()` sources bar text from `data/synthetic/news_corpus.yaml` rather than Postgres, so the escalation bar is scorable in CI.

**Measured, both halves reported.** The stage-0 separability probe **passed** (LOO balanced accuracy 0.898 vs 0.498 random), establishing that the frozen geometry carries a recoverable stance axis. The escalation bar **failed** (`issue_frame_recall` 0.00) while leave-one-out over the 49-row leakage-free training set reached **0.715** (4-way chance 0.25). The two are consistent, and the reason is the substantive finding: **the frozen geometry separates proposition-level policy opposition but does not represent framing divergence at all.** Splitting DIVERGENT by origin, node4 policy pro/con pairs score 22/22 while the institutional-vs-grassroots framing pairs score **0/5 even in-sample** — a linear head over these features cannot memorize them. The bar is made entirely of the latter class. `direction_flip_count = 0` is solved by construction via the symmetric features, against 3–7 flips for every LLM judge.

A final-review finding is recorded rather than quietly fixed: the chunk-level leakage guard was **insufficient**, because the fixture corpus aliases bar chunks under different ids with byte-identical text (`inst_em:0` *is* `n1_doc_005:2`). Three of the bar's four issue_frame pairs were in training verbatim. `build_dataset` now guards on text as well (`bar_text_alias`), which cut the training set from 80 rows to 49. `readiness.py` counts at the chunk-id level and is subject to the same aliasing, so **the readiness gauge overstates readiness** — a correction sub-project C must carry. Spec + full results: **[docs/superpowers/specs/2026-07-24-bi-encoder-frame-detector-design.md](docs/superpowers/specs/2026-07-24-bi-encoder-frame-detector-design.md)**.

---

## Corpus tier

Each GIN node holds a **four-tier corpus stack**. This repo implements three storage tiers locally; canonical edges live in Postgres after Bookkeeper admission.

### Cold tier — immutable archive

**Module:** `gin/corpus/cold.py`  
**Storage:** `data/cold/{hash[0:2]}/{hash}` (content-addressed, append-only)

- Documents and chunks are stored as SHA-256-addressed blobs.
- `content_hash` on warm-tier records points back to cold storage.
- Enables tamper-evident provenance; federation syncs **anchor metadata** via a 16-bucket Merkle tree (`gin/federation/anchor_tree.py`, `anchor_sync.py`) — not full corpus transfer.

### Warm tier — structured records + full-text

**Module:** `gin/corpus/warm.py`  
**Storage:** Postgres tables `documents`, `chunks`, `edges`, `ingest_runs`

- Document metadata: outlet, title, source URI, optional `domain`, ingest timestamp.
- Chunk records: text, `head_sentence`, `eval_layer`, `eval_tag`, `content_hash`.
- Epistemic edges between chunks with typed relationships (Bookkeeper is the sole writer of admitted edges).
- Generated `tsvector` column on `chunks.text` for BM25-style sparse retrieval.

### Hot tier — dense embeddings

**Module:** `gin/corpus/hot.py`  
**Storage:** `chunks.embedding vector(384)` with HNSW index

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings computed at ingest time; query embedding at retrieval time.
- Cosine distance search via pgvector.

### Graph layer

Canonical graph state is `GraphState` in `gin/bookkeeper/`, persisted to Postgres `edges` via `persist.sync_admissions`. A dedicated graph DB (Neo4j or Oxigraph) remains a future option for cross-corpus divergence queries; the PoC uses relational edge rows.

---

## Retrieval and synthesis

**Module:** `gin/corpus/retrieve.py`

### Hybrid retrieval

Two parallel searches merge via **Reciprocal Rank Fusion** (k=60):

1. **Dense** — pgvector cosine distance on chunk embeddings.
2. **Sparse** — `plainto_tsquery` + `ts_rank` on generated `tsvector`.

Optional filters (e.g. `eval_layer: realism`) apply to both legs.

### Synthesis bundling

`retrieve_for_synthesis()` expands seed hits into a `SynthesisBundle`:

1. Retrieve top-`k_seed` chunks (hybrid RRF).
2. **Re-rank seeds** by max per-sentence query keyword score; drop zero-relevance noise.
3. Fetch `contradicts` and `cites` edges among seeds.
4. Pull neighbor chunks linked by those edges.
5. Classify mode:
   - **divergent** — when a `contradicts` pair is **query-relevant on both sides**. The production gate is **IDF-weighted** (`idf_weighted_relevance` ≥ `DIVERGENCE_IDF_FLOOR` = 0.13): one *distinctive* shared word (e.g. `wildfire`) suffices, a generic one (e.g. `district`) does not — this is what lets real reframing pairs that share almost no vocabulary still reach divergent mode. Without corpus IDF (unit tests) it falls back to the lexical gate (`DIVERGENCE_RELEVANCE_FLOOR` + `matched_keyword_count` ≥ 2 for queries with ≥3 keywords). Close RRF competitors without a contradicts edge are corroboration, not divergence.
   - **convergent** — otherwise (including bureau + independent survey agreeing on the same statistic).
6. Build pairs (query-irrelevant contradicts pairs excluded); boost RRF scores for paired chunks; cap at `k_max`.

Legacy path: when `query` is empty, any contradicts edge or close multi-outlet competitors still force divergent mode.

**Module:** `gin/corpus/materialize.py` orders pair-adjacent hits, builds a SEAR `Corpus`, and computes `required_doc_groups` — frozensets of doc indices that must both be quoted when a `contradicts` edge links them.

**Module:** `gin/corpus/generate.py` — convergent decode steers to top query-relevant doc; `competing_same_tag` path emits one AMBIGUOUS corroboration span across bureau + survey outlets. See [ENG 02](docs/GIN_ENG_02_Eval_Baseline_v1.md) and `docs/nc_phase3_divergence_correctness.plan.md`.

**Module:** `gin/corpus/prompts.py` assembles a metadata-only source manifest (chunk bodies live in the SEAR corpus, not the prompt) plus mode-specific task instructions.

---

## SEAR constrained decoding

**Modules:** `sear/corpus.py`, `sear/processor.py`, `sear/connectives.py`

### Corpus as grammar

`Corpus` indexes each document as a token-id sequence and builds `start_index: token_id → [(doc_id, position), ...]`. The logits processor consults this index at every decode step; spans copy token-by-token from real corpus positions, sidestepping cross-document BPE boundary issues for the baseline.

### Cursor finite-state machine

`ExtractiveCopyConstraint` tracks three modes:

| Mode | Behavior |
|------|----------|
| `BOUNDARY` | May start a new extractive span, emit EOS, or (after first closed span) emit connectives / cite markers |
| `IN_SPAN` | Advance cursors; legal tokens = union of `continuation(doc, pos)` across live cursors |
| `IN_CONNECTIVE` | Walk multi-token connective phrases (`but`, `whereas`, `on the other hand`, …) |

On each step, all non-allowed vocabulary positions are masked to `-1e30`.

### Span lifecycle

1. **Start** — token must appear in `start_index` at an unused position.
2. **Continue** — cursors fan out across documents sharing the prefix; prune when tokens diverge.
3. **Close** — at `min_span_len`, allow `|` delimiter or EOS; record `Segment` with `(doc_id, start, end)` sources.
4. **Divergence auto-close** — when `close_on_doc_divergence=True`, close when cursors narrow **at the current token** (not "ever since span start") and only if `_span_close_permitted()` (min length + sentence-end when configured).

### Connectives and citations

- **Connectives** — tokenizer-aligned phrase inventory stratified into contrastive, additive, and concessive categories; gated by edge type at materialisation time; only available after at least one closed extractive span.
- **Cite markers** — `[1]`, `[2]`, … mapped to doc indices; emitted after spans to attribute bracket-style references.
- **Used positions** — `(doc_id, position)` tuples are consumed after span close to prevent verbatim reuse of the same source span.

### Attribution render

`constraint.render(detok)` produces human-readable output:

```
"Emergency services confirmed 142 people received treatment"[1]  <- EXACT: CentralWire[12:24] [steered]
  |  however
"Emergency services confirmed 98 people received treatment"[2]  <- EXACT: MetroDaily[12:24] [steered]
```

`AMBIGUOUS` tags spans whose cursors survived from multiple documents at close time. `[steered]` / `[divergence-steered]` tags indicate the span start was guided by a retrieval heuristic rather than freely chosen by the model; no tag means free selection.

---

## Layered provenance record

The "honest by architecture" property of SEAR applies precisely to the generation layer: given the retrieved corpus, output is verbatim extraction with exact span attribution. Three additional layers influence the output and each now contributes to a structured provenance record emitted alongside the attribution.

### Retrieval manifest

**Module:** `gin/corpus/retrieval_manifest.py`
**Storage:** `data/retrieval_manifests/{hash[0:2]}/{hash}.json` (content-addressed)

At synthesis time, `build_retrieval_manifest(query, bundle)` serialises the query string, its SHA-256 hash, synthesis mode, edge types present, and per-chunk retrieval ranks and RRF scores into a `RetrievalManifest`. The manifest is content-hashed — identical retrieval events produce the same hash and are stored once. `materialize_from_synthesis()` returns the manifest as a fourth value and stores `manifest_hash` on `SynthesisContext`.

A **retrieval confidence floor** (`RETRIEVAL_CONFIDENCE_FLOOR = 0.010`) causes `retrieve_for_synthesis()` to raise `RetrievalConfidenceError` when the top RRF score falls below the absolute threshold, analogous to the zero-cursor grounding failure signal in SEAR. This prevents low-confidence retrieval from silently producing attributed-but-misleading output.

### Edge-type-gated connectives

**Module:** `sear/connectives.py`

The connective inventory is stratified into `CONTRASTIVE_PHRASES`, `ADDITIVE_PHRASES`, and `CONCESSIVE_PHRASES`. `phrases_for_edge_types(edge_type_set)` selects the appropriate subset based on the epistemic relationships present in `bundle.edges`: a `contradicts` edge restricts the model to contrastive connectives; a `cites` edge admits additive and concessive phrases. The selected phrase set is built into the connective inventory at materialisation time and stored on `SynthesisContext` alongside `active_edge_types`. Connective framing is now constrained by the typed graph rather than free model choice.

### Steering guidance tags

**Module:** `sear/processor.py`

`Segment` carries a `guidance` field (`""` free, `"steered"`, `"divergence-steered"`). In `_begin_span`, the opening cursor position is checked against `preferred_starts` and `divergence_starts`; the guidance string is recorded and attached to the closed segment. `render()` appends `[steered]` or `[divergence-steered]` to the attribution line, making the distinction between a model-chosen span and a heuristic-steered span visible in the output.

### Synthesis manifest

**Module:** `gin/corpus/synthesis_manifest.py`

`render_synthesis_manifest(query, ctx, segments, render_output, retrieval_manifest=...)` assembles a single structured text record covering all four layers:

```
=== Synthesis Manifest ===

--- Retrieval ---   manifest hash, mode, per-chunk ranks, confidence floor verdict
--- Graph ---       active edges, required doc groups, groups-satisfied check
--- Steering ---    connective mode (with edge-type derivation), preferred/forbidden starts
--- Generation ---  span count, free vs steered vs divergence-steered breakdown
--- Attribution --- verbatim render() output with [steered] tags
```

`scripts/corpus_generate.py` prints this record after every generation run.

---

## End-to-end data flow

```
YAML / ingest
    → cold.store(bytes)           # content hash
    → warm.upsert_document/chunk  # metadata + tsv
    → hot.embed_and_store         # vector(384)

Query (local)
    → retrieve_for_synthesis
    → materialize_from_synthesis  # Corpus + SynthesisContext
    → build_synthesis_prompt      # manifest + task
    → ExtractiveCopyConstraint    # LogitsProcessor on llama.cpp
    → finalize() + render()       # attribution record

Query (federated)
    → POST /v1/federated/query[/stream]
    → answer_or_delegate → NoContinuationArm (local or peer via mTLS)
```

**Entry points:** `scripts/corpus_generate.py` (local), `scripts/node_serve.py` (federated).

---

## Database schema

Defined in `docker/init-db.sql`:

```
documents
  doc_id UUID PK
  content_hash TEXT UNIQUE
  outlet, title, source_uri, source_type, domain, ingested_at

chunks
  chunk_id TEXT PK
  doc_id → documents
  text, head_sentence, eval_layer, eval_tag, content_hash
  embedding vector(384)
  tsv tsvector GENERATED

edges
  src_chunk_id, dst_chunk_id → chunks
  edge_type, note
  UNIQUE (src, dst, type)

ingest_runs
  run_id, status, stats_json, timestamps

peer_anchors
  peer_node_id, chunk_id, content_hash, outlet, title, bucket_index
  (cached peer Merkle leaves — never peer chunk text)

peer_summaries
  peer_node_id PK
  centroid vector, top_terms JSONB, domains JSONB, updated_at
  (routing summaries for N>2 peer selection)
```

Indexes: GIN on `tsv`, HNSW on `embedding`, B-tree on `eval_layer` and `doc_id`; peer-anchor index on `(peer_node_id, bucket_index)`.

---

## Module map

```
GIN/
├── gin/
│   ├── corpus/
│   │   ├── cold.py                # SHA-256 blob store
│   │   ├── warm.py                # Postgres CRUD, edges, ingest runs, domain
│   │   ├── hot.py                 # SentenceTransformer embeddings
│   │   ├── db.py                  # Connection helpers, env config
│   │   ├── models.py              # ChunkHit, SynthesisBundle, SynthesisContext, …
│   │   ├── ingest.py              # YAML → tiered ingest pipeline
│   │   ├── corpus_manager.py      # Local JSONL/txt ingest + immutable store
│   │   ├── manifest.py            # Versioned snapshot manifests
│   │   ├── retrieve.py            # Hybrid search + synthesis bundling
│   │   ├── materialize.py         # ChunkHit[] → sear.Corpus + SynthesisContext
│   │   ├── generate.py            # decode_bundle / generate_no_continuation
│   │   ├── divergence.py          # Divergence zone + forbidden-start computation
│   │   ├── relevance.py           # Query-sentence + IDF-weighted match scoring
│   │   ├── prompts.py             # Synthesis prompt templates
│   │   ├── retrieval_manifest.py  # Content-addressed retrieval event record
│   │   ├── synthesis_manifest.py  # Human-readable layered provenance render
│   │   └── trace_events.py        # Corpus-tier stream event primitives
│   ├── cartographer/              # Relatedness gate + relation proposers + scan
│   ├── bookkeeper/                # Admission gate, GraphState, persist to edges
│   ├── federation/                # Wire schema, router, mTLS client, anchors, trust
│   ├── curator/                   # Label store, residue source, node4 build/verify
│   └── eval/                      # RAG vs NoContinuation arms, metrics, runners
├── sear/
│   ├── corpus.py                  # Token-indexed document store
│   ├── processor.py               # ExtractiveCopyConstraint FSM + Segment.guidance
│   ├── connectives.py             # Stratified connective inventory
│   └── bias.py                    # BiasedGINLogitsProcessor
├── scripts/                       # CLI: ingest, generate, node_serve, curator_*, eval_*
├── config/                        # Per-node YAML (node_a/b/c, trust-gated variants)
├── tests/
├── data/
│   ├── synthetic/                 # YAML eval corpus
│   ├── cold/                      # Content-addressed blobs (gitignored content)
│   ├── curator/                   # labels.jsonl, node4_sources.yaml
│   ├── retrieval_manifests/
│   └── eval_runs/
└── docker/
    ├── docker-compose.yml         # pgvector Postgres
    └── init-db.sql                # Schema bootstrap
```

---

## Target deployment topology

Aspirational Tier 1/2/3 layout. gRPC/QUIC remains deferred; the measured PoC uses HTTP+JSON over mTLS on localhost.

```mermaid
flowchart TB
    T3A["Tier 3\nPhone / laptop\n1B-8B quantized"]
    T3B["Tier 3\nChromebook"]
    T2["Tier 2 relay\nfairlady\n7B-14B + anchor cache"]
    T1A["Tier 1\nUniversity / archive\n14B-70B + full corpus"]
    T1B["Tier 1\nResearch consortium"]

    T3A --> T2
    T3B --> T2
    T2 -->|"gRPC / QUIC deferred\nMerkle diff sync"| T1A
    T2 --> T1B
    T1A <-->|"divergence exchange"| T1B
```

| Tier | Corpus | Inference | Federation role |
|------|--------|-----------|-----------------|
| **1** | Full hot/warm/cold/graph | Large local model + SEAR adapter | Anchor custodian; publishes diffs |
| **2** | Partial cache | Medium model on always-on relay | Cache-first routing; upstream delegation |
| **3** | Personal cache + annotations | Small quantized model | Offline-first client; queues on disconnect |

### Measured localhost topology (Phase 3)

Three sovereign processes (`config/node_a.yaml`, `node_b.yaml`, `node_c.yaml`), each with its own Postgres DB, cold store, GGUF model, and self-signed cert. Peers pin each other's `cert.pem`. Background tasks sync Merkle anchors and routing summaries; queries use `POST /v1/federated/query` over mTLS.

| Node | Corpus role | Typical peer use |
|------|-------------|------------------|
| A | Split / primary eval corpus | Delegates when local grounding fails |
| B | Complementary split | Answers hop=1 relays |
| C | Monetary-policy corpus | N=3 selection target |

Federation syncs **anchor metadata** via Merkle-tree diffing — not full corpus transfer. Each node retains sovereignty over its own cold archive.

---

## Build phases

### Phase 1 — Dense baseline

Prove SEAR behavior on stock Mistral with grammar-constrained extractive synthesis:

- ✅ `ExtractiveCopyConstraint` with cursor fan-out, pruning, connectives, cites
- ✅ Corpus tier PoC (Postgres + pgvector + cold store)
- ✅ Hybrid retrieval and query-aware divergent/convergent synthesis modes
- ✅ Live generation via `llama-cpp-python`
- ✅ Layered provenance record (retrieval manifest, guidance tags, synthesis manifest)
- ✅ Edge-type-gated connective vocabulary
- ✅ Retrieval confidence floor (`RetrievalConfidenceError`)
- ✅ SEAR vs RAG eval harness (`scripts/eval_run.py`, `gin/eval/`)
- ✅ Preliminary eval baseline (structural prevention + NC epistemic targets) — see [docs/GIN_ENG_02_Eval_Baseline_v1.md](docs/GIN_ENG_02_Eval_Baseline_v1.md)

**Validation:** `pytest`, `python scripts/sear_phase1.py --selftest`

### Phase 2 — Bookkeeper separation

- ✅ Canonical graph admission gate (`gin/bookkeeper/`) — confidence, endpoint/anchor integrity, dedup, DAG acyclicity, provenance stamp
- ✅ Cartographer proposals vs verified edges (`gin/cartographer/` → `Bookkeeper.admit` → `persist.sync_admissions`)
- ✅ Reasoning layer strictly read-only on graph state (SEAR / retrieval consume edges; never write them)
- ✅ Scan production validation + escalation frame-judge closure (issue_frame class is curation-only; see curator tier)

### Phase 3 — Federation

- ✅ Two-node divergence demo (inter-corpus, same machinery) — measured on **real fetched text** (`20260705T043114Z`, `divergence_fidelity` 1.0), generalized across three framing registers and confirmed model-independent on Qwen2.5-7B. See [docs/nc_real_text_divergence_generalization.plan.md](docs/nc_real_text_divergence_generalization.plan.md). This established the divergence *signal* across two corpora; the transport that followed is measured below.
- ✅ Sovereign delegation loop (zero-cursor routing v1) — two node processes,
  HTTP+JSON schema-first transport behind the `PeerClient` seam
  (`gin/federation/`); pre-commitment grounding failures delegate to the
  configured peer; B's answer relays with attribution intact and explicitly
  marked as B's. Measured: `data/eval_runs/20260714T175645Z/federation_metrics.json`.
  Spec: docs/superpowers/specs/2026-07-13-federation-v1-sovereign-delegation-design.md
- ✅ Merkle diff sync of anchor metadata (spec #2) — 16-bucket Merkle tree
  over (chunk_id, content_hash, outlet, title); background asyncio loop per
  node pulls its peer's root, drills into mismatched buckets only. Measured:
  `data/eval_runs/20260715T073932Z/anchor_sync_metrics.json` (0 diff vs. peer ground
  truth; no-op cycle O(1) bytes; single-chunk-change cycle « full corpus).
  Not load-bearing at N=2 (built as the primitive for N>2 peer selection).
  Spec: docs/superpowers/specs/2026-07-14-merkle-anchor-sync-design.md
- ✅ Peer selection at N>2 (spec #3) — third node added (monetary-policy
  corpus); a node that can't ground locally ranks its peers by dense+sparse
  RRF fusion (reusing the retrieval stack's `RRF_K`) over routing summaries
  synced alongside anchors — one background sync loop per peer, each refetching
  a peer's summary until it is cached — and delegates to the best-matching peer
  first, falling back through the ranked list (every hop still `hop_count=1`).
  Measured on three real Mistral-7B nodes (one CPU-only):
  `data/eval_runs/20260715T192750Z/peer_selection_metrics.json` — selection
  precision@1 1.0, avg peers tried 1.0, routing FP 0, fabrication 0.0,
  attribution 1.0, honest refusal 1.0.
  Spec: docs/superpowers/specs/2026-07-15-peer-selection-n3-design.md
- ✅ Trust weights (per-domain peer gating, spec #4) — a node excludes a
  peer from ranked delegation candidates when a configured
  `(peer, domain)` trust weight falls below threshold; domain coverage
  synced automatically alongside the routing summary, gating applied after
  RRF ranking and before delegation. Measured on the live 3-node
  deployment: gated run `data/eval_runs/20260716T004321Z/peer_selection_metrics.json`
  (gated_peer_contacted 0, honest_refusal_rate 1.0 for the affected
  queries); ungated regression run
  `data/eval_runs/20260716T004515Z/peer_selection_metrics.json` reproduces
  sub-project 3's exact bar (precision@1 1.0, avg peers tried 1.0).
  Spec: docs/superpowers/specs/2026-07-15-trust-weights-design.md
- ✅ mTLS: self-signed pinned peer certificates (spec #5) — replaces the
  federation shared-secret bearer scheme (`shared_secret`/`GIN_FED_SECRET`,
  fully removed) with mutual TLS. Each node self-signs its own identity
  cert (`scripts/node_keygen.py`); peer operators exchange `cert.pem`
  out-of-band and pin it (`peers[].pinned_cert_path`) — an unpinned cert
  fails the TLS handshake and never reaches the ASGI app. Measured on the
  live 3-node deployment over real TLS handshakes, not a simulation:
  ungated run
  `data/eval_runs/20260716T095519Z/peer_selection_metrics.json` reproduces
  sub-project 3's exact bar (precision@1 1.0, avg peers tried 1.0, routing
  FP 0, fabrication 0.0, attribution 1.0, honest refusal 1.0); gated run
  `data/eval_runs/20260716T095704Z/peer_selection_metrics.json` reproduces
  sub-project 4's exact bar (gated_peer_contacted 0, same figures
  otherwise).
  Spec: docs/superpowers/specs/2026-07-16-federation-mtls-design.md
- ✅ Streaming reasoning trace: NDJSON incremental claim events over the
  existing mTLS stack — reframes (does not build) the gRPC/QUIC line;
  see docs/superpowers/specs/2026-07-16-streaming-reasoning-trace-design.md
  for why gRPC/QUIC itself remains deferred.

### Phase 4 — SEAR training loop (Tier 1)

Four-stage training per [docs/GIN_Node_Architecture_v1.md](docs/GIN_Node_Architecture_v1.md):

1. Anchor extraction with retrieval traces
2. Grammar-constrained SFT (`citation-before-claim`)
3. Contrastive divergence pairs
4. RLHF/DPO reward pass penalizing false synthesis of conflict

---

## Extension points

| Hook | Location | Purpose |
|------|----------|---------|
| Embedding model | `gin/corpus/hot.py` `EMBEDDING_MODEL` | Swap retriever for domain-specific encoders |
| Connective phrases | `sear/connectives.py` `CONTRASTIVE_PHRASES` / `ADDITIVE_PHRASES` / `CONCESSIVE_PHRASES` | Extend per-category connective vocabulary |
| Connective gating | `sear/connectives.py` `phrases_for_edge_types()` | Map new edge types to connective categories |
| Ambiguity threshold | `gin/corpus/retrieve.py` `AMBIGUITY_SCORE_DELTA` | Tune divergent mode sensitivity (legacy no-query path) |
| Divergence IDF floor | `gin/corpus/retrieve.py` `DIVERGENCE_IDF_FLOOR` | Min IDF-weighted query relevance for a contradicts side to count as divergence |
| Retrieval confidence floor | `gin/corpus/retrieve.py` `RETRIEVAL_CONFIDENCE_FLOOR` | Minimum absolute RRF score before synthesis is declined |
| `min_span_len` | `ExtractiveCopyConstraint` constructor | Minimum tokens before span close |
| Eval layers | `gin/corpus/models.py` `EvalLayer` | Filter retrieval / ingest by corpus slice |
| Zero-cursor / grounding failure | `sear/processor.py` → `gin/federation/router.py` | Federation routing via `answer_or_delegate` (implemented) |
| Trust gate threshold | `gin/federation/config.py` `trust_gate_threshold` | Drop peers whose known domain weight is below threshold |
| Retrieval manifest storage | `gin/corpus/retrieval_manifest.py` `retrieval_manifests_dir()` | Override manifest storage root |

---

## Related documents

| Document | Focus |
|----------|-------|
| [README.md](README.md) | Ops commands, eval status table, curator / federation how-to |
| [docs/GIN_Node_Architecture_v1.md](docs/GIN_Node_Architecture_v1.md) | Tier 1/2/3 specs, SEAR training loop, federation protocol |
| [docs/GIN_ENG_00_Engineering_Register.md](docs/GIN_ENG_00_Engineering_Register.md) | Measured vs quarantined engineering ledger |
| [docs/GIN_ENG_01_SEAR_PoC_Spec.md](docs/GIN_ENG_01_SEAR_PoC_Spec.md) | SEAR proof-of-concept engineering spec |
| [docs/GIN_The_Whole_Frame.md](docs/GIN_The_Whole_Frame.md) | Program-level synthesis and roadmap honesty |
| [docs/GIN_02_Productive_Divergence.md](docs/GIN_02_Productive_Divergence.md) | Divergence-as-feature thesis |
