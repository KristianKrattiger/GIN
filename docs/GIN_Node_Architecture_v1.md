# GIN Node Architecture
### Grounded Intelligence Network — Node Tier Specification v1.0
*Status: Active Design | June 2026*

---

## Preamble

GIN does not distribute a single epistemic authority — it generates authority *relationally*, through the interaction of nodes that maintain genuine independence. The plurality is not a feature bolted on. It is the mechanism. Every architectural decision in this document serves that thesis.

SEAR (Sparse Epistemically Anchored Reasoning) is the inference strategy that makes this honest by architecture rather than by instruction. Sparse means the model reasons only from what it can point to. Anchored means claims have a home in a real corpus, not a latent space. These two constraints together produce grounded synthesis — and when they fail, the failure is visible and traceable.

---

## Tier Overview

| Tier | Role | Corpus | Model Size | Primary Interface |
|------|------|---------|------------|-------------------|
| Tier 1 | Institutional anchor | Full, tiered | 14B–70B | Admin console + API |
| Tier 2 | Relay / aggregator | Partial cache | 7B–14B | API + mesh relay |
| Tier 3 | Handheld / household | Personal cache | 1B–8B quantized | Local API + upstream delegation |

---

## Tier 1 — Institutional Anchor Nodes

*Universities, archives, research consortia, libraries. Tier 1 nodes are corpus custodians and primary synthesis engines. They form the epistemic backbone of the federation.*

---

### 1.1 Corpus Layer

Tier 1 corpus sits in a four-tier storage architecture. Each tier earns its place.

#### Hot Tier — Active Retrieval
**Qdrant** or **Weaviate** (primary vector store)

- Hybrid dense + sparse search — critical for SEAR's anchor-retrieval step, which requires BM25-style sparse retrieval running *alongside* dense embeddings, not instead of them
- Supports payload filtering at retrieval time (date range, source type, trust weight)
- Qdrant preferred for self-hosted institutional deployments; Weaviate where GraphQL query interface adds value
- **Fallback**: Postgres + pgvector — viable for corpora under ~10M chunks; loses performance on complex filtered recall above that threshold

#### Warm Tier — Structured Records + Full-Text
**Postgres** (structured metadata, provenance chains, citation graphs) + **Tantivy** or **Elasticsearch** (lexical full-text search)

- Postgres holds: document metadata, anchor records with trust weights, inter-document citation edges, versioning history
- Full-text layer enables lexical matching that complements vector retrieval — SEAR's sparse anchoring benefits directly from BM25 recall on exact terms and named entities
- Citation graph edges live here as relational rows before being mirrored to the graph layer

#### Cold Tier — Immutable Archive
**Content-addressed object store** (MinIO / S3-compatible)

- Merkle-tree manifest over stored objects — any document cited as an epistemic anchor needs a tamper-evident reference
- Anything that enters the hot/warm tiers has a corresponding cold-tier hash; the hash is the ground truth for provenance disputes
- Append-only by policy — no in-place edits to archived documents

#### Graph Layer — Relational Structure
**Neo4j** or **Oxigraph** (lightweight RDF)

- Explicit edge types: `cites`, `contradicts`, `supersedes`, `translated_from`, `synthesizes`
- This is where GIN's relational half is represented — not as embeddings but as named relationships
- Divergence detection queries run here: find all anchor pairs connected by `contradicts` edges within a topic cluster
- Oxigraph preferred where RDF interoperability with external knowledge bases matters; Neo4j where query expressiveness and tooling matter more

---

### 1.2 Base Model + Inference Stack

**Base model**: mid-size open-weight (14B–70B range depending on institutional compute). Fine-tuned locally — Tier 1 nodes do not run a shared frozen foundation model.

- Diversity of base models across Tier 1 nodes is a feature, not a liability. SEAR's synthesis mechanism needs genuinely independent perspectives. The same base model with different prompts is not independence.
- LoRA/QLoRA adapter slots: the SEAR-trained reasoning layer sits on top of the base model as an adapter. Base model updates and SEAR adapter updates are decoupled.

**Inference serving**: vLLM or TGI

- vLLM preferred for throughput-optimized deployments (PagedAttention, continuous batching)
- TGI where tight HuggingFace ecosystem integration matters
- Adapter hot-swapping: both support LoRA adapter loading without full service restart

---

### 1.3 SEAR Training Loop

*Sparse Epistemically Anchored Reasoning — four-stage training process for Tier 1 nodes.*

**Stage 1 — Anchor Extraction**

From the corpus, extract `(claim, evidence_set)` pairs where evidence is sparse-retrieved — BM25 + filtered vector search, not pure top-k dense similarity. Training data must include the retrieval trace: the model learns to reason from anchors it can actually point to, which means the trace is part of the input, not hidden.

Output: labeled dataset of `(query, sparse_anchor_set, expected_output)` tuples with anchor IDs explicit in the structure.

**Stage 2 — Constrained Generation (SFT)**

Supervised fine-tuning on `(query, sparse_anchor_set, reasoning_trace, conclusion)` tuples where the reasoning trace explicitly references anchor IDs in a grammar-constrained format: `citation-before-claim` structure is enforced at the output grammar level, not just requested in the prompt.

The grammar constraint is what separates SEAR from standard RAG-with-citations. The model cannot generate a claim without having already surfaced an anchor ID for it. Hallucinated citations are structurally impossible within the constrained decoding grammar — they fail format validation, not just evaluation.

**Stage 3 — Contrastive Divergence Pairs**

The hardest stage to generate data for. Training examples where two anchors *disagree*, and the target output holds the tension rather than resolving it falsely. The model must learn to mark genuine epistemic conflict as conflict — not synthesize it away.

Data generation options:
- Human-curated divergence pairs from subject matter experts
- Cross-node validation: Tier 1 nodes generate divergence examples for each other (this is the two-node divergence demo made into a training pipeline)
- Automated extraction from corpus `contradicts` edges in the graph layer — find claim pairs connected by contradiction, surface both anchors, generate resolution-that-preserves-tension as target

**Stage 4 — RLHF / DPO Reward Pass**

Reward signal criteria:
- **Positive**: cites real, retrievable anchors; preserves marked uncertainty when anchors conflict; produces reasoning traces that are replayable (re-retrieving the cited anchors reproduces the claim)
- **Negative**: hallucinated citation IDs; false synthesis of conflicting anchors into a single position; citations that don't actually support the generated claim; overconfident conclusions where the anchor set is ambiguous

The RLHF pass is where "honest by architecture" gets stress-tested. The training signal must penalize confident resolution of genuine ambiguity — this is the behavioral commitment GIN's pluralism requires.

---

### 1.4 Transmission / Peer Connection

**Protocol**: gRPC over QUIC (primary) — low latency for synthesis requests, stream support for long reasoning traces, tolerant of variable network conditions between institutional nodes.

> **v1 implementation note (2026-07):** the shipped two-node loop speaks
> HTTP/1.1 + JSON behind the `PeerClient` seam (`gin/federation/client.py`);
> the Pydantic schema is the protocol contract. gRPC/QUIC remains the
> institutional-deployment target and replaces the transport without touching
> routing logic.

**Federation layer — three interfaces**:

1. **Query/Synthesis API**: accepts a query + optional anchor constraints, returns a SEAR-traced response with anchor IDs, confidence markers, and divergence flags
2. **Corpus-Diff Sync Endpoint**: Merkle-tree diffing of anchor sets — nodes propagate new anchors without full corpus transfer. Delta sync only.

> **v1 implementation note (2026-07):** the shipped sync endpoint is a
> 2-level, 16-bucket prefix tree (`gin/federation/anchor_tree.py`), not a
> full Merkle trie — sufficient at corpus sizes in the tens to low hundreds
> of chunks per node. A deeper tree is a later revision if bucket sizes grow
> enough that a single changed bucket still means transferring hundreds of
> leaves.

3. **Divergence Exchange**: specialized endpoint for sharing `contradicts`-type anchor pairs. Tier 1 nodes actively contribute to each other's Stage 3 training data through this channel.

**Identity and trust**:

- Mutual TLS across all node-to-node connections; federation-wide PKI
- Trust weights are explicit in the protocol — not binary allow/deny. Node A may weight Node B's corpus highly for domain X and skeptically for domain Y. This asymmetry is first-class in the data model.

> **v1 peer selection (2026-07):** when a node routes a query it ranks peers
> by content similarity only — dense (query embedding vs. each peer's synced
> centroid) and sparse (query keywords vs. each peer's distinctive IDF terms),
> RRF-fused with the same constant as hybrid retrieval. Trust weights (the
> per-domain asymmetric weighting described above) remain a separate, later
> mechanism layered on top of this similarity signal.

- Epistemic Council governance maps onto permissioned access: council decisions can modify trust weights between nodes, quarantine a node's anchor contributions, or revoke federation membership

**Sync cadence**: anchor updates propagate async — eventual consistency is appropriate. Epistemic claims don't need real-time consistency; they need traceable versioning. Every anchor has a version vector; conflict resolution on sync follows last-write-wins for metadata, human review for trust-weight changes.

---

### 1.5 Institutional Interface

**Admin Console** (web dashboard):

- Corpus curation: ingest pipelines, anchor approval/rejection queue (human-in-the-loop for what becomes a citable anchor)
- Trust weight management: view and edit per-peer trust weights by domain
- Federation peer management: add/remove peers, view sync status, inspect divergence reports
- Training pipeline monitoring: SEAR stage progress, reward signal trends, adapter version history

**Research API** (OpenAPI-spec'd REST + gRPC):

- Direct query access for institutional researchers — hits Tier 1 inference with full SEAR trace visible in response
- Integrates with Jupyter, internal search tools, institutional SSO
- Rate-limited by department/research group allocation, not globally capped

**Node Setup and Federation Enrollment**:

1. Institution provisions hardware (on-prem or institutional cloud) — minimum spec TBD by corpus size
2. Run bootstrap script: generates node keypair, registers public key with federation CA
3. Bootstrap node list provided by federation coordinator — node connects to 2–3 bootstrap peers to begin sync
4. Corpus ingest: initial bulk load to cold tier → warm tier indexing → hot tier embedding generation (this is the slow step, days to weeks for large corpora)
5. Trust weight initialization: new node starts with default weights; Epistemic Council reviews and sets final weights after a probationary observation period

---

## Tier 3 — Handheld and Household Nodes

*Phones, tablets, laptops, home servers. Tier 3 nodes are personal epistemic environments — they hold the user's annotated corpus cache, run lightweight local inference, and delegate synthesis upstream when connectivity allows.*

---

### 3.1 Corpus Layer

No local corpus of institutional scale. Tier 3 holds a **personal cache + annotation layer**.

**Local vector index**: sqlite-vec or similar embedded vector store — small footprint, no separate service, queries in milliseconds on-device

Contents:
- Cached anchors from recent Tier 1 queries — the answers the node has seen become retrievable locally
- User's personal annotation corpus: notes, highlights, Obsidian/Monolith vault content, bookmarks
- Personalized anchor trust weights: user-level overrides on top of federation defaults

**No cold tier locally.** Tier 3 nodes are clients of Tier 1's archive, not custodians of it. Provenance verification for cited anchors goes upstream.

---

### 3.2 Base Model + Inference

**Local model**: 1B–8B quantized (Q4/Q5) — Phi-3, Gemma 2, or Llama 3 variants at sizes that run on-device without dedicated GPU.

- On-device inference handles: privacy-sensitive local queries, draft reasoning, offline capability, personal corpus search
- Heavy SEAR synthesis tasks — multi-anchor cross-epistemological synthesis, divergence-held responses — get delegated upstream to Tier 1/2 when connectivity exists
- Tier 3 is a thin reasoning client + thick caching layer. The inference asymmetry is intentional.

**Quantized SEAR adapter**: a distilled, compressed version of the SEAR reasoning layer pushed down from Tier 1. Smaller, less capable than Tier 1's full adapter, but sufficient for single-anchor citation tasks and personal corpus queries.

---

### 3.3 Training / Personalization

Tier 3 nodes do not train base models. They receive **distilled SEAR adapters** pushed from Tier 1 on a cadence (weekly or on-demand).

Optional **personal fine-tuning**:
- On-device LoRA training on the user's own annotation corpus — Monolith vault, personal notes, highlighted anchors
- Personalization runs locally; personal corpus never leaves the device unless explicitly exported
- Fine-tuning delta stored as a personal LoRA; combined with the distilled SEAR adapter at inference time via adapter composition

---

### 3.4 Transmission / Connection

**Client protocol**: HTTPS/REST or QUIC to nearest Tier 1 or Tier 2 relay — mobile-friendly, tolerant of intermittent connectivity.

- Request queuing: queries made offline are queued and replayed on reconnect
- Cache-first architecture: every upstream response is cached locally; repeated queries hit cache before network
- Streaming responses: long SEAR traces stream back incrementally — UI renders as the reasoning arrives

**Homelab integration (fairlady as Tier 2 relay)**:

fairlady running on the Tailscale mesh is a natural Tier 2 relay node — more capable than a phone, always-on, Tailscale-routable from anywhere. The recommended household topology:

```
Phone / Chromebook (Tier 3 thin clients)
        │
        ▼  [Tailscale mesh]
    fairlady (Tier 2 relay)
    ├── heavier local model (7B–14B)
    ├── larger anchor cache
    └── upstream federation calls to Tier 1
        │
        ▼  [gRPC / QUIC]
    Tier 1 Federation Nodes
```

Phone and Chromebook treat fairlady as "my GIN node." fairlady handles upstream federation calls transparently. Cache hits on fairlady never go upstream. This topology reduces latency, keeps personal queries off the public network, and lets fairlady accumulate a richer cache than any single device could hold.

---

### 3.5 Household / Personal Interface

**Local API endpoint** (fairlady):

- Single endpoint all household devices point to — phone, Chromebook, ThinkPad all hit `gin.fairlady` (or Tailscale hostname) as their GIN interface
- fairlady routes: personal corpus queries handled locally, synthesis requests with anchor IDs served from cache, cache misses delegated upstream
- The Tailscale mesh handles routing — no new networking model required, just a service running on the existing infrastructure

**Personal console** (lightweight, optional):

- View personal anchor cache, annotation corpus status
- Inspect recent SEAR traces — what anchors did the model cite, were they from personal corpus or federation
- Manage personal fine-tuning runs

---

## Design Principles

These are the constraints the architecture should never violate. Every future component proposal gets tested against them.

1. **Complexity earns its place.** No layer exists for organizational reasons, tooling preference, or future-proofing. If it's here, it's load-bearing.
2. **Plurality is the mechanism, not the goal.** The architecture generates plural outputs because the nodes are genuinely independent. Avoid anything that covertly homogenizes.
3. **Honest by architecture.** SEAR's constraints make hallucination structurally difficult and failure visible. Don't add abstraction layers that obscure the trace.
4. **Minimalism as discipline.** The system's beauty is its parsimony. Resist the pull toward governance UI, visualization dashboards, and admin tooling until the core is proven.
5. **Provenance is first-class.** Every claim has a traceable anchor. Every anchor has a content-addressed home in cold storage. This is non-negotiable.

---

## First Build Target

**Two-node divergence demo.**

Two Tier 1 nodes with distinct corpora on a shared topic. One query. Both nodes return SEAR-traced responses. The synthesis layer holds the divergence rather than resolving it. Output shows both anchor sets, marks the conflict explicitly, and does not produce a false consensus.

This is the thing that demonstrates GIN's actual thesis — not "distributed RAG" but productive epistemic tension made legible. Everything else in this document is context for why this demo matters.

---

*Document lives in: Monolith / GIN / Architecture*
*Prior document: SEAR Engineering Specifications (June 2026)*
*Next: Two-Node Divergence Demo — Build Spec*
