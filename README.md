# GIN — Grounded Intelligence Network

GIN is a federation of independent, place-rooted reasoning nodes that ground every claim in a traceable corpus, hold their disagreements legible instead of dissolving them, and arrive at convergence — when they do — relationally rather than by central decree.

Attribution in GIN is **exact by construction**: enforced at decode time via cursor-based copy constraints, not inferred post-hoc from attention weights. Édouard Glissant's *right to opacity* — the refusal to treat total transparency as a condition of participation — is the philosophical charter: nodes relate across their differences without being required to expose or homogenize their corpora.

For system design, data flow, and module-level detail, see **[architecture.md](architecture.md)**. Broader program documents live in [`docs/`](docs/).

---

## What this repository contains

This repo is the **Phase 1 engineering scaffold**: a working corpus tier, hybrid retrieval, and SEAR constrained decoding on stock Mistral via `llama-cpp-python`. It proves extractive synthesis with span-level attribution before federation, graph admission, or attention surgery.

| Package | Role |
|---------|------|
| [`sear/`](sear/) | SEAR inference core — token-indexed `Corpus`, `ExtractiveCopyConstraint` logits processor, stratified connective inventory, `BiasedGINLogitsProcessor` |
| [`gin/corpus/`](gin/corpus/) | Corpus tier — cold/warm/hot storage, ingest, hybrid retrieval, synthesis bundling, layered provenance records |
| [`gin/eval/`](gin/eval/) | Eval harness — RAG vs SEAR arms, overlap/NLI verifiers, metrics, `data/eval_runs/` reports |
| [`scripts/`](scripts/) | CLI entry points for ingest, query, materialization, and live generation |
| [`tests/`](tests/) | Unit tests for processor, retrieval, prompts, materialization, provenance |
| [`data/synthetic/`](data/synthetic/) | YAML eval corpus (controlled divergence, counterfactual probes) |
| [`data/eval/`](data/eval/) | Shared query set (`queryset.yaml`) for the designed experiment |
| [`data/eval_runs/`](data/eval_runs/) | Timestamped eval reports, per-query JSON, retrieval recall artifacts |
| [`docker/`](docker/) | Postgres + pgvector for local development |

---

## SEAR — Sparse Epistemically Anchored Reasoning

SEAR is the inference discipline that makes GIN honest by architecture rather than by instruction.

- **Sparse** — the model may only emit token spans that occur verbatim in the retrieved corpus.
- **Anchored** — each emitted span carries a pointer `(doc_id, start_pos, end_pos)` back to source positions.
- **Cursor tracking** — live cursors `(doc_id, position)` track which source spans remain consistent with what has been emitted; the legal next token is the union of tokens at `position + 1` across all live cursors.
- **Zero cursors** — a first-class grounding-failure signal: the corpus cannot support the current continuation. This triggers graceful termination or (in the full design) federation routing to a peer node.

In **divergent** synthesis mode, when sources disagree, the constraint can auto-close spans when cursors diverge across documents and require citation of both sides of a `contradicts` edge.

---

## Quick start

### Prerequisites

- Python 3.11+
- Docker (for Postgres + pgvector)
- Optional: a local GGUF model (e.g. `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf`) for live generation

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database

```bash
cd docker
docker compose up -d
```

Default connection: `postgresql://gin:gin@localhost:5432/gin`  
Override with `GIN_DATABASE_URL` in a `.env` file at the repo root.

### 3. Ingest the synthetic corpus

```bash
python scripts/corpus_ingest.py --source data/synthetic
```

This writes content-addressed blobs to `data/cold/`, metadata and edges to Postgres, and dense embeddings (MiniLM-L6-v2, 384-dim) into pgvector.

### 3b. Ingest local corpus files with manifest snapshots

```bash
python scripts/ingest.py --input-dir data/local_corpus --format auto
```

This writes immutable blobs to `data/corpus_store/` and publishes a versioned manifest under `data/manifests/manifest_v{N}.json`.

### 4. Run tests (no model required)

```bash
pytest
```

SEAR cursor logic self-test:

```bash
python scripts/sear_phase1.py --selftest
```

### 5. Query the corpus

```bash
python scripts/corpus_query.py "downtown incident hospital treatment" -k 5
```

### 6. Live constrained generation

```bash
python scripts/corpus_generate.py \
  --model /path/to/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf \
  --query "downtown incident hospital treatment"
```

Output includes raw generated text and an **attribution record** with `EXACT` vs `AMBIGUOUS` span tags and source positions.

### 7. RAG vs SEAR eval (requires model + Postgres)

Batch the shared query set through both generation arms and write a comparison report:

```bash
python scripts/eval_run.py \
  --model /path/to/Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  --arms rag,no_continuation \
  --verifier overlap \
  --threshold 0.5
```

Use `--verifier nli` for entailment scoring (downloads `cross-encoder/nli-deberta-v3-xsmall` on first use). Reports land in `data/eval_runs/<timestamp>/` (`report.md`, `metrics.json`, `results/*.json`, `retrieval/*.json`).

See **[docs/GIN_ENG_02_Eval_Baseline_v1.md](docs/GIN_ENG_02_Eval_Baseline_v1.md)** for measured results and interpretation.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GIN_DATABASE_URL` | `postgresql://gin:gin@localhost:5432/gin` | Postgres connection string |
| `GIN_COLD_PATH` | `data/cold` | Content-addressed blob store root |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/sear_phase1.py` | SEAR baseline — `--selftest` (no model) or `--model` for two-doc Mistral demo |
| `scripts/corpus_ingest.py` | Load YAML corpus into cold/warm/hot tiers |
| `scripts/ingest.py` | Ingest local JSONL/txt into immutable store + versioned manifests |
| `scripts/corpus_query.py` | Hybrid dense + sparse retrieval (RRF merge) |
| `scripts/corpus_to_sear.py` | Materialize retrieved chunks into a SEAR `Corpus` (debug); supports `--manifest-version` with `--all` |
| `scripts/corpus_generate.py` | End-to-end: retrieve → prompt → constrained llama.cpp generation |
| `scripts/eval_run.py` | RAG vs SEAR eval harness — batch query set, overlap or NLI verifier, write `data/eval_runs/` report |
| `scripts/corpus_wipe.py` | Reset warm-tier data (development) |

---

## Synthetic corpus format

YAML files under `data/synthetic/` define documents and epistemic edges:

```yaml
documents:
  - id: incident_centralwire
    outlet: CentralWire
    title: Downtown incident response
    eval_layer: realism
    chunks:
      - |
        RIVERPORT — Officials responded to a downtown incident...

edges:
  - src: incident_centralwire:0
    dst: incident_metrodaily:0
    type: contradicts
    note: conflicting hospital and arrest counts
```

`eval_layer` values: `realism`, `counterfactual`, `out_of_scope`, `convergent`.  
Edge types: `cites`, `contradicts`, `supersedes`, `translated_from`.

---

## Manifest version handoff to cursor resolver

The cursor resolver must read a manifest snapshot first, then load document text by `content_hash`. It should not scan storage directly.

1. Run ingestion to create a new snapshot:

```bash
python scripts/ingest.py --input-dir data/local_corpus --format auto
```

2. Pick a manifest version (for reproducibility or rollback), then pass it when materializing resolver input:

```bash
python scripts/corpus_to_sear.py --all --manifest-version 3
```

`gin.corpus.materialize.materialize_all(..., manifest_version=3)` will hydrate the SEAR corpus from `data/manifests/manifest_v3.json`. Use an older version to roll back while keeping immutable blobs and prior manifests queryable.

---

## Architecture at a glance

SEAR operates across three layers (full design; partial implementation in this repo):

1. **Cartographer** — proposes typed epistemic edges; does not write canonical graph state.
2. **Bookkeeper** — sole admission gate for verified graph edges, anchor integrity, DAG invariants.
3. **Reasoning** — read-only consumer of the verified graph; produces grounded answers via SEAR.

Every synthesis event produces a **layered provenance record** covering four layers: retrieval (content-addressed manifest with per-chunk RRF scores), graph (active edges and required-quote groups), steering (connective mode derived from edge types, preferred/forbidden starts), and generation (verbatim span attribution with free-vs-steered tags). See **[architecture.md — Layered provenance record](architecture.md#layered-provenance-record)**.

**Node topology** (target deployment):

| Tier | Role | Model | Corpus |
|------|------|-------|--------|
| 1 | Institutional anchor | 14B–70B | Full four-tier stack |
| 2 | Relay (e.g. fairlady) | 7B–14B | Partial cache on Tailscale mesh |
| 3 | Handheld / household | 1B–8B quantized | Personal cache, offline-first |

Federation propagates **anchored diffs**, not full corpora — Merkle-tree metadata sync preserves node sovereignty.

See **[architecture.md](architecture.md)** for data-flow diagrams, module map, and schema detail.

---

## Eval baseline

Measured **RAG vs No-Continuation** on the synthetic corpus ([ENG 02](docs/GIN_ENG_02_Eval_Baseline_v1.md)).

| Run | Queries | Role |
|-----|---------|------|
| `20260701T192827Z` | 9 | Structural prevention (overlap) |
| `20260702T012203Z` | 20 | **NC epistemic promotion** (overlap) |
| `20260702T010918Z` | 9 | Regression anchors post–Phase 3 |

**No-Continuation (`20260702T012203Z`, full 20):**

| Metric | NC | Target |
|--------|-----|--------|
| Fabrication rate | **0.000** | 0.000 |
| Query relevance rate | **1.000** | ≥ 0.90 |
| Supported irrelevance rate | **0.000** | ≤ 0.05 |
| Gold chunk coverage | **1.000** | ≥ 0.75 |
| Counterfactual adherence | **1.000** | ≥ 0.90 |
| Divergence fidelity | **1.000** | preserve |

**What this proves:** decode-time prevention (fabrication 0) *and* query-relevant extractive selection on the expanded queryset — via query-aware mode gating (Phases 2–3), convergent steering (Phase 1), and corroboration decode for bureau+survey counterfactuals. RAG still fabricates on overlap (0.238 on full 20) but answers counterfactuals via paraphrase.

NLI confirms NC fabrication 0 on the 9-query structural baseline (`194024Z`); expanded-set NLI re-run outstanding. Details: [ENG 02](docs/GIN_ENG_02_Eval_Baseline_v1.md).

---

| Item | Status |
|------|--------|
| SEAR Phase 1 scaffold (cursor logic, masking, attribution render) | ✅ |
| Corpus tier (cold / warm / hot in Postgres) | ✅ |
| Corpus Manager ingestion (local JSONL/txt, immutable store) | ✅ |
| Versioned manifest snapshots for resolver input | ✅ |
| Manifest-version materialization path (`--manifest-version`) | ✅ |
| Hybrid retrieval + synthesis bundling (convergent / divergent) | ✅ |
| Live Mistral integration via llama-cpp-python | ✅ |
| Synthetic corpus with controlled divergence | ✅ |
| Stratified connective vocabulary (edge-type-gated) | ✅ |
| Steering guidance tags on attribution record | ✅ |
| Content-addressed retrieval manifests | ✅ |
| Retrieval confidence floor (`RetrievalConfidenceError`) | ✅ |
| Synthesis manifest — layered provenance record | ✅ |
| SEAR vs RAG eval harness (`scripts/eval_run.py`, `gin/eval/`) | ✅ |
| Eval baseline — structural prevention + NC epistemic targets (overlap) | ✅ ([ENG 02](docs/GIN_ENG_02_Eval_Baseline_v1.md), run `20260702T012203Z`) |
| Query-relevance / epistemic metrics on expanded query set | ✅ |
| Two-node divergence demo (inter-corpus) | 🔲 |
| Bookkeeper + reasoning layer separation (Phase 2) | 🔲 |
| Federation routing with sync metadata (Phase 3) | 🔲 |

---

## Stack

- **Python** — application code
- **Postgres + pgvector** — warm metadata, full-text (`tsvector`), dense embeddings
- **sentence-transformers** — `all-MiniLM-L6-v2` query/document embeddings
- **llama-cpp-python** — local GGUF inference with custom `LogitsProcessor`
- **Mistral-7B-Instruct-v0.3** (Q4_K_M GGUF) — reference model for Phase 1
- **Tailscale** — mesh networking for Tier 2 relay topology (deployment target)

---

## Documentation

| Document | Contents |
|----------|----------|
| [architecture.md](architecture.md) | Technical architecture — corpus tier, SEAR, layered provenance record, module map |
| [docs/GIN_ENG_01_SEAR_PoC_Spec.md](docs/GIN_ENG_01_SEAR_PoC_Spec.md) | SEAR proof-of-concept engineering spec and staged roadmap |
| [docs/GIN_ENG_02_Eval_Baseline_v1.md](docs/GIN_ENG_02_Eval_Baseline_v1.md) | RAG vs No-Continuation baseline, epistemic promotion runs, deeper implications |
| [docs/nc_mode_gating_retrieval_ordering.plan.md](docs/nc_mode_gating_retrieval_ordering.plan.md) | Phase 2: query-aware divergent gating + seed re-rank |
| [docs/nc_phase3_divergence_correctness.plan.md](docs/nc_phase3_divergence_correctness.plan.md) | Phase 3: divergent correctness + corroboration decode (promoted) |
| [docs/GIN_ENG_00_Engineering_Register.md](docs/GIN_ENG_00_Engineering_Register.md) | Engineering register — unmeasured specs and promotion rule |
| [docs/GIN_Node_Architecture_v1.md](docs/GIN_Node_Architecture_v1.md) | Node tier specification (institutional / relay / client) |
| [docs/GIN_The_Whole_Frame.md](docs/GIN_The_Whole_Frame.md) | Program synthesis — scale, governance, philosophy |
| [docs/GIN_00_Reader.md](docs/GIN_00_Reader.md) | Reading order for the full doc set |

---

## License

See [docs/GIN_12_Ecosystem_Licensing.md](docs/GIN_12_Ecosystem_Licensing.md) for the ecosystem licensing posture.
