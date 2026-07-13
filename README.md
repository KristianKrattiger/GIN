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
| `scripts/corpus_ingest.py` | Load YAML corpus into cold/warm/hot tiers (`--no-edges` for scan-first workflow) |
| `scripts/ingest.py` | Ingest local JSONL/txt into immutable store + versioned manifests |
| `scripts/corpus_query.py` | Hybrid dense + sparse retrieval (RRF merge) |
| `scripts/corpus_to_sear.py` | Materialize retrieved chunks into a SEAR `Corpus` (debug); supports `--manifest-version` with `--all` |
| `scripts/corpus_generate.py` | End-to-end: retrieve → prompt → constrained llama.cpp generation |
| `scripts/eval_run.py` | RAG vs SEAR eval harness — batch query set, overlap or NLI verifier, write `data/eval_runs/` report |
| `scripts/cartographer_scan.py` | Cartographer batch scan — propose edges, Bookkeeper admit, persist to Postgres |
| `scripts/cartographer_eval_scan.py` | Score scan admitted edges against gold hand-curated contradicts |
| `scripts/cartographer_eval_escalation.py` | Calibrate escalation frame judge on issue_frame gold + CLASS_C |
| `scripts/edges_wipe.py` | Truncate edges table only (re-scan without re-ingest) |
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

### Scan-first edge discovery (Cartographer validation)

To validate machine-discovered edges instead of hand-curated YAML `edges:` blocks:

```bash
# 1. Wipe + ingest chunks only (YAML edges kept as gold labels, not ingested)
python scripts/corpus_wipe.py
python scripts/corpus_ingest.py --source data/synthetic --no-edges
python scripts/corpus_ingest.py --source corpus_node1.json --no-edges
python scripts/corpus_ingest.py --source corpus_node2.json --no-edges
python scripts/corpus_ingest.py --source data/fixtures/disclosure_framing.yaml --no-edges
python scripts/corpus_ingest.py --source data/fixtures/housing_framing.yaml --no-edges
python scripts/corpus_ingest.py --source data/fixtures/wildfire_multipara.yaml --no-edges

# 2. Discover + admit edges (default: hybrid IDF+embedding prune, relation re-check, exclude out_of_scope_stub)
#    --curated-edges ingests hand-curated issue_frame contradicts (the class the
#    scan cannot machine-detect; see corpus_edges.yaml relation_class comments)
python scripts/cartographer_scan.py --cross-outlet-only --curated-edges data/corpus_edges.yaml

# Optional flags for A/B:
#   --no-prune              skip IDF+embedding candidate pruning (full O(n²) pair space)
#   --relatedness-floor 0.20  IDF overlap floor for stage-1 prune (default 0.20)
#   --no-relation-recheck   skip Bookkeeper semantic re-check (bidirectional-entailment deny)
#   --exclude-doc-id DOC    exclude chunks from edge discovery (repeatable)
#   --no-exclude-defaults   include out_of_scope_stub in scan
#   --escalation-judge local:path/to/model.gguf  route anchor-less pairs to a
#                           local llama.cpp frame judge (no API billing; default path)
#   --escalation-judge anthropic[:model]  optional API backend (needs ANTHROPIC_API_KEY)
#   --escalation-gpu-layers -1  GPU layers for local judge (-1 = all)
# Calibrate before trusting ANY model (bar: issue_frame_recall, class_c_discrimination,
# unrelated_discrimination all 1.0, mixed labels on the 33-pair breadth set):
#   python scripts/cartographer_eval_escalation.py --judge local:models/foo.gguf
python scripts/edges_wipe.py   # truncate edges only — re-scan without re-ingest

# 3. Score scan vs gold
python scripts/cartographer_eval_scan.py --cross-outlet-only

# 4. Divergence eval on scan-only edges (WSL recommended for llama.cpp)
python scripts/eval_run.py \
  --model models/Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  --queryset data/eval/queryset_twonode.yaml \
  --arms no_continuation \
  --verifier overlap \
  --edge-source cartographer_scan \
  --cartographer-scan-run-id <scan_eval_run_id>
```

**Scan precision pipeline** (2026-07-12, run `20260712T094453Z` — story-gated band with anchor tokens):

| Stage | Module | Effect |
|-------|--------|--------|
| Candidate prune | `RelatednessGate` + embedding cosine | 6,222 → 2,157 cross-outlet pairs |
| Same-story tier | `relatedness.make_same_story` | ≥ 2 shared corpus-rare tokens, ≥ 1 entity-grade anchor (mid-sentence capital / all-caps / multi-digit number); wired by `scan.wire_same_story` |
| Relation typing | `CombinedRelationProposer` | NLI + story-gated divergence band; mid-band default flipped to `related_untyped` |
| Doc-pair dedup | `scan.dedupe_doc_pair_proposals` | one contradicts per doc pair (NLI preferred) |
| Structural gate | `Bookkeeper` | confidence, anchors, dedup, cycles |
| Semantic re-check | `bookkeeper/relation_verify.py` | class-C entailment guard; `FRAMING_BAND_FLOOR=0.35` |

The mid-band-default-contradicts rule inverted at scan scale: measured on the
136-chunk DB, true framing divergences sit *above* the corroborate ceiling
(kestrel cos 0.698) while the mid band is cross-topic noise. Contradicts typing
on both channels now requires a shared story: ≥ 2 corpus-rare tokens including
an entity-grade anchor (Alderflats, Meridian, RIVERPORT, 842 — not 'remain in
effect'-style boilerplate). Gold includes four author-labeled
`news_corpus.yaml` edges (previously scored as false positives). Scan eval
reports both chunk-pair and doc-pair granularity; admitted chunk pairs on a
curated doc pair at a different anchor count as **anchor discoveries**, not
machine false positives (`anchor_discovery_keys` in scan eval output).

**Gold is split by `relation_class`** (`scan_eval.split_gold_by_class`):
machine metrics score the *story* class (same story, conflicting accounts);
the four *issue_frame* pairs (same issue, opposing frames, zero shared
entities — `corpus_edges.yaml`, including alternate emissions anchor
`n1_doc_005:1 ↔ n2_doc_001:1`) are curated-ingest edges. The 2026-07-12
signal audit measured them machine-undetectable locally: six embedding models
(margin −0.39…−0.08 vs adjacent noise), NLI ≈ 0 both directions, register-axis
delta overlapping, Mistral-7B constant-answer zero- and few-shot. Forward path
for corpora where curation does not scale: `--escalation-judge local:model.gguf`
routes the anchor-less topically-close residue (91–338 pairs at floors
0.40–0.30) to a model-agnostic framing judge via llama.cpp
(`gin/cartographer/escalation.py`). Optional `anthropic:model` backend for
overflow. **Calibrate per model** with `cartographer_eval_escalation.py`
(4 issue_frame gold + 6 corroboration + 4 cross-issue controls, both
directions, plus the 33-pair labeled breadth set; judge reasoning stored per
pair). Measured 2026-07-13: the constant-DIVERGENT collapse was substantially
**harness-induced** — the one-word budget (`071808Z`) and a register exemplar
in the DIVERGENT definition (`083941Z`) each reproduce it. With the
register-neutral reasoning prompt Mistral-7B mixes labels but stays far below
the bar: recall 0.5, class_c 0.67, unrelated 0.25, and 7/14 pairs flip label
under argument-order swap (`092010Z`) — 7B verdicts on this class are noise.
Prompt wording is calibration-sensitive; never edit `FRAME_JUDGE_PROMPT`
without a rerun.

**The bar is signal-bound, not scale-bound — closed at every tier including the
frontier** (2026-07-13; local models Q4_K_M with the `[INST]` harness held
constant; Opus 4.8 via API. Small n — 4/6/4 pairs — read direction, not
decimals):

| Model | Active | issue_frame recall | class_c | unrelated | order-flips | labeled acc |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Mistral-7B dense (`092010Z`) | 7B | 0.50 | 0.67 | 0.25 | 7/14 | 0.394 |
| Qwen3.6-14B-A3B MoE (`190431Z`) | ~3B | 0.25 | 0.50 | 0.50 | 7/14 | 0.636 |
| Qwen2.5-14B dense (`192930Z`) | 14B | 0.50 | 0.33 | **1.00** | **3/14** | 0.727 |
| Opus 4.8 frontier (`223653Z`) | — | **0.00** | 0.67 | **1.00** | **3/14** | **0.788** |

Two readings stack. First, **architecture**: active-compute depth (dense 14B)
buys what scale is supposed to — order-flips halve (7→3), `unrelated` goes
perfect, and it's the only *local* model that emits UNRELATED at all (the
3B-active MoE dropped the category, folding different-issue pairs into
AGREE/DIVERGENT); MoE total-param breadth helped only generic classification,
not the targeted stance call.

Second, and decisive, **the frontier inverts the premise**. Opus 4.8 is the best
general judge on every competence metric (labeled acc 0.788, `unrelated` 1.0,
best-tier stability) yet scores `issue_frame` recall **0.00** — the worst of any
model. Recall down the capability ladder is 0.50 → 0.25 → 0.50 → 0.00: it does
not rise with capability. Opus doesn't fail these pairs, it reads them lucidly
and *disagrees with the gold*: the climate pairs it calls AGREE ("both point the
same direction — A the diagnostic, B the prescriptive — they corroborate"), the
wildfire/water pairs UNRELATED ("distinct questions — magnitude of burning vs
public-health vulnerability"). The `issue_frame` "contradicts" edge encodes a
**critical-theory reading** (institutional framing *opposes* justice framing)
that competent judges recover as same-direction corroboration or topic
difference, not stance opposition. It is a contestable curatorial judgment, not
a latent signal — fittingly, a curator-vs-reader divergence of exactly the kind
GIN exists to hold legible.

**Conclusion:** the escalation-judge path is closed at every tier, not for lack
of capability but because the label is an editorial stance no off-the-shelf
judge reproduces. `issue_frame` stays **curation-only by nature.** Wording
remains calibration-sensitive; never edit `FRAME_JUDGE_PROMPT` without a rerun.

**Forward path (much later phase): a purpose-trained judge.** Supervised
fine-tuning *learns the labeler's function*, so a judge trained on the
curatorial framing is the way to scale this class where curation doesn't — but
this finding reframes what that model is (a learned encoding of one editorial
frame, not a universal divergence detector) and names its prerequisite: a
labeled framing corpus at scale (four gold pairs cannot train anything). Not on
the Phase-1/2 path.

| Metric | Old band (`074956Z`) | Story gate (`091415Z`) | Anchors, gold 11 (`094453Z`) | Class split (`202240Z`) | Label closure (`220456Z`) |
|--------|---|---|---|---|---|
| Admitted contradicts | 122 | 20 | 11 | 11 (+3 curated at persist) | 11 (+3 curated; scan also hits `:1↔:1`) |
| False positives (chunk) | 120 | 15 | 3 | **3** | **1** (labor only) |
| Anchor discoveries | — | — | — | — | **1** (`n1_doc_005:1 ↔ n2_doc_001:1`) |
| Machine gold admitted | 2/8 | 5/8 | 8/11 | **8/8 — recall 1.0, 0 missed** | **9/9 — recall 1.0, 0 missed** |
| Curated gold (issue_frame) | — | — | — | 3/3 via `--curated-edges` | **4/4** (3 at persist when scan covers `:1↔:1`) |
| class_c_discrimination | 1.0 | 1.0 | 1.0 | 1.0 | **0.5** (labor control fails; twonode passes) |

Labeled set: precision 1.0, recall 4/7 (legal and housing 1.0; the four
climate pairs are the issue_frame class). The Bookkeeper re-check is now
entailment-only (`relation_verify.py`) — its band/nli branches were circular
(same NLI signal the proposer applied) and are removed; confidence floors live
at the Bookkeeper gate.

**Label decisions closed** (2026-07-12):

- `incident_metrodaily:0 ↔ incident_regionalpost:0` — gold `contradicts` in
  `news_corpus.yaml` (98 vs 142 treated).
- `n1_doc_005:1 ↔ n2_doc_001:1` — second `issue_frame` gold in
  `corpus_edges.yaml`; scan eval reports it as an anchor discovery when the
  scan admits that anchor (primary gold remains `:2 ↔ :4`).
- `labor_bureau_report:0 ↔ labor_independent_survey:0` — **no** gold edge
  (corroboration-with-a-caveat); added to `CLASS_C_CONTROLS` so scan eval
  penalizes admission. Remains the sole true chunk-level FP until the detector
  learns interpretation-caveat vs factual divergence.

Artifact: `data/eval_runs/20260712T220456Z/`.

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
| `20260702T012203Z` | 20 | **NC epistemic promotion** (overlap, CPU) |
| `20260702T010918Z` | 9 | Regression anchors post–Phase 3 |
| `20260711T212721Z` | 20 | Same-day CPU control (overlap, `n_gpu_layers=0`) |
| `20260711T211202Z` | 20 | **GPU hardware artifact** (overlap, RTX 4070, `n_gpu_layers=-1`) — see below |

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

**GPU hardware artifact (`20260711T211202Z`, RTX 4070) — measured and root-caused.** A same-day CPU control (`20260711T212721Z`, identical code and corpus state) isolates the true CPU/GPU gap to **1 of 20 queries**: retrieval is byte-identical, but that one query's refuse-vs-answer decision flips under GPU decode. Fabrication rate is 0.000 on both backends. Root cause: llama.cpp's CPU and CUDA kernels aren't required to be bit-exact even at `temperature=0.0`, and this query is a near-tie at SEAR's first decode step — small logit noise flips it. Naively diffing against the 9-day-old baseline instead made it look like 3 queries regressed; 2 of those were stale-corpus artifacts (retrieval tie-break order isn't stable across separate ingestion runs), not GPU-specific. Details: [ENG 02, Remaining gaps items 3 and 5](docs/GIN_ENG_02_Eval_Baseline_v1.md).

**Generalization beyond the synthetic corpus.** The divergence mechanism was stress-tested on **real fetched two-node text** (institutional statistic vs. grassroots reframing — pairs that share no lede structure) and holds: `divergence_fidelity` **1.000**, `fabrication_rate` **0.000** (`20260705T043114Z`). It generalizes across three framing registers (climate, adversarial/legal, housing) and is **model-independent** — Qwen2.5-7B matches the Mistral baseline exactly on all four divergence querysets. Reconfirmed on **GPU** (`20260711T214751Z`, RTX 4070, `n_gpu_layers=-1`): `divergence_fidelity` **1.000**, `fabrication_rate` **0.000**, 22.2 tok/s — unlike the synthetic-corpus NC run, the two-node divergence result shows zero CPU/GPU gap. Method, root-cause analysis, and per-pair IDF/token tables: [docs/nc_real_text_divergence_generalization.plan.md](docs/nc_real_text_divergence_generalization.plan.md).

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
| Two-node divergence demo (inter-corpus, real fetched text) | ✅ (run `20260705T043114Z`, fidelity 1.0) |
| Divergence generalization across framing registers (climate / legal / housing) | ✅ (`20260705T202450Z`, `20260705T203622Z`) |
| Cross-model confirmation (Qwen2.5-7B) — divergence is model-independent | ✅ (`20260705T211452Z`–`20260705T220525Z`) |
| Representative GPU hardware artifact | ✅ (run `20260711T211202Z`, RTX 4070, vs same-day CPU control `20260711T212721Z`; gap root-caused to 1/20 queries, backend floating-point non-determinism, fabrication unaffected) |
| Retrieval determinism + `corpus_fingerprint` in eval meta | ✅ |
| Convergent sentence-end close (`tn_2023_anomaly` truncation) | ✅ |
| Cartographer scan + Bookkeeper Postgres persist | ✅ (`scripts/cartographer_scan.py`, `gin/bookkeeper/persist.py`) |
| Cartographer scan production validation (scan-only divergence eval) | ✅ Machine recall 1.0 on 9/9 story-class gold, chunk FP 1 (labor CLASS_C; anchor discovery tracked separately) (`20260712T220456Z`); issue_frame class 4/4 curated via `--curated-edges`; twonode divergence eval coverage 1.0 (`20260712T203110Z`); model-agnostic escalation judge (local llama.cpp + optional anthropic) with reasoning-prompt calibration harness — Mistral-7B, Qwen3.6-14B-A3B MoE, Qwen2.5-14B dense, **and Opus 4.8 frontier** all measured below bar (`20260713T092010Z`/`190431Z`/`192930Z`/`223653Z`); `issue_frame` recall does not rise with capability (0.50→0.25→0.50→**0.00**) — the frontier judge lucidly reads the pairs as corroborate/unrelated and disagrees with the gold, so the `contradicts` label is a contestable curatorial stance, not a latent signal; **epistemic phase closed — class is curation-only by nature; forward path is a purpose-trained judge (much later)** |
| Labeled set expanded + threshold calibration (33 pairs, LOO ≥ 0.85) | ✅ (`data/cartographer_thresholds.json`) |
| NLI verifier on expanded 20-query set | ✅ (`20260712T035228Z`, `models/Mistral-7B-Instruct-v0.3-Q6_K.gguf`, WSL+GPU; NC realism fabrication 0.0, overall NLI fabrication 0.056 on counterfactual entailment miss) |
| Bookkeeper + reasoning layer separation (Phase 2) | ✅ (admission gate wired; synthesis reads warm `edges`) |
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
| [docs/nc_real_text_divergence_generalization.plan.md](docs/nc_real_text_divergence_generalization.plan.md) | Two-node real-text divergence, framing generalization, cross-model check |
| [docs/GIN_ENG_00_Engineering_Register.md](docs/GIN_ENG_00_Engineering_Register.md) | Engineering register — unmeasured specs and promotion rule |
| [docs/GIN_Node_Architecture_v1.md](docs/GIN_Node_Architecture_v1.md) | Node tier specification (institutional / relay / client) |
| [docs/GIN_The_Whole_Frame.md](docs/GIN_The_Whole_Frame.md) | Program synthesis — scale, governance, philosophy |
| [docs/GIN_13_Temporal_Sensor_Grounding.md](docs/GIN_13_Temporal_Sensor_Grounding.md) | Forward direction — extending SEAR grounding from text to sensor ground truth |
| [docs/GIN_00_Reader.md](docs/GIN_00_Reader.md) | Reading order for the full doc set (incl. conceptual, engineering, strategy registers) |

---

## License

See [docs/GIN_12_Ecosystem_Licensing.md](docs/GIN_12_Ecosystem_Licensing.md) for the ecosystem licensing posture.
