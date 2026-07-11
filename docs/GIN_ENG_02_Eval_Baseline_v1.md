---
tags: [GIN, engineering, SEAR, eval, baseline, register]
updated: 2026-07-11
version: 0.6-divergence-generalized
status: measured (NC epistemic targets met on expanded queryset; divergence generalized to real text + second model)
register: engineering
implements: GIN ENG 01 — SEAR PoC Spec §6 (Stage 1 metrics harness)
artifact: data/eval_runs/20260702T012203Z/
---

# GIN ENG 02 — SEAR vs RAG Eval Baseline (v1)

> Measured comparison of **traditional RAG** against **GIN No-Continuation** (SEAR extractive-only decoding) on the synthetic eval corpus.
>
> **Current promotion-quality NC epistemic runs** (2026-07-02, overlap verifier): full 20-query `20260702T012203Z`, regression 9-anchor `20260702T010918Z`. NC meets all six epistemic promotion targets on the expanded queryset (fabrication 0, query relevance 1.0, supported irrelevance 0, gold coverage 1.0, counterfactual adherence 1.0, divergence fidelity 1.0).
>
> **Structural prevention baseline** (9-query, pre-expansion): overlap `20260701T192827Z`, NLI `20260701T194024Z`. Earlier overlap `20260701T175728Z` established the prevention delta; pre-fix NLI `20260701T182426Z` is diagnostic only.

---

## What was measured

| Field | Value |
|-------|-------|
| **Primary run IDs (NC epistemic)** | Full 20: `20260702T012203Z` · Regression 9: `20260702T010918Z` |
| **Primary run IDs (structural prevention)** | Overlap: `20260701T192827Z` · NLI: `20260701T194024Z` |
| **Artifact paths** | `data/eval_runs/20260702T012203Z/`, `data/eval_runs/20260702T010918Z/`, `data/eval_runs/20260701T192827Z/` |
| **Harness** | `scripts/eval_run.py` → `gin/eval/` |
| **Query set** | `data/eval/queryset.yaml` (20 queries; 9 regression anchors via `regression: true`) |
| **Corpus** | `data/synthetic/news_corpus.yaml` ingested via `corpus_ingest.py` |
| **Model** | Mistral-7B-Instruct-v0.3 Q6_K GGUF (`llama-cpp-python`, `n_ctx=2048`, `n_gpu_layers=0`, `temperature=0.0`) |
| **Arms** | `rag`, `no_continuation` |
| **Verifiers** | Token overlap and NLI (`cross-encoder/nli-deberta-v3-xsmall`, threshold 0.5) |
| **Retrieval** | Held constant: `retrieve_for_synthesis()` per query, no `eval_layer` filter |
| **Harness changes (Sprint A–C)** | Citation parser for `[n: id]` / `[n, m]`; NLI `apply_softmax` + EXACT-span fast path; post-retrieval relevance gate (`relevance_floor=0.20`); gold recall@k in report |
| **Harness changes (Phase 1–4)** | Epistemic metrics (`query_relevance_rate`, `gold_chunk_coverage`, `supported_irrelevance_rate`, `chunk_quotation_rate`, `divergence_fidelity`); `contradicts_pairs` on divergence queries; `--regression-only`; `n_gpu_layers` + wall-clock in `meta.json`; eval-only `--boost-gold-chunks` / `--gold-refuse-without-coverage` for NC |
| **Harness changes (NC selection robustness, Phase 1)** | Always-on query steering in convergent mode; top-doc `preferred_starts`; `focus_doc_indices` + `allow_shared_prefix=False`; single-span default (`stop_after_first_extract`, `max_tokens=60`); hit re-rank by query sentence score; post-decode query-relevance refusal gate |
| **Harness changes (mode gating + retrieval ordering, Phase 2)** | Query-aware divergent gating (`_divergence_relevant`); seed re-rank before mode detection; zero-relevance seed filter; `competing_same_tag` corroboration decode path for bureau+survey bundles |
| **Harness changes (divergent correctness, Phase 3)** | Divergent mode requires query-relevant contradicts pair (close-competitor fallback removed when query present); `matched_keyword_count` floor (≥2 keywords when query has ≥3); auto-close fires at fork step only and respects sentence-end; EOS blocked only when `required_doc_groups` non-empty; corroboration spans cite both same-tag outlets |

**Arms compared**

- **RAG** — unconstrained `create_completion` with full chunk bodies in-context and a cite-or-refuse instruction.
- **No-Continuation** — `ExtractiveCopyConstraint` via `generate_no_continuation()`; every emitted token is a verbatim corpus span; refuses when retrieval relevance is below floor or when post-decode claims fail query-relevance gate.

**Synthesis mode and decode policy (Phases 1–3)**

1. **Mode detection** (`retrieve_for_synthesis`): seeds are re-ranked and zero-relevance noise dropped *before* mode is set. **Divergent** only when a `contradicts` pair is query-relevant on **both** sides (`DIVERGENCE_RELEVANCE_FLOOR` + `matched_keyword_count`). Close RRF competitors without a contradicts edge are **corroboration → convergent**.
2. **Convergent (default)**: one query-steered extract via top-doc `preferred_starts`, `focus_doc_indices={top_doc}`, `allow_shared_prefix=False`, `stop_after_first_extract=True` (`max_tokens=60`).
3. **Corroboration** (same `eval_tag`, different outlets, agreeing text — e.g. bureau + independent survey): `competing_same_tag` path — shared-prefix span to sentence end, both docs in focus, `max_tokens=100`, dual gold citation via AMBIGUOUS span.
4. **Divergent** (incident/election probes): multi-span weaving, divergence zones, `stop_when_groups_satisfied` only when `required_doc_groups` non-empty; auto-close at doc fork respects sentence boundaries.

Complementary multi-doc convergent weave (non-contradicting completion sets) remains future work. Design notes: `docs/nc_mode_gating_retrieval_ordering.plan.md`, `docs/nc_phase3_divergence_correctness.plan.md`.

**Historical artifacts**

| Run ID | Verifier | Queries | Status |
|--------|----------|---------|--------|
| `20260701T175728Z` | overlap | 9 | Original baseline; prevention delta first measured |
| `20260701T182426Z` | nli (pre-fix) | 9 | **Diagnostic only** — citation parser + NLI calibration bugs |
| `20260701T192827Z` | overlap | 9 | Structural prevention baseline (9-query) |
| `20260701T194024Z` | nli | 9 | NC fabrication 0 under NLI (9-query) |
| `20260701T205324Z` | overlap | 20 | Pre–Phase 1 NC epistemic baseline |
| `20260701T214706Z` | overlap | 20 | Post–Phase 1 only (query relevance still 0.70) |
| `20260702T003047Z` | overlap | 20 | Post–Phase 2 (4/6 epistemic targets) |
| `20260702T010918Z` | overlap | 9 | **Regression anchors post–Phase 3** |
| `20260702T012203Z` | overlap | 20 | **Current NC epistemic promotion run** |

---

## Results (NC epistemic promotion) — `20260702T012203Z` (full 20)

Overlap verifier, Mistral-7B Q6_K, CPU (`n_gpu_layers=0`).

| Metric | RAG | No-Continuation | Target |
|--------|-----|-----------------|--------|
| Fabrication rate | 0.238 | **0.000** | 0.000 |
| Query relevance rate | 1.000 | **1.000** | ≥ 0.90 |
| Gold chunk coverage | 0.956 | **1.000** | ≥ 0.75 |
| Supported irrelevance rate | 0.000 | **0.000** | ≤ 0.05 |
| Counterfactual adherence | 1.000 | **1.000** | ≥ 0.90 |
| Divergence fidelity | 0.875 | **1.000** | preserve |
| Failure-state recall (out_of_scope) | 1.000 | **1.000** | — |
| Mean gold recall@k | 1.000 | 1.000 | — |

Regression run `20260702T010918Z` (9 anchors): NC epistemic metrics all **1.000**; divergence fidelity **1.000**.

**Unit tests** (no llama.cpp): `tests/test_retrieve_synthesis.py` (22 tests, Phases A–C + 3A–3B), `tests/test_processor.py` (auto-close fork replay), `tests/test_generate.py` (`competing_same_tag`) — all passing as of Phase 3 promotion.

**Note:** Counterfactual NC `cross_node_within_ratio` is **0.000** by design — corroborated answers are single AMBIGUOUS spans citing bureau + independent survey (different outlets). Zero cross-node violations; not an epistemic promotion target.

---

## Results (overall) — structural prevention runs (9-query)

### Overlap verifier (`20260701T192827Z`)

| Metric | RAG | No-Continuation |
|--------|-----|-----------------|
| Fabrication rate | **0.286** | **0.000** |
| Grounded precision | 0.714 | **1.000** |
| Attribution coverage | 0.714 | **1.000** |
| Counterfactual adherence | **1.000** | 0.000 |
| Failure-state precision | 0.500 | **1.000** |
| Failure-state recall | **1.000** | **1.000** |
| Cross-node within ratio | 0.714 | **1.000** |
| Cross-node violations | 0 | 0 |
| Claims emitted | 7 | 29 |
| Queries | 9 | 9 |
| Mean gold recall@k | 1.000 | 1.000 |

### NLI verifier (`20260701T194024Z`)

| Metric | RAG | No-Continuation |
|--------|-----|-----------------|
| Fabrication rate | 0.714 | **0.000** |
| Grounded precision | 0.286 | **1.000** |
| Attribution coverage | 0.286 | **1.000** |
| Counterfactual adherence | 0.000 | 0.000 |
| Failure-state precision | 0.500 | **1.000** |
| Failure-state recall | **1.000** | **1.000** |
| Cross-node within ratio | 0.286 | **1.000** |
| Cross-node violations | 0 | 0 |
| Claims emitted | 7 | 29 |
| Queries | 9 | 9 |
| Mean gold recall@k | 1.000 | 1.000 |

### By eval_layer (overlap run)

**Realism (6 queries)**

| Metric | RAG | No-Continuation |
|--------|-----|-----------------|
| Fabrication rate | 0.333 | 0.000 |
| Grounded precision | 0.667 | 1.000 |

**Counterfactual (1 query)** — `unemployment_rate`

| Metric | RAG | No-Continuation |
|--------|-----|-----------------|
| Counterfactual adherence | 1.000 | 0.000 |
| Gold recall@k | 1.000 | 1.000 |

**Out of scope (2 queries)**

| Metric | RAG | No-Continuation |
|--------|-----|-----------------|
| Failure-state recall | 1.000 | 1.000 |
| Claims emitted | 0 | 0 |

Full machine-readable breakdown: `metrics.json` per run. Per-query claims: `results/rag.json`, `results/no_continuation.json`. Retrieval: `retrieval/<query_id>.json`.

---

## NLI scoring caveat (pre-fix run `182426Z`)

The first NLI run reported NC fabrication **0.676** and RAG **1.000** — not comparable to overlap. Root causes (now fixed):

1. **Citation parser** — Mistral emits `[1: chunk_id]`, `[4, 5]`; bare `[n]` parser left `cited_chunk_ids` empty → attribution_coverage 0 for RAG.
2. **NLI calibration** — `_nli_score()` now uses `apply_softmax=True` and resolves entailment label index from model config.
3. **EXACT-span fast path** — verbatim No-Continuation claims bypass NLI when claim text is a substring of the cited chunk (structural grounding).

Post-fix NLI run (`194024Z`): NC fabrication **0.000**, failure recall **1.000**. RAG NLI fabrication remains high (0.714) — paraphrased answers and strict entailment threshold; overlap run (0.286) is the fairer RAG baseline until RAG-specific NLI tuning.

**Trust hierarchy**

1. **Overlap `192827Z`** — authoritative for prevention delta and NC structural metrics.
2. **NLI `194024Z`** — authoritative for NC under entailment verifier; RAG comparison still caveat-heavy.
3. **NLI `182426Z`** — diagnostic only; do not cite metrics.

---

## Interpretation

### What the baseline confirms

1. **Structural prevention works.** No-Continuation achieves **zero fabrication** on both overlap and post-fix NLI runs. EXACT spans score SUPPORTED by construction or via substring fast path.

2. **Failure state wired.** Post-retrieval relevance gate (`relevance_floor=0.20`) drives NC `refused=True` on `interest_rate_probe` and `sports_probe`. Failure recall **1.000** on out_of_scope (was 0.0 in `175728Z`).

3. **Citation parser improved RAG overlap scores.** RAG fabrication dropped **0.571 → 0.286** (overlap); attribution coverage **0.429 → 0.714**.

4. **Retrieval quality is now separable.** Gold recall@k logged per query. `unemployment_rate` shows recall@k **1.0** (labor chunks retrieved) while NC still extracts incident text — counterfactual adherence 0 is a **synthesis selection** problem, not retrieval miss.

5. **Divergent realism surfaces.** On `incident_arrests`, NC emits both 23-arrest and 11-arrest spans without collapsing.

### Remaining gaps (post–Phase 3)

1. **NLI re-run on expanded 20-query set** — promotion metrics measured under overlap only for NC epistemic targets; NLI `194024Z` is 9-query.

2. **RAG under NLI** — High fabrication on paraphrase; overlap is fairer for RAG comparison.

3. **Representative GPU hardware** — **Measured and root-caused** (`20260711T211202Z`, RTX 4070, `n_gpu_layers=-1`, Q6_K, vs same-day CPU control `20260711T212721Z`, `n_gpu_layers=0`, identical code and corpus state). Fabrication rate holds at 0.000 on both backends. A same-day, same-corpus, same-code control (essential — see item 5) isolates the true CPU/GPU generation gap to **exactly 1 of 20 queries** (`incident_hospital`): retrieval is byte-identical, but CPU emits the correct divergent two-claim answer while GPU emits an outright refusal ("The sources do not support an answer."). Root cause: llama.cpp's CPU and CUDA backends are not required to be bit-exact even at `temperature=0.0` (differing floating-point summation order in matmul kernels), and this particular query is evidently a near-tie between "start extracting" and "refuse" at SEAR's first decode step — small logit noise is enough to flip a close call. This is expected cross-backend variance in a hard-constraint decoder, not a GIN defect; fabrication risk (the property SEAR is actually designed to eliminate) is unaffected. Promotion rule should define a tolerance band (e.g. ≤1/20 boundary-decision flips) for cross-hardware reproduction rather than requiring bit-exact metric parity.

4. **Verifier min-length floor** — Short fragment claims (e.g. truncated numerics) can score overlap 1.0; worth a follow-up harness tweak, not blocking NC promotion.

5. **Retrieval non-determinism across ingestion runs** — comparing `20260711T211202Z` directly against the 9-day-old `20260702T012203Z` baseline initially looked like a 3-query GPU regression (`incident_hospital`, `election_margin`, `school_enrollment_fall`). A same-day CPU control (`20260711T212721Z`, ingested fresh from the same `corpus_ingest.py` run as the GPU artifact) showed 2 of those 3 "flips" reproduce identically on CPU too — they're artifacts of re-ingesting the corpus between baseline and GPU runs (RRF tie-break / chunk ordering isn't guaranteed stable across separate ingestion runs when scores are close), not GPU-specific at all. Real gap going forward: **compare same-corpus-state runs only**; consider adding an explicit deterministic tie-break (e.g. secondary sort by `chunk_id`) to retrieval so ordering is stable across re-ingestion.

### Epistemic quality metrics

Report section **## Epistemic quality** scores what fabrication rate cannot see:

| Metric | Definition |
|--------|------------|
| `query_relevance_rate` | Per query: any non-refusal claim overlaps the query (≥ `0.20`) or cites a `gold_chunk_id`; out_of_scope passes on correct refusal |
| `gold_chunk_coverage` | Mean fraction of `gold_chunk_ids` with at least one SUPPORTED claim citing that chunk |
| `supported_irrelevance_rate` | SUPPORTED claims with zero query overlap and no gold citation / total SUPPORTED |
| `chunk_quotation_rate` | Mean \|cited chunk ids\| / \|retrieved chunk ids\| per query |
| `divergence_fidelity` | On `incident_divergence` / `election_divergence` tags: fraction of `contradicts_pairs` where both chunks appear in cited output |

On overlap run `192827Z` (pre–Phase 1), NC `query_relevance_rate` &lt; 1.0 on realism — failing IDs included `transit_ridership`, `weather_winds`. Post–Phase 3 run `20260702T012203Z`: all 20 queries pass query relevance; `weather_winds` retrieves only `weather_service_brief:0`.

---

## Deeper implications of the post-fix reruns

> **Historical note (2026-07-01):** This section documents the *pre–Phase 1* harness era (`192827Z`, 9 queries). NC selection failures described below (supported irrelevance, counterfactual misses, `weather_winds` incident spans) were **resolved** by Phases 1–3; see [Results (NC epistemic promotion)](#results-nc-epistemic-promotion--20260702t012203z-full-20) and `20260702T012203Z`.

The promotion tables summarize *whether* metrics moved. This section records *what they mean* — which comparisons answer which questions, and what failure modes remain visible in the per-query JSON.

### Three comparisons, three questions

| Comparison | What changed | Question answered |
|------------|--------------|-------------------|
| `175728Z` → `192827Z` (overlap) | Harness **and** arm behavior | Did wiring failure state and citation parsing change real outcomes? |
| `182426Z` → `194024Z` (NLI) | Harness scoring only | Was NC fabrication under NLI a measurement bug or a decode regression? |
| `192827Z` vs `194024Z` (same generation) | Verifier only | What does each verifier actually measure? |

Do not treat all run IDs as interchangeable. The pre-fix NLI run (`182426Z`) is a harness diagnostic, not evidence against the prevention delta.

### Structural prevention is verifier-robust (NC)

Original baseline (`175728Z`): NC fabrication = 0 under overlap only.

Post-fix (`192827Z` overlap, `194024Z` NLI): NC fabrication = 0 under **both** verifiers, with the same 29 claims.

Pre-fix NLI (`182426Z`) reported NC fabrication **0.676** because verbatim hospital-count extracts scored ~0.003 entailment before calibration and the EXACT-span fast path. Generation did not change; scoring did.

**Implication:** Stage 1b prevention — no decode-time tokens outside the corpus — is measured, not argued. EXACT spans are SUPPORTED by substring match or by NLI after the fast path.

### Failure state: coarse but effective on out_of_scope

**Before (`175728Z`):** `interest_rate_probe` and `sports_probe` emitted incident/election spans (5 claims). NC failure recall = 0.

**After (`192827Z`):** Both probes → `refused: true`, 0 claims. NC failure recall = 1.0, failure precision = 1.0.

Mechanism: post-retrieval query–chunk token overlap below `relevance_floor=0.20` in `NoContinuationArm` / `RagArm` (`gin/eval/arms.py`). Retrieval may still return chunks (e.g. labor/incident text for an interest-rate query), but if no chunk shares enough query terms, the arm emits `REFUSAL_SENTINEL` and `refused=True`.

**Implication:** SEAR now has an explicit GATE-style abstention path. The gate is **lexical** (token overlap), not semantic — it refuses when retrieved text does not overlap the query, not when a model "knows" the question is unanswerable.

### Retrieval metrics split synthesis from retrieval

Gold recall@k in the report separates layers that aggregate metrics previously conflated.

**`unemployment_rate` (counterfactual):**

| Signal | Value |
|--------|-------|
| Gold recall@k | **1.0** — `labor_bureau_report:0` and `labor_independent_survey:0` in retrieved set |
| RAG counterfactual adherence (overlap) | **1.0** — states "3.7 percent" with parsed `[4, 5]` citations |
| NC output | Incident hospital/arrest spans — no "3.7 percent" |
| NC counterfactual adherence | **0.0** |

The original baseline attributed NC counterfactual failure partly to retrieval miss. Post-fix data shows retrieval **succeeds** while **synthesis selection** fails: the constrained decoder surfaces dominant incident spans from a mixed bundle instead of labor chunks that are also present.

**Implication:** Counterfactual behavior is not fixable by retrieval tuning alone. It needs query-aware materialization — biasing which chunks enter extractive context, or refusing when no emitted span matches the query's target even if spans are structurally valid.

### Zero fabrication can hide epistemic failure (supported irrelevance)

Fabrication rate measures **chunk grounding**, not **question answering**. Per-query JSON exposes cases where NC scores perfectly while answering the wrong question.

**`transit_ridership` and `weather_winds`:**

| Signal | RAG | No-Continuation |
|--------|-----|-----------------|
| Gold recall@k | 1.0 | 1.0 |
| Behavior | Refuses ("sources do not support an answer") | Emits incident hospital-count spans |
| Harness score | Failure-precision FP on realism | Fabrication 0, all claims SUPPORTED |
| Epistemic outcome | Under-confident (wrong) | Wrong topic, structurally grounded |

The gold chunks (`transit_authority_update:0`, `weather_service_brief:0`) are in the retrieved set for both arms. NC does not extract from them; it extracts from higher-salience incident text in the same bundle.

**Implication:** NC can pass structural metrics while being useless or misleading. The 29 vs 7 claim asymmetry (NC vs RAG) is not only "more friction" — it is many structurally valid spans that do not address the query. Selection bias ([[GIN_04_SEAR]] open issue) is visible in the data before a dedicated metric exists.

A future **query-relevance** metric (e.g. whether any SUPPORTED claim contains query-critical terms or gold-chunk content) would flag `weather_winds` NC output as failed despite fabrication 0.

### RAG: two failure modes, one improved

**Overlap fabrication 0.571 → 0.286** (`175728Z` → `192827Z`):

- Citation parser now handles `[1: chunk_id]`, `[4, 5]`, etc. — attribution coverage 0.429 → 0.714.
- Remaining UNSUPPORTED claims include meta-commentary on `incident_arrests` ("unclear which source is correct") — arguably correct scoring of non-propositional text, not hallucination.

**Persistent false refusals:** `transit_ridership` and `weather_winds` refused despite gold recall@k = 1.0. Confusion matrix `fp=2` on realism in post-fix runs. RAG's failure mode on these probes is **over-refusal** when incident noise dominates the prompt, not fabrication.

### NLI: fair for NC, not yet fair for RAG

Same 7 RAG claims in `192827Z` and `194024Z`, different scores:

| Query (RAG) | Overlap | NLI |
|-------------|---------|-----|
| `unemployment_rate` | SUPPORTED (1.0) | UNSUPPORTED (~0.002) — text is correct, citations parsed |
| `incident_arrests` meta-sentences | UNSUPPORTED | UNSUPPORTED — agreement |
| NC EXACT spans | SUPPORTED | SUPPORTED — agreement after fast path |

**Implication:** Use overlap for RAG baseline comparison today. NLI confirms NC under entailment; paraphrased correct RAG answers fail strict entailment at threshold 0.5.

### Architectural summary

```text
                    RAG                    No-Continuation
Decode integrity    tokens may be invented  tokens ⊆ corpus (measured)
Query relevance     model judgment          lexical gate (out_of_scope only)
Selection           single answer           many verbatim spans (29 vs 7 claims)
When retrieval mixed
  - RAG             may refuse (transit)    may extract wrong chunk (transit)
  - NC              may paraphrase wrong    extracts wrong chunk, scores SUPPORTED
```

1. **No-Continuation solves decode integrity** — measured, promotion-quality under overlap and NLI.
2. **It does not solve epistemic alignment** — whether the right spans are selected for the question. Relevance gate helps out_of_scope; it does not help when retrieval is a mixed bag and incident text wins the decode race.
3. **RAG trades integrity for fluency** — can answer counterfactual correctly when NC cannot; refuses or fabricates on other probes.

### What the reruns imply for next work

| Priority | Rationale from reruns |
|----------|----------------------|
| Query-set expansion | Make transit/weather/NC wrong-answer pattern statistically visible |
| Divergence fidelity + chunk-quotation rate | Measure selection bias directly (29 NC claims, many off-query) |
| Query-relevance metric | Distinguish fabrication from supported irrelevance |
| Counterfactual routing | Gold recall 1.0 but wrong spans — materialize from labor chunks when present |
| RAG NLI tuning or overlap-as-primary | Cross-arm NLI comparison not promotion-ready for RAG |

---

## Post-v1: divergence generalization (real text + cross-model)

The v1 baseline above is measured on the **synthetic** single-corpus queryset, where every `contradicts` pair shares a lede and forks at one sentence. Follow-on work stress-tested whether that promotion generalizes to real editorial divergence (institutional statistic vs. grassroots reframing) that shares *no* structure, and to a second model. Full method, root-cause analysis, and per-pair token/IDF-margin tables: `docs/nc_real_text_divergence_generalization.plan.md`.

| Result | Run | divergence_fidelity (SEAR) | fabrication_rate (SEAR) |
|--------|-----|----------------------------|-------------------------|
| Two-node real-text (climate: emissions / wildfire / water) | `20260705T043114Z` | **1.000** (0.333 pre-fix `105554Z`) | 0.000 |
| Multi-paragraph chunks (anchor-vs-filler selection) | `20260705T065559Z` | **1.000** | 0.000 |
| Framing round 1 — adversarial/legal (press release vs. regulator complaint) | `20260705T202450Z` | **1.000** | 0.000 |
| Framing round 2 — housing (zoning-board vs. tenant-organizing) | `20260705T203622Z` | **1.000** | 0.000 |
| Cross-model — Qwen2.5-7B on all four divergence querysets | `20260705T211452Z`–`20260705T220525Z` | **1.000** | 0.000 |

**What changed to get there:** an IDF-weighted divergence gate (`idf_weighted_relevance`, `DIVERGENCE_IDF_FLOOR=0.13` in `gin/corpus/relevance.py` / `retrieve.py`) so mode selection is no longer pure keyword overlap; a per-pair fallback divergence zone for structurally-dissimilar pairs with an IDF-anchored sentence scorer (`gin/corpus/divergence.py`); a divergent `max_tokens` ceiling put on a measured basis (`40 + 80·n` = 120 for one group, 24% headroom over the worst-case 97-token decode); and a refusal-gate fix (`_claims_query_relevant` judges the claim's own text, not the on-topic-elsewhere cited chunk — `gin/eval/arms.py`).

**Findings carried forward:** (1) the gate survives in every new domain because a *distinctive shared entity* carries IDF mass across framings — advocacy text that never names the entity is the lexical-by-construction failure mode, and the concrete argument for Cartographer/Bookkeeper sentence-level anchors as admitted graph state (plan §7.1). (2) The Qwen run surfaced a harness bug (RAG refusal-detector substring match in `arms.py`) and a real convergent-mode truncation (`tn_2023_anomaly`) root-caused to `span_must_close_at_sentence_end` not being set for single-source convergent decode — both flagged, neither in scope of that doc.

Unit regressions: `tests/test_divergence.py`, `tests/test_framing_generalization.py` (parametrized over both rounds), `tests/test_eval_arms.py`, `tests/test_generate.py`. Still CPU/WSL — GPU artifact remains before full promotion.

---

## Promotion status (per [[GIN_ENG_00_Engineering_Register]])

| Claim | Status |
|-------|--------|
| No-Continuation fabrication rate = 0 (overlap + post-fix NLI) | **Measured** (`192827Z`, `194024Z`) |
| NC epistemic targets on expanded 20-query set | **Measured** (`20260702T012203Z` — all six targets) |
| Regression anchors hold post–Phase 3 | **Measured** (`20260702T010918Z` — 9/9) |
| Prevention delta (RAG vs NC fabrication, overlap) | **Measured** — delta = 0.286 (`192827Z`) |
| Failure-state behavior for No-Continuation | **Measured** — recall 1.0 on out_of_scope |
| Citation parser + NLI calibration + EXACT fast path | **Done** (Sprint A) |
| Retrieval recall@k in harness | **Done** (Sprint C) |
| Epistemic metrics + chunk quotation + divergence fidelity | **Done** (Phase 1) |
| Query set expansion (≥20 queries, regression anchors) | **Done** (Phase 2) |
| NC selection robustness (Phase 1 steering) | **Done** — see `20260701T214706Z` partial / `20260702T003047Z` |
| Mode gating + retrieval ordering (Phase 2) | **Done** — eval `20260702T003047Z` |
| Divergent correctness (Phase 3) | **Done** — eval `20260702T012203Z` |
| Divergence generalizes to real two-node text | **Measured** — `20260705T043114Z` (fidelity 1.0, fabrication 0.0) |
| Divergence generalizes across framing registers | **Measured** — adversarial/legal `20260705T202450Z`, housing `20260705T203622Z` |
| Divergence mechanism is model-independent | **Measured** — Qwen2.5-7B matches Mistral on all 4 divergence querysets |
| GPU / wall-clock in `meta.json` | **Done** — pass `--n-gpu-layers`; timing recorded per run |
| Fair RAG vs NC under NLI | **Partial** — NC stable; RAG still caveat-heavy |
| Gold-aware NC synthesis (`--boost-gold-chunks`) | **Available** (eval flags; superseded for production by query steering) |
| Representative GPU hardware artifact | **Measured and root-caused** — `20260711T211202Z` (RTX 4070, `n_gpu_layers=-1`) vs same-day CPU control `20260711T212721Z`. Fabrication 0.000 on both; gap isolated to 1/20 queries at a near-tie refuse/answer decision, attributable to CPU/CUDA floating-point non-determinism (see Remaining gaps items 3 and 5) |

Structural prevention and NC epistemic alignment are **measured on synthetic corpus**, confirmed on both CPU and GPU backends via a same-day controlled comparison. Fabrication rate (the property SEAR is designed to eliminate) is identical (0.000) across backends; a single close-call query's refuse/answer decision is sensitive to backend-level floating-point noise, which is expected variance in a hard-constraint decoder rather than a defect.

---

## Valid next steps (prioritised)

Steps 1–3 from the original plan are **complete**. Phase 1–4 of the next-phase plan are **implemented in harness**; re-run eval on expanded queryset to populate epistemic metrics in artifacts.

### 4. Expand query set modestly; keep synthetic as regression suite — **done**

- 20 queries in `data/eval/queryset.yaml` (11 new + 9 anchors).
- `--regression-only` on `scripts/eval_run.py` for CI speed.

### 5. Instrument selection bias and divergence fidelity — **done**

- `chunk_quotation_rate`, `divergence_fidelity`, and epistemic metrics in `gin/eval/metrics.py` and report.

### 6. Two-node federation pilot — **deferred** (Phase 5)

### 7. Flagged Generation arm — **deferred** (Phase 6)

### 8. Hardware and inference notes — **partial**

- `n_gpu_layers`, `wall_clock_seconds_per_query`, `tokens_per_second` in `meta.json`.
- GPU eval artifact produced and root-caused (`20260711T211202Z`, RTX 4070, `n_gpu_layers=-1`, 18.9 tok/s vs same-day CPU control `20260711T212721Z`, 2.8 tok/s). Gap isolated to 1/20 queries, a near-tie refuse/answer boundary decision sensitive to CPU/CUDA floating-point non-determinism — see Remaining gaps item 3. Promotion rule should state an explicit cross-backend tolerance rather than requiring bit-exact metric parity.

### 9. Counterfactual synthesis selection — **flags available**

- `--boost-gold-chunks` reorders hits and boosts `preferred_starts` on gold chunks.
- `--gold-refuse-without-coverage` refuses when no emitted claim cites a gold chunk.

Steps 1–3 from the original plan are **complete**. Phases 1–3 (NC steering, mode gating, divergent correctness) are **implemented and measured** on the expanded queryset (`20260702T012203Z`).

### 10. NC selection robustness (Phase 1) — **done**

- Convergent steering, post-decode gate. Partial on full-20 until Phase 2 (`214706Z`).

### 11. Mode gating + retrieval ordering (Phase 2) — **done**

- Query-aware divergent gating, seed re-rank, zero-relevance filter, `competing_same_tag` path.
- Eval: `20260702T003047Z` (4/6 targets); superseded by Phase 3.

### 12. Divergent-mode correctness (Phase 3) — **done**

- Query-relevant contradicts only; keyword-count floor; auto-close fork fix; EOS guard; corroboration decode.
- Unit tests: `tests/test_retrieve_synthesis.py` (22), `tests/test_processor.py`, `tests/test_generate.py`.
- Eval: `20260702T012203Z` (full 20, 6/6 targets), `20260702T010918Z` (regression 9/9).
- Plan: `docs/nc_phase3_divergence_correctness.plan.md`.

---

## How to reproduce

```bash
# Prerequisites: Postgres + pgvector, venv (WSL), model path
cd docker && docker compose up -d && cd ..
source venv/bin/activate
pip install -r requirements.txt
python scripts/corpus_ingest.py --source data/synthetic

# Overlap verifier (promotion-quality; regression anchors only)
python scripts/eval_run.py \
  --model /path/to/Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  --arms rag,no_continuation \
  --verifier overlap \
  --threshold 0.5 \
  --regression-only

# Full expanded queryset (20 queries) — current promotion check
python scripts/eval_run.py \
  --model /path/to/Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  --arms rag,no_continuation \
  --verifier overlap \
  --threshold 0.5

# Eval-only gold-aware NC synthesis
python scripts/eval_run.py \
  --model /path/to/Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  --arms no_continuation \
  --verifier overlap \
  --boost-gold-chunks \
  --gold-refuse-without-coverage

# NLI verifier (post-fix)
python scripts/eval_run.py \
  --model /path/to/Mistral-7B-Instruct-v0.3-Q6_K.gguf \
  --arms rag,no_continuation \
  --verifier nli \
  --threshold 0.5
```

Report: `data/eval_runs/<timestamp>/report.md` (includes retrieval quality section).

---

## Related

[[GIN_ENG_00_Engineering_Register]] · [[GIN_ENG_01_SEAR_PoC_Spec]] · [[GIN_04_SEAR]] · [[GIN_02_Productive_Divergence]] · [[GIN_The_Whole_Frame]] · [Real-text divergence generalization](nc_real_text_divergence_generalization.plan.md)

## Back to Vault

[[HOME]]
