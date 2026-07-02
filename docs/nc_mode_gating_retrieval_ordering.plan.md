# Phase 2: NC Mode Gating + Retrieval Ordering

**Status**: PROMOTED — superseded by Phase 3 for remaining epistemic gaps
**Depends on**: Phase 1 (NC selection robustness) — committed at `7d4ad06`
**Baseline runs**: `20260701T214706Z` (full 20, Phase 1 only), `20260701T220243Z` (regression 9)
**Phase 2 eval**: `20260702T003047Z` (full 20 — 4/6 epistemic targets)
**Final promotion**: `20260702T012203Z` (full 20, 6/6) after Phase 3 fixes in `docs/nc_phase3_divergence_correctness.plan.md`

---

## 1. Confirmed Root Causes

### Problem A — False divergent mode (highest leverage)

**Location**: `gin/corpus/retrieve.py:147-149` (`_is_ambiguous`)

```python
if any(e.edge_type == "contradicts" for e in edges):
    return True
```

Any `contradicts` edge among seed hits forces divergent mode, even when the
contradicting pair is query-irrelevant.

**Evidence** (run `214706Z`):

| Query | Retrieved (top 3) | Gold chunk | Mode | Output |
|---|---|---|---|---|
| `port_cargo_throughput` | election_centralwire:0, election_metrodaily:0, port_authority_brief:0 | port_authority_brief:0 | divergent (false) | election turnout spans |
| `school_enrollment_fall` | election_metrodaily:0, election_centralwire:0, school_district_report:0 | school_district_report:0 | divergent (false) | election turnout spans |
| `housing_permits_issued` | incident_rp:0, incident_cw:0, election_cw:0, incident_md:0, election_md:0, housing_permits_office:0 | housing_permits_office:0 | divergent (false) | incident+election weave |

The election chunks have a `contradicts` edge between them. This edge fires
for *every* query that retrieves both election chunks, regardless of whether
the query is about elections.

### Problem B — Hit ordering before mode detection

**Location**: `gin/corpus/retrieve.py:267-313` (`retrieve_for_synthesis`)

`rerank_hits_by_query_score()` runs in `materialize_synthesis_bundle()` *after*
`retrieve_for_synthesis()` has already set `bundle.mode` and built `bundle.pairs`
using RRF order. Mode detection and pair building use RRF order, not query-
sentence score order.

Incident/election chunks often RRF-rank 1-3 despite the gold chunk being
present (recall@k = 1.0 for all failing queries).

**Evidence**: `weather_winds` retrieves incident chunks at ranks 1-3,
`weather_service_brief:0` at rank 4. RRF ordering + contradicts edge →
divergent mode → incident spans → post-decode gate refuses → query_relevance
failure.

Same pattern for `reservoir_storage_level` and `outage_restoration_time`.

### Problem C — Convergent steering cannot overcome noise-first ordering

Even when mode is correctly convergent, `score_starts_for_convergent()` steers
to the top query-scored doc. But if the gold chunk is at position 3+ in the
hit list and incident/election chunks dominate the first 2 positions, the
decoder may still prefer shared-prefix noise from those early docs.

**Affected**: `weather_winds`, `reservoir_storage_level`, `outage_restoration_time`
(after A+B fix, these become convergent but gold chunk must be promoted).

### Problem D — Counterfactual regression

`counterfactual_adherence` checks whether `counterfactual_answer` appears in a
SUPPORTED claim (`gin/eval/metrics.py:131-145`).

**Evidence** (run `214706Z`):

| Query | Output | Expected | Problem |
|---|---|---|---|
| `unemployment_rate` | "stood at 3" (truncated) + inflation + wage | "3.7 percent" | `stop_after_first_extract=True` truncates before "3.7 percent" completes |
| `wage_growth_rate` | inflation + unemployment spans | "4.8 percent" | Top-doc steering picks wrong doc from mixed bureau bundle |
| `export_decline_rate` | "fell 3.2 percent" (correct) + unemployment | "3.2 percent" | Passes adherence, but irrelevant tail |
| `inflation_rate` | "rose 2" (truncated) + unemployment | "2.1 percent" | Same truncation pattern |

Causes:
1. `stop_after_first_extract=True` for convergent mode stops after ~3-8 tokens
   of the first span, truncating "3.7 percent" to "3"
2. Top-doc steering picks wrong doc among multiple same-topic bureau chunks
3. Without re-rank, labor/inflation/wage bureau chunks compete for position 0

---

## 2. Promotion Targets

| Metric | Target | Current (214706Z) | Gap |
|---|---|---|---|
| `query_relevance_rate` | >= 0.90 | 0.700 | 6 queries failing |
| `supported_irrelevance_rate` | <= 0.05 | 0.387 | Need to eliminate off-topic SUPPORTED claims |
| `gold_chunk_coverage` | >= 0.75 | 0.533 | Gold chunks not being selected |
| `counterfactual_adherence` | >= 0.90 | 0.250 | 3/4 counterfactual queries broken |
| `fabrication_rate` | 0.000 | 0.000 | Must not regress |
| `divergence_fidelity` | 1.0 | 1.000 | Must preserve |

---

## 3. Implementation Design

### Phase A — Query-relevant divergent gating

**File**: `gin/corpus/retrieve.py`

**Change**: Replace `_is_ambiguous()` lines 147-149 with query-aware logic.

```python
DIVERGENCE_RELEVANCE_FLOOR = 0.15

def _is_ambiguous(
    seed_hits: list[ChunkHit],
    edges: list[EdgeRecord],
    query: str = "",
) -> bool:
    contradicts_edges = [e for e in edges if e.edge_type == "contradicts"]
    if contradicts_edges and query:
        # Only enter divergent mode when a contradicts pair is query-relevant
        from .relevance import max_sentence_score
        hit_by_id = {h.chunk_id: h for h in seed_hits}
        has_relevant_pair = False
        for edge in contradicts_edges:
            left = hit_by_id.get(edge.src_chunk_id)
            right = hit_by_id.get(edge.dst_chunk_id)
            if left is None or right is None:
                continue
            left_score = max_sentence_score(left.text, query)
            right_score = max_sentence_score(right.text, query)
            if (left_score >= DIVERGENCE_RELEVANCE_FLOOR
                    and right_score >= DIVERGENCE_RELEVANCE_FLOOR):
                has_relevant_pair = True
                break
        if has_relevant_pair:
            return True
    elif contradicts_edges and not query:
        # Legacy path: no query available, fall back to blunt check
        return True
    # ... rest of close-competitors logic unchanged ...
```

**Also change `_build_pairs`**: Filter out pairs where neither chunk is
query-relevant. This prevents `required_doc_groups` from including
query-irrelevant contradicts pairs.

```python
def _build_pairs(
    hits_by_id: dict[str, ChunkHit],
    edges: list[EdgeRecord],
    query: str = "",
) -> list[tuple[ChunkHit, ChunkHit, EdgeRecord]]:
    pairs = []
    for edge in edges:
        if edge.edge_type not in ("contradicts", "cites"):
            continue
        left = hits_by_id.get(edge.src_chunk_id)
        right = hits_by_id.get(edge.dst_chunk_id)
        if left is None or right is None:
            continue
        if edge.edge_type == "contradicts" and query:
            from .relevance import max_sentence_score
            if (max_sentence_score(left.text, query) < DIVERGENCE_RELEVANCE_FLOOR
                    and max_sentence_score(right.text, query) < DIVERGENCE_RELEVANCE_FLOOR):
                continue  # skip query-irrelevant contradicts pair
        pairs.append((left, right, edge))
    return pairs
```

**Regression safety**: When `query == ""` (no query context), the old behavior
is preserved exactly. Incident/election queries that *are* about incidents or
elections will still score both chunks >= 0.15 and enter divergent mode.

**Expected impact**: Fixes `port_cargo_throughput`, `school_enrollment_fall`,
`housing_permits_issued` — these will become convergent instead of false-divergent.

### Phase B — Re-rank seed hits before mode detection

**File**: `gin/corpus/retrieve.py` (`retrieve_for_synthesis`)

**Change**: Accept `query` parameter (already the first positional arg).
After RRF retrieval and relevance floor, re-rank seed hits by query-sentence
score before edge fetch and mode detection.

```python
def retrieve_for_synthesis(
    query: str,
    *,
    k_seed: int = 5,
    k_max: int = 6,
    filters: Optional[dict[str, Any]] = None,
    min_rrf_delta: float = DEFAULT_MIN_RRF_DELTA,
    confidence_floor: float = RETRIEVAL_CONFIDENCE_FLOOR,
) -> SynthesisBundle:
    seed_hits = retrieve(query, k=k_seed, filters=filters)
    # ... confidence floor checks ...
    seed_hits = _apply_relevance_floor(seed_hits, min_rrf_delta)

    # NEW: re-rank by query relevance before mode detection
    from .relevance import rerank_hits_by_query_score
    seed_hits = rerank_hits_by_query_score(seed_hits, query)

    # ... rest of edge fetch, _is_ambiguous(seed_hits, edges, query), etc.
```

**Also**: Pass `query` to `_is_ambiguous()` and `_build_pairs()`.

**Dedup with materialize**: `materialize_synthesis_bundle()` currently calls
`rerank_hits_by_query_score()` again. After this change, the re-rank in
`materialize_synthesis_bundle` becomes a no-op for seeds (already re-ranked)
but still applies to neighbor-expanded hits. Keep it as a secondary pass.

**Expected impact**: Gold chunks promoted in seed order. For `weather_winds`,
`weather_service_brief:0` moves from rank 4 to rank 1, incident chunks drop.
Mode detection now sees query-relevant chunks first.

### Phase C — Filter zero-relevance seeds

**File**: `gin/corpus/retrieve.py` (`retrieve_for_synthesis`)

After re-rank, optionally drop seeds with query score = 0 (completely
irrelevant to the query) unless needed for pair completeness:

```python
from .relevance import max_sentence_score

# Drop seeds with zero query relevance
scored_seeds = [(h, max_sentence_score(h.text, query)) for h in seed_hits]
relevant_seeds = [h for h, s in scored_seeds if s > 0]
if relevant_seeds:
    seed_hits = relevant_seeds
# If all seeds score 0, keep them all (better than empty)
```

This is a lightweight additional filter. It ensures that if the gold chunk
is the only query-relevant hit, it becomes the sole seed and gets convergent
mode with single-doc focus.

**Constraint**: Only filter *before* edge expansion. If an edge links to a
filtered-out seed, the neighbor fetch still finds it.

### Phase D — Counterfactual decode path

**Root cause analysis**:
- `stop_after_first_extract=True` for convergent mode stops the decoder
  after the first extractive span (often 3-8 tokens)
- Counterfactual answers like "3.7 percent" need the full sentence
- Bureau chunks for the same economic topic (labor, inflation, wage) share
  vocabulary, causing wrong-doc selection

**Design choice**: Disable `stop_after_first_extract` when counterfactual
bureau chunks are retrieved and the bundle is convergent with multiple
same-topic chunks.

**Implementation approach**: Use a production-safe heuristic — when the top-2
query-scored seed hits are from different outlets but the same eval_tag
(indicating competing bureau reports), allow 2 spans instead of 1.

**File**: `gin/corpus/generate.py` (`_resolve_decode_params`)

```python
def _resolve_decode_params(
    bundle: SynthesisBundle,
    ctx: SynthesisContext,
    *,
    require_cites: bool,
    stop_when_satisfied: bool,
    min_span_len: Optional[int],
    max_tokens: Optional[int],
) -> dict[str, Any]:
    divergent = bundle.mode == "divergent"
    # ...existing divergent logic...
    stop_after_first_extract = not divergent

    # For convergent bundles with competing same-tag sources,
    # allow a longer first span to avoid truncating numeric answers
    if not divergent and max_tokens is None:
        max_tokens = 60
        # Check if top-2 hits share eval_tag but differ in outlet
        if len(bundle.hits) >= 2:
            h0, h1 = bundle.hits[0], bundle.hits[1]
            if (h0.eval_tag and h0.eval_tag == h1.eval_tag
                    and h0.outlet != h1.outlet):
                max_tokens = 100
                stop_after_first_extract = False
    # ...
```

**Alternative considered**: Passing `eval_layer == counterfactual` from the
eval harness — rejected because this uses gold information in the production
path. The same-tag + different-outlet heuristic is production-safe.

**Expected impact**: `unemployment_rate` gets "3.7 percent" in full;
`wage_growth_rate` gets "4.8 percent"; `inflation_rate` gets "2.1 percent".

### Phase E — Post-decode gate refinement (conditional)

After Phases A-D, re-evaluate whether the gate in `gin/eval/arms.py`
(`_claims_query_relevant`) still causes false refusals. The gate was
correctly refusing off-topic output that arose from false-divergent mode.
With A+B fixing mode detection, these queries should produce on-topic output
that passes the gate naturally.

**Decision**: Implement only if eval run after A-D shows remaining gate
false-positives. No preemptive changes.

---

## 4. API Changes

### `gin/corpus/retrieve.py`

| Function | Change | Backward compatible? |
|---|---|---|
| `_is_ambiguous(seed_hits, edges)` | Add `query: str = ""` parameter | Yes (default empty) |
| `_build_pairs(hits_by_id, edges)` | Add `query: str = ""` parameter | Yes (default empty) |
| `retrieve_for_synthesis(query, ...)` | Internal re-rank before mode detection; pass query to `_is_ambiguous` and `_build_pairs` | Yes (no signature change) |

### `gin/corpus/generate.py`

| Function | Change | Backward compatible? |
|---|---|---|
| `_resolve_decode_params(...)` | Heuristic for competing same-tag convergent bundles | Yes |

### New constant

| Module | Name | Value | Purpose |
|---|---|---|---|
| `gin.corpus.retrieve` | `DIVERGENCE_RELEVANCE_FLOOR` | `0.15` | Min query-sentence score for a contradicts chunk to trigger divergent mode |

---

## 5. Test Matrix

### Unit tests (no llama.cpp required)

| Test | File | Verifies |
|---|---|---|
| Port query + election contradicts in seed → convergent mode | `tests/test_retrieve_synthesis.py` | Phase A |
| Incident query + incident contradicts in seed → divergent mode (preserved) | `tests/test_retrieve_synthesis.py` | Phase A regression |
| Election query + election contradicts → divergent (preserved) | `tests/test_retrieve_synthesis.py` | Phase A regression |
| `_build_pairs` excludes query-irrelevant contradicts | `tests/test_retrieve_synthesis.py` | Phase A |
| Seed re-rank puts port chunk first for port query | `tests/test_retrieve_synthesis.py` | Phase B |
| Seed re-rank puts weather chunk first for weather query | `tests/test_retrieve_synthesis.py` | Phase B |
| Zero-relevance filter drops incident for port query | `tests/test_retrieve_synthesis.py` | Phase C |
| Competing same-tag bureau → stop_after_first_extract=False | `tests/test_generate.py` (new) | Phase D |
| Convergent preferred_starts align with re-ranked doc 0 | `tests/test_materialize_steering.py` | Phase B+C |
| Divergent probes unchanged (existing tests) | `tests/test_materialize_steering.py` | Regression |

### Integration eval (user runs manually)

```bash
# Regression anchors first
python scripts/eval_run.py --model <gguf> --arms no_continuation --verifier overlap --regression-only

# Full 20-query run
python scripts/eval_run.py --model <gguf> --arms rag,no_continuation --verifier overlap
```

**Success criteria**: Full 20-query NC meets all promotion targets; regression
anchors hold.

---

## 6. Regression Anchors — Must Not Break

| Query | Current behavior | Constraint |
|---|---|---|
| `incident_hospital` | divergent, fidelity 1.0 | Incident contradicts must still trigger divergent |
| `incident_arrests` | divergent, fidelity 1.0 | Same |
| `election_margin` | divergent, fidelity 1.0 | Election contradicts must still trigger divergent |
| `election_turnout` | divergent, fidelity 1.0 | Same |
| `transit_ridership` | convergent, passes | Must remain convergent |
| `interest_rate_probe` | out_of_scope refusal | Must refuse |
| `sports_probe` | out_of_scope refusal | Must refuse |

### Why regression anchors are safe

- Incident queries: "incident", "hospital", "treatment", "arrests" all appear
  in incident chunk text → both chunks score >= 0.15 → divergent mode preserved
- Election queries: "harbor", "district", "referendum", "votes", "turnout" all
  appear in election chunk text → both chunks score >= 0.15 → divergent mode preserved
- Port query: "cargo", "throughput", "twenty-foot", "TEU" do NOT appear in
  election chunk text → election contradicts pair scores < 0.15 → convergent

---

## 7. Implementation Order

```
Phase A (query-relevant divergent gating)
    ↓
Phase B (re-rank before mode detection)
    ↓
Phase C (zero-relevance seed filter)
    ├── Phase D (counterfactual decode path)
    ↓
Phase F (tests)
    ↓
Phase E (gate refinement — only if needed after eval)
```

Start with A+B together (directly addresses 6 of 6 failing realism queries).
Then D (fixes 3-4 counterfactual queries). Run tests, then eval.

---

## 8. Out of Scope

- RAG arm fixes
- Federation pilot
- Flagged Generation arm
- Complementary multi-doc convergent weave
- Embedding model / re-ingest
- Any change to the eval queryset or corpus data
