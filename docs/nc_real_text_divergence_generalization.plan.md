# Real-Text Divergence Generalization (two-node corpus)

**Status**: CONFIRMED — measured in full DB eval
**Depends on**: [Phase 3 divergence correctness](nc_phase3_divergence_correctness.plan.md) (PROMOTED on synthetic corpus)
**Baseline**: two-node run `20260704T105554Z` — SEAR `divergence_fidelity` 0.333 (1 of 3 pairs)
**Result run**: `20260705T043114Z` — SEAR `divergence_fidelity` **1.000** (3 of 3 pairs)

Phase 3 promoted the divergent-decode mechanism (`compute_divergence_zones`,
`min_span_len` auto-close, EOS gating) against a synthetic corpus where every
contradicts pair shares a lede and forks at one sentence (e.g. "142" vs "98"
treated). This doc covers the gap Phase 3 didn't know existed: real editorial
divergence — an institutional statistic vs. a grassroots reframing — usually
shares **no** structure at all, and the Phase 3 machinery silently refused on
exactly that case.

---

## 1. Background: how the two-node corpus got here

A second corpus (`corpus_node1.json` / `corpus_node2.json`, institutional vs.
grassroots framing, real fetched text) was built to stress-test whether Phase
3's synthetic-corpus promotion generalizes. Three `contradicts` pairs
(emissions, wildfire, water) were wired via `data/corpus_edges.yaml` and
`data/eval/queryset_twonode.yaml`.

**Retrieval-side gap (prior session, already fixed — see
[[phase3-divergence-correctness]] follow-on).** The `_divergence_relevant`
lexical gate (max_sentence_score + matched_keyword_count) flipped real
minority-side chunks to convergent mode because the grassroots reframing uses
different vocabulary than the query. Fixed with an IDF-weighted relevance gate
(`corpus_idf`, `idf_weighted_relevance` in `gin/corpus/relevance.py`,
`DIVERGENCE_IDF_FLOOR=0.13`) so mode selection is no longer purely
keyword-overlap. This got all three pairs correctly routed to divergent mode
— but exposed the decode-side bug below.

## 2. Root cause: `compute_divergence_zones` returns empty for reframing pairs

**File**: `gin/corpus/divergence.py`

`compute_divergence_zones` marks a divergence point only where **index-aligned
sentences differ AND share >= 3 words** (`_word_overlap`, exact match, no
stemming). This assumes both sides of a contradicts pair share a lede and
fork at a specific sentence — true for synthetic "same story, different
number" pairs, false for real reframing pairs:

| pair | left (institutional) | right (grassroots) | shared words | zone? |
|---|---|---|---|---|
| emissions | "...cuts to predicted 2030 greenhouse gas emissions of roughly 28 percent..." | "...stopped or delayed greenhouse gas pollution equivalent to roughly one-quarter..." | greenhouse, gas, roughly, ... (>=3) | marked |
| wildfire | "56,580 wildfires burned 2,693,910 acres..." | "Elderly, immunocompromised...populations face...risk from wildfire smoke" | only "and" (`wildfire` != `wildfires`, no stemming) | **empty** |
| water | "...snowpack held a snow water equivalent of 61.1 inches..." | "Disadvantaged...communities...affected by water shortages..." | only "water" | **empty** |

With `divergence_starts` empty for a pair, the cascade is fatal
(`materialize_synthesis_bundle` in `gin/corpus/materialize.py` +
`sear/processor.py`):

1. The forbidden-tail net treats every doc-unique sentence as a forbidden
   "tail" (mayor-quote-style filler) **unless** it falls inside a divergence
   zone. With zero zones, this net swallows the pair's own anchor sentences.
2. `_position_start_permitted` (`sear/processor.py`) rejects every span start
   for both docs.
3. `_allowed()` returns `{}` in `BOUNDARY` mode while EOS is blocked
   (`block_eos_until_groups_satisfied`, required because `required_doc_groups`
   is non-empty for a divergent bundle with a contradicts pair) -> all logits
   masked to `-inf` -> decode refuses ("sources do not support an answer").

Emissions survived only because its aligned sentences happen to share >= 3
words. This was never a `min_span_len` or EOS-machinery bug — that
machinery (Phase 3's own fixes) is sound; the zone it depends on was never
being produced upstream.

## 3. Fixes

### Fix 1 — per-pair fallback zone for structurally-dissimilar pairs

**File**: `gin/corpus/divergence.py` (`compute_divergence_zones`)

When the index-aligned >= 3-word-overlap test marks nothing for a
`contradicts` pair, treat the whole chunk on each side as its own divergence
zone (every sentence start in that chunk becomes a divergence-steered start
for that doc). Rationale: in a reframing pair there is no single "fork
sentence" — the divergence *is* the whole framing choice on each side. Only
fires when the aligned test found nothing, so synthetic/aligned pairs
(emissions included) are byte-for-byte unaffected.

### Fix 2 — divergent `max_tokens` budget was tuned for synthetic sentence lengths

**File**: `gin/corpus/generate.py` (`_resolve_decode_params`)

`max_tokens = 40 + 25 * len(required_doc_groups)` gave 65 tokens for one
group. Real institutional sentences run ~50-55 tokens; the first extract
consumed the budget and the second (grassroots) side truncated to a fragment
("Elderly, immunocompromised" instead of the full sentence). Raised to
`40 + 90 * len(required_doc_groups)` (130 for one group) — a ceiling, not a
fixed cost: `stop_when_groups_satisfied` still fires EOS the instant both
sides are quoted, so aligned/synthetic pairs that finish early are unaffected.

## 4. Verification (three layers)

1. **Deterministic repro** (no llama.cpp): whitespace-tokenizer rebuild of the
   real wildfire/water chunk pairs through `compute_divergence_zones` directly.
   Pre-fix: empty zone, both gold anchors `(0,0)`/`(1,0)` forbidden. Post-fix:
   two-sided zone, anchors un-forbidden.
2. **Real Mistral-7B dry run** (hand-built `SynthesisBundle`, no DB): drove the
   actual `ExtractiveCopyConstraint` + `BiasedGINLogitsProcessor` through
   `llm.create_completion` on the wildfire/water pairs. Pre-Fix-2: both sides
   quoted but grassroots side truncated to a fragment. Post-Fix-2: both
   sentences render in full, `quoted_docs=[0,1]`, `groups_satisfied=True`.
3. **Full DB eval** (real Postgres + real retrieval + real model,
   `scripts/eval_run.py --queryset data/eval/queryset_twonode.yaml`): see
   results below.

Unit regression: `tests/test_divergence.py::test_divergence_fallback_for_structurally_dissimilar_pair`
(new). Full suite: 155 passed, 3 skipped (DB-dependent, expected without a
live connection).

## 5. Measured results

Two-node eval (`data/eval/queryset_twonode.yaml`, Mistral-7B-Instruct-v0.3-Q6_K,
overlap verifier):

| Metric | Before (`20260704T105554Z`) | After (`20260705T043114Z`) |
|---|---|---|
| divergence_fidelity (SEAR) | 0.333 | **1.000** |
| fabrication_rate (SEAR) | 0.000 | 0.000 |
| fabrication_rate (RAG) | 0.438 | 0.412 |
| gold_chunk_coverage (SEAR) | 0.600 | 1.000 |
| cross_node_within_ratio (SEAR) | 1.000 | 1.000 |
| wall-clock per query | 34.3s | 51.8s |

All three contradicts pairs (emissions, wildfire, water) now surface both
sides; fabrication stays at zero against RAG's ~41%. Wall-clock roughly +50%
for divergent queries, tracking the larger `max_tokens` ceiling — decode still
terminates on `stop_when_groups_satisfied`, so this is a ceiling increase, not
a per-query fixed cost.

## 6. Open questions / next steps (not yet done)

Ranked by what would most change confidence in this result:

1. **Generalize past 3 pairs.** The whole real-text divergence claim rests on
   three hand-picked pairs from a 19-document corpus. Add more contradicts
   pairs across more topic domains (ideally with a third framing style, not
   just institutional-vs-grassroots) to check the fallback zone and IDF gate
   aren't overfit to this specific corpus's vocabulary distribution.
2. **Stress-test the fallback zone on multi-sentence chunks.** Every chunk in
   the current two-node corpus is a single sentence, so "mark the whole chunk
   as one zone" and "mark every sentence start" are equivalent. On a
   multi-sentence chunk this fallback would mark *every* sentence in both
   docs as divergent, which may pull in irrelevant tail sentences that a
   sharper zone would exclude. Needs a multi-sentence reframing-pair test case.
3. **Investigate `supported_irrelevance_rate` 0.0 -> 0.200 (SEAR).** New in the
   post-fix run; likely tied to `tn_out_of_scope_referendum` (both arms failed
   query_relevance on it) but not yet root-caused. Small in magnitude, not
   zero — worth a follow-up pass before calling this fully clean.
4. **Cross-model check.** Verified only on Mistral-7B-Instruct-v0.3 (Q6_K, CPU,
   temperature 0, greedy). No evidence yet the fix's behavior (or the
   IDF-floor / word-overlap constants) generalizes across model families.
5. **Revisit output fluency.** Divergent answers are two extracted sentences
   joined by a bare `|` delimiter — grounded and citable but not natural
   prose. Inherent to the extractive design (Flagged Generation / Mode 2 is
   the registered-but-unimplemented path for this), not a regression, but
   worth tracking as a product-quality gap separate from correctness.

## 7. Out of scope (this doc)

- RAG arm changes, Flagged Generation arm implementation
- Verifier changes
- Retrieval-side IDF gate implementation detail (see prior-session notes;
  summarized in section 1 only as context for why pairs reach divergent mode
  at all)
