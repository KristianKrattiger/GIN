# Phase 3: Divergent-Mode Correctness (mode gating v2 + span auto-close)

**Status**: PROMOTED — all 6 targets met
**Depends on**: Phase 2 (query-aware gating + retrieval ordering) — working tree, eval run `20260702T003047Z`
**Baseline**: full-20 `20260702T003047Z` (4/6 targets), regression `20260701T224945Z` (9/9)
**Result runs**: full-20 `20260702T012203Z` (6/6 targets), regression `20260702T010918Z` (9/9 anchors hold)

**Unit tests** (all passing): `tests/test_retrieve_synthesis.py` (Phase 3A–3B), `tests/test_processor.py` (`test_no_auto_close_after_early_fork`, `test_auto_close_at_fork_respects_sentence_end`), `tests/test_generate.py` (`competing_same_tag`).

Phase 3 outcome vs targets (no_continuation, full-20):
query_relevance 1.000, supported_irrelevance 0.000, gold_coverage 1.000,
counterfactual_adherence 1.000, fabrication 0.000, divergence_fidelity 1.000.
counterfactual cross_node_within_ratio reads 0.000 by design — corroborated
answers are single AMBIGUOUS spans citing both bureau + independent docs
(different outlets → cross-node); 0 violations, not a promotion target.

---

## 1. Corrected Root-Cause Analysis

The Phase 2 wrap-up attributed the remaining failures to (a) a constrained-decode
tokenization boundary ("3" vs ".7") and (b) a keyword collision with mode
"correctly convergent". Inspection of run `20260702T003047Z` raw outputs
disproves both. All four remaining failures share one root cause chain.

### Evidence: the failing queries ran DIVERGENT, not convergent

Raw output for `unemployment_rate`:

```
 The regional unemployment rate stood at 3 [1] [2] The regional consumer price
 index rose 2.1 percent over the past twelve months.
 | [4] The regional consumer price
```

- `[n]` cite markers ⇒ `require_cites=True`, which is only set for divergent
  decode (`gin/corpus/generate.py:50`, `require_cites = require_cites or divergent`).
- `|` delimiters between spans, multi-doc AMBIGUOUS first claim, spans drawn
  from non-top docs ⇒ `focus_doc_indices` was all-docs, i.e. divergent.
- Therefore the Phase D `competing_same_tag` heuristic **never fired** for
  these queries — it is gated on `not divergent`.

Same signature on `school_enrollment_fall` (`Turnout reached 61 percent of [1]
Turnout reached 58 percent of registered voters. |`) — a classic divergent
both-sides render. The claim "mode is correctly convergent" was wrong.

### Root cause 1 — close-competitor branch of `_is_ambiguous` misfires on corroboration

`gin/corpus/retrieve.py:165-188`. The counterfactual bundles retrieve
bureau + independent-survey chunks: **different outlets, adjacent RRF scores,
identical agreeing text**. The close-competitor branch reads that as
divergence. The 0.15 relevance floor doesn't help: inflation chunks score
0.167 on the unemployment query via the single shared word "regional";
wage chunks score 0.33 via "latest survey".

Corroboration (same numbers, multiple outlets) is the *convergent* success
case, not divergence. No contradicts edge exists among the econ chunks.

### Root cause 2 — `_maybe_auto_close_on_divergence` truncates at exactly `min_span_len`

`sear/processor.py:385-394`. In divergent mode (`close_on_doc_divergence=True`)
it closes the span as soon as `span_len >= min_span_len` (8) **if the cursor
set ever narrowed since span start** — comparing `current_docs` against
`_span_start_docs`, not against the previous step. It also calls
`_close_span()` directly, bypassing `_span_close_permitted()` and the
`span_must_close_at_sentence_end` guarantee.

Proof by contrast within the same run:

| Query | Span start prefix | Docs at start | Narrows? | Result |
|---|---|---|---|---|
| `unemployment_rate` | "The regional" (shared by labor×2 + inflation×2) | 4 | at "unemployment" | truncated at token 8: "…stood at 3" |
| `inflation_rate` | "The regional" | 4 | at "consumer" | truncated: "…rose 2" |
| `school_enrollment_fall` | "Turnout reached " (both election docs) | 2 | at "61"/"58" | truncated at token ~8: "…percent of" |
| `export_decline_rate` | "Regional export volumes" (unique prefix) | 2 | never | **full sentence — PASSES** |

The "3" vs "3.7" pattern is not a tokenization issue; the close lands wherever
token 8 happens to fall.

### Root cause 3 — EOS deadlock when divergent mode has no contradicts pairs

The econ bundles have no contradicts edges ⇒ `required_doc_groups` is empty ⇒
`_groups_satisfied()` returns **False** when there are no groups
(`sear/processor.py:279-282`) ⇒ `block_eos_until_groups_satisfied` blocks EOS
forever ⇒ the decoder rambles off-topic spans until `max_tokens`. This is the
direct driver of `supported_irrelevance_rate` (0.138): the unemployment query
emits inflation spans, the wage query emits inflation + unemployment spans.

### Root cause 4 — `school_enrollment_fall` divergence floor + pair-filter asymmetry

Both election chunks share the wire-copy sentence "The harbor district
referendum passed by 842 votes…", so **both** sides of the election
contradicts pair score 0.20 ≥ 0.15 on "district" (1 of 5 query keywords) and
`_is_ambiguous` returns divergent. Separately, `_build_pairs` keeps a pair
unless **both** sides are below the floor, while `_is_ambiguous` requires
**both** above — an inconsistency that lets one-sided-relevant pairs get
front-loaded by `_prioritize_hits`.

---

## 2. Fixes

### 3A — Divergent mode requires a query-relevant contradicts pair

**File**: `gin/corpus/retrieve.py` (`_is_ambiguous`)

Remove the close-competitor fallback branch (or keep it only for the legacy
`query == ""` path). Divergent mode ⇔ at least one contradicts pair whose
**both** endpoints are query-relevant. Multi-outlet agreement without a
contradicts edge is corroboration → convergent.

Consequences:
- All four counterfactual queries become convergent → `competing_same_tag`
  heuristic finally fires → Phase D path actually exercised.
- Divergent ⇒ pairs exist ⇒ `required_doc_groups` non-empty ⇒ root cause 3
  cannot trigger (still add the defensive guard in 3D).

### 3B — Minimum matched-keyword count for divergence relevance

**Files**: `gin/corpus/relevance.py`, `gin/corpus/retrieve.py`

Add `matched_keyword_count(text, query) -> int` (max over sentences). A
contradicts endpoint counts as query-relevant only if
`max_sentence_score >= DIVERGENCE_RELEVANCE_FLOOR` **and**
`matched_keyword_count >= 2` (when the query has ≥ 3 keywords).

- Fixes `school_enrollment_fall`: "district" alone (1 of 5 keywords) no
  longer qualifies → election pair irrelevant → convergent → top-doc steering
  picks school doc (sentence score 1.0).
- Anchors safe: election queries match election chunks on 3+ keywords
  (harbor, district, referendum, turnout, votes); incident queries likewise.

Also align `_build_pairs` with `_is_ambiguous`: drop a contradicts pair unless
**both** sides pass the same relevance test (currently `and` → keeps
one-sided pairs, which `_prioritize_hits` then front-loads).

### 3C — Fix auto-close: fire at the fork, never bypass sentence-end

**File**: `sear/processor.py` (`_maybe_auto_close_on_divergence`)

- Track the previous step's cursor doc-set; fire only when narrowing occurred
  **at this token**, not "ever since span start".
- Respect `_span_close_permitted()` (min length + sentence-end) instead of
  calling `_close_span()` directly; if close is not yet permitted, let the
  span continue on the surviving cursors.

Result: spans that start on a shared prefix and fork early (fork before
`min_span_len`) simply continue single-doc to sentence end. Genuine divergence
attribution still works because divergence-zone steering places starts inside
per-doc divergence zones.

### 3D — EOS-deadlock guard

**File**: `gin/corpus/generate.py` (`_resolve_decode_params`)

Set `block_eos` / `stop_when_satisfied` only when the bundle actually has
contradicts groups (`ctx.required_doc_groups` non-empty), so a divergent
bundle without groups can terminate. Belt-and-braces with 3A.

### 3E — Corroboration path: cover both gold chunks

**File**: `gin/corpus/generate.py`

Risk introduced by 3A: counterfactual queries have **two** gold chunks
(bureau + independent). Convergent steered decode uses
`focus_docs = {top_doc_idx}` and `allow_shared_prefix=False`, which would
yield single-doc attribution → `gold_chunk_coverage` 0.5 per query.

For `competing_same_tag` bundles: widen `focus_docs` to all docs sharing the
top hit's `eval_tag`, and keep `allow_shared_prefix=True`, so the shared
sentence closes as one AMBIGUOUS span citing both docs (semantically correct:
a corroborated claim). Optionally enable `span_must_close_at_sentence_end`
for convergent decode too, killing residual fragments like "The regional
consumer" (3-token SUPPORTED claims).

---

## 3. Test Matrix (unit, no llama.cpp)

| Test | Verifies |
|---|---|
| Bureau+survey bundle (no contradicts, close RRF, multi-outlet) → convergent | 3A |
| Election contradicts + election query → divergent (anchor) | 3A regression |
| Election contradicts + school query (1 shared keyword) → convergent | 3B |
| `_build_pairs` drops pair when only one side relevant | 3B |
| State-machine replay: span starting on 4 docs, forking at token 3, min_span_len 8 → runs to sentence end | 3C |
| State-machine replay: fork after min_span_len → closes at fork only if sentence-end permitted | 3C |
| Divergent params with empty groups → eos not blocked | 3D |
| competing_same_tag → focus includes both same-tag docs; shared span cites both | 3E |
| Existing anchors: incident/election divergent fidelity tests | regression |

Then: regression eval (9 anchors), full-20 eval.

## 4. Expected Metric Impact

| Metric | 003047Z | Expected | Mechanism |
|---|---|---|---|
| counterfactual_adherence | 0.250 | 1.0 | convergent corroboration path; full sentences (export already proves the happy path) |
| supported_irrelevance_rate | 0.138 | ~0 | no EOS deadlock; focus restricted to same-tag docs |
| query_relevance_rate | 0.950 | 1.0 | school query convergent → school doc selected |
| gold_chunk_coverage | 0.867 | ≥ 0.867 | 3E preserves dual-gold citation via AMBIGUOUS spans |
| fabrication_rate | 0.000 | 0.000 | unchanged — masking untouched |
| divergence_fidelity | 1.000 | 1.000 | anchors keep contradicts-edge divergence; 3C verified by replay tests |

## 5. Out of Scope

- RAG arm, Flagged Generation arm, federation
- Verifier changes (fragment claims scored 1.0 — worth a min-length floor later)
- Corpus/queryset changes
