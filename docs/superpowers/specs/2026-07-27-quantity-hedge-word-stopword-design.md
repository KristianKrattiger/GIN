# Design: Strip "roughly" From Measure Tokens

**Date:** 2026-07-27
**Status:** IMPLEMENTED and measured 2026-07-27.
**Sub-project:** G — one of the two false positives sub-project F left as documented, out-of-scope costs
**Predecessor:** `docs/superpowers/specs/2026-07-27-stance-nli-precedence-design.md`
**Labels this rests on:** the same 24 curator labels over `corpus_node5.json`, committed at `ebceb46`. No new labels.

## Problem

Sub-project F's Results section names two residual false positives on the
node5 labels and treats them as one family — "cross-event, pre-existing,
unrelated to that fix." Direct extraction (not assumption) shows they are two
independent defects with two different fixes available:

| pair | mechanism | fixable at |
|---|---|---|
| `n5_doc_023↔024` | a single unstripped hedge-adverb spuriously aligns two unrelated quantities | the stance layer (`quantity.py`) |
| `n5_doc_023↔026` | stage 1's `same_story` is incorrectly `True` (the union/Union collision); 026 states no quantity at all, so there is nothing for any stance-layer fix to act on | stage 1 only (`relatedness.py`) |

Per your scope decision, this spec addresses only `023↔024`. `023↔026` stays
a documented, out-of-scope cost — fixing it would mean reopening the stage-1
anchor territory that regressed two real SEC-register pairs when tried
during sub-project E's design, which is a separate, larger decision.

## Measured evidence

`gin.cartographer.quantity.extract_mentions`, run directly against the real
`corpus_node5.json` text (not assumed from the docstring that flagged this
pair):

```
n5_doc_023: QuantityMention(value=3200.0, unit_class='count',
  measure=frozenset({'involv', 'dockworker', 'action', 'national', 'acros',
  'coordinat', 'part', 'roughly', 'walkout', 'organizer', 'port'}), ...)
n5_doc_024: QuantityMention(value=45.0, unit_class='count',
  measure=frozenset({'morn', 'disrupt', 'official', 'roughly', 'rush',
  'delay', 'commuter', 'minut'}), ...)
```

The two measure sets share exactly one token: `roughly` ("roughly 3,200
dockworkers" vs. "roughly 45 minutes"). Jaccard = 1/18 ≈ 0.056 — above
`ALIGN_FLOOR` (0.05), so the two unrelated quantities align, and the judge
reads a `conflict` (different values, same "measure," compatible scope).
`stance_for` confirms: `evidence_for(text_023, text_024).conflicts` contains
exactly this pair.

`_STOPWORDS` (`gin/cartographer/quantity.py:73-79`) already contains `about`
— an equivalent hedge-adverb that carries no information about *what* is
being measured, only how precisely. `roughly` plays the identical role and
is simply missing from the list. `_content()`, the only function that reads
`_STOPWORDS`, has exactly one call site (`_measure_tokens`, line 199) — scope
extraction (`_scope_tokens`) uses a separate mechanism entirely (`SCOPE_TOKENS`
membership, not `_content`/`_STOPWORDS`), so this change cannot affect scope,
`unit_class`, or the `revised`/`as_of` detection.

## Goal

Remove `roughly` as a measure-token, so two unrelated quantities no longer
align on that word alone. Verified directly, not assumed: once `roughly` is
stripped, `023↔024`'s stance becomes `quantity.UNALIGNED` — both sides still
state a quantity, they simply share no measure token anymore. `UNALIGNED`
resolves to `RELATED_UNTYPED`/`"abstain"` in `classify_relation` regardless of
whether NLI also fires, so the false positive is removed — but via a
different path than `023↔026`. `023↔026` has `stance=None` (026 states no
quantity at all), which still falls into the degenerate
`CONTRADICTS`/`"band"` branch when NLI doesn't independently catch it —
exactly why it stays a distinct, accepted cost rather than becoming a second
instance of a pattern this fix already resolves.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Mechanism | Add `"roughly"` to `_STOPWORDS` | Matches the existing `about` precedent; single call site, so the blast radius is provably limited to measure-token computation |
| Breadth | Exactly one word, not a curated hedge-word set | The measured problem is `roughly`; kin like `around`/`some` have non-hedging uses elsewhere in English and would need their own individual verification against the full label set before being trusted — speculative work for an unconfirmed benefit |
| `ALIGN_FLOOR` | Untouched | The alternative (raise the floor above 0.056) is fragile threshold-placement between two specific measured Jaccard values, not a fix with independent justification |
| `023↔026` | Out of scope, left as a second documented `stance=None` cost | Requires a stage-1 fix, a separate and larger decision explicitly declined for this spec |

## Architecture

One line, one file:

```python
_STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with without as is are was were
be been being has have had said say says it its this that these those they
their there then than not no nor so if while during after before over under
about up out off down more most less least new newly than which who whom whose
will would can could may might must shall should do does did done also very
roughly
""".split())
```

No other file changes. `type_relation`, `CombinedRelationProposer`,
`classify_relation`, `relatedness.py`, and `text_index.py` are all untouched
— this is entirely inside the extraction step that runs before alignment.

## Measurement plan

1. **`scripts/eval_node5_stance.py`** (real models). Predicted: `P_all`
   0.857 → **0.923** (tp 12, fp 1 — `023↔024` no longer typed CONTRADICTS).
   `P` (within-event) and `R` unaffected — both residual false positives are
   cross-event, so within-event precision was already 1.000 and stays there;
   no gold `contradicts` pair is touched by removing a token that appears in
   zero of their measure sets (verified in the next step).
2. **Full suite**, current baseline 744 passed / 16 skipped.
3. **`data/cartographer_thresholds.json`**: byte-identical (`git status --short`).
4. **14-pair escalation bar pin, 45-pair eval set, scan gold eval**: unaffected —
   `_STOPWORDS` changes are upstream of `classify_relation` entirely, and
   these surfaces were already confirmed independent of the stance mechanism
   during sub-project F.
5. **Held-out 40-pair calibration score** via
   `scripts/recalibrate_cheap_pipeline.py --score-only`. Predicted unchanged
   at 0.725, but must be checked rather than assumed: if any of the 9
   same-story pairs in that set has `roughly` in an aligned measure today,
   removing it could change that pair's stance verdict.

## Success criteria

- `P_all` strictly improves; `P` and `R` do not regress.
- The isolated stance-arm test's exact-pinned false-positive set shrinks from
  `{023↔024 (stance), 023↔026 (band)}` to `{023↔026 (band)}` only — re-pinned
  by name, not just by count, matching this track's established testing style.
- No frozen surface regresses.
- `data/cartographer_thresholds.json` byte-identical; no new tunable
  introduced.

## Failure modes

| Condition | Handling |
|---|---|
| Removing `roughly` breaks a currently-correct alignment elsewhere in node5 or the gold set | Investigate before reverting — report which pair and whether `roughly` was load-bearing for a real conflict (if so, this reveals `roughly` sometimes does carry content, which is new information, not a reason to silently special-case it) |
| Held-out-40 calibration score moves | Report the direction and which pair; do not compensate by writing thresholds |
| `023↔024`'s stance does not become `quantity.UNALIGNED` specifically (e.g. it stays `"conflict"`, or drops to `None`) | Stop. If `None`: at least one side's extraction broke, and the pair still resolves to CONTRADICTS via the same degenerate path as `023↔026` — the fix would not actually have worked despite removing the shared token. If still `"conflict"`: some other token is still aligning and the measured evidence above missed it. |

## Testing

Model-free throughout — this needs no model calls.

- **`tests/test_cartographer_quantity.py`**: a new extraction-level test
  pinning that `roughly` does not appear in a measure token set (e.g.
  extracting from a sentence containing "roughly 45 minutes" yields a
  measure set without `"roughly"`), following the file's existing
  `test_extracts_*` naming convention.
- Same file: a stance-level regression test asserting `stance_for` on the
  real `n5_doc_023`/`n5_doc_024` sentences returns `quantity.UNALIGNED`, not
  `"conflict"` — the direct pin of the fix, mirroring the file's existing
  `test_unaligned_when_both_state_quantities_that_do_not_align` (identical
  semantics: both sides state a quantity, none of them align).
- `tests/test_cartographer_stance_node5.py`: update the pinned
  false-positive set (currently asserts exactly `{023↔024: "stance",
  023↔026: "band"}`) to `{023↔026: "band"}` only.
- Regression: full suite.

## Out of scope

- `n5_doc_023↔026` and any stage-1 (`relatedness.py`, `make_same_story`,
  `anchor_tokens`) change — a separate, larger decision.
- Any hedge word other than `roughly`.
- `ALIGN_FLOOR`'s value.
- Recalibrating or writing `data/cartographer_thresholds.json`.

## Open questions

None blocking.

## Results (measured 2026-07-27)

Implemented in commit `8b3e81a` (`quantity.py: strip roughly as a hedge-adverb
from measure tokens`), re-pinned in `f56e237`. Every pre-registered number in
the Measurement plan landed exactly as predicted. The one place the plan left
two possibilities open rather than a single predicted value — which channel
claims the remaining `023↔026` false positive under real models — resolved to
one of the two named options, for a measured reason (below), not a third
outcome. Reported in full below, per this track's practice of recording every
number whichever way it moves.

### End-to-end node5, real models

Real `all-MiniLM-L6-v2` + `nli-deberta-v3-xsmall`, `scripts/eval_node5_stance.py`:

| | `ebceb46` baseline | pre-this-spec (F, shipped) | measured (this spec) |
|---|---|---|---|
| `P` within-event | 0.632 | 1.000 (tp 12, fp 0, fn 0) | **1.000** (tp 12, fp 0, fn 0) |
| `R` recall | 1.000 | 1.000 | **1.000** |
| `P_all` incl. cross-event | 0.500 | 0.857 (tp 12, fp 2) | **0.923** (tp 12, fp 1) |

Matches this spec's own prediction exactly: `P_all` 0.857 → 0.923 (tp 12, fp 1
— `023↔024` no longer typed CONTRADICTS), `P` and `R` unaffected at 1.000.
Script's own verdict line:

```
  pre-registered bar: PASS  (P and P_all both improve, R >= 0.75)
```

The script always prints "P and P_all both improve" as its generic bar
description; here only `P_all` actually moves — `P` was already at its
ceiling of 1.000 coming in from sub-project F, and this spec's own Success
criteria only required `P` and `R` not to regress, not to improve further.
The bar passes on its real condition: `P_all` strictly improves, `R` holds at
1.000.

Dev/held-out split, unaffected and unchanged from sub-project F: both halves
stay at `P` 1.000 / `R` 1.000, gap +0.000. This spec's Measurement plan didn't
register a prediction for this split (F already closed the gap; this fix
doesn't touch either pair that split turned on), so it's reported here for
completeness, not as a pre-registered number.

### False positives by channel

Predicted: only `n5_doc_023:0<->n5_doc_026:0` remaining, channel `band` or
`nli` depending on which fires first with real models (the design explicitly
left this open rather than picking one). Measured, verbatim:

```
  nli      gold=unrelated    p_contra=0.692 cos=0.261 stance=None  n5_doc_023:0 <-> n5_doc_026:0
  totals: {'nli': 1}
```

Channel is `nli`. Real `p_contra` (0.692) clears `contra_threshold` (0.686,
per the shipped thresholds printed by the calibration run below) by a slim
0.006 margin, so `nli_contradicts` is `True`; since `stance=None` makes
`stance_disagrees` `False` by construction (`combined.py`'s guard requires
`stance is not None`), `classify_relation`'s first branch returns
`CONTRADICTS, "nli"` before the pair ever reaches the `stance is None ->
"band"` fallback. The isolated test (`tests/test_cartographer_stance_node5.py`,
injected `p_contra=0.05`) never clears that threshold, so it correctly
continues to pin this same pair as `"band"` — the two tests measure different
things (isolated stance arm vs. real end-to-end with real models), they are
not disagreeing with each other. `023↔024` no longer appears anywhere in this
list, which is the fix working. Total false positives: 1, down from 2 (F's
measurement) — an exact match to the prediction's `tp 12, fp 1`.

### Held-out 40-pair calibration score

`scripts/recalibrate_cheap_pipeline.py --score-only`, verbatim:

```
samples: 150 {'related_untyped': 62, 'unrelated': 26, 'corroborates': 28, 'contradicts': 34}
same_story corpus: 274 docs, df_ceiling 9
shipped thresholds: Thresholds(gate_floor=0.14, corroborate_ceiling=0.486, contra_threshold=0.686)
held-out (40 eval pairs, never calibrated on) accuracy   0.725
```

Predicted unchanged at 0.725. Measured: **0.725**. Exact match — confirms,
rather than assumes, that none of the 9 same-story pairs in that 40-pair
held-out set had `roughly` in an aligned measure that this change touches.

### Frozen surfaces (from Task 2)

Task 2 measured these; final state at commit `15b0812` (a pure docstring/
comment cleanup on top of `f56e237`, the commit that actually produced the
counts below — no test logic changed in between):

- Full suite: **746 passed / 16 skipped** (baseline 744 passed / 16 skipped
  before Task 1's 2 new test cases).
- `data/cartographer_thresholds.json`: byte-identical (`git status --short`
  empty).
- 45-pair eval set, 14-pair escalation bar pin, and scan gold eval
  (`tests/test_cartographer_eval_pairs.py` + `tests/test_cartographer_scan_gold.py`
  + `tests/test_scan_precision.py`): **26 passed** (7 + 8 + 11), unchanged.
- Isolated stance-arm test (`tests/test_cartographer_stance_node5.py`):
  **3 passed**, re-pinned false-positive set now
  `{n5_doc_023:0<->n5_doc_026:0: "band"}` only — `023↔024` removed from the
  set entirely, as predicted.

### Deviations from prediction

None on any measured number. `P`, `R`, `P_all`, the false-positive count, and
the held-out-40 score all landed exactly on the values this spec's Measured
evidence and Measurement plan predicted. The only item the plan explicitly
left as an open branch rather than a single predicted value — whether the
real end-to-end run's remaining false positive types via `band` or `nli` —
resolved to `nli`, one of the two anticipated outcomes, for the reason
measured above (real `p_contra` clears `contra_threshold` by a 0.006 margin
before the `band` fallback would otherwise be reached). `023↔026` stays
exactly what this spec's own "Out of scope" section said it would: a
documented, un-fixed stage-1 cost.
