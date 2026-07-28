# Design: Strip "roughly" From Measure Tokens

**Date:** 2026-07-27
**Status:** Proposed
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
