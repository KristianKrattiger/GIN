# Design: Stance Channel Outranks NLI on Same-Story Disagreement

**Date:** 2026-07-27
**Status:** Proposed
**Sub-project:** F — the deferred decision from sub-project E
**Predecessor:** `docs/superpowers/specs/2026-07-26-same-story-stance-channel-design.md`
**Labels this rests on:** the same 24 curator labels over `corpus_node5.json`,
committed at `ebceb46`. No new labels.

## Problem

Sub-project E built a fact-aligned stance channel and gated the same-story
CONTRADICTS branch on it, but deliberately left the NLI channel's priority
over that branch untouched: *"the spec fixed NLI's priority and moving it is
a separate decision."* E's own Results section measured the cost of that
choice and named it as the sharpest open question:

> 3 of 4 end-to-end false positives are the PRE-EXISTING NLI channel, not the
> new stance arm — and 2 of those are exactly the pairs the spec's own table
> identified as NLI's highest `p_contra` in the set when concluding NLI can't
> carry this branch.

This spec makes that decision.

## Measured evidence

Real `all-MiniLM-L6-v2` + `nli-deberta-v3-xsmall`, `scripts/eval_node5_stance.py`,
re-run during this design (2026-07-27) to check one specific thing the original
spec didn't isolate: **whether any gold `contradicts` pair currently relies on
NLI winning against a disagreeing stance.** It does not. Every `channel=nli`
pair in the 24 labels falls into exactly one of two buckets:

| gold | stance | NLI fires? | outcome | agreement? |
|---|---|---|---|---|
| contradicts (`coastal_storm_landfall`) | `conflict` | yes | correct | stance agrees |
| contradicts (`harbor_district_referendum`) | `conflict` | yes | correct | stance agrees |
| supersedes (`007↔008`) | `revision` | yes | **wrong** | stance disagrees |
| corroborates (`036↔037`) | `unaligned` | yes | **wrong** | stance disagrees |

Zero exceptions either direction: whenever NLI fires and stance has a decisive
verdict, "stance agrees" perfectly predicts "NLI is right" and "stance
disagrees" perfectly predicts "NLI is wrong." Both wrong cases are same-story
pairs (`same_story=True`); one is a **dev** event (`northgate_hospital_outbreak`)
and one is **held-out** (`stadium_capacity_ruling`), so the pattern isn't purely
dev-fitted. The other two residual false positives (`023↔024`, `023↔026`) both
have `stance` uninvolved in the disagreement sense — one is a stance-channel
false positive in agreement with NLI, one has `stance=None` — and neither is
touched by this change (see Out of scope).

## Goal

When the stance channel renders a decisive verdict that disagrees with a
firing NLI channel on a same-story pair, trust the stance channel. When stance
agrees (`conflict`), or has nothing to say (`None`), leave every existing
branch exactly as it is.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Narrow veto, not a broader precedence rethink | The evidence supports exactly the disagreement case; nothing measured supports removing NLI's role on same-story pairs generally |
| `stance == "agreement"` + firing NLI (untested combination) | Abstain, not corroborate | No labeled example either way; abstaining removes the wrong edge without asserting an equally unproven opposite one, consistent with this branch's existing abstain-over-assert fallback |
| Mechanism | Inline guards in `classify_relation` (Approach A) | Smallest diff; a predicate this size doesn't earn its own extracted function yet |
| `stance is None` | Untouched — the veto can never apply | `stance_disagrees` requires `stance is not None`, so the pre-2026-07-26 contract stays byte-for-byte |
| Measurement | Full frozen-surface discipline (same as sub-project E) | This changes a shared global function; every surface it feeds needs re-checking, not just the node5 numbers |
| `BASELINE_P` / `BASELINE_P_ALL` / `BASELINE_R` in `node5_labels.py` | Left unchanged | They document the pre-stance-channel (`ebceb46`) reference point, shared with E's own tests. This spec's bar is pinned as new exact-count assertions, not a redefinition of those constants |

## Architecture

All changes are inside `gin/cartographer/combined.py`. No other module changes.

```
nli_contradicts  = p_contra >= t.contra_threshold and same_story is not False
stance_disagrees = same_story and stance is not None and stance != "conflict"

if nli_contradicts and not stance_disagrees:
    return Relation.CONTRADICTS, "nli"                                    # unchanged
if same_story:
    if stance is None:
        return Relation.CONTRADICTS, "band"                              # unchanged
    if stance == "conflict":
        return Relation.CONTRADICTS, "stance"                            # unchanged
    if nli_contradicts:
        return Relation.RELATED_UNTYPED, "abstain"                       # NEW: the veto
    if stance == "agreement" and cos >= t.corroborate_ceiling:
        return Relation.CORROBORATES, "band"                             # unchanged
    return Relation.RELATED_UNTYPED, "abstain"                           # unchanged
if cos >= t.corroborate_ceiling:
    return Relation.CORROBORATES, "band"                                 # unchanged
return Relation.RELATED_UNTYPED, "band"                                  # unchanged
```

Every branch's output was traced by hand against all combinations of
`{stance ∈ {None, conflict, revision, partial, agreement, UNALIGNED}} ×
{nli_contradicts ∈ {True, False}}` before writing this spec; only the one
marked NEW changes.

**Documentation to correct, not just add:**
- `type_relation`'s inline comment (`combined.py:269-273`) currently states
  NLI "takes priority over the stance arm below" unconditionally. That
  becomes false and must be corrected to describe the disagreement veto.
- `classify_relation`'s docstring gains a paragraph stating the veto
  condition and citing the measured evidence above (matching this file's
  existing practice of justifying every branch with a measured reason, not
  an assertion).
- The module docstring (lines 1–24) and the rest of `classify_relation`'s
  docstring describe NLI vs. the cosine band and the story gate — neither
  claim is affected, so neither needs to change.

**Explicitly not touched:** `type_relation`'s control flow, `CombinedRelationProposer`
(construction, caching, `_p_contra`), `assess_pair`'s confidence logic (still
`ev.get("p_contra", ev["cos"])` for the new abstain outcome, matching the
existing abstain branches), the function signature, `quantity.py`, `relatedness.py`,
`text_index.py`, and `data/cartographer_thresholds.json`.

## Measurement plan

Re-run and record, in this order:

1. **`scripts/eval_node5_stance.py`** (real models). Predicted from the table
   above, stated as a prediction to be confirmed by actually running the
   modified code, not assumed: within-event `P` 0.857 → **1.000**, `P_all`
   0.750 → **0.857**, `R` stays **1.000**. Dev/held-out split predicted to
   both reach 1.000 (currently 0.900 / 0.750) — report whichever way it
   actually lands.
2. **`tests/test_cartographer_stance_node5.py`** (the model-free isolated
   arm). Predicted **unchanged**: this test injects `p_contra=0.05` for every
   pair, always below threshold, so `nli_contradicts` is always `False` and
   the new veto branch is never reached. If this prediction is wrong, that
   itself is a finding worth stopping on.
3. **45-pair eval set and scan gold eval.**
4. **14-pair escalation bar pin** (`tests/test_cartographer_eval_pairs.py`).
   Confirmed during design that this test never sets `same_story` or `stance`,
   so it is structurally insulated — re-run to confirm, not because the
   mechanism suggests risk.
5. **Held-out 40-pair calibration score** via
   `scripts/recalibrate_cheap_pipeline.py --score-only`. Baseline 0.725
   (post-E). Predicted unchanged, by the same reasoning E's Results section
   used: the 9 same-story pairs in that set are all gold `contradicts`, and
   none of their stance values were among the disagreement cases in table
   above — but this must be checked, not inferred from the node5 table alone.
6. **Full suite** (737 passed / 16 skipped at `befe18b`).

## Success criteria

Reusing the three metrics E defined (`P`, `R`, `P_all` — see that spec for
exact definitions), pre-registered here:

- `P` and `P_all` must not regress below their currently-shipped values
  (0.857 / 0.750) and are predicted to improve to 1.000 / 0.857 — report
  whichever way they land.
- `R` must stay at 1.000. Argued above to be impossible to regress on these
  24 labels by construction (no gold-`contradicts` pair currently reaches the
  `nli` channel with a disagreeing stance) — the measurement confirms the
  argument, not the other way around.
- The two now-fixed false positives are pinned by name and channel change, not
  just by count, matching E's existing testing style: `007↔008` and `036↔037`
  move from `nli`/CONTRADICTS to `abstain`/RELATED_UNTYPED. `023↔024` and
  `023↔026` are pinned as **unchanged**.
- No regression on any frozen surface in the measurement plan.
- `data/cartographer_thresholds.json` byte-identical; no new tunable
  threshold introduced anywhere.

## Failure modes

| Condition | Handling |
|---|---|
| A node1–4 or 45-pair-set contradiction currently caught only by NLI turns out to have a disagreeing, non-`None` stance | Investigate before reverting. If stance is right and NLI was catching it for the wrong reason, that is this fix working as intended elsewhere. If stance is wrong, that is new information about the stance channel's coverage outside node5, and the veto's scope (or the stance channel itself) needs revisiting — not a reason to silently special-case the pair. |
| Held-out 40-pair score moves at all | Report the direction and which commit. Do not compensate by writing thresholds. |
| The isolated test (`test_cartographer_stance_node5.py`) is affected | Stop — the prediction that it can't be reached depends on `p_contra=0.05` always being below threshold in that test's injection, and if that's wrong, the reasoning behind "R can't regress" needs re-checking too. |
| `P` or `P_all` fails to reach the predicted value but still doesn't regress | Report the actual number and the gap; do not tune anything to close it. |

## Testing

All model-free, in `tests/test_cartographer_stance_branch.py`:

- **Invert** `test_nli_still_outranks_the_stance_branch` (currently asserts
  `stance="partial"` + firing NLI → `CONTRADICTS, "nli"`). Rename to reflect
  the new rule and flip the expected outcome to `RELATED_UNTYPED, "abstain"`.
- **New**, parametrized across `["revision", "partial", "agreement", UNALIGNED]`
  at a firing `p_contra`: all four resolve to `RELATED_UNTYPED, "abstain"`.
- **New**: `stance="conflict"` at a firing `p_contra` still resolves via
  `CONTRADICTS, "nli"` (not `"stance"`) — pins the moot-agreement case takes
  the same channel attribution it does today.
- **Unchanged, already covered**: `stance=None` at a firing `p_contra` stays
  `CONTRADICTS, "nli"` (existing parametrized row in
  `test_stance_none_reproduces_the_current_truth_table`) — noted here as
  confirmed-covered, not re-added.
- **Regression**: full suite, plus the two frozen-surface tests named in the
  measurement plan.

## Out of scope

- `n5_doc_023↔024` (stance-channel false positive; both stance and NLI agree
  it's CONTRADICTS — the low-`ALIGN_FLOOR` hazard E pre-registered) and
  `n5_doc_023↔026` (`stance=None`, decided by the pre-stance `band` fallback
  either way NLI or same_story routes it). Neither involves a stance/NLI
  disagreement; both are already documented under different mechanisms in
  E's spec.
- Recalibrating or writing `data/cartographer_thresholds.json`.
- `story_floor`, `df_ceiling`, and anything in `relatedness.py` — stage 1 is
  untouched.
- Any change to `quantity.py`'s extraction, alignment, or judging logic, or
  its `conflict > revision > partial > agreement > None` precedence — this
  spec changes a *different* precedence (which channel's verdict wins), not
  that one.
- Extending the veto to `same_story is False` or `same_story is None` — it
  only ever applies when stage 1 has positively confirmed one story.

## Open questions

None blocking. If measurement (item 1 in the plan) surfaces a same-story
disagreement pattern outside node5 not covered by this design, that is new
evidence for a follow-on, not a reason to hold this one.
