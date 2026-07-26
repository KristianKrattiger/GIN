# Design: Cheap-Pipeline Recalibration (sub-project C)

**Date:** 2026-07-25
**Status:** Implemented and measured 2026-07-25 — see **Results** below.
> The recalibration ran and its output was **rejected, not shipped**. The design
> was sound; the premise was not. Results are authoritative over the sections
> above.
**Sub-project:** C — consumes A/B0's label store; independent of B's detector

## Problem

`CombinedRelationProposer`'s thresholds are calibrated by grid search over
**39 baked sample tuples** hardcoded in `gin/cartographer/calibration.py`
(`_MEASURED`), measured over `labeled_set.py`'s 33 gold pairs plus 6
`related_untyped` samples pulled from a 2026-07-12 scan run. The module's own
comment says "regenerate if the labeled set changes." It has not been
regenerated. The curator store now holds **178 labeled pairs**.

Three concrete defects follow:

1. **The calibration set is stale and small.** 39 samples, hand-transcribed,
   while 133 non-eval labeled pairs sit unused in the store.
2. **The persisted artifact does not match the code.**
   `data/cartographer_thresholds.json` records
   `"leave_one_out_accuracy": 0.923`; recomputing `leave_one_out(default_samples())`
   today yields **0.897**. Something moved after 2026-07-13 without the file
   being regenerated, and nothing detected it.
3. **Calibration and evaluation overlap.** The 39 samples derive from
   `labeled_set`, which is also the set `scan_eval`/`evaluation` measure
   against. Reported accuracy is partly in-sample.

Sub-project A deliberately deferred this: it declined to rewrite
`labeled_set.py`/`gold_edges.py` "so passing evals don't move." That deferral
was correct then and the constraint still binds — this design honors it rather
than discarding it.

## Goal

Recalibrate the cheap pipeline against the full curated corpus, with an honest
held-out number, and make regeneration a command rather than a hand-edit — while
leaving every pre-registered evaluation surface provably frozen.

**Not in scope:** the encoder/representation question sub-project B surfaced
(whether epistemic relation structure is recoverable from a different encoder).
That is a separate research probe with a different risk profile and gets its own
spec. Also out of scope: any change to `gin/frames/`, and any new relation type.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope of "C" | Recalibration, not the encoder probe | Original-C is fully unblocked and improves code already in production; the encoder question is speculative and deserves separate framing. |
| Loader unification | **Split** — eval surfaces frozen, calibration reads the store | The bar is pre-registered and must not move; calibration should track new labels. Full unification would put a pre-registered eval one bad label away from drifting across a 15-file blast radius. |
| Train/eval split | Calibrate on store **minus** the 45 eval pairs | `labeled_set` + `gold_edges` pairs are what `scan_eval`/`evaluation` measure against. Excluding them makes the reported number held-out rather than a restatement. Leaves 133 calibration pairs. |
| Sample persistence | Generated JSON data file + manifest | Keeps the deliberate "calibration reproduces without models" property while making regeneration a command. A generated Python literal invites accidental hand-edits; on-the-fly computation would force models into calibration tests. |
| Disputed pair | Curator adjudicates; run reports sensitivity on the **held-out score** | `inst_em:0 ↔ clim_pledges:0` is a curatorial call, not the implementer's. It is a `labeled_set` member, hence an eval pair excluded from calibration, so flipping it moves the held-out number rather than the thresholds. The run reports it both ways so the cost of the decision is visible. |

## Current state (measured 2026-07-25)

| | value |
|---|---|
| Baked calibration samples | 39 |
| Calibrated thresholds | gate 0.140, ceiling 0.486, contra 0.686 |
| LOO accuracy (recomputed) | **0.897** |
| LOO class_c_discrimination | 1.000 |
| Persisted `thresholds.json` claim | 0.923 — **stale, does not reproduce** |
| Store folded pairs | 178 |
| Eval pairs (`labeled_set` ∪ `gold_edges`) | 45 id-keys, **40 distinct offline-measurable** — see note |
| Available calibration pairs | **133** |

**Note on 45 vs 40.** `gold_edges` and `labeled_set` name the same 5 pairs under
two chunk-id schemes (`disc_northwind_complaint:0` *is* `disc_nw_complaint:0`;
likewise `disc_meridian_*`, `hf_alderflats_*`, `hf_kestrel_*`, `wf_multi_*`). The
store was seeded from both and holds both copies, but only the short-form ones
resolve to offline text. So `eval_pair_keys()` is 45 id-keys while the held-out
set is **40 distinct pairs**. The leakage guarantee is unaffected: long-form
copies drop as `text_unresolved` and their short-form twins as `eval_pair`, so
neither can become a calibration sample. This is the same id-aliasing hazard
sub-project B hit from the other direction, and it is why guards in this codebase
should compare text rather than chunk ids wherever that is possible.

Calibration-pair class mix: `related_untyped` 62, `corroborates` 26,
`contradicts` 22, `unrelated` 21, `supersedes` 2 (the last excluded — not a
classifier output).

## Architecture

```
data/curator/labels.jsonl
        │  Store.gold()  → 178 folded pairs
        ▼
  exclude the 45 eval pairs (labeled_set ∪ gold_edges)  ── frozen, never calibrated on
        │
        ▼  133 pairs
  scripts/regen_calibration_samples.py
        │  real embed + NLI + same_story, once
        ▼
  data/calibration/samples.json   {manifest, samples[]}
        │
        ▼  model-free from here down
  calibration.default_samples() ── manifest gate (model ids must match combined.py)
        │
        ├──► calibrate()      → Thresholds
        └──► leave_one_out()  → honest accuracy
        │
        ▼
  data/cartographer_thresholds.json  (consumed by combined.load_thresholds)
```

### Frozen surfaces — the load-bearing constraint

These do **not** change and gain an explicit guard:

- `gin/cartographer/gold_edges.py` and `escalation_eval.default_calibration_sets()`
  keep their current definitions. A new test pins the escalation bar's exact 14
  pairs **by chunk id**, so any future edit that would move the bar fails loudly.
- `scan_eval` / `evaluation` continue to measure against `gold_edges`. They are
  the cheap pipeline's held-out eval and must not become store readers.
- `labeled_set.py` keeps serving chunk text — `gin/curator/text_index.py`
  depends on it, and sub-project B's whole DB-free path runs through that.

Only **sample generation** reads the curator store. This is the split the
brainstorming settled on: single-source-of-truth where it helps, nowhere it
endangers a pre-registered eval.

### Component 1 — sample generator (`scripts/regen_calibration_samples.py`)

Reads the store, drops the 45 eval pairs and any `supersedes` rows, resolves
text via `gin.curator.text_index.default_text_index()`, and for each remaining
pair computes:

- `cos` — `CombinedRelationProposer.embedding_cosine`
- `p_contra` — max-direction NLI, the same scorer `combined.py` uses
- `same_story` — `relatedness.make_same_story` over the corpus texts, matching
  what the classifier receives at scan time

Writes `data/calibration/samples.json`:

```json
{
  "manifest": {
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "nli_model": "cross-encoder/nli-deberta-v3-xsmall",
    "n_samples": 133,
    "class_counts": {"related_untyped": 62, "corroborates": 26,
                     "contradicts": 22, "unrelated": 21},
    "excluded_eval_pairs": 45,
    "git_sha": "…",
    "created_utc": "…"
  },
  "samples": [{"cos": 0.39, "p_contra": 0.068,
               "same_story": false, "relation": "contradicts"}]
}
```

Network/model-bound, run deliberately — never during tests or calibration.

### Component 2 — sample loading (`calibration.default_samples()`)

Reads the JSON file. **Manifest gate:** if `embed_model` or `nli_model` disagree
with `combined.DEFAULT_EMBED_MODEL` / `DEFAULT_NLI_MODEL`, raise — a sample file
measured with different models must never silently calibrate the live pipeline.
This mirrors the `head_sha256` gate `gin/frames/head.py` already uses, and it is
the direct fix for defect 2 above.

`_MEASURED` is deleted, not kept as a fallback: a silent fallback to 39 stale
samples is precisely the failure being removed.

### Component 3 — recalibration CLI (`scripts/recalibrate_cheap_pipeline.py`)

Loads samples, calibrates, computes LOO, and prints a comparison against the
current baseline. Writes `data/cartographer_thresholds.json` including the
provenance fields the current file lacks (`n_samples`, model ids, git sha,
UTC) so a stale artifact is detectable next time.

**Sensitivity line.** `inst_em:0 ↔ clim_pledges:0` is a member of
`labeled_set` gold, so it is one of the 45 **eval** pairs and is excluded from
calibration — flipping it cannot move the thresholds. What it moves is the
**held-out score**. The CLI therefore reports held-out accuracy computed twice:
once with the pair as `corroborates` (its current store label) and once as
`contradicts`. This changes no label; it prices the open curatorial decision so
the curator can adjudicate on evidence.

## Success criteria

Pre-registered, stated before the numbers are seen:

- **Primary:** LOO accuracy over the 133 calibration pairs, reported against the
  current **0.897 / 39 samples** baseline, together with
  `class_c_discrimination` (currently 1.000).
- **Held-out:** the recalibrated thresholds scored on the 45 frozen eval pairs
  via the existing `scan_eval`/`evaluation` path — the number that was
  previously partly in-sample and now is not.
- **Invariant:** the escalation bar's 14 pairs are byte-identical before and
  after; the pinning test proves it.

**Report the number whichever direction it moves.** More calibration data
reducing accuracy is a live outcome, not a failure — it happened to AGREE in
sub-project B when hard negatives were added, and it would mean the 39 baked
samples were flattering. A drop with a wider, more representative sample is more
trustworthy than the 0.897 it replaces, and the writeup must say so rather than
reaching for the older number.

## Failure modes

| Condition | Handling |
|-----------|----------|
| `samples.json` missing | Hard error naming the regen command — never fall back to baked samples |
| Manifest model ids mismatch | Hard error; a differently-measured sample set must not calibrate the live pipeline |
| A class empties after eval-pair exclusion | Hard error — never grid-search on a degenerate set |
| Chunk text unresolvable during regen | Pair dropped, counted by reason, surfaced in the run summary |
| Escalation bar would change | Pinning test fails |
| Thresholds file written without provenance | Not possible — fields are required by the writer |

## Testing

- **Bar pinning** — the 14 escalation-bar pairs asserted by exact chunk id.
- **Eval-pair exclusion** — no pair in `labeled_set ∪ gold_edges` appears in the
  generated sample set; asserted against the real store (regression guard on the
  133/45 split).
- **Manifest gate** — mismatched model ids raise; matching ids load.
- **Schema round-trip** — samples survive write→read identically.
- **Calibration determinism** — same samples produce the same thresholds.
- **Existing calibration tests** — repointed at a small committed fixture sample
  file rather than the live one, so they stay model-free and stable.
- Everything except `regen_calibration_samples.py` runs without models.

## Results — measured 2026-07-25

Implemented over 5 TDD tasks. All numbers below were reproduced independently by
the controller, not taken from an implementer's report.

### The recalibration was measured and rejected

`data/cartographer_thresholds.json` deliberately retains its previous values.

| | baseline (39 baked) | recalibrated (131 curated) |
|---|:--:|:--:|
| `gate_floor` | 0.140 | 0.279 |
| `corroborate_ceiling` | 0.486 | 0.733 |
| `contra_threshold` | 0.686 | 0.552 |
| LOO accuracy | 0.897 | **0.534** |
| LOO `class_c_discrimination` | 1.000 | 0.731 |
| contradicts precision / recall | — | **0.0 / 0.0** |
| held-out (40 frozen eval pairs) | **0.700** | **0.550** |

The held-out row is the decisive one: the thresholds we kept beat the ones we
computed, on pairs neither was fitted to.

### Why — the calibration set does not exercise the mechanism

`classify_relation` gates both contradicts channels on `same_story`, and that
gate cuts both ways:

- `same_story=False` **blocks** CONTRADICTS entirely. All **22** contradicts
  samples in the new set are `same_story=False`, so no threshold choice can type
  one correctly. That is why recall is 0.0.
- `same_story=True` **forces** CONTRADICTS regardless of thresholds. **11** rows
  are same-story and **none** are gold contradicts, so all 11 are wrong. That is
  why precision is 0.0 rather than undefined (0 TP / 11 FP), and why `class_c`
  lands at exactly 19/26 = 0.731.

**33 of 131 rows are unfixable at any threshold**, capping achievable accuracy at
0.748. The grid's actual maximum is 0.588, against a majority-class baseline of
0.473 — the classifier is barely beating "guess `related_untyped`".

Two further findings from the final review:

- **`contra_threshold` is provably inert on this data.** No value in [0, 1]
  changes any of the 131 predictions; 0.552 is pure max-margin tie-break output.
  One third of the calibrated triple had zero evidential support.
- **Class imbalance is a genuine co-contributor, not an alternative
  explanation.** Even restricted to the 98 non-blocked rows, the best achievable
  is 0.786 — cosine alone does not separate these classes.

### What this means

The spec's premise — *calibrate the cheap pipeline on the full curated corpus* —
is **wrong**, and the reason is structural rather than statistical.

The cheap pipeline's contradicts channel targets **same-story propositional
conflict**: two reports of one event disagreeing on a number. The curated corpus
has grown in the opposite direction, toward **cross-document framing
divergence** (`issue_frame`, node4 policy pro/con). Only **11 of 131** samples
are same-story at all, and **zero** of the contradicts ones are. The corpus and
the detector are aimed at different phenomena, so one cannot calibrate the other.

This rhymes with sub-project B's finding rather than contradicting it. B found
the frozen geometry separates *topical* distinctions and fails on *epistemic*
ones. C finds the automated pipeline is built for *same-story propositional*
conflict while curation has moved to *cross-story framing* conflict. Both say the
same thing from different directions: **the curation effort and the automated
machinery have drifted apart in what they are about.** That drift, not any
threshold, is the thing to fix.

### What ships anyway

The infrastructure is sound and independently valuable:

- Calibration samples are a **generated file with a manifest gate**, not a hand-
  transcribed literal. Regeneration is one command.
- The **eval/calibration split is airtight** — 131 + 40 + 5 + 2 = 178, and
  `load_samples()` structurally cannot read the held-out array.
- The **escalation bar is pinned** by chunk id, so a pre-registered eval can no
  longer move silently.
- Spec defect 2 is fixed: `cartographer_thresholds.json` recorded a
  non-reproducing 0.923; it now records the verified **0.897** with provenance.
- The manifest records the `same_story` parameters, so the signal that decided
  this outcome is no longer invisible.

### Adjudication of the 22 same-story-negative contradicts pairs

The writeup above left one thing indistinguishable: are those 22 pairs *genuinely*
cross-story, or same-story pairs that `make_same_story`'s lexical test simply
misses? Those imply different projects, so they were adjudicated by hand.

**Verdict: genuinely cross-story. The story predicate is correct, not broken.**

All 22 are `n4_doc ↔ n4_doc` — node4's contested-policy pro/con pairs. Reading
them settles it. `n4_doc_001:0` "carbon taxation can correct a market failure and
make the economy more efficient" against `n4_doc_002:0` "Carbon taxes are
nonobjective, they are coercive, and they are impediments to prosperity" is two
*positions on a policy question*, not two *reports of one event*. Three of the 22
(`n4_doc_014 ↔ n4_doc_015`) are not even the same topic — renewables-grid against
gas-bridge-fuel, paired because the stances interact.

The predicate's internals agree. Every one of the 22 fails at the **first**
condition, fewer than 2 shared corpus-rare tokens; none even reaches the anchor
check. Shared tokens run 1–5, of which rare ones run 0–1, and the rare ones that
do appear are topic words — `taxe`, `divestment`, `geoengineering` — never
entities.

No parameterization rescues them:

| `story_floor` | `require_anchor` | `df_ceiling` | pass |
|:--:|:--:|:--:|:--:|
| 2 | True | default (7) | **0/22** ← shipped setting |
| 2 | True | 15 / 40 | 0/22 |
| 1 | True | any | 0/22 |
| 2 | False | 15 / 40 | 5 / 14 |
| 1 | False | default / 15 / 40 | 8 / 17 / 20 |

**`require_anchor=True` gives 0/22 at every other setting.** The entity-anchor
requirement is exactly what separates "same event" from "same topic," and these
pairs share no entity. Recovering any of them means dropping that requirement,
which converts the predicate from a story detector into a topic detector and
reinstates the cross-topic false positives that scan run `20260712T074956Z`
documented as the reason the mid-band was changed in the first place.

**So the branch is settled: the pipeline has no channel for what the corpus
contains.** Not a bug to fix in `make_same_story`, and not a threshold to retune.
Either curation re-aims at same-story propositional conflict, or a cross-story
divergence channel is built. That is a product decision, and it is now the
blocking one.

### Known limitation — the method does not scale

`calibrate()` is O(n⁴) and `leave_one_out()` is O(n⁵). Measured: 0.20 s / 1.40 s
/ 5.05 s at n = 30 / 50 / 70, ~62 s at n = 131. This LOO run took **~2.25
hours**; at 200 samples it would be ~19 h and at 300 ~35 h. The method is
impractical beyond roughly 150 samples, which the curator corpus will pass.

`_score()` is a pure-Python loop over every sample for each of O(n³) threshold
combinations. Vectorizing it preserves the grid and tie-breaking exactly,
provided grid iteration order is kept (the `>` comparison makes ties first-wins)
and `same_story` is encoded three-valued as −1/0/1 rather than as a bool, since
`classify_relation` distinguishes `None` from `False`. Expected ~20–50×.

## Open questions

None blocking. The disputed pair is deliberately left open and priced rather
than decided; adjudicating it is a curator action that needs no code change.
