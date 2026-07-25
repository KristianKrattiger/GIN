# Design: Bi-Encoder Frame Detector (sub-project B)

**Date:** 2026-07-24
**Status:** Implemented and measured 2026-07-25 — see **Results** below.
> The Data/Decisions sections record the design as approved. The chunk-level
> leakage guard they specify proved **insufficient** (text-aliased bar chunks);
> the Results section is authoritative for final counts and numbers.
**Sub-project:** B (bi-encoder frame detector) — consumes A/B0 (curator UI + label store), feeds C (recalibration)

## Problem

The Cartographer's escalation frame judge types the `issue_frame` residue: pairs
taking opposed positions on the same proposition, which the cheap pipeline
(cosine bands + NLI) cannot reach. Every off-the-shelf LLM judge fails the bar,
and the 2026-07-13 sweep established that the failure is **signal, not scale**:

| Model | issue_frame recall | class_c | unrelated | flips |
|-------|:---:|:---:|:---:|:---:|
| Mistral-7B dense | 0.50 | 0.67 | 0.25 | 7/14 |
| Qwen3.6-14B-A3B MoE | 0.25 | 0.50 | 0.50 | 7/14 |
| Qwen2.5-14B dense | 0.50 | 0.33 | 1.00 | 3/14 |
| Opus 4.8 (frontier) | **0.00** | 0.67 | 1.00 | 3/14 |

Recall does not rise with capability. Opus reads the pairs lucidly and
*disagrees with the gold* — it recovers the curator's `contradicts` as
same-direction corroboration or topic difference. The label encodes a
contestable **curatorial stance**, not a latent property any general judge
reproduces.

The forward path is therefore to **learn that stance** from curator labels as a
discriminative embedding geometry, rather than prompt for it. Sub-projects A and
B0 built the label store and the productivity tooling; the readiness gauge is now
green (issue_frame 22, agree 20, unrelated 22 against a 20/class target), so B is
unblocked.

## Goal

Ship a learned, `FrameJudge`-compatible detector that reproduces the curator's
`issue_frame` stance, and measure it honestly against the pre-registered
escalation bar and against the LLM judges it replaces.

Explicitly **not** in scope: recalibrating the cheap pipeline (sub-project C),
fine-tuning the encoder (only if the probe gate fails, under its own spec),
unifying `labeled_set.py`/`gold_edges.py` (deferred to C), or any generative
model.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Train/test split | Chunk-level bar exclusion **(insufficient — superseded, see Results)** | Drop any pair where *either* chunk appears anywhere in the bar. Labeling the residue drew from n1/n2 — the same corpus the bar was built from — so 9 chunks now overlap. `readiness.py` filters exact bar *pairs* only, so this leaked invisibly. Costs 11 rows; buys an honestly held-out bar score. |
| Label schema | 4-way, DIVERGENT = `issue_frame` only | `related_untyped` is retained as its own class because "topically related, no typed relation" is exactly the hard negative every LLM judge collapsed on. `story`-contradicts excluded: NLI already owns propositional conflict upstream (`combined.py`, p_contra 0.899 legal register), so escalation never meets those pairs in production, and mixing them dilutes the stance axis. |
| Null-class seeds | Backfill by register | The 7 seed `contradicts` rows predate `relation_class`. inst↔grass (3) and housing (2) are canonical issue_frame → include; the 2 `disc_*` securities pairs are propositional → tag `story`, stay excluded. |
| Architecture | Frozen embeddings + symmetric pair-head, behind a probe gate | Order-invariance is structural, so `direction_flip_count = 0` comes free rather than being trained for. Embeddings stay precomputable, preserving the cheap-pipeline economics that motivated choosing a bi-encoder over a generative judge. |
| Success criteria | Bar all-green **and** cross-validation reported | The bar stays the headline gate for comparability with the judge sweep, but 14 pairs (4 issue_frame) can be passed by luck. Precedent: `calibration.py:leave_one_out` reported 0.69 against 0.875 in-sample, and the honest number was the valuable one. |
| Module placement | New `gin/frames/` package | `gin.cartographer` must not import `gin.curator` — sub-project A established and verified that invariant. `gin/frames/` composes above both. |

## Data

**Source:** `data/curator/labels.jsonl`, read through `gin.curator.store.Store`.

**Count from the fold, never from raw log lines.** `Store.gold()` folds the
append-only log latest-wins per pair key: 104 raw lines resolve to 102 unique
pairs. Counting lines double-counts relabeled pairs and silently trains on stale
labels. The cross-check that the fold is the correct view: folded DIVERGENT is
22, matching `readiness.py`'s independently computed `new_issue_frame = 22`.

**Text resolution:** union of `labeled_set.chunks()`,
`load_corpus_chunks(corpus_node1..4)`, and `data/synthetic/news_corpus.yaml` →
236 chunks. Verified to resolve every row that survives the other filters, and —
importantly — **all 21 escalation-bar chunks**.

That last point makes the whole of B runnable **without Postgres**. The existing
`scripts/cartographer_eval_escalation.py` calls `ensure_postgres()` and sources
bar text via `chunks_from_db`; 10 bar chunks (`inflation_*`, `labor_*`, `wage_*`,
`export_*`, `school_*`, `transit_*`) live only in the DB or in
`news_corpus.yaml`. Reading the YAML directly keeps B aligned with the
DB-free precedent sub-project A set, and means the bar can be scored in CI.

**Filter chain**, applied in order, each drop counted by reason and reported —
never silent:

1. **Schema map** — rows not matching the 4-way map are dropped.
2. **Bar-chunk exclusion** — either endpoint appearing in the bar drops the row.
3. **Text resolution** — unresolvable chunk id drops the row.

| Label | Source relation | before backfill | after backfill |
|-------|-----------------|--:|--:|
| DIVERGENT | `contradicts` + `relation_class == issue_frame` | 22 | **27** |
| AGREE | `corroborates` | 17 | **17** |
| RELATED_UNTYPED | `related_untyped` | 15 | **15** |
| UNRELATED | `unrelated` | 21 | **21** |
| **Total** | | 75 | **80** |

> Superseded: the text-alias guard added after the final review drops 31 more
> rows. **The honest training set is 49** (DIVERGENT 24, AGREE 9,
> RELATED_UNTYPED 10, UNRELATED 6). See Results.

Dropped from 102 folded pairs: 11 on bar-chunk leak, 11 on schema (9
`story`-contradicts plus the 2 `disc_*` pairs the backfill reclassifies to
`story`). Text resolution drops nothing — every unresolvable chunk id sits in a
row already removed by an earlier filter.

**Backfill mechanism:** append 7 corrected `LabelRecord`s (5 `issue_frame`, 2
`story`) with `supersedes` set to the original seed row ids. The store's
latest-wins fold makes these authoritative; provenance is preserved in the log.
No reclassification logic lives in the trainer.

These counts are asserted in tests as a regression guard — if the label log
changes, the test names the drift rather than silently retraining on different
data.

## Architecture

```
data/curator/labels.jsonl
        │  Store.gold()
        ▼
   dataset.py ── schema → bar-chunk → text → bar-text-alias → 49 rows
        │
        ▼
   encoder.py ── frozen all-MiniLM-L6-v2, cached per chunk_id (384-dim)
        │
        ▼
   symmetric pair features  [ |a-b| , a*b , (a+b)/2 ]  → 1152-dim
        │
        ├──► probe.py  (stage 0 gate: logistic regression, LOO)
        │
        ▼
    head.py ── smallest sufficient head → 4 logits
        │
        ▼
   judge.py ── argmax, RELATED_UNTYPED→UNRELATED → FrameJudge (a,b)->str
        │
        ▼
    eval.py ── evaluate_escalation_judge (bar) + LOO + baseline table
```

### Stage 0 — probe gate (`probe.py`)

Before any head is trained, fit a logistic regression on the symmetric pair
features and report leave-one-out balanced accuracy on the DIVERGENT-vs-rest
axis, against a stratified-random baseline.

This doubles as the **linear baseline**: if a linear model already clears the
bar, that is the shipped model and no MLP is built.

Thresholds are fixed here so the gate cannot be renegotiated after seeing the
number. DIVERGENT-vs-rest is a binary axis, so chance balanced accuracy is 0.50.

- **Pass** — LOO balanced accuracy ≥ **0.65** → proceed to stage 1.
- **Inconclusive** — 0.55–0.65 → report as inconclusive; stage 1 may proceed but
  the writeup carries the caveat that the geometry is weak.
- **Fail** — < 0.55 → stop. The frozen geometry lacks a recoverable stance axis.
  That is a publishable measured finding and the justification for escalating to
  encoder fine-tuning under a separate spec — not a reason to quietly add
  capacity here.

There is prior reason to take the failure branch seriously: `combined.py`'s
cosine bands for divergent (0.134–0.552) and corroborate (0.490–0.727) overlap
substantially. The head sees full 384-dim vectors rather than cosine alone, so
the probe is a genuine question, not a formality.

### Stage 1 — model (`encoder.py`, `head.py`)

- **Encoder:** frozen `all-MiniLM-L6-v2` — the same encoder `combined.py` uses,
  so embeddings are shared and precomputable across the stack.
- **Pair features:** `[|a-b|, a*b, (a+b)/2]`, order-invariant by construction.
- **Head:** the smallest head that clears the probe. Linear first; MLP
  1152→32→4 (ReLU, dropout) only if linear is insufficient. At 80 rows,
  capacity is a liability, and the spec prefers underfitting to an
  unfalsifiable win.
- **Inference:** argmax over 4 classes, then RELATED_UNTYPED → UNRELATED to
  satisfy the 3-way `FrameJudge` contract. The 4th class exists to sharpen the
  DIVERGENT boundary during training, not to be emitted.

**Order invariance is structural.** All three pair features are symmetric in
`a`/`b`, so `judge(a,b) == judge(b,a)` holds identically, giving
`direction_flip_count = 0` without training for it. This is asserted as a
property test rather than measured and hoped for.

### Artifacts

Training writes `data/frames/head.joblib` plus a `manifest.json` recording
encoder id, feature dim, class list, head kind, seed, training row count and
per-class counts, git SHA, and UTC timestamp. Loading a head whose manifest
disagrees with the current encoder or feature dim is a hard error, never a
silent fallback.

The head is a scikit-learn estimator (`LogisticRegression`, or
`MLPClassifier(hidden_layer_sizes=(32,))` if linear proves insufficient), not a
hand-written torch module. At 80 rows a training loop is pure surface area for
bugs; sklearn gives deterministic fits, built-in `LeaveOneOut`, and
`class_weight="balanced"` for free. Model artifacts are gitignored; the manifest
is committed.

## Training protocol

- **Leave-one-out** over the 80 rows — the sole cross-validation scheme
  (cheap at this size; precedent `calibration.py:leave_one_out`). An earlier
  draft also called for stratified 5-fold for training curves and early
  stopping; that was dropped as YAGNI once the head became a scikit-learn
  estimator, since `LogisticRegression` needs no early stopping and LOO is
  strictly more informative than 5-fold at n=80.
- **Class weighting** via `class_weight="balanced"` — the distribution
  (27/17/15/21) is mildly imbalanced.
- **Seed variance reported.** At 80 rows a single-seed result is not
  trustworthy; the writeup reports mean and spread across 5 seeds (0–4). A result
  quoted from one seed is treated as unreported.

## Evaluation

**Primary gate** — `evaluate_escalation_judge` on the fixed bar:
`issue_frame_recall = class_c_discrimination = unrelated_discrimination = 1.0`
and `direction_flip_count = 0`.

**Honesty number** — LOO 4-way balanced accuracy and per-class recall over the
80 rows.

**Baseline table** — reported alongside, every time:

| Baseline | Source |
|----------|--------|
| Qwen2.5-14B dense | best sweep model: 0.50 / 0.33 / 1.00, 3 flips |
| Opus 4.8 | frontier: 0.00 / 0.67 / 1.00, 3 flips |
| Majority class | computed |
| Stratified random | computed |

**Decision rule, fixed in advance:**

4-way chance balanced accuracy is 0.25.

- Bar all-green **and** LOO 4-way balanced accuracy ≥ **0.50** (double chance)
  → success.
- Bar all-green **but** LOO < 0.40 → written up as overfit or lucky, **not**
  shipped as a win. With 4 issue_frame pairs this outcome is live, and naming it
  now removes the temptation to rationalize it later.
- Bar all-green with LOO 0.40–0.50 → shipped with an explicit
  small-data caveat, the way `leave_one_out`'s 0.69 was reported.
- Probe fails → stage 1 is not attempted; the finding is the deliverable.

## Failure modes

| Condition | Handling |
|-----------|----------|
| Chunk text unresolvable | Row dropped, counted by reason, surfaced in the run summary |
| A class empties after filtering | Hard error — never train on a degenerate split |
| Manifest/encoder mismatch on load | Hard error, no silent fallback |
| Probe at chance | Stop and document; do not add capacity to rescue it |
| Label log drifts from asserted counts | Test fails and names the drift |

## Testing

- **Dataset filters** — asserted against exact counts (80 total; 27/17/15/21),
  including each drop reason, as a regression guard.
- **Order invariance** — property test that `judge(a,b) == judge(b,a)` across
  all pairs.
- **Manifest round-trip** — save/load reproduces byte-identical predictions.
- **Protocol conformance** — `BiEncoderFrameJudge` satisfies `FrameJudge` and
  runs end-to-end through `evaluate_escalation_judge` with a stub head.
- **Model-free by default** — every test except the embedding step itself runs
  without downloading a model, matching the curator suite's precedent.

## Layering

`gin/frames/` may import `gin.curator` (labels) and `gin.cartographer` (models,
eval harness). Neither may import `gin.frames`. The invariant that
`gin.cartographer` never imports `gin.curator` is preserved and should be
re-verified in the whole-branch review.

## Results — measured 2026-07-25

Implemented over 8 TDD tasks. All numbers reproduced independently by the
controller, not taken from an implementer's report.

### A leakage defect found by the final review — read this first

The chunk-level bar guard this spec specifies was **insufficient**. The fixture
corpus aliases escalation-bar chunks under different ids with byte-identical
text: `inst_em:0` *is* `n1_doc_005:2`, `grass_wf:0` *is* `n2_doc_005:1`, and six
more. Excluding by chunk id let **3 of the bar's 4 issue_frame pairs into
training verbatim**, labeled DIVERGENT.

`build_dataset` now also guards on text (`bar_text_alias` drop reason, derived
from the canonical index so a caller passing a partial index cannot silently
disable it). That removed **31 further rows: the honest training set is 49, not
80.** All numbers below are from the clean 49-row set. The earlier 80-row run is
superseded and its eval artifacts deleted.

`readiness.py` had the same defect and is **now fixed**: the text guard lives in
`gin/curator/text_index.py` and both the gauge and the training filter use it.
Corrected, the gauge reads **issue_frame 24 / agree 9 / unrelated 6** against a
20/class target — **NOT ready**. B was therefore built on a corpus that had not
actually met its own gate: issue_frame is over target, but AGREE and UNRELATED
are under half. That reframes the negative result below — see "What this does
not license".

### Measured

| | value |
|---|---|
| Stage-0 probe (DIVERGENT-vs-rest, LOO) | **0.898** vs 0.498 random → **PASS** |
| Bar `issue_frame_recall` | **0.00** |
| Bar `class_c_discrimination` | 0.667 |
| Bar `unrelated_discrimination` | 1.00 |
| Bar `direction_flip_count` | **0** |
| Verdict | **`bar_failed`** |
| LOO over 49 rows (4-way, chance 0.25) | **0.715** |

Per-class LOO recall: DIVERGENT 0.917, AGREE 0.778, UNRELATED 0.667,
RELATED_UNTYPED 0.500.

**Update 2026-07-25b — UNRELATED class closed to target (63 rows).** 14
UNRELATED labels were added to close the gauge's UNRELATED shortfall, attributed
`curator: "claude"` (the AGREE shortfall is deliberately left to the human
curator, since corroboration is the more contested of the two — Opus agreed with
the gold 1.00 on UNRELATED but only 0.67 on corroboration).

| | 49-row | 63-row |
|---|:--:|:--:|
| LOO balanced accuracy | 0.715 | **0.799** |
| recall UNRELATED | 0.667 | **1.000** |
| recall DIVERGENT | 0.917 | 0.917 |
| recall AGREE | 0.778 | 0.778 |
| recall RELATED_UNTYPED | 0.500 | 0.500 |
| Bar (all four metrics) | 0.00 / 0.667 / 1.00 / 0 | **unchanged** |

**Read the 0.799 with care — it is partly an artifact of label selection.** The
14 added pairs are unambiguous cross-domain negatives (monetary policy × climate
science, anything × a Mars mission timeline), not hard negatives near the
decision boundary. UNRELATED recall 1.000 substantially reflects that choice,
and it lifts the 4-way mean with it. The classes that actually discriminate —
DIVERGENT and RELATED_UNTYPED — are **unchanged at 0.917 and 0.500**, and the
bar is unchanged in every metric. Nothing about the central negative result
moved. A corpus with genuinely hard UNRELATED negatives would score lower here
and be worth more.

**The corpus still has not met its own gate:** readiness reads issue_frame 24 /
agree 9 / unrelated 20 against 20/class. AGREE remains open.

**Update 2026-07-25c — gate CLOSED, 102 rows.** The human curator labeled the
AGREE gap (16 corroborates) and, in passing, **44 more `related_untyped`**.
Readiness is green (issue_frame 24 / agree 20 / unrelated 20); the training set
is 102 rows: DIVERGENT 24, AGREE 20, RELATED_UNTYPED 38, UNRELATED 20.

| | 63-row | 102-row |
|---|:--:|:--:|
| Stage-0 probe | 0.898 | **0.939** vs 0.497 → PASS |
| Bar `issue_frame_recall` | 0.00 | **0.00** |
| Bar `class_c_discrimination` | 0.667 | **1.00** |
| Bar `unrelated_discrimination` | 1.00 | 1.00 |
| Bar `direction_flip_count` | 0 | 0 |
| LOO balanced accuracy | 0.799 | **0.705** |
| recall DIVERGENT | 0.917 | 0.917 |
| recall AGREE | 0.778 | **0.300** |
| recall RELATED_UNTYPED | 0.500 | **0.605** |
| recall UNRELATED | 1.000 | 1.000 |

**Three of four bar metrics are now green.** Only `issue_frame_recall` fails,
and it fails at 0.00 exactly as before — the framing register remains
unreachable. Note `class_c_discrimination` reaching 1.00 is partly degenerate:
the detector became *less* willing to emit DIVERGENT off-distribution, and a
model that never says DIVERGENT scores 1.00 on both control sets by default.

**The LOO drop from 0.799 to 0.705 is not a regression — it is a more honest
number on a harder corpus.** The confusion matrix locates it precisely:

| true | predicted |
|---|---|
| DIVERGENT (24) | **DIVERGENT 22**, UNRELATED 2 |
| UNRELATED (20) | **UNRELATED 20** |
| AGREE (20) | RELATED_UNTYPED 12, **AGREE 6**, UNRELATED 1, DIVERGENT 1 |
| RELATED_UNTYPED (38) | **RELATED_UNTYPED 23**, AGREE 13, DIVERGENT 2 |

**AGREE and RELATED_UNTYPED are symmetrically confused** — 12 one way, 13 the
other, near chance between those two — while DIVERGENT and UNRELATED are
near-perfect. AGREE's earlier 0.778 was propped up by the absence of nearby
RELATED_UNTYPED examples; adding 44 of them revealed the boundary was never
learned. This is a second negative result of the same shape as the headline one:

> The frozen geometry separates **topical** distinctions (same issue vs
> different issue, opposed policy positions vs not) and fails on **epistemic**
> ones (does B corroborate A's claim, or merely share its topic?). Both
> `issue_frame` and the AGREE/RELATED_UNTYPED boundary are epistemic, and both
> are unlearnable here. `direction_flip_count = 0` and `UNRELATED` 1.000 are the
> topical successes; they are real but they are the easy half.

For sub-project C this narrows the question considerably: the missing signal is
not quantity of labels — the gate is closed and the boundary still is not
learned — so it is representational. Either a different encoder carries
epistemic relation structure, or the frozen-embedding approach is the wrong
instrument for this class and fine-tuning (or a cross-encoder, at the cost of
precomputability) becomes the live option.

### What this means

**The frozen geometry separates proposition-level policy opposition and does not
represent framing divergence at all.** That is the finding, and it is sharper
than "the model failed to generalize."

The evidence comes from the superseded contaminated run, where the framing rows
were still in training. Splitting DIVERGENT by origin there:

| DIVERGENT rows | LOO recall | in-sample |
|---|---|---|
| node4 (proposition-level pro/con policy) | **22/22** | — |
| `inst_*`/`grass_*` + housing (framing divergence) | **0/5** | **0/5** |

The five framing rows were missed **even with themselves in the training set** —
a linear head over these features cannot memorize them, let alone generalize.
Meanwhile node4 was fit perfectly. The probe's 0.870/0.898 is carried entirely
by node4.

The escalation bar is made of exactly the class the geometry cannot represent:
its issue_frame pairs are climate institutional-statistics versus
frontline-justice framing (`n1_doc_*` ↔ `n2_doc_*`). So `issue_frame_recall
0.00` is not distribution shift — it is the same failure the 0/5 in-sample
result already showed, now measured cleanly on genuinely held-out pairs.

**`direction_flip_count = 0` is a real and unambiguous win**, solved by
construction via the symmetric pair features, against 3–7 flips for every LLM
judge including the frontier one.

**`unrelated_discrimination 1.00` should not be read as parity with the LLM
judges.** The detector answers UNRELATED on 3 of the 4 issue_frame pairs; a
constant-UNRELATED predictor also scores 1.00 on the unrelated set. The metric
is uninformative for a predictor biased this way.

### Correction to an earlier draft of this section

An earlier version of this writeup attributed the bar failure to the bar's
issue_frame pairs being an institutional-vs-independent register
(`*_bureau_report` vs `*_independent_survey`) held out by the leakage guard.
**That was wrong twice over**: those are the bar's *corroboration* pairs, not its
issue_frame pairs, and the actual issue_frame pairs were in training rather than
held out. It was a comfortable explanation that the data falsifies, and it is
recorded here rather than quietly replaced.

### What this does not license

This is not "the bi-encoder path failed." It is: **framing divergence and
policy-position opposition are different targets, and only the latter lives in
this frozen geometry.** Node4 was built to supply issue_frame examples and in
fact supplies a different, easier class.

The next move is therefore *not* simply more labels. Sub-project C should first
establish whether framing divergence is recoverable from a frozen encoder at all
— e.g. probe the 5 framing rows directly against alternative encoders — before
spending curation effort. If it is not, encoder fine-tuning becomes the live
question rather than a fallback.

### Methodological caveat

Seed-variance reporting is **vacuous for the linear head**: lbfgs is a
deterministic convex solver and never consumes `random_state`, so spread is
structurally 0.000. That zero is evidence of solver determinism, not model
stability. `loo_report` returns `seed_variance_meaningful` and the CLI labels
the number so it cannot be misread. `per_class_recall_last_seed` is named for
what it is — last-seed, not seed-averaged.

## Open questions

None blocking. Deferred by design: encoder fine-tuning (only on probe failure),
and whether `RELATED_UNTYPED` should eventually become a first-class GIN
relation rather than an inference-time collapse — a question for C, once there
are enough labels to support it.
