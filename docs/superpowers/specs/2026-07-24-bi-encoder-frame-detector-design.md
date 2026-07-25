# Design: Bi-Encoder Frame Detector (sub-project B)

**Date:** 2026-07-24
**Status:** Approved (design), pending implementation plan
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
| Train/test split | Chunk-level bar exclusion | Drop any pair where *either* chunk appears anywhere in the bar. Labeling the residue drew from n1/n2 — the same corpus the bar was built from — so 9 chunks now overlap. `readiness.py` filters exact bar *pairs* only, so this leaked invisibly. Costs 11 rows; buys an honestly held-out bar score. |
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
   dataset.py ── schema map → bar exclusion → text resolution → 80 rows
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

Implemented over 8 TDD tasks. All numbers below were reproduced independently by
the controller, not taken from an implementer's report.

**Stage-0 probe: PASS.** LOO balanced accuracy **0.870** on DIVERGENT-vs-rest
against a stratified-random baseline of 0.502 (gate was ≥0.65). The frozen
all-MiniLM-L6-v2 geometry *does* carry a linearly recoverable stance axis, so
the premise of the whole approach survived its own falsification test.

**Escalation bar: FAILED.** Verdict `bar_failed` under the pre-registered rule.

| metric | bi-encoder | Qwen2.5-14B | Opus 4.8 |
|---|:--:|:--:|:--:|
| issue_frame_recall | **0.00** | 0.50 | 0.00 |
| class_c_discrimination | 0.67 | 0.33 | 0.67 |
| unrelated_discrimination | **1.00** | 1.00 | 1.00 |
| direction_flip_count | **0** | 3 | 3 |

**Leave-one-out over the 80 training rows: 0.676** balanced accuracy (4-way
chance 0.25). Per-class recall: DIVERGENT 0.815, AGREE 0.765, UNRELATED 0.857,
**RELATED_UNTYPED 0.267**.

### What this means

The honest reading is that **the detector learned the curator's stance and
still failed the bar** — and those are not in tension, because they measure
different distributions.

- In-distribution the model is genuinely good: 0.676 4-way against 0.25 chance,
  with DIVERGENT recall 0.815. It reproduces the curatorial frame on the data it
  was trained on.
- On the bar it scores 0.00 issue_frame recall — **Opus 4.8's exact failure
  mode**. Direct diagnosis (loading the trained head and calling it on the four
  gold pairs) returns `UNRELATED, AGREE, UNRELATED, UNRELATED`. This is real
  model behavior, not a harness fault.
- The cause is the chunk-level bar exclusion working as designed. The bar's
  issue_frame pairs are an institutional-vs-independent register
  (`*_bureau_report` vs `*_independent_survey`), and every chunk of that
  register was held out of training. Node4 taught proposition-level pro/con
  policy opposition. The model learned the frame it was shown and did not
  transfer to a register it never saw.

**`direction_flip_count = 0` is a real, unambiguous win.** Every LLM judge
flipped on 3–7 of 14 pairs; the symmetric pair features make order invariance a
mathematical identity. That metric is now solved by construction rather than
hoped for.

**`RELATED_UNTYPED` recall 0.267 is the weakest result** and is diagnostic: the
hard-negative class — topically related but untyped — remains the hardest
discrimination, which is the same boundary every LLM judge collapsed on. It has
only 15 training rows.

### What this does not license

This is not evidence that the bi-encoder path is closed. It is evidence that
**4 held-out pairs in an unseen register cannot be reached by 80 training rows
that do not cover that register.** The distinguishing fact versus the judge
sweep is that recall there did not rise with capability, whereas here the model
demonstrably learns the target concept in-distribution. Do not report the
`bar_failed` verdict as "the bi-encoder failed" without that qualifier.

The obvious next move is more labels in the bar's register — but the bar must
stay held out, so that means labeling *new* institutional-vs-independent pairs,
not the bar's own. That is a sub-project C question.

### Methodological caveat

Seed-variance reporting is **vacuous for the linear head**: lbfgs is a
deterministic convex solver and never consumes `random_state`, so spread is
structurally 0.000 across all seeds. That zero is evidence of solver
determinism, not model stability. `loo_report` now returns
`seed_variance_meaningful` and the CLI labels the number, so it cannot be
misread. The spec's original "report seed variance" honesty measure did not
survive contact with the estimator choice.

## Open questions

None blocking. Deferred by design: encoder fine-tuning (only on probe failure),
and whether `RELATED_UNTYPED` should eventually become a first-class GIN
relation rather than an inference-time collapse — a question for C, once there
are enough labels to support it.
