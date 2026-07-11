# Reasoning-layer robustness under noisy edges

**Status:** steps 1–3 implemented (gate-level characterization,
decode-in-the-loop degradation, and the Cartographer design bridge — all
measured). The real NLI relation detector (the "minimal Cartographer" proper) is
the next step, scoped in [nc_cartographer_design.plan.md](nc_cartographer_design.plan.md) §6.

**Why this is the gate.** Everything in
[nc_real_text_divergence_generalization.plan.md](nc_real_text_divergence_generalization.plan.md)
§7 runs on **hand-curated, fully-trusted edges**. The reasoning layer today
*assumes* every `contradicts` edge is real and query-relevant. The moment a
Cartographer proposes edges automatically, edge quality (precision/recall)
becomes a live variable the reasoning layer has never been stress-tested
against. Per §7's own sequencing — "close [reasoning robustness] before adding a
layer whose value is feeding the reasoning layer *more and noisier* edges" —
this is the work that unblocks Cartographer + Bookkeeper. The architecture's
falsifiability-by-layer only holds if each layer is independently sound first.

This doc measures **how the reasoning layer behaves when fed bad edges**, so we
know which failure classes it already deflects and which must be caught upstream
(Cartographer edge-precision) or at admission (Bookkeeper semantic verification)
rather than by tightening the reasoning gate.

---

## 1. What already defends the reasoning layer

Two mechanisms in `gin/corpus/retrieve.py` decide whether a `contradicts` edge
reaches the divergent decode at all:

- **`_is_ambiguous` / `_pair_divergence_ok`** — a `contradicts` edge only flips
  the synthesis mode to `divergent` if **both** endpoints pass the
  query-relevance test. Mis-curated edges whose partner is off-query are
  supposed to drop back to convergent here.
- **`_divergence_relevant` → `idf_weighted_relevance` (floor 0.13)** — the
  relevance test is IDF-weighted, so one *distinctive* shared word (`wildfire`)
  clears the bar while one *generic* word (`district`) does not.

Neither mechanism inspects whether the two sides *actually contradict*. They
test topical relevance, not the truth of the relation type. That asymmetry is
the whole finding of step 1.

## 2. A taxonomy of noisy edges a Cartographer could emit

| Class | Description | What a robust reasoning layer should do | Caught by the gate? |
|-------|-------------|------------------------------------------|---------------------|
| **A — irrelevant partner** | `contradicts` between a query-relevant chunk and an off-query chunk | Not diverge (drop the pair) | **Yes** — relevance gate rejects the off-query side |
| **B — generic overlap** | Two chunks sharing only a generic query word | Not diverge | **Yes (by design)** — IDF floor rejects generic-only mass |
| **C — mislabeled corroboration** | Two query-relevant chunks that actually *agree*, mistyped as `contradicts` | Not diverge (they don't conflict) | **No** — both sides pass relevance; gate forces divergent on agreeing text |
| **D — dangling anchor** | Edge endpoint absent from retrieval | Not diverge (inert) | **Yes (structurally)** — endpoint lookup returns `None`, pair skipped |
| **E — true contradiction** (control) | A real, query-relevant contradiction | Diverge | **Yes** — retained |

**Class C is the load-bearing result.** The reasoning gate is a *relevance*
filter, not a *relation-type verifier*. It cannot tell a real contradiction from
two agreeing chunks that both happen to be on-topic. So a Cartographer that emits
false `contradicts` edges between corroborating sources will drive the reasoning
layer into spurious divergent mode — quoting two agreeing sentences as if they
disagree — and no amount of tightening the relevance floor fixes it, because both
sides *are* relevant. This is the concrete argument that:

- the **Cartographer** must be measured on **edge precision/recall** on its own
  axis (§7.1 of the divergence plan), not judged only by downstream reasoning
  metrics; and
- the **Bookkeeper** admission gate needs a **relation-type / anchor
  verification** step, because relation correctness is not recoverable at read
  time from relevance alone.

## 3. Step 1 — deterministic gate characterization (implemented)

`gin/eval/edge_robustness.py` — a DB-free, LLM-free harness that runs a taxonomy
of noisy edges through the **actual** reasoning-layer gate
(`retrieve._is_ambiguous`) and reports, per class, whether the edge would have
forced divergent synthesis.

- IDF is built from the **real two-node corpus** (`corpus_node1.json` +
  `corpus_node2.json`) so the gate behaves exactly as in the working demo, not
  against a toy corpus whose IDF would be an artifact (the overfitting risk the
  divergence plan §7.1 flags).
- Cases are grounded in the **same real fixture sentences** the divergence demo
  uses (institutional vs grassroots emissions / wildfire / water).
- Two headline numbers:
  - **`noise_rejection_rate`** — fraction of *should-reject* noisy edges
    (classes A, D; B when present) the gate diverts from synthesis.
  - **`true_positive_retention`** — fraction of real contradictions (class E)
    still admitted. Guards against a gate that "passes" the stress test by
    rejecting everything.
- Class C is reported separately and **pinned as a known gap**
  (`tests/test_edge_robustness.py`): the gate forces divergent on mislabeled
  corroboration. When a future Bookkeeper/Cartographer change makes relation
  correctness checkable, that test flips deliberately.

Regression: `tests/test_edge_robustness.py`.

## 4. Step 2 — decode-in-the-loop degradation (implemented, measured)

Step 1 stops at the gate. Step 2 measures what happens to the *answer* when a
class-C edge gets through to the decoder. `gin/eval/edge_degradation.py` drives
the real materialize + constrained-decode path over three hand-constructed,
in-memory bundles (no DB — bundles are built from real corpus text) and scores
each with the production metrics:

- **clean** — two agreeing institutional wildfire statistics, no edge;
- **noisy (class C)** — the *same* agreeing pair, mistyped `contradicts`;
- **control** — a genuine institutional-vs-grassroots contradiction.

No corpus mutation and no specific model needed: the decode is
constraint-determined, so `GreedyMaskDecoder` (deterministic argmax under the
mask) is a faithful stand-in — **the real Mistral-7B produces byte-identical
answers** (`data/eval_runs/edge_degradation_20260711T223926Z`).

**Measured result** (identical for GreedyMaskDecoder and Mistral-7B-Q4_K_M):

| scenario | edge | mode | fabrication | div_fidelity | supp_irrel | spurious |
|---|---|---|---|---|---|---|
| clean (agreeing pair, no edge) | none | convergent | 0.000 | n/a | 0.000 | False |
| **noisy class-C** | contradicts | **divergent** | **0.000** | **1.000** | 0.000 | **True** |
| control (real contradiction) | contradicts | divergent | 0.000 | 1.000 | 0.500 | False |

The class-C answer quotes **two agreeing wildfire statistics** ("56,580
wildfires burned…" | "one-quarter…occurred on federally protected lands") joined
by the divergence delimiter — a **grounded but epistemically wrong** answer.
Fabrication stays 0 (extractive decode can't fabricate) and every existing metric
passes. Critically, **the noisy and control rows are metric-indistinguishable**
(fabrication 0, divergence_fidelity 1.0): divergence-*fidelity* measures citation
coverage, not relation validity, so it cannot tell a real contradiction from two
agreeing chunks. The only field that separates them is `relation_is_real` —
ground truth the reasoning layer does not possess.

**Conclusion.** This is the quantified motivation for a divergence-*validity*
signal that must come from **Cartographer edge-precision** + **Bookkeeper
relation/anchor verification**, not from any read-time reasoning-layer metric.
Regression: `tests/test_edge_degradation.py`. Artifact script:
`scripts/edge_degradation.py`.

## 5. Step 3 — carry the constraints into Cartographer design (implemented)

Written up as its own spec + first implementation:
**[nc_cartographer_design.plan.md](nc_cartographer_design.plan.md)**. It carries
all three constraints below into the Cartographer:

- The relation-type detector must **not** reuse the retrieval-side IDF signal
  (divergence plan §7.1); the Cartographer is a two-stage pipeline (relatedness
  gate → relation detector) whose stages use different signals. Edge
  precision/recall is measured **independently** (`gin/cartographer/evaluation.py`,
  per framing register), with `class_c_discrimination` as the headline: does the
  proposer avoid re-minting the exact step-2 edge.
- Negatives ("assessed, unrelated") are first-class stored `Assessment`s
  (`gin/cartographer/models.py`), not silence.
- Sentence-level anchors decided: adopt divergence plan §7.1 option (b);
  `EdgeProposal` carries optional token-offset anchors now to avoid a later
  migration.

First implementation ships the relatedness gate, the anti-pattern
`RelatednessProposer` baseline (relatedness-only → class_c_discrimination 0.0,
and it even ranks the agreeing pair above a real contradiction), and the
independent harness. The real NLI relation detector is the next step. Regression:
`tests/test_cartographer.py`.

## 6. Out of scope (this doc)

- The Cartographer / Bookkeeper implementations themselves (this only produces
  the robustness evidence and constraints that gate them).
- RAG-arm and Flagged-Generation changes.
- Retrieval-side gate *tuning* — step 1 characterizes current behavior; it does
  not move the floor.
