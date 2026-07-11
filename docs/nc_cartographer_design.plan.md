# Cartographer design — carrying the noisy-edge constraints

**Status:** design + implementation across two passes. Built: the independent
edge-precision harness, the relatedness gate, the anti-pattern baseline
(`gin/cartographer/`, `tests/test_cartographer.py`), and the NLI relation detector
(`gin/cartographer/nli.py`, `tests/test_cartographer_nli.py`). §6 records the
measured finding that NLI-contradiction is the wrong signal for GIN divergence and
reframes the next detector; the Bookkeeper admission gate is the step after.

This is step 3 of
[nc_reasoning_robustness_noisy_edges.plan.md](nc_reasoning_robustness_noisy_edges.plan.md)
and the design bridge into item (b) of the divergence plan §7 recommended order.
It exists to carry the measured constraints from steps 1–2 into the Cartographer
before any automated edge proposal is wired to the reasoning layer.

---

## 1. The constraint steps 1–2 hand us

Step 1 measured that the reasoning-layer divergence gate is a **relevance
filter, not a relation-type verifier**: it retains true contradictions and
rejects off-query/dangling edges, but cannot catch *mislabeled corroboration*
(two agreeing, on-query chunks mistyped `contradicts`). Step 2 measured that such
an edge decodes into a **grounded-but-wrong divergent answer** that is
metric-indistinguishable from a genuine contradiction (fabrication 0.0,
divergence_fidelity 1.0). Divergence-*fidelity* scores citation coverage, not
relation validity.

**Therefore the Cartographer owns relation validity.** The one thing the whole
downstream stack cannot recover at read time is whether a `contradicts` edge is
*really* a contradiction. That has to be correct at proposal time and verified at
admission. This reframes the Cartographer's job: it is not "find related chunks
and link them," it is "decide the *relation type* between related chunks, and be
independently measurable on that decision."

## 2. Relatedness ≠ relation (the load-bearing separation)

The Cartographer is a **two-stage** pipeline, and the stages must use
**different signals**:

1. **Relatedness gate (cheap).** "Are these two chunks about the same thing?"
   May use the relevance signals the rest of the system already has — IDF-weighted
   token overlap, shared entities, citation overlap, embedding proximity. Its job
   is only to cut the O(n²) pair space down to candidate pairs worth the expensive
   stage, and to emit **negatives** for pairs assessed and found unrelated.

2. **Relation-type detector (expensive).** "Given that they are related, do they
   *contradict*, *corroborate*, *supersede*, or merely *coexist*?" This stage
   **must not reuse the retrieval-side IDF signal** (divergence plan §7.1): IDF
   measures topical overlap, and topical overlap is exactly what a real
   contradiction and a real corroboration *share*. A detector built on IDF would
   reproduce, inside relation detection, the same blind spot step 1 found in the
   reasoning gate — and it would then be correlated with the gating and anchoring
   that already lean on IDF, so a single artifact could hide in all three. The
   relation detector needs a **semantic / NLI-class signal** (entailment vs.
   contradiction), which is orthogonal to relevance.

The `gin/cartographer/proposers.py::RelatednessProposer` baseline deliberately
**collapses these two stages** — it proposes `contradicts` for any related pair —
so the harness can quantify precisely how much precision that collapse costs (§4).

## 3. Negatives are graph content

Per [GIN_Session_Synthesis_v1.md](GIN_Session_Synthesis_v1.md) §1.2, an
"assessed, unrelated" verdict is a **stored `Assessment`**, not silence.
`gin/cartographer/models.py::Assessment` carries the pair, the verdict
(`contradicts` / `corroborates` / `supersedes` / `unrelated` / `related_untyped`),
the method that produced it, and a rationale. The relatedness gate emits
`unrelated` assessments; the relation detector refines related pairs into typed
verdicts. Only typed, non-`unrelated`, sufficiently-confident assessments become
`EdgeProposal`s for the Bookkeeper. Storing the negatives is what stops the same
null from being re-litigated on every query and is the substrate of the
federation cache.

## 4. Independent measurement (the falsifiable target)

The Cartographer is scored on **edge precision/recall per relation type**, on a
labeled pair set — *independently* of any reasoning-layer metric.
`gin/cartographer/evaluation.py` computes this against gold from the
hand-curated edges (`data/corpus_edges.yaml`) plus explicit negatives, and adds
one headline discrimination score:

- **`class_c_discrimination`** — of the agreeing/corroborating pairs (the step-2
  failure case: the two 2023 wildfire statistics `n1_doc_008:0` ↔ `n1_doc_008:2`),
  the fraction the proposer correctly does **not** type as `contradicts`. This is
  the single number that says whether a proposer would have fed the reasoning
  layer the exact edge steps 1–2 proved it cannot survive.

Measured this session (deterministic harness, real corpus text, 5 labeled pairs):

| proposer | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| relatedness-only (anti-pattern) | 0.500 | 0.333 | **0.000** |

Two distinct failures fall out, and both matter:

1. **Stage-2 blind spot (the headline).** class_c_discrimination is **0.0** — the
   proposer types the two agreeing 2023 wildfire statistics
   (`n1_doc_008:0` ↔ `n1_doc_008:2`) as `contradicts`, minting exactly the edge
   step 2 proved is unrecoverable downstream. This is the quantified spec the real
   relation detector (§6) must close: **class_c_discrimination = 1.0 without
   collapsing contradicts recall**.

2. **Stage-1 limitation (a second, independent finding).** Lexical IDF relatedness
   *under-recalls genuine cross-register contradictions*: the grassroots
   reframings share too little surface vocabulary with their institutional
   counterparts, so the wildfire (0.103) and water (0.045) true-contradiction
   pairs fall below the relatedness floor (0.20) and are never even proposed
   (recall 0.333). Damningly, the relatedness score **ranks the agreeing pair
   (0.226) as more related than a real contradiction** — a precise illustration
   that relatedness is not relation, and evidence that the *production* relatedness
   signal must be **embedding-based (semantic), not lexical**. This independently
   corroborates the sparse-surface-overlap concern the housing fixture was built to
   stress (divergence plan §6 #3, round 2).

Precision/recall is reported **per framing register**, so a shared-signal blind
spot cannot hide behind "just noisy data" as corpus size grows.

## 5. Sentence-level anchors (Bookkeeper decision, decide now)

Divergence plan §7.1 option (b): the Cartographer proposes and the Bookkeeper
stamps **token-offset sentence anchors** on `contradicts`/`supersedes` edges, so
the diverging sentence is *admitted graph state* rather than re-derived by the IDF
heuristic on every read. **Recommendation: adopt option (b).** Rationale carried
from step 1: the fallback anchor scorer is a third consumer of the IDF signal, and
step 2 showed the anchor drives which sentences get quoted as "the divergence."
Making the anchor admitted state removes the last IDF re-derivation from the read
path and is cheap to add while chunks are still ~sentence-sized. `EdgeProposal`
therefore carries optional `src_anchor` / `dst_anchor` `(token_start, token_end)`
fields now, so the schema does not need migrating after multi-sentence ingest.

## 6. The relation detector — NLI tried, and the finding that reshapes it

The natural stage-2 detector is an NLI-class pairwise judge, using the same 3-class
cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`, labels
contradiction/entailment/neutral) the eval `Verifier` already loads — a semantic
signal orthogonal to the retrieval IDF, satisfying §2. Built as
`gin/cartographer/nli.py::NliRelationProposer` (injectable scorer, testable
without the model) and measured on the labeled set with the real cross-encoder:

| detector | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| nli_relation (real cross-encoder) | n/a | **0.000** | **1.000** |

**Measured finding — NLI-contradiction is not GIN divergence.** Every labeled pair
types as `related_untyped` (p_contra ≤ 0.068 across all five). The detector scores
class_c_discrimination **1.0**, but only by the degenerate route the §4 recall
co-metric exists to expose: it types **nothing** as `contradicts`, so its recall on
the three genuine framing divergences is **0.0**. The institutional-vs-grassroots
pairs are both *true* statements that emphasize different aspects of a shared
event ("acreage was below average" and "low-income people face smoke risk" do not
logically contradict), so an entailment model rates them neutral — the same verdict
it gives the agreeing pair. Threshold tuning cannot rescue it: the unrelated cross
pair (0.050) scores essentially as contradictory as the true emissions divergence
(0.068). Regression: `tests/test_cartographer_nli.py` (a synthetic
passed-vs-failed pair confirms the typing logic *does* fire on real propositional
contradiction — the miss is a signal property, not a bug).

**Reframe.** The relation the Cartographer must detect is not propositional
contradiction; it is **framing / stance divergence over a shared referent** — two
sources selecting and foregrounding different dimensions of the same event. That
is orthogonal to entailment.

**Second probe — LLM frame judge (also ruled out as-is).**
`gin/cartographer/frame_judge.py::LlmFrameJudge` asks the framing question
directly ("competing perspectives / values vs. agree vs. unrelated") instead of
entailment. Measured with the real Mistral-7B:

| detector | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| llm_frame_judge (Mistral-7B, shipped prompt) | 0.600 | 1.000 | **0.000** |

The exact mirror image of NLI: it types **every** pair `DIVERGENT` — the three
true divergences (recall 1.0) but also the agreeing pair *and* the unrelated cross
pair (class_c 0.0). Confirmed a real judgment, not a parse artifact (raw output is
` DIVERGENT` for all). A stance-axis prompt variant collapses the other way —
every pair `SAME`. So Mistral-7B zero-shot is **prompt-bias-dominated and does not
discriminate the institutional-vs-grassroots stance axis at all**; recall 1.0 here
is the trivial always-divergent artifact, not evidence of a real signal. The
mapping and pipeline are correct (an oracle judge scores perfectly —
`tests/test_cartographer_frame_judge.py`); the signal is simply not cheaply
extractable from a 7B model zero-shot.

**Where this leaves the relation detector.** Two natural signals both collapse in
opposite directions (NLI → nothing divergent; zero-shot LLM → everything
divergent). Neither is the answer. Two conclusions:

1. **The labeled set is too small to develop against.** Five hand-picked pairs
   cannot separate a real signal from prompt bias or tune a threshold without
   overfitting. **Prerequisite for the next step: expand the labeled
   divergence/corroboration/unrelated set** — the framing fixtures
   (`data/fixtures/*.yaml`, legal + housing registers) already supply more
   institutional-vs-grassroots pairs to label, and deliberate corroborating and
   cross-topic negatives must be added.
2. **The lead remaining signal is same-referent + divergent-aspect**: embedding
   proximity on the shared referent combined with *low* similarity on the claim
   content — a structural signal, not a single zero-shot yes/no. This couples with
   the §4 finding that production relatedness should be embedding-based, and unlike
   a bare LLM label it is calibratable against the expanded set. Few-shot / larger
   models are a fallback, not the lead.

The harness (§4) remains the fixed measurement and the target is unchanged:
**class_c_discrimination = 1.0 with non-trivial contradicts recall**. The
Bookkeeper admission gate (anchor verification, DAG invariants, provenance stamp)
is the step after a signal clears that bar.

## 7. Out of scope (this doc)

- The *next* relation signal (same-referent/divergent-aspect or an LLM frame
  judge) and the Bookkeeper admission gate (§6 → next steps). The NLI detector is
  built and measured (§6); it is the ruled-out baseline, not the answer.
- Cross-corpus / federated proposal (the relatedness gate is scoped intra-corpus
  first; the alignment stage is the same machinery at inter-node scope).
- Embedding-based relatedness (the gate is lexical/entity first for a
  deterministic, model-free harness; embeddings are the production upgrade).
