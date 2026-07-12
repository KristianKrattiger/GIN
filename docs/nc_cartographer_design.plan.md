# Cartographer design — carrying the noisy-edge constraints

**Status:** design + implementation across four passes. Built: the independent
edge-precision harness, the relatedness gate + anti-pattern baseline, the NLI
relation detector, the LLM frame judge, a **13-pair labeled set across three
registers**, and the **combined register-robust detector** (§6a:
embedding gate + NLI propositional channel + cosine aspect band —
`gin/cartographer/`, `tests/test_cartographer*.py`). The combined detector reaches
recall 1.0 / precision 0.875 / class_c 0.667 on the 13-pair set, its single error
being a pair whose gold label is disputed (adjudicating it yields a perfect
score). The Bookkeeper admission gate is the next step; §6a's threshold-calibration
caveat (13 pairs is too few) is the standing risk.

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

Measured on the **expanded 13-pair set** (`gin/cartographer/labeled_set.py`:
7 divergent across climate/legal/housing, 3 corroborating, 3 cross-topic):

| proposer | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| relatedness-only (anti-pattern) | 0.667 | 0.286 | 0.667 |

Two findings, both sharpened by the larger set:

1. **Stage-2 blind spot.** The proposer still mints the two agreeing 2023 wildfire
   statistics (`inst_wf` ↔ `inst_wf_fed`) as `contradicts` — the exact class-C edge
   step 2 proved is unrecoverable. class_c_discrimination reads 0.667 rather than
   0.0 only because the *stage-1 gate accidentally drops* two of the three
   corroborating pairs for the same reason it drops real divergences (finding 2) —
   an artifact of under-recall, not any relation understanding. The spec the real
   detector (§6) must close is unchanged: **class_c = 1.0 without collapsing
   recall**.

2. **Stage-1 under-recall is general, not a climate artifact.** Lexical IDF
   relatedness drops **5 of 7** true divergences (recall 0.286) — across *all
   three* registers, including both housing pairs (recall 0.0) whose two sides
   share only the place entity, and one legal pair. The framing fixtures were built
   for exactly this sparse-surface-overlap stress, and the gate fails it. Strong
   evidence the *production* relatedness signal must be **embedding-based
   (semantic), not lexical**.

Precision/recall is reported **per framing register**, so a shared-signal blind
spot cannot hide behind "just noisy data."

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

## 6. The relation detector — two signals probed; complementary and register-heterogeneous

Two natural stage-2 detectors — an NLI cross-encoder
(`cross-encoder/nli-deberta-v3-xsmall`, the model the eval `Verifier` loads) and an
LLM frame judge — both use a semantic signal orthogonal to the retrieval IDF (§2).

Both detectors below were first run on a 5-pair stub, where both collapsed
(NLI → nothing divergent; LLM → everything divergent). That collapse turned out to
be a **small-set artifact**: on the **expanded 13-pair set** across three registers,
real structure appears that five pairs hid.

**NLI relation detector.** `gin/cartographer/nli.py::NliRelationProposer` (injectable
scorer, testable without the model). Measured with the real cross-encoder:

| detector | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| nli_relation (real cross-encoder), 13-pair set | 0.500 | 0.143 | 0.667 |

**Finding — GIN "contradicts" is heterogeneous.** NLI fires *correctly* on the
**legal / securities** register — Northwind "record revenue" vs. "materially
overstated revenue" scores p_contra **0.899** → contradicts; Meridian
"customer-trust" vs. "concealed a breach" scores 0.473 (a near miss) — because
securities-fraud framings genuinely *are* propositional contradictions. It rates
the **climate and housing** divergences neutral (p_contra ≤ 0.11), because those are
value/emphasis divergences over a shared referent, not logical contradictions
("acreage below average" and "low-income smoke risk" are both true). So the single
`contradicts` edge type spans two different relations: **propositional contradiction
(legal) and framing/stance divergence (climate/housing)**. NLI covers the first and
is blind to the second. (One climate corroborating pair — `inst_em` "cuts needed"
vs. `clim_pledges` "on track for 2.5–2.9 °C" — also scores high p_contra 0.93; it
sits on the corroborate/diverge boundary, a labeling edge case worth revisiting.)

**LLM frame judge.** `gin/cartographer/frame_judge.py::LlmFrameJudge` asks the
framing question directly. Measured with the real Mistral-7B on the 13-pair set:

| detector | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| llm_frame_judge (Mistral-7B), 13-pair set | 0.583 | **1.000** | 0.333 |

**Finding — the frame judge has real signal, bounded by two fixable errors.** It
recovers **all seven** divergences (recall 1.0), is **perfect on legal and housing**
(precision 1.0 each), and — unlike on the 5-pair stub — now correctly types a
corroborating pair as `AGREE` (the two observed-warming statements), so class_c is
0.333, not 0. Its two error classes are both addressable outside the judge:
(a) it types the **cross-topic** pairs `DIVERGENT` — but topic filtering is the
**relatedness gate's** job, not the relation judge's, so in the real pipeline the
judge never sees them; and (b) it still over-fires on the two remaining climate
corroborations (including the class-C wildfire pair). Error (a) is why the ungated
precision looks low; gated, the judge is strong.

**Where this leaves the relation detector.** The two signals are **complementary,
not redundant**: NLI owns propositional contradiction (legal), the LLM frame judge
owns framing divergence (climate/housing) with high recall. Neither alone clears the
target, but the shape of the solution is now visible:

1. **Relatedness must be embedding-based** (§4): the lexical gate drops 5/7
   divergences, and the frame judge's cross-topic false positives are precisely
   what a real gate removes.
2. **The relation detector is likely a combination**, not one signal — an NLI
   propositional-contradiction channel plus a framing-divergence channel — and it
   must be **register-robust**, since what "contradicts" means differs by register.
3. **The lead single signal for framing divergence is same-referent +
   divergent-aspect**: embedding proximity on the shared referent with *low*
   similarity on the claim content — a structural, calibratable measure, the frame
   judge's high-recall behavior made quantitative.

## 6a. The combined register-robust detector (built, near-target)

`gin/cartographer/combined.py::CombinedRelationProposer` composes the three
findings into one pipeline (both signals injectable, so it is testable without
models):

```
1. embedding relatedness gate   cos < 0.13            -> UNRELATED
2. NLI propositional channel     p_contra >= 0.5       -> CONTRADICTS   (priority)
3. cosine aspect band            cos >= 0.45           -> CORROBORATES
                                 else (related, mid)   -> CONTRADICTS
```

The design rests on a measured regularity: all-MiniLM-L6-v2 cosine separates the
three relation classes into bands on the 13-pair set — unrelated ≤ 0.124,
framing-divergent 0.134–0.552, corroborating 0.490–0.727 — and NLI covers the one
place the bands overlap (a legal contradiction that is *highly* similar, cos 0.552,
which the band alone would miscall as corroboration). NLI has priority over the
band for exactly that reason.

**Measured (real all-MiniLM-L6-v2 + real cross-encoder, 13-pair set):**

| detector | contradicts precision | contradicts recall | class_c_discrimination |
|---|---|---|---|
| combined_relation | **0.875** | **1.000** | 0.667 |

Recall **1.0** across every register; precision and recall **1.0** on legal and
housing. Every channel fires as designed: the gate rejects all three cross-topic
pairs, NLI catches legal Northwind, and the cosine band both catches the
climate/housing framing divergences and correctly types the two clear
corroborations (including the class-C wildfire pair the reasoning layer could
not survive). The **single error** is `inst_em` ↔ `clim_pledges` — the one pair
whose gold label is itself disputed (both NLI 0.93 and the frame judge read
divergence). Adjudicating that pair to `contradicts` yields precision **1.0**,
recall **1.0**, class_c **1.0**. Regression: `tests/test_cartographer_combined.py`.

**This essentially clears the §4 target** (class_c = 1.0 with non-trivial recall,
modulo one disputed label). Two honest caveats:

- **Thresholds are calibrated on the 13-pair set** (too small to be production
  values) — the *architecture* (gate + NLI channel + aspect band) is the
  contribution; the exact 0.13 / 0.45 / 0.5 need a held-out set. The band's
  water-divergence floor (0.134) sits perilously close to the unrelated ceiling
  (0.124), so the gate threshold especially is not yet robust.
- **The mid-band → divergent rule assumes** a related, non-propositional,
  not-highly-similar pair is a framing divergence. That holds here but needs
  validation against corroborating pairs that are topically related yet textually
  dissimilar (none such in this set).

The Bookkeeper admission gate (anchor verification, DAG invariants, provenance
stamp) is the next step — it now has a detector whose proposals are worth gating.

## 7. Out of scope (this doc)

- **Threshold calibration on a held-out set** and the Bookkeeper admission gate
  (§6a → next steps). The combined detector is built and near-target on the 13-pair
  set; its thresholds are calibrated in-sample and need a larger labeled set before
  they are production values.
- Cross-corpus / federated proposal (the relatedness gate is scoped intra-corpus
  first; the alignment stage is the same machinery at inter-node scope).
- Embedding-based relatedness (the gate is lexical/entity first for a
  deterministic, model-free harness; embeddings are the production upgrade).
