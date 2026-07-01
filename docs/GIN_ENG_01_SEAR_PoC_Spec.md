---
tags: [GIN, engineering, SEAR, poc, spec, register, ssa]
updated: 2026-06-29
version: 0.2-poc
status: working draft
register: engineering
implements: GIN v0.4 — SEAR ([[GIN_04_SEAR]])
---

# GIN ENG 01 — SEAR Proof-of-Concept Specification & Roadmap

> The reality-grounded build plan for [[GIN_04_SEAR|SEAR]]. This document specifies a verify-gate inference layer on a frozen open-weights model as the first measurable stage, the test corpus that makes the grounding guarantee measurable, and a staged roadmap shaped by the long-horizon design direction: a **hybrid subquadratic sparse attention (SSA) architecture** that externalises working memory to global anchor tokens and treats attention weights as grounding certificates. Every design choice in the early stages is made with that destination in mind.

-----

## 0. Promotion-rule notice

Per [[GIN_ENG_00_Engineering_Register]], **nothing in this document is a specification yet.** Every number, target, and threshold below is *unmeasured* — a hypothesis to be tested, not a result. A line graduates to a stated spec only when (1) measured on the representative rig defined in §3, (2) with method recorded, (3) reproducibly. Until then it is an engineering target. Where a figure would normally go, it reads **TBM** (to be measured).

-----

## 1. What this PoC proves — and what it does not

**The one job.** Demonstrate that a frozen LLM, wrapped in SEAR, will emit only claims traceable to retrieved corpus material, and will emit an explicit *"the corpus does not support this"* failure state instead of fabricating when it cannot ground an answer. Everything else in [[GIN_04_SEAR]] — the friction dial, the two modes, selection-bias measurement, federation — is downstream of whether that core grounding guarantee holds on real hardware.

**In scope (v0.1 / Stages 0–1b).**
- Single-node, **convergent** mode.
- The **verify-gate** mechanism: generate → check each claim against its cited source → emit or fail.
- A measured **fabrication rate**, measured **counterfactual adherence**, and a working, measured **failure state**.
- **Attention-weight baseline** recorded from Stage 0 onward, in preparation for Stage 1c.

**Explicitly out of scope (v0.1).**
- Decode-time structural prevention (Stage 1b).
- Inference-time SSA prototype (Stage 1c).
- Divergent mode and multi-node behaviour (Stages 2–3).
- The friction dial, the agentic/MCP surface ([[GIN_09_Agentic_Layer]]), Council governance ([[GIN_10_Epistemic_Council]]).
- Throughput-at-scale and latency-under-concurrency figures.

**Honest framing.** The verify-gate is generate-then-check — the "discouraged" approach SEAR's own prose distinguishes itself from. It is chosen deliberately as **measurement infrastructure**: establish the metrics harness and a fabrication-rate floor, then measure the *delta* when decode-time constraint (Stage 1b) and SSA (Stage 1c) are added. Each delta is a result. The verify-gate baseline is what makes those results legible.

-----

## 2. The mechanism axis

SEAR's claim is *architectural prevention* of ungrounded generation. The PoC starts at the soft end and moves right, measuring what each move buys:

| | Verify-gate (Stage 1) | Decode-time constraint (Stage 1b) | Hybrid SSA (Stage 1c+) |
|---|---|---|---|
| Mechanism | Generate with citation format; verify each claim post-hoc; gate on failure | Corpus automaton masks logits at each step; output can only continue spans present in retrieved corpus | Global anchor tokens per chunk; sparse attention (local window + global); attention weights *are* the grounding certificate |
| Guarantee | Soft — enforced after generation | Hard — ungrounded continuations unreachable | Structural — grounding is a geometric property of the attention space |
| Fluency | High | Low → stilted | Tunable — connective tokens relax the constraint without breaking grounding |
| Novelty | Low (cited-RAG with verifier) | High — SEAR thesis v1 | Highest — **SEAR thesis v2, the design destination** |
| Bottleneck | Verifier passes (GPU) | CPU-side automaton lookup | Global token count × attention heads |
| Divergence enforcement | None | None | Native — minimum attention mass on conflicting global tokens structurally enforces conflict surfacing |

The rightmost column is the design horizon that shapes choices from Stage 0 onward.

-----

## 2b. Design horizon — Hybrid SSA Architecture

This section states the long-horizon architecture so that every earlier stage decision is made with it in mind. Nothing here is measured; everything here is named as the engineering destination.

### The core idea

In standard RAG the model holds retrieved chunks as dense context and attends to them freely. With hybrid SSA, the architecture restructures this:

- **Global anchor tokens** — one designated token per retrieved chunk, attending globally (to and from all positions). These are the externalized working memory of the system. The model's knowledge of the corpus at inference time lives in these tokens, not in its weights.
- **Local window attention** — generation tokens attend within a local neighbourhood for coherence, plus globally to all anchor tokens for corpus grounding. Cross-chunk reasoning routes through the anchors, not through dense cross-chunk attention.
- **No dense cross-chunk attention** — the O(n²) cost of attending across the full retrieved context is replaced by O(n·k) where k is the number of anchor tokens, which is small.

### Attention as grounding certificate

The critical property: **a generated claim token's attention distribution over global anchor tokens is its provenance trace.** A claim token attending 85% to ANCHOR_3 was drawn from chunk 3 — not by post-hoc citation format, but as a geometric property of the forward pass. The citation is intrinsic, not appended.

This collapses the generate → cite → verify pipeline into a single forward pass where grounding is expressed in the attention weights themselves. The verify-gate (Stage 1) measures the baseline against which this is an improvement; Stage 1c measures the delta.

Honest caveat: attention weights are an imperfect proxy for token importance — the literature shows attention ≠ causal importance in all cases. "Attention as certificate" is the architectural goal, not a proven property of existing models. The Stage 1c metric (attention-weight grounding correlation vs. NLI baseline) exists precisely to validate this empirically before asserting it.

### Structural divergence enforcement

In divergent mode (Stage 3+), conflicting accounts from two nodes are represented as two competing global anchor tokens (ANCHOR_A, ANCHOR_B). A minimum-attention-mass constraint requires generation tokens to attend to both above threshold. The model cannot collapse to one source because the attention geometry enforces it. This converts the GIN_04 hard rule — "the dial cannot collapse a legitimate conflict" — from a software assertion into a structural property of the attention space.

### The two-model access stack

SSA at the node level enables a clean two-layer interaction model for researcher access:

**Layer 1 — SEAR node model (small, corpus-specialized).** A lightweight reasoning-focused model running SSA over the node's corpus. Produces the dense frictionized report: every claim grounded, every conflict surfaced, full provenance via attention weights and citation handles. This is the epistemic artifact.

**Layer 2 — Reasoning agent (larger, general).** Takes the frictionized report as *input*, not the corpus directly. Reasons over the verified artifact to produce a query-specific, less-dense response. All citations from the report are inherited. The original frictionized report travels with every response as an inspectable artifact.

The key architectural property: **grounding guarantees were enforced at Layer 1 before the reasoning agent sees anything.** The agent cannot launder friction that is already locked into the artifact it is reasoning from. This solves the friction-laundering problem ([[GIN_09_Agentic_Layer]]) at the architecture level rather than through behavioural monitoring. The original report is always surfaced, making any misrepresentation by the agent immediately auditable.

### Model optimization direction

When working memory is externalized to global anchors, the base model's job becomes: route attention correctly across anchors, detect entailment and contradiction between them, compile output faithful to attended anchors. This task profile does not require encyclopedic parametric knowledge — it requires precise reasoning over provided context. Implications:

- A smaller model specifically trained for reasoning-over-context may outperform a larger generalist model at this task because the generalist's parametric knowledge actively fights the corpus on counterfactual probes.
- The VRAM tradeoff on a 4070 is concrete: a 4-bit 3B model (~1.5–2 GB) leaves ~9–10 GB for corpus context versus ~6–7 GB for a 7B model. The smaller model may hold significantly more global anchor tokens — more corpus in view — at the cost of reasoning depth. The sweet spot is empirically determined by the grounding metrics.
- Once the SEAR task profile is clearly defined by the PoC metrics, it becomes a well-specified optimization target for the broader open-source ML community. GIN benefits from community optimization without having to run it.

### What "SSA-aware" means for early-stage design choices

Because the SSA architecture is the destination, certain choices made in Stages 0–1 should be compatible with it rather than arbitrary:

1. **Chunk sizing with anchor representability**: chunks should have natural semantic boundaries and be sized so a single anchor token can meaningfully represent the chunk's epistemic content (~150–300 tokens as a working target, TBM). Too small = too many anchors, attention diluted. Too large = anchor token cannot represent the chunk cleanly.
2. **Top-k as future global anchor count**: the number of retrieved chunks becomes the number of global anchors. Design top-k with SSA context window in mind — a consistent range of 10–20 is a working target (TBM).
3. **Stable chunk IDs**: citation handles `[chunk_id]` must map stably to chunks across queries. The same chunk always has the same ID. This is essential when attention weights eventually replace post-hoc citation format.
4. **Attention weight logging from Stage 0**: record model attention distributions over retrieved chunks during every generation pass from the start of the harness. This establishes the pre-SSA baseline that Stage 1c measures against.
5. **Model selection**: prefer models with demonstrated strong in-context reasoning over long context rather than encyclopedic recall. Models with clean sparse-attention fine-tuning histories are preferred; see §3.

-----

## 3. Representative rig (the measurement baseline)

- **GPU:** RTX 4070, 12 GB VRAM. Commodity target by design. Note: VRAM split between base model weights and corpus context is a live design variable; §2b discusses the tradeoff explicitly.
- **Base model:** 7–8B instruct model at 4-bit quantization (AWQ or GPTQ) for Stages 0–1b. Selection criteria: strong in-context reasoning over long context (not encyclopedic recall), clean instruction-following, first-class HF `transformers` tooling, good sparse-attention fine-tuning history. Model selection is deferred until the stack is locked. *(A current shortlist can be pulled at commit time.)*
  - *SSA-horizon note:* Stage 1c will test whether a smaller model (3B range, 4-bit) with more VRAM allocated to corpus context outperforms the 7–8B model on grounding metrics. Both configurations should be measurable on the same rig.
- **Decoding surface:** HuggingFace `transformers` + custom `LogitsProcessor`. Full per-token control is necessary because the constraint logic is the artifact. Migration to vLLM / XGrammar is deferred to latency-at-scale stages.
- **Retrieval:** sentence-transformer embeddings + FAISS or Chroma. Chunk sizes per §2b (150–300 tokens, TBM). Stable chunk IDs required from Stage 0.
- **Verifier:** dedicated NLI/entailment model (claim ⊨ cited chunk?). LLM-as-judge is a fallback with its own ungrounding risk; prefer a dedicated entailment model first.
- **Attention logging:** a lightweight hook on the attention mechanism recording per-layer, per-head attention weight distributions over retrieved chunk positions. Required from Stage 0; forms the baseline for Stage 1c.

-----

## 4. Architecture — the verify-gate pipeline

```
query
  │
  ▼
[1] RETRIEVE      embed query → top-k chunks from vector store
  │               (log chunk IDs; record attention baseline hook)
  ▼
[2] GENERATE      constrained format: every sentence/claim must carry a
  │               citation handle [chunk_id] pointing at a retrieved chunk
  │               (enforced by the LogitsProcessor / grammar)
  │               (record attention weights over chunk positions per token)
  ▼
[3] SEGMENT       split output into atomic claims, each with its cited handle
  │
  ▼
[4] VERIFY        for each claim: does the cited chunk entail it?
  │               (NLI score / span-overlap ≥ threshold τ)
  │               (compare: did attention to cited chunk correlate with pass/fail?)
  ▼
[5] GATE          • all claims verified  → emit
                  • some fail            → drop or regenerate those claims
                  • verified fraction < φ → EMIT EXPLICIT FAILURE STATE
                    ("the corpus does not support this")
```

**Design points that will fight you:**
- **Claim segmentation** is harder than it looks. Start crude (sentence-level), refine toward proposition-level. The SSA architecture eventually makes this cleaner because claim tokens self-identify via attention pattern.
- **Citation faithfulness ≠ format compliance.** The model can emit a well-formed `[chunk_id]` that doesn't support the claim. The verifier catches this, not the grammar. Also: attention-weight logging will reveal whether the model was genuinely attending to the cited chunk when it generated the claim, or whether the citation was a format artefact. This is the first signal on whether attention-as-certificate has any empirical basis.
- **Failure-state UX.** Per GIN_04, "not supported" must read as information, not a broken system. Log it as a first-class outcome from Stage 0.
- **Chunk ID stability.** Assign stable IDs at corpus build time, not at retrieval time. Required for SSA compatibility in later stages.

-----

## 5. Corpus design

Hand-built. You are the ground-truth oracle for a domain you know cold. **Recommended domain: film** (carries both convergent and divergent modes across the roadmap; cars is the sharper convergent grounding alternative for Stages 0–1b). Scale: hundreds to low-thousands of chunks is sufficient to test mechanics.

Chunk structure to enforce from the start:
- Natural semantic boundaries (scene, argument, event — not arbitrary token windows).
- A designated **head sentence** per chunk: the most representative sentence. This becomes the global anchor token proxy in Stage 1c.
- Stable chunk IDs assigned at corpus build time.

**Three layers:**

1. **Realism layer.** Hand-selected real material. Exercises chunking, retrieval, and normal in-corpus answering.

2. **Counterfactual probes.** Deliberately altered verifiable details so the corpus contradicts the model's memorized prior. Does SEAR follow the corpus or its weights? This is the [[GIN_07_Governance_Validity]] captured-corpus limit in miniature — grounding test and poison demonstration in one mechanism. Also used in Stage 1c: does SSA attention-as-certificate show higher fidelity on counterfactual probes than the verify-gate baseline?

3. **Out-of-corpus probes.** Adjacent questions the corpus cannot answer. Correct behaviour is the explicit failure state. Half the headline result; most RAG eval skips it.

**Forks for later stages (build the structure now):**
- **Clean / poisoned fork** → Stage 1b/2 poison demonstration.
- **Conflicting-pair corpora** → Stage 3 divergence test. Design these with the SSA minimum-attention-mass constraint in mind: the two conflicting accounts should be in separate chunks with separate IDs so they map to separate global anchor tokens.

-----

## 6. Metrics

Defined up front so the promotion rule can later apply. Headline metrics in **bold**. SSA-specific metrics added for Stage 1c onward.

| Metric | Definition | Stage |
|---|---|---|
| **Fabrication rate** | fraction of emitted claims not entailed by any retrieved chunk | 1 |
| **Counterfactual adherence** | fraction of counterfactual probes where output follows corpus, not prior | 1 |
| **Failure-state precision/recall** | on out-of-corpus probes: correct refusal vs. fabricated guess | 1 |
| Attribution coverage | fraction of emitted claims carrying a valid (verified) citation | 1 |
| **Attention-weight grounding correlation** | correlation between model attention mass on cited chunk and NLI verification pass/fail; baseline pre-SSA | 1 (baseline) |
| Faithfulness↔fluency | readability vs. grounding rate across axis positions | 1b |
| **Prevention delta** | fabrication rate + refusal accuracy: verify-gate vs. decode-time constraint | 1b |
| **SSA grounding delta** | attention-weight grounding correlation: frozen dense vs. inference-time SSA mask | 1c |
| SSA VRAM tradeoff | grounding rate vs. model size at fixed VRAM; 3B vs. 7B configurations | 1c |
| Divergence fidelity | fraction of conflicting-pair queries where both accounts are faithfully surfaced without averaging | 3 |
| **Attention-mass conflict enforcement** | minimum attention mass on both conflicting anchors maintained above threshold in divergent mode | 3 |
| Citation chain depth | agent response → frictionized report → corpus chunk → node corpus; full chain traceable | 4 |
| Latency | per-query wall-clock on the §3 rig, by stage | all |
| Selection bias | instrument retrieval only in v0.1; measurement deferred (GIN_04 open issue) | — |

-----

## 7. Roadmap

**Stage 0 — Harness.** Retrieval + base model + metrics pipeline + three-layer corpus. Attention-weight logging hook installed from day one — not optional, required for Stage 1c baseline. Stable chunk IDs assigned. Exit: end-to-end query scores automatically; attention distributions recorded per query.

**Stage 1 — Single-node convergent verify-gate.** Full pipeline of §4. Exit: measured fabrication rate; failure state fires correctly on out-of-corpus probes; counterfactual-adherence measured; **attention-weight grounding correlation baseline established.** *This is where SEAR is or isn't real.*

**Stage 1b — Decode-time constraint + delta.** Build the corpus automaton (Aho-Corasick / suffix-automaton / FM-index) and constrained decode loop. Measure **prevention delta** vs. Stage 1 and fluency cost. Exit: defensible answer to "does structural prevention beat verify-gate enough to justify its cost?" — the result that earns SEAR the "structurally prevented" language.

> **Implementation note (2026-06).** The cursor-based decode-time constraint described here is implemented in this repository as **SEAR** (`sear/processor.py`, `sear/corpus.py`) running on llama-cpp-python rather than HuggingFace `transformers`. The mechanism differs from the Aho-Corasick automaton approach (SEAR uses live `(doc_id, position)` cursor pairs rather than a compiled string automaton), but satisfies the same structural guarantee: ungrounded continuations are masked to `-1e30` and are unreachable. The **layered provenance record** (`gin/corpus/retrieval_manifest.py`, `gin/corpus/synthesis_manifest.py`) now provides the metrics harness infrastructure for the §6 measurements — retrieval manifest hash, per-span guidance tags, and grouped confidence signals are recorded at every synthesis event.

**Stage 1c — Inference-time SSA prototype.** Apply sparse attention mask to the frozen model: local window within generation, full attention to head-sentence-of-chunk as global anchor proxy. Measure **SSA grounding delta** vs. Stage 1 attention-weight correlation baseline. Measure latency. Measure VRAM tradeoff (3B vs 7B). Cost: days, not weeks. Purpose: cheap empirical signal on whether the SSA structural idea is sound before investing in fine-tuning. Exit: evidence either supporting or falsifying the attention-as-certificate hypothesis.

**Stage 2 — Two-node convergent.** Federate two complementary verified corpora; merge provenance across nodes; no conflict yet. SSA-aware: each node's retrieved chunks contribute anchor tokens to a shared global attention space; measure whether grounding survives cross-node retrieval. Exit: cross-node attribution plumbing works; grounding guarantee survives federation.

**Stage 2b — SSA fine-tuning validation.** Fine-tune a small model (3B range) with sparse attention patterns active, using the SEAR grounding corpus generated by Stages 0–1c as training data. The PoC generates its own fine-tuning signal. Measure grounding improvement over frozen-model SSA (Stage 1c); measure whether the model learns to use global anchor patterns effectively. Exit: evidence on whether SSA trained-in beats SSA bolted-on; defines the model optimization direction for GIN nodes.

**Stage 3 — Two-node divergent.** Conflicting-pair corpora. SSA minimum-attention-mass constraint active: model must maintain attention above threshold on both conflicting anchor tokens simultaneously in divergent mode. Measure **divergence fidelity** and **attention-mass conflict enforcement**. This is the real test of productive divergence — and the first test of the SSA architecture's ability to enforce the GIN_04 hard rule geometrically rather than by assertion. Exit: the dial cannot collapse a legitimate conflict; measured, not claimed.

**Stage 4 — Two-model access layer.** Deploy a reasoning agent (larger general model) over frictionized multi-node SEAR reports from Stage 3. Validate the frictionized-report-as-traveling-artifact design: the agent receives the report as input, not the corpus; produces a query-specific tailored response; all citations from the report are inherited; the original report is surfaced alongside the response. Measure **citation chain depth** (agent → report → chunk → corpus). Validate that friction-laundering is architecturally blocked by the artifact structure rather than by behavioural monitoring. Exit: four-level citation chain measured end-to-end; friction-laundering solution validated.

**Stage 5 (horizon) — GIN-native model training.** A purpose-trained lightweight reasoning model optimized from scratch for the SSA task profile: reasoning over provided context, entailment, contradiction detection, calibrated uncertainty, minimal parametric leakage. Training objective explicitly differs from standard pre-training: no encyclopedic knowledge instillation; corpus-grounded generation with counterfactual probe signal trained in; divergence surfacing as a positive capability. This is the long-horizon research direction and is not scheduled — it follows from the empirical results of Stages 1c and 2b.

-----

## 8. Known limitations — stated honestly

- **Verify-gate is generate-then-check**, not architectural prevention. v0.1 does not validate SEAR's distinctive thesis — that is Stage 1b's job.
- **Inherited base-model priors.** The frozen model's biases sit beneath everything (GIN_08 critique 1). Counterfactual probes measure leakage but cannot remove it. A GIN-native trained model (Stage 5) is the eventual answer; the PoC stages quantify the problem.
- **Inference-time SSA on a frozen model may degrade quality.** The model was not trained for sparse patterns; quality degradation is possible. Stage 1c exists to measure this empirically before asserting the benefit. If degradation is significant, Stage 2b (fine-tuning) is the answer.
- **Attention weights as grounding certificates are an imperfect proxy.** The literature on attention interpretability is mixed — attention ≠ causal importance in all cases. The attention-as-certificate property is the architectural goal, not a given. Stage 1c's attention-weight grounding correlation metric is the empirical test; do not assert the property before measuring it.
- **The reasoning agent (Stage 4) has its own parametric biases.** It operates over a grounded artifact but may emphasise or misrepresent it. The traveling-report design makes this auditable; the interface must make inspecting the original report genuinely easy, not just technically possible.
- **Selection bias is unmeasured.** Structural fidelity bounds fabrication, not framing. GIN_04 open issue; deferred.
- **Corpus integrity is governance, not engineering** ([[GIN_07_Governance_Validity]]). A clean grounding result on a poisoned corpus faithfully reproduces the poison. The PoC demonstrates this limit; it does not solve it.

-----

## 9. Prerequisite knowledge (learn these deliberately)

**Core four — required for Stages 0–1b:**
1. **Decoding mechanics** — logits, greedy vs. sampling, where a `LogitsProcessor` sits in the generation loop. The surface being modified.
2. **Textual entailment / NLI** — the verify-gate is an entailment check at heart.
3. **Embeddings & vector similarity** — the retrieval substrate.
4. **Standard RAG failure modes** — retrieval miss, context dilution, lost-in-the-middle.

**Add before Stage 1c:**
5. **Sparse attention architectures** — Longformer (local window + global tokens) and BigBird are the direct precedents. Read these before implementing Stage 1c; the pattern is established and the failure modes are documented.
6. **Attention weight interpretation** — the literature on whether attention weights reflect causal importance (Jain & Wallace 2019; Wiegreffe & Pinter 2019 are the key papers). Know the limitations before treating attention as a grounding certificate.

Everything else can be picked up reactively as the build surfaces it.

-----

## Related

[[GIN_ENG_00_Engineering_Register]] · [[GIN_04_SEAR]] · [[GIN_02_Productive_Divergence]] · [[GIN_07_Governance_Validity]] · [[GIN_03_Node_Identity]] · [[GIN_09_Agentic_Layer]] · [[GIN_00_Reader]]

## Back to Vault

[[HOME]]
