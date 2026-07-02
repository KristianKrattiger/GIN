---
tags: [GIN, research, architecture, inference, ml]
updated: 2026-06-13
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN 04 — SEAR

> Sparse Epistemically Anchored Reasoning. The inference layer that constrains a node's model to extractive synthesis from verified corpus material. Reframed around **structural fidelity** rather than the over-strong "deterministic," and in v0.4 specified across both modes of the duality.

---

## What SEAR does

SEAR constrains the generation process so that a node's output is assembled from, and grounded in, material that actually exists in its verified corpus. Grammar-constrained decoding restricts the model's token choices to extractive synthesis; attention is masked to retrieved context; ungrounded generation is structurally prevented rather than discouraged. When the corpus does not support an answer, SEAR emits an explicit failure state instead of fabricating fluent text.

The point is to make a node's reading of its corpus *faithful to that corpus*. Hallucination — confident generation untethered from any source — is the failure mode SEAR is built to eliminate at the architectural level, not at the prompt level.

---

## Structural fidelity, not determinism

v0.2 called this "deterministic." That word overpromised and v0.3 retired it; v0.4 keeps it retired.

Even with grammar-constrained extractive decoding, three things remain consequential and non-deterministic: *which* passages are retrieved, in *what order* they are assembled, and *which* extractive paraphrases the grammar admits. A constrained model can still mislead through selection and framing without ever violating its grounding.

The accurate claim is **structural fidelity**: SEAR guarantees that output is *structurally grounded* in corpus material — every claim traceable to a source the node holds — but it does not guarantee *truth*, and it does not guarantee freedom from selection bias. Structural fidelity is a real and valuable property. It is also narrower than "deterministic" implied, and stating the narrower claim is what makes it defensible.

---

## The honest limit, stated up front

**If a corpus is captured or poisoned, SEAR will output the distortion with full structural grounding and high confidence.** Structural fidelity to a corrupted corpus faithfully reproduces the corruption. Every technical guarantee in GIN floats on top of corpus integrity, which the architecture cannot technically enforce — only govern ([[GIN_07_Governance_Validity]]). This is not a weakness to hide; it is the boundary that defines what kind of system GIN is.

---

## SEAR across the two modes

v0.4 specifies how SEAR behaves under the convergent/divergent duality ([[GIN_02_Productive_Divergence]]). The decoding constraint is identical in both modes — structural grounding in verified corpus material. What differs is what the corpus holds and what synthesis across nodes produces.

**Divergent mode.** The corpus holds situated accounts. Cross-node synthesis surfaces and attributes conflict; SEAR assembles a structured map of disagreement, each perspective in its own terms. Selection and framing risk is high and is the central thing the validity layer must watch.

**Convergent mode.** The corpus holds verified empirical knowledge and data. Cross-node synthesis aggregates concentration: derivation chains, experimental results with methodology and raw data, replication records. SEAR assembles a high-integrity research synthesis with full provenance. Structural fidelity is doing its strongest work here — an agent cannot fabricate a citation or invent a result, because ungrounded generation is structurally prevented. The value density is higher in this mode, which raises the stakes on corpus integrity and on the agentic access controls of [[GIN_09_Agentic_Layer]].

The boundary between "needs the model" and "needs only retrieval" remains an open design question, sharper in convergent mode where much of the work is location and verification rather than fluent situated synthesis.

---

## The agentic / friction-dial layer

SEAR output density is adjustable. For research workflows and agent-to-agent consumption, it produces dense, fully-attributed, frictionised reports — every claim sourced, every cross-node disagreement surfaced. For casual human communication, it produces smoother synthesis. The dial adjusts *presentation density*, never *grounding* and never the surfacing of legitimate conflict. A smoother output is still structurally faithful and still honest about disagreement; it is merely less dense. The constraint that the dial cannot collapse a legitimate conflict into false consensus is a hard rule, not a preference.

v0.4 moves the full treatment of agentic consumers — who may operate them, how access is restricted, and how the dial is prevented from being abused to launder frictions downstream — into its own document, [[GIN_09_Agentic_Layer]]. The hard rule above is the local guarantee; the agentic layer is where it is enforced against external agents at scale.

---

## Caliche binding

SEAR is the *remnant* principle from [[CALICHE_INDEX|Caliche]] rendered as inference: *we use remnants to create something new.* The corpus is the bolt-end of inherited material; SEAR synthesises from it without fabricating new cloth, and attributes what it draws on. Extractive synthesis with attribution is the inheritance-and-acknowledgement ethic expressed in a decoding constraint.

---

## Engineering issues (not specs)

- **Selection-bias measurement.** Structural fidelity does not bound selection bias. Epistemic metrics (`query_relevance_rate`, `supported_irrelevance_rate`, `gold_chunk_coverage`) now measured on the synthetic eval corpus — NC meets targets on full 20-query run `20260702T012203Z` ([[GIN_ENG_02_Eval_Baseline_v1]]). Generalization beyond synthetic corpus TBM.
- **Retrieval/synthesis boundary.** Where exactly does a query stop needing the model and need only retrieval? Undefined; sharper in convergent mode.
- **Failure-state UX.** An explicit "the corpus does not support this" is correct but must be presented so users read it as information, not as system failure. Design problem.
- **Grammar expressiveness vs fidelity.** A more permissive extractive grammar produces more fluent output but admits more paraphrase drift away from sources. The trade-off needs characterising.

(Latency targets, model sizes, decoding implementation, and grammar definitions belong in [[GIN_ENG_00_Engineering_Register]].)

## Related

[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_03_Node_Identity]] · [[GIN_07_Governance_Validity]] · [[GIN_09_Agentic_Layer]] · [[CALICHE_INDEX]]

## Back to Vault

[[HOME]]
