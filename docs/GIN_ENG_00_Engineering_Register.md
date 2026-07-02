---
tags: [GIN, research, engineering, specs, quarantine]
updated: 2026-07-02
version: 0.4-preliminary
status: working draft
register: engineering
---

# GIN ENG 00 — Engineering Register

> Index and quarantine. Every reality-grounded specification lives here and nowhere else, so that nothing in the conceptual register (GIN 00–10) can be mistaken for a shipped design. Every entry below is **unmeasured** until the promotion rule is satisfied.

---

## Why this register is separate

The conceptual papers argue. This register measures. Keeping them apart prevents an argued figure from being read as a measured one. A claim crosses into this register only when measured, never when argued.

---

## Spec categories (all unmeasured)

**Transport — MOCAP** ([[GIN_05_MOCAP]])
- Reticulum frame layout actually assumed; real link-frame overhead.
- Chunk size and overhead ratio on representative lossy links.
- Throughput figures per channel type.

**Transport — radio / duty cycle** ([[GIN_05_MOCAP]])
- Sub-GHz ISM duty-cycle math (EU 868 MHz 1%, US 915 MHz regional caps).
- Users-per-day a node can serve before hitting duty-cycle vs throughput limits.
- Power budget per node topology.

**Transport — physical / DTN** ([[GIN_06_Mule_Architecture]])
- Carrier scheduling and custody logistics.
- Latency distributions over intermittent topology.
- Storage-media models and capacity.

**Inference — SEAR** ([[GIN_04_SEAR]])
- Model sizes, adapter sizes, memory budgets.
- Grammar definition and decoding implementation. *(Cursor-based decode-time constraint implemented as SEAR in this repo — see [[GIN_ENG_01_SEAR_PoC_Spec]] Stage 1b note. Fabrication rate and prevention delta remain TBM.)*
- Latency targets per mode (divergent surfacing vs convergent synthesis).
- Selection-bias measurement method.
- SEAR grounding rate vs RAG baseline — **preliminary measurement recorded** in [[GIN_ENG_02_Eval_Baseline_v1]] (structural runs `20260701T192827Z` overlap, `20260701T194024Z` NLI; NC epistemic promotion `20260702T012203Z` full 20-query overlap on synthetic corpus). Prevention delta, failure state, epistemic metrics (query relevance, gold coverage, supported irrelevance, counterfactual adherence, divergence fidelity) measured on CPU; representative GPU artifact remains before full promotion rule.

**Federation** ([[GIN_03_Node_Identity]])
- Adapter-switching cost and concurrency limits.
- Federation breadth-control: how many nodes constitute a "full picture," selected how.

**(v0.4) Agentic / MCP server** ([[GIN_09_Agentic_Layer]])
- MCP server protocol surface and the published behavioural-control specification.
- Anomaly-detection thresholds for query-rate, breadth, corpus-mapping, and friction-laundering patterns — what measurable property separates legitimate dense research from probing.
- Concurrency and rate-limit budgets under many concurrent Tier 1 agents.
- Suspension and appeals workflow timing.

**(v0.4) Mode routing** ([[GIN_02_Productive_Divergence]], [[GIN_07_Governance_Validity]])
- How the empirical/situated classification is operationalised at query time given that the *criteria* are Council-set and non-neutral — i.e. the mechanism that applies a political decision, not the decision itself.

**(ENG 01) SEAR PoC spec** ([[GIN_ENG_01_SEAR_PoC_Spec]])
- All figures in [[GIN_ENG_01_SEAR_PoC_Spec]] are **TBM** (to be measured) under the promotion rule; that document is the staged build plan and metrics harness for SEAR, not a stated specification.

**(ENG 02) Eval baseline v1** ([[GIN_ENG_02_Eval_Baseline_v1]])
- RAG vs No-Continuation on synthetic corpus. Structural prevention measured (overlap fabrication 0.286 vs 0.000, run `192827Z`). NC epistemic targets met on expanded 20-query set (`20260702T012203Z`: query relevance 1.0, supported irrelevance 0, gold coverage 1.0, counterfactual adherence 1.0, fabrication 0, divergence fidelity 1.0). GPU reproducibility outstanding.

**Sustainability**
- Tier 1 standing costs: power, storage growth, curation labour.
- Break-even volume per deployment topology.
- Funding-mix model beyond device retail.

---

## The promotion rule

A line moves from "unmeasured" to a stated specification when, and only when:

1. it has been measured on representative hardware against representative channels;
2. the measurement method is recorded alongside the result;
3. the result is reproducible.

Until all three hold, the item remains an *engineering issue* (nameable in mechanism papers) rather than a *specification* (stated only here, once measured).

## Related

[[GIN_00_Reader]] · [[GIN_05_MOCAP]] · [[GIN_06_Mule_Architecture]] · [[GIN_09_Agentic_Layer]] · [[GIN_ENG_01_SEAR_PoC_Spec]] · [[GIN_ENG_02_Eval_Baseline_v1]]

## Back to Vault

[[HOME]]
