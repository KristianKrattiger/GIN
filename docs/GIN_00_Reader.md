---
tags: [GIN, research, index, architecture, caliche]
updated: 2026-07-20
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN v0.4 — Reader

> The v0.4 rework is not a single document. It is a set of preliminary mechanism papers, each developing one part of the architecture in depth, bound together by a single principle, a single philosophical ground, and — new in this revision — a single governing institution. This file is the map.

**Measured system (code + ops):** start with [`architecture.md`](../architecture.md) (per-layer graphs) and [`README.md`](../README.md) (commands + status table). The engineering register ([[GIN_ENG_00_Engineering_Register]]) tracks what has been promoted from quarantine with artifact paths.

---

## What changed from v0.3

v0.3 established the narrative reframe (GIN as public-interest knowledge infrastructure, not an internet replacement), the two-register rule (conceptual vs engineering), and productive divergence as the founding principle. All of that holds.

v0.4 adds three things that change what GIN *is*:

1. **The convergent/divergent duality.** GIN is not only a plurality engine. The same infrastructure runs in two modes. In **divergent mode** it preserves and surfaces situated disagreement for political, social, historical, and culturally-weighted questions. In **convergent mode** it acts as a concentrated, institutionally-verified, high-integrity knowledge web for empirical research in mathematics and the sciences — aggregating verified knowledge rather than surfacing perspective. Same transport, same governance, same nodes; radically different epistemic behaviour depending on what is asked. See [[GIN_02_Productive_Divergence]] and [[GIN_04_SEAR]].

2. **The agentic layer as first-class surface.** Other AI agents consuming GIN is no longer a footnote to SEAR. It is a primary product surface with its own access model, restricted to Tier 1 institutions, mediated by a GIN-native MCP server with behavioural controls, and governed rather than bolted on. See [[GIN_09_Agentic_Layer]].

3. **The GIN Epistemic Council.** The governance problem that v0.3 named as the honest centre of gravity now has an institutional answer: a standing inter-institutional body that holds the validity layer, classifies empirical vs situated, defines what falls outside both frames, and governs agentic access. This converts GIN from an architecture into an institution — with everything that implies. See [[GIN_10_Epistemic_Council]].

The v0.4 reframe in one line: **GIN is governed epistemic infrastructure — not a network, not a tool, but the substrate on which legitimate global knowledge production can run, and the institution that holds it accountable.**

---

## The registers

**Conceptual register** — mechanism papers, philosophy, adversarial analysis. May discuss problems; never lists specs.

**Engineering register** (separate) — the reality-grounded specifications: hardware models, protocol byte layouts, throughput figures, power budgets, duty-cycle math, routing algorithms. Held apart so that nothing in the conceptual register can be mistaken for a shipped design. See [[GIN_ENG_00_Engineering_Register]].

**Strategy register** (separate, added post-v0.4) — go-to-market, consulting, and market-positioning content: commercial *bets*, not mechanism claims or measured specs, and quarantined for the same reason the engineering register is. See [[GIN_STRAT_00_Strategy_Register]].

A claim only crosses from conceptual to engineering register when it has been measured, not when it has been argued. Strategy claims cross into neither — a bet stays a bet.

---

## The document set

| Document | Mechanism | Register |
|---|---|---|
| [[GIN_01_Foundations]] | Philosophical ground; the Caliche binding; the narrative reframe | conceptual |
| [[GIN_02_Productive_Divergence]] | The core principle; agonistic pluralism; empirical/non-empirical routing; the convergent/divergent duality | conceptual |
| [[GIN_03_Node_Identity]] | Epistemic-professional nodes; non-merging federated adapters | conceptual |
| [[GIN_04_SEAR]] | Constrained inference; structural fidelity; the two modes; the agentic friction dial | conceptual |
| [[GIN_05_MOCAP]] | Content-addressed transport over constrained links | conceptual |
| [[GIN_06_Mule_Architecture]] | Physical transport as a network layer; multi-modal DTN | conceptual |
| [[GIN_07_Governance_Validity]] | The validity layer; the non-pluralist core; corpus governance; held by the Council | conceptual |
| [[GIN_08_Adversarial_Analysis]] | Stated critiques and their answers | conceptual |
| [[GIN_09_Agentic_Layer]] | Agentic consumers; Tier 1 access; the MCP server; behavioural controls | conceptual |
| [[GIN_10_Epistemic_Council]] | The governing institution; classification authority; the donation target | conceptual |
| [[GIN_11_Comparative_Case]] | Why multi-institutional governed epistemic power beats the corporate baseline | conceptual |
| [[GIN_12_Ecosystem_Licensing]] | Scaling gradients; who buys in; corporate adoption; the licensing fork | conceptual |
| [[GIN_13_Temporal_Sensor_Grounding]] | Extending SEAR from editorial text to sensor ground truth; domain applications; calibration-as-governance | conceptual |
| [[GIN_ENG_00_Engineering_Register]] | Index and quarantine for reality-grounded specs; Phase 2/3 federation promotions recorded | engineering |
| [[GIN_ENG_01_SEAR_PoC_Spec]] | SEAR proof-of-concept specification; staged roadmap; all figures TBM | engineering |
| [[GIN_ENG_02_Eval_Baseline_v1]] | Measured RAG vs No-Continuation baseline; NC epistemic promotion; two-node + real-text divergence generalization | engineering |
| [[GIN_STRAT_00_Strategy_Register]] | Index and quarantine notice for go-to-market and consulting bets | strategy |
| [[GIN_STRAT_01_Consulting_Model]] | The consulting motion; the node-vs-client fit filter | strategy |
| [[GIN_STRAT_02_Strategic_Positioning]] | Sensor-first vs. AI-first; market bifurcation; timing bets | strategy |

---

## How to read this set

Read [[GIN_01_Foundations]] and [[GIN_02_Productive_Divergence]] first. Everything else is downstream of those two. The mechanism papers (03–06) each show how one component expresses the principle. [[GIN_09_Agentic_Layer]] and [[GIN_10_Epistemic_Council]] are the v0.4 additions and should be read after 07, since both are extensions of the governance problem it frames. [[GIN_11_Comparative_Case]] is the argument that *justifies* the institutional turn — why this location for epistemic power beats the corporate baseline — and [[GIN_12_Ecosystem_Licensing]] is its political-economy consequence: who adopts GIN, who funds it, and the undecided licensing fork that determines whether corporate adoption feeds or mines the commons. [[GIN_07_Governance_Validity]] remains the honest centre of gravity; the Council in [[GIN_10_Epistemic_Council]] is its institutional answer, not its dissolution. [[GIN_08_Adversarial_Analysis]] is the stress test.

If a reader takes only one sentence: *GIN operationalises agonistic pluralism as knowledge infrastructure for contested questions and verified concentration for empirical ones, governed by a standing inter-institutional council that holds the boundary between them.*

---

## Status

Preliminary. Working drafts. v0.4 supersedes v0.3 as the organising structure. It does not yet constitute a specification. The principle is being developed toward rigour; the engineering register toward measurement; the Council toward a charter. None is finished. The Council in particular is a *proposal*, not a constituted body — naming it does not make it exist, and the hardest questions about it (capture, legitimacy, the donation target) are open.

## Related

[[CALICHE_INDEX]] · [[PriceRiot_Overview]] · [[POLYMATHY_INDEX]]

## Back to Vault

[[HOME]]
