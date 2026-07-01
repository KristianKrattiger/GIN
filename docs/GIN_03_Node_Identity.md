---
tags: [GIN, research, architecture, federation, ml]
updated: 2026-06-13
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN 03 — Node Identity

> How [[GIN_02_Productive_Divergence|productive divergence]] is implemented: nodes as epistemic professionals, holding non-merging adapters that preserve rather than average their situated knowledge.

---

## The epistemic professional

A Tier 1 node is not a generic assistant pointed at a regional corpus. It is an *epistemic professional*: an institution with domain expertise, a curated corpus, and accountability for what it holds. Source retrieval competence and topic expertise are what let each node speak with authority in its domain rather than as an interchangeable endpoint. A node's identity is the union of its corpus, its curation choices, and the institution accountable for both.

This framing does double duty across the duality ([[GIN_02_Productive_Divergence]]). In **divergent mode** the epistemic professional holds a situated perspective — a region's history, a community's account — and speaks it in its own terms. In **convergent mode** the same node structure holds verified empirical knowledge and data — a laboratory's results, a medical centre's records — and speaks with the authority of institutional verification rather than standpoint. The node architecture does not change between modes; what the node is accountable *for* does.

---

## Non-merging federated adapters

The mechanism is adapter switching, not weight averaging. Each node holds its own adapter over a shared base model. In-region queries use the node's own adapter; cross-regional queries load peer adapters explicitly. Adapters are never merged, because merging is averaging, and averaging destroys the divergence the architecture exists to preserve.

This is closer to a federation *directory* with semantic adapter-loading than to federated learning in the weight-sharing sense, and v0.4 states that honestly. The benefit is not improved shared weights; it is the preservation of distinct, attributable institutional voices that can be queried side by side.

---

## Engineering issues (not specs)

- **Adapter switching at scale.** Loading peer adapters on demand for cross-regional queries is plausible but unvalidated at scale. Open.
- **Base-model dependency.** All nodes sit on top of a shared base model (an open-weights model and its successors). Productive divergence operates in the adapters, but the foundational reasoning capacity is still shaped by a handful of labs. GIN constrains and adapts these models; it does not replace them. This dependency must be stated honestly — it is the axis along which GIN is *not* an answer to centralisation (see [[GIN_08_Adversarial_Analysis]]).

(Reality-grounded specifications — adapter sizes, memory budgets, model selection — belong in [[GIN_ENG_00_Engineering_Register]].)

## Related

[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_04_SEAR]] · [[GIN_07_Governance_Validity]]

## Back to Vault

[[HOME]]
