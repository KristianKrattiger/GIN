---
tags: [GIN, research, architecture, networking, dtn]
updated: 2026-05-30
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN 06 — Mule Architecture

> Physical transport as a network layer. Where links are absent or too constrained, content moves on storage carried by people and vehicles — a multi-modal delay-tolerant network. The lineage (data mules, DTN, sneakernet) is credited to existing literature.

---

## The idea

Not every connectivity gap can be bridged by radio. Where links are absent, intermittent, or duty-cycle-limited beyond usefulness, content moves physically: verified chunks ([[GIN_05_MOCAP]]) on storage media carried by buses, postal routes, field workers, or any regular human movement between nodes. Because chunks are content-addressed and self-verifying, the carrier need not be trusted — only the hash. Physical transport becomes just another link in the delay-tolerant network, with high latency and high bandwidth.

This is data-mule / DTN thinking, not a GIN invention. What GIN adds is treating physical transport as a *first-class network layer* coequal with radio and IP links, unified by content addressing, rather than as a fallback hack.

---

## The connectivity spectrum, and where GIN runs on existing backbones

GIN does not assume one transport medium. It spans a spectrum:

- **High-speed research backbones** (Internet2 in North America, GÉANT in Europe, and equivalents). Where these reach, GIN's transport constraints largely disappear. Federation queries resolve fast and agentic workflows ([[GIN_09_Agentic_Layer]]) become significantly more capable. GIN does not compete with these networks; it rides them. They are epistemically neutral transport — they move bits and do not touch what the bits mean — and GIN supplies the epistemic and governance layer above them ([[GIN_02_Productive_Divergence]], [[GIN_07_Governance_Validity]]). Internet2 member institutions are also natural Tier 1 node candidates: research universities with existing corpus infrastructure, governance, and technical capacity, which GIN inherits rather than rebuilds.
- **Constrained radio links** (sub-GHz mesh). The MOCAP regime of [[GIN_05_MOCAP]].
- **Physical transport** (this document). Where nothing else reaches.

---

## The honest geographic tension

The combination is powerful but uneven, and v0.4 names this rather than hiding it. The richest, fastest knowledge exchange happens where research backbones already reach — well-connected institutions in North America and Europe. The *most important* knowledge for GIN's mission — the situated knowledge currently most absent from global research — lives disproportionately where those backbones do not reach, in exactly the regions that depend on the constrained-link and physical-transport layers.

This raises a real question the architecture must answer rather than wave away: does GIN + research-backbone integration accelerate the already-connected institutions further and widen the gap? Or does the GIN epistemic layer specifically counteract that by making underrepresented corpus material valuable enough — in both divergent and convergent modes — that well-connected institutions have a genuine incentive to reach toward it? The design intent is the latter; whether the incentive is strong enough is unproven.

---

## Engineering issues (not specs)

- **Custody and chain-of-handling.** Untrusted carriers are fine for integrity (the hash protects it) but availability and timeliness depend on custody logistics that are unmodelled.
- **Routing over intermittent topology.** Scheduling chunk movement across a network whose links appear and disappear on human timescales is a hard DTN routing problem, only partially addressed by existing literature.

(Storage models, carrier scheduling, and latency figures belong in [[GIN_ENG_00_Engineering_Register]].)

## Related

[[GIN_00_Reader]] · [[GIN_05_MOCAP]] · [[GIN_09_Agentic_Layer]] · [[GIN_ENG_00_Engineering_Register]]

## Back to Vault

[[HOME]]
