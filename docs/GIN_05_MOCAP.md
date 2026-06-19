---
tags: [GIN, research, architecture, networking, protocol]
updated: 2026-06-13
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN 05 — MOCAP

> Mesh-Optimised Content-Addressing Protocol. The transport-layer mechanism for moving verified content across constrained, intermittent links. This document is conceptual; the byte-level specification lives in [[GIN_ENG_00_Engineering_Register]] and nowhere else.

---

## The idea

MOCAP addresses content by what it *is*, not where it lives. Each chunk of corpus material is identified by a cryptographic hash of its contents. A node requesting material asks for a hash, not a location; any peer holding that chunk can serve it; the requester verifies the chunk against the hash on receipt. This is content addressing in the established sense (IPFS, Reticulum, named-data networking), credited to that literature rather than claimed as novel.

Content addressing earns its place in GIN for two reasons. First, **integrity**: a chunk that hashes correctly is the chunk that was published, which matters acutely given that every guarantee floats on corpus integrity ([[GIN_04_TRAC]], [[GIN_07_Governance_Validity]]). Second, **link tolerance**: on intermittent, low-bandwidth links, being able to fetch a verified chunk from whatever peer is reachable — rather than from a specific origin server — is what makes the network function at all.

---

## Why it suits constrained links

A content-addressed chunk is self-verifying and origin-independent. That means it can travel by any path, including the physical-transport paths of [[GIN_06_Mule_Architecture]], and be reassembled and verified at the destination regardless of how it arrived. The same chunk fetched over a fast research backbone or carried on a storage device across a connectivity gap is the same verified chunk. This origin-independence is what lets GIN treat radically different transport media as one logical network.

---

## Engineering issues (not specs)

- **Frame overhead.** The clean packet arithmetic of earlier drafts (header + hash + payload summing to a round chunk size) was chosen to look neat, not measured. Real Reticulum link-frame overhead — destination hash, context byte, hop count, MAC — runs larger in most configurations. The assumed frame layout must be cited and measured.
- **Duty-cycle limits.** Sub-GHz ISM bands carry strict regulatory duty-cycle caps (EU 868 MHz at 1%, regional US 915 MHz caps). A node serving even modest daily users may hit duty-cycle limits before throughput limits. A serious capacity analysis is owed.
- **Chunk-size tuning.** The trade-off between chunk size, overhead ratio, and retransmission cost on lossy links is uncharacterised.

(Byte layouts, throughput figures, and duty-cycle math belong in [[GIN_ENG_00_Engineering_Register]].)

## Related

[[GIN_00_Reader]] · [[GIN_04_TRAC]] · [[GIN_06_Mule_Architecture]] · [[GIN_ENG_00_Engineering_Register]]

## Back to Vault

[[HOME]]
