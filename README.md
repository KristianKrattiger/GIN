# GIN — Grounded Intelligence Network

## What this is

GIN is a federation of independent, place-rooted reasoning nodes that ground every claim in a traceable corpus, hold their disagreements legible instead of dissolving them, and arrive at convergence — when they do — relationally rather than by central decree. It solves the problem of language model outputs that cannot be traced to verifiable source material: attribution in GIN is exact by construction, enforced at decode time, not inferred post-hoc from attention weights. Édouard Glissant's right to opacity — the refusal to treat total transparency as a condition of participation — is the philosophical charter for the network's design: nodes relate across their differences without being required to expose or homogenize their corpora.

## SEAR — Sparse Epistemically Anchored Reasoning

SEAR is the inference discipline that makes GIN honest by architecture rather than by instruction: the model may only emit token spans that occur verbatim in the corpus, and each emitted span carries a pointer back to its source position. Live cursors `(doc_id, position)` track which source spans remain consistent with what has been emitted so far; the legal next token at each decode step is the union of whatever tokens sit at `position + 1` across all live cursors. Attribution is exact by construction — the cursor set surviving to end-of-span is precisely the set of source documents the model drew from, a different epistemic guarantee than post-hoc attention tracing. Zero live cursors is a first-class signal: it means the corpus cannot support the current continuation, and it triggers either graceful termination or federation routing to a peer node whose corpus can ground the claim.

## Architecture

SEAR operates across three layers. The Relation-Finder proposes typed epistemic edges (`cites`, `contradicts`, `supersedes`, `translated_from`) between documents or across corpora; it does not write edges. The Bookkeeper is the sole admission gate: it maintains canonical graph state, verifies anchor integrity, enforces DAG invariants, and stamps provenance — it makes nothing. The Reasoning layer is a strictly read-only consumer of the verified graph; it produces grounded answers and may feed new proposals back through the discovery pipeline, but it never writes canonical edges. This separation makes each layer independently falsifiable and prevents the reasoning model from inflating its own grounding record.

## Node topology

Tier 1 nodes are institutional anchors — universities, archives, research consortia — running full four-tier corpus stacks (hot vector index, warm structured records, cold content-addressed archive, graph layer) and 14B–70B base models fine-tuned locally. Tier 2 nodes are relays: fairlady, a Beelink mini PC running EndeavourOS on the Tailscale mesh, is the reference Tier 2 deployment, sitting between household thin clients and the wider federation with a 7B–14B model and a larger anchor cache. Tier 3 nodes are handheld and household clients running 1B–8B quantized models with personal corpus caches and an offline-first sync posture. The federation propagates anchored diffs, not corpora — nodes synchronize knowledge graph metadata (topic fingerprints, cursor density estimates, staleness timestamps) via Merkle-tree diffing, so each node retains sovereignty over its own corpus while the network stays coherent.

## Current build status

- ✅ SEAR Phase 1 scaffold validated (self-test passes, cursor logic correct)
- 🔲 Synthetic corpus — verify fan-out/prune under long shared spans, specify zero-cursor fallback
- 🔲 Live Mistral integration via llama-cpp-python (~half-day from current scaffold)
- 🔲 DRAC grounding rate measured against RAG baseline — the number that makes GIN real
- 🔲 Two-node divergence demo — same machinery, scope dialed to inter-corpus
- 🔲 Bookkeeper + reasoning layer separation (Phase 2)
- 🔲 Federation routing with sync metadata (Phase 3)

## Stack

- Python
- llama-cpp-python
- Mistral-7B-Instruct-v0.3 Q4_K_M GGUF
- Tailscale mesh
- fairlady (Beelink mini PC, EndeavourOS) as inference host

## Getting started

See `scripts/run_phase1.py`. Model setup instructions TBD. Run tests with `pytest`.
