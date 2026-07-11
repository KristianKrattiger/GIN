# GIN — Session Synthesis
### Ungraduated Architecture Sessions → Monolith Record
*Status: Active Design | June 2026 | Monolith / GIN*
*Companion documents: GIN_The_Whole_Frame.md · GIN_Node_Architecture_v1.md · sear_phase1.py*

---

## Preamble — What This Document Is

This is the working record of architectural decisions developed across four conversation sessions that have not been graduated into standalone specification documents. The sessions cover: SEAR multi-agent layer separation, the logits processor design and language stack, federation routing via cursor collapse, and knowledge graph metadata sync with staleness-as-governance. A fifth section captures the current session's corpus strategy and the "warm rigor" characterization of the overall design.

Each section marks its session of origin. Decisions are recorded with their rationale. Open questions are named explicitly rather than smoothed over — that's GIN's own discipline turned back on the document.

---

## 1. SEAR Multi-Agent Architecture
*Session: Multi-agent routing architecture for corpus analysis*

### 1.1 The convergence point

The architecture moved from a MoE-style routing intuition toward a clean three-component separation. The key move was pulling edge-creation out of the reasoning layer entirely, which makes the knowledge graph an independent artifact — testable on its own terms regardless of reasoning quality, and impossible for the reasoning model to inflate its own grounding by minting edges.

### 1.2 The three components

**Cartographer**
Proposes typed epistemic edges at any scope — local (intra-corpus) or cross-corpus (inter-node). Does not make edges. Proposes them. The distinction is load-bearing: the Cartographer's reliability profile and test method (edge precision/recall) is different from the Bookkeeper's, and fusing them would make the graph untestable as an independent artifact.

The Cartographer runs a cheap relatedness gate before expensive pairwise alignment: embedding/entity/citation overlap assessed first. Negative assessments ("assessed, unrelated") are stored as valid graph content, not silence. If this isn't done, the same null gets re-litigated on every query.

Cross-corpus alignment is an explicit, expensive stage that belongs to the Cartographer — not hidden inside "receiving diffs." When a Tier 1 node queries a peer, the Cartographer owns the alignment work, proposes the edges it finds, and the Bookkeeper adjudicates.

**Bookkeeper**
Sole admission gate. Maintains canonical graph state. Verifies anchor integrity. Enforces DAG invariants. Stamps provenance. Makes nothing.

The admission gate is uniform for both local and federated edges — there is no separate local-trust / federated-trust path. This is the sovereignty membrane. The trust-link layer sits above it as a defined socket that governance fills later. The Bookkeeper is governance-ready without being governance-complete.

The Bookkeeper's stored assessments also function as a federation cache. The Cartographer only performs live cross-corpus jumps on cache misses or staleness — making most federated queries fast and cheap.

**Reasoning Layer**
Strictly read-only consumer of the verified graph. Produces grounded answers. Does not create edges.

When the reasoning model notices an n-hop conflict the pairwise extractor never surfaced, that finding does not become an edge directly. It re-enters as a proposal through the same verify path as everything else. Reasoning never writes canonical edges; it is allowed to feed the discovery pipeline. This keeps the loop closed without reopening the unfalsifiability problem.

### 1.3 Divergence and convergence fall out of edge types

Supersedes is an ordering — convergence resolves conflict along it; the superseding claim wins. Conflict with no ordering is genuine unresolved divergence, which the reasoning layer surfaces rather than collapses. This is not two passes or two agents. It is one model reading two relation types off one graph.

### 1.4 The baseline build

For Phase 1: **Bookkeeper + one inference model**. One reasoning context means no lossy handoff and span provenance never has to survive serialization across an agent boundary. The graph store is code; only NLI-class judgments (claim spotting, pairwise supersedes/conflict) are model calls. The graph is not a prompt.

### 1.5 Falsifiability is now structural

Each layer has independent measurable outputs:
- Cartographer: edge precision/recall
- Bookkeeper: invariant maintenance (no cycles, anchor integrity, correct admit/deny)  
- Reasoning: SEAR grounding rate over a fixed admitted graph

You cannot hide an unfalsifiable claim in a stack where every layer's job is separately measurable.

---

## 2. Logits Processor — Design Rationale and Language Stack
*Session: Building SEAR with Mistral: grammar constraints and sparse attention*

### 2.1 Why a hand-rolled LogitsProcessor

Standard constrained generation tools (Outlines, XGrammar, llama.cpp GBNF) are built for static automata. SEAR's copy-constraint is dynamic: the set of legal next tokens is derivable only by looking at what actually continues live source spans in the corpus at inference time. The corpus is the grammar, and it changes as documents are added.

Encoding a copy-constraint as a static grammar requires an alternation over every verbatim span in the corpus — tens of thousands of alternatives for a non-trivial corpus, recompiled every time the corpus changes. The tools were not designed for this.

The `LogitsProcessor` contract — `(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor` — is minimal and stable. The same processor runs unchanged under llama-cpp-python now and HF transformers/vLLM later. You write the constraint once.

### 2.2 The cursor mechanism

Instead of compiling the corpus into a grammar, live cursors `(doc_id, position)` track which source spans are still consistent with what has been emitted so far. At each decode step, the legal next tokens are the union of whatever token sits at `position + 1` across all live cursors. After each emission, the cursor set is advanced and pruned.

Cursor set growth is O(|live_cursors|) per step rather than O(|corpus_spans|) to compile. The cursor set typically shrinks as the span lengthens — it is self-pruning. Ambiguous prefixes (a token sequence appearing in more than one document) fan out cursor sets correctly; cursors stay live across all matching documents until a diverging token forces pruning.

**Attribution is exact by construction.** The cursor set surviving to end-of-span is precisely the set of source documents the model drew from. This is a different epistemic guarantee than post-hoc attention tracing. Attention weights describe correlation; cursor provenance describes causation.

### 2.3 Zero cursors — the grounding failure signal

When cursors collapse to zero, the model tried to emit a token that does not appear anywhere in the corpus at the position where it would need to appear to continue any live span. The corpus cannot support the claim. The model cannot continue synthesizing unsourced content.

Fallback policy (unresolved in Phase 1 scaffold, to be specified):
- **Backtrack and choose from remaining legal tokens** — if any exist at the current position
- **Force EOS** — claim terminates at the grounded portion
- **Error with synthesis-time estimate** — see Section 3

The fallback decision has semantic consequences for what SEAR claims about its outputs. It belongs in the specification before Phase 2.

### 2.4 Language compatibility

The processor interface lives where the inference runtime lives.

**Python** — native home. Both llama-cpp-python and HuggingFace transformers have native `LogitsProcessor` hooks. The Phase 1 scaffold (`sear_phase1.py`) is Python. This is the correct stack for research phase.

**Rust** — viable longer term. llama.cpp has a clean C API with quality Rust bindings (llama-rs, kalosm). The cursor logic ports cleanly because it is pure data structure work with no ML framework dependency. A Rust-native GIN node daemon is a reasonable longer-term direction given the homelab stack and correctness requirements.

**C/C++** — possible via llama.cpp callback system but against an internal API that moves. Not justified for research phase.

**Go/TypeScript** — not viable for processor work. Fine for consuming a SEAR endpoint; insufficient for building the constraint.

### 2.5 Timeline estimate

Starting from scratch: 1–3 days of focused work for a validated processor.
- Day 1: LogitsProcessor interface, basic masking, initial corpus data structure
- Day 2: Token-level inverted index, cursor fan-out/prune logic, multi-token span tracking
- Day 3: Self-test with toy corpus, edge case handling, fallback policy implementation

Starting from `sear_phase1.py` with validated self-test: the remaining work to hook into a live Mistral forward pass with llama-cpp-python is approximately a half-day. The longer pole is corpus quality, not implementation.

### 2.6 Open: corpus stress test

The Phase 1 self-test corpus (fox/dog divergence example) is too clean — one token of shared prefix before instant pruning. Real corpus behavior involves long shared spans where the cursor set stays large across many documents before pruning. Cursor set growth under realistic corpus density has not been tested. The fallback at zero cursors has not been specified.

---

## 3. Federation Routing — Cursor Collapse as Network Signal
*Session: This conversation (June 18, 2026)*

### 3.1 The design

When cursors collapse to zero at a single node, rather than treating it as a dead end, the system pivots to the federation. Cursor collapse is the signal that the local corpus cannot ground a claim — which becomes a routing trigger, not a failure state.

**The routing sequence on cursor collapse:**

1. Zero cursors detected — local corpus cannot continue this span
2. Check local knowledge graph metadata for backup nodes with cursor density on this topic
3. If no candidate nodes exist: error, return "cannot ground at current time," recommend recursive query with estimated synthesis time
4. If candidate nodes exist: query ranked by relatedness score, receive candidate continuations with provenance stamps, synthesize across sources

Attribution stays exact across the network. Each candidate continuation carries its source node stamp. The system is not hallucinating; it is composing from verified, multi-sourced material with explicit provenance. This is architecturally stronger than single-node grounding.

### 3.2 Estimated synthesis time as first-class output

The synthesis time estimate is not a UX nicety — it is epistemic transparency. The user sees the cost of grounding depth:
- Local query (cursor hit): fast, provenance certain
- Local query (cursor miss, backtrack): fast, provenance from surviving cursors
- Federated query (neighbor cache hit): medium, provenance from neighbor's verified graph
- Federated query (live cross-corpus alignment): slow, provenance chain explicit but expensive

This is something RAG systems do not expose. The transparency about grounding cost is part of what GIN claims to offer.

### 3.3 Epistemic routing as emergent specialization

Nodes become specialists over time not by design but by corpus accumulation. A node ingesting epidemiology literature develops high cursor density there. The network learns this through knowledge graph metadata sync (Section 4). Queries self-organize toward the nodes best equipped to ground them.

This is Glissant's relational opacity as infrastructure. Each node maintains its own grounded particularity; the network relates across those particularities without flattening them into a single unified corpus. The diversity of nodes is the strength, not a coordination problem to be solved.

### 3.4 Federated cursor granularity — open question

When Node A asks Node B to continue a span, three options exist:

**Option 1 — Next token only:** Node B returns legal continuations. Fast, loses provenance detail.

**Option 2 — Provenance chain:** Node B returns continuations plus source docs in B's corpus containing the span. Medium cost, preserves attribution.

**Option 3 — Re-constrained forward pass:** Node B treats A's prefix as an external anchor and runs a constrained decode through B's local SEAR. Highest cost, produces federated cursor divergence signal — enables detection of cases where the same span grounds differently in different corpora.

Option 3 is the most principled for GIN's thesis (federated divergence as a first-class output) but also the most expensive. The synthesis-time estimate should track which mode is in use. Default for Phase 1 federation: Option 2.

---

## 4. Knowledge Graph Metadata Sync — Staleness as Governance
*Session: This conversation (June 18, 2026)*

### 4.1 What syncs

The periodic sync is not corpus replication — that would be expensive and defeat distributed sovereignty. It syncs **knowledge graph metadata**: each node publishing a structured summary sufficient for routing decisions.

Sync payload per node:
- Topic/domain fingerprint (dense enough for relatedness scoring)
- Cursor density estimates per topic (how well can this node ground claims in this area)
- Known edge relationships to peer nodes (strength, directness, last-verified timestamp)
- Staleness timestamp (so receiving nodes know whether to trust the cached view)

The Bookkeeper stamps and stores incoming sync records. The relatedness gate reads from them. The federation cache from Section 1.2 is this structure.

### 4.2 Dynamic sync trigger

Fixed schedule sync is the wrong model for the general case. The correct trigger is:

**Sync when:**
- Corpus growth exceeds a delta threshold (significant new material warrants publishing an updated fingerprint)
- OR maximum staleness ceiling is reached (hard floor — "I will not serve metadata older than X regardless of growth rate")
- Whichever comes first

This makes sync frequency a function of actual epistemic change rate rather than clock time. A node with a static corpus almost never syncs. A node ingesting new material daily syncs frequently. The network stays current where it matters without unnecessary overhead.

### 4.3 Max staleness ceiling as governance

The staleness ceiling is where governance lives. Different nodes will have different requirements:
- A high-integrity Tier 1 institutional node might refuse to federate with nodes whose staleness ceiling exceeds 7 days
- A personal Tier 3 node might tolerate 2-week-old metadata for casual queries
- Research nodes working on contested empirical topics might require daily freshness

There is no neutral technical answer to staleness tolerance. It is a trust negotiation between nodes. The parameter belongs in the Bookkeeper as a configurable field — governance fills it; the architecture does not pre-fill it.

Practical starting range for a homogeneous trusted network: 2–3 syncs per week to every 2 days. This is a default, not a spec.

### 4.4 Staleness as a trust filter

"I do not federate with nodes whose staleness ceiling exceeds X" is a legitimate policy a high-integrity node might enforce. This means staleness tolerance becomes a visible property of a node's federation posture — part of how nodes signal trustworthiness to each other. The architecture supports this without requiring it.

---

## 5. Corpus Strategy — Building Toward the Two-Node Demo
*Session: This conversation (June 18, 2026)*

### 5.1 What "large enough to be interesting" means

For the cursor divergence signal to be meaningful, the corpus needs:

**Shared prefixes across documents.** If every document uses completely distinct vocabulary, cursors never fan out. The cursor set never grows large enough to produce interesting pruning events. You need documents that share common phrasings so the cursor set actually fans across sources before pruning.

**Conflicting continuations.** The interesting case is two documents sharing an opening span but diverging on a factual claim. "The unemployment rate was 3.7%" vs. "the unemployment rate was 4.2%." This is where the divergence instrument earns its place. A corpus without genuine factual conflict is too easy.

**Traceable provenance.** Attribution must be verifiable — which means a corpus where ground truth is knowable independently of the model's output.

### 5.2 Corpus candidates

**Wikipedia revision histories** — nearly ideal. Same article, different timestamps, genuine factual drift, shared sentence structure. Pull several hundred revisions of a contentious article (one that updates frequently). This produces exactly the needed properties: shared prefixes, genuine factual divergence, known ground truth.

**News wire corpora** (Reuters, AP archives covering the same events across outlets) — shared facts, different phrasings, occasional genuine contradiction. MIND dataset or CC-News subset of Common Crawl are accessible and documented.

**Synthetic corpus (recommended for Phase 1 validation)** — generate N documents with a shared "backbone" of sentences plus controlled injected variations. Lets you set divergence density precisely and verify ground truth mechanically. Slower to build but makes self-tests ironclad and cursor behavior deterministic enough to characterize.

### 5.3 Recommended sequence

1. **Synthetic corpus for mechanical validation** — build in an afternoon, verify cursor fan-out/prune behavior under controlled conditions, stress-test cursor set growth under long shared spans, specify the zero-cursor fallback policy
2. **Wikipedia revision slice for the divergence demo** — once instrumentation is validated against known ground truth, the real signal becomes meaningful rather than artifactual

The Phase 1 self-test (fox/dog) is correct in design but too sparse to characterize real behavior. The synthetic corpus is the next step before any live model integration.

---

## 6. Design Character — Warm Rigor
*Emerged: This conversation*

The phrase "warm rigor" landed as an accurate description of GIN's overall design character and is worth recording as a design constraint, not just a label.

**Rigorous** in that every design decision carries real epistemic consequence. Zero cursors means something specific. Staleness ceilings mean something specific. The Bookkeeper's admission gate is not bureaucratic overhead — it is the integrity of the network's claims. The weight is load-bearing. Nothing is decorative.

**Warm** in that nothing is over-engineered for its own sake. The cursor mechanism is as simple as it can be while still being correct. The sync is metadata not corpus. The governance socket exists but does not demand to be filled before the system runs. Every component earns its place.

The Glissant thread running through GIN is structurally operative, not philosophical decoration. Opacity-as-precondition-of-relation means nodes do not expose their full corpus to participate in the network. The federation relates across particularity rather than flattening it. That is why the design is distributed rather than a shared corpus with access controls.

The zero-cursor failure mode is honest rather than embarrassing. Most systems optimize for never saying "I don't know." GIN makes "I don't know, but Node B might — estimated synthesis time X" a first-class, well-formed response. That is rare. It is the warmth in the rigor.

---

## Open Questions — Explicit Register

These are not gaps in thinking. They are the identified frontier.

| Question | Session of origin | Status |
|---|---|---|
| Zero-cursor fallback policy (backtrack / EOS / error+estimate) | SEAR Phase 1 | Unresolved — must be specified before Phase 2 |
| Federated cursor granularity (Option 1/2/3 for cross-node span continuation) | Federation routing | Default Option 2 for Phase 1; Option 3 as Phase 3 target |
| Cursor set growth under large shared-prefix corpora | SEAR Phase 1 | Untested — synthetic corpus work will characterize |
| Staleness ceiling defaults for a homogeneous trusted network | Sync governance | 2–3x/week to every 2 days as starting range; not a spec |
| Who funds Tier 1 anchors and how trust weights get set without recreating centralization | Multi-agent session | Governance question, not architecture — open frontier |

---

## Build Sequence — Current State

1. ✅ SEAR Phase 1 scaffold validated (self-test passes, cursor logic correct)  
2. ✅ Synthetic corpus — fan-out/prune verified; zero-cursor fallback wired as post-decode refusal gate  
3. ✅ Live Mistral integration via llama-cpp-python (runs via WSL Ubuntu; Windows-python crashes at model load)  
4. ✅ SEAR grounding rate measured against RAG baseline — NC fabrication 0.0 vs RAG 0.238–0.286 ([[GIN_ENG_02_Eval_Baseline_v1]], `20260702T012203Z`)  
5. ✅ Two-node divergence demo — same machinery, scope dialed to inter-corpus; **real fetched text**, `divergence_fidelity` 1.0 (`20260705T043114Z`), generalized across three framing registers and confirmed model-independent on Qwen2.5-7B ([Real-text divergence generalization](nc_real_text_divergence_generalization.plan.md))  
6. 🔲 Bookkeeper + reasoning layer separation (Phase 2)  
7. 🔲 Federation routing with sync metadata (Phase 3)  

*The two-node divergence demo was the empirical keystone. That number now exists (item 5) — so this document and its companions have crossed from architecture into record for the reasoning/divergence layer. What remains architecture is the federation transport (Bookkeeper admission, Cartographer discovery, Merkle-diff sync, zero-cursor peer routing): items 6–7. The real-text plan §7 argues the sequencing — reasoning-layer robustness gates the whole chain, then Cartographer before Bookkeeper.*

---

*Monolith / GIN — session synthesis*  
*Companions: GIN_The_Whole_Frame.md · GIN_Node_Architecture_v1.md · sear_phase1.py*  
*Next real artifact: synthetic corpus + zero-cursor fallback specification*
