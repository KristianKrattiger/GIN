---
tags: [GIN, research, engineering, specs, quarantine]
updated: 2026-07-20
version: 0.5-preliminary
status: working draft
register: engineering
---

# GIN ENG 00 — Engineering Register

> Index and quarantine. Every reality-grounded specification lives here and nowhere else, so that nothing in the conceptual register (GIN 00–10) can be mistaken for a shipped design. Entries below are **unmeasured** until the promotion rule is satisfied — except where a bullet explicitly records a measured promotion with an artifact path.

---

## Why this register is separate

The conceptual papers argue. This register measures. Keeping them apart prevents an argued figure from being read as a measured one. A claim crosses into this register only when measured, never when argued.

---

## Spec categories (quarantine + promotions)

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
- **Divergence generalizes beyond the synthetic corpus** — real fetched two-node text (institutional vs. grassroots framing) reaches `divergence_fidelity` 1.000 / `fabrication_rate` 0.000 in full DB eval `20260705T043114Z`, and holds across two further framing registers (adversarial/legal `20260705T202450Z`, housing `20260705T203622Z`). The mechanism is **model-independent** — Qwen2.5-7B matches the Mistral baseline exactly on all four divergence querysets (`20260705T211452Z`–`20260705T220525Z`). Method and per-pair token/IDF-margin tables: `docs/nc_real_text_divergence_generalization.plan.md`. Still CPU/WSL — representative GPU artifact remains before promotion.
- Convergent-mode early-close permissiveness — **fixed** (2026-07-12): convergent decode sets `span_must_close_at_sentence_end=True`; regression in `tests/test_generate.py::test_convergent_numeric_sentence_closes_at_sentence_end`.

**Temporal / sensor grounding** ([[GIN_13_Temporal_Sensor_Grounding]])
- Architectural fork: derived-claim conversion vs. native temporal nodes (current lean: native — a parallel time-series reasoning pathway, not an extension of the text path). Unbuilt.
- Baseline/reference layer shape (per-crop / per-region / per-sensor-model / domain-specific) — undetermined.
- Sensor calibration metadata registry: schema, required fields (calibration date, drift parameters, error margins, validation records), and whether it is council-hosted shared infrastructure or self-published-and-audited. This extends Bookkeeper's existing calibration/audit role rather than adding a subsystem. Open.
- Divergence scoring mechanics for time-series vs. time-series and time-series vs. text claim.
- First node pair / dataset — not identified. Status: many phases out, not sequenced.

**Federation** ([[GIN_03_Node_Identity]])
- Adapter-switching cost and concurrency limits — still unmeasured.
- Federation breadth-control: how many nodes constitute a "full picture," selected how — open beyond N=3 content+trust ranking.
- Two-node divergence demo (inter-corpus, same machinery) — **measured** (`20260705T043114Z`, real two-node corpus). This established the divergence *signal* across two corpora.
- **Phase 3 transport — measured promotions** (see [architecture.md](../architecture.md) Phase 3; ops in [README.md](../README.md)):
  - Sovereign delegation loop — `data/eval_runs/20260714T175645Z/federation_metrics.json` (routing FP 0, recall 1.0, routed fabrication 0.0, honest refusal 1.0).
  - Merkle anchor sync — `data/eval_runs/20260715T073932Z/anchor_sync_metrics.json` (0 diff vs peer ground truth; no-op O(1) bytes).
  - Peer selection at N=3 — `data/eval_runs/20260715T192750Z/peer_selection_metrics.json` (precision@1 1.0).
  - Trust weights (per-domain gating) — gated `20260716T004321Z` (gated_peer_contacted 0); ungated regression `20260716T004515Z`.
  - mTLS (pinned self-signed certs; bearer `shared_secret` removed) — ungated `20260716T095519Z`, gated `20260716T095704Z` reproduce prior bars over real TLS.
  - Streaming reasoning trace — `POST /v1/federated/query/stream` NDJSON events (spec `2026-07-16-streaming-reasoning-trace-design.md`).
  - Still deferred: gRPC/QUIC transport, geographic multi-host deployment, adapter-switching.

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
- RAG vs No-Continuation on synthetic corpus. Structural prevention measured (overlap fabrication 0.286 vs 0.000, run `192827Z`). NC epistemic targets met on expanded 20-query set (`20260702T012203Z`: query relevance 1.0, supported irrelevance 0, gold coverage 1.0, counterfactual adherence 1.0, fabrication 0, divergence fidelity 1.0). GPU reproducibility **measured and root-caused**: `20260711T211202Z` (RTX 4070) vs same-day CPU control `20260711T212721Z` (identical code/corpus) — fabrication 0.0 on both; gap isolated to 1/20 queries at a near-tie refuse/answer decision, attributable to CPU/CUDA floating-point non-determinism in llama.cpp, not a GIN defect. The initial 3-query comparison against the 9-day-old baseline was largely a stale-baseline artifact (corpus re-ingestion changes retrieval tie-break order) — see [[GIN_ENG_02_Eval_Baseline_v1]] Remaining gaps items 3 and 5.
- **Generalization (post-v1)**: two-node real-text divergence and two additional framing registers measured at `divergence_fidelity` 1.0 / `fabrication_rate` 0.0; cross-model confirmation on Qwen2.5-7B. Full method in `docs/nc_real_text_divergence_generalization.plan.md`.

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

**Cross-eval comparison hygiene (2026-07-12):**

- Compare eval runs only when `meta.json` reports the same `corpus_fingerprint` (`chunk_count`, `edge_count`, `content_hash` over sorted `(chunk_id, content_hash)` pairs).
- Cross-backend reproduction (CPU vs GPU at `temperature=0.0`): `fabrication_rate` must remain **0.000** on both; up to **1/20** queries may flip on refuse-vs-answer boundary decisions without blocking promotion.

**Convergent truncation** (`tn_2023_anomaly`): fixed — convergent decode now sets `span_must_close_at_sentence_end=True` for all modes ([`gin/corpus/generate.py`](../gin/corpus/generate.py)).

**Cartographer → Bookkeeper → Postgres**: `scripts/cartographer_scan.py` + [`gin/bookkeeper/persist.py`](../gin/bookkeeper/persist.py) persist admitted edges with provenance; labeled set expanded to 33 pairs with LOO ≥ 0.85 on baked calibration samples.

**Cartographer scan production validation** (2026-07-12): scan-first workflow (`corpus_ingest.py --no-edges` → `cartographer_scan.py --cross-outlet-only` → `cartographer_eval_scan.py`) measured on mixed 136-chunk DB. **Baseline** (`20260712T053645Z`): gold contradicts recall **0.50**, **135** false-positive admitted edges, `class_c_discrimination` **1.0**. **After scan pruning + Bookkeeper relation re-check** (`20260712T074956Z`): hybrid IDF+embedding candidate prune (6222 → 2157 pairs), doc-pair dedup (NLI preferred), `FRAMING_BAND_FLOOR` **0.35**, default `out_of_scope_stub` exclusion — recall still **0.50**, false positives **120** (122 admitted), `class_c_discrimination` **1.0**; `hf_alderflats_staff:0` ↔ `hf_alderflats_tenants:0` recovered. Remaining gaps closed later the same day: story-gate + label decisions (`20260712T220456Z` machine recall 1.0 on 9/9 story-class gold); issue_frame class closed as curation-only after frontier judges stayed below bar (`20260713T223653Z`). See README Cartographer section and `nc_cartographer_design.plan.md`.

**Curator + node4** (2026-07-17…20): framing-corpus spine (`gin/curator/`, `data/curator/labels.jsonl`); B0 residue source + readiness gauge; node4 contested-policy polar corpus (`corpus_node4.json`, `scripts/build_node4.py`, `scripts/verify_node4_surfacing.py`). Bi-encoder frame detector remains gated on labels.

## Related

[[GIN_00_Reader]] · [[GIN_05_MOCAP]] · [[GIN_06_Mule_Architecture]] · [[GIN_09_Agentic_Layer]] · [[GIN_ENG_01_SEAR_PoC_Spec]] · [[GIN_ENG_02_Eval_Baseline_v1]]

## Back to Vault

[[HOME]]
