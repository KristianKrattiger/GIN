# GIN — Executive Summary

**GIN (Grounded Intelligence Network)** is a federation of independent reasoning nodes that ground every claim in a traceable corpus and hold disagreement between sources visible instead of averaging it away. Below is what has actually been built and measured — nothing aspirational.

## Fabrication elimination (SEAR vs. RAG, 20-query eval set)

| Metric | RAG | SEAR (No-Continuation) |
|---|---|---|
| Fabrication rate | 0.238 | **0.000** |
| Query relevance | 1.000 | 1.000 |
| Gold-chunk coverage | 0.956 | 1.000 |
| Counterfactual adherence | 1.000 | 1.000 |
| Divergence fidelity (both sides of a disagreement preserved) | 0.875 | **1.000** |

**Generalizes beyond the synthetic corpus:** holds on real fetched text across three framing registers (climate, legal, housing) — fidelity 1.000, fabrication 0.000 — and is model-independent (Qwen2.5-7B reproduces the Mistral-7B baseline exactly). Reconfirmed on GPU (RTX 4070), not just CPU.

## Federation (3 live nodes, sovereign delegation)

- Routing false positives: 0 · routing recall: 1.000 · delegated-answer fabrication: 0.000 · honest-refusal rate: 1.000
- Peer selection at N=3: precision@1 1.000
- Merkle anchor sync: 0 diff vs. ground truth, O(1) bytes per no-op cycle
- Transport: mutual TLS with pinned certs, measured live across all 3 nodes

## Negative findings (reported, not smoothed over)

- One eval query of 20 flips its refuse/answer decision between CPU and GPU backends — root-caused to floating-point non-determinism in the underlying decode kernels, not a fabrication risk (fabrication measured 0.000 on both backends).
- Automated "framing divergence" classification failed at every model tier tested, including a frontier model (0.00 recall) — the label encodes an editorial judgment no off-the-shelf judge reproduces. Closed as curation-only rather than claimed solved; a purpose-trained judge is future work, not something we did.
- Under the stricter NLI verifier on the expanded query set, fabrication reads 0.056 (1 of 18 graded claims) — a verbatim-extract entailment miss on one counterfactual query that scores 0.000 under the primary overlap verifier.

## Explicitly out of scope for this report

Phase 4 (a training loop for SEAR), multi-host geographic federation, and any institutional or governance layer are not covered here — they are either unmeasured, not designed in implementation detail, or both.
