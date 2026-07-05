# SEAR vs RAG Evaluation Report

- Generated: 2026-07-05T21:51:12.771889+00:00
- Model: models/Qwen2.5-7B-Instruct-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_framing2.yaml
- Queries: 2
- Wall-clock per query: 99.55s
- Tokens/s (approx): 1.0

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | n/a | 0.000 |
| grounded_precision | n/a | 1.000 |
| attribution_coverage | n/a | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | n/a | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 0 | 4 |
| n_queries | 2 | 2 |

## By eval_layer

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | n/a | 0.000 |
| grounded_precision | n/a | 1.000 |
| attribution_coverage | n/a | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | n/a | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 0 | 4 |
| n_queries | 2 | 2 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 0.000 | 1.000 |
| gold_chunk_coverage | 0.000 | 1.000 |
| supported_irrelevance_rate | n/a | 0.000 |
| chunk_quotation_rate | 0.000 | 0.833 |
| divergence_fidelity | 0.000 | 1.000 |

- **rag** failing query relevance: fr_northwind_revenue, fr_meridian_breach

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| fr_northwind_revenue | 1.000 | disc_northwind_pr:0, disc_northwind_complaint:0 | disc_northwind_pr:0, disc_northwind_complaint:0 |
| fr_meridian_breach | 1.000 | disc_meridian_pr:0, disc_meridian_complaint:0, n1_doc_003:4 | disc_meridian_pr:0, disc_meridian_complaint:0 |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
