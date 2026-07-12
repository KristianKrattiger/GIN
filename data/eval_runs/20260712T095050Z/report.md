# SEAR vs RAG Evaluation Report

- Generated: 2026-07-12T09:50:50.810106+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_framing2.yaml
- Queries: 2
- Wall-clock per query: 77.92s
- Tokens/s (approx): 0.6

## Overall

| Metric | no_continuation |
|---|---|
| fabrication_rate | 0.000 |
| grounded_precision | 1.000 |
| attribution_coverage | 1.000 |
| counterfactual_adherence | n/a |
| failure_precision | n/a |
| failure_recall | n/a |
| cross_node_within_ratio | 1.000 |
| cross_node_violations | 0 |
| n_claims | 4 |
| n_queries | 2 |

## By eval_layer

### realism

| Metric | no_continuation |
|---|---|
| fabrication_rate | 0.000 |
| grounded_precision | 1.000 |
| attribution_coverage | 1.000 |
| counterfactual_adherence | n/a |
| failure_precision | n/a |
| failure_recall | n/a |
| cross_node_within_ratio | 1.000 |
| cross_node_violations | 0 |
| n_claims | 4 |
| n_queries | 2 |

## Epistemic quality

| Metric | no_continuation |
|---|---|
| query_relevance_rate | 1.000 |
| gold_chunk_coverage | 1.000 |
| supported_irrelevance_rate | 0.000 |
| chunk_quotation_rate | 0.833 |
| divergence_fidelity | 1.000 |


## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| fr_northwind_revenue | 1.000 | disc_northwind_complaint:0, disc_northwind_pr:0 | disc_northwind_pr:0, disc_northwind_complaint:0 |
| fr_meridian_breach | 1.000 | disc_meridian_complaint:0, disc_meridian_pr:0, n1_doc_003:4 | disc_meridian_pr:0, disc_meridian_complaint:0 |

- **no_continuation** mean gold recall@k: 1.000
