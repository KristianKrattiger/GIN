# SEAR vs RAG Evaluation Report

- Generated: 2026-07-04T09:13:11.159111+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_node1.yaml
- Queries: 7
- Wall-clock per query: 26.14s
- Tokens/s (approx): 0.6

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.000 | 0.000 |
| grounded_precision | 1.000 | 1.000 |
| attribution_coverage | 1.000 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 1.000 | 1.000 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | 1.000 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 6 | 6 |
| n_queries | 7 | 7 |

## By eval_layer

### out_of_scope

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | n/a | n/a |
| grounded_precision | n/a | n/a |
| attribution_coverage | n/a | n/a |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 1.000 | 1.000 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | n/a | n/a |
| cross_node_violations | 0 | 0 |
| n_claims | 0 | 0 |
| n_queries | 1 | 1 |

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.000 | 0.000 |
| grounded_precision | 1.000 | 1.000 |
| attribution_coverage | 1.000 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | n/a | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 1.000 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 6 | 6 |
| n_queries | 6 | 6 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 1.000 | 1.000 |
| gold_chunk_coverage | 0.944 | 0.889 |
| supported_irrelevance_rate | 0.000 | 0.000 |
| chunk_quotation_rate | 0.240 | 0.212 |
| divergence_fidelity | n/a | n/a |


## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| n1_warming_cause | 1.000 | doc_001:0, doc_001:1, doc_006:0, doc_001:3 | doc_001:0 |
| n1_2023_anomaly | 1.000 | doc_002:1, doc_005:0, doc_007:0, doc_007:1 | doc_002:1 |
| n1_warmest_year | 1.000 | doc_002:0, doc_007:0, doc_006:2, doc_007:2, doc_002:2 | doc_002:0, doc_006:2, doc_007:0 |
| n1_ocean_acidification | 1.000 | doc_004:0, doc_004:4, doc_004:6 | doc_004:0 |
| n1_paris_track | 1.000 | doc_005:1, doc_005:4, doc_002:1, doc_001:0 | doc_005:1 |
| n1_co2_concentration | 1.000 | doc_001:2, doc_004:1, doc_001:7, doc_007:4, doc_004:2 | doc_001:2 |
| n1_out_of_scope_referendum | n/a | doc_004:5 | — |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
