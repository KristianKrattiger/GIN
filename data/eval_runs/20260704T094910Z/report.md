# SEAR vs RAG Evaluation Report

- Generated: 2026-07-04T09:49:10.001175+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_twonode.yaml
- Queries: 6
- Wall-clock per query: 31.65s
- Tokens/s (approx): 1.3

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.444 | 0.000 |
| grounded_precision | 0.556 | 1.000 |
| attribution_coverage | 0.556 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 1.000 | 1.000 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | 0.556 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 18 | 5 |
| n_queries | 6 | 6 |

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
| fabrication_rate | 0.444 | 0.000 |
| grounded_precision | 0.556 | 1.000 |
| attribution_coverage | 0.556 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | n/a | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.556 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 18 | 5 |
| n_queries | 5 | 5 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 1.000 | 0.833 |
| gold_chunk_coverage | 0.600 | 0.500 |
| supported_irrelevance_rate | 0.200 | 0.000 |
| chunk_quotation_rate | 0.381 | 0.206 |
| divergence_fidelity | 0.667 | 0.000 |

- **no_continuation** failing query relevance: tn_water_framing

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| tn_emissions_framing | 0.000 | n1_doc_001:7, n1_doc_005:2, n2_doc_001:4, n1_doc_005:4, n1_doc_007:4 | n1_doc_005:1, n2_doc_001:1 |
| tn_wildfire_framing | 1.000 | n1_doc_008:0, n1_doc_008:2, n1_doc_008:1, n2_doc_005:1 | n1_doc_008:0, n2_doc_005:1 |
| tn_water_framing | 1.000 | n2_doc_008:0, n2_doc_008:1, n2_doc_008:4, n2_doc_008:2, n1_doc_009:0 | n1_doc_009:0, n2_doc_008:2 |
| tn_2023_anomaly | 1.000 | n1_doc_002:1, n1_doc_005:0, n1_doc_007:0, n1_doc_007:1 | n1_doc_002:1 |
| tn_ocean_acidification | 1.000 | n1_doc_004:0, n1_doc_004:4, n1_doc_004:6 | n1_doc_004:0 |
| tn_out_of_scope_referendum | n/a | n2_doc_010:5, n1_doc_009:4, n2_doc_010:3, n2_doc_010:4, n1_doc_009:3 | — |

- **rag** mean gold recall@k: 0.800
- **no_continuation** mean gold recall@k: 0.800
