# SEAR vs RAG Evaluation Report

- Generated: 2026-07-05T21:14:52.377317+00:00
- Model: models/Qwen2.5-7B-Instruct-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_twonode.yaml
- Queries: 6
- Wall-clock per query: 57.48s
- Tokens/s (approx): 1.4

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.545 | 0.000 |
| grounded_precision | 0.455 | 1.000 |
| attribution_coverage | 0.455 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | 1.000 |
| failure_recall | 0.000 | 1.000 |
| cross_node_within_ratio | 0.455 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 11 | 8 |
| n_queries | 6 | 6 |

## By eval_layer

### out_of_scope

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.667 | n/a |
| grounded_precision | 0.333 | n/a |
| attribution_coverage | 0.333 | n/a |
| counterfactual_adherence | n/a | n/a |
| failure_precision | n/a | 1.000 |
| failure_recall | 0.000 | 1.000 |
| cross_node_within_ratio | 0.333 | n/a |
| cross_node_violations | 0 | 0 |
| n_claims | 3 | 0 |
| n_queries | 1 | 1 |

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.500 | 0.000 |
| grounded_precision | 0.500 | 1.000 |
| attribution_coverage | 0.500 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.500 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 8 | 8 |
| n_queries | 5 | 5 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 0.167 | 1.000 |
| gold_chunk_coverage | 0.000 | 1.000 |
| supported_irrelevance_rate | 0.000 | 0.000 |
| chunk_quotation_rate | 0.083 | 0.314 |
| divergence_fidelity | 0.000 | 1.000 |

- **rag** failing query relevance: tn_emissions_framing, tn_wildfire_framing, tn_2023_anomaly, tn_ocean_acidification, tn_out_of_scope_referendum

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| tn_emissions_framing | 1.000 | n1_doc_005:2, n2_doc_001:4, n1_doc_001:7, n1_doc_005:4, n1_doc_007:4 | n1_doc_005:2, n2_doc_001:4 |
| tn_wildfire_framing | 1.000 | n1_doc_008:0, n2_doc_005:1, n1_doc_008:2, n1_doc_008:1 | n1_doc_008:0, n2_doc_005:1 |
| tn_water_framing | 1.000 | n1_doc_009:0, n2_doc_008:2, n2_doc_008:0, n2_doc_008:1, n2_doc_008:4 | n1_doc_009:0, n2_doc_008:2 |
| tn_2023_anomaly | 1.000 | n1_doc_002:1, n1_doc_005:0, n1_doc_007:0, n1_doc_007:1 | n1_doc_002:1 |
| tn_ocean_acidification | 1.000 | n1_doc_004:0, n1_doc_004:4, n1_doc_004:6 | n1_doc_004:0 |
| tn_out_of_scope_referendum | n/a | election_centralwire:0, election_metrodaily:0, port_authority_brief:0, school_district_report:0 | — |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
