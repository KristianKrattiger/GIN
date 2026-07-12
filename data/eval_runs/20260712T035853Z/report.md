# SEAR vs RAG Evaluation Report

- Generated: 2026-07-12T03:58:53.292140+00:00
- Model: models/Qwen2.5-7B-Instruct-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_twonode.yaml
- Queries: 6
- GPU layers: -1
- Wall-clock per query: 51.11s
- Tokens/s (approx): 0.7

## Overall

| Metric | no_continuation |
|---|---|
| fabrication_rate | 0.000 |
| grounded_precision | 1.000 |
| attribution_coverage | 1.000 |
| counterfactual_adherence | n/a |
| failure_precision | 1.000 |
| failure_recall | 1.000 |
| cross_node_within_ratio | 1.000 |
| cross_node_violations | 0 |
| n_claims | 8 |
| n_queries | 6 |

## By eval_layer

### out_of_scope

| Metric | no_continuation |
|---|---|
| fabrication_rate | n/a |
| grounded_precision | n/a |
| attribution_coverage | n/a |
| counterfactual_adherence | n/a |
| failure_precision | 1.000 |
| failure_recall | 1.000 |
| cross_node_within_ratio | n/a |
| cross_node_violations | 0 |
| n_claims | 0 |
| n_queries | 1 |

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
| n_claims | 8 |
| n_queries | 5 |

## Epistemic quality

| Metric | no_continuation |
|---|---|
| query_relevance_rate | 1.000 |
| gold_chunk_coverage | 1.000 |
| supported_irrelevance_rate | 0.000 |
| chunk_quotation_rate | 0.314 |
| divergence_fidelity | 1.000 |


## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| tn_emissions_framing | 1.000 | n1_doc_005:2, n2_doc_001:4, n1_doc_001:7, n1_doc_005:4, n1_doc_007:4 | n1_doc_005:2, n2_doc_001:4 |
| tn_wildfire_framing | 1.000 | n1_doc_008:0, n2_doc_005:1, n1_doc_008:2, n1_doc_008:1 | n1_doc_008:0, n2_doc_005:1 |
| tn_water_framing | 1.000 | n1_doc_009:0, n2_doc_008:2, n2_doc_008:0, n2_doc_008:1, n2_doc_008:4 | n1_doc_009:0, n2_doc_008:2 |
| tn_2023_anomaly | 1.000 | n1_doc_002:1, n1_doc_005:0, n1_doc_007:0, n1_doc_007:1 | n1_doc_002:1 |
| tn_ocean_acidification | 1.000 | n1_doc_004:0, n1_doc_004:4, n1_doc_004:6 | n1_doc_004:0 |
| tn_out_of_scope_referendum | n/a | election_centralwire:0, election_metrodaily:0, port_authority_brief:0, school_district_report:0 | — |

- **no_continuation** mean gold recall@k: 1.000
