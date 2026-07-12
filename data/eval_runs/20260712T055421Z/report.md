# SEAR vs RAG Evaluation Report

- Generated: 2026-07-12T05:54:21.090416+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_twonode.yaml
- Queries: 6
- GPU layers: -1
- Wall-clock per query: 91.41s
- Tokens/s (approx): 0.4

## Overall

| Metric | no_continuation |
|---|---|
| fabrication_rate | 0.000 |
| grounded_precision | 1.000 |
| attribution_coverage | 1.000 |
| counterfactual_adherence | n/a |
| failure_precision | n/a |
| failure_recall | 0.000 |
| cross_node_within_ratio | 1.000 |
| cross_node_violations | 0 |
| n_claims | 11 |
| n_queries | 6 |

## By eval_layer

### out_of_scope

| Metric | no_continuation |
|---|---|
| fabrication_rate | 0.000 |
| grounded_precision | 1.000 |
| attribution_coverage | 1.000 |
| counterfactual_adherence | n/a |
| failure_precision | n/a |
| failure_recall | 0.000 |
| cross_node_within_ratio | 1.000 |
| cross_node_violations | 0 |
| n_claims | 4 |
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
| n_claims | 7 |
| n_queries | 5 |

## Epistemic quality

| Metric | no_continuation |
|---|---|
| query_relevance_rate | 0.500 |
| gold_chunk_coverage | 0.500 |
| supported_irrelevance_rate | 0.182 |
| chunk_quotation_rate | 0.319 |
| divergence_fidelity | 0.000 |

- **no_continuation** failing query relevance: tn_wildfire_framing, tn_water_framing, tn_out_of_scope_referendum

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| tn_emissions_framing | 1.000 | n1_doc_005:1, n2_doc_001:4, n1_doc_001:7, n1_doc_005:2, n1_doc_005:4, n1_doc_007:4 | n1_doc_005:2, n2_doc_001:4 |
| tn_wildfire_framing | 0.500 | n1_doc_008:2, n2_doc_008:3, n1_doc_008:0, n1_doc_008:1, election_centralwire:0, out_of_scope_stub:0 | n1_doc_008:0, n2_doc_005:1 |
| tn_water_framing | 0.500 | n2_doc_008:0, n2_doc_008:1, n2_doc_008:2, n2_doc_008:4, n1_doc_002:4, n1_doc_004:5 | n1_doc_009:0, n2_doc_008:2 |
| tn_2023_anomaly | 1.000 | n1_doc_002:1, n1_doc_005:0, n1_doc_007:0, n1_doc_007:1 | n1_doc_002:1 |
| tn_ocean_acidification | 1.000 | n1_doc_004:0, n1_doc_004:4, n1_doc_004:6, n2_doc_001:2, n2_doc_004:3, n2_doc_005:2 | n1_doc_004:0 |
| tn_out_of_scope_referendum | n/a | election_centralwire:0, election_metrodaily:0, school_district_report:0, water_utility_update:0, port_authority_brief:0, incident_metrodaily:0 | — |

- **no_continuation** mean gold recall@k: 0.800
