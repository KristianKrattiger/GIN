# SEAR vs RAG Evaluation Report

- Generated: 2026-07-12T03:52:28.195718+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: nli (threshold 0.5)
- Query set: /mnt/c/Users/krist/Projects/gin/GIN/data/eval/queryset.yaml
- Queries: 20
- GPU layers: -1
- Wall-clock per query: 37.93s
- Tokens/s (approx): 0.6

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.714 | 0.056 |
| grounded_precision | 0.286 | 0.944 |
| attribution_coverage | 0.286 | 0.944 |
| counterfactual_adherence | 0.500 | 0.750 |
| failure_precision | 1.000 | 0.714 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | 0.286 | 0.833 |
| cross_node_violations | 0 | 0 |
| n_claims | 21 | 18 |
| n_queries | 20 | 20 |

## By eval_layer

### counterfactual

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.500 | 0.250 |
| grounded_precision | 0.500 | 0.750 |
| attribution_coverage | 0.500 | 0.750 |
| counterfactual_adherence | 0.500 | 0.750 |
| failure_precision | n/a | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.500 | 0.250 |
| cross_node_violations | 0 | 0 |
| n_claims | 4 | 4 |
| n_queries | 4 | 4 |

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
| n_queries | 5 | 5 |

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.765 | 0.000 |
| grounded_precision | 0.235 | 1.000 |
| attribution_coverage | 0.235 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | n/a | 0.000 |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.235 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 17 | 14 |
| n_queries | 11 | 11 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 1.000 | 0.900 |
| gold_chunk_coverage | 0.378 | 0.767 |
| supported_irrelevance_rate | 0.000 | 0.000 |
| chunk_quotation_rate | 0.419 | 0.384 |
| divergence_fidelity | 0.875 | 0.750 |

- **no_continuation** failing query relevance: election_margin, school_enrollment_fall

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| incident_hospital | 1.000 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| incident_arrests | 1.000 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| election_margin | 1.000 | election_centralwire:0, election_metrodaily:0, port_authority_brief:0, school_district_report:0 | election_centralwire:0, election_metrodaily:0 |
| election_turnout | 1.000 | election_centralwire:0, election_metrodaily:0, port_authority_brief:0, school_district_report:0 | election_centralwire:0, election_metrodaily:0 |
| transit_ridership | 1.000 | transit_authority_update:0, port_authority_brief:0, power_grid_status:0, wage_independent_survey:0 | transit_authority_update:0 |
| weather_winds | 1.000 | weather_service_brief:0 | weather_service_brief:0 |
| unemployment_rate | 1.000 | labor_bureau_report:0, labor_independent_survey:0, wage_independent_survey:0, inflation_bureau_report:0, inflation_independent_survey:0 | labor_bureau_report:0, labor_independent_survey:0 |
| interest_rate_probe | n/a | disc_northwind_pr:0, disc_northwind_complaint:0 | — |
| sports_probe | n/a | n1_doc_002:0 | — |
| port_cargo_throughput | 1.000 | port_authority_brief:0, election_centralwire:0, election_metrodaily:0, housing_permits_office:0, transit_authority_update:0 | port_authority_brief:0 |
| school_enrollment_fall | 1.000 | election_centralwire:0, election_metrodaily:0, school_district_report:0 | school_district_report:0 |
| reservoir_storage_level | 1.000 | water_utility_update:0, n1_doc_009:3, port_authority_brief:0 | water_utility_update:0 |
| housing_permits_issued | 1.000 | housing_permits_office:0 | housing_permits_office:0 |
| outage_restoration_time | 1.000 | power_grid_status:0 | power_grid_status:0 |
| inflation_rate | 1.000 | inflation_bureau_report:0, inflation_independent_survey:0, export_trade_report:0, labor_bureau_report:0, labor_independent_survey:0 | inflation_bureau_report:0, inflation_independent_survey:0 |
| wage_growth_rate | 1.000 | wage_bureau_report:0, wage_independent_survey:0, labor_bureau_report:0, labor_independent_survey:0, n1_doc_009:4 | wage_bureau_report:0, wage_independent_survey:0 |
| export_decline_rate | 1.000 | export_independent_review:0, export_trade_report:0, labor_bureau_report:0, labor_independent_survey:0, inflation_bureau_report:0 | export_trade_report:0, export_independent_review:0 |
| crypto_price_probe | n/a | disc_northwind_pr:0, disc_northwind_complaint:0, inflation_bureau_report:0, inflation_independent_survey:0 | — |
| olympic_medals_probe | n/a | n1_doc_002:2, n1_doc_002:5, n1_doc_007:0, n1_doc_007:2, n1_doc_009:2 | — |
| dinosaur_fossil_probe | n/a | n1_doc_001:2, n1_doc_002:0, n1_doc_002:5, n1_doc_006:0, n1_doc_006:3 | — |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
