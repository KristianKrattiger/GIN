# SEAR vs RAG Evaluation Report

- Generated: 2026-07-11T21:27:21.031787+00:00
- Model: data/models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: C:\Users\krist\Projects\GIN\data\eval\queryset.yaml
- Queries: 20
- Wall-clock per query: 6.95s
- Tokens/s (approx): 2.8

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.250 | 0.000 |
| grounded_precision | 0.750 | 1.000 |
| attribution_coverage | 0.750 | 1.000 |
| counterfactual_adherence | 1.000 | 1.000 |
| failure_precision | 0.833 | 0.714 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | 0.750 | 0.778 |
| cross_node_violations | 0 | 0 |
| n_claims | 20 | 18 |
| n_queries | 20 | 20 |

## By eval_layer

### counterfactual

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.000 | 0.000 |
| grounded_precision | 1.000 | 1.000 |
| attribution_coverage | 1.000 | 1.000 |
| counterfactual_adherence | 1.000 | 1.000 |
| failure_precision | n/a | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 1.000 | 0.000 |
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
| fabrication_rate | 0.312 | 0.000 |
| grounded_precision | 0.688 | 1.000 |
| attribution_coverage | 0.688 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | 0.000 |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.688 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 16 | 14 |
| n_queries | 11 | 11 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 0.950 | 0.900 |
| gold_chunk_coverage | 0.889 | 0.867 |
| supported_irrelevance_rate | 0.000 | 0.000 |
| chunk_quotation_rate | 0.375 | 0.367 |
| divergence_fidelity | 0.875 | 0.750 |

- **rag** failing query relevance: school_enrollment_fall
- **no_continuation** failing query relevance: election_margin, school_enrollment_fall

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| incident_hospital | 1.000 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| incident_arrests | 1.000 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| election_margin | 1.000 | election_centralwire:0, election_metrodaily:0, port_authority_brief:0, school_district_report:0 | election_centralwire:0, election_metrodaily:0 |
| election_turnout | 1.000 | election_centralwire:0, election_metrodaily:0, school_district_report:0, port_authority_brief:0 | election_centralwire:0, election_metrodaily:0 |
| transit_ridership | 1.000 | transit_authority_update:0, power_grid_status:0, wage_independent_survey:0, port_authority_brief:0, wage_bureau_report:0 | transit_authority_update:0 |
| weather_winds | 1.000 | weather_service_brief:0 | weather_service_brief:0 |
| unemployment_rate | 1.000 | labor_bureau_report:0, labor_independent_survey:0, wage_independent_survey:0, inflation_bureau_report:0, inflation_independent_survey:0 | labor_bureau_report:0, labor_independent_survey:0 |
| interest_rate_probe | n/a | inflation_bureau_report:0, labor_independent_survey:0, labor_bureau_report:0, school_district_report:0, wage_bureau_report:0 | — |
| sports_probe | n/a | out_of_scope_stub:0 | — |
| port_cargo_throughput | 1.000 | port_authority_brief:0, election_centralwire:0, housing_permits_office:0, election_metrodaily:0, transit_authority_update:0 | port_authority_brief:0 |
| school_enrollment_fall | 1.000 | election_centralwire:0, election_metrodaily:0, school_district_report:0 | school_district_report:0 |
| reservoir_storage_level | 1.000 | water_utility_update:0, port_authority_brief:0 | water_utility_update:0 |
| housing_permits_issued | 1.000 | housing_permits_office:0, incident_metrodaily:0, incident_centralwire:0 | housing_permits_office:0 |
| outage_restoration_time | 1.000 | power_grid_status:0 | power_grid_status:0 |
| inflation_rate | 1.000 | inflation_bureau_report:0, inflation_independent_survey:0, labor_bureau_report:0, labor_independent_survey:0, export_trade_report:0 | inflation_bureau_report:0, inflation_independent_survey:0 |
| wage_growth_rate | 1.000 | wage_bureau_report:0, wage_independent_survey:0, labor_bureau_report:0, labor_independent_survey:0, inflation_bureau_report:0 | wage_bureau_report:0, wage_independent_survey:0 |
| export_decline_rate | 1.000 | export_trade_report:0, export_independent_review:0, labor_bureau_report:0, labor_independent_survey:0, inflation_bureau_report:0 | export_trade_report:0, export_independent_review:0 |
| crypto_price_probe | n/a | inflation_bureau_report:0, inflation_independent_survey:0 | — |
| olympic_medals_probe | n/a | school_district_report:0, labor_bureau_report:0 | — |
| dinosaur_fossil_probe | n/a | out_of_scope_stub:0, school_district_report:0, labor_independent_survey:0, labor_bureau_report:0, wage_bureau_report:0 | — |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
