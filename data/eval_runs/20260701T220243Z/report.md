# SEAR vs RAG Evaluation Report

- Generated: 2026-07-01T22:02:43.036442+00:00
- Model: /mnt/c/models/Mistral-7B-INstruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: /mnt/c/Users/krist/Projects/gin/GIN/data/eval/queryset.yaml
- Queries: 9
- Wall-clock per query: 36.51s
- Tokens/s (approx): 0.6

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.545 | 0.000 |
| grounded_precision | 0.455 | 1.000 |
| attribution_coverage | 0.455 | 1.000 |
| counterfactual_adherence | 1.000 | 0.000 |
| failure_precision | 0.667 | 0.667 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | 0.455 | 0.929 |
| cross_node_violations | 0 | 0 |
| n_claims | 11 | 14 |
| n_queries | 9 | 9 |

## By eval_layer

### counterfactual

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.000 | 0.000 |
| grounded_precision | 1.000 | 1.000 |
| attribution_coverage | 1.000 | 1.000 |
| counterfactual_adherence | 1.000 | 0.000 |
| failure_precision | n/a | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 1.000 | 0.667 |
| cross_node_violations | 0 | 0 |
| n_claims | 1 | 3 |
| n_queries | 1 | 1 |

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
| n_queries | 2 | 2 |

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.600 | 0.000 |
| grounded_precision | 0.400 | 1.000 |
| attribution_coverage | 0.400 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | 0.000 |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.400 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 10 | 11 |
| n_queries | 6 | 6 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 0.889 | 0.889 |
| gold_chunk_coverage | 0.595 | 0.857 |
| supported_irrelevance_rate | 0.000 | 0.071 |
| chunk_quotation_rate | 0.267 | 0.333 |
| divergence_fidelity | 0.875 | 1.000 |

- **rag** failing query relevance: weather_winds
- **no_continuation** failing query relevance: weather_winds

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| incident_hospital | 1.000 | incident_regionalpost:0, incident_centralwire:0, incident_metrodaily:0, power_grid_status:0, housing_permits_office:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| incident_arrests | 1.000 | incident_regionalpost:0, incident_centralwire:0, incident_metrodaily:0, housing_permits_office:0, transit_authority_update:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| election_margin | 1.000 | election_metrodaily:0, election_centralwire:0, port_authority_brief:0, school_district_report:0, housing_permits_office:0 | election_centralwire:0, election_metrodaily:0 |
| election_turnout | 1.000 | election_centralwire:0, election_metrodaily:0, school_district_report:0, port_authority_brief:0, inflation_independent_survey:0 | election_centralwire:0, election_metrodaily:0 |
| transit_ridership | 1.000 | transit_authority_update:0, power_grid_status:0, wage_independent_survey:0, port_authority_brief:0, wage_bureau_report:0 | transit_authority_update:0 |
| weather_winds | 1.000 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0, weather_service_brief:0, port_authority_brief:0, power_grid_status:0 | weather_service_brief:0 |
| unemployment_rate | 1.000 | labor_bureau_report:0, labor_independent_survey:0, inflation_bureau_report:0, wage_independent_survey:0, inflation_independent_survey:0 | labor_bureau_report:0, labor_independent_survey:0 |
| interest_rate_probe | n/a | inflation_bureau_report:0, labor_independent_survey:0, labor_bureau_report:0, school_district_report:0, wage_bureau_report:0 | — |
| sports_probe | n/a | election_metrodaily:0, election_centralwire:0, transit_authority_update:0, out_of_scope_stub:0, water_utility_update:0, power_grid_status:0 | — |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
