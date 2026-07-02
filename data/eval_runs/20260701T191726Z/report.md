# SEAR vs RAG Evaluation Report

- Generated: 2026-07-01T19:17:26.758040+00:00
- Model: /mnt/c/models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: nli (threshold 0.5)
- Query set: /mnt/c/Users/krist/Projects/gin/GIN/data/eval/queryset.yaml
- Queries: 9

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.714 | 0.000 |
| grounded_precision | 0.286 | 1.000 |
| attribution_coverage | 0.286 | 1.000 |
| counterfactual_adherence | 0.000 | 0.000 |
| failure_precision | 0.500 | 0.286 |
| failure_recall | 1.000 | 1.000 |
| cross_node_within_ratio | 0.286 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 7 | 8 |
| n_queries | 9 | 9 |

## By eval_layer

### counterfactual

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 1.000 | n/a |
| grounded_precision | 0.000 | n/a |
| attribution_coverage | 0.000 | n/a |
| counterfactual_adherence | 0.000 | 0.000 |
| failure_precision | n/a | 0.000 |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.000 | n/a |
| cross_node_violations | 0 | 0 |
| n_claims | 1 | 0 |
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
| fabrication_rate | 0.667 | 0.000 |
| grounded_precision | 0.333 | 1.000 |
| attribution_coverage | 0.333 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | 0.000 |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.333 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 6 | 8 |
| n_queries | 6 | 6 |

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| incident_hospital | 1.000 | incident_regionalpost:0, incident_centralwire:0, incident_metrodaily:0, transit_authority_update:0, labor_independent_survey:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| incident_arrests | 1.000 | incident_regionalpost:0, incident_centralwire:0, incident_metrodaily:0, election_centralwire:0, election_metrodaily:0, transit_authority_update:0 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 |
| election_margin | 1.000 | election_metrodaily:0, election_centralwire:0, incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 | election_centralwire:0, election_metrodaily:0 |
| election_turnout | 1.000 | election_metrodaily:0, election_centralwire:0, incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0 | election_centralwire:0, election_metrodaily:0 |
| transit_ridership | 1.000 | incident_regionalpost:0, incident_centralwire:0, incident_metrodaily:0, transit_authority_update:0, labor_bureau_report:0 | transit_authority_update:0 |
| weather_winds | 1.000 | incident_centralwire:0, incident_metrodaily:0, incident_regionalpost:0, election_centralwire:0, election_metrodaily:0, weather_service_brief:0 | weather_service_brief:0 |
| unemployment_rate | 1.000 | incident_regionalpost:0, incident_centralwire:0, incident_metrodaily:0, labor_bureau_report:0, labor_independent_survey:0, transit_authority_update:0 | labor_bureau_report:0, labor_independent_survey:0 |
| interest_rate_probe | n/a | — | — |
| sports_probe | n/a | — | — |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 0.286
