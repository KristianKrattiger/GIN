# SEAR vs RAG Evaluation Report

- Generated: 2026-07-01T17:57:28.968144+00:00
- Model: /mnt/c/models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: /mnt/c/Users/krist/Projects/gin/GIN/data/eval/queryset.yaml
- Queries: 9

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.571 | 0.000 |
| grounded_precision | 0.429 | 1.000 |
| attribution_coverage | 0.429 | 1.000 |
| counterfactual_adherence | 1.000 | 0.000 |
| failure_precision | 0.500 | n/a |
| failure_recall | 1.000 | 0.000 |
| cross_node_within_ratio | 0.429 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 7 | 34 |
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
| cross_node_within_ratio | 1.000 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 1 | 3 |
| n_queries | 1 | 1 |

### out_of_scope

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | n/a | 0.000 |
| grounded_precision | n/a | 1.000 |
| attribution_coverage | n/a | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 1.000 | n/a |
| failure_recall | 1.000 | 0.000 |
| cross_node_within_ratio | n/a | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 0 | 5 |
| n_queries | 2 | 2 |

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | 0.667 | 0.000 |
| grounded_precision | 0.333 | 1.000 |
| attribution_coverage | 0.333 | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | 0.333 | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 6 | 26 |
| n_queries | 6 | 6 |
