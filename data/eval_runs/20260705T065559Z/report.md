# SEAR vs RAG Evaluation Report

- Generated: 2026-07-05T06:55:59.907193+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_multipara.yaml
- Queries: 1
- Wall-clock per query: 107.79s
- Tokens/s (approx): 0.4

## Overall

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
| n_claims | 1 | 2 |
| n_queries | 1 | 1 |

## By eval_layer

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
| n_claims | 1 | 2 |
| n_queries | 1 | 1 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 1.000 | 1.000 |
| gold_chunk_coverage | 0.500 | 1.000 |
| supported_irrelevance_rate | 0.000 | 0.000 |
| chunk_quotation_rate | 0.333 | 0.667 |
| divergence_fidelity | 0.000 | 1.000 |


## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| mp_wildfire_multipara | 1.000 | wf_multi_inst:0, wf_multi_grass:0, n1_doc_008:2 | wf_multi_inst:0, wf_multi_grass:0 |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
