# SEAR vs RAG Evaluation Report

- Generated: 2026-07-12T06:00:11.248671+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_multipara.yaml
- Queries: 1
- GPU layers: -1
- Wall-clock per query: 416.13s
- Tokens/s (approx): 0.1

## Overall

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
| n_claims | 2 |
| n_queries | 1 |

## By eval_layer

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
| n_claims | 2 |
| n_queries | 1 |

## Epistemic quality

| Metric | no_continuation |
|---|---|
| query_relevance_rate | 0.000 |
| gold_chunk_coverage | 0.000 |
| supported_irrelevance_rate | 0.500 |
| chunk_quotation_rate | 0.333 |
| divergence_fidelity | 0.000 |

- **no_continuation** failing query relevance: mp_wildfire_multipara

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| mp_wildfire_multipara | 1.000 | n1_doc_008:2, n2_doc_008:3, wf_multi_grass:0, wf_multi_inst:0, out_of_scope_stub:0, water_utility_update:0 | wf_multi_inst:0, wf_multi_grass:0 |

- **no_continuation** mean gold recall@k: 1.000
