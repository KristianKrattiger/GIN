# SEAR vs RAG Evaluation Report

- Generated: 2026-07-05T21:41:08.334066+00:00
- Model: models/Qwen2.5-7B-Instruct-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_multipara.yaml
- Queries: 1
- Wall-clock per query: 184.68s
- Tokens/s (approx): 0.5

## Overall

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | n/a | 0.000 |
| grounded_precision | n/a | 1.000 |
| attribution_coverage | n/a | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | n/a | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 0 | 2 |
| n_queries | 1 | 1 |

## By eval_layer

### realism

| Metric | rag | no_continuation |
|---|---|---|
| fabrication_rate | n/a | 0.000 |
| grounded_precision | n/a | 1.000 |
| attribution_coverage | n/a | 1.000 |
| counterfactual_adherence | n/a | n/a |
| failure_precision | 0.000 | n/a |
| failure_recall | n/a | n/a |
| cross_node_within_ratio | n/a | 1.000 |
| cross_node_violations | 0 | 0 |
| n_claims | 0 | 2 |
| n_queries | 1 | 1 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 0.000 | 1.000 |
| gold_chunk_coverage | 0.000 | 1.000 |
| supported_irrelevance_rate | n/a | 0.000 |
| chunk_quotation_rate | 0.000 | 0.667 |
| divergence_fidelity | 0.000 | 1.000 |

- **rag** failing query relevance: mp_wildfire_multipara

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| mp_wildfire_multipara | 1.000 | wf_multi_inst:0, wf_multi_grass:0, n1_doc_008:2 | wf_multi_inst:0, wf_multi_grass:0 |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
