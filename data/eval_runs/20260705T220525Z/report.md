# SEAR vs RAG Evaluation Report

- Generated: 2026-07-05T22:05:25.173163+00:00
- Model: models/Qwen2.5-7B-Instruct-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_framing3.yaml
- Queries: 2
- Wall-clock per query: 108.36s
- Tokens/s (approx): 1.1

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
| n_claims | 0 | 4 |
| n_queries | 2 | 2 |

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
| n_claims | 0 | 4 |
| n_queries | 2 | 2 |

## Epistemic quality

| Metric | rag | no_continuation |
|---|---|---|
| query_relevance_rate | 0.000 | 1.000 |
| gold_chunk_coverage | 0.000 | 1.000 |
| supported_irrelevance_rate | n/a | 0.000 |
| chunk_quotation_rate | 0.000 | 0.583 |
| divergence_fidelity | 0.000 | 1.000 |

- **rag** failing query relevance: hf_alderflats_rezoning, hf_kestrel_conditions

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| hf_alderflats_rezoning | 1.000 | hf_alderflats_staff:0, hf_alderflats_tenants:0, n2_doc_004:0 | hf_alderflats_staff:0, hf_alderflats_tenants:0 |
| hf_kestrel_conditions | 1.000 | hf_kestrel_inspection:0, hf_kestrel_tenants:0, hf_alderflats_staff:0, hf_alderflats_tenants:0 | hf_kestrel_inspection:0, hf_kestrel_tenants:0 |

- **rag** mean gold recall@k: 1.000
- **no_continuation** mean gold recall@k: 1.000
