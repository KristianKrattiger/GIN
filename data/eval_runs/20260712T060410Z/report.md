# SEAR vs RAG Evaluation Report

- Generated: 2026-07-12T06:04:10.631121+00:00
- Model: models/Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Verifier: overlap (threshold 0.5)
- Query set: data/eval/queryset_framing3.yaml
- Queries: 2
- GPU layers: -1
- Wall-clock per query: 73.56s
- Tokens/s (approx): 0.9

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
| n_claims | 4 |
| n_queries | 2 |

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
| n_claims | 4 |
| n_queries | 2 |

## Epistemic quality

| Metric | no_continuation |
|---|---|
| query_relevance_rate | 0.500 |
| gold_chunk_coverage | 0.500 |
| supported_irrelevance_rate | 0.250 |
| chunk_quotation_rate | 0.333 |
| divergence_fidelity | 0.500 |

- **no_continuation** failing query relevance: hf_kestrel_conditions

## Retrieval quality

| Query | gold_recall@k | retrieved | gold |
|---|---|---|---|
| hf_alderflats_rezoning | 1.000 | hf_alderflats_staff:0, hf_alderflats_tenants:0, n2_doc_004:0, n2_doc_001:2, n2_doc_002:3, n2_doc_002:4 | hf_alderflats_staff:0, hf_alderflats_tenants:0 |
| hf_kestrel_conditions | 1.000 | hf_alderflats_staff:0, n2_doc_001:2, hf_kestrel_inspection:0, hf_kestrel_tenants:0, water_utility_update:0, n1_doc_002:2 | hf_kestrel_inspection:0, hf_kestrel_tenants:0 |

- **no_continuation** mean gold recall@k: 1.000
