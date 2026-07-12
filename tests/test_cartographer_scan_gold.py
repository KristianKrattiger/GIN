"""Deterministic scan-vs-gold evaluation on twonode chunk ids."""
from pathlib import Path

import yaml

from gin.bookkeeper import AdmissionCode, Bookkeeper
from gin.cartographer.combined import CombinedRelationProposer, Thresholds
from gin.cartographer.evaluation import _key
from gin.cartographer.gold_edges import gold_contradicts_keys
from gin.cartographer.models import EdgeProposal, LabeledChunk, Relation
from gin.cartographer.scan import proposals_from_pairs, whitespace_token_count

ROOT = Path(__file__).resolve().parents[1]
CORPUS_EDGES = ROOT / "data" / "corpus_edges.yaml"

_TWONODE_COS = {
    frozenset({"n1_doc_005:2", "n2_doc_001:4"}): 0.390,
    frozenset({"n1_doc_008:0", "n2_doc_005:1"}): 0.418,
    frozenset({"n1_doc_009:0", "n2_doc_008:2"}): 0.134,
    frozenset({"n1_doc_008:0", "n1_doc_008:2"}): 0.727,
}
_TWONODE_NLI = {
    frozenset({"n1_doc_005:2", "n2_doc_001:4"}): (0.90, 0.02, 0.08),
    frozenset({"n1_doc_008:0", "n2_doc_005:1"}): (0.88, 0.03, 0.09),
    frozenset({"n1_doc_009:0", "n2_doc_008:2"}): (0.85, 0.04, 0.11),
    frozenset({"n1_doc_008:0", "n1_doc_008:2"}): (0.05, 0.90, 0.05),
}


def _chunk_id_from_text(text: str) -> str:
    return text.split("::", 1)[0]


def _load_twonode_chunks() -> list[LabeledChunk]:
    data = yaml.safe_load(CORPUS_EDGES.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for edge in data["edges"]:
        ids.add(edge["src"])
        ids.add(edge["dst"])
    ids.add("n1_doc_008:2")
    return [LabeledChunk(cid, f"{cid}::stub") for cid in sorted(ids)]


def _scorers():
    def embed_cos(a: str, b: str) -> float:
        key = frozenset({_chunk_id_from_text(a), _chunk_id_from_text(b)})
        return _TWONODE_COS.get(key, 0.05)

    def nli_scores(premise: str, hypothesis: str) -> tuple[float, float, float]:
        key = frozenset({_chunk_id_from_text(premise), _chunk_id_from_text(hypothesis)})
        return _TWONODE_NLI.get(key, (0.02, 0.90, 0.08))

    return embed_cos, nli_scores


def test_gold_twonode_pairs_proposed_and_admitted():
    chunks = _load_twonode_chunks()
    embed_cos, nli_scores = _scorers()
    proposer = CombinedRelationProposer(
        embed_cos=embed_cos,
        nli_scores=nli_scores,
        thresholds=Thresholds(gate_floor=0.13, corroborate_ceiling=0.60, contra_threshold=0.50),
    )
    pairs = [(chunks[i], chunks[j]) for i in range(len(chunks)) for j in range(i + 1, len(chunks))]
    proposals = proposals_from_pairs(proposer, pairs)
    registry = {ch.chunk_id: whitespace_token_count(ch.text) for ch in chunks}
    bk = Bookkeeper(min_confidence=0.5)
    results = bk.admit_all(proposals, registry=registry)
    admitted = {
        _key(r.edge.src_chunk_id, r.edge.dst_chunk_id)
        for r in results
        if r.code == AdmissionCode.ADMITTED and r.edge and r.edge.relation == Relation.CONTRADICTS
    }
    gold = gold_contradicts_keys([CORPUS_EDGES])
    assert gold <= admitted

    class_c = _key("n1_doc_008:0", "n1_doc_008:2")
    assert class_c not in admitted
    for p in proposals:
        if _key(p.src_chunk_id, p.dst_chunk_id) == class_c:
            assert p.relation != Relation.CONTRADICTS


def test_doc_pair_keys_collapse_chunk_keys():
    from gin.cartographer.scan_eval import doc_pair_keys

    keys = {
        _key("n1_doc_005:1", "n2_doc_001:1"),
        _key("n1_doc_005:2", "n2_doc_001:4"),
        _key("hf_alderflats_staff:0", "hf_alderflats_tenants:0"),
    }
    assert doc_pair_keys(keys) == {
        frozenset({"n1_doc_005", "n2_doc_001"}),
        frozenset({"hf_alderflats_staff", "hf_alderflats_tenants"}),
    }


def test_doc_level_metrics_count_anchor_disagreement_as_discovery():
    """Gold n1_doc_005:2~n2_doc_001:4 admitted at different chunk anchors
    (:1~:1) is a doc-pair discovery, not a miss plus a false positive — the
    dedupe keeps one chunk pair per doc pair, the gold YAML picks another."""
    from gin.cartographer.scan_eval import doc_level_metrics

    gold = {_key("n1_doc_005:2", "n2_doc_001:4"), _key("a:0", "b:0")}
    admitted = {_key("n1_doc_005:1", "n2_doc_001:1"), _key("c:0", "d:0")}
    tp, fp_keys, missed_keys = doc_level_metrics(admitted, gold)
    assert tp == 1
    assert fp_keys == [frozenset({"c", "d"})]
    assert missed_keys == [frozenset({"a", "b"})]


def test_machine_metrics_exclude_issue_frame_gold():
    """Machine scan metrics are scored against story-class gold only; the
    issue_frame class is curated (machine-undetectable) and tracked apart."""
    from gin.cartographer.scan_eval import split_gold_by_class
    from gin.cartographer.gold_edges import gold_pairs

    machine, curated = split_gold_by_class(gold_pairs())
    machine_keys = {_key(g.src_chunk_id, g.dst_chunk_id) for g in machine}
    curated_keys = {_key(g.src_chunk_id, g.dst_chunk_id) for g in curated}
    assert _key("n1_doc_009:0", "n2_doc_008:2") in curated_keys
    assert _key("hf_alderflats_staff:0", "hf_alderflats_tenants:0") in machine_keys
    assert len(curated) == 3
    assert len(machine) == 8


def test_scan_eval_result_reports_curated_class_separately():
    from gin.cartographer.evaluation import CartographerMetrics
    from gin.cartographer.scan_eval import ScanEvalResult

    result = ScanEvalResult(
        metrics=CartographerMetrics(None, None, None, None),
        admitted_contradicts=[],
        curated_gold_keys=[frozenset({"n1_doc_009:0", "n2_doc_008:2"})],
    )
    d = result.to_dict()
    assert d["curated_gold_count"] == 1
    assert d["curated_gold_keys"] == [["n1_doc_009:0", "n2_doc_008:2"]]


def test_gold_pairs_typed_contradicts_directly():
    chunks = {c.chunk_id: c for c in _load_twonode_chunks()}
    embed_cos, nli_scores = _scorers()
    proposer = CombinedRelationProposer(
        embed_cos=embed_cos,
        nli_scores=nli_scores,
        thresholds=Thresholds(gate_floor=0.13, corroborate_ceiling=0.60, contra_threshold=0.50),
    )
    for src, dst in (
        ("n1_doc_005:2", "n2_doc_001:4"),
        ("n1_doc_008:0", "n2_doc_005:1"),
        ("n1_doc_009:0", "n2_doc_008:2"),
    ):
        assessment = proposer.assess_pair(chunks[src], chunks[dst])
        assert assessment.relation == Relation.CONTRADICTS
        proposal = EdgeProposal.from_assessment(assessment)
        result = Bookkeeper(min_confidence=0.5).admit(
            proposal,
            registry={cid: whitespace_token_count(c.text) for cid, c in chunks.items()},
        )
        assert result.code == AdmissionCode.ADMITTED
