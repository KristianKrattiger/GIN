"""Unit tests for scan pruning, doc-pair dedup, and relation re-check."""
from pathlib import Path

import yaml

from gin.bookkeeper import AdmissionCode, Bookkeeper
from gin.bookkeeper.relation_verify import verify_contradicts
from gin.cartographer.combined import CombinedRelationProposer, Thresholds
from gin.cartographer.models import EdgeProposal, LabeledChunk, Relation
from gin.cartographer.scan import (
    dedupe_doc_pair_proposals,
    filter_chunks,
    proposals_from_pairs,
    prune_pairs_by_relatedness,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_EDGES = ROOT / "data" / "corpus_edges.yaml"

_TWONODE_COS = {
    frozenset({"n1_doc_005:2", "n2_doc_001:4"}): 0.390,
    frozenset({"n1_doc_008:0", "n2_doc_005:1"}): 0.418,
    frozenset({"n1_doc_009:0", "n2_doc_008:2"}): 0.134,
}
_TWONODE_NLI = {
    frozenset({"n1_doc_005:2", "n2_doc_001:4"}): (0.90, 0.02, 0.08),
    frozenset({"n1_doc_008:0", "n2_doc_005:1"}): (0.88, 0.03, 0.09),
    frozenset({"n1_doc_009:0", "n2_doc_008:2"}): (0.85, 0.04, 0.11),
}


def _load_twonode_chunks() -> list[LabeledChunk]:
    data = yaml.safe_load(CORPUS_EDGES.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for edge in data["edges"]:
        ids.add(edge["src"])
        ids.add(edge["dst"])
    return [LabeledChunk(cid, f"{cid}::stub") for cid in sorted(ids)]


def _twonode_scorers():
    def _cid(text: str) -> str:
        return text.split("::", 1)[0]

    def embed_cos(a: str, b: str) -> float:
        return _TWONODE_COS.get(frozenset({_cid(a), _cid(b)}), 0.05)

    def nli_scores(a: str, b: str) -> tuple[float, float, float]:
        return _TWONODE_NLI.get(frozenset({_cid(a), _cid(b)}), (0.02, 0.90, 0.08))

    return embed_cos, nli_scores


def test_filter_chunks_excludes_doc_ids():
    chunks = [
        LabeledChunk("keep:0", "wildfire acreage report"),
        LabeledChunk("stub:0", "mars rover timeline"),
    ]
    outlets = {"keep:0": "A", "stub:0": "B"}
    kept, kept_outlets = filter_chunks(chunks, outlets, exclude_doc_ids=["stub"])
    assert [c.chunk_id for c in kept] == ["keep:0"]
    assert kept_outlets == {"keep:0": "A"}


def test_prune_pairs_by_relatedness_drops_unrelated():
    chunks = [
        LabeledChunk("a:0", "wildfire burned fifty thousand acres in Oregon"),
        LabeledChunk("b:0", "harbor district referendum turnout statistics"),
        LabeledChunk("c:0", "wildfire suppression costs exceeded budget"),
    ]
    pairs = [(chunks[0], chunks[1]), (chunks[0], chunks[2])]
    kept, pruned = prune_pairs_by_relatedness(chunks, pairs, floor=0.20)
    assert pruned >= 1
    assert len(kept) < len(pairs)
    kept_ids = {tuple(sorted((a.chunk_id, b.chunk_id))) for a, b in kept}
    assert ("a:0", "b:0") not in kept_ids


def test_dedupe_doc_pair_proposals_prefers_nli_over_band():
    band = EdgeProposal(
        src_chunk_id="n1_doc_005:2",
        dst_chunk_id="n2_doc_001:2",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.90,
    )
    nli = EdgeProposal(
        src_chunk_id="n1_doc_005:2",
        dst_chunk_id="n2_doc_001:4",
        relation=Relation.CONTRADICTS,
        method="combined_relation:nli",
        confidence=0.70,
    )
    kept, dropped = dedupe_doc_pair_proposals([band, nli])
    assert dropped == 1
    contradicts = [p for p in kept if p.relation == Relation.CONTRADICTS]
    assert len(contradicts) == 1
    assert contradicts[0].method.endswith(":nli")
    assert contradicts[0].dst_chunk_id == "n2_doc_001:4"


def test_dedupe_doc_pair_proposals_keeps_highest_confidence():
    low = EdgeProposal(
        src_chunk_id="n1_doc_005:2",
        dst_chunk_id="n2_doc_001:2",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.55,
    )
    high = EdgeProposal(
        src_chunk_id="n1_doc_005:2",
        dst_chunk_id="n2_doc_001:4",
        relation=Relation.CONTRADICTS,
        method="combined_relation:nli",
        confidence=0.90,
    )
    kept, dropped = dedupe_doc_pair_proposals([low, high])
    assert dropped == 1
    contradicts = [p for p in kept if p.relation == Relation.CONTRADICTS]
    assert len(contradicts) == 1
    assert contradicts[0].dst_chunk_id == "n2_doc_001:4"


def test_verify_contradicts_denies_bidirectional_entailment():
    proposal = EdgeProposal(
        src_chunk_id="a:0",
        dst_chunk_id="b:0",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.60,
    )

    def nli(_a: str, _b: str) -> tuple[float, float, float]:
        return (0.05, 0.90, 0.05)

    result = verify_contradicts(
        proposal,
        src_text="wildfires burned 56580 acres",
        dst_text="56580 acres burned in wildfires",
        nli_scores=nli,
    )
    assert not result.ok


def test_verify_contradicts_passes_nli_channel():
    proposal = EdgeProposal(
        src_chunk_id="a:0",
        dst_chunk_id="b:0",
        relation=Relation.CONTRADICTS,
        method="combined_relation:nli",
        confidence=0.95,
    )

    def nli(_a: str, _b: str) -> tuple[float, float, float]:
        return (0.95, 0.02, 0.03)

    result = verify_contradicts(
        proposal,
        src_text="revenue was overstated",
        dst_text="revenue was accurate",
        nli_scores=nli,
    )
    assert result.ok


def test_verify_no_longer_owns_confidence_floors():
    """The re-check is entailment-only: confidence floors are the Bookkeeper's
    job (it denies band conf < FRAMING_BAND_FLOOR), and the old band/nli
    branches were circular — same NLI signal, thresholds the proposer already
    applied — so they could never catch the proposer's systematic errors."""
    proposal = EdgeProposal(
        src_chunk_id="a:0",
        dst_chunk_id="b:0",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.20,
    )

    def nli(_a: str, _b: str) -> tuple[float, float, float]:
        return (0.10, 0.20, 0.70)

    result = verify_contradicts(
        proposal,
        src_text="topic A claim",
        dst_text="topic B claim",
        nli_scores=nli,
    )
    assert result.ok


def test_bookkeeper_floor_denies_band_below_framing_floor():
    """The denial the old verify branch duplicated lives at the Bookkeeper
    confidence gate."""
    registry = {"a:0": 10, "b:0": 10}
    proposal = EdgeProposal(
        src_chunk_id="a:0",
        dst_chunk_id="b:0",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.20,
    )
    bk = Bookkeeper(min_confidence=0.5)
    result = bk.admit(proposal, registry=registry)
    assert result.code == AdmissionCode.DENIED_LOW_CONFIDENCE


def test_verify_contradicts_passes_framing_band():
    proposal = EdgeProposal(
        src_chunk_id="a:0",
        dst_chunk_id="b:0",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.39,
    )

    def nli(_a: str, _b: str) -> tuple[float, float, float]:
        return (0.10, 0.20, 0.70)

    result = verify_contradicts(
        proposal,
        src_text="institutional emissions framing",
        dst_text="grassroots emissions framing",
        nli_scores=nli,
    )
    assert result.ok


def test_bookkeeper_denies_relation_mismatch():
    registry = {"a:0": 10, "b:0": 10}
    proposal = EdgeProposal(
        src_chunk_id="a:0",
        dst_chunk_id="b:0",
        relation=Relation.CONTRADICTS,
        method="combined_relation:band",
        confidence=0.55,
    )

    def verifier(_p: EdgeProposal):
        from gin.bookkeeper.relation_verify import RelationVerifyResult

        return RelationVerifyResult(ok=False, reason="test deny")

    bk = Bookkeeper(min_confidence=0.5, relation_verifier=verifier)
    result = bk.admit(proposal, registry=registry)
    assert result.code == AdmissionCode.DENIED_RELATION_MISMATCH


def test_pruning_keeps_twonode_gold_contradicts_proposable():
    chunks = _load_twonode_chunks()
    all_pairs = [
        (chunks[i], chunks[j])
        for i in range(len(chunks))
        for j in range(i + 1, len(chunks))
    ]
    kept, _pruned = prune_pairs_by_relatedness(chunks, all_pairs, floor=0.20)
    embed_cos, nli_scores = _twonode_scorers()
    proposer = CombinedRelationProposer(
        embed_cos=embed_cos,
        nli_scores=nli_scores,
        thresholds=Thresholds(gate_floor=0.13, corroborate_ceiling=0.60, contra_threshold=0.50),
    )
    proposals = proposals_from_pairs(proposer, kept)
    admitted_keys = {
        frozenset({p.src_chunk_id, p.dst_chunk_id})
        for p in proposals
        if p.relation == Relation.CONTRADICTS
    }
    for src, dst in (
        ("n1_doc_005:2", "n2_doc_001:4"),
        ("n1_doc_008:0", "n2_doc_005:1"),
        ("n1_doc_009:0", "n2_doc_008:2"),
    ):
        assert frozenset({src, dst}) in admitted_keys
