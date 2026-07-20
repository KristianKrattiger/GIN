"""node4 surfacing verifier: thesis-pair identification + PASS/SINK over a
model-free proposer (same injection pattern as test_curator_residue)."""
import pytest

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk
from gin.curator.models import pair_key
from gin.curator.node4_verify import intended_thesis_pairs, verify_surfacing

DOCS = [
    {"doc_id": "n4_doc_001", "metadata": {"topic": "carbon_tax", "stance": "pro"},
     "chunks": [{"position": "0", "text": "carbon tax pro thesis"},
                {"position": "1", "text": "carbon tax pro support"}]},
    {"doc_id": "n4_doc_002", "metadata": {"topic": "carbon_tax", "stance": "con"},
     "chunks": [{"position": "0", "text": "carbon tax con thesis"},
                {"position": "1", "text": "carbon tax con support"}]},
]


def test_intended_thesis_pairs_uses_position_zero():
    got = intended_thesis_pairs(DOCS)
    assert got == {"carbon_tax": pair_key("n4_doc_001:0", "n4_doc_002:0")}


def test_missing_position_zero_raises_named_value_error():
    malformed = [{
        "doc_id": "n4_doc_099",
        "metadata": {"topic": "carbon_tax", "stance": "pro"},
        "chunks": [{"position": 1, "text": "no thesis chunk here"}],
    }]
    with pytest.raises(ValueError, match="n4_doc_099"):
        intended_thesis_pairs(malformed)


def _proposer(cos_map):
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos_map.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: False,
        nli_scores=lambda p, h: (0.0, 0.0, 1.0),  # neutral => model-free ranking
    )


def test_pass_when_thesis_pair_surfaces():
    chunks = [LabeledChunk("n4_doc_001:0", "carbon tax pro thesis"),
              LabeledChunk("n4_doc_002:0", "carbon tax con thesis")]
    cos = {frozenset({"carbon tax pro thesis", "carbon tax con thesis"}): 0.40}
    results = verify_surfacing(chunks, DOCS, _proposer(cos))
    assert len(results) == 1
    assert results[0].topic == "carbon_tax"
    assert results[0].passed is True
    assert results[0].rank == 0


def test_sink_when_below_floor():
    chunks = [LabeledChunk("n4_doc_001:0", "carbon tax pro thesis"),
              LabeledChunk("n4_doc_002:0", "carbon tax con thesis")]
    cos = {frozenset({"carbon tax pro thesis", "carbon tax con thesis"}): 0.05}
    results = verify_surfacing(chunks, DOCS, _proposer(cos))
    assert results[0].passed is False
    assert results[0].rank is None
