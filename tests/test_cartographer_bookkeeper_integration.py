"""Cartographer -> Bookkeeper handoff — the layer separation, end to end.

The Cartographer proposes typed edges; the Bookkeeper is the sole gate that admits
them to canonical graph state. This exercises the real seam: assessments ->
EdgeProposals -> admission, with the reasoning layer never in the loop.
"""
from gin.bookkeeper import AdmissionCode, Bookkeeper
from gin.cartographer import RelatednessProposer, default_chunks
from gin.cartographer.models import EdgeProposal, Relation


def _registry():
    # Token count stand-in: whitespace word count per chunk.
    return {c.chunk_id: len(c.text.split()) for c in default_chunks()}


def _typed_proposals():
    chunks = default_chunks()
    assessments = RelatednessProposer().propose(chunks)
    return [
        EdgeProposal.from_assessment(a) for a in assessments if a.is_typed_edge
    ]


def test_cartographer_proposals_flow_through_admission():
    proposals = _typed_proposals()
    assert proposals  # the proposer emits some typed edges
    bk = Bookkeeper()
    results = bk.admit_all(proposals, registry=_registry())
    # Every proposal references real chunks and is loop-free, so all admit.
    assert all(r.admitted for r in results)
    assert len(bk.graph) == len(proposals)


def test_bookkeeper_rejects_a_proposal_over_a_stale_chunk():
    """A proposal referencing a chunk absent from the registry is denied — the
    anchor-integrity boundary that protects canonical state from stale edges."""
    bk = Bookkeeper()
    stale = EdgeProposal(
        src_chunk_id="inst_wf:0", dst_chunk_id="removed_doc:0",
        relation=Relation.CONTRADICTS, method="cartographer:test", confidence=0.9,
    )
    r = bk.admit(stale, registry=_registry())
    assert r.code == AdmissionCode.DENIED_UNKNOWN_CHUNK
    assert len(bk.graph) == 0


def test_reproposing_an_admitted_edge_is_idempotent():
    proposals = _typed_proposals()
    bk = Bookkeeper()
    bk.admit_all(proposals, registry=_registry())
    before = len(bk.graph)
    # A second sync of the same proposals admits nothing new.
    second = bk.admit_all(proposals, registry=_registry())
    assert all(r.code == AdmissionCode.DENIED_DUPLICATE for r in second)
    assert len(bk.graph) == before
