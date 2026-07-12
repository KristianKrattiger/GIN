"""Bookkeeper admission gate — invariant maintenance, its falsifiable job.

Each layer of GIN is separately measurable (GIN_Session_Synthesis_v1.md §1.5); the
Bookkeeper's is: no cycles, anchor integrity, correct admit/deny, provenance
stamped. These tests exercise exactly those, independent of Cartographer quality.
"""
import pytest

from gin.bookkeeper import (
    AdmissionCode,
    Bookkeeper,
    GraphState,
    edge_key,
)
from gin.cartographer.models import EdgeProposal, Relation

# chunk_id -> token count
REGISTRY = {"a:0": 20, "b:0": 25, "c:0": 15, "d:0": 30}


def _proposal(src, dst, relation=Relation.CONTRADICTS, *, confidence=0.9,
              src_anchor=None, dst_anchor=None, method="cartographer:test"):
    return EdgeProposal(
        src_chunk_id=src, dst_chunk_id=dst, relation=relation, method=method,
        confidence=confidence, src_anchor=src_anchor, dst_anchor=dst_anchor,
    )


def test_admits_a_valid_proposal_and_stamps_provenance():
    bk = Bookkeeper()
    result = bk.admit(_proposal("a:0", "b:0"), registry=REGISTRY)
    assert result.admitted
    assert result.edge is not None
    assert result.edge.provenance.proposer == "cartographer:test"
    assert result.edge.provenance.confidence == 0.9
    assert result.edge.provenance.admitted_at  # timestamped
    assert result.edge.provenance.content_hash
    assert len(bk.graph) == 1


def test_bookkeeper_is_the_only_writer():
    """A denied proposal never reaches graph state."""
    bk = Bookkeeper()
    bk.admit(_proposal("a:0", "ghost:0"), registry=REGISTRY)
    assert len(bk.graph) == 0


def test_denies_unknown_chunk():
    bk = Bookkeeper()
    r = bk.admit(_proposal("a:0", "ghost:0"), registry=REGISTRY)
    assert r.code == AdmissionCode.DENIED_UNKNOWN_CHUNK


def test_denies_self_loop():
    bk = Bookkeeper()
    r = bk.admit(_proposal("a:0", "a:0"), registry=REGISTRY)
    assert r.code == AdmissionCode.DENIED_SELF_LOOP


def test_verifies_anchor_integrity():
    bk = Bookkeeper()
    # a:0 has 20 tokens; (5, 12) is valid, (5, 40) overruns.
    assert bk.admit(
        _proposal("a:0", "b:0", src_anchor=(5, 12), dst_anchor=(0, 8)),
        registry=REGISTRY,
    ).admitted
    r = bk.admit(_proposal("a:0", "c:0", src_anchor=(5, 40)), registry=REGISTRY)
    assert r.code == AdmissionCode.DENIED_INVALID_ANCHOR
    # empty / inverted spans are invalid too
    assert bk.admit(
        _proposal("a:0", "d:0", src_anchor=(7, 7)), registry=REGISTRY
    ).code == AdmissionCode.DENIED_INVALID_ANCHOR


def test_dedups_symmetric_edges_regardless_of_direction():
    bk = Bookkeeper()
    assert bk.admit(_proposal("a:0", "b:0"), registry=REGISTRY).admitted
    # contradicts is symmetric: b->a is the same edge.
    r = bk.admit(_proposal("b:0", "a:0"), registry=REGISTRY)
    assert r.code == AdmissionCode.DENIED_DUPLICATE
    assert len(bk.graph) == 1


def test_enforces_acyclicity_on_ordering_relations():
    bk = Bookkeeper()
    assert bk.admit(_proposal("a:0", "b:0", Relation.SUPERSEDES), registry=REGISTRY).admitted
    assert bk.admit(_proposal("b:0", "c:0", Relation.SUPERSEDES), registry=REGISTRY).admitted
    # a->b->c admitted; c->a would close a cycle in the supersedes ordering.
    r = bk.admit(_proposal("c:0", "a:0", Relation.SUPERSEDES), registry=REGISTRY)
    assert r.code == AdmissionCode.DENIED_CYCLE
    assert len(bk.graph) == 2


def test_symmetric_relations_are_not_cycle_checked():
    """contradicts carries no ordering: a-b and b-c and c-a all coexist."""
    bk = Bookkeeper()
    assert bk.admit(_proposal("a:0", "b:0"), registry=REGISTRY).admitted
    assert bk.admit(_proposal("b:0", "c:0"), registry=REGISTRY).admitted
    assert bk.admit(_proposal("c:0", "a:0"), registry=REGISTRY).admitted
    assert len(bk.graph) == 3


def test_confidence_floor_denies_low_confidence():
    bk = Bookkeeper(min_confidence=0.5)
    assert bk.admit(_proposal("a:0", "b:0", confidence=0.4), registry=REGISTRY).code \
        == AdmissionCode.DENIED_LOW_CONFIDENCE
    assert bk.admit(_proposal("a:0", "b:0", confidence=0.6), registry=REGISTRY).admitted


def test_admit_all_reports_each_outcome():
    bk = Bookkeeper()
    results = bk.admit_all(
        [_proposal("a:0", "b:0"), _proposal("a:0", "a:0"), _proposal("c:0", "d:0")],
        registry=REGISTRY,
    )
    codes = [r.code for r in results]
    assert codes == [
        AdmissionCode.ADMITTED,
        AdmissionCode.DENIED_SELF_LOOP,
        AdmissionCode.ADMITTED,
    ]
    assert len(bk.graph) == 2


def test_edge_key_normalizes_symmetric_but_not_ordering():
    assert edge_key("a:0", "b:0", Relation.CONTRADICTS) == edge_key("b:0", "a:0", Relation.CONTRADICTS)
    assert edge_key("a:0", "b:0", Relation.SUPERSEDES) != edge_key("b:0", "a:0", Relation.SUPERSEDES)
