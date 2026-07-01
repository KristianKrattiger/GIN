"""Tests for synthesis manifest rendering."""
from uuid import uuid4

from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle, SynthesisContext
from gin.corpus.retrieval_manifest import build_retrieval_manifest
from gin.corpus.synthesis_manifest import render_synthesis_manifest
from sear.processor import Segment


def _hit(chunk_id: str, outlet: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=uuid4(),
        text="Emergency services confirmed 142 people received treatment",
        head_sentence="Emergency services confirmed 142 people received treatment",
        eval_layer="realism",
        eval_tag=None,
        content_hash="abc",
        outlet=outlet,
        title="Incident",
        dense_rank=1,
        sparse_rank=2,
        rrf_score=0.0162,
    )


def test_render_synthesis_manifest_key_phrases():
    left = _hit("incident_centralwire:0", "CentralWire")
    right = _hit("incident_metrodaily:0", "MetroDaily")
    edges = [
        EdgeRecord(
            "incident_centralwire:0",
            "incident_metrodaily:0",
            "contradicts",
            note="conflicting hospital and arrest counts",
        )
    ]
    ctx = SynthesisContext(
        doc_index_to_hit={0: left, 1: right},
        cite_index_to_doc={1: 0, 2: 1},
        mode="divergent",
        edges=edges,
        required_doc_groups=[frozenset({0, 1})],
        preferred_starts={(0, 0), (1, 0)},
        active_edge_types={"contradicts"},
    )
    segments = [
        Segment([1, 2, 3], [(0, 0, 3)], "extract", guidance="steered"),
        Segment([4, 5, 6], [(1, 0, 3)], "extract", guidance="steered"),
    ]
    render_output = (
        '"Emergency services confirmed 142 people"  <- EXACT: CentralWire[0:3] [steered]\n'
        '"Emergency services confirmed 98 people"  <- EXACT: MetroDaily[0:3] [steered]'
    )
    bundle_hits = [left, right]
    bundle = SynthesisBundle(
        hits=bundle_hits,
        edges=edges,
        mode="divergent",
    )
    retrieval_manifest = build_retrieval_manifest(
        "downtown incident hospital treatment", bundle
    )

    output = render_synthesis_manifest(
        "downtown incident hospital treatment",
        ctx,
        segments,
        render_output,
        retrieval_manifest=retrieval_manifest,
    )

    assert "Manifest hash:" in output
    assert "contradicts" in output
    assert "Groups satisfied: YES" in output
    assert "[steered]" in output
    assert "Free selections: 0" in output
