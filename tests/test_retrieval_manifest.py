"""Tests for retrieval manifest build and persistence."""
import json
from uuid import uuid4

from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from gin.corpus.retrieval_manifest import (
    build_retrieval_manifest,
    write_retrieval_manifest,
)


def _hit(chunk_id: str, score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=uuid4(),
        text="Officials responded to a downtown incident",
        head_sentence="Officials responded to a downtown incident",
        eval_layer="realism",
        eval_tag=None,
        content_hash="abc",
        outlet="CentralWire",
        title="Incident",
        dense_rank=1,
        sparse_rank=2,
        rrf_score=score,
    )


def _bundle(score: float = 0.0162) -> SynthesisBundle:
    left = _hit("incident_centralwire:0", score)
    right = _hit("incident_metrodaily:0", score - 0.0007)
    edges = [EdgeRecord("incident_centralwire:0", "incident_metrodaily:0", "contradicts")]
    return SynthesisBundle(
        hits=[left, right],
        edges=edges,
        mode="divergent",
        pairs=[(left, right, edges[0])],
    )


def test_build_retrieval_manifest_deterministic_hash():
    query = "downtown incident hospital treatment"
    bundle = _bundle()
    m1 = build_retrieval_manifest(query, bundle)
    m2 = build_retrieval_manifest(query, bundle)
    assert m1.manifest_hash == m2.manifest_hash
    assert m1.manifest_hash
    assert m1.query_hash == m2.query_hash


def test_changing_rrf_score_changes_hash():
    query = "downtown incident hospital treatment"
    m1 = build_retrieval_manifest(query, _bundle(0.0162))
    m2 = build_retrieval_manifest(query, _bundle(0.0163))
    assert m1.manifest_hash != m2.manifest_hash


def test_write_retrieval_manifest(tmp_path):
    query = "downtown incident hospital treatment"
    manifest = build_retrieval_manifest(query, _bundle())
    store_root = tmp_path / "retrieval_manifests"
    path = write_retrieval_manifest(manifest, base_dir=store_root)
    expected = store_root / manifest.manifest_hash[:2] / f"{manifest.manifest_hash}.json"
    assert path == expected
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == manifest.to_dict()
    path2 = write_retrieval_manifest(manifest, base_dir=store_root)
    assert path2 == path
