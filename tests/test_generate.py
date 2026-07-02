"""Tests for No-Continuation decode parameter resolution (gin.corpus.generate)."""
from uuid import uuid4

from gin.corpus.generate import _resolve_decode_params
from gin.corpus.models import ChunkHit, SynthesisBundle, SynthesisContext

DOC = uuid4()


def _hit(chunk_id: str, outlet: str, eval_tag: str | None = None) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=DOC,
        text="text",
        head_sentence="head",
        eval_layer="counterfactual",
        eval_tag=eval_tag,
        content_hash="x",
        outlet=outlet,
        title="title",
        rrf_score=0.5,
    )


def _bundle(hits: list[ChunkHit], mode: str) -> SynthesisBundle:
    return SynthesisBundle(hits=hits, edges=[], mode=mode, pairs=[])


def _ctx(hits: list[ChunkHit], mode: str, groups=None) -> SynthesisContext:
    return SynthesisContext(
        doc_index_to_hit={i: h for i, h in enumerate(hits)},
        cite_index_to_doc={i + 1: i for i in range(len(hits))},
        mode=mode,
        required_doc_groups=groups or [],
    )


def _resolve(hits, mode, groups=None, **overrides):
    bundle = _bundle(hits, mode)
    ctx = _ctx(hits, mode, groups)
    defaults = dict(
        require_cites=False,
        stop_when_satisfied=False,
        min_span_len=None,
        max_tokens=None,
    )
    defaults.update(overrides)
    return _resolve_decode_params(bundle, ctx, **defaults)


def test_competing_same_tag_detected():
    hits = [
        _hit("labor_bureau:0", "NationalLaborBureau", "unemployment_probe"),
        _hit("labor_survey:0", "IndependentEconomicReview", "unemployment_probe"),
    ]
    params = _resolve(hits, "convergent")
    assert params["competing_same_tag"] is True
    assert params["stop_after_first_extract"] is False
    assert params["max_tokens"] == 100
    assert params["divergent"] is False


def test_competing_same_tag_requires_different_outlets():
    hits = [
        _hit("a:0", "SameOutlet", "unemployment_probe"),
        _hit("b:0", "SameOutlet", "unemployment_probe"),
    ]
    params = _resolve(hits, "convergent")
    assert params["competing_same_tag"] is False
    assert params["stop_after_first_extract"] is True
    assert params["max_tokens"] == 60


def test_competing_same_tag_requires_matching_tags():
    hits = [
        _hit("school:0", "MetroSchoolBoard", "enrollment_realism"),
        _hit("election:0", "CentralWire", "election_divergence"),
    ]
    params = _resolve(hits, "convergent")
    assert params["competing_same_tag"] is False


def test_divergent_without_groups_does_not_block_eos():
    """A divergent bundle with no contradicts groups must be able to stop —
    blocking EOS on never-satisfiable groups forced max_tokens rambling."""
    hits = [_hit("a:0", "A"), _hit("b:0", "B")]
    params = _resolve(hits, "divergent")
    assert params["block_eos"] is False
    assert params["stop_when_satisfied"] is False


def test_divergent_with_groups_blocks_eos():
    hits = [_hit("a:0", "A"), _hit("b:0", "B")]
    params = _resolve(hits, "divergent", groups=[frozenset({0, 1})])
    assert params["block_eos"] is True
    assert params["stop_when_satisfied"] is True
    assert params["max_tokens"] == 40 + 25
