"""Tests for No-Continuation decode parameter resolution (gin.corpus.generate)."""
from uuid import uuid4

from gin.corpus.generate import _resolve_decode_params, decode_bundle
from gin.corpus.models import ChunkHit, SynthesisBundle, SynthesisContext
from gin.eval.edge_degradation import GreedyMaskDecoder
from sear.corpus import Corpus

DOC = uuid4()

ANOMALY_TEXT = (
    "In 2023, global surface temperature was about 2.12 degrees F "
    "(1.18 degrees C) above the 20th-century average, beating the next warmest "
    "year (2016) by roughly 0.27 degrees F."
)


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
    # One group == one contradicts pair == two full sentences of budget; EOS
    # still fires the moment both sides are quoted, so this is a ceiling.
    # 40 + 80 = 120: measured max full divergent decode is 97 tokens (water
    # pair), so this is 24% headroom over the corpus worst case (plan §6 #5).
    assert params["max_tokens"] == 40 + 80


def test_convergent_numeric_sentence_closes_at_sentence_end():
    """tn_2023_anomaly regression: convergent decode must not close after '2.'"""
    llm = GreedyMaskDecoder()
    corpus = Corpus.from_texts({"anomaly": ANOMALY_TEXT}, tokenize=llm.tokenize)
    hit = ChunkHit(
        chunk_id="n1_doc_002:1",
        doc_id=DOC,
        text=ANOMALY_TEXT,
        head_sentence=ANOMALY_TEXT.split(",")[0] + ".",
        eval_layer="realism",
        eval_tag=None,
        content_hash="x",
        outlet="NOAA",
        title="2023 anomaly",
        rrf_score=0.9,
    )
    bundle = SynthesisBundle(hits=[hit], edges=[], mode="convergent", pairs=[])
    spans = corpus.sentence_starts
    preferred = {(0, pos) for (doc, pos) in spans if doc == 0}
    ctx = SynthesisContext(
        doc_index_to_hit={0: hit},
        cite_index_to_doc={1: 0},
        mode="convergent",
        preferred_starts=preferred,
        ranked_sentence_starts=[(0, pos, 1.0) for (doc, pos) in spans if doc == 0],
        top_doc_idx=0,
    )
    result = decode_bundle(
        "2023 global surface temperature anomaly",
        corpus,
        ctx,
        bundle,
        llm,
        chat_template="plain",
        query_steered=True,
    )
    assert "2.12 degrees" in result.raw_text
    assert "20th-century average" in result.raw_text
