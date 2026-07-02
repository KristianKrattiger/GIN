"""Tests for materialize steering and divergence alignment."""
from uuid import uuid4

import numpy as np

from gin.corpus.divergence import compute_divergence_zones, shared_sentence_starts
from gin.corpus.materialize import materialize_synthesis_bundle
from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from gin.corpus.relevance import score_starts_by_sentence_match, score_starts_for_convergent
from sear.corpus import Corpus, sentence_token_spans
from sear.processor import ExtractiveCopyConstraint, NEG_INF

DOC = uuid4()


def _tok(b: bytes) -> list[int]:
    vocab: dict[str, int] = {}
    ids: list[int] = []
    for w in b.decode().split():
        if w not in vocab:
            vocab[w] = len(vocab)
        ids.append(vocab[w])
    return ids


def _hit(cid: str, text: str, outlet: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=cid,
        doc_id=DOC,
        text=text,
        head_sentence="",
        eval_layer="realism",
        eval_tag="incident_divergence",
        content_hash="",
        outlet=outlet,
        title="t",
    )


def _incident_bundle():
    shared = (
        "Officials responded to a downtown incident Tuesday evening. "
        "Emergency services confirmed"
    )
    central = _hit("incident_centralwire:0", shared + " 142 people received treatment at area hospitals. Police said 23 arrests were made.", "CentralWire")
    metro = _hit("incident_metrodaily:0", shared + " 98 people received treatment at area hospitals. Police said 11 arrests were made.", "MetroDaily")
    regional = _hit("incident_regionalpost:0", shared + " 142 people received treatment at area hospitals. Police said 11 arrests were made.", "RegionalPost")
    edges = [
        EdgeRecord("incident_centralwire:0", "incident_metrodaily:0", "contradicts"),
        EdgeRecord("incident_centralwire:0", "incident_regionalpost:0", "contradicts"),
    ]
    bundle = SynthesisBundle(
        hits=[regional, central, metro],
        edges=edges,
        mode="divergent",
        pairs=[
            (central, metro, edges[0]),
            (central, regional, edges[1]),
        ],
    )
    return bundle


def test_forbidden_never_intersects_divergence_starts():
    bundle = _incident_bundle()
    corpus, ctx = materialize_synthesis_bundle(
        bundle,
        _tok,
        query="downtown incident hospital treatment",
    )
    all_div = {
        (d, p) for d, positions in ctx.divergence_starts.items() for p in positions
    }
    assert not (ctx.forbidden_starts & all_div)


def test_relevance_and_divergence_share_sentence_positions():
    bundle = _incident_bundle()
    hits = [bundle.hits[1], bundle.hits[2]]
    corpus, _ctx = materialize_synthesis_bundle(
        SynthesisBundle(hits=hits, edges=[], mode="divergent", pairs=[]),
        _tok,
    )
    texts = [h.text for h in hits]
    preferred, ranked = score_starts_by_sentence_match(
        corpus, texts, "hospital treatment", _tok
    )
    for doc, pos in preferred:
        spans = sentence_token_spans(texts[doc], _tok)
        assert any(start == pos for start, _end in spans)


def test_preferred_includes_both_docs_in_first_group():
    bundle = _incident_bundle()
    _corpus, ctx = materialize_synthesis_bundle(
        bundle,
        _tok,
        query="downtown incident hospital treatment",
    )
    first_group = ctx.required_doc_groups[0]
    preferred_docs = {d for d, _p in ctx.preferred_starts}
    assert preferred_docs == first_group


def test_convergent_preferred_from_top_doc_only():
    port = _hit(
        "port_harbor:0",
        "Harbor cargo throughput rose 12 percent in Q2. Container volume reached 2.1 million TEU.",
        "HarborTimes",
    )
    election = _hit(
        "election_harbor:0",
        "The harbor district mayor race tightened ahead of Tuesday's vote.",
        "ElectionWire",
    )
    bundle = SynthesisBundle(hits=[election, port], edges=[], mode="convergent", pairs=[])
    corpus, ctx = materialize_synthesis_bundle(
        bundle,
        _tok,
        query="port cargo throughput container volume",
    )
    assert ctx.doc_index_to_hit[0].chunk_id == "port_harbor:0"
    preferred_docs = {d for d, _p in ctx.preferred_starts}
    assert preferred_docs == {0}
    assert ctx.top_doc_idx == 0


def test_ambiguous_shared_token_resolves_to_steered_doc():
    corpus = Corpus.from_texts(
        {
            "A": "Emergency services confirmed 142 people received treatment at area hospitals.",
            "B": "Emergency services confirmed 98 people received treatment at area hospitals.",
        },
        tokenize=_tok,
    )
    start_a = next(p for d, p in corpus.sentence_starts if d == 0)
    start_b = next(p for d, p in corpus.sentence_starts if d == 1)
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=999,
        delim_id=998,
        min_span_len=3,
        reject_ambiguous_spans=True,
        allow_shared_prefix=False,
        required_doc_groups=[frozenset({0, 1})],
        divergence_starts={0: {start_a}, 1: {start_b}},
        require_divergence_after_first=True,
    )
    flat = np.zeros(1000, dtype=np.float32)
    allowed = {
        i for i in range(1000)
        if c(np.array([], dtype=np.intc), flat.copy())[i] > NEG_INF / 2
    }
    assert allowed
    assert not c.cursors or c.mode == "BOUNDARY"
