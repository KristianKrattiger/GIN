"""
Tests for sear.processor — cursor tracking, masking, connectives, cites, divergence.
"""
import numpy as np
from sear.corpus import Corpus
from sear.processor import ExtractiveCopyConstraint, NEG_INF, BOUNDARY, IN_SPAN, IN_CONNECTIVE

_VOCAB = {w: i for i, w in enumerate(
    ["<pad>", "<eos>", "|", "the", "fox", "ran", "fast", "dog",
     "slept", "all", "day", "and", "but", "[1]", "[2]"])}
_INV = {i: w for w, i in _VOCAB.items()}
_V = len(_VOCAB)


def _tok(b: bytes) -> list[int]:
    return [_VOCAB[w] for w in b.decode().split()]


def _detok(ids: list[int]) -> str:
    return " ".join(_INV[i] for i in ids)


def _make_corpus() -> Corpus:
    return Corpus.from_texts(
        {"A": "the fox ran fast", "B": "the fox ran and the dog slept all day"},
        tokenize=_tok,
    )


def _make_constraint(
    corpus: Corpus,
    *,
    connective_starts: frozenset[int] | None = None,
    connective_phrases: dict[int, list[int]] | None = None,
    cite_ids: dict[int, int] | None = None,
    close_on_doc_divergence: bool = False,
    required_doc_groups: list[frozenset[int]] | None = None,
) -> ExtractiveCopyConstraint:
    starts = connective_starts if connective_starts is not None else frozenset({_VOCAB["but"]})
    phrases = connective_phrases if connective_phrases is not None else {_VOCAB["but"]: [_VOCAB["but"]]}
    cite = cite_ids or {}
    cite_sequences = {doc: [tok] for tok, doc in cite.items()}
    return ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=3,
        connective_starts=starts,
        connective_phrases=phrases,
        cite_ids=cite,
        cite_sequences_by_doc=cite_sequences,
        close_on_doc_divergence=close_on_doc_divergence,
        required_doc_groups=required_doc_groups,
    )


def _step(c: ExtractiveCopyConstraint, generated: list[int]) -> set[str]:
    flat = np.zeros(_V, dtype=np.float32)
    scores = c(np.array(generated, dtype=np.intc), flat.copy())
    return {_INV[i] for i in range(_V) if scores[i] > NEG_INF / 2}


def test_boundary_initial_allowed():
    corpus = _make_corpus()
    c = _make_constraint(corpus)
    allowed = _step(c, [])
    assert "the" in allowed and "fox" in allowed
    assert "<eos>" in allowed
    assert "|" not in allowed
    assert "but" not in allowed


def test_cursor_fanout_shared_prefix():
    corpus = _make_corpus()
    c = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"]]

    allowed1 = _step(c, seq[:1])
    assert allowed1 == {"fox", "dog"}, allowed1

    allowed2 = _step(c, seq[:2])
    assert allowed2 == {"ran"}, allowed2

    allowed3 = _step(c, seq[:3])
    assert {"fast", "and", "|", "<eos>"} == allowed3, allowed3


def test_multi_token_attribution():
    corpus = _make_corpus()
    c2 = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"]]
    flat = np.zeros(_V, dtype=np.float32)

    for i in range(3):
        c2(np.array(seq[:i], dtype=np.intc), flat.copy())
    c2(np.array(seq, dtype=np.intc), flat.copy())
    c2(np.array(seq + [_VOCAB["|"]], dtype=np.intc), flat.copy())

    segs = c2.finalize()
    ext = [s for s in segs if s.kind == "extract"][0]
    assert _detok(ext.token_ids) == "the fox ran"
    assert len(ext.sources) == 2
    assert {d for d, _, _ in ext.sources} == {0, 1}


def test_cursor_pruning_on_divergence():
    corpus = _make_corpus()
    c3 = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"]]
    flat = np.zeros(_V, dtype=np.float32)
    full = seq + [_VOCAB["fast"]]

    for i in range(len(full)):
        c3(np.array(full[:i], dtype=np.intc), flat.copy())
    c3(np.array(full, dtype=np.intc), flat.copy())
    c3(np.array(full + [_VOCAB["<eos>"]], dtype=np.intc), flat.copy())

    ext3 = [s for s in c3.finalize() if s.kind == "extract"][0]
    assert len(ext3.sources) == 1 and ext3.sources[0][0] == 0


def test_connective_allowed_after_closed_span():
    corpus = _make_corpus()
    c = _make_constraint(
        corpus,
        connective_starts=frozenset({_VOCAB["but"], _VOCAB["and"]}),
        connective_phrases={
            _VOCAB["but"]: [_VOCAB["but"]],
            _VOCAB["and"]: [_VOCAB["and"]],
        },
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "but" in allowed
    assert "and" in allowed


def test_connective_between_spans_not_extract():
    corpus = _make_corpus()
    c = _make_constraint(
        corpus,
        connective_starts=frozenset({_VOCAB["but"]}),
        connective_phrases={_VOCAB["but"]: [_VOCAB["but"]]},
    )
    seq = [
        _VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"],
        _VOCAB["but"],
    ]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    kinds = [s.kind for s in c.segments]
    assert kinds.count("connective") >= 2
    assert not any(s.kind == "extract" and _detok(s.token_ids) == "but" for s in c.segments)
    assert c.mode == BOUNDARY


def test_connective_not_in_span_mask():
    corpus = _make_corpus()
    c = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "but" not in allowed
    assert c.mode == IN_SPAN


def test_cite_after_closed_span():
    corpus = _make_corpus()
    cite_ids = {_VOCAB["[1]"]: 0, _VOCAB["[2]"]: 1}
    c = _make_constraint(corpus, cite_ids=cite_ids)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "[1]" in allowed or "[2]" in allowed

    c(np.array(seq + [_VOCAB["[1]"]], dtype=np.intc), flat.copy())
    cite_segs = [s for s in c.segments if s.kind == "cite"]
    assert len(cite_segs) == 1
    assert cite_segs[0].sources[0][0] == 0


def test_cite_not_allowed_inside_span():
    corpus = _make_corpus()
    cite_ids = {_VOCAB["[1]"]: 0}
    c = _make_constraint(corpus, cite_ids=cite_ids)
    seq = [_VOCAB["the"], _VOCAB["fox"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "[1]" not in allowed


def test_close_on_doc_divergence():
    corpus = _make_corpus()
    c = _make_constraint(corpus, close_on_doc_divergence=True)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["fast"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    assert c.mode == BOUNDARY
    extracts = [s for s in c.segments if s.kind == "extract"]
    assert len(extracts) == 1
    assert len(extracts[0].sources) == 1
    assert extracts[0].sources[0][0] == 0


def test_required_doc_groups_tracking():
    corpus = _make_corpus()
    c = _make_constraint(
        corpus,
        required_doc_groups=[frozenset({0, 1})],
    )
    assert not c.groups_satisfied()
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    assert 0 in c.quoted_docs() and 1 in c.quoted_docs()
    assert c.groups_satisfied()


def test_render_includes_cite_suffix():
    corpus = _make_corpus()
    cite_ids = {_VOCAB["[1]"]: 0}
    c = _make_constraint(corpus, cite_ids=cite_ids)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"], _VOCAB["[1]"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    rendered = c.render(_detok)
    assert "[1]" in rendered


def test_reject_ambiguous_spans_after_first_extract():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        reject_ambiguous_spans=True,
        allow_shared_prefix=True,
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"], _VOCAB["the"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    assert c.allow_shared_prefix is False
    assert len(c.cursors) == 1
    assert c.cursors[0][0] == 1


def test_require_cite_gate_blocks_new_extract():
    cite_ids = {_VOCAB["[1]"]: 0, _VOCAB["[2]"]: 1}
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        cite_ids=cite_ids,
        cite_sequences_by_doc={0: [_VOCAB["[1]"]], 1: [_VOCAB["[2]"]]},
        require_cite_after_extract=True,
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "the" not in allowed
    assert "[1]" in allowed or "[2]" in allowed
    assert "<eos>" not in allowed


def test_cite_gate_allows_eos_only_when_no_cite_tokens():
    cite_ids = {_VOCAB["[1]"]: 0}
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        cite_ids=cite_ids,
        cite_sequences_by_doc={0: [_VOCAB["[1]"]]},
        require_cite_after_extract=True,
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "<eos>" not in allowed
    assert "[1]" in allowed


def test_connectives_blocked_until_groups_satisfied():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        connective_starts=frozenset({_VOCAB["but"]}),
        connective_phrases={_VOCAB["but"]: [_VOCAB["but"]]},
        required_doc_groups=[frozenset({0, 1})],
        block_eos_until_groups_satisfied=True,
        close_on_doc_divergence=True,
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["fast"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "but" not in allowed
    assert "dog" in allowed or "the" in allowed


def test_multi_token_cite_shared_prefix_disambiguates():
    """When [1] and [2] share a leading token, continuations disambiguate."""
    corpus = _make_corpus()
    open_bracket, one, two, close0, close1 = 20, 21, 22, 23, 24
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        cite_ids={close0: 0, close1: 1},
        cite_sequences_by_doc={
            0: [open_bracket, one, close0],
            1: [open_bracket, two, close1],
        },
        require_cite_after_extract=True,
    )
    seq = [
        _VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"],
        open_bracket, one, close0,
    ]
    flat = np.zeros(32, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    cite_segs = [s for s in c.segments if s.kind == "cite"]
    assert len(cite_segs) == 1
    assert cite_segs[0].sources[0][0] == 0
    assert cite_segs[0].token_ids == [open_bracket, one, close0]


def test_block_eos_until_groups_satisfied():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        required_doc_groups=[frozenset({0, 1})],
        block_eos_until_groups_satisfied=True,
        close_on_doc_divergence=True,
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["fast"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    allowed = _step(c, seq)
    assert "<eos>" not in allowed
    assert not c.groups_satisfied()


def test_stop_when_groups_satisfied():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        required_doc_groups=[frozenset({0, 1})],
        stop_when_groups_satisfied=True,
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    assert c.groups_satisfied()
    allowed = _step(c, seq)
    assert allowed == {"<eos>"}


def test_doc_steering_prefers_unquoted_group_member():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus, 0, _VOCAB["<eos>"], _VOCAB["|"], min_span_len=3,
        required_doc_groups=[frozenset({0, 1})],
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["fast"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    assert 0 in c.quoted_docs()
    assert 1 not in c.quoted_docs()
    allowed = _step(c, seq)
    assert "dog" in allowed
    assert 1 not in c.quoted_docs()


def test_sentence_starts_indexed():
    corpus = Corpus.from_texts(
        {"A": "the fox ran fast"},
        tokenize=_tok,
    )
    assert (0, 0) in corpus.sentence_starts


def _emit_extract_span(c: ExtractiveCopyConstraint, seq: list[int]) -> None:
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())


def test_guidance_steered_when_preferred_start():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=3,
        preferred_starts={(0, 0)},
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    _emit_extract_span(c, seq)
    ext = [s for s in c.finalize() if s.kind == "extract"][0]
    assert ext.guidance == "steered"


def test_guidance_empty_without_steering():
    corpus = _make_corpus()
    c = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    _emit_extract_span(c, seq)
    ext = [s for s in c.finalize() if s.kind == "extract"][0]
    assert ext.guidance == ""


def test_guidance_divergence_steered():
    corpus = _make_corpus()
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=3,
        divergence_starts={0: {0}},
    )
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    _emit_extract_span(c, seq)
    ext = [s for s in c.finalize() if s.kind == "extract"][0]
    assert ext.guidance == "divergence-steered"
