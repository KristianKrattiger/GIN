"""End-to-end synthesis constraint tests with mock incident corpus."""
import numpy as np
from sear.corpus import Corpus
from sear.processor import ExtractiveCopyConstraint, NEG_INF

_VOCAB = {w: i for i, w in enumerate(
    ["<pad>", "<eos>", "|", "Officials", "responded", "to", "a", "downtown",
     "incident", "Tuesday", "evening.", "Emergency", "services", "confirmed",
     "142", "people", "received", "treatment", "at", "area", "hospitals.",
     "Police", "said", "23", "arrests", "were", "made", "before", "midnight.",
     "98", "11", "[1]", "[2]", "[3]", "but", "however", "The", "north", "line",
     "extension", "opened", "to", "passengers", "on", "March", "14.",
     "Daily", "ridership", "averaged", "18400", "boardings.",
     "mayor", "scheduled", "briefing", "Wednesday", "morning."])}
_V = len(_VOCAB)


def _tok(b: bytes) -> list[int]:
    words = b.decode().split()
    return [_VOCAB[w] for w in words]


def _make_incident_corpus() -> Corpus:
    texts = {
        "central": (
            "Officials responded to a downtown incident Tuesday evening. "
            "Emergency services confirmed 142 people received treatment at area hospitals. "
            "Police said 23 arrests were made before midnight. "
            "The mayor scheduled briefing Wednesday morning."
        ),
        "metro": (
            "Officials responded to a downtown incident Tuesday evening. "
            "Emergency services confirmed 98 people received treatment at area hospitals. "
            "Police said 11 arrests were made before midnight."
        ),
        "transit": (
            "The north line extension opened to passengers on March 14. "
            "Daily ridership averaged 18400 boardings."
        ),
    }
    return Corpus.from_texts(texts, tokenize=_tok)


def _allowed_tokens(c: ExtractiveCopyConstraint, generated: list[int]) -> set[int]:
    flat = np.zeros(_V, dtype=np.float32)
    scores = c(np.array(generated, dtype=np.intc), flat.copy())
    return {i for i in range(_V) if scores[i] > NEG_INF / 2}


def test_divergent_focus_excludes_transit_doc_index():
    corpus = _make_incident_corpus()
    transit_idx = corpus.doc_names.index("transit")
    incident_indices = frozenset({
        corpus.doc_names.index("central"),
        corpus.doc_names.index("metro"),
    })

    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        focus_doc_indices=incident_indices,
        reject_ambiguous_spans=True,
        span_must_start_at_sentence=True,
        close_on_doc_divergence=True,
    )

    flat = np.zeros(_V, dtype=np.float32)
    allowed = {
        i for i in range(_V)
        if c(np.array([], dtype=np.intc), flat.copy())[i] > NEG_INF / 2
    }
    transit_exclusive = {
        _VOCAB["north"], _VOCAB["extension"], _VOCAB["Daily"], _VOCAB["ridership"],
    }
    assert not (allowed & transit_exclusive)


def test_span_must_start_at_sentence_blocks_mid_sentence_token():
    corpus = _make_incident_corpus()
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=3,
        span_must_start_at_sentence=True,
    )
    flat = np.zeros(_V, dtype=np.float32)
    allowed = {
        i for i in range(_V)
        if c(np.array([], dtype=np.intc), flat.copy())[i] > NEG_INF / 2
    }
    assert _VOCAB["142"] not in allowed
    assert _VOCAB["Officials"] in allowed


def test_preferred_starts_rank_emergency_above_mayor():
    corpus = _make_incident_corpus()
    central_idx = corpus.doc_names.index("central")
    emergency_pos = next(
        p for d, p in corpus.sentence_starts if d == central_idx and corpus.docs[d][p] == _VOCAB["Emergency"]
    )
    mayor_pos = next(
        p for d, p in corpus.sentence_starts if d == central_idx and corpus.docs[d][p] == _VOCAB["The"]
    )
    preferred = {(central_idx, emergency_pos)}
    forbidden = {(central_idx, mayor_pos)}

    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        preferred_starts=preferred,
        forbidden_starts=forbidden,
        span_must_start_at_sentence=True,
    )
    allowed = _allowed_tokens(c, [])
    assert _VOCAB["Emergency"] in allowed
    assert _VOCAB["The"] not in allowed


def test_cite_gate_no_eos_when_cites_available():
    corpus = _make_incident_corpus()
    cite_ids = {_VOCAB["[1]"]: 0, _VOCAB["[2]"]: 1}
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        cite_ids=cite_ids,
        cite_sequences_by_doc={0: [_VOCAB["[1]"]], 1: [_VOCAB["[2]"]]},
        require_cite_after_extract=True,
        preferred_starts={
            (d, p) for d, p in corpus.sentence_starts
            if corpus.docs[d][p] == _VOCAB["Emergency"]
        },
        span_must_start_at_sentence=True,
    )
    seq = [
        _VOCAB["Emergency"], _VOCAB["services"], _VOCAB["confirmed"], _VOCAB["142"],
        _VOCAB["people"], _VOCAB["received"], _VOCAB["treatment"], _VOCAB["at"],
        _VOCAB["area"], _VOCAB["hospitals."], _VOCAB["|"],
    ]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    allowed = _allowed_tokens(c, seq)
    assert _VOCAB["<eos>"] not in allowed
    assert _VOCAB["[1]"] in allowed or _VOCAB["[2]"] in allowed


def test_block_eos_prevents_stop_before_two_docs_quoted():
    corpus = _make_incident_corpus()
    central_idx = corpus.doc_names.index("central")
    metro_idx = corpus.doc_names.index("metro")
    emergency_central = next(
        p for d, p in corpus.sentence_starts if d == central_idx and corpus.docs[d][p] == _VOCAB["Emergency"]
    )
    emergency_metro = next(
        p for d, p in corpus.sentence_starts if d == metro_idx and corpus.docs[d][p] == _VOCAB["Emergency"]
    )
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        required_doc_groups=[frozenset({central_idx, metro_idx})],
        block_eos_until_groups_satisfied=True,
        close_on_doc_divergence=True,
        reject_ambiguous_spans=True,
        preferred_starts={(central_idx, emergency_central)},
        span_must_start_at_sentence=True,
        focus_doc_indices=frozenset({central_idx, metro_idx}),
    )
    seq = [
        _VOCAB["Emergency"], _VOCAB["services"], _VOCAB["confirmed"], _VOCAB["142"],
        _VOCAB["people"], _VOCAB["received"], _VOCAB["treatment"], _VOCAB["at"],
        _VOCAB["area"], _VOCAB["hospitals."], _VOCAB["|"],
    ]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())
    allowed = _allowed_tokens(c, seq)
    assert _VOCAB["<eos>"] not in allowed
    assert central_idx in c.quoted_docs()
    assert metro_idx not in c.quoted_docs()
    assert _VOCAB["Emergency"] in allowed or _VOCAB["98"] in allowed


def test_span_must_close_at_sentence_end_blocks_early_delim():
    corpus = _make_incident_corpus()
    central_idx = corpus.doc_names.index("central")
    start = next(
        p for d, p in corpus.sentence_starts
        if d == central_idx and corpus.docs[d][p] == _VOCAB["Emergency"]
    )
    end = corpus.sentence_end_by_start[(central_idx, start)]
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        span_must_close_at_sentence_end=True,
        span_must_start_at_sentence=True,
    )
    partial = corpus.docs[central_idx][start : start + 5]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(partial)):
        c(np.array(partial[:i], dtype=np.intc), flat.copy())
    c(np.array(partial, dtype=np.intc), flat.copy())
    allowed = _allowed_tokens(c, partial)
    assert _VOCAB["|"] not in allowed

    full = corpus.docs[central_idx][start : end + 1]
    c2 = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        span_must_close_at_sentence_end=True,
        span_must_start_at_sentence=True,
    )
    for i in range(len(full)):
        c2(np.array(full[:i], dtype=np.intc), flat.copy())
    c2(np.array(full, dtype=np.intc), flat.copy())
    allowed_full = _allowed_tokens(c2, full)
    assert _VOCAB["|"] in allowed_full


def test_divergence_sentence_end_requires_full_conflict_line():
    corpus = _make_incident_corpus()
    central_idx = corpus.doc_names.index("central")
    start = next(
        p for d, p in corpus.sentence_starts
        if d == central_idx and corpus.docs[d][p] == _VOCAB["Emergency"]
    )
    end = corpus.sentence_end_by_start[(central_idx, start)]
    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        span_must_close_at_sentence_end=True,
        divergence_sentence_ends={central_idx: {start: end}},
    )
    partial = corpus.docs[central_idx][start : start + 5]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(partial)):
        c(np.array(partial[:i], dtype=np.intc), flat.copy())
    c(np.array(partial, dtype=np.intc), flat.copy())
    assert _VOCAB["|"] not in _allowed_tokens(c, partial)

    full = corpus.docs[central_idx][start : end + 1]
    c2 = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=5,
        span_must_close_at_sentence_end=True,
        divergence_sentence_ends={central_idx: {start: end}},
    )
    for i in range(len(full)):
        c2(np.array(full[:i], dtype=np.intc), flat.copy())
    c2(np.array(full, dtype=np.intc), flat.copy())
    c2(np.array(full + [_VOCAB["|"]], dtype=np.intc), flat.copy())
    extracts = [s for s in c2.segments if s.kind == "extract"]
    assert len(extracts) == 1
    assert _VOCAB["142"] in extracts[0].token_ids


def test_corpus_sentence_ends_align_with_starts():
    corpus = _make_incident_corpus()
    central_idx = corpus.doc_names.index("central")
    for doc, start in corpus.sentence_starts:
        if doc != central_idx:
            continue
        end = corpus.sentence_end_by_start[(doc, start)]
        assert (doc, end) in corpus.sentence_ends
        assert end >= start


def test_steered_convergent_focus_excludes_incident_docs():
    """Convergent query steering on transit query must not start incident lede."""
    corpus = _make_incident_corpus()
    transit_idx = corpus.doc_names.index("transit")
    incident_indices = {
        corpus.doc_names.index("central"),
        corpus.doc_names.index("metro"),
    }
    north_pos = next(
        p for d, p in corpus.sentence_starts
        if d == transit_idx and corpus.docs[d][p] == _VOCAB["The"]
    )
    preferred = {(transit_idx, north_pos)}

    c = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=3,
        focus_doc_indices=frozenset({transit_idx}),
        allow_shared_prefix=False,
        preferred_starts=preferred,
        stop_after_first_extract=True,
    )
    flat = np.zeros(_V, dtype=np.float32)
    allowed = {
        i for i in range(_V)
        if c(np.array([], dtype=np.intc), flat.copy())[i] > NEG_INF / 2
    }
    incident_exclusive = {_VOCAB["Officials"], _VOCAB["Emergency"], _VOCAB["142"]}
    assert not (allowed & incident_exclusive)
    assert _VOCAB["The"] in allowed or _VOCAB["north"] in allowed
    for idx in incident_indices:
        assert idx != transit_idx

