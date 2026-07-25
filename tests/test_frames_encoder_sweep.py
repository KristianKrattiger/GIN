"""Frozen-encoder sweep: does it measure framing, or just restate node4?

Model-free throughout -- ChunkEncoder takes an encode_fn, so the whole sweep is
exercised without downloading weights.
"""
import numpy as np
import pytest

from gin.frames.dataset import FrameExample
from gin.frames.encoder import ChunkEncoder
from gin.frames.encoder_sweep import (
    FRAMING_ORIGIN,
    NODE4_ORIGIN,
    ORIGIN_SAMPLE_FLOOR,
    EncoderResult,
    OriginRecall,
    _loo_predictions,
    divergent_origin,
    format_result,
    recall_by_origin,
    sweep_encoder,
    verdict,
)
from gin.frames.labels import FrameClass
from gin.frames.probe import divergent_vs_rest, run_probe


def test_node4_pairs_are_their_own_origin():
    assert divergent_origin("n4_doc_001:0", "n4_doc_002:1") == NODE4_ORIGIN


def test_housing_and_institutional_pairs_are_framing():
    assert divergent_origin("hf_af_staff:0", "hf_af_tenants:0") == FRAMING_ORIGIN
    assert divergent_origin("inst_em:0", "grass_em:0") == FRAMING_ORIGIN


def test_mixed_pair_is_framing_not_node4():
    # A node4 chunk paired against anything else is not the node4 phenomenon;
    # requiring BOTH endpoints keeps the easy class from absorbing hard rows.
    assert divergent_origin("n4_doc_001:0", "hf_af_staff:0") == FRAMING_ORIGIN


def _example(src, dst, label):
    return FrameExample(src, dst, f"text-{src}", f"text-{dst}", label)


def test_recall_by_origin_splits_and_lists_misses():
    examples = [
        _example("n4_doc_001:0", "n4_doc_002:0", FrameClass.DIVERGENT),
        _example("n4_doc_003:0", "n4_doc_004:0", FrameClass.DIVERGENT),
        _example("hf_af_staff:0", "hf_af_tenants:0", FrameClass.DIVERGENT),
        _example("a:0", "b:0", FrameClass.AGREE),
    ]
    target = np.array([1, 1, 1, 0])
    predictions = np.array([1, 1, 0, 0])  # both node4 recovered, framing missed

    rows = {r.origin: r for r in recall_by_origin(examples, target, predictions)}
    assert rows[NODE4_ORIGIN].n == 2 and rows[NODE4_ORIGIN].n_recovered == 2
    assert rows[FRAMING_ORIGIN].n == 1 and rows[FRAMING_ORIGIN].n_recovered == 0
    assert rows[FRAMING_ORIGIN].chunk_pairs == ["hf_af_staff:0 <-> hf_af_tenants:0"]


def test_small_origin_is_not_decisive():
    assert OriginRecall(FRAMING_ORIGIN, ORIGIN_SAMPLE_FLOOR - 1, 0).decisive is False
    assert OriginRecall(NODE4_ORIGIN, ORIGIN_SAMPLE_FLOOR, 0).decisive is True


def test_aggregate_matches_the_published_probe():
    # Load-bearing: the origin split must describe the same fitted model the
    # headline number describes, or the two cannot be compared.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6))
    y = np.where(np.arange(30) % 3 == 0, "DIVERGENT", "AGREE")
    X[y == "DIVERGENT"] += 4.0

    target = divergent_vs_rest(y)
    predictions = _loo_predictions(X, target, seed=0)
    from sklearn.metrics import balanced_accuracy_score

    assert balanced_accuracy_score(target, predictions) == pytest.approx(
        run_probe(X, y, seed=0).balanced_accuracy
    )


def _fake_encoder(dim=8):
    """Deterministic per-text vectors; node4 texts are made trivially separable."""

    def encode(text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=dim)
        if "n4_doc" in text:
            vec += 5.0
        return vec / np.linalg.norm(vec)

    return ChunkEncoder("fake/encoder", encode_fn=encode)


def test_sweep_runs_model_free_and_reports_both_origins():
    examples = [
        _example(f"n4_doc_{i:03d}:0", f"n4_doc_{i+1:03d}:0", FrameClass.DIVERGENT)
        for i in range(6)
    ] + [
        _example("hf_af_staff:0", "hf_af_tenants:0", FrameClass.DIVERGENT),
        _example("hf_kc_inspection:0", "hf_kc_tenants:0", FrameClass.DIVERGENT),
    ] + [
        _example(f"x_{i}:0", f"y_{i}:0", FrameClass.AGREE) for i in range(8)
    ]

    result = sweep_encoder(examples, _fake_encoder(), score_bar=False)
    origins = {r.origin for r in result.by_origin}
    assert origins == {NODE4_ORIGIN, FRAMING_ORIGIN}
    assert result.bar is None
    assert 0.0 <= result.aggregate_balanced_accuracy <= 1.0


def test_format_flags_the_aggregate_as_not_the_answer():
    result = EncoderResult(
        "fake/encoder",
        0.939,
        [OriginRecall(NODE4_ORIGIN, 22, 22), OriginRecall(FRAMING_ORIGIN, 2, 0)],
        {
            "issue_frame_recall": 0.0,
            "class_c_discrimination": 1.0,
            "unrelated_discrimination": 1.0,
            "direction_flip_count": 0,
        },
    )
    text = format_result(result)
    assert "not the answer" in text
    assert "screen only" in text  # the 2-row framing bucket is labelled in place
    assert "clean framing measurement" in text


def test_verdict_is_decided_by_the_bar_not_the_aggregate():
    high_aggregate_no_framing = EncoderResult(
        "a", 0.99, [], {"issue_frame_recall": 0.0}
    )
    low_aggregate_some_framing = EncoderResult(
        "b", 0.55, [], {"issue_frame_recall": 0.25}
    )
    assert verdict([high_aggregate_no_framing]) == "framing_not_recoverable_frozen"
    assert verdict([high_aggregate_no_framing, low_aggregate_some_framing]) == (
        "framing_recoverable:b"
    )


def test_verdict_ignores_failed_encoders():
    failed = EncoderResult("broken", float("nan"), [], None, error="OSError: no weights")
    assert verdict([failed]) == "no_measurement"
