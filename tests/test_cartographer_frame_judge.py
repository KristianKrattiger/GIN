"""LLM frame-judge — reframed relation detector, and its measured collapse.

§6 next-signal probe. The mapping (DIVERGENT->contradicts, AGREE->corroborates,
UNRELATED->unrelated) and pipeline are correct — an oracle judge scores perfectly
— but the real Mistral-7B zero-shot judge collapses to a constant answer driven by
prompt bias (always DIVERGENT on the shipped prompt; always SAME on a stance-axis
variant), so it provides no discrimination. Deterministic via injected judges.
"""
import pytest

from gin.cartographer import default_chunks, default_gold_pairs, evaluate
from gin.cartographer.frame_judge import LlmFrameJudge, _parse_label
from gin.cartographer.models import Relation


def _pairs():
    by_id = {c.chunk_id: c for c in default_chunks()}
    return [(by_id[g.src_chunk_id], by_id[g.dst_chunk_id]) for g in default_gold_pairs()]


def test_requires_a_judge_or_llm():
    with pytest.raises(ValueError):
        LlmFrameJudge()


def test_label_mapping_covers_the_three_verdicts():
    judge = LlmFrameJudge(judge=lambda a, b: "AGREE")
    assert judge.type_relation("x", "y")[0] == Relation.CORROBORATES
    judge = LlmFrameJudge(judge=lambda a, b: "DIVERGENT")
    assert judge.type_relation("x", "y")[0] == Relation.CONTRADICTS
    judge = LlmFrameJudge(judge=lambda a, b: "UNRELATED")
    assert judge.type_relation("x", "y")[0] == Relation.UNRELATED


def test_always_divergent_is_the_degenerate_baseline():
    """An always-DIVERGENT judge gets trivial recall and zero discrimination.

    This is the failure mode the class-C co-metric exists to expose. (The real
    Mistral is near this but not identical on the expanded set — it correctly
    says AGREE on one corroborating pair, class_c 0.333 — recorded in §6.)
    """
    judge = LlmFrameJudge(judge=lambda a, b: "DIVERGENT")
    metrics = evaluate(judge.propose_over(_pairs()), default_gold_pairs())
    assert metrics.contradicts_recall == 1.0        # trivially — everything is divergent
    assert metrics.class_c_discrimination == 0.0    # every corroborating pair mislabeled
    assert (metrics.tp, metrics.fp, metrics.fn) == (7, 6, 0)  # 7 real + 6 spurious


def test_an_oracle_judge_would_score_perfectly():
    """Proves the harness and mapping reward a correct judge — the signal is the
    open problem, not the code."""
    gold_by_key = {
        frozenset({g.src_chunk_id, g.dst_chunk_id}): g.relation for g in default_gold_pairs()
    }

    def oracle(a_text: str, b_text: str) -> str:
        return "AGREE"  # placeholder; overridden per-pair below via chunk ids

    # Build an oracle keyed on the actual pair, mapping gold -> judge label.
    by_text = {c.text: c.chunk_id for c in default_chunks()}
    label_for = {
        Relation.CONTRADICTS: "DIVERGENT",
        Relation.CORROBORATES: "AGREE",
        Relation.UNRELATED: "UNRELATED",
    }

    def keyed_oracle(a_text: str, b_text: str) -> str:
        key = frozenset({by_text[a_text], by_text[b_text]})
        return label_for[gold_by_key[key]]

    judge = LlmFrameJudge(judge=keyed_oracle)
    metrics = evaluate(judge.propose_over(_pairs()), default_gold_pairs())
    assert metrics.class_c_discrimination == 1.0
    assert metrics.contradicts_recall == 1.0
    assert metrics.contradicts_precision == 1.0


def test_parse_label_extracts_first_keyword():
    assert _parse_label(" DIVERGENT") == "DIVERGENT"
    assert _parse_label("The answer is AGREE.") == "AGREE"
    assert _parse_label("hmm, unclear") == "UNRELATED"  # conservative default
