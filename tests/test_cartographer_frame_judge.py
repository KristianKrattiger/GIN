"""LLM frame-judge — reframed relation detector, and its measured collapse.

§6 next-signal probe. The mapping (DIVERGENT->contradicts, AGREE->corroborates,
UNRELATED->unrelated) and pipeline are correct — an oracle judge scores perfectly
— but the real Mistral-7B zero-shot judge collapses to a constant answer driven by
prompt bias (always DIVERGENT on the shipped prompt; always SAME on a stance-axis
variant), so it provides no discrimination. Deterministic via injected judges.
"""
import pytest

from gin.cartographer import default_chunks, default_gold_pairs, evaluate
from gin.cartographer.frame_judge import (
    LlmFrameJudge,
    _parse_label,
    format_frame_judge_prompt,
)
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
    assert (metrics.tp, metrics.fp, metrics.fn) == (7, 26, 0)  # 7 real + 10 corr + 16 unrel spurious


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


def test_parse_label_single_keyword_and_conservative_default():
    assert _parse_label(" DIVERGENT") == "DIVERGENT"
    assert _parse_label("The answer is AGREE.") == "AGREE"
    assert _parse_label("hmm, unclear") == "UNRELATED"  # conservative default


def test_parse_label_final_line_wins_over_reasoning_mentions():
    # Reasoning text routinely names labels it then rejects; the FINAL line
    # is the verdict. First-keyword parsing would return DIVERGENT here.
    text = (
        "1. Both discuss wildfire impacts.\n"
        "2. A reports acreage; B reports smoke risk.\n"
        "3. One could call these DIVERGENT, but the stances do not compete.\n"
        "FINAL: AGREE"
    )
    assert _parse_label(text) == "AGREE"


def test_parse_label_final_tolerates_markup_and_case():
    assert _parse_label("FINAL: **UNRELATED**") == "UNRELATED"
    assert _parse_label("final: <AGREE>") == "AGREE"


def test_parse_label_without_final_uses_last_keyword():
    # A conclusion sits at the end of reasoning text, not the start.
    assert _parse_label("They are not DIVERGENT; they simply AGREE.") == "AGREE"


def test_prompt_requests_reasoning_then_final_line():
    p = format_frame_judge_prompt("text-a", "text-b")
    assert "text-a" in p and "text-b" in p
    assert "FINAL:" in p
    assert "Reason" in p
    assert "[INST]" not in p
    wrapped = format_frame_judge_prompt("text-a", "text-b", llama_inst=True)
    assert wrapped.startswith("[INST]") and wrapped.rstrip().endswith("[/INST]")


class _ReasoningLlm:
    """Fake llama.cpp handle returning a canned reasoning completion."""

    def __init__(self, text: str):
        self.text = text
        self.calls: list = []

    def create_completion(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"choices": [{"text": self.text}]}


def test_llm_judge_reads_final_from_reasoning_and_keeps_text():
    llm = _ReasoningLlm(
        "1. Same issue.\n2. Stances could look DIVERGENT at first.\nFINAL: AGREE"
    )
    judge = LlmFrameJudge(llm=llm)
    assert judge.label("a", "b") == "AGREE"
    assert "FINAL: AGREE" in judge.last_completion_text
    # Reasoning needs budget; the old one-word budget (4 tokens) starved it.
    assert llm.calls[0][1]["max_tokens"] == 256


def test_llm_frame_judge_instance_is_callable():
    judge = LlmFrameJudge(judge=lambda a, b: "AGREE")
    assert judge("x", "y") == "AGREE"
