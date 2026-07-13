"""Escalation tier: model-agnostic LLM judge for anchor-less, topically-close pairs."""
import sys

import pytest

from gin.cartographer.combined import CombinedRelationProposer, Thresholds
from gin.cartographer.escalation import (
    AnthropicFrameJudge,
    escalate_proposals,
    escalation_candidates,
    make_local_frame_judge,
    resolve_escalation_judge,
)
from gin.cartographer.models import LabeledChunk, Relation

T = Thresholds(gate_floor=0.14, corroborate_ceiling=0.485, contra_threshold=0.686)

_CHUNKS = {
    "inst:0": LabeledChunk("inst:0", "Agency reported acreage below the ten-year average."),
    "grass:0": LabeledChunk("grass:0", "Low-income residents face heightened smoke risk."),
    "story_a:0": LabeledChunk("story_a:0", "The Kestrel Court inspection cited mold."),
    "story_b:0": LabeledChunk("story_b:0", "Kestrel Court tenants say mold was ignored."),
    "far:0": LabeledChunk("far:0", "Transit ridership rose this quarter."),
}

_COS = {
    frozenset({"inst:0", "grass:0"}): 0.42,   # anchor-less, topically close
    frozenset({"story_a:0", "story_b:0"}): 0.60,  # same-story (cheap path)
    frozenset({"inst:0", "far:0"}): 0.10,     # below escalation floor
    frozenset({"grass:0", "far:0"}): 0.12,
    frozenset({"story_a:0", "far:0"}): 0.05,
    frozenset({"story_b:0", "far:0"}): 0.05,
    frozenset({"inst:0", "story_a:0"}): 0.15,
    frozenset({"inst:0", "story_b:0"}): 0.15,
    frozenset({"grass:0", "story_a:0"}): 0.15,
    frozenset({"grass:0", "story_b:0"}): 0.15,
}

_STORY = {frozenset({"story_a:0", "story_b:0"})}


def _proposer() -> CombinedRelationProposer:
    ids = {c.text: cid for cid, c in _CHUNKS.items()}
    return CombinedRelationProposer(
        embed_cos=lambda a, b: _COS[frozenset({ids[a], ids[b]})],
        nli_scores=lambda a, b: (0.01, 0.001, 0.989),
        same_story=lambda a, b: frozenset({ids[a], ids[b]}) in _STORY,
        thresholds=T,
    )


def _all_pairs():
    items = list(_CHUNKS.values())
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def test_escalation_candidates_are_anchorless_and_topically_close():
    cands = escalation_candidates(_all_pairs(), _proposer(), cos_floor=0.30)
    keys = {frozenset({a.chunk_id, b.chunk_id}) for a, b in cands}
    assert keys == {frozenset({"inst:0", "grass:0"})}


def test_escalation_candidates_respect_cap():
    cands = escalation_candidates(
        _all_pairs(), _proposer(), cos_floor=0.05, max_candidates=2
    )
    assert len(cands) == 2
    # Highest-cosine candidates first, so the cap keeps the best ones.
    keys = {frozenset({a.chunk_id, b.chunk_id}) for a, b in cands}
    assert frozenset({"inst:0", "grass:0"}) in keys


def test_escalate_proposals_types_only_divergent_labels():
    pairs = [
        (_CHUNKS["inst:0"], _CHUNKS["grass:0"]),
        (_CHUNKS["inst:0"], _CHUNKS["far:0"]),
    ]
    labels = {
        frozenset({"inst:0", "grass:0"}): "DIVERGENT",
        frozenset({"inst:0", "far:0"}): "UNRELATED",
    }

    def judge(a_text: str, b_text: str) -> str:
        ids = {c.text: cid for cid, c in _CHUNKS.items()}
        return labels[frozenset({ids[a_text], ids[b_text]})]

    proposals = escalate_proposals(pairs, judge, method_suffix="test")
    assert len(proposals) == 1
    p = proposals[0]
    assert p.relation == Relation.CONTRADICTS
    assert p.method == "llm_frame_judge:escalation:test"
    assert p.confidence >= 0.5  # must clear the Bookkeeper floor


def test_anthropic_judge_parses_model_reply():
    class FakeMessages:
        def create(self, **kwargs):
            class Block:
                text = "DIVERGENT"

            class Resp:
                content = [Block()]

            self.last_kwargs = kwargs
            return Resp()

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    judge = AnthropicFrameJudge(client=FakeClient())
    assert judge("official framing text", "justice framing text") == "DIVERGENT"


def test_anthropic_judge_reasoning_budget_and_text_capture():
    class FakeMessages:
        def create(self, **kwargs):
            self.last_kwargs = kwargs

            class Block:
                text = "1. Same issue; one might say DIVERGENT.\nFINAL: AGREE"

            class Resp:
                content = [Block()]

            return Resp()

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = FakeClient()
    judge = AnthropicFrameJudge(client=client)
    assert judge("a", "b") == "AGREE"
    # Reasoning prompt needs budget; one-word budget (8) starves it.
    assert client.messages.last_kwargs["max_tokens"] == 256
    assert "FINAL: AGREE" in judge.last_completion_text


def test_local_judge_defaults_to_reasoning_budget(monkeypatch):
    captured: dict = {}

    class FakeLlm:
        def create_completion(self, prompt, **kwargs):
            captured.update(kwargs)
            captured["prompt"] = prompt
            return {"choices": [{"text": "FINAL: UNRELATED"}]}

    fake_module = type(sys)("llama_cpp")
    fake_module.Llama = lambda **kwargs: FakeLlm()
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    judge = make_local_frame_judge("/fake/model.gguf")
    assert judge("a", "b") == "UNRELATED"
    assert captured["max_tokens"] == 256
    assert "FINAL:" in captured["prompt"]


def test_anthropic_judge_without_client_or_key_raises():
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("key present in environment")
    with pytest.raises(RuntimeError):
        AnthropicFrameJudge()


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown escalation backend"):
        resolve_escalation_judge("openai:gpt-4")


def test_resolve_local_without_path_raises():
    import os

    env = os.environ.pop("GIN_ESCALATION_MODEL", None)
    try:
        with pytest.raises(RuntimeError, match="local escalation judge needs"):
            resolve_escalation_judge("local")
    finally:
        if env is not None:
            os.environ["GIN_ESCALATION_MODEL"] = env


def test_make_local_frame_judge_uses_injected_llm(monkeypatch):
    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLlm:
        def create_completion(self, prompt, **kwargs):
            assert "[INST]" in prompt
            return {"choices": [{"text": "DIVERGENT"}]}

    fake_module = type(sys)("llama_cpp")
    fake_module.Llama = lambda **kwargs: FakeLlm()
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    judge = make_local_frame_judge("/fake/model.gguf", n_ctx=2048, n_gpu_layers=0)
    assert judge("official text", "grassroots text") == "DIVERGENT"


def test_resolve_local_returns_local_suffix(monkeypatch):
    captured: dict = {}

    def fake_make(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return lambda a, b: "AGREE"

    monkeypatch.setattr(
        "gin.cartographer.escalation.make_local_frame_judge",
        fake_make,
    )
    judge, suffix = resolve_escalation_judge(
        "local:/models/test.gguf", n_ctx=8192, n_gpu_layers=-1
    )
    assert suffix == "local"
    assert judge("a", "b") == "AGREE"
    assert captured["path"] == "/models/test.gguf"
    assert captured["n_ctx"] == 8192
