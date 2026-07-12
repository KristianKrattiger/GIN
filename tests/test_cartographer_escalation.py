"""Escalation tier: API-class LLM judge for anchor-less, topically-close pairs.

The 2026-07-12 signal audit showed issue-frame divergence (same issue, opposing
frames, no shared story entities) is undetectable with local models: six
bi-encoders (margin -0.39..-0.08), NLI, register axis, and zero-/few-shot
Mistral-7B all failed. The escalation tier routes only the small candidate set
the cheap path cannot type (cross-outlet, not same-story, cosine above a floor
— 91-338 pairs on the 136-chunk corpus depending on floor) to an API-class
judge, off by default.
"""
import pytest

from gin.cartographer.combined import CombinedRelationProposer, Thresholds
from gin.cartographer.escalation import (
    AnthropicFrameJudge,
    escalation_candidates,
    escalate_proposals,
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

    proposals = escalate_proposals(pairs, judge)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.relation == Relation.CONTRADICTS
    assert p.method == "llm_frame_judge:escalation"
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


def test_anthropic_judge_without_client_or_key_raises():
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("key present in environment")
    with pytest.raises(RuntimeError):
        AnthropicFrameJudge()
