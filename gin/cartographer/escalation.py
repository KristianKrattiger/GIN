"""Escalation tier — API-class LLM judge for issue-frame divergence.

The 2026-07-12 signal audit ruled out every on-hand signal for issue-frame
divergence (same issue, opposing frames, zero shared story entities): six
bi-encoders separate it from topically-adjacent noise with margin -0.39..-0.08,
NLI scores ~0 in both directions, a register-axis embedding delta overlaps all
classes, and Mistral-7B collapses to a constant answer zero- and few-shot. The
class is machine-undetectable locally; curated YAML edges cover it today
(scan --curated-edges).

This module is the forward path for corpora where curation does not scale: the
cheap pipeline already types every same-story pair, so only the small residue —
cross-outlet, NOT same-story, cosine above a floor (91-338 pairs on the
136-chunk corpus for floors 0.40-0.30) — escalates to an API-class model, which
handles the framing question that 7B models cannot. Off by default; requires
ANTHROPIC_API_KEY. Labels reuse the frame-judge vocabulary (DIVERGENT / AGREE /
UNRELATED) and only DIVERGENT becomes a proposal.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

from .combined import CombinedRelationProposer
from .frame_judge import FrameJudge, _parse_label
from .models import EdgeProposal, LabeledChunk, Relation

DEFAULT_ESCALATION_COS_FLOOR = 0.30
DEFAULT_MAX_CANDIDATES = 400
# Above the Bookkeeper floor (0.5), below NLI-channel confidences: an API-judge
# verdict is human-adjacent but not propositional evidence.
ESCALATION_CONFIDENCE = 0.7
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

_JUDGE_PROMPT = """You compare how two report excerpts frame a topic.

Excerpt A: {a}

Excerpt B: {b}

Decide the relationship:
- DIVERGENT: same underlying issue, but competing perspectives, priorities, or values (for example an official/technical framing versus a community/justice framing).
- AGREE: same issue and the same perspective; they corroborate each other.
- UNRELATED: different issues, even if both are statistics or reports from the same broad domain.

Answer with exactly one word: DIVERGENT, AGREE, or UNRELATED."""


def escalation_candidates(
    pairs: Iterable[tuple[LabeledChunk, LabeledChunk]],
    proposer: CombinedRelationProposer,
    *,
    cos_floor: float = DEFAULT_ESCALATION_COS_FLOOR,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[tuple[LabeledChunk, LabeledChunk]]:
    """Pairs the cheap path cannot type: not same-story, cosine >= floor.

    Sorted by descending cosine so a budget cap keeps the closest pairs. The
    caller supplies pairs already filtered to cross-outlet candidates (the
    scan's pair source).
    """
    if proposer.same_story is None:
        raise ValueError(
            "escalation needs the stage-1 same-story provider wired "
            "(scan.wire_same_story) to know which pairs the cheap path handled"
        )
    scored: list[tuple[float, tuple[LabeledChunk, LabeledChunk]]] = []
    for a, b in pairs:
        if proposer.same_story(a.text, b.text):
            continue
        cos = proposer.embedding_cosine(a.text, b.text)
        if cos >= cos_floor:
            scored.append((cos, (a, b)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [pair for _cos, pair in scored[:max_candidates]]


def escalate_proposals(
    pairs: Iterable[tuple[LabeledChunk, LabeledChunk]],
    judge: FrameJudge,
    *,
    confidence: float = ESCALATION_CONFIDENCE,
) -> list[EdgeProposal]:
    """Run the judge over escalated pairs; DIVERGENT verdicts become proposals."""
    from .scan import sentence_anchor

    proposals: list[EdgeProposal] = []
    for a, b in pairs:
        label = judge(a.text, b.text)
        if label != "DIVERGENT":
            continue
        proposals.append(
            EdgeProposal(
                src_chunk_id=a.chunk_id,
                dst_chunk_id=b.chunk_id,
                relation=Relation.CONTRADICTS,
                method="llm_frame_judge:escalation",
                confidence=confidence,
                rationale=f"escalation judge label={label}",
                src_anchor=sentence_anchor(a.text),
                dst_anchor=sentence_anchor(b.text),
            )
        )
    return proposals


class AnthropicFrameJudge:
    """FrameJudge backed by the Anthropic API (for the escalation tier).

    Duck-typed ``client`` is injectable for tests; otherwise the ``anthropic``
    SDK and ``ANTHROPIC_API_KEY`` are required at construction time so a
    misconfigured scan fails before spending any model calls.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 8,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "AnthropicFrameJudge needs ANTHROPIC_API_KEY (or an injected client)"
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "AnthropicFrameJudge needs the 'anthropic' package (pip install anthropic)"
            ) from exc
        self._client = anthropic.Anthropic()

    def __call__(self, a_text: str, b_text: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(a=a_text, b=b_text),
                }
            ],
        )
        text = "".join(getattr(block, "text", "") for block in resp.content)
        return _parse_label(text)
