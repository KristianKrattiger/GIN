"""Escalation tier — model-agnostic LLM judge for issue-frame divergence.

The 2026-07-12 signal audit ruled out every on-hand cheap signal for
issue-frame divergence (same issue, opposing frames, zero shared story entities).
Curated YAML edges cover it today (``scan --curated-edges``).

This module is the forward path for corpora where curation does not scale: the
cheap pipeline already types every same-story pair, so only the small residue —
cross-outlet, NOT same-story, cosine above a floor (91-338 pairs on the
136-chunk corpus for floors 0.40-0.30) — escalates to a framing judge.

**Local-first:** ``resolve_escalation_judge("local:path/to/model.gguf")`` uses
llama.cpp via ``LlmFrameJudge`` — no API billing. ``anthropic:model`` is an
optional backend. Off by default. Labels reuse the frame-judge vocabulary
(DIVERGENT / AGREE / UNRELATED); only DIVERGENT becomes a proposal.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from .combined import CombinedRelationProposer
from .frame_judge import (
    REASONING_MAX_TOKENS,
    FrameJudge,
    LlmFrameJudge,
    _parse_label,
    format_frame_judge_prompt,
)
from .models import EdgeProposal, LabeledChunk, Relation

DEFAULT_ESCALATION_COS_FLOOR = 0.30
DEFAULT_MAX_CANDIDATES = 400
DEFAULT_ESCALATION_N_CTX = 4096
DEFAULT_ESCALATION_GPU_LAYERS = -1
ESCALATION_CONFIDENCE = 0.7
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ENV_ESCALATION_MODEL = "GIN_ESCALATION_MODEL"


def escalation_candidates(
    pairs: Iterable[tuple[LabeledChunk, LabeledChunk]],
    proposer: CombinedRelationProposer,
    *,
    cos_floor: float = DEFAULT_ESCALATION_COS_FLOOR,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[tuple[LabeledChunk, LabeledChunk]]:
    """Pairs the cheap path cannot type: not same-story, cosine >= floor."""
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
    method_suffix: str = "unknown",
) -> list[EdgeProposal]:
    """Run the judge over escalated pairs; DIVERGENT verdicts become proposals."""
    from .scan import sentence_anchor

    method = f"llm_frame_judge:escalation:{method_suffix}"
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
                method=method,
                confidence=confidence,
                rationale=f"escalation judge label={label}",
                src_anchor=sentence_anchor(a.text),
                dst_anchor=sentence_anchor(b.text),
            )
        )
    return proposals


def make_local_frame_judge(
    model_path: str,
    *,
    n_ctx: int = DEFAULT_ESCALATION_N_CTX,
    n_gpu_layers: int = DEFAULT_ESCALATION_GPU_LAYERS,
    max_tokens: int = REASONING_MAX_TOKENS,
) -> FrameJudge:
    """Build a FrameJudge backed by a local llama.cpp GGUF (no API billing).

    Returns the callable ``LlmFrameJudge`` instance so callers can read
    ``last_completion_text`` (calibration artifacts store the reasoning).
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "local escalation judge needs llama-cpp-python (pip install llama-cpp-python)"
        ) from exc
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return LlmFrameJudge(llm=llm, max_tokens=max_tokens)


def resolve_escalation_judge(
    spec: str,
    *,
    n_ctx: int = DEFAULT_ESCALATION_N_CTX,
    n_gpu_layers: int = DEFAULT_ESCALATION_GPU_LAYERS,
) -> tuple[FrameJudge, str]:
    """Parse ``BACKEND[:SPEC]`` and return ``(judge_callable, method_suffix)``.

    Supported backends:
    - ``local:path/to/model.gguf`` — llama.cpp (primary, no API billing)
    - ``local`` — reads model path from ``GIN_ESCALATION_MODEL``
    - ``anthropic:model`` — optional API backend
    """
    backend, _, rest = spec.partition(":")
    backend = backend.strip().lower()
    if backend == "local":
        model_path = rest.strip() or os.environ.get(ENV_ESCALATION_MODEL, "")
        if not model_path:
            raise RuntimeError(
                f"local escalation judge needs a model path "
                f"(local:/path/to/model.gguf or {ENV_ESCALATION_MODEL})"
            )
        return make_local_frame_judge(
            model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
        ), "local"
    if backend == "anthropic":
        model = rest.strip() or DEFAULT_ANTHROPIC_MODEL
        return AnthropicFrameJudge(model=model), "anthropic"
    raise ValueError(f"unknown escalation backend {backend!r} (use local or anthropic)")


class AnthropicFrameJudge:
    """Optional FrameJudge backed by the Anthropic API."""

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = REASONING_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.last_completion_text: Optional[str] = None
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
                    "content": format_frame_judge_prompt(a_text, b_text, llama_inst=False),
                }
            ],
        )
        text = "".join(getattr(block, "text", "") for block in resp.content)
        self.last_completion_text = text
        return _parse_label(text)
