"""LLM frame-divergence judge — the reframed relation detector.

§6 ruled out entailment-NLI: GIN "divergence" is not propositional contradiction
but *framing / stance divergence over a shared referent*. This detector asks that
question directly — do two statements take competing perspectives on the same
issue (e.g. official/technical vs. community/justice framing), simply agree, or
cover unrelated topics — instead of asking whether one entails the other.

``judge`` is an injectable ``(a_text, b_text) -> str`` returning a label
(``DIVERGENT`` / ``AGREE`` / ``UNRELATED``), so the mapping is testable without a
model. Without one, an ``llm`` (duck-typed to the llama.cpp interface) is prompted
with a greedy, single-word constrained completion.

MEASURED STATUS (ruled out as-is, like the NLI detector): Mistral-7B zero-shot is
prompt-bias-dominated and collapses to a constant answer — always ``DIVERGENT`` on
the prompt below (recall 1.0 but class_c_discrimination 0.0, and it even labels an
unrelated pair divergent), and always ``SAME`` on a stance-axis variant. It does
not discriminate the institutional-vs-grassroots stance axis. The mapping and
pipeline are correct (an oracle judge scores perfectly); the signal is not cheaply
available from a 7B model zero-shot. See docs/nc_cartographer_design.plan.md §6.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .models import Assessment, LabeledChunk, Relation

# (a_text, b_text) -> one of {"DIVERGENT", "AGREE", "UNRELATED"}
FrameJudge = Callable[[str, str], str]

_LABEL_TO_RELATION = {
    "DIVERGENT": Relation.CONTRADICTS,   # framing divergence == GIN contradicts edge
    "AGREE": Relation.CORROBORATES,
    "UNRELATED": Relation.UNRELATED,
}

_PROMPT = """[INST] You compare how two statements frame a topic.

Statement 1: {a}
Statement 2: {b}

Decide the relationship:
- DIVERGENT: same topic, but competing perspectives, priorities, or values (for example an official/technical framing versus a community/justice framing).
- AGREE: same topic and the same perspective; they corroborate each other.
- UNRELATED: different topics.

Answer with exactly one word: DIVERGENT, AGREE, or UNRELATED. [/INST]"""


def _parse_label(text: str) -> str:
    up = text.upper()
    for label in ("DIVERGENT", "AGREE", "UNRELATED"):
        if label in up:
            return label
    return "UNRELATED"


class LlmFrameJudge:
    """Type related pairs by asking an LLM for the framing relationship."""

    name = "llm_frame_judge"

    def __init__(
        self,
        *,
        judge: Optional[FrameJudge] = None,
        llm: Any = None,
        max_tokens: int = 4,
    ) -> None:
        if judge is None and llm is None:
            raise ValueError("LlmFrameJudge needs either a judge callable or an llm")
        self._judge = judge
        self._llm = llm
        self.max_tokens = max_tokens

    def _llm_label(self, a_text: str, b_text: str) -> str:
        prompt = _PROMPT.format(a=a_text, b=b_text)
        out = self._llm.create_completion(
            prompt, max_tokens=self.max_tokens, temperature=0.0, echo=False
        )
        return _parse_label(out["choices"][0]["text"])

    def label(self, a_text: str, b_text: str) -> str:
        if self._judge is not None:
            return self._judge(a_text, b_text)
        return self._llm_label(a_text, b_text)

    def type_relation(self, a_text: str, b_text: str) -> tuple[Relation, dict]:
        label = self.label(a_text, b_text)
        return _LABEL_TO_RELATION.get(label, Relation.UNRELATED), {"label": label}

    def assess_pair(self, a: LabeledChunk, b: LabeledChunk) -> Assessment:
        relation, ev = self.type_relation(a.text, b.text)
        return Assessment(
            src_chunk_id=a.chunk_id,
            dst_chunk_id=b.chunk_id,
            relation=relation,
            method="llm_frame_judge",
            confidence=1.0 if relation != Relation.UNRELATED else 0.0,
            rationale=f"frame label={ev['label']}",
        )

    def propose_over(
        self, pairs: Iterable[tuple[LabeledChunk, LabeledChunk]]
    ) -> list[Assessment]:
        return [self.assess_pair(a, b) for a, b in pairs]
