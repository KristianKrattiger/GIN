"""LLM frame-divergence judge — the reframed relation detector.

§6 ruled out entailment-NLI: GIN "divergence" is not propositional contradiction
but *framing / stance divergence over a shared referent*. This detector asks that
question directly — do two statements take competing perspectives on the same
issue (e.g. official/technical vs. community/justice framing), simply agree, or
cover unrelated topics — instead of asking whether one entails the other.

``judge`` is an injectable ``(a_text, b_text) -> str`` returning a label
(``DIVERGENT`` / ``AGREE`` / ``UNRELATED``), so the mapping is testable without a
model. Without one, an ``llm`` (duck-typed to the llama.cpp interface) is prompted
to reason briefly and close with a ``FINAL: <label>`` line, which is parsed as
the verdict.

MEASURED STATUS: the §6 one-word constrained variant was prompt-bias-dominated —
Mistral-7B collapsed to a constant answer (always ``DIVERGENT``; recall 1.0 but
class_c_discrimination 0.0), and a stance-axis variant collapsed to ``SAME``.
The 2026-07-13 probe showed the collapse was substantially a harness artifact:
with reasoning room the same model mixes labels (though it still confuses
"different facet of the same issue" with the divergence axis in both
directions). Calibrate any model with ``scripts/cartographer_eval_escalation.py``
before trusting it. See docs/nc_cartographer_design.plan.md §6.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional

from .models import Assessment, LabeledChunk, Relation

# (a_text, b_text) -> one of {"DIVERGENT", "AGREE", "UNRELATED"}
FrameJudge = Callable[[str, str], str]

_LABEL_TO_RELATION = {
    "DIVERGENT": Relation.CONTRADICTS,   # framing divergence == GIN contradicts edge
    "AGREE": Relation.CORROBORATES,
    "UNRELATED": Relation.UNRELATED,
}

# Canonical framing-judge prompt (local llama.cpp and optional API backends).
# Reasoning-then-FINAL: the 2026-07-13 probe showed the one-word variant starved
# the judge into a constant answer (always DIVERGENT); with reasoning room the
# same 7B mixes labels. Only the FINAL line is parsed as the verdict.
#
# Wording is calibration-sensitive (run 20260713T083941Z): a register exemplar
# in the DIVERGENT definition ("official/technical vs community/justice") made
# Mistral-7B pattern-match the register axis and label every cross-register
# pair DIVERGENT regardless of issue match, and the "frame a topic" intro
# presupposed a shared topic, suppressing UNRELATED. Keep definitions
# register-neutral and the intro presupposition-free; calibrate any wording
# change with scripts/cartographer_eval_escalation.py.
FRAME_JUDGE_PROMPT = """Two excerpts from news reports.

Excerpt A: {a}

Excerpt B: {b}

Reason briefly:
1. Name the shared issue in one sentence, or say the issues differ.
2. State each excerpt's evaluative stance or priority in one sentence each.
3. Do the stances COMPETE (opposing priorities or values) or CORROBORATE (same direction)?

Definitions:
- DIVERGENT: same issue, competing or opposing stances, priorities, or values.
- AGREE: same issue, same stance; they corroborate, even if one adds a caveat.
- UNRELATED: different issues, even if both are statistics or reports from the same broad domain.

End with a final line exactly: FINAL: <DIVERGENT|AGREE|UNRELATED>"""

# Token budget that leaves room for the reasoning steps before FINAL.
REASONING_MAX_TOKENS = 256


def format_frame_judge_prompt(a_text: str, b_text: str, *, llama_inst: bool = False) -> str:
    """Format the canonical prompt; llama_inst wraps with Mistral [INST] tags."""
    body = FRAME_JUDGE_PROMPT.format(a=a_text, b=b_text)
    if llama_inst:
        return f"[INST] {body} [/INST]"
    return body


# Back-compat alias for tests and internal use.
_PROMPT = FRAME_JUDGE_PROMPT  # LlmFrameJudge uses format_frame_judge_prompt instead


_FINAL_RE = re.compile(r"FINAL:\s*\W*(DIVERGENT|AGREE|UNRELATED)")
_LABEL_RE = re.compile(r"DIVERGENT|AGREE|UNRELATED")


def _parse_label(text: str) -> str:
    """FINAL-line verdict wins; else last keyword (conclusions come last);
    else conservative UNRELATED. Reasoning text names labels it rejects, so
    first-keyword scanning would bias toward DIVERGENT."""
    up = text.upper()
    finals = _FINAL_RE.findall(up)
    if finals:
        return finals[-1]
    hits = _LABEL_RE.findall(up)
    return hits[-1] if hits else "UNRELATED"


class LlmFrameJudge:
    """Type related pairs by asking an LLM for the framing relationship."""

    name = "llm_frame_judge"

    def __init__(
        self,
        *,
        judge: Optional[FrameJudge] = None,
        llm: Any = None,
        max_tokens: int = REASONING_MAX_TOKENS,
    ) -> None:
        if judge is None and llm is None:
            raise ValueError("LlmFrameJudge needs either a judge callable or an llm")
        self._judge = judge
        self._llm = llm
        self.max_tokens = max_tokens
        # Raw completion of the most recent call — calibration artifacts store
        # it so every model test is self-diagnosing.
        self.last_completion_text: Optional[str] = None

    def _llm_label(self, a_text: str, b_text: str) -> str:
        prompt = format_frame_judge_prompt(a_text, b_text, llama_inst=True)
        out = self._llm.create_completion(
            prompt, max_tokens=self.max_tokens, temperature=0.0, echo=False
        )
        text = out["choices"][0]["text"]
        self.last_completion_text = text
        return _parse_label(text)

    def label(self, a_text: str, b_text: str) -> str:
        if self._judge is not None:
            return self._judge(a_text, b_text)
        return self._llm_label(a_text, b_text)

    # Instances are directly callable as a FrameJudge.
    __call__ = label

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
