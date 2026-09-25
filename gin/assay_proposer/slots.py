"""
gin.assay_proposer.slots
------------------------
Drive one Assay proposal pass as a sequence of small constrained decodes.

SEAR has no structured-output mode, and one JSON stream would put its copy
cursor over raw corpus tokens while the output needed JSON-escaped text. So
the structure lives here: one decode per field, each appended to a growing
assistant turn so llama.cpp reuses the shared prompt prefix. Quote fields run
under SEAR's ExtractiveCopyConstraint; enumerated fields (relation type,
confidence, continue?) under ChoiceConstraint; free-text fields (topic,
statement, rationale) are single lines. A quote's docId is read off the span
SEAR closed, never generated, so the model cannot mislabel its source.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np

from sear.processor import NEG_INF, ExtractiveCopyConstraint

from .corpus import ExcerptLike, LineCorpus, build_line_corpus

RELATION_TYPES = ["contradicts", "corroborates", "updates", "unsupported"]
CONFIDENCES = [f"{i / 20:.2f}" for i in range(1, 21)]  # 0.05 .. 1.00
MIN_SPAN_TOKENS = 4
MAX_QUOTE_TOKENS = 96
FREE_TEXT_TOKENS = {"topic": 16, "statement": 32, "rationale": 64}

SCAFFOLD_NOTE = (
    "\n\nAnswer one proposal at a time by filling in the labelled fields. "
    "Quotes are copied from the excerpts above; the decoder only lets you "
    "copy text that is really there."
)

RenderPrompt = Callable[[str, str], str]


def prompt_token_count(llm: Any, prompt: str) -> int:
    """Tokens llama.cpp's create_completion evaluates for this prompt.

    create_completion tokenizes with special=True (a chat template's control
    tokens are single special tokens, not literal text) and prepends BOS only
    when the model asks for one; tokenize(add_bos=True, special=True) matches.
    """
    return len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))


def _mask(scores, allowed: set[int]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    out = np.full(scores.shape, NEG_INF, dtype=np.float32)
    idx = np.fromiter((t for t in allowed if t < scores.shape[0]), dtype=np.int64)
    out[idx] = scores[idx]
    return out


class ChoiceConstraint:
    """Allow exactly one of a fixed set of token sequences, then EOS."""

    def __init__(self, choices: list[list[int]], prompt_len: int, eos_id: int):
        self.choices = [c for c in choices if c]
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.generated: list[int] = []

    def _allowed(self) -> set[int]:
        n = len(self.generated)
        allowed = {c[n] for c in self.choices if len(c) > n and c[:n] == self.generated}
        if self.generated in self.choices:
            allowed.add(self.eos_id)
        return allowed or {self.eos_id}

    def __call__(self, input_ids, scores):
        self.generated = list(input_ids)[self.prompt_len:]
        return _mask(scores, self._allowed())


class _PromptLengthGuard:
    """Fail loudly if llama.cpp's prompt token count differs from ours.

    SEAR treats every input id past prompt_len as generated; an off-by-one
    would feed it a prompt token as the first quoted token and silently
    misattribute the quote.
    """

    def __init__(self, inner, prompt_len: int):
        self.inner = inner
        self.prompt_len = prompt_len
        self.checked = False

    def __call__(self, input_ids, scores):
        if not self.checked:
            if len(input_ids) != self.prompt_len:
                raise RuntimeError(
                    f"assay_proposer: prompt is {len(input_ids)} tokens to llama.cpp "
                    f"but {self.prompt_len} to the constraint"
                )
            self.checked = True
        return self.inner(input_ids, scores)


def _complete(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    processor=None,
    stop: Optional[list[str]] = None,
) -> tuple[str, str]:
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if processor is not None:
        from llama_cpp import LogitsProcessorList  # lazy, as gin/corpus/generate.py does

        kwargs["logits_processor"] = LogitsProcessorList([processor])
    if stop is not None:
        kwargs["stop"] = stop
    choice = llm.create_completion(prompt, **kwargs)["choices"][0]
    return choice["text"], choice.get("finish_reason") or "stop"


def _quote(
    llm: Any,
    lc: LineCorpus,
    prompt: str,
    focus: frozenset[int],
    temperature: float,
) -> Optional[tuple[str, str]]:
    """(docId, quote) for one span copied from the focus lines, or None if none was closed."""
    if not focus:
        return None
    n = prompt_token_count(llm, prompt)
    constraint = ExtractiveCopyConstraint(
        lc.corpus,
        prompt_len=n,
        eos_id=llm.token_eos(),
        delim_id=llm.tokenize(b"|", add_bos=False)[-1],
        min_span_len=MIN_SPAN_TOKENS,
        focus_doc_indices=focus,
        forbidden_starts=lc.forbidden_starts,
        span_must_start_at_sentence=True,
        span_must_close_at_sentence_end=True,
        stop_after_first_extract=True,
    )
    _text, finish = _complete(
        llm, prompt, max_tokens=MAX_QUOTE_TOKENS, temperature=temperature,
        processor=_PromptLengthGuard(constraint, n),
    )
    if finish == "length":
        return None  # cut off mid-span: a fragment, not a quote
    extracts = [s for s in constraint.finalize() if s.kind == "extract"]
    if not extracts or not extracts[0].sources or len(extracts[0].token_ids) < MIN_SPAN_TOKENS:
        return None
    seg = extracts[0]
    quote = llm.detokenize(seg.token_ids).decode("utf-8", errors="replace").strip()
    return lc.sources[seg.sources[0][0]].doc_id, quote


def _choose(llm: Any, prompt: str, choices: list[str], temperature: float) -> str:
    n = prompt_token_count(llm, prompt)
    seqs = [llm.tokenize(c.encode("utf-8"), add_bos=False) for c in choices]
    constraint = ChoiceConstraint(seqs, n, llm.token_eos())
    text, _finish = _complete(
        llm, prompt, max_tokens=max(len(s) for s in seqs) + 1, temperature=temperature,
        processor=_PromptLengthGuard(constraint, n),
    )
    text = text.strip()
    if text not in choices:
        raise RuntimeError(f"assay_proposer: choice decode produced {text!r}, not one of {choices}")
    return text


def _line(llm: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    text, _finish = _complete(llm, prompt, max_tokens=max_tokens, temperature=temperature, stop=["\n"])
    return text.strip()


def propose_pass(
    llm: Any,
    render_prompt: RenderPrompt,
    *,
    system: str,
    user: str,
    excerpts: Iterable[ExcerptLike],
    temperature: float = 0.8,
    max_proposals: int = 8,
) -> list[dict]:
    lc = build_line_corpus(excerpts, lambda b: llm.tokenize(b, add_bos=False))
    base = render_prompt(system, user + SCAFFOLD_NOTE)
    turn = ""
    proposals: list[dict] = []
    for n in range(1, max_proposals + 1):
        turn += f"Proposal {n}\nClaimant quote: "
        claim = _quote(llm, lc, base + turn, lc.claimant, temperature)
        if claim is None:
            break
        from_doc, from_quote = claim
        turn += f"{from_quote}\nRelation: "
        rtype = _choose(llm, base + turn, RELATION_TYPES, temperature)
        turn += f"{rtype}\n"

        to: Optional[dict] = None
        if rtype != "unsupported":
            turn += "Independent quote: "
            evidence = _quote(llm, lc, base + turn, lc.independent, temperature)
            if evidence is None:
                turn += "none\n"
            else:
                to = {"docId": evidence[0], "quote": evidence[1]}
                turn += f"{evidence[1]}\n"

        if rtype == "unsupported" or to is not None:
            fields: dict[str, str] = {}
            for name in ("topic", "statement", "rationale"):
                turn += f"{name.capitalize()}: "
                fields[name] = _line(llm, base + turn, FREE_TEXT_TOKENS[name], temperature)
                turn += f"{fields[name]}\n"
            turn += "Confidence: "
            confidence = _choose(llm, base + turn, CONFIDENCES, temperature)
            turn += f"{confidence}\n"
            proposals.append({
                "type": rtype,
                "topic": fields["topic"],
                "statement": fields["statement"],
                "from": {"docId": from_doc, "quote": from_quote},
                "to": to,
                "rationale": fields["rationale"],
                "confidence": float(confidence),
            })

        if n == max_proposals:
            break
        turn += "Another proposal? "
        more = _choose(llm, base + turn, ["yes", "no"], temperature)
        turn += f"{more}\n\n"
        if more == "no":
            break
    return proposals
