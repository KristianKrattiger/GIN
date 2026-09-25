"""Tests for gin.assay_proposer.slots, driven by word-level and subword stand-ins for llama_cpp.Llama."""
import re
from dataclasses import dataclass

import numpy as np
import pytest

from gin.assay_proposer.corpus import MAX_QUOTE_WORDS
from gin.assay_proposer.slots import ChoiceConstraint, _complete, _Loud, propose_pass
from sear.processor import NEG_INF


class FakeLlm:
    """Word-level stand-in for llama_cpp.Llama.

    Each create_completion call has a 'wish': the text this fake model would
    write if nothing constrained it. At every step the next wished word scores
    highest, other tokens score 1 and EOS 0 (so a model whose wish is masked
    keeps writing rather than giving up); the logits processors mask the
    rest and the best survivor wins (lowest id on ties). A test scripts what
    the model wants and observes what the constraint lets through.
    """

    EOS = 1
    V = 4096

    def __init__(self, wishes: list[str]):
        self.wishes = list(wishes)
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        self.prompts: list[str] = []

    def _id(self, w: str) -> int:
        if w not in self.vocab:
            i = len(self.vocab) + 2
            self.vocab[w] = i
            self.inv[i] = w
        return self.vocab[w]

    def tokenize(self, b: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return [self._id(w) for w in b.decode("utf-8").split()]

    def detokenize(self, ids) -> bytes:
        return " ".join(self.inv[i] for i in ids if i != self.EOS).encode("utf-8")

    def token_eos(self) -> int:
        return self.EOS

    def create_completion(self, prompt, max_tokens=16, temperature=0.8, logits_processor=None, stop=None, **_):
        self.prompts.append(prompt)
        wish_text = self.wishes.pop(0) if self.wishes else ""
        if stop and "\n" in stop:
            wish_text = wish_text.split("\n")[0]
        wish = self.tokenize(wish_text.encode("utf-8"))
        ids = self.tokenize(prompt.encode("utf-8"))
        out: list[int] = []
        finish = "length"
        for step in range(max_tokens):
            scores = np.ones(self.V, dtype=np.float32)
            scores[self.EOS] = 0.0
            scores[wish[step] if step < len(wish) else self.EOS] = 10.0
            if logits_processor is not None:
                scores = logits_processor(np.array(ids + out, dtype=np.intc), scores)
            tok = int(np.argmax(scores))
            if tok == self.EOS:
                finish = "stop"
                break
            out.append(tok)
        return {"choices": [{"text": self.detokenize(out).decode("utf-8"), "finish_reason": finish}]}


def _render(system: str, user: str) -> str:
    return f"SYSTEM {system} USER {user} ASSISTANT"


@dataclass
class Ex:
    docId: str
    role: str
    text: str
    label: str = "L"


EXCERPTS = [
    Ex("vendor", "claimant", "Acme uptime is 99.99% everywhere.\nAcme support answers in one hour."),
    Ex("status", "independent", "Acme reported four outages this quarter."),
]


def _run(wishes, excerpts=EXCERPTS, max_proposals=8):
    llm = FakeLlm(wishes)
    out = propose_pass(llm, _render, system="S", user="U", excerpts=excerpts, max_proposals=max_proposals)
    return llm, out


def test_a_full_proposal_comes_back_with_exact_quotes_and_docids_from_the_spans():
    llm, out = _run([
        "Acme uptime is 99.99% everywhere.", "contradicts", "Acme reported four outages this quarter.",
        "uptime", "uptime guarantee", "outages contradict it", "0.90", "no",
    ])
    assert out == [{
        "type": "contradicts",
        "topic": "uptime",
        "statement": "uptime guarantee",
        "from": {"docId": "vendor", "quote": "Acme uptime is 99.99% everywhere."},
        "to": {"docId": "status", "quote": "Acme reported four outages this quarter."},
        "rationale": "outages contradict it",
        "confidence": 0.9,
    }]
    assert llm.wishes == []


def test_a_paraphrase_the_model_wants_is_unreachable_it_can_only_copy():
    llm, out = _run([
        "Acme promises perfect uptime forever.", "unsupported", "t", "s", "r", "0.50", "no",
    ])
    assert out[0]["from"] == {"docId": "vendor", "quote": "Acme uptime is 99.99% everywhere."}
    assert out[0]["to"] is None


def test_unsupported_skips_the_independent_quote_slot():
    llm, _ = _run(["Acme uptime is 99.99% everywhere.", "unsupported", "t", "s", "r", "0.50", "no"])
    assert not any(p.endswith("Independent quote: ") for p in llm.prompts)
    assert len(llm.prompts) == 7


def test_each_quote_slot_only_reaches_its_own_role():
    _, out = _run([
        "Acme reported four outages this quarter.",   # wants the independent line as the claim
        "contradicts",
        "Acme uptime is 99.99% everywhere.",          # wants the claimant line as the evidence
        "t", "s", "r", "0.50", "no",
    ])
    assert out[0]["from"]["docId"] == "vendor"
    assert out[0]["to"] == {"docId": "status", "quote": "Acme reported four outages this quarter."}


def test_a_quote_never_crosses_a_line_break():
    excerpts = [
        Ex("vendor", "claimant", "Acme uptime is great\nAcme support is slow today"),
        Ex("status", "independent", "Acme reported four outages this quarter."),
    ]
    _, out = _run(
        ["Acme uptime is great Acme support is slow today", "unsupported", "t", "s", "r", "0.50", "no"],
        excerpts=excerpts,
    )
    assert out[0]["from"]["quote"] == "Acme uptime is great"


def test_an_over_long_sentence_is_never_quoted():
    long = " ".join(f"w{i}" for i in range(MAX_QUOTE_WORDS + 1)) + "."
    llm, out = _run([long], excerpts=[Ex("vendor", "claimant", long)])
    assert out == []
    assert len(llm.prompts) == 1


def test_an_empty_claim_slot_ends_the_pass():
    llm, out = _run([""])
    assert out == []
    assert len(llm.prompts) == 1


def test_a_missing_independent_quote_drops_only_that_proposal():
    only_claimant = [Ex("vendor", "claimant", "Acme uptime is 99.99% everywhere.\nAcme support answers in one hour.")]
    _, out = _run(
        [
            "Acme uptime is 99.99% everywhere.", "contradicts", "yes",
            "Acme support answers in one hour.", "unsupported", "t", "s", "r", "0.50", "no",
        ],
        excerpts=only_claimant,
    )
    assert [p["type"] for p in out] == ["unsupported"]
    assert out[0]["from"]["quote"] == "Acme support answers in one hour."


def test_the_per_pass_cap_stops_without_asking_to_continue():
    llm, out = _run(
        [
            "Acme uptime is 99.99% everywhere.", "unsupported", "t", "s", "r", "0.50", "yes",
            "Acme support answers in one hour.", "unsupported", "t", "s", "r", "0.50",
        ],
        max_proposals=2,
    )
    assert len(out) == 2
    assert llm.wishes == []
    assert llm.prompts[-1].endswith("Confidence: ")


def test_a_prompt_length_mismatch_fails_loudly_instead_of_misattributing():
    class OffByOne(FakeLlm):
        def tokenize(self, b, add_bos=True, special=False):
            ids = super().tokenize(b, add_bos, special)
            return ids + [2] if special else ids

    with pytest.raises(RuntimeError, match="prompt is"):
        propose_pass(OffByOne(["Acme uptime is 99.99% everywhere."]), _render,
                     system="S", user="U", excerpts=EXCERPTS)


def test_choice_constraint_allows_only_listed_sequences_then_eos():
    c = ChoiceConstraint([[5, 6], [5, 7], [8]], prompt_len=2, eos_id=1)

    def allowed(ids):
        scores = c(np.array(ids, dtype=np.intc), np.zeros(10, dtype=np.float32))
        return {t for t in range(10) if scores[t] > NEG_INF / 2}

    assert allowed([0, 0]) == {5, 8}
    assert allowed([0, 0, 5]) == {6, 7}
    assert allowed([0, 0, 5, 6]) == {1}
    assert allowed([0, 0, 8]) == {1}


class SubwordFakeLlm(FakeLlm):
    """FakeLlm with GPT-style subword tokens: a word carries its leading space."""

    def tokenize(self, b, add_bos=True, special=False):
        return [self._id(p) for p in re.findall(r" ?[A-Za-z0-9%]+| ?[^\sA-Za-z0-9]|\s+", b.decode("utf-8"))]

    def detokenize(self, ids):
        return "".join(self.inv[i] for i in ids if i != self.EOS).encode("utf-8")


def test_a_second_sentence_on_a_line_is_quoted_whole_under_a_subword_tokenizer():
    llm = SubwordFakeLlm([" Acme is down.", "unsupported", "t", "s", "r", "0.50", "no"])
    out = propose_pass(llm, _render, system="S", user="U",
                       excerpts=[Ex("vendor", "claimant", "Acme is up. Acme is down.")])
    assert out[0]["from"]["quote"] == "Acme is down."


def test_a_quote_stops_at_the_first_sentence_end_even_when_the_model_wants_to_keep_writing():
    # Reviewer's repro: the model's natural continuation after the first sentence
    # end is the corpus's next token (the start of sentence 2), and that beats the
    # masked "stop" the model actually wished for. The quote must still stop at the
    # first sentence.
    line = (
        "Acme uptime held steady across every region this entire quarter without exception. "
        "Acme support answered every single ticket within one hour during the same quarter."
    )
    excerpts = [Ex("vendor", "claimant", line)]
    _, out = _run(
        [line, "unsupported", "t", "s", "r", "0.50", "no"],
        excerpts=excerpts,
    )
    assert out[0]["from"]["quote"] == (
        "Acme uptime held steady across every region this entire quarter without exception."
    )


def test_a_quote_from_the_second_sentence_does_not_bleed_into_a_third():
    line = (
        "Acme uptime held steady this quarter. "
        "Acme support answered every ticket within an hour. "
        "Acme shipped three releases without incident."
    )
    excerpts = [Ex("vendor", "claimant", line)]
    wish = (
        "Acme support answered every ticket within an hour. "
        "Acme shipped three releases without incident."
    )
    _, out = _run(
        [wish, "unsupported", "t", "s", "r", "0.50", "no"],
        excerpts=excerpts,
    )
    assert out[0]["from"]["quote"] == "Acme support answered every ticket within an hour."


class _Boom:
    def __call__(self, input_ids, scores):
        raise RuntimeError("boom")


def test_loud_masks_to_eos_only_and_records_the_inner_exception_without_raising():
    loud = _Loud(_Boom(), eos_id=1)
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    out = loud(np.array([0], dtype=np.intc), scores)  # does not raise
    assert isinstance(loud.error, RuntimeError) and str(loud.error) == "boom"
    assert out[1] == 5.0
    assert out[0] <= NEG_INF / 2 and out[2] <= NEG_INF / 2


class SwallowingFakeLlm:
    """Stands in for llama-cpp-python 0.3.30's ctypes C callback, which prints
    and drops any exception a logits processor raises inside it -- decoding
    would otherwise carry on with unmasked logits."""

    def token_eos(self) -> int:
        return 1

    def create_completion(self, prompt, max_tokens=16, temperature=0.8, logits_processor=None, stop=None, **_):
        ids = np.array([0], dtype=np.intc)
        scores = np.ones(4, dtype=np.float32)
        for lp in logits_processor or []:
            try:
                lp(ids, scores)
            except Exception:
                pass  # the ctypes callback swallows it here
        return {"choices": [{"text": "", "finish_reason": "stop"}]}


def test_complete_reraises_an_exception_a_ctypes_callback_would_otherwise_swallow():
    with pytest.raises(RuntimeError, match="boom"):
        _complete(SwallowingFakeLlm(), "prompt", max_tokens=4, temperature=0.1, processor=_Boom())
