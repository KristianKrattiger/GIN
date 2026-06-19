#!/usr/bin/env python3
"""
SEAR Phase 1 dense baseline
===========================

Goal of this file: prove SEAR *behavior* (grammar-constrained extractive
synthesis + exact span-level attribution) on stock Mistral with stock dense
attention, on hardware you actually have. No attention surgery yet.

Core design decision
--------------------
For an *extractive* system, attribution is not a post-hoc attention-tracing
problem. You enforce it at decode time: the model may only emit token spans
that occur verbatim in the corpus, and each emitted span carries a pointer
back to its source position(s). Attribution is then exact *by construction*,
and it survives whatever you later do to the attention mechanism (Phase 3),
because it lives entirely at the decoding layer.

Why a hand-rolled LogitsProcessor instead of Outlines / XGrammar
----------------------------------------------------------------
Outlines and XGrammar are excellent for *static* grammars (regex / JSON / CFG).
The copy-constraint here is a *dynamic* automaton over the corpus token stream:
"legal next tokens = the set of tokens that continue some still-live source
span." Expressing that as a static regex/CFG means an alternation over every
corpus span -- impractical. A LogitsProcessor backed by cursors into the corpus
is the right tool, and it ports unchanged from llama-cpp-python (today, CPU /
quantized GGUF on fairlady) to HF transformers / vLLM (later, GPU retrofit),
because both runtimes accept the same (input_ids, scores) -> scores contract.

Checkpoint
----------
mistralai/Mistral-7B-Instruct-v0.3 as a GGUF (e.g. Q4_K_M). Rationale:
  - Full attention (v0.1's sliding-window is dropped), 32k context -> your
    Phase 2 long-context regime is reachable without the SWA confound.
  - Instruct (not base) so it cooperates with the "select supporting spans"
    framing while the constraint guarantees extractiveness regardless.
  - Q4_K_M runs on CPU / modest hardware via llama-cpp-python. Move to bf16 on
    a rented GPU only once Phase 2 profiling says you need the sequence length.

Install:  pip install llama-cpp-python numpy
Model:    download Mistral-7B-Instruct-v0.3.Q4_K_M.gguf into ./models/
Run:      python sear_phase1.py --selftest      # no model needed, validates logic
          python sear_phase1.py --model ./models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Optional
import numpy as np

NEG_INF = -1e30


# --------------------------------------------------------------------------
# Corpus index: token-level, so span boundaries are token boundaries.
# We index the *actual* token sequence of each document, so verbatim copy is
# tokenization-consistent. (Cross-document span starts mid-sentence can still
# tokenize differently due to leading-space BPE; for the baseline we only let
# spans start at real corpus positions and copy token-by-token, which sidesteps
# that. Flag for later: a content-aware start would need re-tokenization care.)
# --------------------------------------------------------------------------
@dataclass
class Corpus:
    docs: list[list[int]]                       # doc_id -> token ids
    doc_names: list[str]
    start_index: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_texts(cls, texts: dict[str, str], tokenize: Callable[[bytes], list[int]]):
        docs, names = [], []
        for name, text in texts.items():
            names.append(name)
            docs.append(tokenize(text.encode("utf-8")))
        index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for d, toks in enumerate(docs):
            for p, t in enumerate(toks):
                index[t].append((d, p))
        return cls(docs=docs, doc_names=names, start_index=dict(index))

    def continuation(self, doc: int, pos: int) -> Optional[int]:
        nxt = pos + 1
        return self.docs[doc][nxt] if nxt < len(self.docs[doc]) else None


# --------------------------------------------------------------------------
# The constraint. Maintains a set of live cursors -- (doc, pos) of the last
# matched token -- so the legal continuation set is the union of next tokens
# across all live cursors. A span present in N documents keeps N cursors live
# until the emitted continuation diverges; the surviving set at span-close *is*
# the corpus-situatedness / divergence signal, not noise.
# --------------------------------------------------------------------------
BOUNDARY, IN_SPAN = "BOUNDARY", "IN_SPAN"


@dataclass
class Segment:
    token_ids: list[int]
    sources: list[tuple[int, int, int]]   # (doc_id, start_pos, end_pos_exclusive)
    kind: str                             # "extract" or "connective"


class ExtractiveCopyConstraint:
    def __init__(self, corpus: Corpus, prompt_len: int, eos_id: int,
                 delim_id: int, min_span_len: int = 3):
        self.corpus = corpus
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.delim_id = delim_id            # boundary marker the model may emit
        self.min_span_len = min_span_len
        self.structural = {eos_id, delim_id}

        self.mode = BOUNDARY
        self.cursors: list[tuple[int, int]] = []   # live (doc, pos)
        self.span_start: list[tuple[int, int]] = []  # cursors at span start
        self.span_len = 0
        self.segments: list[Segment] = []
        self._cur_tokens: list[int] = []
        self._seen = prompt_len

    # ---- legal next-token set given current state -------------------------
    def _allowed(self) -> set[int]:
        if self.mode == BOUNDARY:
            allowed = set(self.corpus.start_index.keys())  # start any span
            allowed.add(self.eos_id)                       # or stop
            return allowed
        # IN_SPAN: continue any live cursor ...
        allowed = set()
        for (d, p) in self.cursors:
            c = self.corpus.continuation(d, p)
            if c is not None:
                allowed.add(c)
        # ... and, once the span is long enough to be a real extraction,
        # allow closing it (delimiter) or ending generation.
        if self.span_len >= self.min_span_len:
            allowed.add(self.delim_id)
            allowed.add(self.eos_id)
        if not allowed:           # dead end (span ran to doc end, too short)
            allowed.add(self.eos_id)
        return allowed

    # ---- advance state by one *generated* token ---------------------------
    def _consume(self, tok: int):
        if self.mode == BOUNDARY:
            if tok in self.structural:
                if tok == self.delim_id:
                    self.segments.append(
                        Segment([tok], [], "connective"))
                return
            # start a new span
            self.cursors = list(self.corpus.start_index.get(tok, []))
            self.span_start = list(self.cursors)
            self.span_len = 1
            self._cur_tokens = [tok]
            self.mode = IN_SPAN
            return
        # IN_SPAN
        if tok in self.structural:
            self._close_span()
            if tok == self.delim_id:
                self.segments.append(Segment([tok], [], "connective"))
            self.mode = BOUNDARY
            return
        # continue: keep only cursors whose continuation == tok, advance them
        new_cursors = []
        for (d, p) in self.cursors:
            if self.corpus.continuation(d, p) == tok:
                new_cursors.append((d, p + 1))
        self.cursors = new_cursors
        self.span_len += 1
        self._cur_tokens.append(tok)

    def _close_span(self):
        if self.span_len == 0:
            return
        sources = []
        for (d, end_pos) in self.cursors:          # end_pos = last matched
            start = end_pos - (self.span_len - 1)
            sources.append((d, start, end_pos + 1))
        self.segments.append(Segment(list(self._cur_tokens), sources, "extract"))
        self.span_len = 0
        self._cur_tokens = []

    # ---- runtime hook: (input_ids, scores) -> scores ----------------------
    def __call__(self, input_ids, scores):
        ids = list(input_ids)
        # consume any generated tokens we haven't processed yet
        for i in range(self._seen, len(ids)):
            self._consume(ids[i])
        self._seen = len(ids)

        allowed = self._allowed()
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.full(scores.shape, NEG_INF, dtype=np.float32)
        idx = np.fromiter((t for t in allowed if t < scores.shape[0]),
                          dtype=np.int64)
        mask[idx] = scores[idx]
        return mask

    # ---- read the attribution record -------------------------------------
    def finalize(self):
        if self.mode == IN_SPAN:
            self._close_span()
        return self.segments

    def render(self, detok: Callable[[list[int]], str]) -> str:
        out = []
        for seg in self.segments:
            if seg.kind == "connective":
                out.append("  |  ")
                continue
            text = detok(seg.token_ids)
            srcs = ", ".join(f"{self.corpus.doc_names[d]}[{s}:{e}]"
                             for (d, s, e) in seg.sources) or "UNATTRIBUTED"
            tag = "AMBIGUOUS" if len(seg.sources) > 1 else "EXACT"
            out.append(f'"{text}"  <- {tag}: {srcs}')
        return "\n".join(out)


# ==========================================================================
# Self-test: exercises the constraint with a stub tokenizer + tiny corpus,
# no model. Proves masking, multi-token attribution, and divergence pruning.
# ==========================================================================
def _selftest():
    # toy vocab
    vocab = {w: i for i, w in enumerate(
        ["<pad>", "<eos>", "|", "the", "fox", "ran", "fast", "dog",
         "slept", "all", "day", "and"])}
    inv = {i: w for w, i in vocab.items()}
    tok = lambda b: [vocab[w] for w in b.decode().split()]
    detok = lambda ids: " ".join(inv[i] for i in ids)

    # "the fox ran" appears in doc A and doc B; they diverge after "ran".
    corpus = Corpus.from_texts(
        {"A": "the fox ran fast", "B": "the fox ran and the dog slept all day"},
        tokenize=tok)

    c = ExtractiveCopyConstraint(
        corpus, prompt_len=0, eos_id=vocab["<eos>"],
        delim_id=vocab["|"], min_span_len=3)

    V = len(vocab)
    flat = np.zeros(V, dtype=np.float32)   # uniform logits; mask does the work

    def step(generated):
        scores = c(np.array(generated, dtype=np.intc), flat.copy())
        return {inv[i] for i in range(V) if scores[i] > NEG_INF / 2}

    # at BOUNDARY: may start any token that begins a corpus position
    allowed0 = step([])
    assert "the" in allowed0 and "fox" in allowed0
    assert "<eos>" in allowed0 and "|" not in allowed0, allowed0

    # script the path: the fox ran  (shared by A and B)
    seq = [vocab["the"], vocab["fox"], vocab["ran"]]
    allowed1 = step(seq[:1])      # after "the"
    # "the" occurs twice in B (the fox / the dog), so both continuations live:
    assert allowed1 == {"fox", "dog"}, allowed1
    allowed2 = step(seq[:2])      # after "the fox"
    assert allowed2 == {"ran"}, allowed2
    allowed3 = step(seq[:3])      # after "the fox ran" (span_len=3 >= min)
    # may continue (A->"fast", B->"and") or close (| / <eos>)
    assert {"fast", "and", "|", "<eos>"} == allowed3, allowed3

    # close the span here -> should be AMBIGUOUS across A and B
    c2 = ExtractiveCopyConstraint(
        corpus, 0, vocab["<eos>"], vocab["|"], min_span_len=3)
    for i in range(3):
        c2(np.array(seq[:i], dtype=np.intc), flat.copy())
    c2(np.array(seq, dtype=np.intc), flat.copy())          # state synced
    c2(np.array(seq + [vocab["|"]], dtype=np.intc), flat.copy())  # close
    segs = c2.finalize()
    ext = [s for s in segs if s.kind == "extract"][0]
    assert detok(ext.token_ids) == "the fox ran", detok(ext.token_ids)
    assert len(ext.sources) == 2, ext.sources       # divergence signal: A and B
    assert {d for d, _, _ in ext.sources} == {0, 1}

    # now extend into "fast" -> diverges, prunes to A only
    c3 = ExtractiveCopyConstraint(
        corpus, 0, vocab["<eos>"], vocab["|"], min_span_len=3)
    full = seq + [vocab["fast"]]
    for i in range(len(full)):
        c3(np.array(full[:i], dtype=np.intc), flat.copy())
    c3(np.array(full, dtype=np.intc), flat.copy())
    c3(np.array(full + [vocab["<eos>"]], dtype=np.intc), flat.copy())
    ext3 = [s for s in c3.finalize() if s.kind == "extract"][0]
    assert len(ext3.sources) == 1 and ext3.sources[0][0] == 0, ext3.sources

    print("self-test OK: masking, exact multi-token attribution, and "
          "cross-source divergence pruning all behave.")
    print("  render of ambiguous span:")
    print("   ", c2.render(detok).replace("\n", "\n    "))


# ==========================================================================
# Live run against a Mistral GGUF via llama-cpp-python.
# ==========================================================================
def _run(model_path: str):
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=0, verbose=False)
    tok = lambda b: llm.tokenize(b, add_bos=False)
    detok = lambda ids: llm.detokenize(ids).decode("utf-8", errors="replace")

    # toy corpus -- swap in your real documents here
    corpus = Corpus.from_texts({
        "munchies_sop": "Closing requires reconciling the Toast drawer against the "
                        "shift report before the safe drop is logged.",
        "toast_doc":    "The drawer must be reconciled against the shift report at "
                        "end of day; discrepancies over five dollars are flagged.",
    }, tokenize=tok)

    prompt = (
        "[INST] Using only verbatim spans from the source documents, assemble an "
        "answer to: what has to happen before the safe drop is logged at close? "
        "Separate extracted spans with '|'. [/INST]")
    prompt_ids = llm.tokenize(prompt.encode(), add_bos=True)

    eos_id = llm.token_eos()
    delim_id = tok(b" |")[-1]
    constraint = ExtractiveCopyConstraint(
        corpus, prompt_len=len(prompt_ids), eos_id=eos_id,
        delim_id=delim_id, min_span_len=4)

    from llama_cpp import LogitsProcessorList
    out = llm.create_completion(
        prompt, max_tokens=200, temperature=0.0,
        logits_processor=LogitsProcessorList([constraint]))
    print("RAW:", out["choices"][0]["text"])
    print("\nATTRIBUTION RECORD\n" + "=" * 60)
    print(constraint.render(detok))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()
    if args.selftest or not args.model:
        _selftest()
    else:
        _run(args.model)
