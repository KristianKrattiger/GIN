#!/usr/bin/env python3
"""
SEAR Phase 1 dense baseline
===========================

Goal of this file: prove SEAR *behavior* (grammar-constrained extractive
synthesis + exact span-level attribution) on stock Mistral with stock dense
attention, on hardware you actually have. No attention surgery yet.

Run:      python scripts/sear_phase1.py --selftest      # no model needed
          python scripts/sear_phase1.py --model ./models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sear.corpus import Corpus
from sear.processor import ExtractiveCopyConstraint, NEG_INF


def _selftest():
    vocab = {w: i for i, w in enumerate(
        ["<pad>", "<eos>", "|", "the", "fox", "ran", "fast", "dog",
         "slept", "all", "day", "and"])}
    inv = {i: w for w, i in vocab.items()}
    tok = lambda b: [vocab[w] for w in b.decode().split()]
    detok = lambda ids: " ".join(inv[i] for i in ids)

    corpus = Corpus.from_texts(
        {"A": "the fox ran fast", "B": "the fox ran and the dog slept all day"},
        tokenize=tok)

    c = ExtractiveCopyConstraint(
        corpus, prompt_len=0, eos_id=vocab["<eos>"],
        delim_id=vocab["|"], min_span_len=3)

    V = len(vocab)
    flat = np.zeros(V, dtype=np.float32)

    def step(generated):
        scores = c(np.array(generated, dtype=np.intc), flat.copy())
        return {inv[i] for i in range(V) if scores[i] > NEG_INF / 2}

    allowed0 = step([])
    assert "the" in allowed0 and "fox" in allowed0
    assert "<eos>" in allowed0 and "|" not in allowed0, allowed0

    seq = [vocab["the"], vocab["fox"], vocab["ran"]]
    allowed1 = step(seq[:1])
    assert allowed1 == {"fox", "dog"}, allowed1
    allowed2 = step(seq[:2])
    assert allowed2 == {"ran"}, allowed2
    allowed3 = step(seq[:3])
    assert {"fast", "and", "|", "<eos>"} == allowed3, allowed3

    c2 = ExtractiveCopyConstraint(
        corpus, 0, vocab["<eos>"], vocab["|"], min_span_len=3)
    for i in range(3):
        c2(np.array(seq[:i], dtype=np.intc), flat.copy())
    c2(np.array(seq, dtype=np.intc), flat.copy())
    c2(np.array(seq + [vocab["|"]], dtype=np.intc), flat.copy())
    segs = c2.finalize()
    ext = [s for s in segs if s.kind == "extract"][0]
    assert detok(ext.token_ids) == "the fox ran", detok(ext.token_ids)
    assert len(ext.sources) == 2, ext.sources
    assert {d for d, _, _ in ext.sources} == {0, 1}

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


def _run(model_path: str):
    from llama_cpp import Llama, LogitsProcessorList

    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=0, verbose=False)
    tok = lambda b: llm.tokenize(b, add_bos=False)
    detok = lambda ids: llm.detokenize(ids).decode("utf-8", errors="replace")

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
