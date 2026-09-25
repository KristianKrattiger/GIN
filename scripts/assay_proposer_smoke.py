#!/usr/bin/env python3
"""Run one real SEAR proposer pass and check every quote is copied from the right role.

    python scripts/assay_proposer_smoke.py --model models/Qwen2.5-7B-Instruct-Q6_K.gguf

This is the only check of slots.prompt_token_count against a real tokenizer:
if llama.cpp counts the prompt differently, the first constrained slot's
_PromptLengthGuard raises, _Loud stops decoding right there, and _complete
re-raises once create_completion returns -- instead of the mismatch silently
misattributing a quote. Exit 0 = every quote verified; 1 = a quote
failed; 2 = the model proposed nothing, so nothing was checked; 3 = every
quote verified, but one quote pair got more than one relation type.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEM = "You compare a vendor's own claims against independent reports about that vendor."
EXCERPTS = [
    {
        "docId": "tesla", "role": "claimant", "label": "Tesla Vehicle Safety Report",
        "text": "Tesla vehicles with FSD (Supervised) engaged experience fewer collisions than those driven without.\n"
                "Eight external cameras provide a 360-degree view of the environment around the vehicle.\n"
                "Full Self-Driving is available across North America, Europe and parts of Asia today. "
                "The system does not yet operate reliably in heavy snow or dense fog. "
                "Tesla plans to widen availability further next year.",
    },
    {
        "docId": "hn", "role": "independent", "label": "Hacker News - FSD",
        "text": "Tesla Full Self Driving requires human intervention every 13 miles\n"
                "Tesla 'Full Self-Driving' crashed through railroad gate seconds before train",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=4096)
    args = ap.parse_args()

    from gin.assay_proposer.consistency import conflicting_pairs
    from gin.assay_proposer.corpus import MAX_QUOTE_WORDS
    from gin.assay_proposer.runtime import load_model
    from gin.assay_proposer.schema import Excerpt
    from gin.assay_proposer.slots import propose_pass
    from sear.corpus import SENTENCE_BOUNDARY

    llm, render, model_id = load_model(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    excerpts = [Excerpt(**e) for e in EXCERPTS]
    user = (
        "Subject: Tesla FSD\nTask: propose contradicts, corroborates and updates relations between the "
        "claimant excerpts and the independent excerpts below.\n\nExcerpts:\n\n"
        + "\n\n".join(f"--- docId: {e.docId} | role: {e.role} | source: {e.label}\n{e.text}" for e in excerpts)
    )

    t0 = time.time()
    proposals = propose_pass(llm, render, system=SYSTEM, user=user, excerpts=excerpts,
                             temperature=0.8, max_proposals=2)
    seconds = round(time.time() - t0, 1)
    print(json.dumps({"model": model_id, "seconds": seconds, "proposals": proposals}, indent=2))

    lines = {e.docId: e.text.split("\n") for e in excerpts}
    roles = {e.docId: e.role for e in excerpts}

    def line_sentences(line: str) -> list[str]:
        return [s.strip() for s in SENTENCE_BOUNDARY.split(line) if s.strip()]

    def matching_line(quote: str, doc_lines: list[str]) -> str | None:
        return next((line for line in doc_lines if quote in line), None)

    failures = []
    for p in proposals:
        for field, want_role in (("from", "claimant"), ("to", "independent")):
            span = p[field]
            if span is None:
                continue
            if roles.get(span["docId"]) != want_role:
                failures.append(f"{field} is not a {want_role} doc: {span}")
                continue
            line = matching_line(span["quote"], lines[span["docId"]])
            if line is None:
                failures.append(f"{field} quote not copied from its doc: {span}")
                continue
            if len(span["quote"].split()) > MAX_QUOTE_WORDS:
                failures.append(f"{field} quote is over {MAX_QUOTE_WORDS} words: {span}")
            if span["quote"].strip() not in line_sentences(line):
                failures.append(f"{field} quote is not exactly one sentence of its line: {span}")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    if not proposals:
        print("the model proposed nothing; nothing was checked")
        return 2
    conflicts = conflicting_pairs(proposals)
    if conflicts:
        print("CONFLICT: quotes verified, but a pair got more than one relation type:\n  " + "\n  ".join(
            f"{c['types']} on {c['from']['quote']!r} -> {c['to']['quote']!r}" for c in conflicts))
        return 3
    print(f"OK: {len(proposals)} proposal(s) in {seconds}s, every quote copied from a line of the right role")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
