#!/usr/bin/env python3
"""Run one real SEAR proposer pass and check every quote is copied from the right role.

    python scripts/assay_proposer_smoke.py --model models/Qwen2.5-7B-Instruct-Q6_K.gguf

This is the only check of slots.prompt_token_count against a real tokenizer:
if llama.cpp counts the prompt differently, the first constrained slot's
_PromptLengthGuard raises, _Loud stops decoding right there, and _complete
re-raises once create_completion returns -- instead of the mismatch silently
misattributing a quote. Exit 0 = every quote verified; 1 = a quote
failed; 2 = the model proposed nothing, so nothing was checked.
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
                "Eight external cameras provide a 360-degree view of the environment around the vehicle.",
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

    from gin.assay_proposer.runtime import load_model
    from gin.assay_proposer.schema import Excerpt
    from gin.assay_proposer.slots import propose_pass

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
    failures = []
    for p in proposals:
        if roles.get(p["from"]["docId"]) != "claimant":
            failures.append(f"from is not a claimant doc: {p['from']}")
        elif not any(p["from"]["quote"] in line for line in lines[p["from"]["docId"]]):
            failures.append(f"from quote not copied from its doc: {p['from']}")
        if p["to"] is not None:
            if roles.get(p["to"]["docId"]) != "independent":
                failures.append(f"to is not an independent doc: {p['to']}")
            elif not any(p["to"]["quote"] in line for line in lines[p["to"]["docId"]]):
                failures.append(f"to quote not copied from its doc: {p['to']}")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    if not proposals:
        print("the model proposed nothing; nothing was checked")
        return 2
    print(f"OK: {len(proposals)} proposal(s) in {seconds}s, every quote copied from a line of the right role")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
