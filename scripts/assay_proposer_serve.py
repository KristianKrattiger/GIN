#!/usr/bin/env python3
"""Serve the SEAR-constrained Assay proposer on localhost.

Usage:
    python scripts/assay_proposer_serve.py --model models/Qwen2.5-7B-Instruct-Q6_K.gguf
    python scripts/assay_proposer_serve.py --model ... --n-gpu-layers -1   # on a GPU rig

Receipts reaches it with --client sear, SEAR_HOST=http://127.0.0.1:8766 and
SEAR_MODEL set to the id printed at startup (sear/<gguf file stem>). Binds
127.0.0.1 only: a local sidecar with no peers, so no mTLS.

app.py's propose lock serializes passes but does not cancel one already in
flight: if Receipts is interrupted mid-request, a queued or in-progress pass
keeps decoding under the lock until it finishes on its own -- there is
nothing here to stop it early. Restarting the sidecar is what clears it.

--n-ctx may need raising for large passes: a long excerpt set plus a full
proposal turn can outgrow the default context window, and llama.cpp handles
that two different ways depending on how it's overgrown. A prompt that is
already too long on its own fails loudly: create_completion raises
ValueError("Requested tokens ... exceed context window ...") before decoding
starts, which comes back to Receipts as a failed pass with that error as the
reason given. A prompt that fits but would run past n_ctx before max_tokens
is reached is not rejected -- llama.cpp silently clamps max_tokens down to
what's left instead. There the decode does not fail outright: a quote slot
just finishes early with finish_reason "length" and _quote returns None (on
the from-slot this ends the pass with no error at all), and a choice slot can
come back empty and fail with "choice decode produced '', not one of [...]".
Both are symptoms of the same silent truncation, not of a bug in the slot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a GGUF")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--n-gpu-layers", type=int, default=0, help="0 = CPU (GIN's default); -1 = all on GPU")
    ap.add_argument("--n-ctx", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-proposals", type=int, default=8)
    args = ap.parse_args()

    import uvicorn

    from gin.assay_proposer.app import create_app
    from gin.assay_proposer.runtime import load_model
    from gin.assay_proposer.slots import propose_pass

    llm, render, model_id = load_model(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)

    def propose_fn(req):
        user = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
        if user is None:
            raise ValueError("request carries no user message")
        return propose_pass(
            llm, render, system=req.system, user=user, excerpts=req.excerpts,
            temperature=args.temperature, max_proposals=args.max_proposals,
        )

    print(f"assay proposer: serving {model_id} on http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(propose_fn, model_id), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
