#!/usr/bin/env python3
"""Serve the SEAR-constrained Assay proposer on localhost.

Usage:
    python scripts/assay_proposer_serve.py --model models/Qwen2.5-7B-Instruct-Q6_K.gguf
    python scripts/assay_proposer_serve.py --model ... --n-gpu-layers -1   # on a GPU rig

Receipts reaches it with --client sear, SEAR_HOST=http://127.0.0.1:8766 and
SEAR_MODEL set to the id printed at startup (sear/<gguf file stem>). Binds
127.0.0.1 only: a local sidecar with no peers, so no mTLS.
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
