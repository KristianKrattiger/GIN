"""
scripts/edge_degradation.py
Step 2 artifact: decode-in-the-loop degradation under a noisy (class-C) edge.

Runs the clean / noisy / control scenarios through the real materialize +
constrained-decode path and writes a report + JSON under data/eval_runs. Needs a
GGUF model but no Postgres (bundles are constructed in memory from real corpus
text). With --deterministic, uses GreedyMaskDecoder instead of a model — the
answers are byte-identical, so this is a fast no-model artifact.

See docs/nc_reasoning_robustness_noisy_edges.plan.md §4.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.eval.edge_degradation import (
    GreedyMaskDecoder,
    format_degradation_report,
    run_degradation,
)

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="Path to local GGUF model")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use GreedyMaskDecoder instead of a model (byte-identical answers).",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if args.deterministic or not args.model:
        model_label = "GreedyMaskDecoder (deterministic)"
        llm = GreedyMaskDecoder()
    else:
        from llama_cpp import Llama

        print(f"[*] Loading model from {args.model}...")
        model_label = args.model
        llm = Llama(
            model_path=args.model,
            n_ctx=2048,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )

    results = run_degradation(llm)
    report = format_degradation_report(results)
    print(report)

    run_dir = Path(args.out) / (
        "edge_degradation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps(
            {"model": model_label, "results": [r.to_dict() for r in results]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[*] Wrote artifact to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
