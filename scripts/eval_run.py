"""
scripts/eval_run.py
Run the RAG vs SEAR designed experiment over a query set and write a report.

Retrieval is held constant across arms; only the generation mechanism varies.
Requires Postgres (retrieval) and a GGUF model (generation). The NLI verifier
downloads a small cross-encoder on first use; use --verifier overlap to avoid.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from llama_cpp import Llama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres
from gin.eval.arms import ArmConfig, build_arm
from gin.eval.queryset import filter_regression_queries, load_query_set
from gin.eval.runner import make_meta, run_experiment, write_run
from gin.eval.verifier import Verifier

DEFAULT_QUERYSET = ROOT / "data" / "eval" / "queryset.yaml"
DEFAULT_OUT = ROOT / "data" / "eval_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Path to local GGUF model")
    parser.add_argument("--queryset", type=str, default=str(DEFAULT_QUERYSET))
    parser.add_argument(
        "--arms",
        type=str,
        default="rag,no_continuation",
        help="Comma-separated arm names (rag, no_continuation, flagged_generation)",
    )
    parser.add_argument("--verifier", type=str, default="nli", choices=["nli", "overlap"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--k-seed", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument(
        "--eval-layer",
        type=str,
        default=None,
        help="Optional retrieval filter; default searches the whole corpus so "
        "out_of_scope probes can genuinely miss.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--chat-template", type=str, default="mistral", choices=["mistral", "plain"])
    parser.add_argument("--no-logit-bias", action="store_true")
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="Run only queries tagged regression: true (9 anchors for CI).",
    )
    parser.add_argument(
        "--boost-gold-chunks",
        action="store_true",
        help="Eval-only: prioritize gold chunks in NC materialization.",
    )
    parser.add_argument(
        "--gold-refuse-without-coverage",
        action="store_true",
        help="Eval-only: NC refuses when no emitted claim cites a gold chunk.",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    queries = load_query_set(args.queryset)
    queries = filter_regression_queries(queries, regression_only=args.regression_only)
    if not queries:
        print(f"[!] No queries loaded from {args.queryset}", file=sys.stderr)
        return 1

    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    filters = {"eval_layer": args.eval_layer} if args.eval_layer else None
    config = ArmConfig(
        k_seed=args.k_seed,
        k_max=args.k_max,
        filters=filters,
        max_tokens=args.max_tokens,
        chat_template=args.chat_template,
        use_logit_bias=not args.no_logit_bias,
        boost_gold_chunks=args.boost_gold_chunks,
        gold_refuse_without_coverage=args.gold_refuse_without_coverage,
    )
    arms = {name: build_arm(name, config) for name in arm_names}

    verifier = Verifier(mode=args.verifier, threshold=args.threshold)

    print(f"[*] Loading model from {args.model}...")
    llm = Llama(
        model_path=args.model,
        n_ctx=2048,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    print(f"[*] Running {len(queries)} queries across arms: {', '.join(arm_names)}")
    t0 = time.perf_counter()
    results_by_arm = run_experiment(queries, arms, llm, verifier)
    elapsed = time.perf_counter() - t0
    n_runs = len(queries) * len(arm_names)
    wall_per_query = elapsed / n_runs if n_runs else None
    total_tokens = sum(
        len(r.raw_text.split()) for rows in results_by_arm.values() for r in rows
    )
    tokens_per_second = (total_tokens / elapsed) if elapsed > 0 else None

    meta = make_meta(
        model=args.model,
        verifier_mode=args.verifier,
        threshold=args.threshold,
        queryset=args.queryset,
        arms=arm_names,
        n_queries=len(queries),
        n_gpu_layers=args.n_gpu_layers,
        wall_clock_seconds_per_query=wall_per_query,
        tokens_per_second=tokens_per_second,
    )
    run_dir = write_run(results_by_arm, meta, args.out)
    print(f"[*] Wrote results to {run_dir}")
    print(f"[*] Report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
