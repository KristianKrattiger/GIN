"""Sweep frozen encoders for recoverable framing divergence (sub-project B's handoff).

    python scripts/frames_encoder_sweep.py
    python scripts/frames_encoder_sweep.py --models sentence-transformers/all-mpnet-base-v2

Downloads each candidate on first use. Exit code 0 if any encoder recovers a
held-out bar issue_frame pair, 1 if none does -- a 1 is a real result, not a
crash: it retires the frozen-encoder path for framing and promotes fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder
from gin.frames.encoder_sweep import (
    CANDIDATE_ENCODERS,
    EncoderResult,
    format_result,
    sweep_encoder,
    verdict,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen-encoder sweep for framing divergence")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--models", nargs="+", default=list(CANDIDATE_ENCODERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bar", action="store_true", help="skip bar scoring (faster)")
    ap.add_argument("--json", type=Path, help="also write results here")
    args = ap.parse_args()

    report = build_dataset(Store(args.log))
    print(f"dataset: {len(report.examples)} examples {report.counts}")
    print(f"drops:   {report.drops}\n")

    results: list[EncoderResult] = []
    for name in args.models:
        try:
            result = sweep_encoder(
                report.examples,
                ChunkEncoder(name),
                seed=args.seed,
                score_bar=not args.no_bar,
            )
        except Exception as exc:  # a model that will not load must not kill the sweep
            result = EncoderResult(name, float("nan"), [], None, error=f"{type(exc).__name__}: {exc}")
        results.append(result)
        print(format_result(result))
        print()

    outcome = verdict(results)
    print(f"VERDICT: {outcome}")
    if outcome == "framing_not_recoverable_frozen":
        print("\nNo frozen encoder recovered a held-out bar issue_frame pair. The")
        print("frozen-geometry path is exhausted for framing divergence; escalate to")
        print("encoder fine-tuning under a separate spec rather than curating more")
        print("framing labels. This null result is the deliverable.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "verdict": outcome,
                    "dataset": {"n": len(report.examples), "counts": report.counts},
                    "results": [
                        {
                            "model": r.model_name,
                            "aggregate_balanced_accuracy": r.aggregate_balanced_accuracy,
                            "by_origin": [
                                {
                                    "origin": o.origin,
                                    "n": o.n,
                                    "n_recovered": o.n_recovered,
                                    "recall": o.recall,
                                    "decisive": o.decisive,
                                    "missed": o.chunk_pairs,
                                }
                                for o in r.by_origin
                            ],
                            "bar": r.bar,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0 if outcome.startswith("framing_recoverable") else 1


if __name__ == "__main__":
    sys.exit(main())
