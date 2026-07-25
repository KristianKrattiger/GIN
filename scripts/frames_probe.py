"""Stage-0 gate: can a linear model recover the stance axis?

    python scripts/frames_probe.py

Loads the real encoder (downloads on first run). Exit code 0 on pass or
inconclusive, 1 on fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder, feature_matrix
from gin.frames.probe import PROBE_FLOOR, PROBE_PASS, run_probe


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen-geometry separability probe")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = build_dataset(Store(args.log))
    print(f"dataset: {len(report.examples)} examples {report.counts}")
    print(f"drops:   {report.drops}")

    X, y = feature_matrix(report.examples, ChunkEncoder())
    result = run_probe(X, y, seed=args.seed)

    print(f"\nDIVERGENT-vs-rest, leave-one-out over n={result.n} ({result.n_positive} positive)")
    print(f"  balanced accuracy : {result.balanced_accuracy:.3f}")
    print(f"  stratified random : {result.baseline:.3f}")
    print(f"  bands             : fail < {PROBE_FLOOR} <= inconclusive < {PROBE_PASS} <= pass")
    print(f"  VERDICT           : {result.verdict.upper()}")
    if result.verdict == "fail":
        print("\nThe frozen geometry has no recoverable stance axis. Do not add")
        print("capacity to rescue this — escalate to encoder fine-tuning under a")
        print("separate spec. This null result is the deliverable.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
