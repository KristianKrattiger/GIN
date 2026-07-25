"""Score the trained head: bar + leave-one-out + baseline table.

    python scripts/frames_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder, feature_matrix
from gin.frames.eval import (
    BAR_METRIC_KEYS,
    PUBLISHED_BASELINES,
    bar_all_green,
    bar_metrics,
    decide,
    loo_report,
)
from gin.frames.judge import load_judge

DEFAULT_HEAD = Path("data/frames")
DEFAULT_OUT = Path("data/eval_runs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the bi-encoder frame detector")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    ap.add_argument("--kind", default="linear")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    encoder = ChunkEncoder()
    report = build_dataset(Store(args.log))
    X, y = feature_matrix(report.examples, encoder)

    judge = load_judge(args.head, encoder=encoder)
    bar = bar_metrics(judge)
    loo = loo_report(X, y, kind=args.kind)
    verdict = decide(bar, loo["balanced_accuracy_mean"])

    print("=== escalation bar ===")
    for key in BAR_METRIC_KEYS:
        print(f"  {key:28s} {bar.get(key)}")
    print(f"  ALL GREEN: {bar_all_green(bar)}")

    print("\n=== leave-one-out (honest generalization) ===")
    print(f"  n                      {loo['n']}")
    print(f"  balanced acc (mean)    {loo['balanced_accuracy_mean']:.3f}")
    print(f"  spread across seeds    {loo['balanced_accuracy_spread']:.3f}")
    for name, value in loo["per_class_recall"].items():
        print(f"  recall {name:18s} {value:.3f}")

    print("\n=== baselines (2026-07-13 sweep) ===")
    for row in PUBLISHED_BASELINES:
        print(f"  {row['model']:22s} recall {row['issue_frame_recall']:.2f}  "
              f"class_c {row['class_c_discrimination']:.2f}  "
              f"unrel {row['unrelated_discrimination']:.2f}  "
              f"flips {row['direction_flip_count']}")

    print(f"\nVERDICT: {verdict}")
    if verdict == "suspect":
        print("Bar is green but cross-validation is at chance. Report as overfit")
        print("or lucky — do NOT ship this as a win.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bar": {k: bar.get(k) for k in BAR_METRIC_KEYS},
        "bar_all_green": bar_all_green(bar),
        "loo": loo,
        "baselines": list(PUBLISHED_BASELINES),
        "verdict": verdict,
        "dataset_counts": report.counts,
        "dataset_drops": report.drops,
    }
    (run_dir / "frame_detector_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {run_dir}/frame_detector_metrics.json")


if __name__ == "__main__":
    main()
