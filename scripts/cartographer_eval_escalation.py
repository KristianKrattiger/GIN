"""Calibrate an escalation frame judge: issue_frame gold + expanded controls.

Pass bar: issue_frame_recall 1.0 AND class_c_discrimination 1.0 AND
unrelated_discrimination 1.0, with mixed labels on the 33-pair labeled set.
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

from gin.cartographer.escalation import resolve_escalation_judge
from gin.cartographer.escalation_eval import (
    default_calibration_sets,
    evaluate_escalation_judge,
    labeled_set_pairs,
)
from gin.cartographer.scan import chunks_from_db
from gin.corpus.db import DatabaseUnavailableError, connect, ensure_postgres

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge",
        required=True,
        metavar="BACKEND[:SPEC]",
        help="Escalation judge spec (local:model.gguf, anthropic:model, etc.)",
    )
    parser.add_argument(
        "--escalation-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers for local judge (-1 = all)",
    )
    parser.add_argument(
        "--escalation-n-ctx",
        type=int,
        default=4096,
        help="Context window for local judge",
    )
    parser.add_argument(
        "--skip-labeled-set",
        action="store_true",
        help="Skip the 33-pair in-memory breadth block (faster run)",
    )
    parser.add_argument(
        "--single-direction",
        action="store_true",
        help="Judge each pair in the listed direction only (no flip diagnostic)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        judge, method_suffix = resolve_escalation_judge(
            args.judge,
            n_ctx=args.escalation_n_ctx,
            n_gpu_layers=args.escalation_gpu_layers,
        )
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    with connect() as conn:
        chunks, _outlets = chunks_from_db(conn)
    text_by_chunk = {ch.chunk_id: ch.text for ch in chunks}

    sets = default_calibration_sets()
    metrics = evaluate_escalation_judge(
        judge,
        text_by_chunk,
        issue_frame_pairs=sets["issue_frame"],
        corroboration_pairs=sets["corroboration"],
        unrelated_pairs=sets["unrelated"],
        labeled_pairs=None if args.skip_labeled_set else labeled_set_pairs(),
        both_directions=not args.single_direction,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "escalation_judge_metrics.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": ts,
        "judge_spec": args.judge,
        "method_suffix": method_suffix,
        **metrics,
    }
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[*] Escalation judge calibration — {ts}")
    print(f"    backend: {method_suffix}")
    print(f"    issue_frame recall: {metrics['issue_frame_recall']}")
    print(f"    class_c_discrimination: {metrics['class_c_discrimination']}")
    print(f"    unrelated_discrimination: {metrics['unrelated_discrimination']}")
    print(f"    direction_flip_count: {metrics['direction_flip_count']}")
    print(f"    label_distribution: {metrics['label_distribution']}")
    if "labeled_set" in metrics:
        ls = metrics["labeled_set"]
        by = {
            k: f"{v['correct']}/{v['total']}" for k, v in ls["by_expected"].items()
        }
        print(f"    labeled_set accuracy: {ls['accuracy']:.3f} ({by})")
    print(f"    artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
