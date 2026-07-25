"""Recalibrate the cheap pipeline from the generated samples.

    venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py

Model-free: reads data/calibration/samples.json, grid-searches thresholds,
reports leave-one-out, and writes thresholds with provenance so a stale artifact
is detectable next time.

Pre-registered: report the number whichever way it moves. More calibration data
reducing accuracy is a real outcome, not a failure — it would mean the previous
39 baked samples were flattering.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from gin.cartographer.calibration import calibrate, leave_one_out
from gin.cartographer.calibration_samples import (
    DEFAULT_SAMPLES_PATH,
    EvalSample,
    load_eval_samples,
    load_samples,
)
from gin.cartographer.combined import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_NLI_MODEL,
    Thresholds,
    classify_relation,
)
from gin.cartographer.models import Relation

THRESHOLDS_PATH = ROOT / "data" / "cartographer_thresholds.json"
DISPUTED_PAIR = {"inst_em:0", "clim_pledges:0"}


def _score_held_out(eval_samples: list[EvalSample], t: Thresholds) -> float:
    """Fraction of held-out eval pairs the thresholds classify correctly."""
    if not eval_samples:
        return float("nan")
    correct = sum(
        classify_relation(e.cos, e.p_contra, t, same_story=e.same_story)[0] == e.relation
        for e in eval_samples
    )
    return correct / len(eval_samples)

# Measured 2026-07-25 on the 39 baked samples, before this recalibration.
BASELINE = {"n_samples": 39, "gate_floor": 0.140, "corroborate_ceiling": 0.486,
            "contra_threshold": 0.686, "loo_accuracy": 0.897, "loo_class_c": 1.000}


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Recalibrate cheap-pipeline thresholds")
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    ap.add_argument("--out", type=Path, default=THRESHOLDS_PATH)
    ap.add_argument("--write", action="store_true",
                    help="write the thresholds file (default: report only)")
    args = ap.parse_args()

    samples, manifest = load_samples(
        args.samples,
        expect_embed_model=DEFAULT_EMBED_MODEL,
        expect_nli_model=DEFAULT_NLI_MODEL,
    )
    thresholds = calibrate(samples)
    loo = leave_one_out(samples)

    print(f"samples: {len(samples)} {manifest.class_counts}")
    print(f"excluded eval pairs: {manifest.excluded_eval_pairs}")
    print()
    print("=== thresholds ===")
    print(f"  {'':22s} {'baseline(39)':>14s} {'recalibrated':>14s}")
    for name, base_key in (("gate_floor", "gate_floor"),
                           ("corroborate_ceiling", "corroborate_ceiling"),
                           ("contra_threshold", "contra_threshold")):
        print(f"  {name:22s} {BASELINE[base_key]:14.3f} {getattr(thresholds, name):14.3f}")
    print()
    print("=== leave-one-out ===")
    print(f"  {'accuracy':22s} {BASELINE['loo_accuracy']:14.3f} {loo.accuracy:14.3f}")
    cc = loo.class_c_discrimination
    cc_s = f"{cc:14.3f}" if cc is not None else f"{'n/a':>14s}"
    print(f"  {'class_c_discrimination':22s} {BASELINE['loo_class_c']:14.3f} {cc_s}")
    prec, rec = loo.contradicts_precision, loo.contradicts_recall
    print(f"  contradicts precision  {prec if prec is None else round(prec, 3)}")
    print(f"  contradicts recall     {rec if rec is None else round(rec, 3)}")

    eval_samples = load_eval_samples(args.samples)
    held_out = _score_held_out(eval_samples, thresholds)
    print()
    print(f"=== held-out ({len(eval_samples)} eval pairs, never calibrated on) ===")
    print(f"  accuracy               {held_out:14.3f}")

    print()
    print("=== disputed pair sensitivity ===")
    print("  inst_em:0 <-> clim_pledges:0 is a labeled_set member, hence an EVAL")
    print("  pair excluded from calibration — flipping it cannot move the")
    print("  thresholds. It moves the held-out score only.")
    flipped = [
        replace(e, relation=Relation.CONTRADICTS)
        if {e.src, e.dst} == DISPUTED_PAIR
        else e
        for e in eval_samples
    ]
    if flipped == eval_samples:
        print("  (pair not present in the eval samples — nothing to flip)")
    else:
        alt = _score_held_out(flipped, thresholds)
        print(f"  held-out as corroborates (current)  {held_out:.3f}")
        print(f"  held-out as contradicts             {alt:.3f}")
        print(f"  cost of adjudicating it to contradicts: {alt - held_out:+.3f}")

    if args.write:
        payload = {
            "gate_floor": thresholds.gate_floor,
            "corroborate_ceiling": thresholds.corroborate_ceiling,
            "contra_threshold": thresholds.contra_threshold,
            "n_samples": len(samples),
            "leave_one_out_accuracy": round(loo.accuracy, 4),
            "leave_one_out_class_c_discrimination": None if cc is None else round(cc, 4),
            "held_out_accuracy": round(held_out, 4),
            "held_out_n": len(eval_samples),
            "embed_model": manifest.embed_model,
            "nli_model": manifest.nli_model,
            "git_sha": git_sha(),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    else:
        print("\n(report only — pass --write to update the thresholds file)")


if __name__ == "__main__":
    main()
