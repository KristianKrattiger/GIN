"""Recalibrate the cheap pipeline from the generated samples.

    venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py

Model-free: reads data/calibration/samples.json, grid-searches thresholds,
reports leave-one-out, and writes thresholds with provenance so a stale artifact
is detectable next time.

Pre-registered: report the number whichever way it moves. More calibration data
reducing accuracy is a real outcome, not a failure — it would mean the previous
39 baked samples were flattering.

STATUS 2026-07-25: this script has been run once, on the 131 curated samples in
data/calibration/samples.json. Its output was REJECTED and NOT shipped —
data/cartographer_thresholds.json deliberately retains the earlier
(baked-39-fixture) threshold values.

Reason: classify_relation() story-gates the contradicts channels on
same_story. All 22 contradicts samples in the 131-sample set are
same_story=False, which makes CONTRADICTS unreachable for them at any
threshold; the 11 rows that ARE same_story get forced to CONTRADICTS
regardless of threshold, and none of those 11 are gold contradicts. That is
33 of 131 rows that cannot be classified correctly no matter where the
thresholds are set.

Direct evidence: on the 40 held-out eval pairs, the retained (baked-39)
thresholds score 0.700; the thresholds this script recalibrated from the
131-sample set score 0.550. See commit c30f910 for the measurement.

Consequence: --write should not be used again until the calibration corpus
contains same-story contradicts pairs (i.e. gold contradicts pairs where
same_story=True actually occur). Until then, running with --write would
silently regress the shipped thresholds.

STATUS 2026-07-26: the precondition above is now SATISFIED, and --write is
STILL not to be used from this session's work.

What changed. Registering node5 in CORPUS_NODES took the calibration corpus to
150 rows including 12 same-story contradicts, so the 33-unclassifiable-rows
problem no longer holds. And combined.py's same-story arm no longer forces
CONTRADICTS on story membership alone -- it now requires per-fact quantity
stance evidence (gin/cartographer/quantity.py), so the "11 rows forced to
CONTRADICTS regardless of threshold" mechanism is gone too. Both halves of the
2026-07-25 reasoning are therefore obsolete as stated.

Why --write is still wrong here. Recalibrating under a pipeline that changed in
the same branch restates the change rather than evaluating it. The threshold
decision was deliberately left to its own spec, to be made with this branch's
numbers in hand rather than folded into them.

Held-out movement under the SHIPPED thresholds, via --score-only (which skips
calibrate()/leave_one_out(), both impractical at n=150): 0.700 at the 131-sample
baseline -> 0.725 after node5 registration -> 0.725 after the stance channel.

The score is unchanged, but not because stance rarely fires on these pairs --
it is non-None on 5 of the 9 same-story held-out pairs, so it fires often. It
is unchanged because all 9 same-story held-out pairs are gold contradicts, and
stance reads "conflict" on 5 of them and None on the other 4 -- "conflict"
types CONTRADICTS through the stance channel, and None types CONTRADICTS
through the pre-stance band fallback, so both paths agree on every one of the
9. See docs/superpowers/specs/2026-07-26-same-story-stance-channel-design.md's
Results section for the full correction.

See docs/superpowers/specs/2026-07-26-same-story-stance-channel-design.md.
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
    load_thresholds,
)
from gin.cartographer.models import Relation

THRESHOLDS_PATH = ROOT / "data" / "cartographer_thresholds.json"
DISPUTED_PAIR = {"inst_em:0", "clim_pledges:0"}


def _score_held_out(eval_samples: list[EvalSample], t: Thresholds) -> float:
    """Fraction of held-out eval pairs the thresholds classify correctly.

    Passes each sample's ``stance`` through to classify_relation so this scores
    the rule the pipeline actually runs, not the pre-stance branch. Verified
    2026-07-26: the score is 0.725 either way on the current 40 held-out pairs
    (9 are same-story; stance is non-None on 5 of those 9, and all 9 are gold
    contradicts, so both the "conflict" and the "None -> band" paths already
    gave the correct answer) -- so this change makes the reported number
    honestly measured without moving it.
    """
    if not eval_samples:
        return float("nan")
    correct = sum(
        classify_relation(
            e.cos, e.p_contra, t, same_story=e.same_story, stance=e.stance
        )[0] == e.relation
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
    ap.add_argument("--score-only", action="store_true",
                    help="score the SHIPPED thresholds on the held-out pairs and exit; "
                         "skips calibrate() and leave_one_out(), which are O(n^4)/O(n^5) "
                         "and impractical past ~150 samples")
    args = ap.parse_args()

    samples, manifest = load_samples(
        args.samples,
        expect_embed_model=DEFAULT_EMBED_MODEL,
        expect_nli_model=DEFAULT_NLI_MODEL,
    )

    if args.score_only:
        # The pre-registered comparison is "what does the SHIPPED pipeline score
        # on the frozen held-out pairs", so this reads thresholds from
        # data/cartographer_thresholds.json rather than recalibrating. Nothing
        # is written and no grid search runs.
        shipped = load_thresholds()
        eval_samples = load_eval_samples(args.samples)
        held_out = _score_held_out(eval_samples, shipped)
        print(f"samples: {len(samples)} {manifest.class_counts}")
        print(f"same_story corpus: {manifest.same_story_corpus_size} docs, "
              f"df_ceiling {manifest.df_ceiling}")
        print(f"shipped thresholds: {shipped}")
        print(f"held-out ({len(eval_samples)} eval pairs, never calibrated on) "
              f"accuracy   {held_out:.3f}")
        return

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
