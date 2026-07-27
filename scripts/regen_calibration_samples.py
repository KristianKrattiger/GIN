"""Measure calibration samples from the curator store. Model-bound; run rarely.

    venv/Scripts/python.exe scripts/regen_calibration_samples.py

Loads embed + NLI models, so this is the ONE place in the calibration path that
touches a model. Everything downstream reads the file it writes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.cartographer.calibration_samples import (
    DEFAULT_SAMPLES_PATH,
    EvalSample,
    Sample,
    SampleManifest,
    write_samples,
)
from gin.cartographer.combined import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_NLI_MODEL,
    CombinedRelationProposer,
)
from gin.cartographer.models import Relation
from gin.cartographer.quantity import stance_for
from gin.cartographer.relatedness import (
    DEFAULT_STORY_FLOOR,
    _rare_df_ceiling,
    make_same_story,
)
from gin.curator.calibration_export import export_calibration_rows
from gin.curator.store import Store
from gin.curator.text_index import default_text_index

DEFAULT_LABELS = ROOT / "data" / "curator" / "labels.jsonl"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate calibration samples")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_SAMPLES_PATH)
    args = ap.parse_args()

    text = default_text_index()
    corpus_texts = list(text.values())
    story_floor = DEFAULT_STORY_FLOOR
    df_ceiling = _rare_df_ceiling(len(corpus_texts))
    require_anchor = True
    proposer = CombinedRelationProposer()
    same_story = make_same_story(
        corpus_texts,
        story_floor=story_floor,
        df_ceiling=df_ceiling,
        require_anchor=require_anchor,
    )

    def signals(a_text: str, b_text: str) -> tuple[float, float, bool, Optional[str]]:
        story = same_story(a_text, b_text)
        return (
            proposer.embedding_cosine(a_text, b_text),
            proposer._p_contra(a_text, b_text),  # noqa: SLF001 - same scorer the classifier uses
            story,
            # Only same-story pairs reach the stance arm, so only they need it
            # measured. Mirrors CombinedRelationProposer.type_relation.
            stance_for(a_text, b_text) if story else None,
        )

    report = export_calibration_rows(Store(args.log), signals, text_index=text)
    samples = [
        Sample(
            cos=r["cos"], p_contra=r["p_contra"],
            relation=Relation(r["relation"]), same_story=r["same_story"],
            stance=r.get("stance"),
        )
        for r in report.rows
    ]
    eval_samples = [
        EvalSample(
            src=r["src"], dst=r["dst"], cos=r["cos"], p_contra=r["p_contra"],
            relation=Relation(r["relation"]), same_story=r["same_story"],
            stance=r.get("stance"),
        )
        for r in report.eval_rows
    ]
    manifest = SampleManifest(
        embed_model=DEFAULT_EMBED_MODEL,
        nli_model=DEFAULT_NLI_MODEL,
        n_samples=len(samples),
        class_counts=report.class_counts,
        excluded_eval_pairs=report.drops.get("eval_pair", 0),
        git_sha=git_sha(),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        same_story_corpus_size=len(corpus_texts),
        story_floor=story_floor,
        df_ceiling=df_ceiling,
        require_anchor=require_anchor,
        stance_provider="quantity.stance_for",
    )
    write_samples(args.out, manifest, samples, eval_samples)
    print(f"measured {len(samples)} calibration samples {report.class_counts}")
    print(f"measured {len(eval_samples)} held-out eval samples")
    print(f"drops: {report.drops}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
