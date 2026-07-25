"""Train the pair-head and write head.joblib + manifest.json.

    python scripts/frames_train.py --kind linear
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder, feature_matrix
from gin.frames.head import HEAD_KINDS, Manifest, git_sha, save_head, train_head

DEFAULT_OUT = Path("data/frames")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the bi-encoder pair-head")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kind", choices=HEAD_KINDS, default="linear")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = build_dataset(Store(args.log))
    encoder = ChunkEncoder()
    X, y = feature_matrix(report.examples, encoder)
    model = train_head(X, y, kind=args.kind, seed=args.seed)

    manifest = Manifest(
        encoder_model=encoder.model_name,
        feature_dim=int(X.shape[1]),
        classes=sorted(set(y.tolist())),
        kind=args.kind,
        seed=args.seed,
        n_train=len(report.examples),
        class_counts=report.counts,
        git_sha=git_sha(),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    save_head(args.out, model, manifest)
    print(f"trained {args.kind} head on {manifest.n_train} rows {manifest.class_counts}")
    print(f"wrote {args.out}/head.joblib + manifest.json")


if __name__ == "__main__":
    main()
