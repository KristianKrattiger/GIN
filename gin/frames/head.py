"""The pair-head: a small scikit-learn estimator plus a gating manifest.

Deliberately not a hand-written torch module. At 80 rows a training loop is pure
surface area for bugs; sklearn gives deterministic fits, LeaveOneOut, and
class_weight="balanced" for free.

Capacity is a liability here, so "linear" is the default and "mlp" exists only
for the case where linear provably underfits.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

HEAD_KINDS: tuple[str, ...] = ("linear", "mlp")
HEAD_FILENAME = "head.joblib"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class Manifest:
    encoder_model: str
    feature_dim: int
    classes: list[str]
    kind: str
    seed: int
    n_train: int
    class_counts: dict[str, int]
    git_sha: str
    created_utc: str
    head_sha256: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Manifest":
        return cls(**data)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_estimator(kind: str, seed: int):
    if kind == "linear":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,), alpha=1.0, max_iter=5000,
            early_stopping=False, random_state=seed,
        )
    raise ValueError(f"unknown head kind: {kind!r} (expected one of {HEAD_KINDS})")


def train_head(X: np.ndarray, y: np.ndarray, kind: str = "linear", seed: int = 0):
    model = build_estimator(kind, seed)
    model.fit(X, y)
    return model


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_head(directory: Path, model, manifest: Manifest) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    head_path = directory / HEAD_FILENAME
    joblib.dump(model, head_path)
    # Hash the bytes actually written to disk, not the in-memory model, so the
    # manifest attests to exactly what a later load_head() will read back.
    digest = _sha256_of(head_path)
    final_manifest = replace(manifest, head_sha256=digest)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(final_manifest.to_json(), indent=2) + "\n", encoding="utf-8"
    )


def load_head(
    directory: Path,
    *,
    expect_encoder: Optional[str] = None,
    expect_dim: Optional[int] = None,
) -> tuple[object, Manifest]:
    """Load head + manifest. Incompatibility is a hard error, never a fallback."""
    directory = Path(directory)
    head_path = directory / HEAD_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    if not head_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"no trained head in {directory}")
    manifest = Manifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
    # Integrity check: catches a joblib/manifest pair left mismatched by a crash
    # between the two writes in save_head (e.g. a retrain that dies mid-write).
    # This is not a compatibility check, so it runs unconditionally.
    actual_digest = _sha256_of(head_path)
    if manifest.head_sha256 != actual_digest:
        raise ValueError(
            "head artifact does not match its manifest (stale or partial write): "
            f"manifest records {manifest.head_sha256!r}, head.joblib hashes to {actual_digest!r}"
        )
    if expect_encoder is not None and manifest.encoder_model != expect_encoder:
        raise ValueError(
            f"encoder mismatch: head trained on {manifest.encoder_model!r}, "
            f"caller supplied {expect_encoder!r}"
        )
    if expect_dim is not None and manifest.feature_dim != expect_dim:
        raise ValueError(
            f"feature dim mismatch: head expects {manifest.feature_dim}, got {expect_dim}"
        )
    return joblib.load(head_path), manifest
