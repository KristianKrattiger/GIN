"""Frozen sentence embeddings and the order-invariant pair representation.

The encoder is never fine-tuned, so embeddings stay precomputable and shared
with the cheap pipeline (same model as combined.py).

pair_features is symmetric in a/b by construction. That is a design commitment,
not an implementation detail: it makes judge(a,b) == judge(b,a) an identity, so
direction_flip_count = 0 without training for it. Every model in the 2026-07-13
sweep flipped on 3-7 of 14 pairs.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from gin.cartographer.combined import DEFAULT_EMBED_MODEL

from .dataset import FrameExample


class ChunkEncoder:
    """Lazily-loaded frozen encoder with a per-text cache.

    Pass ``encode_fn`` to run model-free (tests, CI).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        encode_fn: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> None:
        self.model_name = model_name
        self._encode_fn = encode_fn
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if self._encode_fn is not None:
            vec = np.asarray(self._encode_fn(text), dtype=np.float64)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            vec = np.asarray(
                self._model.encode([text], normalize_embeddings=True)[0],
                dtype=np.float64,
            )
        self._cache[text] = vec
        return vec


def pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric pair representation: [|a-b|, a*b, (a+b)/2]."""
    return np.concatenate([np.abs(a - b), a * b, (a + b) / 2.0])


def feature_matrix(
    examples: list[FrameExample], encoder: ChunkEncoder
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for the given examples; y holds FrameClass *values* as strings."""
    X = np.vstack(
        [
            pair_features(encoder.encode(e.src_text), encoder.encode(e.dst_text))
            for e in examples
        ]
    )
    y = np.array([e.label.value for e in examples])
    return X, y
