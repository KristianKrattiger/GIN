"""Drop-in FrameJudge backed by the trained pair-head.

Satisfies the same (a_text, b_text) -> {DIVERGENT, AGREE, UNRELATED} contract
the LLM judges used, so evaluate_escalation_judge scores it unchanged and the
comparison against the 2026-07-13 sweep is apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .encoder import ChunkEncoder, pair_features
from .head import load_head
from .labels import JUDGE_LABEL, FrameClass


class BiEncoderFrameJudge:
    """Frozen embeddings -> symmetric pair features -> head -> 3-way label."""

    def __init__(self, model, encoder: ChunkEncoder) -> None:
        self.model = model
        self.encoder = encoder

    def __call__(self, a_text: str, b_text: str) -> str:
        features = pair_features(
            self.encoder.encode(a_text), self.encoder.encode(b_text)
        ).reshape(1, -1)
        predicted = str(self.model.predict(features)[0])
        return JUDGE_LABEL[FrameClass(predicted)]


def load_judge(directory: Path, encoder: Optional[ChunkEncoder] = None) -> BiEncoderFrameJudge:
    """Load a trained head, verifying it matches the encoder it was trained on."""
    encoder = ChunkEncoder() if encoder is None else encoder
    model, _manifest = load_head(directory, expect_encoder=encoder.model_name)
    return BiEncoderFrameJudge(model, encoder)
