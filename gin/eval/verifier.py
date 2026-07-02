"""Claim grounding verifier.

Decides whether an emitted claim is entailed by any retrieved chunk. Two
backends:

- ``nli``     a dedicated entailment cross-encoder (via sentence-transformers).
              Premise = chunk text, hypothesis = claim; score = P(entailment).
- ``overlap`` a token-containment heuristic: fraction of the claim's content
              words that appear in the chunk. No model, cheap, cruder.

The verifier returns the best-matching chunk and its confidence so every
claim record can carry ``matched_chunk_id`` + ``score``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .claims import Verdict

# Default entailment model: small, CPU-friendly, 3-class NLI.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
# Label order for the cross-encoder/nli-* family.
DEFAULT_NLI_LABELS = ("contradiction", "entailment", "neutral")

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with""".split()
)

# (chunk_id, text) pair.
Chunk = tuple[str, str]
Scorer = Callable[[str, str], float]


@dataclass
class ClaimVerdict:
    verdict: str
    matched_chunk_id: Optional[str]
    score: float


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def token_overlap(claim_text: str, chunk_text: str) -> float:
    """Fraction of the claim's content words contained in the chunk."""
    claim_words = _content_words(claim_text)
    if not claim_words:
        return 0.0
    chunk_words = set(_content_words(chunk_text))
    hits = sum(1 for w in claim_words if w in chunk_words)
    return hits / len(claim_words)


def max_query_overlap(query: str, chunks: Sequence[Chunk]) -> float:
    """Best token-overlap between a query and any retrieved chunk."""
    if not chunks:
        return 0.0
    return max(token_overlap(query, text) for _, text in chunks)


class Verifier:
    """Score claim grounding against candidate chunks.

    Parameters
    ----------
    mode:
        ``"nli"`` or ``"overlap"``.
    threshold:
        Minimum score for a SUPPORTED verdict.
    scorer:
        Optional injected ``(claim, chunk) -> float`` callable. When provided it
        overrides the backend entirely (used in tests and to plug in custom
        verifiers). If a claim/chunk score reaches ``threshold`` the claim is
        SUPPORTED.
    model_name / entail_index:
        NLI cross-encoder identifier and the index of the entailment label in
        the model's output.
    """

    def __init__(
        self,
        *,
        mode: str = "nli",
        threshold: float = 0.5,
        scorer: Optional[Scorer] = None,
        model_name: str = DEFAULT_NLI_MODEL,
        entail_index: int = 1,
    ) -> None:
        if mode not in {"nli", "overlap"}:
            raise ValueError(f"unknown verifier mode: {mode!r}")
        self.mode = mode
        self.threshold = threshold
        self.model_name = model_name
        self.entail_index = entail_index
        self._scorer = scorer
        self._model = None  # lazy-loaded CrossEncoder

    # -- scoring backends ---------------------------------------------------

    def _nli_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self.model_name)
            self._resolve_nli_entail_index()
        return self._model

    def _resolve_nli_entail_index(self) -> None:
        """Align entail_index with the loaded model's label order when known."""
        model = self._model
        if model is None:
            return
        labels = getattr(model, "config", None)
        id2label = getattr(labels, "id2label", None) if labels else None
        if not id2label:
            return
        for idx, label in id2label.items():
            if str(label).lower() == "entailment":
                self.entail_index = int(idx)
                return

    def _nli_entailment_prob(self, row) -> float:
        """Map CrossEncoder predict output to P(entailment)."""
        import numpy as np

        values = np.asarray(row, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return 0.0
        if values.size == 1:
            # Single-label relevance models: score is already in [0, 1].
            return float(np.clip(values[0], 0.0, 1.0))
        idx = min(self.entail_index, values.size - 1)
        if np.all((values >= 0.0) & (values <= 1.0)) and abs(values.sum() - 1.0) < 0.05:
            return float(values[idx])
        exp = np.exp(values - values.max())
        probs = exp / exp.sum()
        return float(probs[idx])

    def _nli_score(self, claim_text: str, chunk_text: str) -> float:
        model = self._nli_model()
        raw = model.predict(
            [(chunk_text, claim_text)],
            apply_softmax=True,
            convert_to_numpy=True,
        )
        row = raw[0] if getattr(raw, "ndim", 0) == 2 else raw
        return self._nli_entailment_prob(row)

    def _score(self, claim_text: str, chunk_text: str) -> float:
        if self._scorer is not None:
            return float(self._scorer(claim_text, chunk_text))
        if self.mode == "overlap":
            return token_overlap(claim_text, chunk_text)
        return self._nli_score(claim_text, chunk_text)

    # -- public API ---------------------------------------------------------

    def verify(self, claim_text: str, chunks: Sequence[Chunk]) -> ClaimVerdict:
        """Return the best-supported chunk for ``claim_text`` (or UNSUPPORTED)."""
        best_score = 0.0
        best_id: Optional[str] = None
        for chunk_id, text in chunks:
            score = self._score(claim_text, text)
            if score > best_score:
                best_score = score
                best_id = chunk_id
        if best_id is not None and best_score >= self.threshold:
            return ClaimVerdict(Verdict.SUPPORTED.value, best_id, best_score)
        return ClaimVerdict(Verdict.UNSUPPORTED.value, None, best_score)
