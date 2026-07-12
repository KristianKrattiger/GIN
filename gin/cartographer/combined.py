"""Combined register-robust relation detector.

The two single-signal probes (§6) were complementary, not redundant: NLI owns
propositional contradiction (legal/securities register), the framing signal owns
value/emphasis divergence (climate/housing). Sentence-embedding cosine, measured
across the 13-pair set, separates the three relation classes into bands —
unrelated low (≤0.12), framing-divergent middle (0.13–0.42), corroborating high
(≥0.49) — and where cosine cannot separate (a real legal contradiction is highly
similar), NLI covers exactly that gap. This detector composes them:

    1. embedding relatedness gate   cos < gate_floor           -> UNRELATED
    2. NLI propositional channel     p_contra >= contra_thresh  -> CONTRADICTS
    3. cosine aspect band            cos >= corroborate_ceiling -> CORROBORATES
                                     else (related, mid-band)   -> CONTRADICTS

The NLI channel has priority over the band so a propositional contradiction that
is also highly similar (legal) is not misread as corroboration.

Both signals are injectable ((a,b)->cosine and (premise,hypothesis)->(c,e,n)), so
the composition is testable without models. Thresholds are calibrated on the
13-pair set — too small to be production values; the architecture is the
contribution, the thresholds await a larger labeled set (design §6).
See docs/nc_cartographer_design.plan.md.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .models import Assessment, LabeledChunk, Relation

CosineScorer = Callable[[str, str], float]
# (premise, hypothesis) -> (p_contradiction, p_entailment, p_neutral)
NliScorer = Callable[[str, str], tuple[float, float, float]]

DEFAULT_GATE_FLOOR = 0.13
DEFAULT_CORROBORATE_CEILING = 0.45
DEFAULT_CONTRA_THRESHOLD = 0.5
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class CombinedRelationProposer:
    name = "combined_relation"

    def __init__(
        self,
        *,
        embed_cos: Optional[CosineScorer] = None,
        nli_scores: Optional[NliScorer] = None,
        gate_floor: float = DEFAULT_GATE_FLOOR,
        corroborate_ceiling: float = DEFAULT_CORROBORATE_CEILING,
        contra_threshold: float = DEFAULT_CONTRA_THRESHOLD,
        embed_model: str = DEFAULT_EMBED_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
    ) -> None:
        self.gate_floor = gate_floor
        self.corroborate_ceiling = corroborate_ceiling
        self.contra_threshold = contra_threshold
        self.embed_model = embed_model
        self.nli_model = nli_model
        self._embed_cos = embed_cos
        self._nli_scores = nli_scores
        self._embedder = None
        self._emb_cache: dict[str, Any] = {}
        self._nli = None
        self._nli_label_index: dict[str, int] = {}

    # -- backends -----------------------------------------------------------

    def _embedding(self, text: str):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embed_model)
        if text not in self._emb_cache:
            self._emb_cache[text] = self._embedder.encode(
                [text], normalize_embeddings=True
            )[0]
        return self._emb_cache[text]

    def _cosine(self, a_text: str, b_text: str) -> float:
        if self._embed_cos is not None:
            return self._embed_cos(a_text, b_text)
        import numpy as np

        return float(np.dot(self._embedding(a_text), self._embedding(b_text)))

    def _nli_model_scores(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        import numpy as np

        if self._nli is None:
            from sentence_transformers import CrossEncoder

            self._nli = CrossEncoder(self.nli_model)
            id2label = getattr(getattr(self._nli, "config", None), "id2label", {}) or {}
            self._nli_label_index = {str(v).lower(): int(k) for k, v in id2label.items()}
        raw = self._nli.predict(
            [(premise, hypothesis)], apply_softmax=True, convert_to_numpy=True
        )
        row = np.asarray(raw[0] if getattr(raw, "ndim", 0) == 2 else raw, dtype=float).reshape(-1)
        c = self._nli_label_index.get("contradiction", 0)
        e = self._nli_label_index.get("entailment", 1)
        n = self._nli_label_index.get("neutral", 2)
        return float(row[c]), float(row[e]), float(row[n])

    def _p_contra(self, a_text: str, b_text: str) -> float:
        scorer = self._nli_scores or self._nli_model_scores
        return max(scorer(a_text, b_text)[0], scorer(b_text, a_text)[0])

    # -- relation typing ----------------------------------------------------

    def type_relation(self, a_text: str, b_text: str) -> tuple[Relation, dict]:
        cos = self._cosine(a_text, b_text)
        if cos < self.gate_floor:
            return Relation.UNRELATED, {"cos": cos, "channel": "gate"}
        p_contra = self._p_contra(a_text, b_text)
        if p_contra >= self.contra_threshold:
            return Relation.CONTRADICTS, {"cos": cos, "p_contra": p_contra, "channel": "nli"}
        if cos >= self.corroborate_ceiling:
            return Relation.CORROBORATES, {"cos": cos, "p_contra": p_contra, "channel": "band"}
        return Relation.CONTRADICTS, {"cos": cos, "p_contra": p_contra, "channel": "band"}

    def assess_pair(self, a: LabeledChunk, b: LabeledChunk) -> Assessment:
        relation, ev = self.type_relation(a.text, b.text)
        return Assessment(
            src_chunk_id=a.chunk_id,
            dst_chunk_id=b.chunk_id,
            relation=relation,
            method=f"combined_relation:{ev['channel']}",
            confidence=ev.get("p_contra", ev["cos"]),
            rationale=f"cos={ev['cos']:.3f} channel={ev['channel']}"
            + (f" p_contra={ev['p_contra']:.3f}" if "p_contra" in ev else ""),
        )

    def propose_over(
        self, pairs: Iterable[tuple[LabeledChunk, LabeledChunk]]
    ) -> list[Assessment]:
        return [self.assess_pair(a, b) for a, b in pairs]
