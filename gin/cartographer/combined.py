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

from dataclasses import dataclass
from pathlib import Path
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

_THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "data" / "cartographer_thresholds.json"


@dataclass(frozen=True)
class Thresholds:
    gate_floor: float = DEFAULT_GATE_FLOOR
    corroborate_ceiling: float = DEFAULT_CORROBORATE_CEILING
    contra_threshold: float = DEFAULT_CONTRA_THRESHOLD


def load_thresholds(path: Optional[Path] = None) -> Thresholds:
    """Load calibrated thresholds from data/cartographer_thresholds.json if present."""
    p = path or _THRESHOLDS_PATH
    if not p.is_file():
        return Thresholds()
    import json

    raw = json.loads(p.read_text(encoding="utf-8"))
    return Thresholds(
        gate_floor=float(raw.get("gate_floor", DEFAULT_GATE_FLOOR)),
        corroborate_ceiling=float(raw.get("corroborate_ceiling", DEFAULT_CORROBORATE_CEILING)),
        contra_threshold=float(raw.get("contra_threshold", DEFAULT_CONTRA_THRESHOLD)),
    )


def classify_relation(
    cos: float, p_contra: float, t: Thresholds, *, same_story: Optional[bool] = None
) -> tuple[Relation, str]:
    """Pure combined-detector decision. Returns (relation, channel).

    Shared by CombinedRelationProposer and the calibrator so both apply the exact
    same rule. ``same_story`` is the stage-1 relatedness tier (shared rare story
    entities — design §2 allows entity signals there): True/False when a stage-1
    provider is wired, None when there is no story evidence either way.

    Scan-scale measurements (run 20260712T074956Z) showed cosine measures
    topicality, not stance: true framing divergences sit ABOVE the corroborate
    ceiling while the mid band is cross-topic noise. So contradicts typing is
    story-gated on both channels — the band types any ungated same-story pair
    as divergent (framing pairs are exactly the highly similar ones), the NLI
    propositional channel is blocked only when stage 1 positively says the pair
    is NOT one story (the cross-topic numeric-claim artifact) — and the old
    mid-band default flips from CONTRADICTS to RELATED_UNTYPED (no edge).
    """
    if cos < t.gate_floor:
        return Relation.UNRELATED, "gate"
    if p_contra >= t.contra_threshold and same_story is not False:
        return Relation.CONTRADICTS, "nli"
    if same_story:
        return Relation.CONTRADICTS, "band"
    if cos >= t.corroborate_ceiling:
        return Relation.CORROBORATES, "band"
    return Relation.RELATED_UNTYPED, "band"


class CombinedRelationProposer:
    name = "combined_relation"

    def __init__(
        self,
        *,
        embed_cos: Optional[CosineScorer] = None,
        nli_scores: Optional[NliScorer] = None,
        same_story: Optional[Callable[[str, str], bool]] = None,
        gate_floor: float = DEFAULT_GATE_FLOOR,
        corroborate_ceiling: float = DEFAULT_CORROBORATE_CEILING,
        contra_threshold: float = DEFAULT_CONTRA_THRESHOLD,
        thresholds: Optional[Thresholds] = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
    ) -> None:
        t = thresholds or load_thresholds()
        self.thresholds = t
        # Stage-1 same-story provider (see relatedness.make_same_story); the
        # scan wires this from the scanned corpus. None = no story evidence.
        self.same_story = same_story
        self.gate_floor = t.gate_floor
        self.corroborate_ceiling = t.corroborate_ceiling
        self.contra_threshold = t.contra_threshold
        self.embed_model = embed_model
        self.nli_model = nli_model
        self._embed_cos = embed_cos
        self._nli_scores = nli_scores
        self._embedder = None
        self._emb_cache: dict[str, Any] = {}
        self._nli = None
        self._nli_label_index: dict[str, int] = {}
        self._p_contra_cache: dict[frozenset, float] = {}

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

    def embedding_cosine(self, a_text: str, b_text: str) -> float:
        """Public cosine accessor for scan-stage candidate pruning."""
        return self._cosine(a_text, b_text)

    def nli_p_contra(self, a_text: str, b_text: str) -> float:
        """Public NLI-contradiction accessor (max over both directions).

        Lets candidate ranking consult the propositional channel directly (e.g.
        to float high-cosine contradictions), reusing the same cross-encoder the
        typer uses. Injected nli_scores keep this model-free under test.
        """
        return self._p_contra(a_text, b_text)

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
        # Memoized on the UNORDERED pair of texts: the result already takes the
        # max over both directions, so a(b) and b(a) share one cache entry.
        key = frozenset((a_text, b_text))
        if key in self._p_contra_cache:
            return self._p_contra_cache[key]
        scorer = self._nli_scores or self._nli_model_scores
        value = max(scorer(a_text, b_text)[0], scorer(b_text, a_text)[0])
        self._p_contra_cache[key] = value
        return value

    # -- relation typing ----------------------------------------------------

    def type_relation(self, a_text: str, b_text: str) -> tuple[Relation, dict]:
        cos = self._cosine(a_text, b_text)
        # The gate needs only cosine; compute NLI lazily so a gated pair never
        # pays for a cross-encoder call.
        if cos < self.thresholds.gate_floor:
            return Relation.UNRELATED, {"cos": cos, "channel": "gate"}
        story = self.same_story(a_text, b_text) if self.same_story is not None else None
        if story is False:
            # Both contradicts channels are story-blocked: the outcome is
            # decided by the corroborate ceiling alone, no cross-encoder call.
            relation, channel = classify_relation(cos, 0.0, self.thresholds, same_story=False)
            return relation, {"cos": cos, "channel": channel, "same_story": False}
        p_contra = self._p_contra(a_text, b_text)
        relation, channel = classify_relation(
            cos, p_contra, self.thresholds, same_story=story
        )
        ev = {"cos": cos, "p_contra": p_contra, "channel": channel}
        if story is not None:
            ev["same_story"] = story
        return relation, ev

    def assess_pair(self, a: LabeledChunk, b: LabeledChunk) -> Assessment:
        relation, ev = self.type_relation(a.text, b.text)
        channel = ev["channel"]
        if channel == "band" and relation == Relation.CONTRADICTS:
            confidence = ev["cos"]
        else:
            confidence = ev.get("p_contra", ev["cos"])
        return Assessment(
            src_chunk_id=a.chunk_id,
            dst_chunk_id=b.chunk_id,
            relation=relation,
            method=f"combined_relation:{channel}",
            confidence=confidence,
            rationale=f"cos={ev['cos']:.3f} channel={channel}"
            + (f" p_contra={ev['p_contra']:.3f}" if "p_contra" in ev else ""),
        )

    def propose_over(
        self, pairs: Iterable[tuple[LabeledChunk, LabeledChunk]]
    ) -> list[Assessment]:
        return [self.assess_pair(a, b) for a, b in pairs]
