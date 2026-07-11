"""NLI-based relation typing — the natural stage-2 relation detector attempt.

Uses a 3-class entailment cross-encoder (the same model family the eval Verifier
uses: contradiction / entailment / neutral) as a semantic signal orthogonal to
the retrieval IDF signal, satisfying the design §2 independence constraint.

The measured result is a *negative* one worth recording: GIN "divergence" is not
propositional NLI-contradiction. Institutional-vs-grassroots framing pairs are
both true statements that emphasize different aspects of a shared event, so an
entailment model rates them neutral — indistinguishable from genuine
corroboration. This detector therefore lands the framing-divergence pairs in
RELATED_UNTYPED, achieving class_c_discrimination = 1.0 only by the degenerate
route of typing almost nothing as contradicts (recall collapses). That is the
finding that reshapes the relation-detector design (see
docs/nc_cartographer_design.plan.md §6). Kept with an injectable scorer so it is
testable without the model.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from .models import Assessment, LabeledChunk, Relation

# (premise, hypothesis) -> (p_contradiction, p_entailment, p_neutral)
NliScorer = Callable[[str, str], tuple[float, float, float]]

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
DEFAULT_CONTRA_THRESHOLD = 0.5
DEFAULT_ENTAIL_THRESHOLD = 0.5


class NliRelationProposer:
    """Type related pairs by pairwise NLI, both directions.

    ``scorer`` is an injectable ``(premise, hypothesis) -> (contra, entail,
    neutral)`` callable (mirrors ``Verifier``'s injectable scorer), so the
    relation logic is testable deterministically. Without one, a CrossEncoder is
    lazily loaded.
    """

    name = "nli_relation"

    def __init__(
        self,
        *,
        scorer: Optional[NliScorer] = None,
        model_name: str = DEFAULT_NLI_MODEL,
        contra_threshold: float = DEFAULT_CONTRA_THRESHOLD,
        entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.contra_threshold = contra_threshold
        self.entail_threshold = entail_threshold
        self._scorer = scorer
        self._model = None
        self._label_index: dict[str, int] = {}

    # -- NLI backend --------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self.model_name)
            id2label = getattr(getattr(self._model, "config", None), "id2label", {}) or {}
            self._label_index = {
                str(v).lower(): int(k) for k, v in id2label.items()
            }
        return self._model

    def _model_scores(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        import numpy as np

        model = self._load_model()
        raw = model.predict(
            [(premise, hypothesis)], apply_softmax=True, convert_to_numpy=True
        )
        row = np.asarray(raw[0] if getattr(raw, "ndim", 0) == 2 else raw, dtype=float).reshape(-1)
        c = self._label_index.get("contradiction", 0)
        e = self._label_index.get("entailment", 1)
        n = self._label_index.get("neutral", 2)
        return float(row[c]), float(row[e]), float(row[n])

    def _scores(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        if self._scorer is not None:
            return self._scorer(premise, hypothesis)
        return self._model_scores(premise, hypothesis)

    # -- relation typing ----------------------------------------------------

    def type_relation(self, a_text: str, b_text: str) -> tuple[Relation, dict]:
        """Symmetric relation type + the NLI evidence behind it."""
        c_ab, e_ab, _ = self._scores(a_text, b_text)
        c_ba, e_ba, _ = self._scores(b_text, a_text)
        p_contra = max(c_ab, c_ba)
        p_entail = max(e_ab, e_ba)
        if p_contra >= self.contra_threshold:
            relation = Relation.CONTRADICTS
        elif p_entail >= self.entail_threshold:
            relation = Relation.CORROBORATES
        else:
            # Related but neither entailed nor contradicted: framing divergence
            # lands here, which is the finding.
            relation = Relation.RELATED_UNTYPED
        evidence = {"p_contra": p_contra, "p_entail": p_entail}
        return relation, evidence

    def assess_pair(self, a: LabeledChunk, b: LabeledChunk) -> Assessment:
        relation, ev = self.type_relation(a.text, b.text)
        return Assessment(
            src_chunk_id=a.chunk_id,
            dst_chunk_id=b.chunk_id,
            relation=relation,
            method="nli_relation:cross_encoder",
            confidence=max(ev["p_contra"], ev["p_entail"]),
            rationale=f"p_contra={ev['p_contra']:.3f} p_entail={ev['p_entail']:.3f}",
        )

    def propose_over(
        self, pairs: Iterable[tuple[LabeledChunk, LabeledChunk]]
    ) -> list[Assessment]:
        """Type an explicit candidate set — isolates the relation signal from
        the stage-1 relatedness gate's (separately recorded) recall limits."""
        return [self.assess_pair(a, b) for a, b in pairs]
