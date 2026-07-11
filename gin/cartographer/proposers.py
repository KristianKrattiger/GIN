"""Cartographer proposers — stage 2, relation typing.

A proposer takes the corpus chunks, runs the relatedness gate (stage 1), types
the related candidates, and returns one Assessment per assessed pair (negatives
included). Only typed, non-``unrelated`` assessments are eligible to become
EdgeProposals for the Bookkeeper.

``RelatednessProposer`` is the deliberate ANTI-PATTERN: it collapses stage 1 and
stage 2 by typing every related pair as ``contradicts``. It exists so the
evaluation harness can quantify exactly how much precision that collapse costs and
show it failing class-C discrimination (design §2/§4) — it is the negative
baseline the real NLI relation detector (design §6) must beat, not a component to
ship. See docs/nc_cartographer_design.plan.md.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Protocol, runtime_checkable

from .models import Assessment, LabeledChunk, Relation
from .relatedness import DEFAULT_RELATEDNESS_FLOOR, RelatednessGate


@runtime_checkable
class Proposer(Protocol):
    name: str

    def propose(self, chunks: Iterable[LabeledChunk]) -> list[Assessment]: ...


class RelatednessProposer:
    """ANTI-PATTERN: types every related pair as ``contradicts``.

    Relatedness is not relation: a true contradiction and a true corroboration are
    equally related (they share a topic), so this proposer cannot tell them apart.
    High contradicts recall, poor precision, and class_c_discrimination = 0.
    """

    name = "relatedness_only"

    def __init__(
        self,
        *,
        floor: float = DEFAULT_RELATEDNESS_FLOOR,
        idf_corpus: Iterable[str] | None = None,
    ) -> None:
        self.floor = floor
        self.idf_corpus = list(idf_corpus) if idf_corpus is not None else None

    def propose(self, chunks: Iterable[LabeledChunk]) -> list[Assessment]:
        gate = RelatednessGate(chunks, floor=self.floor, idf_corpus=self.idf_corpus)
        assessments: list[Assessment] = []
        for a in gate.assess_all():
            if a.relation == Relation.RELATED_UNTYPED:
                # The blind spot: related -> assumed contradicting.
                assessments.append(
                    replace(
                        a,
                        relation=Relation.CONTRADICTS,
                        method="relatedness_only:related_implies_contradicts",
                    )
                )
            else:
                assessments.append(a)  # negatives pass through, stored
        return assessments
