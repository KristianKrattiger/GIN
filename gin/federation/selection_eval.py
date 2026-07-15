"""Peer-selection bar: four-class queryset, outcomes, and metrics.

The correct peer for a b_only/c_only query is implied by its class label
(b_only -> node_b, c_only -> node_c). Selection precision@1 asks whether A's
FIRST contacted peer was that correct one; avg peers tried asks how far down
the ranked list A had to go. Attribution reuses gin.federation.eval's verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

VALID_CLASSES = {"a_answerable", "b_only", "c_only", "neither"}
CLASS_TO_PEER = {"b_only": "node_b", "c_only": "node_c"}


@dataclass(frozen=True)
class SelectionQuery:
    id: str
    query: str
    federation_class: str


def load_selection_queryset(path: str | Path) -> list[SelectionQuery]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: list[SelectionQuery] = []
    for q in raw["queries"]:
        cls = q["federation_class"]
        if cls not in VALID_CLASSES:
            raise ValueError(f"query {q['id']!r}: class {cls!r} not in {sorted(VALID_CLASSES)}")
        out.append(SelectionQuery(id=str(q["id"]), query=str(q["query"]), federation_class=cls))
    return out


@dataclass
class SelectionOutcome:
    id: str
    federation_class: str
    refused: bool
    routed: bool
    source_node: str = ""
    peers_attempted: list[str] = field(default_factory=list)
    attribution_verified: Optional[bool] = None
    refusal_reasons: dict = field(default_factory=dict)


def compute_selection_metrics(outcomes: list[SelectionOutcome]) -> dict:
    a = [o for o in outcomes if o.federation_class == "a_answerable"]
    neither = [o for o in outcomes if o.federation_class == "neither"]
    routed_grounded = [
        o for o in outcomes
        if o.federation_class in CLASS_TO_PEER and o.routed and not o.refused
    ]
    correct_first = [
        o for o in routed_grounded
        if o.peers_attempted[:1] == [CLASS_TO_PEER[o.federation_class]]
    ]
    verified = [o for o in routed_grounded if o.attribution_verified]
    tried_counts = [len(o.peers_attempted) for o in routed_grounded]
    return {
        "n_queries": len(outcomes),
        # Bar: 1.0 — the correct peer is contacted first.
        "selection_precision_at_1": (len(correct_first) / len(routed_grounded)) if routed_grounded else None,
        # Bar: ~1.0 — selection beats blind sequential fan-out.
        "avg_peers_tried": (sum(tried_counts) / len(tried_counts)) if tried_counts else None,
        # Bar: 0 — A-answerable queries never route.
        "routing_false_positives": sum(1 for o in a if o.routed),
        # Bar: 1.0 / 0.0 — routed answers verify against the answering node's corpus.
        "routed_attribution_verified": (len(verified) / len(routed_grounded)) if routed_grounded else None,
        "routed_fabrication_rate": (1.0 - len(verified) / len(routed_grounded)) if routed_grounded else None,
        # Bar: 1.0 — neither-class queries end in refusal.
        "honest_refusal_rate": (sum(1 for o in neither if o.refused) / len(neither)) if neither else None,
        "per_query": [o.__dict__ for o in outcomes],
    }
