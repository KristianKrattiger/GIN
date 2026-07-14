"""Federation v1 bar: class-labeled queryset, outcomes, metrics.

The eval driver — unlike Node A — legitimately holds credentials for BOTH
node databases, so it performs the attribution verification A architecturally
cannot: every routed claim's text must appear verbatim (whitespace-normalized)
in every chunk it cites, fetched from the answering node's own database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import psycopg
import yaml

from .schema import WireClaim

VALID_CLASSES = {"a_answerable", "b_only", "neither"}


@dataclass(frozen=True)
class FederationQuery:
    id: str
    query: str
    federation_class: str
    gold_chunk_ids: tuple[str, ...] = ()


def load_federation_queryset(path: str | Path) -> list[FederationQuery]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: list[FederationQuery] = []
    for q in raw["queries"]:
        cls = q["federation_class"]
        if cls not in VALID_CLASSES:
            raise ValueError(
                f"query {q['id']!r}: federation_class {cls!r} not in "
                f"{sorted(VALID_CLASSES)}"
            )
        out.append(
            FederationQuery(
                id=str(q["id"]),
                query=str(q["query"]),
                federation_class=cls,
                gold_chunk_ids=tuple(q.get("gold_chunk_ids", []) or []),
            )
        )
    return out


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def claims_verify(
    claims: list[WireClaim], fetch_text: Callable[[str], Optional[str]]
) -> bool:
    """True iff every claim cites >=1 chunk and its text appears
    (whitespace-normalized) in EVERY chunk it cites. No claims = False —
    an answer with nothing to verify is not a verified answer."""
    if not claims:
        return False
    for claim in claims:
        if not claim.cited_chunk_ids:
            return False
        for chunk_id in claim.cited_chunk_ids:
            text = fetch_text(chunk_id)
            if text is None or _normalize(claim.text) not in _normalize(text):
                return False
    return True


def verify_claims_in_db(claims: list[WireClaim], database_url: str) -> bool:
    """DB-backed claims_verify against the answering node's chunks table."""
    with psycopg.connect(database_url) as conn:

        def fetch(chunk_id: str) -> Optional[str]:
            row = conn.execute(
                "SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,)
            ).fetchone()
            return row[0] if row else None

        return claims_verify(claims, fetch)


@dataclass
class QueryOutcome:
    id: str
    federation_class: str
    refused: bool
    routed: bool
    source_node: str = ""
    attribution_verified: Optional[bool] = None
    refusal_reasons: dict = field(default_factory=dict)


def compute_metrics(outcomes: list[QueryOutcome]) -> dict:
    a = [o for o in outcomes if o.federation_class == "a_answerable"]
    b = [o for o in outcomes if o.federation_class == "b_only"]
    n = [o for o in outcomes if o.federation_class == "neither"]
    routed_answers_b = [o for o in b if o.routed and not o.refused]
    verified = [o for o in routed_answers_b if o.attribution_verified]
    return {
        "n_queries": len(outcomes),
        # Bar: 0 — an A-answerable query must never consult the peer.
        "routing_false_positives": sum(1 for o in a if o.routed),
        # Bar: 1.0 — every B-only query must reach the peer.
        "routing_recall": (sum(1 for o in b if o.routed) / len(b)) if b else None,
        # Bar: 1.0 / 0.0 — routed answers verify against B's corpus.
        "routed_answer_attribution_verified": (
            len(verified) / len(routed_answers_b)
        ) if routed_answers_b else None,
        "routed_fabrication_rate": (
            1.0 - len(verified) / len(routed_answers_b)
        ) if routed_answers_b else None,
        # Bar: 1.0 — neither-class queries end in refusal, never an answer.
        "honest_refusal_rate": (
            sum(1 for o in n if o.refused) / len(n)
        ) if n else None,
        "a_answered_locally": sum(
            1 for o in a if not o.refused and not o.routed
        ),
        "per_query": [o.__dict__ for o in outcomes],
    }
