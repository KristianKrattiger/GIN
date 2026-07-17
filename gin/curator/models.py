"""The atomic unit the curator emits: one immutable labeling act."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gin.cartographer.models import Relation


def pair_key(src: str, dst: str) -> tuple[str, str]:
    """Order-independent key so A->B and B->A fold to the same pair."""
    a, b = sorted((src, dst))
    return (a, b)


def _anchor_to_json(anchor: Optional[tuple[int, int]]) -> Optional[list[int]]:
    return None if anchor is None else [anchor[0], anchor[1]]


def _anchor_from_json(raw) -> Optional[tuple[int, int]]:
    if not raw:
        return None
    return int(raw[0]), int(raw[1])


@dataclass(frozen=True)
class LabelRecord:
    """One label / relabel / adjudication. Appended, never mutated in place."""

    id: str
    src_chunk_id: str
    dst_chunk_id: str
    relation: Relation
    relation_class: Optional[str]
    rationale: str
    curator: str
    ts: str  # UTC ISO-8601
    supersedes: Optional[str] = None
    src_anchor: Optional[tuple[int, int]] = None
    dst_anchor: Optional[tuple[int, int]] = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "src_chunk_id": self.src_chunk_id,
            "dst_chunk_id": self.dst_chunk_id,
            "relation": self.relation.value,
            "relation_class": self.relation_class,
            "rationale": self.rationale,
            "curator": self.curator,
            "ts": self.ts,
            "supersedes": self.supersedes,
            "src_anchor": _anchor_to_json(self.src_anchor),
            "dst_anchor": _anchor_to_json(self.dst_anchor),
        }

    @classmethod
    def from_json(cls, d: dict) -> "LabelRecord":
        return cls(
            id=d["id"],
            src_chunk_id=d["src_chunk_id"],
            dst_chunk_id=d["dst_chunk_id"],
            relation=Relation(d["relation"]),
            relation_class=d.get("relation_class"),
            rationale=d.get("rationale", ""),
            curator=d.get("curator", "unknown"),
            ts=d["ts"],
            supersedes=d.get("supersedes"),
            src_anchor=_anchor_from_json(d.get("src_anchor")),
            dst_anchor=_anchor_from_json(d.get("dst_anchor")),
        )
