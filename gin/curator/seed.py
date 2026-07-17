"""One-time import of the existing gold into the store as seed records.

labeled_set carries `register` (not a story/issue_frame class), so its
contradicts pairs seed with relation_class=None — the importer never GUESSES a
class. gold_edges carries the YAML relation_class and keeps it. labeled_set is
emitted first, so on any pair collision the labeled_set relation wins (this is
what the regression guard asserts).
"""
from __future__ import annotations

import uuid

from gin.cartographer import gold_edges, labeled_set
from gin.cartographer.models import Relation

from .models import LabelRecord, pair_key
from .store import Store


def seed_records(curator: str = "seed", ts: str = "2026-07-17T00:00:00Z") -> list[LabelRecord]:
    records: list[LabelRecord] = []
    for src, dst, relation, _register in labeled_set.gold():
        records.append(
            LabelRecord(
                id=str(uuid.uuid4()), src_chunk_id=src, dst_chunk_id=dst,
                relation=relation, relation_class=None,
                rationale="", curator=curator, ts=ts,
            )
        )
    for e in gold_edges.load_all_gold_contradicts():
        records.append(
            LabelRecord(
                id=str(uuid.uuid4()), src_chunk_id=e.src_chunk_id, dst_chunk_id=e.dst_chunk_id,
                relation=Relation.CONTRADICTS, relation_class=e.relation_class,
                rationale=e.note, curator=curator, ts=ts,
            )
        )
    return records


def seed_store(store: Store, curator: str = "seed", ts: str = "2026-07-17T00:00:00Z") -> int:
    present = set(store.fold_current().keys())
    appended = 0
    for rec in seed_records(curator=curator, ts=ts):
        key = pair_key(rec.src_chunk_id, rec.dst_chunk_id)
        if key in present:
            continue
        store.append(rec)
        present.add(key)
        appended += 1
    return appended
