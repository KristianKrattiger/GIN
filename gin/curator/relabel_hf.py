"""Correct two housing pairs from issue_frame back to story.

Sub-project B's Task 1 backfill classified hf_af_* (Alder Flats rezoning) and
hf_kc_* (Kestrel Court habitability) as issue_frame. Two independent sources say
story: the labeling guide, corrected against labels.jsonl on 2026-07-20
(b9e0079), lists rezoning and habitability as story examples; and gold_edges
labels the same content story under its long-form ids (hf_alderflats_*,
hf_kestrel_*). Both pairs pass make_same_story, which is what story means.

Appended as superseding records, never edited in place — the same mechanism the
original backfill used, so the earlier judgment stays auditable.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from gin.cartographer.models import Relation

from .models import LabelRecord, pair_key
from .store import Store

HF_STORY_RELABEL: dict[tuple[str, str], str] = {
    pair_key("hf_af_staff:0", "hf_af_tenants:0"): "story",
    pair_key("hf_kc_inspection:0", "hf_kc_tenants:0"): "story",
}

RATIONALE = (
    "relabel: same-story institutional-vs-community divergence; matches the "
    "labeling guide's rezoning/habitability story examples and the gold_edges "
    "long-form ids. Supersedes an issue_frame backfill."
)


def relabel_hf_to_story(store: Store, *, curator: str = "relabel") -> int:
    """Append superseding records fixing the two housing pairs. Idempotent."""
    current = store.fold_current()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    appended = 0
    for key, relation_class in sorted(HF_STORY_RELABEL.items()):
        rec = current.get(key)
        if rec is None:
            continue
        if rec.relation is not Relation.CONTRADICTS or rec.relation_class == relation_class:
            continue
        store.append(
            LabelRecord(
                id=str(uuid.uuid4()),
                src_chunk_id=rec.src_chunk_id,
                dst_chunk_id=rec.dst_chunk_id,
                relation=Relation.CONTRADICTS,
                relation_class=relation_class,
                rationale=RATIONALE,
                curator=curator,
                ts=ts,
                supersedes=rec.id,
                src_anchor=rec.src_anchor,
                dst_anchor=rec.dst_anchor,
            )
        )
        appended += 1
    return appended
