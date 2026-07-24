"""One-shot: classify the 7 seed contradicts that predate relation_class.

Five are framing divergences over a shared issue (institutional vs grassroots,
landlord vs tenant) and are canonical issue_frame. Two are securities-fraud
pairs — propositional contradictions that NLI already types upstream — so they
are tagged `story` and stay out of the frame training set.

The store is append-only: this appends superseding records rather than editing,
so the original seed judgments remain auditable in the log.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.store import Store

# pair_key (sorted) -> relation_class
SEED_CLASS_BACKFILL: dict[tuple[str, str], str] = {
    pair_key("inst_em:0", "grass_em:0"): "issue_frame",
    pair_key("inst_wf:0", "grass_wf:0"): "issue_frame",
    pair_key("inst_wa:0", "grass_wa:0"): "issue_frame",
    pair_key("hf_af_staff:0", "hf_af_tenants:0"): "issue_frame",
    pair_key("hf_kc_inspection:0", "hf_kc_tenants:0"): "issue_frame",
    pair_key("disc_nw_pr:0", "disc_nw_complaint:0"): "story",
    pair_key("disc_mer_pr:0", "disc_mer_complaint:0"): "story",
}

RATIONALE = {
    "issue_frame": "backfill: framing divergence over a shared issue (pre-relation_class seed)",
    "story": "backfill: propositional contradiction, NLI-typed register (pre-relation_class seed)",
}


def backfill_seed_classes(store: Store, *, curator: str = "backfill") -> int:
    """Append superseding records tagging untyped seed contradicts. Idempotent."""
    current = store.fold_current()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    appended = 0
    for key, relation_class in sorted(SEED_CLASS_BACKFILL.items()):
        rec = current.get(key)
        if rec is None:
            continue
        if rec.relation is not Relation.CONTRADICTS or rec.relation_class is not None:
            continue
        store.append(
            LabelRecord(
                id=str(uuid.uuid4()),
                src_chunk_id=rec.src_chunk_id,
                dst_chunk_id=rec.dst_chunk_id,
                relation=Relation.CONTRADICTS,
                relation_class=relation_class,
                rationale=RATIONALE[relation_class],
                curator=curator,
                ts=ts,
                supersedes=rec.id,
                src_anchor=rec.src_anchor,
                dst_anchor=rec.dst_anchor,
            )
        )
        appended += 1
    return appended
