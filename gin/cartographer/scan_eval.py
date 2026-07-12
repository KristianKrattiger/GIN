"""Evaluate Cartographer scan output against gold hand-curated edges."""

from __future__ import annotations



from dataclasses import dataclass, field

from pathlib import Path

from typing import Iterable, Optional, Sequence



import psycopg



from gin.bookkeeper import AdmissionCode, Bookkeeper

from gin.bookkeeper.models import AdmittedEdge



from .combined import CombinedRelationProposer

from .evaluation import CartographerMetrics, GoldPair, _key, evaluate

from .gold_edges import CLASS_C_CONTROLS, gold_pairs, load_all_gold_contradicts

from .models import Assessment, Relation

from .scan import (

    DEFAULT_EXCLUDED_DOC_IDS,

    candidate_pairs,

    chunks_from_db,

    dedupe_doc_pair_proposals,

    doc_id_from_chunk,

    make_relation_verifier,

    proposals_from_pairs,

    prune_pairs_by_relatedness,

    whitespace_token_count,

    wire_same_story,

)





def split_gold_by_class(

    gold: Iterable[GoldPair],

) -> tuple[list[GoldPair], list[GoldPair]]:

    """(machine_gold, curated_gold): scan metrics are scored on the story class

    only; issue_frame pairs are curated-ingest edges the scan cannot reach."""

    gold = list(gold)

    machine = [g for g in gold if g.relation_class != "issue_frame"]

    curated = [g for g in gold if g.relation_class == "issue_frame"]

    return machine, curated





def doc_pair_keys(keys: Iterable[frozenset]) -> set[frozenset]:

    """Collapse chunk-pair keys to unordered doc-pair keys."""

    return {frozenset(doc_id_from_chunk(cid) for cid in key) for key in keys}





def doc_level_metrics(

    admitted_keys: Iterable[frozenset], gold_keys: Iterable[frozenset]

) -> tuple[int, list[frozenset], list[frozenset]]:

    """Score discovery at doc-pair granularity: (tp, fp_keys, missed_keys).

    The dedupe keeps one chunk pair per doc pair while the gold YAML picks a

    specific chunk pair — anchor disagreement on the same doc pair should count

    as a discovery, not a miss plus a false positive.

    """

    admitted_docs = doc_pair_keys(admitted_keys)

    gold_docs = doc_pair_keys(gold_keys)

    tp = len(admitted_docs & gold_docs)

    fp = sorted(admitted_docs - gold_docs)

    missed = sorted(gold_docs - admitted_docs)

    return tp, fp, missed


def split_machine_false_positives(
    admitted_keys: set[frozenset],
    gold_keys: set[frozenset],
    curated_keys: Iterable[frozenset],
) -> tuple[list[frozenset], list[frozenset]]:
    """Split admitted-not-gold into (false_positive_keys, anchor_discovery_keys).

    Chunk pairs on a doc pair that already has curated issue_frame gold are
    anchor discoveries, not machine false positives.
    """
    curated_doc_pairs = doc_pair_keys(curated_keys)
    not_gold = admitted_keys - gold_keys
    anchor_discoveries: list[frozenset] = []
    false_positives: list[frozenset] = []
    for key in sorted(not_gold):
        doc_pair = frozenset(doc_id_from_chunk(cid) for cid in key)
        if doc_pair in curated_doc_pairs:
            anchor_discoveries.append(key)
        else:
            false_positives.append(key)
    return false_positives, anchor_discoveries


@dataclass

class ScanEvalResult:

    metrics: CartographerMetrics

    admitted_contradicts: list[AdmittedEdge]

    false_positive_keys: list[frozenset] = field(default_factory=list)

    missed_gold_keys: list[frozenset] = field(default_factory=list)

    class_c_discrimination: Optional[float] = None

    class_c_total: int = 0

    class_c_pass: int = 0

    gold_count: int = 0

    admitted_count: int = 0

    pair_count_before_prune: int = 0

    pair_count_after_prune: int = 0

    doc_tp: int = 0

    doc_false_positive_keys: list[frozenset] = field(default_factory=list)

    doc_missed_gold_keys: list[frozenset] = field(default_factory=list)

    # issue_frame gold: curated-ingest class, excluded from machine metrics.
    curated_gold_keys: list[frozenset] = field(default_factory=list)
    # Admitted chunk pairs on curated doc pairs but at different anchors.
    anchor_discovery_keys: list[frozenset] = field(default_factory=list)

    def to_dict(self) -> dict:

        return {

            "metrics": self.metrics.to_dict(),

            "gold_count": self.gold_count,

            "admitted_contradicts_count": self.admitted_count,

            "false_positive_count": len(self.false_positive_keys),

            "missed_gold_count": len(self.missed_gold_keys),

            "false_positive_keys": [sorted(k) for k in self.false_positive_keys],

            "missed_gold_keys": [sorted(k) for k in self.missed_gold_keys],

            "class_c_discrimination": self.class_c_discrimination,

            "class_c_total": self.class_c_total,

            "class_c_pass": self.class_c_pass,

            "pair_count_before_prune": self.pair_count_before_prune,

            "pair_count_after_prune": self.pair_count_after_prune,

            "doc_tp": self.doc_tp,

            "doc_false_positive_count": len(self.doc_false_positive_keys),

            "doc_missed_gold_count": len(self.doc_missed_gold_keys),

            "doc_false_positive_keys": [sorted(k) for k in self.doc_false_positive_keys],

            "doc_missed_gold_keys": [sorted(k) for k in self.doc_missed_gold_keys],

            "curated_gold_count": len(self.curated_gold_keys),

            "curated_gold_keys": [sorted(k) for k in self.curated_gold_keys],

            "anchor_discovery_count": len(self.anchor_discovery_keys),

            "anchor_discovery_keys": [sorted(k) for k in self.anchor_discovery_keys],

            "admitted_edges": [

                {

                    "src": e.src_chunk_id,

                    "dst": e.dst_chunk_id,

                    "relation": e.relation.value,

                    "confidence": e.provenance.confidence,

                    "proposer": e.provenance.proposer,

                }

                for e in self.admitted_contradicts

            ],

        }





def _class_c_from_proposals(

    proposals: Iterable[Assessment],

) -> tuple[Optional[float], int, int]:

    proposed = {

        _key(p.src_chunk_id, p.dst_chunk_id): p.relation for p in proposals

    }

    total = passed = 0

    for src, dst, _reg in CLASS_C_CONTROLS:

        total += 1

        if proposed.get(_key(src, dst), Relation.UNRELATED) != Relation.CONTRADICTS:

            passed += 1

    rate = passed / total if total else None

    return rate, total, passed





def evaluate_scan_on_conn(

    conn: psycopg.Connection,

    *,

    doc_id: Optional[str] = None,

    min_confidence: float = 0.5,

    cross_outlet_only: bool = False,

    prune_relatedness: bool = True,

    relatedness_floor: float = 0.20,

    exclude_doc_ids: Optional[Sequence[str]] = None,

    relation_recheck: bool = True,

    proposer: Optional[CombinedRelationProposer] = None,

    gold_sources: Optional[Iterable[Path]] = None,

) -> ScanEvalResult:

    proposer = proposer or CombinedRelationProposer()

    if exclude_doc_ids is None:

        exclude_doc_ids = DEFAULT_EXCLUDED_DOC_IDS



    chunks, outlets = chunks_from_db(

        conn, doc_id=doc_id, exclude_doc_ids=exclude_doc_ids

    )

    wire_same_story(proposer, chunks)

    registry = {ch.chunk_id: whitespace_token_count(ch.text) for ch in chunks}

    text_by_chunk = {ch.chunk_id: ch.text for ch in chunks}

    pairs = list(

        candidate_pairs(chunks, outlets=outlets, cross_outlet_only=cross_outlet_only)

    )

    pair_count_before_prune = len(pairs)

    if prune_relatedness:

        pairs, _pruned = prune_pairs_by_relatedness(

            chunks, pairs, floor=relatedness_floor, proposer=proposer

        )



    proposals = proposals_from_pairs(proposer, pairs)

    proposals, _dropped = dedupe_doc_pair_proposals(proposals)

    assessments = [proposer.assess_pair(a, b) for a, b in pairs]



    verifier = (

        make_relation_verifier(proposer, text_by_chunk, min_confidence=min_confidence)

        if relation_recheck

        else None

    )

    bk = Bookkeeper(min_confidence=min_confidence, relation_verifier=verifier)

    results = bk.admit_all(proposals, registry=registry)

    admitted = [

        r.edge

        for r in results

        if r.code == AdmissionCode.ADMITTED and r.edge is not None

    ]

    admitted_contradicts = [

        e for e in admitted if e.relation == Relation.CONTRADICTS

    ]



    machine_gold, curated_gold = split_gold_by_class(gold_pairs(gold_sources))

    gold = machine_gold

    gold_keys = {_key(g.src_chunk_id, g.dst_chunk_id) for g in gold}

    curated_keys = sorted(

        _key(g.src_chunk_id, g.dst_chunk_id) for g in curated_gold

    )

    admitted_keys = {

        _key(e.src_chunk_id, e.dst_chunk_id) for e in admitted_contradicts

    }



    false_positives, anchor_discoveries = split_machine_false_positives(
        admitted_keys, gold_keys, curated_keys
    )

    missed = sorted(gold_keys - admitted_keys)

    doc_tp, doc_fp, doc_missed = doc_level_metrics(admitted_keys, gold_keys)



    metrics = evaluate(assessments, gold)

    class_c, class_c_total, class_c_pass = _class_c_from_proposals(assessments)



    return ScanEvalResult(

        metrics=metrics,

        admitted_contradicts=admitted_contradicts,

        false_positive_keys=false_positives,

        missed_gold_keys=missed,

        class_c_discrimination=class_c,

        class_c_total=class_c_total,

        class_c_pass=class_c_pass,

        gold_count=len(gold_keys),

        admitted_count=len(admitted_keys),

        pair_count_before_prune=pair_count_before_prune,

        pair_count_after_prune=len(pairs),

        doc_tp=doc_tp,

        doc_false_positive_keys=doc_fp,

        doc_missed_gold_keys=doc_missed,

        curated_gold_keys=curated_keys,

        anchor_discovery_keys=anchor_discoveries,

    )


