"""Cartographer batch scan — propose, admit, persist."""

from __future__ import annotations



import itertools

import time

from dataclasses import dataclass, field

from pathlib import Path

from typing import Callable, Iterable, Optional, Sequence



import psycopg



from gin.bookkeeper import AdmissionCode, Bookkeeper

from gin.bookkeeper.persist import sync_admissions

from gin.bookkeeper.relation_verify import RelationVerifyResult, verify_contradicts

from gin.cartographer.combined import CombinedRelationProposer

from gin.cartographer.models import EdgeProposal, GRAPH_EDGE_RELATIONS, LabeledChunk, Relation

from gin.cartographer.relatedness import (
    DEFAULT_RELATEDNESS_FLOOR,
    RelatednessGate,
    make_same_story,
)

from gin.corpus.db import transaction



DEFAULT_EXCLUDED_DOC_IDS: tuple[str, ...] = ("out_of_scope_stub",)





def whitespace_token_count(text: str) -> int:

    return len(text.split())





def doc_id_from_chunk(chunk_id: str) -> str:

    return chunk_id.split(":", 1)[0]





def sentence_anchor(text: str) -> tuple[int, int]:

    """Whitespace-token span covering the first sentence (or full chunk)."""

    words = text.split()

    if not words:

        return (0, 0)

    for i, w in enumerate(words):

        if w.endswith((".", "!", "?")):

            return (0, i + 1)

    return (0, len(words))





def filter_chunks(

    chunks: list[LabeledChunk],

    outlets: dict[str, str],

    *,

    exclude_doc_ids: Optional[Sequence[str]] = None,

) -> tuple[list[LabeledChunk], dict[str, str]]:

    if not exclude_doc_ids:

        return chunks, outlets

    excluded = frozenset(exclude_doc_ids)

    kept = [ch for ch in chunks if doc_id_from_chunk(ch.chunk_id) not in excluded]

    kept_ids = {ch.chunk_id for ch in kept}

    return kept, {cid: o for cid, o in outlets.items() if cid in kept_ids}





def chunks_from_db(

    conn: psycopg.Connection,

    *,

    doc_id: Optional[str] = None,

    exclude_doc_ids: Optional[Sequence[str]] = None,

) -> tuple[list[LabeledChunk], dict[str, str]]:

    if doc_id:

        rows = conn.execute(

            """

            SELECT c.chunk_id, c.text, d.outlet

            FROM chunks c

            JOIN documents d ON d.doc_id = c.doc_id

            WHERE c.chunk_id LIKE %s

            ORDER BY c.chunk_id

            """,

            (f"{doc_id}:%",),

        ).fetchall()

    else:

        rows = conn.execute(

            """

            SELECT c.chunk_id, c.text, d.outlet

            FROM chunks c

            JOIN documents d ON d.doc_id = c.doc_id

            ORDER BY c.chunk_id

            """

        ).fetchall()

    chunks = [LabeledChunk(chunk_id=r[0], text=r[1]) for r in rows]

    outlets = {r[0]: r[2] or "" for r in rows}

    return filter_chunks(chunks, outlets, exclude_doc_ids=exclude_doc_ids)





def candidate_pairs(

    chunks: list[LabeledChunk],

    *,

    outlets: Optional[dict[str, str]] = None,

    cross_outlet_only: bool = False,

) -> Iterable[tuple[LabeledChunk, LabeledChunk]]:

    for a, b in itertools.combinations(chunks, 2):

        if cross_outlet_only and outlets is not None:

            if outlets.get(a.chunk_id, "") == outlets.get(b.chunk_id, ""):

                continue

        yield a, b





def prune_pairs_by_relatedness(
    chunks: list[LabeledChunk],
    pairs: list[tuple[LabeledChunk, LabeledChunk]],
    *,
    floor: float = DEFAULT_RELATEDNESS_FLOOR,
    proposer: Optional[CombinedRelationProposer] = None,
) -> tuple[list[tuple[LabeledChunk, LabeledChunk]], int]:
    """Stage-1 prune: IDF overlap floor, or embedding cosine when proposer given."""
    gate = RelatednessGate(chunks, floor=floor)
    embed_floor = proposer.thresholds.gate_floor if proposer is not None else None
    kept: list[tuple[LabeledChunk, LabeledChunk]] = []
    pruned = 0
    for a, b in pairs:
        idf_ok = gate.assess_pair(a, b).relation == Relation.RELATED_UNTYPED
        embed_ok = (
            proposer is not None
            and embed_floor is not None
            and proposer.embedding_cosine(a.text, b.text) >= embed_floor
        )
        if idf_ok or embed_ok:
            kept.append((a, b))
        else:
            pruned += 1
    return kept, pruned





def wire_same_story(
    proposer: CombinedRelationProposer, chunks: list[LabeledChunk]
) -> None:
    """Wire the stage-1 same-story provider from the scanned corpus.

    Story-gates both contradicts channels (combined.classify_relation): built
    from the chunks actually under scan so token rarity reflects this corpus.
    An injected provider is left untouched.
    """
    if proposer.same_story is None:
        proposer.same_story = make_same_story([ch.text for ch in chunks])



def proposals_from_pairs(

    proposer: CombinedRelationProposer,

    pairs: Iterable[tuple[LabeledChunk, LabeledChunk]],

) -> list[EdgeProposal]:

    proposals: list[EdgeProposal] = []

    for a, b in pairs:

        assessment = proposer.assess_pair(a, b)

        if assessment.relation not in GRAPH_EDGE_RELATIONS:

            continue

        proposals.append(

            EdgeProposal.from_assessment(

                assessment,

                src_anchor=sentence_anchor(a.text),

                dst_anchor=sentence_anchor(b.text),

            )

        )

    return proposals





CURATED_CONFIDENCE = 0.95


def curated_issue_frame_proposals(
    sources: "Sequence[Path]",
    text_by_chunk: Optional[dict[str, str]] = None,
) -> list[EdgeProposal]:
    """Load hand-curated issue_frame contradicts edges as EdgeProposals.

    Only the issue_frame class is ingested: story-class edges are
    machine-recoverable and stay the scan's job (2026-07-12 signal audit —
    issue-frame pairs share no story entities and no on-hand model detects
    them). Anchors are derived from chunk texts when available.
    """
    from gin.cartographer.gold_edges import load_all_gold_contradicts

    proposals: list[EdgeProposal] = []
    for edge in load_all_gold_contradicts(sources):
        if edge.relation_class != "issue_frame":
            continue
        src_anchor = dst_anchor = None
        if text_by_chunk is not None:
            src_text = text_by_chunk.get(edge.src_chunk_id)
            dst_text = text_by_chunk.get(edge.dst_chunk_id)
            src_anchor = sentence_anchor(src_text) if src_text else None
            dst_anchor = sentence_anchor(dst_text) if dst_text else None
        proposals.append(
            EdgeProposal(
                src_chunk_id=edge.src_chunk_id,
                dst_chunk_id=edge.dst_chunk_id,
                relation=Relation.CONTRADICTS,
                method="curated:issue_frame",
                confidence=CURATED_CONFIDENCE,
                src_anchor=src_anchor,
                dst_anchor=dst_anchor,
            )
        )
    return proposals


def _proposal_rank(p: EdgeProposal) -> tuple[float, float]:
    """Prefer NLI propositional channel over band framing signal, then confidence."""
    nli_first = 1.0 if p.method.endswith(":nli") else 0.0
    return (nli_first, p.confidence)


def dedupe_doc_pair_proposals(proposals: list[EdgeProposal]) -> tuple[list[EdgeProposal], int]:
    """Keep at most one contradicts proposal per unordered doc pair (best rank)."""
    best_by_docs: dict[frozenset[str], EdgeProposal] = {}
    other: list[EdgeProposal] = []
    dropped = 0
    for p in proposals:
        if p.relation != Relation.CONTRADICTS:
            other.append(p)
            continue
        doc_key = frozenset(
            {doc_id_from_chunk(p.src_chunk_id), doc_id_from_chunk(p.dst_chunk_id)}
        )
        prev = best_by_docs.get(doc_key)
        if prev is None or _proposal_rank(p) > _proposal_rank(prev):
            if prev is not None:
                dropped += 1
            best_by_docs[doc_key] = p
        else:
            dropped += 1
    return other + list(best_by_docs.values()), dropped





def make_relation_verifier(

    proposer: CombinedRelationProposer,

    text_by_chunk: dict[str, str],

    *,

    min_confidence: float,

) -> Callable[[EdgeProposal], RelationVerifyResult]:

    def _nli_scores(a_text: str, b_text: str) -> tuple[float, float, float]:

        scorer = proposer._nli_scores or proposer._nli_model_scores  # noqa: SLF001

        return scorer(a_text, b_text)



    def _verify(proposal: EdgeProposal) -> RelationVerifyResult:

        return verify_contradicts(

            proposal,

            src_text=text_by_chunk[proposal.src_chunk_id],

            dst_text=text_by_chunk[proposal.dst_chunk_id],

            nli_scores=_nli_scores,

        )



    return _verify





@dataclass

class ScanResult:

    counts: dict[str, int]

    admitted_edges: list[dict] = field(default_factory=list)

    elapsed_seconds: float = 0.0

    pair_count: int = 0





def run_scan(

    *,

    doc_id: Optional[str] = None,

    min_confidence: float = 0.5,

    dry_run: bool = False,

    cross_outlet_only: bool = False,

    prune_relatedness: bool = True,

    relatedness_floor: float = DEFAULT_RELATEDNESS_FLOOR,

    exclude_doc_ids: Optional[Sequence[str]] = None,

    relation_recheck: bool = True,

    proposer: Optional[CombinedRelationProposer] = None,

    curated_sources: Optional[Sequence[Path]] = None,

    escalation_judge: Optional[Callable[[str, str], str]] = None,

    escalation_cos_floor: Optional[float] = None,

    escalation_method_suffix: str = "unknown",

    conn: Optional[psycopg.Connection] = None,

) -> ScanResult:

    """Scan corpus chunks, admit proposals, optionally persist."""

    proposer = proposer or CombinedRelationProposer()

    if exclude_doc_ids is None:

        exclude_doc_ids = DEFAULT_EXCLUDED_DOC_IDS

    started = time.perf_counter()



    def _execute(c: psycopg.Connection) -> ScanResult:

        chunks, outlets = chunks_from_db(

            c, doc_id=doc_id, exclude_doc_ids=exclude_doc_ids

        )

        if len(chunks) < 2:

            return ScanResult(

                counts={"chunks": len(chunks), "proposals": 0, "pairs": 0},

                elapsed_seconds=time.perf_counter() - started,

            )



        wire_same_story(proposer, chunks)

        registry = {ch.chunk_id: whitespace_token_count(ch.text) for ch in chunks}

        text_by_chunk = {ch.chunk_id: ch.text for ch in chunks}

        pairs = list(

            candidate_pairs(chunks, outlets=outlets, cross_outlet_only=cross_outlet_only)

        )

        pair_count_before_prune = len(pairs)

        pruned_unrelated = 0

        if prune_relatedness:

            pairs, pruned_unrelated = prune_pairs_by_relatedness(

                chunks, pairs, floor=relatedness_floor, proposer=proposer

            )



        proposals = proposals_from_pairs(proposer, pairs)

        proposals, doc_pair_dropped = dedupe_doc_pair_proposals(proposals)

        curated_count = 0

        if curated_sources:

            # After dedupe: curated issue_frame edges are human-asserted and

            # must not compete with scan proposals for their doc pair.

            curated = curated_issue_frame_proposals(curated_sources, text_by_chunk)

            proposals = proposals + curated

            curated_count = len(curated)

        escalated_count = 0

        if escalation_judge is not None:

            from gin.cartographer.escalation import (

                DEFAULT_ESCALATION_COS_FLOOR,

                escalate_proposals,

                escalation_candidates,

            )

            floor = (

                escalation_cos_floor

                if escalation_cos_floor is not None

                else DEFAULT_ESCALATION_COS_FLOOR

            )

            esc_pairs = escalation_candidates(pairs, proposer, cos_floor=floor)

            escalated = escalate_proposals(
                esc_pairs,
                escalation_judge,
                method_suffix=escalation_method_suffix,
            )

            escalated, _esc_dropped = dedupe_doc_pair_proposals(escalated)

            # A doc pair the cheap path (or curation) already typed keeps its

            # higher-evidence edge; the judge only fills uncovered doc pairs.

            covered = {

                frozenset(

                    {doc_id_from_chunk(p.src_chunk_id), doc_id_from_chunk(p.dst_chunk_id)}

                )

                for p in proposals

                if p.relation == Relation.CONTRADICTS

            }

            escalated = [

                p

                for p in escalated

                if frozenset(

                    {doc_id_from_chunk(p.src_chunk_id), doc_id_from_chunk(p.dst_chunk_id)}

                )

                not in covered

            ]

            proposals = proposals + escalated

            escalated_count = len(escalated)



        verifier = (

            make_relation_verifier(proposer, text_by_chunk, min_confidence=min_confidence)

            if relation_recheck

            else None

        )

        bk = Bookkeeper(

            min_confidence=min_confidence,

            relation_verifier=verifier,

        )

        results = bk.admit_all(proposals, registry=registry)



        notes = {

            (p.src_chunk_id, p.dst_chunk_id, p.relation.value): p.rationale

            for p in proposals

        }

        counts: dict[str, int] = {

            "chunks": len(chunks),

            "pairs": pair_count_before_prune,

            "pairs_after_prune": len(pairs),

            "pruned_unrelated": pruned_unrelated,

            "doc_pair_dropped": doc_pair_dropped,

            "proposals": len(proposals),

            "cross_outlet_only": int(cross_outlet_only),

            "prune_relatedness": int(prune_relatedness),

            "relation_recheck": int(relation_recheck),

            "curated_proposals": curated_count,

            "escalated_proposals": escalated_count,

        }



        admitted_edges: list[dict] = []

        for r in results:

            counts[r.code.value] = counts.get(r.code.value, 0) + 1

            if r.code == AdmissionCode.ADMITTED and r.edge is not None:

                e = r.edge

                admitted_edges.append(

                    {

                        "src_chunk_id": e.src_chunk_id,

                        "dst_chunk_id": e.dst_chunk_id,

                        "relation": e.relation.value,

                        "confidence": e.provenance.confidence,

                        "proposer": e.provenance.proposer,

                    }

                )



        if not dry_run:

            admission_counts = sync_admissions(c, results, notes=notes)

            for code, n in admission_counts.items():

                counts[code.value] = n



        return ScanResult(

            counts=counts,

            admitted_edges=admitted_edges,

            elapsed_seconds=time.perf_counter() - started,

            pair_count=len(pairs),

        )



    if conn is not None:

        return _execute(conn)

    with transaction() as c:

        return _execute(c)


