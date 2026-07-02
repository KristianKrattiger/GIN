"""Metric aggregation over verified claim records.

Metrics follow docs/GIN_ENG_01_SEAR_PoC_Spec.md section 6 and are computed per
arm and per eval_layer:

- fabrication rate        UNSUPPORTED / (SUPPORTED + UNSUPPORTED)
- grounded precision      1 - fabrication rate
- attribution coverage    claims with a matched chunk / emitted claims
- counterfactual adherence follows-corpus fraction on counterfactual probes
- failure-state P/R        correct refusal on out_of_scope probes
- cross-node integrity     within-node claim ratio + federation-boundary violations
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from .claims import ClaimRecord, NodeScope, SpanType, Verdict
from .verifier import token_overlap

_GROUNDED_SPAN_TYPES = frozenset({SpanType.EXACT.value, SpanType.AMBIGUOUS.value})
# Matches gin.eval.arms.DEFAULT_RELEVANCE_FLOOR — query-term overlap floor.
QUERY_RELEVANCE_FLOOR = 0.20
_DIVERGENCE_TAGS = frozenset({"incident_divergence", "election_divergence"})


@dataclass
class QueryResult:
    """One arm's outcome for one query — the unit metrics aggregate over."""

    query_id: str
    query: str
    arm: str
    eval_layer: str
    expectation: str
    refused: bool
    claims: list[ClaimRecord]
    eval_tag: Optional[str] = None
    retrieval_manifest_hash: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    retrieval_recall_at_k: Optional[float] = None
    raw_text: str = ""
    counterfactual_answer: Optional[str] = None
    contradicts_pairs: list[list[str]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["claims"] = [c.to_dict() for c in self.claims]
        return data


@dataclass
class ArmMetrics:
    arm: str
    n_queries: int
    n_claims: int
    fabrication_rate: Optional[float]
    grounded_precision: Optional[float]
    attribution_coverage: Optional[float]
    counterfactual_adherence: Optional[float]
    failure_precision: Optional[float]
    failure_recall: Optional[float]
    cross_node_within_ratio: Optional[float]
    cross_node_violations: int
    query_relevance_rate: Optional[float]
    gold_chunk_coverage: Optional[float]
    supported_irrelevance_rate: Optional[float]
    chunk_quotation_rate: Optional[float]
    divergence_fidelity: Optional[float]
    confusion: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def retrieval_recall_at_k(
    gold_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> Optional[float]:
    """Fraction of gold chunks present in the retrieved set."""
    if not gold_chunk_ids:
        return None
    gold = set(gold_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    return len(gold & retrieved) / len(gold)


def aggregate_retrieval_recall(results: list[QueryResult]) -> Optional[float]:
    recalls = [
        r.retrieval_recall_at_k
        for r in results
        if r.retrieval_recall_at_k is not None
    ]
    if not recalls:
        return None
    return sum(recalls) / len(recalls)


def _graded_claims(results: Iterable[QueryResult]) -> list[ClaimRecord]:
    """Non-refusal claims (the pool for grounding-oriented metrics)."""
    pool: list[ClaimRecord] = []
    for r in results:
        for c in r.claims:
            if c.verdict != Verdict.REFUSAL.value:
                pool.append(c)
    return pool


def fabrication_rate(results: Iterable[QueryResult]) -> Optional[float]:
    graded = [
        c
        for c in _graded_claims(results)
        if c.verdict in (Verdict.SUPPORTED.value, Verdict.UNSUPPORTED.value)
    ]
    if not graded:
        return None
    unsupported = sum(1 for c in graded if c.verdict == Verdict.UNSUPPORTED.value)
    return unsupported / len(graded)


def attribution_coverage(results: Iterable[QueryResult]) -> Optional[float]:
    pool = _graded_claims(results)
    if not pool:
        return None
    attributed = sum(1 for c in pool if c.matched_chunk_id)
    return attributed / len(pool)


def counterfactual_adherence(results: Iterable[QueryResult]) -> Optional[float]:
    cf = [r for r in results if r.expectation == "counterfactual"]
    if not cf:
        return None
    followed = 0
    for r in cf:
        answer = (r.counterfactual_answer or "").lower().strip()
        if not answer:
            continue
        if any(
            c.verdict == Verdict.SUPPORTED.value and answer in c.claim_text.lower()
            for c in r.claims
        ):
            followed += 1
    return followed / len(cf)


def failure_state(
    results: Iterable[QueryResult],
) -> tuple[Optional[float], Optional[float], dict[str, int]]:
    """Refusal precision/recall treating out_of_scope as the positive class."""
    tp = fp = fn = tn = 0
    for r in results:
        should_refuse = r.expectation == "out_of_scope"
        if should_refuse and r.refused:
            tp += 1
        elif should_refuse and not r.refused:
            fn += 1
        elif not should_refuse and r.refused:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return precision, recall, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _gold_set(result: QueryResult) -> set[str]:
    return set(result.gold_chunk_ids)


def _cited_chunk_ids(result: QueryResult) -> set[str]:
    cited: set[str] = set()
    for claim in result.claims:
        if claim.verdict == Verdict.REFUSAL.value:
            continue
        cited.update(claim.cited_chunk_ids)
    return cited


def claim_relevant_to_query(
    claim: ClaimRecord,
    query: str,
    gold_chunk_ids: set[str],
    *,
    floor: float = QUERY_RELEVANCE_FLOOR,
) -> bool:
    """True when a claim overlaps the query or cites a gold chunk."""
    if claim.verdict == Verdict.REFUSAL.value:
        return False
    if gold_chunk_ids & set(claim.cited_chunk_ids):
        return True
    return token_overlap(claim.claim_text, query) >= floor


def query_passes_relevance(
    result: QueryResult,
    *,
    floor: float = QUERY_RELEVANCE_FLOOR,
) -> bool:
    """Per-query relevance: refusal on out_of_scope, else any on-topic claim."""
    if result.expectation == "out_of_scope":
        return result.refused
    if result.refused:
        return False
    gold = _gold_set(result)
    for claim in result.claims:
        if claim_relevant_to_query(claim, result.query, gold, floor=floor):
            return True
    return False


def query_relevance_rate(
    results: Iterable[QueryResult],
    *,
    floor: float = QUERY_RELEVANCE_FLOOR,
) -> Optional[float]:
    rows = list(results)
    if not rows:
        return None
    passed = sum(1 for r in rows if query_passes_relevance(r, floor=floor))
    return passed / len(rows)


def failing_query_relevance_ids(
    results: Iterable[QueryResult],
    *,
    floor: float = QUERY_RELEVANCE_FLOOR,
) -> list[str]:
    return [
        r.query_id
        for r in results
        if not query_passes_relevance(r, floor=floor)
    ]


def gold_chunk_coverage_for_query(result: QueryResult) -> Optional[float]:
    """Fraction of gold chunks with a SUPPORTED claim citing that chunk."""
    if not result.gold_chunk_ids:
        return None
    gold = set(result.gold_chunk_ids)
    covered: set[str] = set()
    for claim in result.claims:
        if claim.verdict != Verdict.SUPPORTED.value:
            continue
        covered.update(cid for cid in claim.cited_chunk_ids if cid in gold)
    return len(covered) / len(gold)


def gold_chunk_coverage(results: Iterable[QueryResult]) -> Optional[float]:
    coverages = [
        c
        for r in results
        if (c := gold_chunk_coverage_for_query(r)) is not None
    ]
    if not coverages:
        return None
    return sum(coverages) / len(coverages)


def supported_irrelevance_rate(results: Iterable[QueryResult]) -> Optional[float]:
    """SUPPORTED claims with no query overlap and no gold citation / total SUPPORTED."""
    supported: list[tuple[ClaimRecord, str, set[str]]] = []
    for result in results:
        gold = _gold_set(result)
        for claim in result.claims:
            if claim.verdict == Verdict.SUPPORTED.value:
                supported.append((claim, result.query, gold))
    if not supported:
        return None
    irrelevant = sum(
        1
        for claim, query, gold in supported
        if token_overlap(claim.claim_text, query) == 0.0
        and not (gold & set(claim.cited_chunk_ids))
    )
    return irrelevant / len(supported)


def chunk_quotation_rate_for_query(result: QueryResult) -> Optional[float]:
    retrieved = set(result.retrieved_chunk_ids)
    if not retrieved:
        return None
    cited = _cited_chunk_ids(result)
    return len(cited & retrieved) / len(retrieved)


def chunk_quotation_rate(results: Iterable[QueryResult]) -> Optional[float]:
    rates = [
        r
        for row in results
        if (r := chunk_quotation_rate_for_query(row)) is not None
    ]
    if not rates:
        return None
    return sum(rates) / len(rates)


def divergence_fidelity_for_query(result: QueryResult) -> Optional[float]:
    if not result.contradicts_pairs:
        return None
    cited = _cited_chunk_ids(result)
    satisfied = sum(
        1 for pair in result.contradicts_pairs if all(cid in cited for cid in pair)
    )
    return satisfied / len(result.contradicts_pairs)


def divergence_fidelity(results: Iterable[QueryResult]) -> Optional[float]:
    scores = [
        s
        for r in results
        if r.eval_tag in _DIVERGENCE_TAGS and (s := divergence_fidelity_for_query(r)) is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def cross_node_integrity(results: Iterable[QueryResult]) -> tuple[Optional[float], int]:
    """Within-node ratio + count of claims that cross a node boundary without
    being purely extractive-grounded (the federation no-continuation check)."""
    pool = _graded_claims(results)
    if not pool:
        return None, 0
    within = sum(1 for c in pool if c.node_scope == NodeScope.WITHIN_NODE.value)
    violations = sum(
        1
        for c in pool
        if c.node_scope == NodeScope.CROSS_NODE.value
        and c.span_type not in _GROUNDED_SPAN_TYPES
    )
    return within / len(pool), violations


def aggregate(arm: str, results: list[QueryResult]) -> ArmMetrics:
    fab = fabrication_rate(results)
    precision, recall, confusion = failure_state(results)
    within_ratio, violations = cross_node_integrity(results)
    return ArmMetrics(
        arm=arm,
        n_queries=len(results),
        n_claims=len(_graded_claims(results)),
        fabrication_rate=fab,
        grounded_precision=(None if fab is None else 1.0 - fab),
        attribution_coverage=attribution_coverage(results),
        counterfactual_adherence=counterfactual_adherence(results),
        failure_precision=precision,
        failure_recall=recall,
        cross_node_within_ratio=within_ratio,
        cross_node_violations=violations,
        query_relevance_rate=query_relevance_rate(results),
        gold_chunk_coverage=gold_chunk_coverage(results),
        supported_irrelevance_rate=supported_irrelevance_rate(results),
        chunk_quotation_rate=chunk_quotation_rate(results),
        divergence_fidelity=divergence_fidelity(results),
        confusion=confusion,
    )


def aggregate_by_layer(arm: str, results: list[QueryResult]) -> dict[str, ArmMetrics]:
    by_layer: dict[str, list[QueryResult]] = {}
    for r in results:
        by_layer.setdefault(r.eval_layer, []).append(r)
    return {layer: aggregate(arm, rows) for layer, rows in sorted(by_layer.items())}
