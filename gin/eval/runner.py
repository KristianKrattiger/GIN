"""Experiment orchestration: run arms over a query set, score, and report.

The runner is model-agnostic — it takes any ``llm`` accepted by the arms and
any ``Verifier`` — so it can be driven by a live Llama or a fake in tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .arms import REFUSAL_SENTINEL, Arm, ArmOutput
from .claims import ClaimRecord, NodeScope, SpanType, Verdict, node_scope_for
from .metrics import (
    ArmMetrics,
    QueryResult,
    aggregate,
    aggregate_by_layer,
    aggregate_retrieval_recall,
    failing_query_relevance_ids,
    retrieval_recall_at_k,
)
from .queryset import EvalQuery
from .verifier import ClaimVerdict, Verifier

_EXTRACTIVE = frozenset({SpanType.EXACT.value, SpanType.AMBIGUOUS.value})


def _normalize_for_substring(text: str) -> str:
    return " ".join(text.lower().split())


def _exact_span_verdict(
    claim_text: str,
    cited_chunk_ids: list[str],
    chunk_text_by_id: dict[str, str],
) -> Optional[ClaimVerdict]:
    """Structural shortcut: verbatim EXACT spans are SUPPORTED by construction."""
    normalized = _normalize_for_substring(claim_text)
    if not normalized:
        return None
    for chunk_id in cited_chunk_ids:
        chunk_text = chunk_text_by_id.get(chunk_id, "")
        if normalized in _normalize_for_substring(chunk_text):
            return ClaimVerdict(Verdict.SUPPORTED.value, chunk_id, 1.0)
    return None


def _chunks_for_verification(
    raw,
    chunk_text_by_id: dict[str, str],
) -> list[tuple[str, str]]:
    """Prefer cited chunks for RAG claims; fall back to full retrieval set."""
    if raw.cited_chunk_ids:
        cited = [
            (cid, chunk_text_by_id[cid])
            for cid in raw.cited_chunk_ids
            if cid in chunk_text_by_id
        ]
        if cited:
            return cited
    return list(chunk_text_by_id.items())


@dataclass
class RunMeta:
    generated_at: str
    model: str
    verifier_mode: str
    threshold: float
    queryset: str
    arms: list[str] = field(default_factory=list)
    n_queries: int = 0
    n_gpu_layers: int = 0
    wall_clock_seconds_per_query: Optional[float] = None
    tokens_per_second: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "model": self.model,
            "verifier_mode": self.verifier_mode,
            "threshold": self.threshold,
            "queryset": self.queryset,
            "arms": self.arms,
            "n_queries": self.n_queries,
            "n_gpu_layers": self.n_gpu_layers,
            "wall_clock_seconds_per_query": self.wall_clock_seconds_per_query,
            "tokens_per_second": self.tokens_per_second,
        }


def _refusal_record() -> ClaimRecord:
    return ClaimRecord(
        claim_text=REFUSAL_SENTINEL,
        verdict=Verdict.REFUSAL.value,
        matched_chunk_id=None,
        score=0.0,
        node_scope=NodeScope.NONE.value,
        span_type=SpanType.REFUSAL.value,
        cited_chunk_ids=[],
    )


def verify_output(output: ArmOutput, verifier: Verifier) -> list[ClaimRecord]:
    """Turn an arm's raw claims into scored, node-classified claim records."""
    node_of = lambda cid: output.node_of.get(cid, "unknown")
    chunk_text_by_id = dict(output.chunks)
    records: list[ClaimRecord] = []
    for raw in output.claims:
        verdict = None
        if raw.span_type == SpanType.EXACT.value:
            verdict = _exact_span_verdict(
                raw.text, raw.cited_chunk_ids, chunk_text_by_id
            )
        if verdict is None:
            chunks = _chunks_for_verification(raw, chunk_text_by_id)
            verdict = verifier.verify(raw.text, chunks)
            if (
                verdict.verdict != Verdict.SUPPORTED.value
                and raw.cited_chunk_ids
                and chunks != list(chunk_text_by_id.items())
            ):
                verdict = verifier.verify(raw.text, list(chunk_text_by_id.items()))
        if raw.span_type in _EXTRACTIVE:
            # Extractive spans carry their real sources — authoritative for the
            # federation node-boundary check.
            ground_ids = raw.cited_chunk_ids
        elif verdict.matched_chunk_id:
            ground_ids = [verdict.matched_chunk_id]
        else:
            ground_ids = []
        records.append(
            ClaimRecord(
                claim_text=raw.text,
                verdict=verdict.verdict,
                matched_chunk_id=verdict.matched_chunk_id,
                score=verdict.score,
                node_scope=node_scope_for(ground_ids, node_of).value,
                span_type=raw.span_type,
                cited_chunk_ids=raw.cited_chunk_ids,
            )
        )
    return records


def run_query(
    arm: Arm,
    query: EvalQuery,
    llm: Any,
    verifier: Verifier,
) -> QueryResult:
    base = dict(
        query_id=query.id,
        query=query.query,
        arm=arm.name,
        eval_layer=query.eval_layer,
        eval_tag=query.eval_tag,
        expectation=query.expectation,
        counterfactual_answer=query.counterfactual_answer,
        contradicts_pairs=[list(pair) for pair in query.contradicts_pairs],
    )
    config = getattr(arm, "config", None)
    if config is not None:
        config.gold_chunk_ids = list(query.gold_chunk_ids)
    try:
        output = arm.run(query.query, llm)
    except NotImplementedError as exc:
        return QueryResult(refused=False, claims=[], error=str(exc), **base)

    if output.refused:
        retrieved_ids = [cid for cid, _ in output.chunks]
        return QueryResult(
            refused=True,
            claims=[_refusal_record()],
            retrieval_manifest_hash=output.retrieval_manifest_hash,
            retrieved_chunk_ids=retrieved_ids,
            gold_chunk_ids=list(query.gold_chunk_ids),
            retrieval_recall_at_k=retrieval_recall_at_k(
                query.gold_chunk_ids, retrieved_ids
            ),
            raw_text=output.raw_text,
            **base,
        )

    records = verify_output(output, verifier)
    retrieved_ids = [cid for cid, _ in output.chunks]
    return QueryResult(
        refused=False,
        claims=records,
        retrieval_manifest_hash=output.retrieval_manifest_hash,
        retrieved_chunk_ids=retrieved_ids,
        gold_chunk_ids=list(query.gold_chunk_ids),
        retrieval_recall_at_k=retrieval_recall_at_k(
            query.gold_chunk_ids, retrieved_ids
        ),
        raw_text=output.raw_text,
        **base,
    )


def run_experiment(
    queries: list[EvalQuery],
    arms: dict[str, Arm],
    llm: Any,
    verifier: Verifier,
) -> dict[str, list[QueryResult]]:
    results: dict[str, list[QueryResult]] = {name: [] for name in arms}
    for query in queries:
        for name, arm in arms.items():
            results[name].append(run_query(arm, query, llm, verifier))
    return results


# --- Reporting --------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


_METRIC_ROWS = [
    ("fabrication_rate", lambda m: _fmt(m.fabrication_rate)),
    ("grounded_precision", lambda m: _fmt(m.grounded_precision)),
    ("attribution_coverage", lambda m: _fmt(m.attribution_coverage)),
    ("counterfactual_adherence", lambda m: _fmt(m.counterfactual_adherence)),
    ("failure_precision", lambda m: _fmt(m.failure_precision)),
    ("failure_recall", lambda m: _fmt(m.failure_recall)),
    ("cross_node_within_ratio", lambda m: _fmt(m.cross_node_within_ratio)),
    ("cross_node_violations", lambda m: str(m.cross_node_violations)),
    ("n_claims", lambda m: str(m.n_claims)),
    ("n_queries", lambda m: str(m.n_queries)),
]


_EPISTEMIC_ROWS = [
    ("query_relevance_rate", lambda m: _fmt(m.query_relevance_rate)),
    ("gold_chunk_coverage", lambda m: _fmt(m.gold_chunk_coverage)),
    ("supported_irrelevance_rate", lambda m: _fmt(m.supported_irrelevance_rate)),
    ("chunk_quotation_rate", lambda m: _fmt(m.chunk_quotation_rate)),
    ("divergence_fidelity", lambda m: _fmt(m.divergence_fidelity)),
]


def _metric_table(
    arm_names: list[str],
    metrics: dict[str, ArmMetrics],
    rows: list[tuple[str, Any]] | None = None,
) -> list[str]:
    rows = rows or _METRIC_ROWS
    header = "| Metric | " + " | ".join(arm_names) + " |"
    sep = "|" + "---|" * (len(arm_names) + 1)
    lines = [header, sep]
    for label, getter in rows:
        cells = [getter(metrics[a]) for a in arm_names if a in metrics]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def format_report(
    results_by_arm: dict[str, list[QueryResult]],
    meta: RunMeta,
) -> str:
    arm_names = list(results_by_arm.keys())
    overall = {a: aggregate(a, rows) for a, rows in results_by_arm.items()}

    lines = ["# SEAR vs RAG Evaluation Report", ""]
    lines.append(f"- Generated: {meta.generated_at}")
    lines.append(f"- Model: {meta.model}")
    lines.append(f"- Verifier: {meta.verifier_mode} (threshold {meta.threshold})")
    lines.append(f"- Query set: {meta.queryset}")
    lines.append(f"- Queries: {meta.n_queries}")
    if meta.n_gpu_layers:
        lines.append(f"- GPU layers: {meta.n_gpu_layers}")
    if meta.wall_clock_seconds_per_query is not None:
        lines.append(
            f"- Wall-clock per query: {meta.wall_clock_seconds_per_query:.2f}s"
        )
    if meta.tokens_per_second is not None:
        lines.append(f"- Tokens/s (approx): {meta.tokens_per_second:.1f}")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.extend(_metric_table(arm_names, overall))
    lines.append("")

    layers: set[str] = set()
    by_layer_by_arm: dict[str, dict[str, ArmMetrics]] = {}
    for arm, rows in results_by_arm.items():
        by_layer_by_arm[arm] = aggregate_by_layer(arm, rows)
        layers.update(by_layer_by_arm[arm].keys())

    lines.append("## By eval_layer")
    lines.append("")
    for layer in sorted(layers):
        lines.append(f"### {layer}")
        lines.append("")
        layer_metrics = {
            arm: by_layer_by_arm[arm][layer]
            for arm in arm_names
            if layer in by_layer_by_arm[arm]
        }
        lines.extend(_metric_table(list(layer_metrics.keys()), layer_metrics))
        lines.append("")

    lines.append("## Epistemic quality")
    lines.append("")
    lines.extend(_metric_table(arm_names, overall, _EPISTEMIC_ROWS))
    lines.append("")
    for arm in arm_names:
        failed = failing_query_relevance_ids(results_by_arm[arm])
        if failed:
            lines.append(
                f"- **{arm}** failing query relevance: {', '.join(failed)}"
            )
    lines.append("")

    lines.append("## Retrieval quality")
    lines.append("")
    lines.append("| Query | gold_recall@k | retrieved | gold |")
    lines.append("|---|---|---|---|")
    query_rows = results_by_arm[arm_names[0]]
    for row in query_rows:
        gold = ", ".join(row.gold_chunk_ids) if row.gold_chunk_ids else "—"
        retrieved = ", ".join(row.retrieved_chunk_ids) if row.retrieved_chunk_ids else "—"
        recall = _fmt(row.retrieval_recall_at_k)
        lines.append(f"| {row.query_id} | {recall} | {retrieved} | {gold} |")
    lines.append("")
    for arm in arm_names:
        recall = aggregate_retrieval_recall(results_by_arm[arm])
        lines.append(f"- **{arm}** mean gold recall@k: {_fmt(recall)}")
    lines.append("")

    return "\n".join(lines)


def write_run(
    results_by_arm: dict[str, list[QueryResult]],
    meta: RunMeta,
    out_dir: str | Path,
) -> Path:
    """Persist per-arm query JSON, aggregate metrics JSON, and report.md."""
    run_dir = Path(out_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = run_dir / "results"
    retrieval_dir = run_dir / "retrieval"
    results_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    for arm, rows in results_by_arm.items():
        payload = [r.to_dict() for r in rows]
        (results_dir / f"{arm}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    if results_by_arm:
        first_arm_rows = next(iter(results_by_arm.values()))
        for row in first_arm_rows:
            retrieval_payload = {
                "query_id": row.query_id,
                "query": row.query,
                "eval_layer": row.eval_layer,
                "gold_chunk_ids": row.gold_chunk_ids,
                "retrieved_chunk_ids": row.retrieved_chunk_ids,
                "retrieval_recall_at_k": row.retrieval_recall_at_k,
                "retrieval_manifest_hash": row.retrieval_manifest_hash,
            }
            (retrieval_dir / f"{row.query_id}.json").write_text(
                json.dumps(retrieval_payload, indent=2), encoding="utf-8"
            )

    metrics_payload = {
        arm: {
            "overall": aggregate(arm, rows).to_dict(),
            "by_layer": {
                layer: m.to_dict()
                for layer, m in aggregate_by_layer(arm, rows).items()
            },
        }
        for arm, rows in results_by_arm.items()
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        format_report(results_by_arm, meta), encoding="utf-8"
    )
    return run_dir


def make_meta(
    *,
    model: str,
    verifier_mode: str,
    threshold: float,
    queryset: str,
    arms: list[str],
    n_queries: int,
    n_gpu_layers: int = 0,
    wall_clock_seconds_per_query: Optional[float] = None,
    tokens_per_second: Optional[float] = None,
) -> RunMeta:
    return RunMeta(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        verifier_mode=verifier_mode,
        threshold=threshold,
        queryset=queryset,
        arms=arms,
        n_queries=n_queries,
        n_gpu_layers=n_gpu_layers,
        wall_clock_seconds_per_query=wall_clock_seconds_per_query,
        tokens_per_second=tokens_per_second,
    )
