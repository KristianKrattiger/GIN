"""Evaluation query set: schema and YAML loader.

A query set is the shared stimulus for every arm in the designed experiment.
Each entry declares an ``expectation`` so the harness knows how to score it:

- ``answerable``    the corpus contains the answer; arms should ground it.
- ``counterfactual`` the corpus deliberately contradicts a likely model prior;
                     arms should follow the corpus, not the prior.
- ``out_of_scope``  the corpus cannot answer; arms should refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

VALID_EXPECTATIONS = frozenset({"answerable", "counterfactual", "out_of_scope"})


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    eval_layer: str
    eval_tag: Optional[str] = None
    expectation: str = "answerable"
    gold_chunk_ids: list[str] = field(default_factory=list)
    counterfactual_answer: Optional[str] = None
    contradicts_pairs: list[list[str]] = field(default_factory=list)
    regression: bool = True

    def __post_init__(self) -> None:
        if self.expectation not in VALID_EXPECTATIONS:
            raise ValueError(
                f"query {self.id!r}: expectation {self.expectation!r} not in "
                f"{sorted(VALID_EXPECTATIONS)}"
            )


def _parse_entry(raw: dict) -> EvalQuery:
    try:
        qid = raw["id"]
        query = raw["query"]
        eval_layer = raw["eval_layer"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"query set entry missing required key: {exc}") from exc

    return EvalQuery(
        id=str(qid),
        query=str(query),
        eval_layer=str(eval_layer),
        eval_tag=raw.get("eval_tag"),
        expectation=str(raw.get("expectation", "answerable")),
        gold_chunk_ids=list(raw.get("gold_chunk_ids", []) or []),
        counterfactual_answer=raw.get("counterfactual_answer"),
        contradicts_pairs=[
            list(pair) for pair in (raw.get("contradicts_pairs") or [])
        ],
        regression=bool(raw.get("regression", True)),
    )


def parse_query_set(data: dict | list) -> list[EvalQuery]:
    """Parse an already-loaded YAML/JSON structure into EvalQuery objects."""
    if isinstance(data, dict):
        entries = data.get("queries", [])
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError("query set must contain a list of queries")

    parsed = [_parse_entry(entry) for entry in entries]
    seen: set[str] = set()
    for q in parsed:
        if q.id in seen:
            raise ValueError(f"duplicate query id: {q.id!r}")
        seen.add(q.id)
    return parsed


def load_query_set(path: str | Path) -> list[EvalQuery]:
    """Load and validate a query set from a YAML file."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return []
    return parse_query_set(data)


def filter_regression_queries(
    queries: list[EvalQuery],
    *,
    regression_only: bool,
) -> list[EvalQuery]:
    """Keep regression anchors only, or the full expanded set."""
    if not regression_only:
        return queries
    return [q for q in queries if q.regression]
