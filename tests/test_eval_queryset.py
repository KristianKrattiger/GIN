"""Tests for the eval query set loader."""
from pathlib import Path

import pytest

from gin.eval.queryset import EvalQuery, load_query_set, parse_query_set

ROOT = Path(__file__).resolve().parents[1]


def test_load_bundled_queryset():
    queries = load_query_set(ROOT / "data" / "eval" / "queryset.yaml")
    assert queries
    assert all(isinstance(q, EvalQuery) for q in queries)
    expectations = {q.expectation for q in queries}
    assert expectations == {"answerable", "counterfactual", "out_of_scope"}


def test_parse_rejects_bad_expectation():
    with pytest.raises(ValueError):
        parse_query_set([{"id": "x", "query": "q", "eval_layer": "realism", "expectation": "bogus"}])


def test_parse_rejects_duplicate_ids():
    entries = [
        {"id": "dup", "query": "a", "eval_layer": "realism"},
        {"id": "dup", "query": "b", "eval_layer": "realism"},
    ]
    with pytest.raises(ValueError):
        parse_query_set(entries)


def test_parse_defaults_expectation_answerable():
    queries = parse_query_set([{"id": "x", "query": "q", "eval_layer": "realism"}])
    assert queries[0].expectation == "answerable"
    assert queries[0].gold_chunk_ids == []
    assert queries[0].regression is True
    assert queries[0].contradicts_pairs == []


def test_parse_contradicts_pairs_and_regression_flag():
    queries = parse_query_set(
        [
            {
                "id": "x",
                "query": "q",
                "eval_layer": "realism",
                "regression": False,
                "contradicts_pairs": [["a:0", "b:0"]],
            }
        ]
    )
    assert queries[0].regression is False
    assert queries[0].contradicts_pairs == [["a:0", "b:0"]]


def test_load_bundled_queryset_has_twenty_queries():
    from gin.eval.queryset import filter_regression_queries

    queries = load_query_set(ROOT / "data" / "eval" / "queryset.yaml")
    assert len(queries) == 20
    anchors = filter_regression_queries(queries, regression_only=True)
    assert len(anchors) == 9
