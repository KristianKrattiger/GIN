"""Tests for stratified connective vocabulary."""
from sear.connectives import (
    ADDITIVE_PHRASES,
    CONCESSIVE_PHRASES,
    CONTRASTIVE_PHRASES,
    DEFAULT_CONNECTIVE_PHRASES,
    phrases_for_edge_types,
)


def test_phrases_for_contradicts_returns_contrastive_only():
    result = phrases_for_edge_types({"contradicts"})
    assert result == CONTRASTIVE_PHRASES
    assert set(result).isdisjoint(ADDITIVE_PHRASES)
    assert set(result).isdisjoint(CONCESSIVE_PHRASES)


def test_phrases_for_cites_returns_no_contrastive():
    result = phrases_for_edge_types({"cites"})
    assert set(result).isdisjoint(CONTRASTIVE_PHRASES)
    assert set(ADDITIVE_PHRASES) <= set(result)
    assert set(CONCESSIVE_PHRASES) <= set(result)


def test_phrases_for_empty_returns_default():
    result = phrases_for_edge_types(set())
    assert result == DEFAULT_CONNECTIVE_PHRASES


def test_phrases_subset_of_default():
    for edge_types in (
        {"contradicts"},
        {"cites"},
        {"supersedes"},
        {"cites", "supersedes"},
        set(),
        {"unknown"},
    ):
        result = phrases_for_edge_types(edge_types)
        assert set(result) <= set(DEFAULT_CONNECTIVE_PHRASES)
