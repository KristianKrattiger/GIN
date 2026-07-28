"""node5 builder: manifest -> corpus dict, validation, composition floor."""
import pytest

from gin.curator.node5_build import (
    NODE_ID,
    VALID_KINDS,
    build_node5,
    compute_global_id,
    pair_inventory,
)


def _event(name, outlets, intent, lede="SHARED LEDE."):
    return {
        "event": name,
        "domain": "incident",
        "shared_lede": lede,
        "reports": [
            {"outlet": o, "published": f"2026-03-0{i + 1}T12:00Z",
             "chunks": [f"{lede} Report from {o}."]}
            for i, o in enumerate(outlets)
        ],
        "intent": intent,
    }


def _minimal_ok():
    """Two events carrying one conflict and one negative — floors relaxed in tests."""
    return [
        _event("e1", ["A", "B"], [{"pair": ["A", "B"], "kind": "conflict",
                                   "varied_fact": "count"}]),
        _event("e2", ["A", "B"], [{"pair": ["A", "B"], "kind": "corroboration",
                                   "varied_fact": None}]),
    ]


def test_node_id_and_schema_match_node1_4():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    assert corpus["node_id"] == NODE_ID
    assert set(corpus) == {"node_id", "documents"}
    doc = corpus["documents"][0]
    assert set(doc) == {"doc_id", "global_id", "source", "url", "node", "metadata", "chunks"}
    assert set(doc["chunks"][0]) == {"chunk_id", "position", "text"}


def test_positions_are_strings():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    assert corpus["documents"][0]["chunks"][0]["position"] == "0"


def test_doc_and_chunk_ids_follow_the_convention():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    assert corpus["documents"][0]["doc_id"] == "n5_doc_001"
    assert corpus["documents"][0]["chunks"][0]["chunk_id"] == "n5_doc_001_c000"


def test_global_id_shape():
    gid = compute_global_id("Some Source", "CentralWire", "2026-03-04T21:10Z")
    assert gid.startswith("gid_")
    assert len(gid) == 4 + 16


def test_global_id_is_deterministic_and_outlet_sensitive():
    a = compute_global_id("S", "CentralWire", "T")
    assert a == compute_global_id("S", "CentralWire", "T")
    assert a != compute_global_id("S", "MetroDaily", "T")


def test_build_is_deterministic():
    assert build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1) == build_node5(
        _minimal_ok(), min_conflicts=1, min_negatives=1
    )


def test_metadata_carries_event_context_not_labels():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    meta = corpus["documents"][0]["metadata"]
    assert set(meta) == {"outlet", "published", "event", "domain"}
    # The intent matrix must never leak a relation label into the corpus.
    blob = str(corpus)
    for banned in ("conflict", "corroboration", "contradicts", "varied_fact"):
        assert banned not in blob


def test_pair_inventory_counts_by_kind():
    inv = pair_inventory(_minimal_ok())
    assert inv == {"conflict": 1, "corroboration": 1}


def test_composition_floor_is_enforced():
    with pytest.raises(ValueError, match="conflict pairs"):
        build_node5(_minimal_ok(), min_conflicts=20, min_negatives=1)
    with pytest.raises(ValueError, match="negative pairs"):
        build_node5(_minimal_ok(), min_conflicts=1, min_negatives=20)


def test_unknown_kind_is_rejected():
    bad = [_event("e1", ["A", "B"], [{"pair": ["A", "B"], "kind": "vibes",
                                      "varied_fact": None}])]
    with pytest.raises(ValueError, match="unknown kind"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_intent_referencing_an_unknown_outlet_is_rejected():
    bad = [_event("e1", ["A", "B"], [{"pair": ["A", "Z"], "kind": "conflict",
                                      "varied_fact": "count"}])]
    with pytest.raises(ValueError, match="unknown outlet"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_duplicate_outlet_within_event_is_rejected():
    # node5_verify's doc_of lookup is keyed by (event, outlet); a duplicate
    # outlet in one event would silently collapse onto the last document.
    bad = [_event("e1", ["A", "A"], [])]
    with pytest.raises(ValueError, match="duplicate outlet"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_missing_shared_lede_is_rejected():
    bad = [{"event": "e", "domain": "incident", "reports": [], "intent": []}]
    with pytest.raises(ValueError, match="shared_lede"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_valid_kinds_are_the_four_from_the_spec():
    assert VALID_KINDS == frozenset(
        {"conflict", "corroboration", "update", "compatible_partial"}
    )


def test_build_is_parametrizable_for_later_nodes():
    """Node6 (and any later synthetic node) reuses this builder with its own
    node id and doc prefix — scripts/build_node6.py depends on these params."""
    corpus = build_node5(
        _minimal_ok(),
        min_conflicts=1,
        min_negatives=1,
        node_id="node_6_samestory",
        doc_prefix="n6_doc",
        url_tag="node6",
    )
    assert corpus["node_id"] == "node_6_samestory"
    assert corpus["documents"][0]["doc_id"] == "n6_doc_001"
    assert corpus["documents"][0]["chunks"][0]["chunk_id"] == "n6_doc_001_c000"
    assert corpus["documents"][0]["url"].startswith("synthetic://node6/")
