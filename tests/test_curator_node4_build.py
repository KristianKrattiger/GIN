"""Deterministic manifest -> corpus_node4 builder."""
import pytest

from gin.curator.node4_build import build_node4, compute_global_id


def _entry(topic, stance, n=2, **over):
    e = {
        "topic": topic, "stance": stance,
        "source": f"{topic} {stance} source", "author": f"{topic} author",
        "date": "2023-05", "url": f"https://example.org/{topic}/{stance}",
        "domain": "climate_policy", "type": "opinion",
        "chunks": [f"{topic} {stance} claim {i}" for i in range(n)],
    }
    e.update(over)
    return e


def _pair(topic):
    return [_entry(topic, "pro"), _entry(topic, "con")]


def test_global_id_matches_known_anchor():
    got = compute_global_id(
        "Indigenous Environmental Network: Frontline Communities Demand Real Climate Solutions",
        "Indigenous Environmental Network", "2023-12",
    )
    assert got == "gid_f5842fdb72d6327a"


def test_builds_node_id_and_doc_ids_pro_then_con():
    out = build_node4(_pair("carbon_tax") + _pair("nuclear_power"))
    assert out["node_id"] == "node_4_contested"
    docs = out["documents"]
    assert [d["doc_id"] for d in docs] == [
        "n4_doc_001", "n4_doc_002", "n4_doc_003", "n4_doc_004",
    ]
    assert docs[0]["metadata"]["stance"] == "pro"
    assert docs[1]["metadata"]["stance"] == "con"
    assert docs[0]["metadata"]["topic"] == "carbon_tax"
    assert docs[0]["node"] == "node_4_contested"


def test_chunk_ids_and_string_positions():
    out = build_node4(_pair("carbon_tax"))
    chunks = out["documents"][0]["chunks"]
    assert [c["chunk_id"] for c in chunks] == ["n4_doc_001_c000", "n4_doc_001_c001"]
    assert [c["position"] for c in chunks] == ["0", "1"]
    assert chunks[0]["text"] == "carbon_tax pro claim 0"


def test_computes_global_id_per_doc():
    out = build_node4(_pair("carbon_tax"))
    d = out["documents"][0]
    assert d["global_id"] == compute_global_id(d["source"], d["metadata"]["author"], d["metadata"]["date"])


def test_missing_key_raises():
    bad = _pair("carbon_tax")
    del bad[0]["url"]
    with pytest.raises(ValueError, match="url"):
        build_node4(bad)


def test_bad_stance_raises():
    with pytest.raises(ValueError, match="stance"):
        build_node4([_entry("carbon_tax", "maybe"), _entry("carbon_tax", "con")])


def test_topic_not_exactly_pro_con_raises():
    with pytest.raises(ValueError, match="carbon_tax"):
        build_node4([_entry("carbon_tax", "pro"), _entry("carbon_tax", "pro")])


def test_non_adjacent_topic_raises():
    manifest = [_entry("carbon_tax", "pro"), _entry("nuclear_power", "pro"),
                _entry("carbon_tax", "con"), _entry("nuclear_power", "con")]
    with pytest.raises(ValueError, match="adjacent"):
        build_node4(manifest)


def test_global_id_collision_raises():
    dup = _pair("carbon_tax")
    dup[1]["source"] = dup[0]["source"]
    dup[1]["author"] = dup[0]["author"]
    dup[1]["date"] = dup[0]["date"]
    with pytest.raises(ValueError, match="global_id"):
        build_node4(dup)
