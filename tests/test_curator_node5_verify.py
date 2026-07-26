"""Surfacing gate: every authored pair must reach the curator backlog."""
from gin.curator.node5_verify import authored_pair_chunk_ids, verify_surfacing


def _manifest():
    return [
        {
            "event": "e1", "domain": "incident", "shared_lede": "L.",
            "reports": [
                {"outlet": "A", "published": "t1", "chunks": ["L. a"]},
                {"outlet": "B", "published": "t2", "chunks": ["L. b"]},
            ],
            "intent": [
                {"pair": ["A", "B"], "kind": "conflict", "varied_fact": "n"},
            ],
        },
        {
            "event": "e2", "domain": "incident", "shared_lede": "M.",
            "reports": [
                {"outlet": "A", "published": "t1", "chunks": ["M. a"]},
                {"outlet": "B", "published": "t2", "chunks": ["M. b"]},
            ],
            "intent": [
                {"pair": ["A", "B"], "kind": "corroboration", "varied_fact": None},
            ],
        },
    ]


def test_authored_pairs_map_to_chunk_ids_in_build_order():
    pairs = authored_pair_chunk_ids(_manifest())
    # Loader-normalised ids ("n5_doc_001:0"), not the raw JSON "n5_doc_001_c000".
    assert pairs == [
        ("n5_doc_001:0", "n5_doc_002:0", "conflict"),
        ("n5_doc_003:0", "n5_doc_004:0", "corroboration"),
    ]


def test_all_surfaced_passes():
    offered = {
        frozenset(("n5_doc_001:0", "n5_doc_002:0")),
        frozenset(("n5_doc_003:0", "n5_doc_004:0")),
    }
    report = verify_surfacing(_manifest(), offered)
    report_ok = report["passed"]
    assert report_ok is True
    assert report["missing"] == []


def test_a_missing_negative_fails_as_loudly_as_a_missing_conflict():
    # The negatives are why this corpus exists. If only conflicts surface, the
    # curator never labels a same-story non-contradiction.
    offered = {frozenset(("n5_doc_001:0", "n5_doc_002:0"))}
    report = verify_surfacing(_manifest(), offered)
    assert report["passed"] is False
    assert report["missing"] == [
        ("n5_doc_003:0", "n5_doc_004:0", "corroboration")
    ]


def test_missing_by_kind_is_reported():
    report = verify_surfacing(_manifest(), set())
    assert report["missing_by_kind"] == {"conflict": 1, "corroboration": 1}
    assert report["authored"] == 2
