"""Surfacing gate: every authored pair must reach the curator backlog."""
from pathlib import Path

import yaml

from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node5_verify import authored_pair_chunk_ids, verify_surfacing

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "corpus_node5.json"
MANIFEST = REPO_ROOT / "data" / "curator" / "node5_events.yaml"


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


def test_authored_pair_chunk_ids_match_the_real_corpus():
    # node5_verify re-implements the builder's document numbering and hardcodes
    # chunk position ":0", with nothing pinning the two together -- that seam
    # already produced one bug (id form mismatch vs. the loader's normalised
    # ids). This is model-free: it builds the ids from the real manifest and
    # checks every one is actually present in the real built corpus, without
    # loading any embedding/NLI model.
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    chunks = load_corpus_chunks([CORPUS])
    chunk_ids = {c.chunk_id for c in chunks}
    for src, dst, _kind in authored_pair_chunk_ids(manifest):
        assert src in chunk_ids, f"{src} from authored_pair_chunk_ids not in built corpus"
        assert dst in chunk_ids, f"{dst} from authored_pair_chunk_ids not in built corpus"
