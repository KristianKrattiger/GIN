"""Regression guard on the built node5 corpus."""
from pathlib import Path

import yaml

from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node5_build import NODE_ID, pair_inventory

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "corpus_node5.json"
MANIFEST = REPO_ROOT / "data" / "curator" / "node5_events.yaml"


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_corpus_loads_through_the_standard_loader():
    chunks = load_corpus_chunks([CORPUS])
    assert len(chunks) == 38
    assert all(c.chunk_id.startswith("n5_doc_") for c in chunks)


def test_intent_matrix_totals():
    assert pair_inventory(_manifest()) == {
        "conflict": 21, "corroboration": 11, "update": 4, "compatible_partial": 6,
    }


def test_twelve_events_thirty_eight_reports():
    # Ten 3-outlet events and two 4-outlet events.
    m = _manifest()
    assert len(m) == 12
    assert sum(len(e["reports"]) for e in m) == 38
    assert sum(len(e["intent"]) for e in m) == 42


def test_every_report_opens_with_its_events_shared_lede():
    # The shared lede is what makes make_same_story fire; without it the pair is
    # not same-story and the corpus does not test what it was built to test.
    for ev in _manifest():
        for rep in ev["reports"]:
            assert rep["chunks"], f"{ev['event']}/{rep['outlet']} has no chunk"
            assert rep["chunks"][0].startswith(ev["shared_lede"]), (
                f"{ev['event']}/{rep['outlet']} does not open with the shared lede"
            )


def test_update_pairs_are_ordered_in_time():
    # An update that is not later than what it revises is just a conflict.
    for ev in _manifest():
        published = {r["outlet"]: r["published"] for r in ev["reports"]}
        for entry in ev["intent"]:
            if entry["kind"] == "update":
                first, second = entry["pair"]
                assert published[first] != published[second], (
                    f"{ev['event']}: update pair {entry['pair']} shares a timestamp"
                )


def test_corpus_carries_no_relation_labels():
    text = CORPUS.read_text(encoding="utf-8")
    for banned in ("conflict", "corroboration", "compatible_partial", "varied_fact"):
        assert banned not in text
