"""Curator endpoints: ordered /next, /label append + supersede, class validation."""
from pathlib import Path

from fastapi.testclient import TestClient

from gin.cartographer.models import LabeledChunk
from gin.curator.app import create_curator_app
from gin.curator.candidates import OfflineCandidateSource
from gin.curator.models import pair_key
from gin.curator.store import Store

CHUNKS = [LabeledChunk("a:0", "alpha text"), LabeledChunk("b:0", "bravo text"),
          LabeledChunk("c:0", "charlie text")]


def _fake_signals(a_text: str, b_text: str) -> dict:
    # Deterministic per-pair signals so ordering is assertable without a model.
    table = {
        frozenset({"alpha text", "bravo text"}): {"cosine": 0.55, "nli_p_contra": 0.9,
                                                   "same_story": None, "cheap_verdict": "contradicts"},
        frozenset({"alpha text", "charlie text"}): {"cosine": 0.30, "nli_p_contra": None,
                                                     "same_story": None, "cheap_verdict": "related_untyped"},
        frozenset({"bravo text", "charlie text"}): {"cosine": 0.80, "nli_p_contra": 0.05,
                                                     "same_story": None, "cheap_verdict": "corroborates"},
    }
    return table[frozenset({a_text, b_text})]


def _client(tmp_path: Path) -> TestClient:
    store = Store(tmp_path / "labels.jsonl")
    app = create_curator_app(
        store=store, source=OfflineCandidateSource(CHUNKS), signals_fn=_fake_signals,
    )
    return TestClient(app)


def test_next_returns_hard_cases_first_with_text_and_signals(tmp_path):
    r = _client(tmp_path).get("/curator/next?n=10")
    assert r.status_code == 200
    data = r.json()
    assert data["labeled"] == 0
    first = data["pairs"][0]
    assert {first["src"], first["dst"]} == {"a:0", "b:0"}  # the disagreement pair ranks first
    assert first["src_text"] and first["dst_text"]
    assert first["signals"]["cosine"] == 0.55


def test_label_appends_one_record_and_reflects_in_next(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    app = create_curator_app(store=store, source=OfflineCandidateSource(CHUNKS), signals_fn=_fake_signals)
    client = TestClient(app)
    r = client.post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0",
        "relation": "contradicts", "relation_class": "issue_frame", "rationale": "why",
    })
    assert r.status_code == 200 and r.json()["ok"] is True
    assert len(store.read_log()) == 1
    assert store.fold_current()[pair_key("a:0", "b:0")].relation.value == "contradicts"
    assert client.get("/curator/next?n=10").json()["labeled"] == 1


def test_contradicts_without_class_is_rejected(tmp_path):
    r = _client(tmp_path).post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0", "relation": "contradicts",
    })
    assert r.status_code == 422


def test_unknown_relation_is_rejected(tmp_path):
    r = _client(tmp_path).post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0", "relation": "banana",
    })
    assert r.status_code == 422


def test_relabel_supersedes_prior(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    app = create_curator_app(store=store, source=OfflineCandidateSource(CHUNKS), signals_fn=_fake_signals)
    client = TestClient(app)
    first = client.post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0", "relation": "corroborates",
    }).json()["id"]
    client.post("/curator/label", json={
        "src_chunk_id": "b:0", "dst_chunk_id": "a:0",
        "relation": "contradicts", "relation_class": "story",
    })
    log = store.read_log()
    assert len(log) == 2
    assert log[1].supersedes == first
    assert store.fold_current()[pair_key("a:0", "b:0")].relation.value == "contradicts"


def test_index_page_served(tmp_path):
    r = _client(tmp_path).get("/curator/")
    assert r.status_code == 200
    assert "GIN Curator" in r.text
