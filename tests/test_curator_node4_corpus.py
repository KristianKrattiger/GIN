"""corpus_node4.json loads and normalizes like node1-3, with the expected shape."""
import json
from pathlib import Path

from gin.curator.corpus_json import load_corpus_chunks

CORPUS = Path("corpus_node4.json")


def test_loads_and_normalizes():
    chunks = load_corpus_chunks([CORPUS])
    assert chunks, "node4 produced no chunks"
    for c in chunks:
        assert c.chunk_id.startswith("n4_doc_")
        assert ":" in c.chunk_id  # normalized to {doc_id}:{position}
        assert c.text.strip()


def test_twenty_docs_ten_topics_pro_con():
    docs = json.loads(CORPUS.read_text(encoding="utf-8"))["documents"]
    assert len(docs) == 20
    topics = {}
    for d in docs:
        topics.setdefault(d["metadata"]["topic"], []).append(d["metadata"]["stance"])
    assert len(topics) == 10
    for topic, stances in topics.items():
        assert sorted(stances) == ["con", "pro"], topic
