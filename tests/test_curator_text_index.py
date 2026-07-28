"""df_corpus_texts: the same-story df corpus must not double-count node texts.

verify_node5_surfacing.py and curator_serve.py used to concatenate their
chunks' texts onto default_text_index(), which already contains every
CORPUS_NODES corpus. Doubling a node's document frequencies pushes its tokens
above the rare ceiling and MASKS cross-event false positives — it reported the
union anchor mode at 0/5 on node5 when the truth was 4/5. See
docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md, "Known defect".
"""
from __future__ import annotations

from collections import Counter

from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.text_index import CORPUS_NODES, default_text_index, df_corpus_texts


def test_registered_node_texts_are_not_double_counted():
    node5 = [c.text for c in load_corpus_chunks([CORPUS_NODES[4]])]
    texts = df_corpus_texts(node5)
    counts = Counter(texts)
    index_counts = Counter(default_text_index().values())
    for text in node5:
        assert counts[text] == index_counts[text], (
            "a registered node5 text appears more often in the df corpus than "
            "in the index — document frequencies are doubled again"
        )


def test_unregistered_texts_are_added_exactly_once():
    novel = "A text that exists in no registered corpus node at all."
    texts = df_corpus_texts([novel, novel])
    assert Counter(texts)[novel] == 1


def test_index_is_always_included():
    index_texts = set(default_text_index().values())
    texts = set(df_corpus_texts([]))
    assert index_texts <= texts
