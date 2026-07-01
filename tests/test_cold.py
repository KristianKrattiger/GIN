"""Tests for cold tier content-addressed storage."""
from gin.corpus import cold


def test_content_hash_stable():
    data = b"RIVERPORT officials responded."
    assert cold.content_hash(data) == cold.content_hash(data)


def test_store_idempotent(tmp_cold_root):
    data = b"hello cold tier"
    digest1, created1 = cold.store(data, tmp_cold_root)
    digest2, created2 = cold.store(data, tmp_cold_root)
    assert digest1 == digest2
    assert created1 is True
    assert created2 is False
    assert cold.load(digest1, tmp_cold_root) == data
