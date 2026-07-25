"""Frozen embeddings + order-invariant pair features."""
import zlib

import numpy as np

from gin.frames.dataset import FrameExample
from gin.frames.encoder import ChunkEncoder, feature_matrix, pair_features
from gin.frames.labels import FrameClass


def _stub(dim=4):
    """Deterministic pseudo-embedding, no model download."""
    def encode(text: str):
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        vec = rng.normal(size=dim)
        return vec / np.linalg.norm(vec)
    return encode


def test_pair_features_are_symmetric():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 4.0])
    assert np.allclose(pair_features(a, b), pair_features(b, a))


def test_pair_features_have_three_blocks():
    a, b = np.ones(5), np.zeros(5)
    assert pair_features(a, b).shape == (15,)


def test_pair_feature_blocks_are_abs_diff_product_mean():
    a, b = np.array([3.0, 1.0]), np.array([1.0, 5.0])
    got = pair_features(a, b)
    assert np.allclose(got[:2], [2.0, 4.0])     # |a-b|
    assert np.allclose(got[2:4], [3.0, 5.0])    # a*b
    assert np.allclose(got[4:], [2.0, 3.0])     # (a+b)/2


def test_encoder_caches_per_text():
    calls = []

    def counting(text):
        calls.append(text)
        return [1.0, 0.0]

    enc = ChunkEncoder(encode_fn=counting)
    enc.encode("same")
    enc.encode("same")
    enc.encode("other")
    assert calls == ["same", "other"]


def test_encoder_returns_float_array():
    enc = ChunkEncoder(encode_fn=lambda t: [1, 0, 0])
    vec = enc.encode("x")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64


def test_feature_matrix_shapes_and_labels():
    examples = [
        FrameExample("a:0", "b:0", "alpha", "beta", FrameClass.DIVERGENT),
        FrameExample("c:0", "d:0", "gamma", "delta", FrameClass.AGREE),
    ]
    X, y = feature_matrix(examples, ChunkEncoder(encode_fn=_stub(4)))
    assert X.shape == (2, 12)
    assert list(y) == ["DIVERGENT", "AGREE"]


def test_feature_matrix_is_row_order_invariant_per_example():
    enc = ChunkEncoder(encode_fn=_stub(4))
    forward = FrameExample("a:0", "b:0", "alpha", "beta", FrameClass.AGREE)
    reversed_ = FrameExample("b:0", "a:0", "beta", "alpha", FrameClass.AGREE)
    Xf, _ = feature_matrix([forward], enc)
    Xr, _ = feature_matrix([reversed_], enc)
    assert np.allclose(Xf, Xr)


def test_encoder_returns_defensive_copy_not_cached_reference():
    """Regression test: mutating returned array must not corrupt cache."""
    enc = ChunkEncoder(encode_fn=_stub(4))

    # Encode text and get the first result
    first = enc.encode("test_text")
    first_copy = first.copy()

    # Mutate the returned array in place (this should NOT affect cache)
    first += 100.0

    # Encode the same text again
    second = enc.encode("test_text")

    # The second result should be identical to the first (before mutation)
    assert np.allclose(second, first_copy), (
        "Cached embedding was corrupted by in-place mutation of returned array"
    )
