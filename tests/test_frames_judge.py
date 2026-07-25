"""BiEncoderFrameJudge: FrameJudge contract + structural order invariance."""
import numpy as np

from gin.frames.encoder import ChunkEncoder
from gin.frames.judge import BiEncoderFrameJudge


class _StubModel:
    """Predicts by a deterministic function of the feature vector."""

    def __init__(self, label):
        self._label = label

    def predict(self, X):
        return np.array([self._label] * X.shape[0])


def _stub_encoder(dim=4):
    def encode(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=dim)
        return vec / np.linalg.norm(vec)
    return ChunkEncoder(encode_fn=encode)


def test_emits_the_three_contract_labels():
    for internal, expected in [
        ("DIVERGENT", "DIVERGENT"),
        ("AGREE", "AGREE"),
        ("UNRELATED", "UNRELATED"),
        ("RELATED_UNTYPED", "UNRELATED"),
    ]:
        judge = BiEncoderFrameJudge(_StubModel(internal), _stub_encoder())
        assert judge("alpha text", "beta text") == expected


def test_related_untyped_is_never_emitted():
    judge = BiEncoderFrameJudge(_StubModel("RELATED_UNTYPED"), _stub_encoder())
    assert judge("a", "b") not in {"RELATED_UNTYPED"}


def test_order_invariance_is_structural():
    # Not a trained property: the pair features are symmetric, so this is an
    # identity. direction_flip_count = 0 follows by construction.
    class _Echo:
        def predict(self, X):
            return np.array(["DIVERGENT" if X[0].sum() > 0 else "AGREE"])

    judge = BiEncoderFrameJudge(_Echo(), _stub_encoder())
    for a, b in [("one", "two"), ("longer text here", "x"), ("same", "same")]:
        assert judge(a, b) == judge(b, a)


def test_is_callable_as_a_frame_judge():
    from gin.cartographer.frame_judge import FrameJudge  # noqa: F401  (protocol alias)

    judge = BiEncoderFrameJudge(_StubModel("AGREE"), _stub_encoder())
    assert callable(judge)
    assert isinstance(judge("a", "b"), str)
