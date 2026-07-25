"""Head training, persistence, and the manifest compatibility gate."""
import numpy as np
import pytest

from gin.frames.head import (
    HEAD_KINDS,
    Manifest,
    build_estimator,
    load_head,
    save_head,
    train_head,
)


def _data(n=40, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    labels = np.array(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"] * (n // 4))
    for offset, name in enumerate(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"]):
        X[labels == name] += offset * 5.0
    return X, labels


def _manifest(dim=6, kind="linear"):
    return Manifest(
        encoder_model="stub-encoder", feature_dim=dim,
        classes=["AGREE", "DIVERGENT", "RELATED_UNTYPED", "UNRELATED"],
        kind=kind, seed=0, n_train=40,
        class_counts={"DIVERGENT": 10, "AGREE": 10, "RELATED_UNTYPED": 10, "UNRELATED": 10},
        git_sha="abc1234", created_utc="2026-07-24T00:00:00Z",
    )


def test_both_head_kinds_are_supported():
    assert HEAD_KINDS == ("linear", "mlp")
    for kind in HEAD_KINDS:
        assert build_estimator(kind, seed=0) is not None


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown head kind"):
        build_estimator("transformer", seed=0)


def test_trains_and_predicts_all_four_classes():
    X, y = _data()
    model = train_head(X, y, kind="linear", seed=0)
    assert set(model.classes_) == {"DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"}


def test_round_trip_reproduces_identical_predictions(tmp_path):
    X, y = _data()
    model = train_head(X, y, kind="linear", seed=0)
    before = model.predict(X)
    save_head(tmp_path, model, _manifest())
    loaded, manifest = load_head(tmp_path)
    assert list(loaded.predict(X)) == list(before)
    assert manifest.encoder_model == "stub-encoder"


def test_manifest_json_round_trip():
    m = _manifest()
    assert Manifest.from_json(m.to_json()) == m


def test_encoder_mismatch_is_a_hard_error(tmp_path):
    X, y = _data()
    save_head(tmp_path, train_head(X, y, seed=0), _manifest())
    with pytest.raises(ValueError, match="encoder mismatch"):
        load_head(tmp_path, expect_encoder="different-encoder")


def test_feature_dim_mismatch_is_a_hard_error(tmp_path):
    X, y = _data()
    save_head(tmp_path, train_head(X, y, seed=0), _manifest())
    with pytest.raises(ValueError, match="feature dim mismatch"):
        load_head(tmp_path, expect_dim=999)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_head(tmp_path)
