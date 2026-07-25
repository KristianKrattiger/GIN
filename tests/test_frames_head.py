"""Head training, persistence, and the manifest compatibility gate."""
import json
from dataclasses import replace

import numpy as np
import pytest

from gin.frames.head import (
    HEAD_FILENAME,
    HEAD_KINDS,
    MANIFEST_FILENAME,
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
    manifest = _manifest()
    save_head(tmp_path, model, manifest)
    loaded, loaded_manifest = load_head(tmp_path)
    assert list(loaded.predict(X)) == list(before)
    # Full equality, not just one field: the only field save_head is allowed to
    # change from what the caller passed in is head_sha256 (computed on write).
    assert loaded_manifest.head_sha256 != ""
    assert loaded_manifest == replace(manifest, head_sha256=loaded_manifest.head_sha256)


def test_manifest_json_round_trip():
    m = _manifest()
    # Go through real json.dumps/json.loads, not just asdict(), so a field that
    # survives dataclass round-tripping but breaks under real JSON encoding
    # (e.g. tuple vs list, non-string dict keys) would be caught.
    reloaded = Manifest.from_json(json.loads(json.dumps(m.to_json())))
    assert reloaded == m


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


def test_stale_pairing_is_a_hard_error_when_joblib_is_tampered(tmp_path):
    # Simulates a crash between the joblib write and the manifest write during
    # a retrain: the bytes on disk no longer match what the manifest attests to.
    X, y = _data()
    save_head(tmp_path, train_head(X, y, seed=0), _manifest())
    head_path = tmp_path / HEAD_FILENAME
    head_path.write_bytes(head_path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_head(tmp_path)


def test_stale_pairing_is_a_hard_error_when_manifest_digest_is_wrong(tmp_path):
    X, y = _data()
    save_head(tmp_path, train_head(X, y, seed=0), _manifest())
    manifest_path = tmp_path / MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["head_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_head(tmp_path)
