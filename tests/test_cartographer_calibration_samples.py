"""Calibration sample file: schema round-trip and the manifest compatibility gate."""
import json

import pytest

from gin.cartographer.calibration_samples import (
    EvalSample,
    Sample,
    SampleManifest,
    load_eval_samples,
    load_samples,
    write_samples,
)
from gin.cartographer.models import Relation


def _eval_samples():
    return [
        EvalSample(src="inst_em:0", dst="clim_pledges:0", cos=0.52,
                   p_contra=0.93, relation=Relation.CORROBORATES, same_story=True),
    ]


def _manifest(embed="embed-x", nli="nli-y"):
    return SampleManifest(
        embed_model=embed, nli_model=nli, n_samples=2,
        class_counts={"contradicts": 1, "unrelated": 1},
        excluded_eval_pairs=45, git_sha="abc1234",
        created_utc="2026-07-25T00:00:00Z",
    )


def _samples():
    return [
        Sample(cos=0.39, p_contra=0.068, relation=Relation.CONTRADICTS, same_story=True),
        Sample(cos=0.10, p_contra=0.010, relation=Relation.UNRELATED, same_story=False),
    ]


def test_round_trip_preserves_samples(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, manifest = load_samples(path)
    assert loaded == _samples()
    assert manifest == _manifest()


def test_eval_samples_round_trip_and_carry_ids(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    assert load_eval_samples(path) == _eval_samples()


def test_load_samples_never_returns_eval_samples(tmp_path):
    # The whole point of the split: calibrate() must not see an eval pair.
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, _ = load_samples(path)
    assert len(loaded) == 2
    assert all(not hasattr(s, "src") for s in loaded)


def test_round_trip_goes_through_real_json(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"manifest", "samples", "eval_samples"}
    assert raw["samples"][0]["relation"] == "contradicts"


def test_embed_model_mismatch_is_a_hard_error(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    with pytest.raises(ValueError, match="embed model mismatch"):
        load_samples(path, expect_embed_model="different-embed")


def test_nli_model_mismatch_is_a_hard_error(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    with pytest.raises(ValueError, match="NLI model mismatch"):
        load_samples(path, expect_nli_model="different-nli")


def test_matching_models_load_cleanly(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, _ = load_samples(path, expect_embed_model="embed-x", expect_nli_model="nli-y")
    assert len(loaded) == 2


def test_missing_file_names_the_regen_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="regen_calibration_samples"):
        load_samples(tmp_path / "absent.json")


def test_manifest_json_round_trip():
    m = _manifest()
    assert SampleManifest.from_json(json.loads(json.dumps(m.to_json()))) == m
