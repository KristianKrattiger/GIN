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


def test_stance_provider_mismatch_is_a_hard_error(tmp_path):
    path = tmp_path / "samples.json"
    manifest = SampleManifest(
        embed_model="embed-x", nli_model="nli-y", n_samples=2,
        class_counts={"contradicts": 1, "unrelated": 1}, excluded_eval_pairs=45,
        git_sha="abc1234", created_utc="2026-07-25T00:00:00Z",
        stance_provider="quantity.stance_for",
    )
    write_samples(path, manifest, _samples(), _eval_samples())
    with pytest.raises(ValueError, match="stance provider mismatch"):
        load_samples(path, expect_stance_provider="something-else")


def test_matching_stance_provider_loads_cleanly(tmp_path):
    path = tmp_path / "samples.json"
    manifest = SampleManifest(
        embed_model="embed-x", nli_model="nli-y", n_samples=2,
        class_counts={"contradicts": 1, "unrelated": 1}, excluded_eval_pairs=45,
        git_sha="abc1234", created_utc="2026-07-25T00:00:00Z",
        stance_provider="quantity.stance_for",
    )
    write_samples(path, manifest, _samples(), _eval_samples())
    loaded, _ = load_samples(path, expect_stance_provider="quantity.stance_for")
    assert len(loaded) == 2


def test_stance_provider_check_is_skipped_when_not_requested(tmp_path):
    # expect_stance_provider=None (the default) means "don't gate" -- the
    # committed 39-sample fixture's manifest carries stance_provider="none"
    # and must keep loading with no expectation passed.
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, manifest = load_samples(path)
    assert manifest.stance_provider == "none"
    assert len(loaded) == 2


def test_missing_file_names_the_regen_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="regen_calibration_samples"):
        load_samples(tmp_path / "absent.json")


def test_manifest_json_round_trip():
    m = _manifest()
    assert SampleManifest.from_json(json.loads(json.dumps(m.to_json()))) == m


def test_empty_samples_array_raises_valueerror(tmp_path):
    # A valid manifest with an empty samples array is a silent-degradation trap.
    path = tmp_path / "samples.json"
    manifest = _manifest()
    write_samples(path, manifest, [], _eval_samples())  # Empty samples
    with pytest.raises(ValueError, match="calibration samples array is empty"):
        load_samples(path)


def test_mismatched_sample_count_raises_valueerror(tmp_path):
    # Manifest says n_samples=5 but file only has 2: file is truncated or corrupted.
    path = tmp_path / "samples.json"
    manifest = SampleManifest(
        embed_model="embed-x", nli_model="nli-y", n_samples=5,
        class_counts={"contradicts": 3, "unrelated": 2},
        excluded_eval_pairs=45, git_sha="abc1234",
        created_utc="2026-07-25T00:00:00Z",
    )
    samples = _samples()  # Only 2 samples
    write_samples(path, manifest, samples, _eval_samples())
    with pytest.raises(ValueError, match="samples count.*does not match"):
        load_samples(path)


def test_stance_round_trips_including_the_none_versus_unaligned_distinction(tmp_path):
    # None and "unaligned" are different answers, not one missing value: None
    # means the stance channel had no quantitative claim to judge and
    # classify_relation falls through to its pre-stance branch, while
    # "unaligned" means it looked and found no shared fact and the branch
    # abstains. Collapsing them in serialization would silently change what a
    # regenerated sample file says the pipeline did.
    path = tmp_path / "samples.json"
    manifest = SampleManifest(
        embed_model="embed-x", nli_model="nli-y", n_samples=2,
        class_counts={"contradicts": 1, "corroborates": 1}, excluded_eval_pairs=0,
        git_sha="abc1234", created_utc="2026-07-26T00:00:00Z",
        stance_provider="quantity.stance_for",
    )
    write_samples(
        path, manifest,
        [
            Sample(cos=0.9, p_contra=0.1, relation=Relation.CONTRADICTS,
                   same_story=True, stance="conflict"),
            Sample(cos=0.8, p_contra=0.1, relation=Relation.CORROBORATES,
                   same_story=True, stance=None),
        ],
        [
            EvalSample(src="a:0", dst="b:0", cos=0.9, p_contra=0.1,
                       relation=Relation.CONTRADICTS, same_story=True,
                       stance="unaligned"),
        ],
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["samples"][1]["stance"] is None, "None must serialize as JSON null"
    assert raw["eval_samples"][0]["stance"] == "unaligned"

    samples, loaded = load_samples(path)
    assert [s.stance for s in samples] == ["conflict", None]
    assert loaded.stance_provider == "quantity.stance_for"
    assert load_eval_samples(path)[0].stance == "unaligned"


def test_the_committed_fixture_manifest_shape_still_loads_without_stance_fields():
    # The 39-sample manifest predates same_story_corpus_size AND stance_provider.
    # from_json hard-errors on missing non-defaulted keys, so defaulting is the
    # only thing keeping that fixture loadable -- a required field would break it.
    manifest = SampleManifest.from_json({
        "embed_model": "e", "nli_model": "n", "n_samples": 39,
        "class_counts": {}, "excluded_eval_pairs": 0,
        "git_sha": "x", "created_utc": "2026-07-25T00:00:00Z",
    })
    assert manifest.stance_provider == "none"
    assert manifest.same_story_corpus_size == 0
