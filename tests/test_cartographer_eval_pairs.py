"""The frozen eval surfaces: the 45-pair eval set and the 14-pair bar pin."""
from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.eval_pairs import BAR_PAIR_IDS, eval_pair_keys


def test_eval_set_is_45_pairs():
    # labeled_set gold (33) union gold_edges pairs; calibration must never
    # train on any of these or its reported accuracy is a restatement.
    assert len(eval_pair_keys()) == 45


def test_eval_keys_are_unordered_pairs():
    for key in eval_pair_keys():
        assert isinstance(key, frozenset)
        assert len(key) == 2


def test_known_eval_members():
    assert frozenset(("inst_em:0", "clim_pledges:0")) in eval_pair_keys()


def test_eval_pair_keys_is_cached():
    assert eval_pair_keys() is eval_pair_keys()


def test_bar_is_exactly_14_pairs():
    sets = default_calibration_sets()
    assert len(sets["issue_frame"]) == 4
    assert len(sets["corroboration"]) == 6
    assert len(sets["unrelated"]) == 4


def test_bar_pairs_are_pinned_by_chunk_id():
    # The escalation bar is pre-registered. If this fails, some change moved a
    # pre-registered eval — revert it rather than updating this expectation.
    sets = default_calibration_sets()
    live = tuple(
        (src, dst, register)
        for group in ("issue_frame", "corroboration", "unrelated")
        for src, dst, register in sets[group]
    )
    assert live == BAR_PAIR_IDS


def test_every_bar_pair_is_in_the_eval_set_or_is_a_control():
    # The 4 issue_frame bar pairs come from curated gold_edges, so they must be
    # inside the eval set; controls are separate tuples and need not be.
    keys = eval_pair_keys()
    for src, dst, _register in default_calibration_sets()["issue_frame"]:
        assert frozenset((src, dst)) in keys
