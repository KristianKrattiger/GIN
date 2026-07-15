"""Trust gating: a peer is excluded only if every domain it's known to
serve falls below the configured weight; absence of domain information
never gates; order is preserved for peers that pass."""
from gin.federation.trust_gate import filter_trusted, is_trusted


def test_is_trusted_true_when_all_domains_clear_threshold():
    assert is_trusted(["monetary_policy"], {"monetary_policy": 0.9}, 0.5) is True


def test_is_trusted_false_when_any_domain_below_threshold():
    assert is_trusted(
        ["monetary_policy", "inflation"],
        {"monetary_policy": 0.9, "inflation": 0.1},
        0.5,
    ) is False


def test_is_trusted_true_for_unconfigured_domain_default_full_trust():
    # No entry for this domain -> implicit weight 1.0, clears any threshold <= 1.0.
    assert is_trusted(["monetary_policy"], {}, 0.5) is True


def test_is_trusted_true_for_no_known_domains():
    # Absence of domain information never gates.
    assert is_trusted([], {"monetary_policy": 0.0}, 0.5) is True


def test_filter_trusted_excludes_gated_peer_preserves_order():
    order = filter_trusted(
        ["node_c", "node_b"],
        {"node_c": ["monetary_policy"], "node_b": ["environmental_impact"]},
        {"node_c": {"monetary_policy": 0.1}},
        0.5,
    )
    assert order == ["node_b"]


def test_filter_trusted_keeps_peer_with_no_domain_entry():
    # node_b has no entry in domains_by_peer at all (e.g. no synced summary).
    order = filter_trusted(
        ["node_c", "node_b"],
        {"node_c": ["monetary_policy"]},
        {"node_c": {"monetary_policy": 0.1}},
        0.5,
    )
    assert order == ["node_b"]


def test_filter_trusted_empty_trust_weights_keeps_everyone():
    order = filter_trusted(
        ["node_c", "node_b"],
        {"node_c": ["monetary_policy"], "node_b": ["environmental_impact"]},
        {},
        0.5,
    )
    assert order == ["node_c", "node_b"]
