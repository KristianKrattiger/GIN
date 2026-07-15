"""Pure trust-gate logic — no I/O, no network, no DB.

A peer is gated out of consideration only if a domain it's known to serve
falls below the configured trust weight; absence of domain information
(no synced summary, or a summary with no tagged domains) never gates a
peer, matching peer_selection.py's "no-summary peers never dropped"
invariant. This is a filter applied AFTER ranking, not a re-ranking: order
is preserved for every peer that passes.
"""
from __future__ import annotations


def is_trusted(
    peer_domains: list[str], peer_weights: dict[str, float], threshold: float
) -> bool:
    """True unless some known domain of this peer falls below threshold.
    An unconfigured domain defaults to full trust (1.0); a peer with no
    known domains passes vacuously."""
    return all(peer_weights.get(d, 1.0) >= threshold for d in peer_domains)


def filter_trusted(
    ranked_peer_ids: list[str],
    domains_by_peer: dict[str, list[str]],
    trust_weights: dict[str, dict[str, float]],
    threshold: float,
) -> list[str]:
    """Ranked peer ids with any gated peer removed; relative order preserved."""
    return [
        nid
        for nid in ranked_peer_ids
        if is_trusted(domains_by_peer.get(nid, []), trust_weights.get(nid, {}), threshold)
    ]
