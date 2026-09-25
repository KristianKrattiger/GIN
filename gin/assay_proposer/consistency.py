"""Cross-proposal consistency: a claimant/independent quote pair carries one relation type.

The decoder grounds each proposal's quotes on its own, so nothing stops a pass from
proposing both "corroborates" and "updates" for the very same pair. This reports those
pairs; it does not drop them -- deciding which (if either) is right is downstream's job.
"""
from __future__ import annotations


def _pair_key(p: dict) -> tuple[str, str, str, str] | None:
    to = p.get("to")
    if to is None:
        return None
    frm = p["from"]
    return (frm["docId"], frm["quote"].strip(), to["docId"], to["quote"].strip())


def conflicting_pairs(proposals: list[dict]) -> list[dict]:
    """Every quote pair that received more than one distinct relation type, in first-seen order."""
    types: dict[tuple[str, str, str, str], set[str]] = {}
    for p in proposals:
        key = _pair_key(p)
        if key is not None:
            types.setdefault(key, set()).add(p["type"])
    return [
        {"from": {"docId": k[0], "quote": k[1]}, "to": {"docId": k[2], "quote": k[3]}, "types": sorted(ts)}
        for k, ts in types.items()
        if len(ts) > 1
    ]
