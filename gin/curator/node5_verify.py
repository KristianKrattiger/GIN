"""Hard gate: every authored node5 pair must reach the curator backlog.

Conflicts and negatives are held to the same standard, and that is the point. If
only the conflicts surface, the curator never labels a same-story
non-contradiction, and the corpus cannot falsify combined.py's unconditional
"same_story => CONTRADICTS" branch — which is the reason it was built.
"""
from __future__ import annotations

from collections import Counter


def authored_pair_chunk_ids(manifest: list[dict]) -> list[tuple[str, str, str]]:
    """(src_chunk_id, dst_chunk_id, kind) per authored pair, in build order.

    Mirrors node5_build's document numbering: events in order, reports within an
    event in order, one document each.
    """
    # NOTE the id form. corpus_json.load_corpus_chunks NORMALISES the raw
    # "n5_doc_001_c000" written into the JSON to "n5_doc_001:0", and the
    # candidate source yields the normalised form. Emitting the raw form here
    # would make every pair look missing.
    doc_of: dict[tuple[str, str], str] = {}
    index = 0
    for ev in manifest:
        for rep in ev["reports"]:
            index += 1
            doc_of[(ev["event"], rep["outlet"])] = f"n5_doc_{index:03d}:0"

    pairs: list[tuple[str, str, str]] = []
    for ev in manifest:
        for entry in ev["intent"]:
            a, b = entry["pair"]
            pairs.append(
                (doc_of[(ev["event"], a)], doc_of[(ev["event"], b)], entry["kind"])
            )
    return pairs


def verify_surfacing(manifest: list[dict], offered_pairs: set[frozenset]) -> dict:
    """Which authored pairs reached the backlog, and which did not."""
    authored = authored_pair_chunk_ids(manifest)
    missing = [
        (src, dst, kind)
        for src, dst, kind in authored
        if frozenset((src, dst)) not in offered_pairs
    ]
    return {
        "authored": len(authored),
        "surfaced": len(authored) - len(missing),
        "missing": missing,
        "missing_by_kind": dict(Counter(kind for _s, _d, kind in missing)),
        "passed": not missing,
    }
