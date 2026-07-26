"""Deterministic builder: event manifest -> corpus_node5.json dict.

Pure and network-free. data/curator/node5_events.yaml is the reviewable
artifact; this turns it into the schema node1-4 use, so load_corpus_chunks and
the whole curator path work unchanged.

The manifest's ``intent`` matrix records which pairs were authored as conflicts
and which as negatives. It drives validation and the surfacing gate and is
DELIBERATELY NOT written into the corpus: the curator labels these pairs, and
shipping the intended answer alongside the text would defeat that. Node4 set the
precedent with ``stance``.
"""
from __future__ import annotations

import hashlib
from collections import Counter

NODE_ID = "node_5_samestory"

# One positive kind and three negative kinds. The negatives are the point: a
# corpus of pure conflicts would confirm combined.py's unconditional
# "same_story => CONTRADICTS" branch rather than test it.
VALID_KINDS = frozenset({"conflict", "corroboration", "update", "compatible_partial"})
NEGATIVE_KINDS = frozenset({"corroboration", "update", "compatible_partial"})

_REQUIRED_EVENT = ("event", "domain", "shared_lede", "reports", "intent")
_REQUIRED_REPORT = ("outlet", "published", "chunks")


def compute_global_id(source: str, outlet: str, published: str) -> str:
    digest = hashlib.sha256(f"{source}|{outlet}|{published}".encode()).hexdigest()
    return "gid_" + digest[:16]


def _validate(manifest: list[dict]) -> None:
    for i, ev in enumerate(manifest):
        for key in _REQUIRED_EVENT:
            if key not in ev:
                raise ValueError(f"event {i} missing required key {key!r}")
        if not ev["shared_lede"]:
            raise ValueError(f"event {ev['event']!r} has an empty shared_lede")
        outlets = set()
        for j, rep in enumerate(ev["reports"]):
            for key in _REQUIRED_REPORT:
                if key not in rep:
                    raise ValueError(
                        f"event {ev['event']!r} report {j} missing key {key!r}"
                    )
            if not rep["chunks"]:
                raise ValueError(f"event {ev['event']!r} report {j} has no chunks")
            if rep["outlet"] in outlets:
                raise ValueError(
                    f"event {ev['event']!r} has duplicate outlet {rep['outlet']!r}"
                )
            outlets.add(rep["outlet"])
        for entry in ev["intent"]:
            if entry["kind"] not in VALID_KINDS:
                raise ValueError(
                    f"event {ev['event']!r} unknown kind {entry['kind']!r} "
                    f"(expected one of {sorted(VALID_KINDS)})"
                )
            for outlet in entry["pair"]:
                if outlet not in outlets:
                    raise ValueError(
                        f"event {ev['event']!r} intent references unknown outlet "
                        f"{outlet!r} (event has {sorted(outlets)})"
                    )


def pair_inventory(manifest: list[dict]) -> dict[str, int]:
    """How many authored pairs of each kind the manifest declares."""
    counts: Counter[str] = Counter()
    for ev in manifest:
        for entry in ev["intent"]:
            counts[entry["kind"]] += 1
    return dict(counts)


def build_node5(
    manifest: list[dict], *, min_conflicts: int = 20, min_negatives: int = 20
) -> dict:
    """Manifest -> corpus dict. Raises ValueError on malformed input or thin composition."""
    _validate(manifest)

    inventory = pair_inventory(manifest)
    n_conflict = inventory.get("conflict", 0)
    n_negative = sum(inventory.get(k, 0) for k in NEGATIVE_KINDS)
    if n_conflict < min_conflicts:
        raise ValueError(
            f"only {n_conflict} conflict pairs authored, need at least {min_conflicts}"
        )
    if n_negative < min_negatives:
        raise ValueError(
            f"only {n_negative} negative pairs authored, need at least {min_negatives}"
        )

    documents: list[dict] = []
    index = 0
    for ev in manifest:
        for rep in ev["reports"]:
            index += 1
            doc_id = f"n5_doc_{index:03d}"
            source = f"{ev['event']} ({rep['outlet']})"
            documents.append(
                {
                    "doc_id": doc_id,
                    "global_id": compute_global_id(source, rep["outlet"], rep["published"]),
                    "source": source,
                    "url": f"synthetic://node5/{ev['event']}/{rep['outlet']}",
                    "node": NODE_ID,
                    "metadata": {
                        "outlet": rep["outlet"],
                        "published": rep["published"],
                        "event": ev["event"],
                        "domain": ev["domain"],
                    },
                    "chunks": [
                        {
                            "chunk_id": f"{doc_id}_c{pos:03d}",
                            "position": str(pos),
                            "text": text,
                        }
                        for pos, text in enumerate(rep["chunks"])
                    ],
                }
            )
    return {"node_id": NODE_ID, "documents": documents}
