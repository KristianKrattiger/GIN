# Same-Story Corpus (node5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synthetic same-story corpus containing both propositional conflicts and same-story negatives, surface it to the curator, and verify every authored pair arrives.

**Architecture:** Two-stage and deterministic, matching node4 exactly — a hand-authored, reviewable event manifest feeds a pure network-free builder that emits `corpus_node5.json` in the same schema node1–4 use. A new `SameStoryCandidateSource` surfaces the pairs, because the existing residue source filters *to* the not-same-story band. A hard surfacing gate fails on a missing negative as loudly as on a missing conflict.

**Tech Stack:** Python 3.12, PyYAML, sentence-transformers + `cross-encoder/nli-deberta-v3-xsmall` (surfacing smoke only), pytest.

## Global Constraints

- **Layering:** `gin.curator` may import `gin.cartographer`. `gin.cartographer` must **never** import `gin.curator` — check with `git grep -nE '^\s*(from|import)\s+gin\.curator' -- gin/cartographer` (no output = clean). Nothing may import `gin.frames`. Scripts may import anything.
- **Strictly additive:** node1–4, `gold_edges.py`, `escalation_eval.py`, `scan_eval.py`, `evaluation.py`, `labeled_set.py` and `gin/frames/` are not modified. The escalation bar's 14 pairs must not move — `tests/test_cartographer_eval_pairs.py` must still pass.
- **Model-free by default:** only `scripts/verify_node5_surfacing.py` may load models. Every test passes without a model download.
- **The manifest never carries relation labels.** It records divergence *intent* (`kind`, `varied_fact`) as build/verify metadata only. Encoding the intended answer would defeat the curation this corpus exists to enable. Node4 set this precedent with `stance`.
- **Corpus schema** is identical to node1–4: top level `{"node_id", "documents"}`; each document `{"doc_id", "global_id", "source", "url", "node", "metadata", "chunks"}`; each chunk `{"chunk_id", "position", "text"}` with `position` a **string**.
- **`node_id` is `"node_5_samestory"`.** Doc ids are `n5_doc_001`…; chunk ids **in the JSON** are `n5_doc_00X_c000`. Note `corpus_json.load_corpus_chunks` **normalises** these on load to `n5_doc_00X:0` — anything comparing against loaded chunks (the candidate source, the surfacing gate) must use the normalised form.
- **`global_id = "gid_" + sha256(f"{source}|{outlet}|{published}").hexdigest()[:16]`** — same shape as `gin/curator/node4_build.py:compute_global_id`.
- **Composition floor:** the built corpus must yield **≥20 conflict pairs and ≥20 same-story negative pairs**. The builder hard-errors otherwise.
- Run all commands from the repo root with `venv/Scripts/python.exe`. Scripts importing through `gin.curator.store` need the repo-root `sys.path` prelude used by `scripts/frames_probe.py`.
- Full suite currently passes at **622 passed / 16 skipped / 0 failed**.

## File Structure

| File | Responsibility |
|------|----------------|
| `gin/curator/node5_build.py` | Pure manifest→corpus builder, validation, composition floor |
| `gin/curator/node5_verify.py` | Surfacing verification over a candidate source |
| `gin/curator/same_story.py` | `SameStoryCandidateSource` — selects and ranks same-story pairs |
| `gin/curator/readiness.py` | Modified: `story` target and count |
| `scripts/build_node5.py` | Thin CLI for the builder |
| `scripts/verify_node5_surfacing.py` | Thin CLI for the gate (loads models) |
| `scripts/curator_serve.py` | Modified: `--source same-story` |
| `data/curator/node5_events.yaml` | The authored, reviewable event manifest |
| `corpus_node5.json` | Generated corpus (committed, like node4) |
| `tests/test_curator_node5_build.py` | Builder validation, determinism, schema |
| `tests/test_curator_node5_corpus.py` | Regression guard on the built corpus |
| `tests/test_curator_same_story.py` | Candidate source, model-free |
| `tests/test_curator_node5_verify.py` | Surfacing verification logic |

---

### Task 1: Correct the `hf_*` labels from `issue_frame` to `story`

Independent of the rest, and first so downstream counts are stable.

**Files:**
- Create: `gin/curator/relabel_hf.py`
- Create: `scripts/relabel_hf_story.py`
- Test: `tests/test_curator_relabel_hf.py`

**Interfaces:**
- Consumes: `gin.curator.store.Store`, `gin.curator.models.{LabelRecord, pair_key}`, `gin.cartographer.models.Relation`
- Produces: `HF_STORY_RELABEL: dict[tuple[str, str], str]`, `relabel_hf_to_story(store: Store, *, curator: str = "relabel") -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_curator_relabel_hf.py`:

```python
"""hf_af_*/hf_kc_* are story, not issue_frame — restoring guide/seed consistency."""
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.relabel_hf import HF_STORY_RELABEL, relabel_hf_to_story
from gin.curator.store import Store


def _rec(src, dst, relation_class, rid="orig-1"):
    return LabelRecord(
        id=rid, src_chunk_id=src, dst_chunk_id=dst, relation=Relation.CONTRADICTS,
        relation_class=relation_class, rationale="", curator="backfill",
        ts="2026-07-24T00:00:00Z",
    )


def test_map_covers_exactly_the_two_housing_pairs():
    assert len(HF_STORY_RELABEL) == 2
    assert set(HF_STORY_RELABEL.values()) == {"story"}
    assert pair_key("hf_af_staff:0", "hf_af_tenants:0") in HF_STORY_RELABEL
    assert pair_key("hf_kc_inspection:0", "hf_kc_tenants:0") in HF_STORY_RELABEL


def test_relabels_issue_frame_to_story(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("hf_af_staff:0", "hf_af_tenants:0", "issue_frame"))
    assert relabel_hf_to_story(store) == 1
    current = store.fold_current()[pair_key("hf_af_staff:0", "hf_af_tenants:0")]
    assert current.relation_class == "story"
    assert current.relation is Relation.CONTRADICTS
    assert current.supersedes == "orig-1"


def test_is_idempotent(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("hf_af_staff:0", "hf_af_tenants:0", "issue_frame"))
    assert relabel_hf_to_story(store) == 1
    assert relabel_hf_to_story(store) == 0
    assert len(store.read_log()) == 2


def test_leaves_already_story_alone(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("hf_af_staff:0", "hf_af_tenants:0", "story"))
    assert relabel_hf_to_story(store) == 0


def test_does_not_touch_other_pairs(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0", "issue_frame", rid="keep"))
    assert relabel_hf_to_story(store) == 0
    assert store.fold_current()[pair_key("inst_em:0", "grass_em:0")].relation_class == "issue_frame"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_relabel_hf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.relabel_hf'`

- [ ] **Step 3: Write the implementation**

Create `gin/curator/relabel_hf.py`:

```python
"""Correct two housing pairs from issue_frame back to story.

Sub-project B's Task 1 backfill classified hf_af_* (Alder Flats rezoning) and
hf_kc_* (Kestrel Court habitability) as issue_frame. Two independent sources say
story: the labeling guide, corrected against labels.jsonl on 2026-07-20
(b9e0079), lists rezoning and habitability as story examples; and gold_edges
labels the same content story under its long-form ids (hf_alderflats_*,
hf_kestrel_*). Both pairs pass make_same_story, which is what story means.

Appended as superseding records, never edited in place — the same mechanism the
original backfill used, so the earlier judgment stays auditable.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from gin.cartographer.models import Relation

from .models import LabelRecord, pair_key
from .store import Store

HF_STORY_RELABEL: dict[tuple[str, str], str] = {
    pair_key("hf_af_staff:0", "hf_af_tenants:0"): "story",
    pair_key("hf_kc_inspection:0", "hf_kc_tenants:0"): "story",
}

RATIONALE = (
    "relabel: same-story institutional-vs-community divergence; matches the "
    "labeling guide's rezoning/habitability story examples and the gold_edges "
    "long-form ids. Supersedes an issue_frame backfill."
)


def relabel_hf_to_story(store: Store, *, curator: str = "relabel") -> int:
    """Append superseding records fixing the two housing pairs. Idempotent."""
    current = store.fold_current()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    appended = 0
    for key, relation_class in sorted(HF_STORY_RELABEL.items()):
        rec = current.get(key)
        if rec is None:
            continue
        if rec.relation is not Relation.CONTRADICTS or rec.relation_class == relation_class:
            continue
        store.append(
            LabelRecord(
                id=str(uuid.uuid4()),
                src_chunk_id=rec.src_chunk_id,
                dst_chunk_id=rec.dst_chunk_id,
                relation=Relation.CONTRADICTS,
                relation_class=relation_class,
                rationale=RATIONALE,
                curator=curator,
                ts=ts,
                supersedes=rec.id,
                src_anchor=rec.src_anchor,
                dst_anchor=rec.dst_anchor,
            )
        )
        appended += 1
    return appended
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_relabel_hf.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Write the CLI**

Create `scripts/relabel_hf_story.py`:

```python
"""Relabel the two housing pairs from issue_frame to story.

    venv/Scripts/python.exe scripts/relabel_hf_story.py

Idempotent: re-running appends nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.curator.relabel_hf import relabel_hf_to_story
from gin.curator.store import Store

DEFAULT_LOG = ROOT / "data" / "curator" / "labels.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel hf_* pairs issue_frame -> story")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    print(f"appended {relabel_hf_to_story(Store(args.log))} record(s) to {args.log}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it against the real label log**

Confirm `git status` is clean first, then run:
`venv/Scripts/python.exe scripts/relabel_hf_story.py`
Expected: `appended 2 record(s) to ...data/curator/labels.jsonl`

Verify the downstream effect:
```bash
venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'.'); from pathlib import Path; from gin.curator.store import Store; from gin.frames.dataset import build_dataset; r=build_dataset(Store(Path('data/curator/labels.jsonl'))); print(len(r.examples), r.counts)"
```
Expected: `100 {'DIVERGENT': 22, 'AGREE': 20, 'RELATED_UNTYPED': 38, 'UNRELATED': 20}` — the training set drops 102→100 and DIVERGENT 24→22, exactly as the spec predicts. If the numbers differ, STOP and report.

- [ ] **Step 7: Update the frames regression guard**

`tests/test_frames_dataset.py::test_real_label_log_yields_expected_counts` asserts the old 102/24. Update it to the new values and add the reason:

```python
    # 100 not 102: hf_af_*/hf_kc_* were relabeled issue_frame -> story
    # (scripts/relabel_hf_story.py), which removes them from DIVERGENT.
    assert len(report.examples) == 100
    assert report.counts == {
        "DIVERGENT": 22, "AGREE": 20, "RELATED_UNTYPED": 38, "UNRELATED": 20,
    }
```

- [ ] **Step 8: Run the full suite and commit**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass.

```bash
git add gin/curator/relabel_hf.py scripts/relabel_hf_story.py tests/test_curator_relabel_hf.py tests/test_frames_dataset.py data/curator/labels.jsonl
git commit -m "Curator: relabel hf_* housing pairs issue_frame -> story"
```

---

### Task 2: The node5 builder

**Files:**
- Create: `gin/curator/node5_build.py`
- Create: `scripts/build_node5.py`
- Test: `tests/test_curator_node5_build.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `NODE_ID = "node_5_samestory"`, `VALID_KINDS: frozenset[str]`, `compute_global_id(source: str, outlet: str, published: str) -> str`, `build_node5(manifest: list[dict]) -> dict`, `pair_inventory(manifest: list[dict]) -> dict[str, int]`

`build_node5` raises `ValueError` on malformed manifests and on failing the composition floor.

- [ ] **Step 1: Write the failing test**

Create `tests/test_curator_node5_build.py`:

```python
"""node5 builder: manifest -> corpus dict, validation, composition floor."""
import pytest

from gin.curator.node5_build import (
    NODE_ID,
    VALID_KINDS,
    build_node5,
    compute_global_id,
    pair_inventory,
)


def _event(name, outlets, intent, lede="SHARED LEDE."):
    return {
        "event": name,
        "domain": "incident",
        "shared_lede": lede,
        "reports": [
            {"outlet": o, "published": f"2026-03-0{i + 1}T12:00Z",
             "chunks": [f"{lede} Report from {o}."]}
            for i, o in enumerate(outlets)
        ],
        "intent": intent,
    }


def _minimal_ok():
    """Two events carrying one conflict and one negative — floors relaxed in tests."""
    return [
        _event("e1", ["A", "B"], [{"pair": ["A", "B"], "kind": "conflict",
                                   "varied_fact": "count"}]),
        _event("e2", ["A", "B"], [{"pair": ["A", "B"], "kind": "corroboration",
                                   "varied_fact": None}]),
    ]


def test_node_id_and_schema_match_node1_4():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    assert corpus["node_id"] == NODE_ID
    assert set(corpus) == {"node_id", "documents"}
    doc = corpus["documents"][0]
    assert set(doc) == {"doc_id", "global_id", "source", "url", "node", "metadata", "chunks"}
    assert set(doc["chunks"][0]) == {"chunk_id", "position", "text"}


def test_positions_are_strings():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    assert corpus["documents"][0]["chunks"][0]["position"] == "0"


def test_doc_and_chunk_ids_follow_the_convention():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    assert corpus["documents"][0]["doc_id"] == "n5_doc_001"
    assert corpus["documents"][0]["chunks"][0]["chunk_id"] == "n5_doc_001_c000"


def test_global_id_shape():
    gid = compute_global_id("Some Source", "CentralWire", "2026-03-04T21:10Z")
    assert gid.startswith("gid_")
    assert len(gid) == 4 + 16


def test_global_id_is_deterministic_and_outlet_sensitive():
    a = compute_global_id("S", "CentralWire", "T")
    assert a == compute_global_id("S", "CentralWire", "T")
    assert a != compute_global_id("S", "MetroDaily", "T")


def test_build_is_deterministic():
    assert build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1) == build_node5(
        _minimal_ok(), min_conflicts=1, min_negatives=1
    )


def test_metadata_carries_event_context_not_labels():
    corpus = build_node5(_minimal_ok(), min_conflicts=1, min_negatives=1)
    meta = corpus["documents"][0]["metadata"]
    assert set(meta) == {"outlet", "published", "event", "domain"}
    # The intent matrix must never leak a relation label into the corpus.
    blob = str(corpus)
    for banned in ("conflict", "corroboration", "contradicts", "varied_fact"):
        assert banned not in blob


def test_pair_inventory_counts_by_kind():
    inv = pair_inventory(_minimal_ok())
    assert inv == {"conflict": 1, "corroboration": 1}


def test_composition_floor_is_enforced():
    with pytest.raises(ValueError, match="conflict pairs"):
        build_node5(_minimal_ok(), min_conflicts=20, min_negatives=1)
    with pytest.raises(ValueError, match="negative pairs"):
        build_node5(_minimal_ok(), min_conflicts=1, min_negatives=20)


def test_unknown_kind_is_rejected():
    bad = [_event("e1", ["A", "B"], [{"pair": ["A", "B"], "kind": "vibes",
                                      "varied_fact": None}])]
    with pytest.raises(ValueError, match="unknown kind"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_intent_referencing_an_unknown_outlet_is_rejected():
    bad = [_event("e1", ["A", "B"], [{"pair": ["A", "Z"], "kind": "conflict",
                                      "varied_fact": "count"}])]
    with pytest.raises(ValueError, match="unknown outlet"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_missing_shared_lede_is_rejected():
    bad = [{"event": "e", "domain": "incident", "reports": [], "intent": []}]
    with pytest.raises(ValueError, match="shared_lede"):
        build_node5(bad, min_conflicts=0, min_negatives=0)


def test_valid_kinds_are_the_four_from_the_spec():
    assert VALID_KINDS == frozenset(
        {"conflict", "corroboration", "update", "compatible_partial"}
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_node5_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.node5_build'`

- [ ] **Step 3: Write the implementation**

Create `gin/curator/node5_build.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_node5_build.py -v`
Expected: PASS — 12 passed

- [ ] **Step 5: Write the CLI**

Create `scripts/build_node5.py`:

```python
"""Build corpus_node5.json from the event manifest.

    venv/Scripts/python.exe scripts/build_node5.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from gin.curator.node5_build import build_node5, pair_inventory


def main() -> None:
    ap = argparse.ArgumentParser(description="Build corpus_node5.json from manifest")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "curator" / "node5_events.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "corpus_node5.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    corpus = build_node5(manifest)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pair inventory: {pair_inventory(manifest)}")
    print(f"wrote {len(corpus['documents'])} docs to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add gin/curator/node5_build.py scripts/build_node5.py tests/test_curator_node5_build.py
git commit -m "Curator: node5 event-manifest builder + composition floor"
```

---

### Task 3: Author the event manifest — structure and intent, chunks pending

This is the review gate. The event list and intent matrix are settled here, before any chunk text is written, mirroring node4's approved-source-list gate.

**Files:**
- Create: `data/curator/node5_events.yaml`

**Interfaces:**
- Consumes: the manifest schema `gin/curator/node5_build.py` validates
- Produces: the manifest that Task 4 fills with text

- [ ] **Step 1: Write the manifest skeleton**

Create `data/curator/node5_events.yaml` with all 12 events, every report's `outlet` and `published`, the full `intent` matrix, and **`chunks: []` throughout**. Use exactly these events and intents:

| # | event | domain | outlets | intent pairs |
|---|---|---|---|---|
| 1 | `riverport_warehouse_fire` | incident | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `evacuee_count`; CW–RP corroboration; MD–RP conflict `evacuee_count` |
| 2 | `harbor_district_referendum` | election | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `turnout_pct`; CW–RP update `margin`; MD–RP conflict `turnout_pct` |
| 3 | `northgate_hospital_outbreak` | public_health | CentralWire, MetroDaily, RegionalPost, CivicLedger | CW–MD conflict `case_count`; CW–RP compatible_partial `ward_vs_hospital`; MD–CL corroboration; RP–CL compatible_partial `ward_vs_hospital`; CW–CL conflict `case_count`; MD–RP corroboration |
| 4 | `sable_bridge_closure` | infrastructure | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `reopening_date`; CW–RP corroboration; MD–RP conflict `reopening_date` |
| 5 | `meridian_civil_verdict` | courts | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `damages_awarded`; CW–RP compatible_partial `counts_vs_charges`; MD–RP conflict `damages_awarded` |
| 6 | `coastal_storm_landfall` | weather | CentralWire, MetroDaily, RegionalPost, CivicLedger | CW–MD update `wind_speed`; CW–RP conflict `outage_count`; MD–CL corroboration; RP–CL conflict `outage_count`; CW–CL update `wind_speed`; MD–RP corroboration |
| 7 | `dockworkers_walkout` | labor | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `participant_count`; CW–RP compatible_partial `local_vs_national`; MD–RP conflict `participant_count` |
| 8 | `crosstown_line_suspension` | transit | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `delay_minutes`; CW–RP corroboration; MD–RP corroboration |
| 9 | `district_enrollment_report` | education | CentralWire, MetroDaily, RegionalPost | CW–MD compatible_partial `district_vs_school`; CW–RP conflict `enrollment_figure`; MD–RP conflict `enrollment_figure` |
| 10 | `lakeshore_algae_bloom` | environment | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `affected_area_km2`; CW–RP update `affected_area_km2`; MD–RP corroboration |
| 11 | `civic_bond_audit` | finance | CentralWire, MetroDaily, RegionalPost | CW–MD conflict `shortfall_amount`; CW–RP corroboration; MD–RP conflict `shortfall_amount` |
| 12 | `stadium_capacity_ruling` | municipal | CentralWire, MetroDaily, RegionalPost | CW–MD compatible_partial `seated_vs_total`; CW–RP conflict `capacity_figure`; MD–RP corroboration |

Outlet shorthand: CW = CentralWire, MD = MetroDaily, RP = RegionalPost, CL = CivicLedger.

Totals: **21 conflict**, **11 corroboration**, **4 update**, **6 compatible_partial** — 21 conflicts and 21 negatives across **42 pairs** over **38 reports** (ten 3-outlet events give C(3,2)=3 pairs each, two 4-outlet events give C(4,2)=6 each). Clears the spec's ≥20/≥20 floor with margin. Verify by counting the table rather than trusting this line: an earlier draft of it asserted 20/10/4/6 over 40 pairs/40 reports, which the table never supported.

Each event also needs a one-sentence `shared_lede` that every report in that event will open with verbatim. Write those now; they are structure, not divergent content. Example for event 1:

```yaml
- event: riverport_warehouse_fire
  domain: incident
  shared_lede: "RIVERPORT — Fire crews responded to a warehouse blaze on the east waterfront Tuesday evening."
  reports:
    - outlet: CentralWire
      published: "2026-03-04T21:10Z"
      chunks: []
    - outlet: MetroDaily
      published: "2026-03-04T21:40Z"
      chunks: []
    - outlet: RegionalPost
      published: "2026-03-04T22:05Z"
      chunks: []
  intent:
    - pair: [CentralWire, MetroDaily]
      kind: conflict
      varied_fact: evacuee_count
    - pair: [CentralWire, RegionalPost]
      kind: corroboration
      varied_fact: null
    - pair: [MetroDaily, RegionalPost]
      kind: conflict
      varied_fact: evacuee_count
```

- [ ] **Step 2: Verify the intent matrix totals**

Run:
```bash
venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'.'); import yaml; from pathlib import Path; from gin.curator.node5_build import pair_inventory; m=yaml.safe_load(Path('data/curator/node5_events.yaml').read_text(encoding='utf-8')); inv=pair_inventory(m); print(inv); print('conflicts', inv.get('conflict',0), 'negatives', sum(v for k,v in inv.items() if k!='conflict')); print('events', len(m), 'reports', sum(len(e['reports']) for e in m))"
```
Expected: `{'conflict': 21, 'corroboration': 11, 'update': 4, 'compatible_partial': 6}`, `conflicts 21 negatives 21`, `events 12 reports 38`.

If the totals differ, the matrix was transcribed wrong — fix the manifest, not the expectation.

- [ ] **Step 3: Commit the skeleton for review**

The build will not succeed yet (reports have no chunks); that is expected at this gate.

```bash
git add data/curator/node5_events.yaml
git commit -m "Curator: node5 approved event list + intent matrix (chunks pending)"
```

---

### Task 4: Author the chunk text and build the corpus

**Files:**
- Modify: `data/curator/node5_events.yaml` (fill every `chunks: []`)
- Create: `corpus_node5.json`
- Test: `tests/test_curator_node5_corpus.py`

**Interfaces:**
- Consumes: `gin.curator.node5_build.{build_node5, pair_inventory}`
- Produces: `corpus_node5.json` loadable by `gin.curator.corpus_json.load_corpus_chunks`

- [ ] **Step 1: Author the chunks**

Each report gets **exactly one chunk**, 3–4 sentences, and it MUST:

1. **Open with the event's `shared_lede` verbatim.** This is what makes
   `make_same_story` fire — it needs ≥2 shared corpus-rare tokens including an
   entity-grade anchor, and the shared lede supplies the place name and event
   noun that serve as that anchor.
2. **Carry the varied fact as a specific number, date, or name** so a conflict is
   a real propositional disagreement, not a tone difference.
3. **Read like wire copy**, not like an argument. No stance, no evaluation.

The four kinds, and what makes each one right:

- **conflict** — the two reports state *incompatible* values for the same fact.
  "Officials confirmed 34 people were evacuated" against "Officials confirmed 19
  people were evacuated."
- **corroboration** — the same fact, worded differently. "34 people were
  evacuated" against "Evacuations totaled 34 residents."
- **update** — a *later* report revises an earlier figure, and says so. "Officials
  initially reported 34 evacuees; the count was revised to 41 Wednesday." The
  `published` timestamps must order correctly, or this is indistinguishable from
  a conflict.
- **compatible_partial** — the two reports state different numbers that are both
  true because they measure different scopes, and **each report names its scope
  explicitly**. "23 arrests were made downtown" against "31 arrests were made
  citywide." **This is the hardest and most valuable kind**: a naive
  numeric-conflict detector calls it a contradiction and it is not. If the scope
  is not stated in the text, the pair is indistinguishable from a conflict and is
  worthless — spend the extra clause.

Two authoring traps:

- A `corroboration` pair whose two reports are near-identical strings teaches
  nothing. Vary sentence structure and wording while keeping the fact identical.
- A `conflict` pair must not *also* differ in scope, or it becomes a
  `compatible_partial` by accident. One varied fact per conflict pair, same scope.

Consistency requirement: every report participates in the pairs its event's
intent matrix declares, so a single report's numbers must satisfy *all* of its
pairings simultaneously. Write each event's reports together, not one at a time.

- [ ] **Step 2: Build the corpus**

Run: `venv/Scripts/python.exe scripts/build_node5.py`
Expected: `pair inventory: {'conflict': 21, 'corroboration': 11, 'update': 4, 'compatible_partial': 6}` and `wrote 38 docs to ...corpus_node5.json`

- [ ] **Step 3: Write the corpus regression guard**

Create `tests/test_curator_node5_corpus.py`:

```python
"""Regression guard on the built node5 corpus."""
from pathlib import Path

import yaml

from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node5_build import NODE_ID, pair_inventory

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "corpus_node5.json"
MANIFEST = REPO_ROOT / "data" / "curator" / "node5_events.yaml"


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_corpus_loads_through_the_standard_loader():
    chunks = load_corpus_chunks([CORPUS])
    assert len(chunks) == 38
    assert all(c.chunk_id.startswith("n5_doc_") for c in chunks)


def test_intent_matrix_totals():
    assert pair_inventory(_manifest()) == {
        "conflict": 21, "corroboration": 11, "update": 4, "compatible_partial": 6,
    }


def test_twelve_events_thirty_eight_reports():
    # Ten 3-outlet events and two 4-outlet events.
    m = _manifest()
    assert len(m) == 12
    assert sum(len(e["reports"]) for e in m) == 38
    assert sum(len(e["intent"]) for e in m) == 42


def test_every_report_opens_with_its_events_shared_lede():
    # The shared lede is what makes make_same_story fire; without it the pair is
    # not same-story and the corpus does not test what it was built to test.
    for ev in _manifest():
        for rep in ev["reports"]:
            assert rep["chunks"], f"{ev['event']}/{rep['outlet']} has no chunk"
            assert rep["chunks"][0].startswith(ev["shared_lede"]), (
                f"{ev['event']}/{rep['outlet']} does not open with the shared lede"
            )


def test_update_pairs_are_ordered_in_time():
    # An update that is not later than what it revises is just a conflict.
    for ev in _manifest():
        published = {r["outlet"]: r["published"] for r in ev["reports"]}
        for entry in ev["intent"]:
            if entry["kind"] == "update":
                first, second = entry["pair"]
                assert published[first] != published[second], (
                    f"{ev['event']}: update pair {entry['pair']} shares a timestamp"
                )


def test_corpus_carries_no_relation_labels():
    text = CORPUS.read_text(encoding="utf-8")
    for banned in ("conflict", "corroboration", "compatible_partial", "varied_fact"):
        assert banned not in text
```

- [ ] **Step 4: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_node5_corpus.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Confirm the same-story predicate actually fires**

This is the load-bearing check for the whole corpus. Run:
```bash
venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
import yaml
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.text_index import default_text_index
from gin.cartographer.relatedness import make_same_story
chunks = load_corpus_chunks([Path('corpus_node5.json')])
by_id = {c.chunk_id: c.text for c in chunks}
corpus = list(default_text_index().values()) + [c.text for c in chunks]
ss = make_same_story(corpus)
m = yaml.safe_load(Path('data/curator/node5_events.yaml').read_text(encoding='utf-8'))
docs = {}
i = 0
for ev in m:
    for rep in ev['reports']:
        i += 1
        docs[(ev['event'], rep['outlet'])] = f'n5_doc_{i:03d}:0'  # loader-normalised
ok = bad = 0
for ev in m:
    for entry in ev['intent']:
        a = by_id[docs[(ev['event'], entry['pair'][0])]]
        b = by_id[docs[(ev['event'], entry['pair'][1])]]
        if ss(a, b): ok += 1
        else:
            bad += 1
            print('NOT same-story:', ev['event'], entry['pair'], entry['kind'])
print(f'{ok}/{ok+bad} authored pairs pass make_same_story')
"
```
Expected: `42/42 authored pairs pass make_same_story`.

If any pair fails, the shared lede is too thin for that event — **strengthen the lede** (add a place name or a distinctive event noun). Do NOT loosen `make_same_story`; its entity-anchor requirement is what separates same-story from same-topic, and weakening it reinstates the cross-topic false positives that scan run `20260712T074956Z` documented.

- [ ] **Step 6: Commit**

```bash
git add data/curator/node5_events.yaml corpus_node5.json tests/test_curator_node5_corpus.py
git commit -m "Curator: node5 same-story corpus built (12 events, 38 reports, 42 pairs)"
```

---

### Task 5: `SameStoryCandidateSource`

**Files:**
- Create: `gin/curator/same_story.py`
- Test: `tests/test_curator_same_story.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.LabeledChunk`, `gin.cartographer.combined.CombinedRelationProposer`
- Produces: `SameStoryCandidateSource(chunks, *, same_story=None, proposer=None, max_candidates=2000)` with `pre_ranked = True`, `.chunks() -> list[LabeledChunk]`, `.pairs() -> list[tuple[LabeledChunk, LabeledChunk]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_curator_same_story.py`:

```python
"""SameStoryCandidateSource: selects same-story pairs, ranks conflicts first."""
from gin.cartographer.models import LabeledChunk
from gin.curator.same_story import SameStoryCandidateSource


def _chunks(*ids):
    return [LabeledChunk(chunk_id=i, text=f"text {i}") for i in ids]


def _same_story_for(pairs):
    """True only for the given unordered text pairs."""
    keys = {frozenset(p) for p in pairs}

    def predicate(a_text, b_text):
        return frozenset((a_text, b_text)) in keys

    return predicate


def test_only_same_story_pairs_are_offered():
    chunks = _chunks("a:0", "b:0", "c:0")
    src = SameStoryCandidateSource(
        chunks, same_story=_same_story_for([("text a:0", "text b:0")])
    )
    pairs = src.pairs()
    assert len(pairs) == 1
    assert {pairs[0][0].chunk_id, pairs[0][1].chunk_id} == {"a:0", "b:0"}


def test_negatives_are_included_not_filtered():
    # The whole point of the corpus is same-story pairs that are NOT conflicts.
    # A source that kept only high-p_contra pairs would drop them.
    chunks = _chunks("a:0", "b:0", "c:0", "d:0")
    src = SameStoryCandidateSource(
        chunks,
        same_story=_same_story_for([("text a:0", "text b:0"), ("text c:0", "text d:0")]),
        p_contra=lambda x, y: 0.9 if "a:0" in x else 0.01,
    )
    assert len(src.pairs()) == 2


def test_conflicts_rank_before_negatives():
    chunks = _chunks("lo1:0", "lo2:0", "hi1:0", "hi2:0")
    src = SameStoryCandidateSource(
        chunks,
        same_story=_same_story_for(
            [("text lo1:0", "text lo2:0"), ("text hi1:0", "text hi2:0")]
        ),
        p_contra=lambda x, y: 0.95 if "hi" in x else 0.02,
    )
    first = src.pairs()[0]
    assert {first[0].chunk_id, first[1].chunk_id} == {"hi1:0", "hi2:0"}


def test_is_pre_ranked_so_the_app_does_not_resort():
    assert SameStoryCandidateSource.pre_ranked is True


def test_chunks_round_trip():
    chunks = _chunks("a:0", "b:0")
    src = SameStoryCandidateSource(chunks, same_story=lambda x, y: True)
    assert [c.chunk_id for c in src.chunks()] == ["a:0", "b:0"]


def test_max_candidates_caps_the_backlog():
    chunks = _chunks(*[f"c{i}:0" for i in range(10)])
    src = SameStoryCandidateSource(
        chunks, same_story=lambda x, y: True, p_contra=lambda x, y: 0.5,
        max_candidates=4,
    )
    assert len(src.pairs()) == 4


def test_empty_when_nothing_is_same_story():
    src = SameStoryCandidateSource(_chunks("a:0", "b:0"), same_story=lambda x, y: False)
    assert src.pairs() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_same_story.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.same_story'`

- [ ] **Step 3: Write the implementation**

Create `gin/curator/same_story.py`:

```python
"""Candidate source over same-story pairs — node5's counterpart to the residue source.

EscalationResidueCandidateSource cannot serve this corpus for two independent
reasons. It filters TO the anchor-less residue (escalation_candidates returns
"pairs the cheap path cannot type: not same-story"), and it ranks mid-band cosine
first, so same-story pairs — high cosine from a shared lede — would rank last
even if they survived the filter.

Ranking is NLI-contradiction-descending so genuine conflicts reach the curator
before the negatives. Note it RANKS but never FILTERS on p_contra: the negatives
are the reason this corpus exists, and dropping low-p_contra pairs would discard
exactly the rows that can falsify combined.py's unconditional
"same_story => CONTRADICTS" branch.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional

from gin.cartographer.models import LabeledChunk

DEFAULT_MAX_CANDIDATES = 2000


class SameStoryCandidateSource:
    """A.CandidateSource over pairs the stage-1 story predicate accepts."""

    # pairs() returns an evidence-based ranking; app.next_pairs must not re-sort.
    pre_ranked = True

    def __init__(
        self,
        chunks: list[LabeledChunk],
        *,
        same_story: Optional[Callable[[str, str], bool]] = None,
        p_contra: Optional[Callable[[str, str], float]] = None,
        proposer=None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._chunks = list(chunks)
        self._max_candidates = max_candidates
        # Only construct a proposer when the caller supplied no predicate. An
        # injected same_story must never drag a model load into a test.
        if same_story is None and proposer is None:
            from gin.cartographer.combined import CombinedRelationProposer

            proposer = CombinedRelationProposer()
        if same_story is None:
            if proposer.same_story is None:
                raise ValueError(
                    "SameStoryCandidateSource needs a same-story provider: pass "
                    "same_story=, or a proposer with scan.wire_same_story applied"
                )
            same_story = proposer.same_story
        if p_contra is None:
            # No scorer and no proposer means no ranking evidence: keep every
            # pair (the negatives are the point) in a stable, unranked order.
            p_contra = (
                (lambda a, b: proposer._p_contra(a, b))  # noqa: SLF001
                if proposer is not None
                else (lambda a, b: 0.0)
            )
        self._same_story = same_story
        self._p_contra = p_contra

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        scored: list[tuple[float, tuple[LabeledChunk, LabeledChunk]]] = []
        for a, b in combinations(self._chunks, 2):
            if not self._same_story(a.text, b.text):
                continue
            scored.append((self._p_contra(a.text, b.text), (a, b)))
        scored.sort(key=lambda row: -row[0])
        return [pair for _score, pair in scored[: self._max_candidates]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_same_story.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/same_story.py tests/test_curator_same_story.py
git commit -m "Curator: same-story candidate source (ranks conflicts first, keeps negatives)"
```

---

### Task 6: Readiness `story` target and launcher wiring

**Files:**
- Modify: `gin/curator/readiness.py`
- Modify: `scripts/curator_serve.py`
- Test: `tests/test_curator_readiness.py`

**Interfaces:**
- Consumes: `gin.curator.same_story.SameStoryCandidateSource`
- Produces: `ReadinessTarget(issue_frame=20, agree=20, unrelated=20, story=20)`, `ReadinessReport(..., new_story: int)`, `--source same-story` on the launcher

- [ ] **Step 1: Write the failing test**

Add to `tests/test_curator_readiness.py`:

```python
def test_story_class_is_counted(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("s1:0", "s2:0", Relation.CONTRADICTS, "2026-07-25T00:00:00Z",
                      relation_class="story"))
    rep = readiness(store)
    assert rep.new_story == 1
    assert rep.new_issue_frame == 0


def test_ready_requires_story_too(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    target = ReadinessTarget(issue_frame=1, agree=1, unrelated=1, story=1)
    store.append(_rec("a:0", "b:0", Relation.CONTRADICTS, "2026-07-25T00:00:01Z",
                      relation_class="issue_frame"))
    store.append(_rec("c:0", "d:0", Relation.CORROBORATES, "2026-07-25T00:00:02Z"))
    store.append(_rec("e:0", "f:0", Relation.UNRELATED, "2026-07-25T00:00:03Z"))
    assert readiness(store, target).ready is False   # story still 0
    store.append(_rec("g:0", "h:0", Relation.CONTRADICTS, "2026-07-25T00:00:04Z",
                      relation_class="story"))
    assert readiness(store, target).ready is True
```

Add `ReadinessTarget` to that file's imports if it is not already there.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_readiness.py -v`
Expected: FAIL — `AttributeError: 'ReadinessReport' object has no attribute 'new_story'`

- [ ] **Step 3: Add the story target**

In `gin/curator/readiness.py`, extend the two dataclasses and the count loop:

```python
@dataclass(frozen=True)
class ReadinessTarget:
    issue_frame: int = 20
    agree: int = 20
    unrelated: int = 20
    story: int = 20


@dataclass(frozen=True)
class ReadinessReport:
    new_issue_frame: int
    new_agree: int
    new_unrelated: int
    new_story: int
    target: ReadinessTarget
    ready: bool
```

In `readiness()`, initialise `n_st = 0`, add the branch, and extend the verdict and return:

```python
        if relation is Relation.CONTRADICTS and relation_class == "issue_frame":
            n_if += 1
        elif relation is Relation.CONTRADICTS and relation_class == "story":
            n_st += 1
        elif relation is Relation.CORROBORATES:
            n_ag += 1
        elif relation is Relation.UNRELATED:
            n_un += 1
    ready = (
        n_if >= target.issue_frame
        and n_ag >= target.agree
        and n_un >= target.unrelated
        and n_st >= target.story
    )
    return ReadinessReport(n_if, n_ag, n_un, n_st, target, ready)
```

- [ ] **Step 4: Update the readiness endpoint**

`gin/curator/app.py` builds the readiness JSON response. Add the story fields alongside the existing ones so the UI progress line reports them:

```python
            "new_story": rep.new_story,
```
in the counts block, and
```python
                "story": rep.target.story,
```
in the target block.

- [ ] **Step 5: Wire the launcher**

In `scripts/curator_serve.py`, add `"same-story"` to the `--source` choices and construct the source. Alongside the existing `escalation-residue` branch:

```python
    elif args.source == "same-story":
        try:
            chunks = load_corpus_chunks(args.corpus)
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"error: {exc}")
        from gin.cartographer.scan import wire_same_story
        from gin.curator.same_story import SameStoryCandidateSource

        wire_same_story(proposer, chunks)
        source = SameStoryCandidateSource(chunks, proposer=proposer)
        print(f"same-story source over {len(source.chunks())} corpus chunks")
```

`wire_same_story(proposer, chunks)` takes the `LabeledChunk` list itself, not texts — verified at `gin/cartographer/scan.py:232`. It is a no-op when the proposer already has a provider, so calling it twice is safe.

- [ ] **Step 6: Run tests and the full suite**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_readiness.py tests/test_curator_app.py -v`
Expected: PASS.

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass. Any test asserting a 3-field `ReadinessReport` needs its expectation extended, not the feature reverted.

- [ ] **Step 7: Commit**

```bash
git add gin/curator/readiness.py gin/curator/app.py scripts/curator_serve.py tests/test_curator_readiness.py
git commit -m "Curator: readiness story target + --source same-story"
```

---

### Task 7: Surfacing gate

**Files:**
- Create: `gin/curator/node5_verify.py`
- Create: `scripts/verify_node5_surfacing.py`
- Test: `tests/test_curator_node5_verify.py`

**Interfaces:**
- Consumes: `gin.curator.same_story.SameStoryCandidateSource`, `gin.curator.node5_build.pair_inventory`
- Produces: `authored_pair_chunk_ids(manifest) -> list[tuple[str, str, str]]` (src_chunk_id, dst_chunk_id, kind), `verify_surfacing(manifest, offered_pairs) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_curator_node5_verify.py`:

```python
"""Surfacing gate: every authored pair must reach the curator backlog."""
from gin.curator.node5_verify import authored_pair_chunk_ids, verify_surfacing


def _manifest():
    return [
        {
            "event": "e1", "domain": "incident", "shared_lede": "L.",
            "reports": [
                {"outlet": "A", "published": "t1", "chunks": ["L. a"]},
                {"outlet": "B", "published": "t2", "chunks": ["L. b"]},
            ],
            "intent": [
                {"pair": ["A", "B"], "kind": "conflict", "varied_fact": "n"},
            ],
        },
        {
            "event": "e2", "domain": "incident", "shared_lede": "M.",
            "reports": [
                {"outlet": "A", "published": "t1", "chunks": ["M. a"]},
                {"outlet": "B", "published": "t2", "chunks": ["M. b"]},
            ],
            "intent": [
                {"pair": ["A", "B"], "kind": "corroboration", "varied_fact": None},
            ],
        },
    ]


def test_authored_pairs_map_to_chunk_ids_in_build_order():
    pairs = authored_pair_chunk_ids(_manifest())
    # Loader-normalised ids ("n5_doc_001:0"), not the raw JSON "n5_doc_001_c000".
    assert pairs == [
        ("n5_doc_001:0", "n5_doc_002:0", "conflict"),
        ("n5_doc_003:0", "n5_doc_004:0", "corroboration"),
    ]


def test_all_surfaced_passes():
    offered = {
        frozenset(("n5_doc_001:0", "n5_doc_002:0")),
        frozenset(("n5_doc_003:0", "n5_doc_004:0")),
    }
    report = verify_surfacing(_manifest(), offered)
    report_ok = report["passed"]
    assert report_ok is True
    assert report["missing"] == []


def test_a_missing_negative_fails_as_loudly_as_a_missing_conflict():
    # The negatives are why this corpus exists. If only conflicts surface, the
    # curator never labels a same-story non-contradiction.
    offered = {frozenset(("n5_doc_001:0", "n5_doc_002:0"))}
    report = verify_surfacing(_manifest(), offered)
    assert report["passed"] is False
    assert report["missing"] == [
        ("n5_doc_003:0", "n5_doc_004:0", "corroboration")
    ]


def test_missing_by_kind_is_reported():
    report = verify_surfacing(_manifest(), set())
    assert report["missing_by_kind"] == {"conflict": 1, "corroboration": 1}
    assert report["authored"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_node5_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.node5_verify'`

- [ ] **Step 3: Write the implementation**

Create `gin/curator/node5_verify.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_node5_verify.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Write the gate CLI**

Create `scripts/verify_node5_surfacing.py`:

```python
"""Hard gate: every authored node5 pair must reach the curator backlog.

    venv/Scripts/python.exe scripts/verify_node5_surfacing.py

Loads the real embedding + NLI proposer. Exit 0 iff every authored pair surfaces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.relatedness import make_same_story
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node5_verify import verify_surfacing
from gin.curator.same_story import SameStoryCandidateSource


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify node5 pairs surface")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "curator" / "node5_events.yaml")
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus_node5.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    chunks = load_corpus_chunks([args.corpus])
    proposer = CombinedRelationProposer()
    same_story = make_same_story([c.text for c in chunks])
    source = SameStoryCandidateSource(chunks, same_story=same_story, proposer=proposer)

    offered = {frozenset((a.chunk_id, b.chunk_id)) for a, b in source.pairs()}
    report = verify_surfacing(manifest, offered)

    print(f"authored {report['authored']} | surfaced {report['surfaced']}")
    if report["missing"]:
        print(f"missing by kind: {report['missing_by_kind']}")
        for src, dst, kind in report["missing"]:
            print(f"  MISSING [{kind}] {src} <-> {dst}")
        print("\nA missing negative is as serious as a missing conflict: without")
        print("them the curator never labels a same-story non-contradiction.")
        return 1
    print("PASS: every authored pair reaches the curator backlog")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the gate for real**

Run: `venv/Scripts/python.exe scripts/verify_node5_surfacing.py`
Expected: `authored 42 | surfaced 42` and `PASS: every authored pair reaches the curator backlog`.

If pairs are missing, the fix is in the corpus (strengthen that event's shared lede) or the cap, never in loosening `make_same_story`.

- [ ] **Step 7: Smoke the curator end to end**

Run: `venv/Scripts/python.exe scripts/curator_serve.py --source same-story --corpus corpus_node5.json --curator kristian`
Then in another shell: `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8600/curator/` (expect `200`) and `curl -s http://127.0.0.1:8600/curator/readiness` (expect a JSON body including `new_story` and a `story` target). Stop the server afterwards.

- [ ] **Step 8: Run the full suite and commit**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass, and `tests/test_cartographer_eval_pairs.py` still passes — the escalation bar has not moved.

```bash
git add gin/curator/node5_verify.py scripts/verify_node5_surfacing.py tests/test_curator_node5_verify.py
git commit -m "Curator: node5 surfacing gate (negatives held to the same bar)"
```

---

## Post-Implementation

Record in `architecture.md` and in the spec's own results section: the corpus's event and pair counts, the surfacing gate result, and the readiness gauge's `story` reading. State plainly that no detector metric was measured — this ships a corpus and the means to label it, and the labeling has not happened yet.

Then the next actionable step is a curator session:
`venv/Scripts/python.exe scripts/curator_serve.py --source same-story --corpus corpus_node5.json --curator kristian`
