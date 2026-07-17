# Curator UI + Label Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI labeling tool that writes curator judgments (relation + relation_class + rationale) over chunk pairs to a durable, git-trackable, append-only JSONL event log — the shared labeled-framing-corpus substrate that later unblocks the bi-encoder frame detector (sub-project B) and the larger-set Cartographer recalibration (sub-project C).

**Architecture:** A new `gin/curator/` package, sibling to `gin/cartographer/`, with one clear job per module: `models` (the record type), `store` (append-only JSONL + latest-wins fold), `signals` (a thin read-only wrapper over the existing `CombinedRelationProposer`), `candidates` (a pluggable source + hard-cases-first ordering), `seed` (import the existing ~33 labels), and `app` (a local-only FastAPI app serving one no-build-step HTML page + JSON endpoints). The package **consumes** `gin/cartographer` (for `Relation`, `CombinedRelationProposer`, `sentence_anchor`, and the existing gold loaders) but is **imported by nothing** — it is a new top layer. It is strictly additive: no existing file is modified.

**Tech Stack:** Python 3.10+, FastAPI + Starlette (already in the federation stack), stdlib `json`/`uuid`/`datetime`/`itertools`, and `gin.cartographer`. No new dependencies.

## Global Constraints

- **No new dependencies.** FastAPI, Starlette (`TestClient`, `HTMLResponse`), `httpx`, and `pydantic` are already in `requirements.txt`.
- **Strictly additive.** Do not modify `gin/cartographer/labeled_set.py`, `gin/cartographer/gold_edges.py`, `gin/cartographer/calibration.py`, or `gin/cartographer/combined.py`. The loader-unification refactor is explicitly deferred to sub-project C. The full existing test suite must stay green with zero modifications to any existing test file.
- **Layering:** `gin/curator/` may import from `gin/cartographer/`. Nothing in `gin/cartographer/`, `gin/eval/`, `gin/bookkeeper/`, or `gin/federation/` may import from `gin/curator/`.
- **`relation_class` is required if and only if `relation == "contradicts"`.** It is `"story"` or `"issue_frame"`; `None` otherwise. This is the bi-encoder's target label and is enforced server-side, not just in the page.
- **The label store is a git-tracked append-only JSONL file** at `data/curator/labels.jsonl`. Pairs are keyed order-independently (A↔B and B↔A are the same pair).
- **Relation vocabulary** is exactly `gin.cartographer.models.Relation`: `contradicts`, `corroborates`, `supersedes`, `related_untyped`, `unrelated` (string values identical to the member names lowercased).
- **Test runner:** `venv/Scripts/python.exe -m pytest` (Windows venv; `PYTHONIOENCODING=utf-8` if console emoji/box output appears).

---

## Task 1: Label record type

**Files:**
- Create: `gin/curator/__init__.py`
- Create: `gin/curator/models.py`
- Test: `tests/test_curator_models.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.Relation` (existing str-enum).
- Produces:
  - `pair_key(src: str, dst: str) -> tuple[str, str]` — order-independent 2-tuple (`tuple(sorted((src, dst)))`).
  - `LabelRecord` frozen dataclass: `id: str`, `src_chunk_id: str`, `dst_chunk_id: str`, `relation: Relation`, `relation_class: Optional[str]`, `rationale: str`, `curator: str`, `ts: str`, `supersedes: Optional[str] = None`, `src_anchor: Optional[tuple[int, int]] = None`, `dst_anchor: Optional[tuple[int, int]] = None`.
  - `LabelRecord.to_json(self) -> dict` and `LabelRecord.from_json(cls, d: dict) -> LabelRecord` (round-trip; `Relation` (de)serialized by `.value`, anchors as 2-element lists).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_models.py
"""LabelRecord (de)serialization and order-independent pair keys."""
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key


def _rec(**kw):
    base = dict(
        id="r1", src_chunk_id="b:0", dst_chunk_id="a:0",
        relation=Relation.CONTRADICTS, relation_class="issue_frame",
        rationale="opposing frames", curator="kristian", ts="2026-07-17T00:00:00Z",
    )
    base.update(kw)
    return LabelRecord(**base)


def test_pair_key_is_order_independent():
    assert pair_key("a:0", "b:0") == pair_key("b:0", "a:0") == ("a:0", "b:0")


def test_to_json_round_trips():
    rec = _rec(src_anchor=(0, 3), dst_anchor=(0, 5))
    d = rec.to_json()
    assert d["relation"] == "contradicts"
    assert d["relation_class"] == "issue_frame"
    assert d["src_anchor"] == [0, 3]
    back = LabelRecord.from_json(d)
    assert back == rec


def test_from_json_handles_null_class_and_anchors():
    rec = _rec(relation=Relation.CORROBORATES, relation_class=None)
    back = LabelRecord.from_json(rec.to_json())
    assert back.relation is Relation.CORROBORATES
    assert back.relation_class is None
    assert back.src_anchor is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator'`

- [ ] **Step 3: Create the package and the record type**

Create `gin/curator/__init__.py`:

```python
"""Curator tier — a human-in-the-loop labeling substrate for the framing corpus.

Produces the durable labeled pair set that later feeds the bi-encoder frame
detector (sub-project B) and the larger-set Cartographer recalibration
(sub-project C). Consumes gin.cartographer; imported by nothing.
See docs/superpowers/specs/2026-07-17-curator-ui-label-store-design.md.
"""
```

Create `gin/curator/models.py`:

```python
"""The atomic unit the curator emits: one immutable labeling act."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gin.cartographer.models import Relation


def pair_key(src: str, dst: str) -> tuple[str, str]:
    """Order-independent key so A->B and B->A fold to the same pair."""
    a, b = sorted((src, dst))
    return (a, b)


def _anchor_to_json(anchor: Optional[tuple[int, int]]) -> Optional[list[int]]:
    return None if anchor is None else [anchor[0], anchor[1]]


def _anchor_from_json(raw) -> Optional[tuple[int, int]]:
    if not raw:
        return None
    return int(raw[0]), int(raw[1])


@dataclass(frozen=True)
class LabelRecord:
    """One label / relabel / adjudication. Appended, never mutated in place."""

    id: str
    src_chunk_id: str
    dst_chunk_id: str
    relation: Relation
    relation_class: Optional[str]
    rationale: str
    curator: str
    ts: str  # UTC ISO-8601
    supersedes: Optional[str] = None
    src_anchor: Optional[tuple[int, int]] = None
    dst_anchor: Optional[tuple[int, int]] = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "src_chunk_id": self.src_chunk_id,
            "dst_chunk_id": self.dst_chunk_id,
            "relation": self.relation.value,
            "relation_class": self.relation_class,
            "rationale": self.rationale,
            "curator": self.curator,
            "ts": self.ts,
            "supersedes": self.supersedes,
            "src_anchor": _anchor_to_json(self.src_anchor),
            "dst_anchor": _anchor_to_json(self.dst_anchor),
        }

    @classmethod
    def from_json(cls, d: dict) -> "LabelRecord":
        return cls(
            id=d["id"],
            src_chunk_id=d["src_chunk_id"],
            dst_chunk_id=d["dst_chunk_id"],
            relation=Relation(d["relation"]),
            relation_class=d.get("relation_class"),
            rationale=d.get("rationale", ""),
            curator=d.get("curator", "unknown"),
            ts=d["ts"],
            supersedes=d.get("supersedes"),
            src_anchor=_anchor_from_json(d.get("src_anchor")),
            dst_anchor=_anchor_from_json(d.get("dst_anchor")),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/__init__.py gin/curator/models.py tests/test_curator_models.py
git commit -m "Curator: LabelRecord type + order-independent pair keys (curator UI, task 1)."
```

---

## Task 2: Append-only JSONL store

**Files:**
- Create: `gin/curator/store.py`
- Test: `tests/test_curator_store.py`

**Interfaces:**
- Consumes: `LabelRecord`, `pair_key` (Task 1).
- Produces `Store`:
  - `Store(path: Path)`
  - `append(self, rec: LabelRecord) -> None` — one JSON line; creates the parent dir on first write.
  - `read_log(self) -> list[LabelRecord]` — all records in file order; raises `ValueError` naming the 1-based line number on a malformed line.
  - `fold_current(self) -> dict[tuple[str, str], LabelRecord]` — latest-wins per `pair_key`, ordered by `(ts, line_index)` so a later record supersedes an earlier one.
  - `gold(self) -> list[tuple[str, str, Relation, Optional[str]]]` — the folded view as `(src, dst, relation, relation_class)` tuples (the reader B and C consume).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_store.py
"""Append-only JSONL store: round-trip, latest-wins fold, loud on corruption."""
import pytest

from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.store import Store


def _rec(id, src, dst, relation, ts, relation_class=None):
    return LabelRecord(
        id=id, src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator="t", ts=ts,
    )


def test_append_then_read_round_trips(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    r = _rec("1", "a:0", "b:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "story")
    store.append(r)
    assert store.read_log() == [r]


def test_fold_is_latest_wins_per_pair(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("1", "a:0", "b:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "story"))
    store.append(_rec("2", "a:0", "b:0", Relation.CORROBORATES, "2026-07-17T01:00:00Z"))
    fold = store.fold_current()
    assert set(fold.keys()) == {pair_key("a:0", "b:0")}
    assert fold[pair_key("a:0", "b:0")].relation is Relation.CORROBORATES


def test_fold_collapses_reversed_pair(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("1", "a:0", "b:0", Relation.UNRELATED, "2026-07-17T00:00:00Z"))
    store.append(_rec("2", "b:0", "a:0", Relation.CORROBORATES, "2026-07-17T02:00:00Z"))
    fold = store.fold_current()
    assert len(fold) == 1
    assert fold[pair_key("a:0", "b:0")].relation is Relation.CORROBORATES


def test_gold_returns_reader_shape(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("1", "a:0", "b:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "issue_frame"))
    assert store.gold() == [("a:0", "b:0", Relation.CONTRADICTS, "issue_frame")]


def test_read_log_raises_loudly_on_malformed_line(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id": "1"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        Store(path).read_log()


def test_empty_store_reads_empty(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    assert store.read_log() == []
    assert store.fold_current() == {}
    assert store.gold() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.store'`

- [ ] **Step 3: Write `gin/curator/store.py`**

```python
"""Append-only JSONL label store. Source of truth for the framing corpus.

The current gold is DERIVED by folding the log latest-wins per pair; a relabel
or adjudication is a new record superseding an earlier one, never an in-place
edit — so labeling history (including contested-then-adjudicated pairs) survives
and the file stays git-diffable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from gin.cartographer.models import Relation

from .models import LabelRecord, pair_key


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, rec: LabelRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")

    def read_log(self) -> list[LabelRecord]:
        if not self.path.is_file():
            return []
        records: list[LabelRecord] = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(LabelRecord.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{self.path}: malformed record on line {lineno}: {exc}") from exc
        return records

    def fold_current(self) -> dict[tuple[str, str], LabelRecord]:
        current: dict[tuple[str, str], tuple[str, int, LabelRecord]] = {}
        for idx, rec in enumerate(self.read_log()):
            key = pair_key(rec.src_chunk_id, rec.dst_chunk_id)
            stamp = (rec.ts, idx)
            prev = current.get(key)
            if prev is None or stamp >= (prev[0], prev[1]):
                current[key] = (rec.ts, idx, rec)
        return {key: value[2] for key, value in current.items()}

    def gold(self) -> list[tuple[str, str, Relation, Optional[str]]]:
        return [
            (r.src_chunk_id, r.dst_chunk_id, r.relation, r.relation_class)
            for r in self.fold_current().values()
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_store.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/store.py tests/test_curator_store.py
git commit -m "Curator: append-only JSONL store with latest-wins fold (curator UI, task 2)."
```

---

## Task 3: Cheap-signal wrapper

**Files:**
- Create: `gin/curator/signals.py`
- Test: `tests/test_curator_signals.py`

**Interfaces:**
- Consumes: `gin.cartographer.combined.CombinedRelationProposer` (existing; injectable `embed_cos`/`nli_scores`/`same_story` callables make it model-free for tests).
- Produces: `pair_signals(a_text: str, b_text: str, proposer: CombinedRelationProposer) -> dict` returning `{"cosine": float, "nli_p_contra": Optional[float], "same_story": Optional[bool], "cheap_verdict": str, "channel": str}`. `nli_p_contra`/`same_story` are `None` when the proposer short-circuits before computing them (gated pair, or no story provider).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_signals.py
"""pair_signals surfaces the cheap detector's cosine / NLI / verdict for display."""
from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.signals import pair_signals


def _proposer(cos, p_contra):
    # Injected scorers make this model-free (same seam combined.py's own tests use).
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos,
        nli_scores=lambda a, b: (p_contra, 0.0, 1.0 - p_contra),
    )


def test_gated_pair_reports_cosine_and_no_nli():
    sig = pair_signals("x", "y", _proposer(cos=0.05, p_contra=0.9))
    assert sig["cheap_verdict"] == "unrelated"
    assert sig["cosine"] == 0.05
    assert sig["nli_p_contra"] is None  # gate short-circuits before NLI


def test_related_pair_reports_nli_p_contra():
    sig = pair_signals("x", "y", _proposer(cos=0.55, p_contra=0.9))
    assert sig["nli_p_contra"] == 0.9
    assert sig["cheap_verdict"] in {"contradicts", "corroborates", "related_untyped"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.signals'`

- [ ] **Step 3: Write `gin/curator/signals.py`**

```python
"""Read-only view of the cheap pipeline's signals, for display + ordering.

Wraps CombinedRelationProposer.type_relation — no new model code. Whatever the
detector already computes for a pair (cosine always; NLI p_contra only when the
pair passes the gate and isn't story-blocked) is surfaced; unavailable signals
are reported as None rather than recomputed.
"""
from __future__ import annotations

from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer


def pair_signals(a_text: str, b_text: str, proposer: CombinedRelationProposer) -> dict:
    relation, ev = proposer.type_relation(a_text, b_text)
    same_story: Optional[bool] = ev.get("same_story")
    return {
        "cosine": ev.get("cos"),
        "nli_p_contra": ev.get("p_contra"),
        "same_story": same_story,
        "cheap_verdict": relation.value,
        "channel": ev.get("channel"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_signals.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/signals.py tests/test_curator_signals.py
git commit -m "Curator: cheap-signal wrapper over CombinedRelationProposer (curator UI, task 3)."
```

---

## Task 4: Candidate source + hard-cases-first ordering

**Files:**
- Create: `gin/curator/candidates.py`
- Test: `tests/test_curator_candidates.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.LabeledChunk` (existing); `pair_key` (Task 1).
- Produces:
  - `CandidateSource` `Protocol` with `chunks(self) -> list[LabeledChunk]` and `pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]`.
  - `OfflineCandidateSource(chunks: list[LabeledChunk])` implementing it (`pairs()` = all unordered 2-combinations).
  - `informativeness(sig: dict) -> float` — tier score: `2.0` signal-disagreement (`nli_p_contra >= 0.5` and `cosine >= 0.45`), `1.0` ambiguous mid-band (`0.13 <= cosine < 0.45`), else `0.0`.
  - `order_backlog(scored, already_labeled)` where `scored: list[tuple[tuple[LabeledChunk, LabeledChunk], dict]]` and `already_labeled: set[tuple[str, str]]` — drops already-labeled pairs, sorts by `informativeness` desc then `cosine` desc then pair-id for determinism.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_candidates.py
"""Offline pair enumeration + hard-cases-first ordering."""
from gin.cartographer.models import LabeledChunk
from gin.curator.candidates import (
    OfflineCandidateSource,
    informativeness,
    order_backlog,
)
from gin.curator.models import pair_key

A = LabeledChunk("a:0", "alpha")
B = LabeledChunk("b:0", "bravo")
C = LabeledChunk("c:0", "charlie")


def test_offline_source_enumerates_unordered_pairs():
    src = OfflineCandidateSource([A, B, C])
    keys = {pair_key(x.chunk_id, y.chunk_id) for x, y in src.pairs()}
    assert keys == {pair_key("a:0", "b:0"), pair_key("a:0", "c:0"), pair_key("b:0", "c:0")}


def test_informativeness_tiers():
    assert informativeness({"cosine": 0.55, "nli_p_contra": 0.9}) == 2.0   # disagreement
    assert informativeness({"cosine": 0.30, "nli_p_contra": None}) == 1.0  # mid-band
    assert informativeness({"cosine": 0.80, "nli_p_contra": 0.05}) == 0.0  # obvious corroboration
    assert informativeness({"cosine": 0.05, "nli_p_contra": None}) == 0.0  # gated


def test_order_ranks_hard_cases_first_and_excludes_labeled():
    disagreement = ((A, B), {"cosine": 0.55, "nli_p_contra": 0.9})
    midband = ((A, C), {"cosine": 0.30, "nli_p_contra": None})
    obvious = ((B, C), {"cosine": 0.80, "nli_p_contra": 0.05})
    ordered = order_backlog([obvious, midband, disagreement], already_labeled=set())
    assert [p for p, _ in ordered] == [(A, B), (A, C), (B, C)]

    # Exclude an already-labeled pair.
    ordered2 = order_backlog(
        [obvious, midband, disagreement],
        already_labeled={pair_key("a:0", "b:0")},
    )
    assert (A, B) not in [p for p, _ in ordered2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.candidates'`

- [ ] **Step 3: Write `gin/curator/candidates.py`**

```python
"""Where pairs to label come from, and in what order to show them.

Pluggable source (offline DB-free default; a Postgres/live-residue adapter is a
later addition behind the same Protocol). Ordering is a static hard-cases-first
heuristic over the cheap pipeline's own signals: signal disagreements first,
then the ambiguous mid-band, then everything obvious — so curator time buys the
most boundary signal. No retraining loop (that needs the bi-encoder to exist).
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Protocol

from gin.cartographer.models import LabeledChunk

from .models import pair_key

GATE_FLOOR = 0.13
CORROBORATE_CEILING = 0.45
CONTRA_THRESHOLD = 0.5


class CandidateSource(Protocol):
    def chunks(self) -> list[LabeledChunk]: ...
    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]: ...


class OfflineCandidateSource:
    """DB-free source over an in-memory chunk set (the default)."""

    def __init__(self, chunks: list[LabeledChunk]) -> None:
        self._chunks = list(chunks)

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        return list(combinations(self._chunks, 2))


def informativeness(sig: dict) -> float:
    """Static tier score; higher is more worth a curator's attention."""
    cos = sig.get("cosine") or 0.0
    p = sig.get("nli_p_contra")
    if p is not None and p >= CONTRA_THRESHOLD and cos >= CORROBORATE_CEILING:
        return 2.0  # signal disagreement: NLI says contradict, cosine says corroborate
    if GATE_FLOOR <= cos < CORROBORATE_CEILING:
        return 1.0  # ambiguous mid-band (includes the not-same-story residue)
    return 0.0


def order_backlog(
    scored: list[tuple[tuple[LabeledChunk, LabeledChunk], dict]],
    already_labeled: set[tuple[str, str]],
) -> list[tuple[tuple[LabeledChunk, LabeledChunk], dict]]:
    unlabeled = [
        (pair, sig)
        for pair, sig in scored
        if pair_key(pair[0].chunk_id, pair[1].chunk_id) not in already_labeled
    ]
    unlabeled.sort(
        key=lambda item: (
            -informativeness(item[1]),
            -(item[1].get("cosine") or 0.0),
            item[0][0].chunk_id,
            item[0][1].chunk_id,
        )
    )
    return unlabeled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_candidates.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/candidates.py tests/test_curator_candidates.py
git commit -m "Curator: pluggable candidate source + hard-cases-first ordering (curator UI, task 4)."
```

---

## Task 5: Seed importer + regression guard

**Files:**
- Create: `gin/curator/seed.py`
- Test: `tests/test_curator_seed.py`

**Interfaces:**
- Consumes: `Store` (Task 2), `LabelRecord`, `pair_key` (Task 1); existing `gin.cartographer.labeled_set.gold()` and `gin.cartographer.gold_edges.load_all_gold_contradicts()`.
- Produces:
  - `seed_records(curator: str = "seed", ts: str = "2026-07-17T00:00:00Z") -> list[LabelRecord]` — one record per existing gold pair. `labeled_set` pairs (which carry `register`, not a class) seed with `relation_class=None`, **never a guessed class**; `gold_edges` contradicts pairs carry their YAML `relation_class`. `labeled_set` is emitted first so it wins any pair collision.
  - `seed_store(store: Store, **kw) -> int` — appends each seed record whose `pair_key` is not already in the store's fold; returns the count appended (idempotent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_seed.py
"""Seeding imports the existing ~33 labels; the regression guard proves the
store reproduces today's gold before it grows it."""
from gin.cartographer import gold_edges, labeled_set
from gin.cartographer.models import Relation
from gin.curator.models import pair_key
from gin.curator.seed import seed_store
from gin.curator.store import Store


def test_seed_reproduces_labeled_set_relations(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    fold = store.fold_current()
    for src, dst, relation, _register in labeled_set.gold():
        key = pair_key(src, dst)
        assert key in fold, f"seeded pair missing: {key}"
        assert fold[key].relation is relation


def test_labeled_set_contradicts_seed_with_null_class(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    fold = store.fold_current()
    # inst_em:0 <-> grass_em:0 is a labeled_set CONTRADICTS pair; labeled_set
    # carries no story/issue_frame tag, so it seeds as None (not a guess).
    rec = fold[pair_key("inst_em:0", "grass_em:0")]
    assert rec.relation is Relation.CONTRADICTS
    assert rec.relation_class is None


def test_gold_edges_class_is_preserved(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    fold = store.fold_current()
    edges = gold_edges.load_all_gold_contradicts()
    assert edges  # fixtures exist in-repo
    classed = [
        fold[pair_key(e.src_chunk_id, e.dst_chunk_id)]
        for e in edges
        if pair_key(e.src_chunk_id, e.dst_chunk_id) in fold
    ]
    assert classed
    assert all(r.relation_class in {"story", "issue_frame"} for r in classed)


def test_seed_is_idempotent(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    first = seed_store(store)
    assert first > 0
    assert seed_store(store) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.seed'`

- [ ] **Step 3: Write `gin/curator/seed.py`**

```python
"""One-time import of the existing gold into the store as seed records.

labeled_set carries `register` (not a story/issue_frame class), so its
contradicts pairs seed with relation_class=None — the importer never GUESSES a
class. gold_edges carries the YAML relation_class and keeps it. labeled_set is
emitted first, so on any pair collision the labeled_set relation wins (this is
what the regression guard asserts).
"""
from __future__ import annotations

import uuid

from gin.cartographer import gold_edges, labeled_set
from gin.cartographer.models import Relation

from .models import LabelRecord, pair_key
from .store import Store


def seed_records(curator: str = "seed", ts: str = "2026-07-17T00:00:00Z") -> list[LabelRecord]:
    records: list[LabelRecord] = []
    for src, dst, relation, _register in labeled_set.gold():
        records.append(
            LabelRecord(
                id=str(uuid.uuid4()), src_chunk_id=src, dst_chunk_id=dst,
                relation=relation, relation_class=None,
                rationale="", curator=curator, ts=ts,
            )
        )
    for e in gold_edges.load_all_gold_contradicts():
        records.append(
            LabelRecord(
                id=str(uuid.uuid4()), src_chunk_id=e.src_chunk_id, dst_chunk_id=e.dst_chunk_id,
                relation=Relation.CONTRADICTS, relation_class=e.relation_class,
                rationale=e.note, curator=curator, ts=ts,
            )
        )
    return records


def seed_store(store: Store, curator: str = "seed", ts: str = "2026-07-17T00:00:00Z") -> int:
    present = set(store.fold_current().keys())
    appended = 0
    for rec in seed_records(curator=curator, ts=ts):
        key = pair_key(rec.src_chunk_id, rec.dst_chunk_id)
        if key in present:
            continue
        store.append(rec)
        present.add(key)
        appended += 1
    return appended
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_seed.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/seed.py tests/test_curator_seed.py
git commit -m "Curator: seed importer + gold-reproduction regression guard (curator UI, task 5)."
```

---

## Task 6: FastAPI app — endpoints + labeling page

**Files:**
- Create: `gin/curator/app.py`
- Test: `tests/test_curator_app.py`

**Interfaces:**
- Consumes: `Store` (Task 2), `CandidateSource`/`order_backlog` (Task 4), `pair_key`/`LabelRecord` (Task 1); `gin.cartographer.scan.sentence_anchor` (existing, `text -> (int, int)`); `gin.cartographer.models.Relation`.
- Produces: `create_curator_app(*, store: Store, source: CandidateSource, signals_fn: Callable[[str, str], dict], curator: str = "curator", scan_limit: int = 500) -> FastAPI` with:
  - `GET /curator/` → `HTMLResponse` (the labeling page).
  - `GET /curator/next?n=<int>` → JSON `{"pairs": [{"src", "dst", "src_text", "dst_text", "signals"}], "labeled": int, "remaining": int}`.
  - `POST /curator/label` (body `LabelRequest`: `src_chunk_id`, `dst_chunk_id`, `relation`, `relation_class=None`, `rationale=""`) → `{"ok": true, "id": <str>}`; `422` if `relation` is not in the vocab, or `relation == "contradicts"` with no `relation_class`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_app.py
"""Curator endpoints: ordered /next, /label append + supersede, class validation."""
from pathlib import Path

from fastapi.testclient import TestClient

from gin.cartographer.models import LabeledChunk
from gin.curator.app import create_curator_app
from gin.curator.candidates import OfflineCandidateSource
from gin.curator.models import pair_key
from gin.curator.store import Store

CHUNKS = [LabeledChunk("a:0", "alpha text"), LabeledChunk("b:0", "bravo text"),
          LabeledChunk("c:0", "charlie text")]


def _fake_signals(a_text: str, b_text: str) -> dict:
    # Deterministic per-pair signals so ordering is assertable without a model.
    table = {
        frozenset({"alpha text", "bravo text"}): {"cosine": 0.55, "nli_p_contra": 0.9,
                                                   "same_story": None, "cheap_verdict": "contradicts"},
        frozenset({"alpha text", "charlie text"}): {"cosine": 0.30, "nli_p_contra": None,
                                                     "same_story": None, "cheap_verdict": "related_untyped"},
        frozenset({"bravo text", "charlie text"}): {"cosine": 0.80, "nli_p_contra": 0.05,
                                                     "same_story": None, "cheap_verdict": "corroborates"},
    }
    return table[frozenset({a_text, b_text})]


def _client(tmp_path: Path) -> TestClient:
    store = Store(tmp_path / "labels.jsonl")
    app = create_curator_app(
        store=store, source=OfflineCandidateSource(CHUNKS), signals_fn=_fake_signals,
    )
    return TestClient(app)


def test_next_returns_hard_cases_first_with_text_and_signals(tmp_path):
    r = _client(tmp_path).get("/curator/next?n=10")
    assert r.status_code == 200
    data = r.json()
    assert data["labeled"] == 0
    first = data["pairs"][0]
    assert {first["src"], first["dst"]} == {"a:0", "b:0"}  # the disagreement pair ranks first
    assert first["src_text"] and first["dst_text"]
    assert first["signals"]["cosine"] == 0.55


def test_label_appends_one_record_and_reflects_in_next(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    app = create_curator_app(store=store, source=OfflineCandidateSource(CHUNKS), signals_fn=_fake_signals)
    client = TestClient(app)
    r = client.post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0",
        "relation": "contradicts", "relation_class": "issue_frame", "rationale": "why",
    })
    assert r.status_code == 200 and r.json()["ok"] is True
    assert len(store.read_log()) == 1
    assert store.fold_current()[pair_key("a:0", "b:0")].relation.value == "contradicts"
    assert client.get("/curator/next?n=10").json()["labeled"] == 1


def test_contradicts_without_class_is_rejected(tmp_path):
    r = _client(tmp_path).post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0", "relation": "contradicts",
    })
    assert r.status_code == 422


def test_unknown_relation_is_rejected(tmp_path):
    r = _client(tmp_path).post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0", "relation": "banana",
    })
    assert r.status_code == 422


def test_relabel_supersedes_prior(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    app = create_curator_app(store=store, source=OfflineCandidateSource(CHUNKS), signals_fn=_fake_signals)
    client = TestClient(app)
    first = client.post("/curator/label", json={
        "src_chunk_id": "a:0", "dst_chunk_id": "b:0", "relation": "corroborates",
    }).json()["id"]
    client.post("/curator/label", json={
        "src_chunk_id": "b:0", "dst_chunk_id": "a:0",
        "relation": "contradicts", "relation_class": "story",
    })
    log = store.read_log()
    assert len(log) == 2
    assert log[1].supersedes == first
    assert store.fold_current()[pair_key("a:0", "b:0")].relation.value == "contradicts"


def test_index_page_served(tmp_path):
    r = _client(tmp_path).get("/curator/")
    assert r.status_code == 200
    assert "GIN Curator" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.app'`

- [ ] **Step 3: Write `gin/curator/app.py`**

```python
"""Local-only FastAPI labeling app: one page + two JSON endpoints.

signals_fn is injected (real: pair_signals bound to a CombinedRelationProposer;
tests: a fake) so the endpoints are model-free under test. The store is the
source of truth; the candidate source only drives ORDERING, so a label for a
pair the source doesn't enumerate is still accepted and stored.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from gin.cartographer.models import Relation
from gin.cartographer.scan import sentence_anchor

from .candidates import CandidateSource, order_backlog
from .models import LabelRecord, pair_key
from .store import Store

_VALID_RELATIONS = {r.value for r in Relation}


class LabelRequest(BaseModel):
    src_chunk_id: str
    dst_chunk_id: str
    relation: str
    relation_class: Optional[str] = None
    rationale: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_curator_app(
    *,
    store: Store,
    source: CandidateSource,
    signals_fn: Callable[[str, str], dict],
    curator: str = "curator",
    scan_limit: int = 500,
) -> FastAPI:
    app = FastAPI(title="GIN Curator")
    text_by_id = {c.chunk_id: c.text for c in source.chunks()}

    @app.get("/curator/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE_HTML

    @app.get("/curator/next")
    def next_pairs(n: int = 20) -> dict:
        labeled = set(store.fold_current().keys())
        scored = []
        for a, b in source.pairs():
            if pair_key(a.chunk_id, b.chunk_id) in labeled:
                continue
            scored.append(((a, b), signals_fn(a.text, b.text)))
            if len(scored) >= scan_limit:
                break
        ordered = order_backlog(scored, already_labeled=set())
        pairs = [
            {
                "src": a.chunk_id, "dst": b.chunk_id,
                "src_text": a.text, "dst_text": b.text, "signals": sig,
            }
            for (a, b), sig in ordered[:n]
        ]
        return {"pairs": pairs, "labeled": len(labeled), "remaining": len(ordered)}

    @app.post("/curator/label")
    def label(req: LabelRequest) -> dict:
        if req.relation not in _VALID_RELATIONS:
            raise HTTPException(status_code=422, detail=f"unknown relation {req.relation!r}")
        if req.relation == Relation.CONTRADICTS.value and not req.relation_class:
            raise HTTPException(
                status_code=422, detail="relation_class (story|issue_frame) required for contradicts"
            )
        if req.relation_class is not None and req.relation_class not in {"story", "issue_frame"}:
            raise HTTPException(status_code=422, detail=f"bad relation_class {req.relation_class!r}")

        prior = store.fold_current().get(pair_key(req.src_chunk_id, req.dst_chunk_id))
        src_text = text_by_id.get(req.src_chunk_id, "")
        dst_text = text_by_id.get(req.dst_chunk_id, "")
        rec = LabelRecord(
            id=str(uuid.uuid4()),
            src_chunk_id=req.src_chunk_id,
            dst_chunk_id=req.dst_chunk_id,
            relation=Relation(req.relation),
            relation_class=req.relation_class if req.relation == Relation.CONTRADICTS.value else None,
            rationale=req.rationale,
            curator=curator,
            ts=_now_iso(),
            supersedes=prior.id if prior is not None else None,
            src_anchor=sentence_anchor(src_text) if src_text else None,
            dst_anchor=sentence_anchor(dst_text) if dst_text else None,
        )
        store.append(rec)
        return {"ok": True, "id": rec.id}

    return app


PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>GIN Curator</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
 #progress{color:#666;margin-bottom:1rem}
 .panels{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
 .panel{border:1px solid #ccc;border-radius:6px;padding:1rem;background:#fafafa}
 .panel h3{margin:0 0 .5rem;font-size:.8rem;text-transform:uppercase;color:#888}
 #signals{font-family:ui-monospace,monospace;font-size:.85rem;color:#444;margin:1rem 0;padding:.5rem;background:#f0f0f0;border-radius:4px}
 .rel{margin:.2rem;padding:.5rem .8rem;border:1px solid #999;border-radius:4px;background:#fff;cursor:pointer}
 .rel.sel{background:#2a6;color:#fff;border-color:#2a6}
 #class-row{margin:.6rem 0;display:none}
 textarea{width:100%;min-height:3rem;margin:.5rem 0}
 #save{padding:.5rem 1.2rem;font-size:1rem}
 kbd{background:#eee;border:1px solid #bbb;border-radius:3px;padding:0 .3rem;font-size:.8rem}
</style></head><body>
<h2>GIN Curator</h2>
<div id="progress"></div>
<div class="panels">
  <div class="panel"><h3>A</h3><p id="a"></p></div>
  <div class="panel"><h3>B</h3><p id="b"></p></div>
</div>
<div id="signals"></div>
<div id="rels">
  <button class="rel" data-rel="contradicts">1 contradicts</button>
  <button class="rel" data-rel="corroborates">2 corroborates</button>
  <button class="rel" data-rel="supersedes">3 supersedes</button>
  <button class="rel" data-rel="related_untyped">4 related_untyped</button>
  <button class="rel" data-rel="unrelated">5 unrelated</button>
</div>
<div id="class-row">
  class: <label><input type="radio" name="cls" value="story"> story</label>
  <label><input type="radio" name="cls" value="issue_frame"> issue_frame</label>
</div>
<textarea id="rationale" placeholder="rationale (optional)"></textarea>
<div><button id="save">Save &amp; next <kbd>Enter</kbd></button></div>
<script>
const RELS=["contradicts","corroborates","supersedes","related_untyped","unrelated"];
let queue=[],cur=null,pending=null;
function fmt(x){return x==null?"\\u2013":Number(x).toFixed(3);}
async function loadNext(){
  const d=await (await fetch("/curator/next?n=20")).json();
  queue=d.pairs;
  document.getElementById("progress").textContent=`labeled ${d.labeled} \\u00b7 remaining ${d.remaining}`;
  show();
}
function show(){
  pending=null;
  document.querySelectorAll(".rel").forEach(b=>b.classList.remove("sel"));
  document.getElementById("class-row").style.display="none";
  document.querySelectorAll("input[name=cls]").forEach(i=>i.checked=false);
  document.getElementById("rationale").value="";
  if(queue.length===0){loadNext();return;}
  cur=queue.shift();
  document.getElementById("a").textContent=cur.src_text;
  document.getElementById("b").textContent=cur.dst_text;
  const s=cur.signals;
  document.getElementById("signals").textContent=
    `cheap=${s.cheap_verdict}  cos=${fmt(s.cosine)}  p_contra=${fmt(s.nli_p_contra)}  same_story=${s.same_story}`;
}
function pick(rel){
  pending=rel;
  document.querySelectorAll(".rel").forEach(b=>b.classList.toggle("sel",b.dataset.rel===rel));
  document.getElementById("class-row").style.display=(rel==="contradicts")?"block":"none";
}
async function save(){
  if(!pending||!cur)return;
  let cls=null;
  if(pending==="contradicts"){
    const c=document.querySelector("input[name=cls]:checked");
    if(!c){alert("pick story or issue_frame");return;}
    cls=c.value;
  }
  await fetch("/curator/label",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({src_chunk_id:cur.src,dst_chunk_id:cur.dst,relation:pending,
      relation_class:cls,rationale:document.getElementById("rationale").value})});
  show();
}
document.querySelectorAll(".rel").forEach(b=>b.addEventListener("click",()=>pick(b.dataset.rel)));
document.getElementById("save").addEventListener("click",save);
document.addEventListener("keydown",e=>{
  const n=parseInt(e.key);
  if(n>=1&&n<=RELS.length){pick(RELS[n-1]);}
  else if(e.key==="Enter"){e.preventDefault();save();}
});
loadNext();
</script></body></html>
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_app.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite to confirm zero regression**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: every pre-existing test still passes (this package is additive; nothing existing imports `gin.curator`), plus the new curator tests.

- [ ] **Step 6: Commit**

```bash
git add gin/curator/app.py tests/test_curator_app.py
git commit -m "Curator: FastAPI labeling endpoints + no-build-step page (curator UI, task 6)."
```

---

## Task 7: Launcher script + manual smoke run

**Files:**
- Create: `scripts/curator_serve.py`
- Modify: `.gitignore` (ignore nothing new — `data/curator/labels.jsonl` is intentionally tracked; this step only confirms it is not caught by an existing ignore)

**Interfaces:**
- Consumes: `Store`, `OfflineCandidateSource`, `pair_signals` (bound to a real `CombinedRelationProposer`), `create_curator_app`, `labeled_set.chunks()`.
- Produces: a runnable local server. No unit test — a launcher, verified by the manual smoke run below (same posture as prior sub-projects' live-eval steps).

The default offline chunk set is `labeled_set.chunks()` (the fixture chunks already embedded in the repo — no new data file invented). `--chunks corpus_node1.json` can point it at a corpus export later; for v1 the launcher wires the fixture chunks.

- [ ] **Step 1: Write `scripts/curator_serve.py`**

```python
"""Launch the local curator labeling app.

    venv/Scripts/python.exe scripts/curator_serve.py

Serves http://127.0.0.1:8600/curator/ over the fixture chunk set, appending
labels to data/curator/labels.jsonl. Seeds the ~33 existing labels on first run
so already-known pairs are not re-surfaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from gin.cartographer import labeled_set
from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.app import create_curator_app
from gin.curator.candidates import OfflineCandidateSource
from gin.curator.seed import seed_store
from gin.curator.signals import pair_signals
from gin.curator.store import Store

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description="GIN curator labeling app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--no-seed", action="store_true", help="skip seeding existing gold")
    args = ap.parse_args()

    store = Store(args.log)
    if not args.no_seed:
        added = seed_store(store)
        print(f"seeded {added} existing labels into {args.log}")

    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded
    source = OfflineCandidateSource(labeled_set.chunks())
    app = create_curator_app(
        store=store,
        source=source,
        signals_fn=lambda a, b: pair_signals(a, b, proposer),
        curator="kristian",
    )
    print(f"curator UI: http://{args.host}:{args.port}/curator/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the label path is git-trackable**

Run: `git check-ignore data/curator/labels.jsonl; echo "exit=$?"`
Expected: `exit=1` (not ignored). If it prints a matching rule and `exit=0`, add a negation `!data/curator/labels.jsonl` to `.gitignore` so the label corpus is tracked.

- [ ] **Step 3: Manual smoke run**

```bash
venv/Scripts/python.exe scripts/curator_serve.py
```

Expected: prints `seeded N existing labels ...` (N > 0 on first run, 0 afterward) then `curator UI: http://127.0.0.1:8600/curator/`. Open that URL, confirm a pair renders with both panels and a signal line, press `1`–`5` to pick a relation (a `contradicts` pick reveals the class toggle), press Enter, and confirm the progress counter's `labeled` increments. Stop with Ctrl-C.

- [ ] **Step 4: Confirm a real labeling act persisted, then commit the script + seeded log**

Run: `venv/Scripts/python.exe -c "from pathlib import Path; from gin.curator.store import Store; print(len(Store(Path('data/curator/labels.jsonl')).read_log()), 'records')"`
Expected: a record count ≥ the seed count.

```bash
git add scripts/curator_serve.py data/curator/labels.jsonl
git commit -m "Curator: local launcher script + seeded label log (curator UI, task 7)."
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`
- Modify: `architecture.md`

- [ ] **Step 1: Add a "Curator labeling tool" subsection to `README.md`**

Add a subsection covering: launching `scripts/curator_serve.py` (`venv/Scripts/python.exe scripts/curator_serve.py`, then open `http://127.0.0.1:8600/curator/`); the `data/curator/labels.jsonl` record shape (the `LabelRecord.to_json()` keys); that pairs are keyed order-independently and the current gold is the latest-wins fold of the log; the `relation` vocabulary and that `relation_class` (`story`|`issue_frame`) is required for `contradicts`; and that `gin.curator.store.Store.gold()` is the reader later work (bi-encoder / recalibration) consumes.

- [ ] **Step 2: Add a note to `architecture.md`**

Add a short note that the curator + JSONL label store is the shared framing-corpus spine feeding the (still-open) bi-encoder frame detector (sub-project B) and the cheap-pipeline recalibration (sub-project C), and that it operationalizes the escalation-judge conclusion that issue_frame is curation-only by nature (no off-the-shelf judge, 7B→Opus-4.8, reproduces the stance). Note the package is strictly additive — it consumes `gin.cartographer` and is imported by nothing.

- [ ] **Step 3: Commit**

```bash
git add README.md architecture.md
git commit -m "Docs: curator UI + label store (framing-corpus spine, sub-project A)."
```

---

## Self-Review Notes

**Spec coverage:** every spec section maps to a task — falsifiable-claim bars: store round-trip (Task 2), seed regression guard (Task 5), ordering (Task 4), reader shape `gold()` (Task 2), end-to-end append via app (Task 6), existing-suite-green (Task 6 Step 5), no new deps (Global Constraints). Architecture modules: `models` (Task 1), `store` (Task 2), `signals` (Task 3), `candidates` (Task 4), `seed` (Task 5), `app` (Task 6), `scripts/curator_serve.py` (Task 7). Data flow (serve → /next fold+order → label append+supersede): Tasks 6–7. Error handling: contradicts-without-class 422 + unknown-relation 422 + label-for-unknown-pair-accepted (Task 6), malformed-line loud failure + empty store (Task 2). Out-of-scope items (bi-encoder, recalibration/loader-unification, active learning, Postgres source, manual anchors, multi-curator) are left unbuilt by construction. Docs (Task 8).

**Additive-only, concretely:** no task modifies `labeled_set.py`, `gold_edges.py`, `calibration.py`, or `combined.py`; `gin/curator/` is imported by nothing (new top layer). Task 6 Step 5 runs the full suite to prove zero regression. The seed importer reads the existing loaders but does not change them.

**Type/interface consistency:** `pair_key` returns a 2-tuple used identically as a dict key in `store.fold_current`, `candidates.order_backlog`, `seed.seed_store`, and `app` (supersede lookup). `Store.gold()` and `LabelRecord` field names (`src_chunk_id`, `dst_chunk_id`, `relation`, `relation_class`, `supersedes`) are used consistently across Tasks 2/5/6. `signals_fn: Callable[[str, str], dict]` (injected in Task 6, bound to `pair_signals` in Task 7) has the same shape the fake returns in `test_curator_app.py`. `informativeness`'s keys (`cosine`, `nli_p_contra`) match what `pair_signals` (Task 3) emits and what the app forwards in each pair's `signals`. `create_curator_app` is keyword-only in both its test (Task 6) and its caller (Task 7).

**Placeholder scan:** no TBD/TODO; every code step is complete and grounded in verified interfaces — `CombinedRelationProposer.type_relation`'s return shape, `sentence_anchor(text) -> (int, int)`, `labeled_set.gold()`'s 4-tuple, and `gold_edges.load_all_gold_contradicts()`'s `relation_class` were all read from source before writing this plan; the 33 labeled_set pairs / 13 gold_edges contradicts (classes `story`+`issue_frame`) counts that Task 5's guard relies on were confirmed by running the loaders.
