# Curator issue_frame Labeling Productivity (B0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make issue_frame labeling reachable and measurable so sub-project B (the bi-encoder) can eventually train on real held-out data — via an escalation-residue candidate source over the full corpus plus a no-model class-count readiness gauge.

**Architecture:** Additive to sub-project A. New `gin/curator/` modules — `corpus_json` (load + normalize corpus chunk ids), `residue` (an `EscalationResidueCandidateSource` reusing `cartographer.escalation.escalation_candidates`), and `readiness` (a count-only gauge) — plus a readiness endpoint on A's existing app and two CLI extensions. B0 trains no model; the readiness gauge is pure counting over the folded label store.

**Tech Stack:** Python 3.10+, stdlib `json`/`pathlib`/`itertools`/`dataclasses`, FastAPI (already present), and `gin.cartographer` + sub-project A's `gin.curator`. No new dependencies.

## Global Constraints

- **No new dependencies.**
- **B0 trains no model and expands no corpus.** The readiness gauge is pure counting; no embedding-head/classifier is built. Bi-encoder training (B proper), corpus expansion, active-learning retraining, and a live-Postgres residue source are all out of scope.
- **Additive:** only two existing files are modified — `gin/curator/app.py` and `scripts/curator_serve.py` — plus `README.md`/`architecture.md` in the docs task. No other existing file (in `gin/cartographer`, `gin/curator`, `gin/eval`, `gin/federation`, `sear`) may be modified. The full existing suite stays green.
- **Layering:** `gin/curator/` may import from `gin/cartographer/`; nothing in cartographer/eval/bookkeeper/federation may import from `gin/curator/`.
- **chunk-id normalization:** corpus chunks load as `LabeledChunk(f"{doc_id}:{position}", text)` (e.g. `n1_doc_005:2`), matching the gold/escalation-bar/store convention. `corpus_node*.json` stores them as `n1_doc_005_c002` — never use that raw form.
- **Reuse, don't re-implement:** the residue definition is `gin.cartographer.escalation.escalation_candidates` (not same-story, cosine ≥ `cos_floor`, cosine-sorted, top `max_candidates`; `DEFAULT_ESCALATION_COS_FLOOR=0.30`, `DEFAULT_MAX_CANDIDATES=400`). The excluded bar pairs come from `gin.cartographer.escalation_eval.default_calibration_sets()`.
- **Readiness counting rule:** `new_issue_frame` = store labels with `relation==CONTRADICTS` **and** `relation_class=="issue_frame"`, pair ∉ bar (None-class contradicts are NOT counted). `new_agree` = `corroborates`, pair ∉ bar. `new_unrelated` = `unrelated`, pair ∉ bar (these DO include the disjoint labeled_set seed controls). Only the 14 bar pairs are excluded. `ready` iff all three ≥ target.
- **Test runner:** `venv/Scripts/python.exe -m pytest` (Windows venv; `PYTHONIOENCODING=utf-8` if box/emoji output appears). Model-backed paths are exercised only in a manual smoke; automated tests inject fakes.

---

## Task 1: Corpus JSON loader with chunk-id normalization

**Files:**
- Create: `gin/curator/corpus_json.py`
- Test: `tests/test_curator_corpus_json.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.LabeledChunk` (existing; `LabeledChunk(chunk_id: str, text: str)`).
- Produces: `load_corpus_chunks(paths: Iterable[Path | str]) -> list[LabeledChunk]` — flattens each `corpus_node*.json`'s `documents[].chunks[]` into `LabeledChunk(f"{doc_id}:{position}", text)`, deduped by chunk id (first wins), order preserved. Raises `FileNotFoundError` for a missing path and `ValueError` for a document missing `doc_id` or a chunk missing `position`/`text`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_corpus_json.py
"""Corpus JSON loading + chunk-id normalization to {doc_id}:{position}."""
import json

import pytest

from gin.curator.corpus_json import load_corpus_chunks

FIXTURE = {
    "node_id": "node_1_institutional",
    "documents": [
        {"doc_id": "n1_doc_005", "chunks": [
            {"chunk_id": "n1_doc_005_c000", "position": 0, "text": "alpha text"},
            {"chunk_id": "n1_doc_005_c002", "position": 2, "text": "gamma text"},
        ]},
        {"doc_id": "n1_doc_008", "chunks": [
            {"chunk_id": "n1_doc_008_c000", "position": 0, "text": "delta text"},
        ]},
    ],
}


def _write(tmp_path, data, name="corpus.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_flattens_and_normalizes_chunk_ids(tmp_path):
    chunks = load_corpus_chunks([_write(tmp_path, FIXTURE)])
    ids = [c.chunk_id for c in chunks]
    assert ids == ["n1_doc_005:0", "n1_doc_005:2", "n1_doc_008:0"]
    assert chunks[1].text == "gamma text"


def test_dedupes_by_chunk_id_first_wins(tmp_path):
    dup = {"documents": [
        {"doc_id": "d", "chunks": [
            {"position": 0, "text": "first"},
            {"position": 0, "text": "second"},
        ]},
    ]}
    chunks = load_corpus_chunks([_write(tmp_path, dup)])
    assert [c.chunk_id for c in chunks] == ["d:0"]
    assert chunks[0].text == "first"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus_chunks([tmp_path / "nope.json"])


def test_missing_position_raises(tmp_path):
    bad = {"documents": [{"doc_id": "d", "chunks": [{"text": "no position"}]}]}
    with pytest.raises(ValueError, match="position"):
        load_corpus_chunks([_write(tmp_path, bad)])


def test_missing_doc_id_raises(tmp_path):
    bad = {"documents": [{"chunks": [{"position": 0, "text": "x"}]}]}
    with pytest.raises(ValueError, match="doc_id"):
        load_corpus_chunks([_write(tmp_path, bad)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_corpus_json.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.corpus_json'`

- [ ] **Step 3: Write `gin/curator/corpus_json.py`**

```python
"""Load corpus_node*.json exports into LabeledChunks, DB-free.

Normalizes chunk ids to the {doc_id}:{position} convention the gold, the
escalation bar, and the curator store all use — the JSON stores them as
n1_doc_005_c002, which would never match those keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from gin.cartographer.models import LabeledChunk


def load_corpus_chunks(paths: Iterable[Union[Path, str]]) -> list[LabeledChunk]:
    chunks: dict[str, LabeledChunk] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"corpus file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for doc in data.get("documents", []):
            if "doc_id" not in doc:
                raise ValueError(f"{path}: document missing 'doc_id'")
            doc_id = doc["doc_id"]
            for ch in doc.get("chunks", []):
                if "position" not in ch:
                    raise ValueError(f"{path}: chunk in {doc_id} missing 'position'")
                if "text" not in ch:
                    raise ValueError(f"{path}: chunk {doc_id}:{ch['position']} missing 'text'")
                cid = f"{doc_id}:{ch['position']}"
                if cid not in chunks:
                    chunks[cid] = LabeledChunk(cid, ch["text"])
    return list(chunks.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_corpus_json.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/corpus_json.py tests/test_curator_corpus_json.py
git commit -m "Curator: corpus_node*.json loader with chunk-id normalization (B0, task 1)."
```

---

## Task 2: Readiness gauge

**Files:**
- Create: `gin/curator/readiness.py`
- Test: `tests/test_curator_readiness.py`

**Interfaces:**
- Consumes: `gin.curator.store.Store` (existing; `store.gold() -> list[tuple[str, str, Relation, Optional[str]]]`), `gin.curator.models.pair_key`, `gin.cartographer.models.Relation`, `gin.cartographer.escalation_eval.default_calibration_sets()` (existing; returns `{"issue_frame": [(src,dst,reg)...], "corroboration": [...], "unrelated": [...]}`).
- Produces:
  - `ReadinessTarget(issue_frame: int = 20, agree: int = 20, unrelated: int = 20)` frozen dataclass.
  - `ReadinessReport(new_issue_frame: int, new_agree: int, new_unrelated: int, target: ReadinessTarget, ready: bool)` frozen dataclass.
  - `bar_pair_keys() -> set[tuple[str, str]]` — the 14 escalation-bar pairs as `pair_key`s.
  - `readiness(store: Store, target: ReadinessTarget = ReadinessTarget()) -> ReadinessReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_readiness.py
"""No-model readiness gauge: counts NEW (non-bar) labels per class vs a target."""
from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord
from gin.curator.readiness import ReadinessTarget, bar_pair_keys, readiness
from gin.curator.store import Store


def _rec(src, dst, relation, ts, relation_class=None):
    return LabelRecord(id=f"{src}-{dst}", src_chunk_id=src, dst_chunk_id=dst,
                       relation=relation, relation_class=relation_class,
                       rationale="", curator="t", ts=ts)


def test_bar_pairs_have_14_keys():
    assert len(bar_pair_keys()) == 14  # 4 issue_frame + 6 corroboration + 4 unrelated


def test_seeded_bar_issue_frame_counts_as_zero_new(tmp_path):
    # A store holding ONLY the 4 escalation-bar issue_frame pairs => 0 new.
    store = Store(tmp_path / "labels.jsonl")
    for i, (src, dst, _reg) in enumerate(default_calibration_sets()["issue_frame"]):
        store.append(_rec(src, dst, Relation.CONTRADICTS, f"2026-07-17T00:00:0{i}Z",
                          relation_class="issue_frame"))
    rep = readiness(store)
    assert rep.new_issue_frame == 0
    assert rep.ready is False


def test_counts_new_labels_and_verdict_flips(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    target = ReadinessTarget(issue_frame=2, agree=1, unrelated=1)
    # Two NEW issue_frame (non-bar), one agree, one unrelated.
    store.append(_rec("x:0", "y:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "issue_frame"))
    store.append(_rec("x:1", "y:1", Relation.CONTRADICTS, "2026-07-17T00:00:01Z", "issue_frame"))
    store.append(_rec("p:0", "q:0", Relation.CORROBORATES, "2026-07-17T00:00:02Z"))
    store.append(_rec("m:0", "n:0", Relation.UNRELATED, "2026-07-17T00:00:03Z"))
    rep = readiness(store, target)
    assert (rep.new_issue_frame, rep.new_agree, rep.new_unrelated) == (2, 1, 1)
    assert rep.ready is True
    # One short on issue_frame => not ready.
    assert readiness(store, ReadinessTarget(issue_frame=3, agree=1, unrelated=1)).ready is False


def test_none_class_contradicts_not_counted_as_issue_frame(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("x:0", "y:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", None))
    assert readiness(store).new_issue_frame == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_readiness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.readiness'`

- [ ] **Step 3: Write `gin/curator/readiness.py`**

```python
"""No-model readiness gauge for sub-project B (bi-encoder).

Counts NEW labeled pairs per frame class, EXCLUDING the fixed escalation-bar
14 pairs (so the bar's own data never counts as progress toward training a
detector that will be measured on it). Pure counting — trains nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.models import Relation

from .models import pair_key
from .store import Store


@dataclass(frozen=True)
class ReadinessTarget:
    issue_frame: int = 20
    agree: int = 20
    unrelated: int = 20


@dataclass(frozen=True)
class ReadinessReport:
    new_issue_frame: int
    new_agree: int
    new_unrelated: int
    target: ReadinessTarget
    ready: bool


def bar_pair_keys() -> set[tuple[str, str]]:
    """The fixed escalation-bar pairs (issue_frame + corroboration + unrelated)."""
    keys: set[tuple[str, str]] = set()
    for group in default_calibration_sets().values():
        for src, dst, _reg in group:
            keys.add(pair_key(src, dst))
    return keys


def readiness(store: Store, target: ReadinessTarget = ReadinessTarget()) -> ReadinessReport:
    bar = bar_pair_keys()
    n_if = n_ag = n_un = 0
    for src, dst, relation, relation_class in store.gold():
        if pair_key(src, dst) in bar:
            continue
        if relation is Relation.CONTRADICTS and relation_class == "issue_frame":
            n_if += 1
        elif relation is Relation.CORROBORATES:
            n_ag += 1
        elif relation is Relation.UNRELATED:
            n_un += 1
    ready = (
        n_if >= target.issue_frame
        and n_ag >= target.agree
        and n_un >= target.unrelated
    )
    return ReadinessReport(n_if, n_ag, n_un, target, ready)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_readiness.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/readiness.py tests/test_curator_readiness.py
git commit -m "Curator: no-model class-count readiness gauge, bar pairs excluded (B0, task 2)."
```

---

## Task 3: Escalation-residue candidate source

**Files:**
- Create: `gin/curator/residue.py`
- Test: `tests/test_curator_residue.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.LabeledChunk`; `gin.cartographer.combined.CombinedRelationProposer` (injectable `same_story`/`embed_cos`); `gin.cartographer.scan.wire_same_story(proposer, chunks)`; `gin.cartographer.escalation.escalation_candidates(pairs, proposer, *, cos_floor, max_candidates)` with `DEFAULT_ESCALATION_COS_FLOOR=0.30`, `DEFAULT_MAX_CANDIDATES=400`.
- Produces: `EscalationResidueCandidateSource` implementing sub-project A's `CandidateSource` protocol (`chunks() -> list[LabeledChunk]`, `pairs() -> list[tuple[LabeledChunk, LabeledChunk]]`). Constructor: `EscalationResidueCandidateSource(chunks, *, proposer=None, cos_floor=DEFAULT_ESCALATION_COS_FLOOR, max_candidates=DEFAULT_MAX_CANDIDATES)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_residue.py
"""EscalationResidueCandidateSource reuses escalation_candidates (model-free via
an injected proposer)."""
from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk
from gin.curator.models import pair_key
from gin.curator.residue import EscalationResidueCandidateSource

A = LabeledChunk("n1_doc_005:0", "institutional framing of the issue")
B = LabeledChunk("n2_doc_001:0", "grassroots framing of the same issue")
C = LabeledChunk("n1_doc_008:0", "an unrelated topic entirely")


def _proposer(same_story, cos):
    # Injected scorers => model-free. escalation_candidates needs same_story wired.
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: same_story.get(frozenset({a, b}), False),
    )


def test_pairs_excludes_same_story_and_below_floor():
    cos = {frozenset({A.text, B.text}): 0.40,   # residue: not same-story, above floor
           frozenset({A.text, C.text}): 0.05,   # below floor -> dropped
           frozenset({B.text, C.text}): 0.50}   # same-story -> dropped
    same_story = {frozenset({B.text, C.text}): True}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer(same_story, cos), cos_floor=0.30,
    )
    keys = {pair_key(a.chunk_id, b.chunk_id) for a, b in src.pairs()}
    assert keys == {pair_key("n1_doc_005:0", "n2_doc_001:0")}


def test_pairs_sorted_by_cosine_desc():
    cos = {frozenset({A.text, B.text}): 0.40,
           frozenset({A.text, C.text}): 0.60,
           frozenset({B.text, C.text}): 0.50}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos), cos_floor=0.30,
    )
    ordered = [(a.chunk_id, b.chunk_id) for a, b in src.pairs()]
    assert ordered[0] == ("n1_doc_005:0", "n1_doc_008:0")  # 0.60 first


def test_chunks_returns_input():
    src = EscalationResidueCandidateSource([A, B, C], proposer=_proposer({}, {}))
    assert src.chunks() == [A, B, C]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_residue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.residue'`

- [ ] **Step 3: Write `gin/curator/residue.py`**

```python
"""Candidate source that surfaces the issue_frame residue for labeling.

Reuses cartographer.escalation.escalation_candidates (the already-measured
residue: not same-story, cosine >= floor, cosine-sorted) so what the curator
labels stays aligned with what the escalation bar tests. Implements
sub-project A's CandidateSource protocol.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.escalation import (
    DEFAULT_ESCALATION_COS_FLOOR,
    DEFAULT_MAX_CANDIDATES,
    escalation_candidates,
)
from gin.cartographer.models import LabeledChunk
from gin.cartographer.scan import wire_same_story


class EscalationResidueCandidateSource:
    """A.CandidateSource over the escalation residue of a corpus."""

    def __init__(
        self,
        chunks: list[LabeledChunk],
        *,
        proposer: Optional[CombinedRelationProposer] = None,
        cos_floor: float = DEFAULT_ESCALATION_COS_FLOOR,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._chunks = list(chunks)
        if proposer is None:
            proposer = CombinedRelationProposer()
        # escalation_candidates needs the stage-1 same-story provider; wire it
        # from this corpus unless one was injected (tests inject a fake).
        if proposer.same_story is None:
            wire_same_story(proposer, self._chunks)
        self._proposer = proposer
        self._cos_floor = cos_floor
        self._max_candidates = max_candidates

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        return escalation_candidates(
            combinations(self._chunks, 2),
            self._proposer,
            cos_floor=self._cos_floor,
            max_candidates=self._max_candidates,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_residue.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add gin/curator/residue.py tests/test_curator_residue.py
git commit -m "Curator: EscalationResidueCandidateSource reusing escalation_candidates (B0, task 3)."
```

---

## Task 4: Readiness endpoint + progress-line wiring

**Files:**
- Modify: `gin/curator/app.py`
- Test: `tests/test_curator_app.py`

**Interfaces:**
- Consumes: `readiness`, `ReadinessTarget` (Task 2).
- Produces: `create_curator_app` gains a keyword param `readiness_target: ReadinessTarget = ReadinessTarget()`; a new `GET /curator/readiness` endpoint returning `{"new_issue_frame", "new_agree", "new_unrelated", "target": {"issue_frame", "agree", "unrelated"}, "ready"}`; the served page's progress line appends the readiness summary.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_curator_app.py` (reuses the file's existing `CHUNKS`, `_fake_signals`, `_client` helpers from sub-project A's Task 6):

```python
def test_readiness_endpoint_returns_report_shape(tmp_path):
    from gin.cartographer.models import Relation
    from gin.curator.models import LabelRecord
    from gin.curator.store import Store
    from gin.curator.app import create_curator_app
    from gin.curator.candidates import OfflineCandidateSource
    from gin.curator.readiness import ReadinessTarget
    from fastapi.testclient import TestClient

    store = Store(tmp_path / "labels.jsonl")
    store.append(LabelRecord(id="1", src_chunk_id="x:0", dst_chunk_id="y:0",
                             relation=Relation.CONTRADICTS, relation_class="issue_frame",
                             rationale="", curator="t", ts="2026-07-17T00:00:00Z"))
    app = create_curator_app(store=store, source=OfflineCandidateSource(CHUNKS),
                             signals_fn=_fake_signals,
                             readiness_target=ReadinessTarget(issue_frame=1, agree=1, unrelated=1))
    r = TestClient(app).get("/curator/readiness")
    assert r.status_code == 200
    body = r.json()
    assert body["new_issue_frame"] == 1
    assert body["target"] == {"issue_frame": 1, "agree": 1, "unrelated": 1}
    assert body["ready"] is False  # agree/unrelated still 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_app.py::test_readiness_endpoint_returns_report_shape -v`
Expected: FAIL — `TypeError: create_curator_app() got an unexpected keyword argument 'readiness_target'`

- [ ] **Step 3: Add the import, the constructor param, and the endpoint**

Edit `gin/curator/app.py`. Add to the intra-package imports (next to `from .store import Store`):

```python
from .readiness import ReadinessTarget, readiness
from .store import Store
```

Change the `create_curator_app` signature to add the parameter:

```python
def create_curator_app(
    *,
    store: Store,
    source: CandidateSource,
    signals_fn: Callable[[str, str], dict],
    curator: str = "curator",
    scan_limit: int = 500,
    readiness_target: ReadinessTarget = ReadinessTarget(),
) -> FastAPI:
```

Add the endpoint immediately after the `next_pairs` function (before the `@app.post("/curator/label")` block):

```python
    @app.get("/curator/readiness")
    def readiness_report() -> dict:
        rep = readiness(store, readiness_target)
        return {
            "new_issue_frame": rep.new_issue_frame,
            "new_agree": rep.new_agree,
            "new_unrelated": rep.new_unrelated,
            "target": {
                "issue_frame": rep.target.issue_frame,
                "agree": rep.target.agree,
                "unrelated": rep.target.unrelated,
            },
            "ready": rep.ready,
        }
```

- [ ] **Step 4: Wire the readiness summary into the page progress line**

Edit `gin/curator/app.py`. In `PAGE_HTML`, replace the `loadNext` function:

```javascript
async function loadNext(){
  const d=await (await fetch("/curator/next?n=20")).json();
  queue=d.pairs;
  document.getElementById("progress").textContent=`labeled ${d.labeled} \\u00b7 remaining ${d.remaining}`;
  show();
}
```

with:

```javascript
async function loadNext(){
  const d=await (await fetch("/curator/next?n=20")).json();
  queue=d.pairs;
  let rtxt="";
  try{
    const r=await (await fetch("/curator/readiness")).json();
    const t=r.target;
    rtxt=`  |  issue_frame ${r.new_issue_frame}/${t.issue_frame} \\u00b7 agree ${r.new_agree}/${t.agree}`
      +` \\u00b7 unrelated ${r.new_unrelated}/${t.unrelated} \\u00b7 ${r.ready?"READY":"not ready"}`;
  }catch(e){}
  document.getElementById("progress").textContent=`labeled ${d.labeled} \\u00b7 remaining ${d.remaining}${rtxt}`;
  show();
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_app.py -v`
Expected: all pass (the 6 existing app tests + the new readiness test)

- [ ] **Step 6: Run the full suite to confirm zero regression**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: same pass count as before plus the new tests; no existing test changes behavior.

- [ ] **Step 7: Commit**

```bash
git add gin/curator/app.py tests/test_curator_app.py
git commit -m "Curator: GET /curator/readiness + progress-line readiness summary (B0, task 4)."
```

---

## Task 5: CLI — readiness script + launcher residue source

**Files:**
- Create: `scripts/curator_readiness.py`
- Modify: `scripts/curator_serve.py`
- Test: `tests/test_curator_readiness_cli.py`

**Interfaces:**
- Consumes: `readiness`, `ReadinessTarget` (Task 2), `Store` (A); `load_corpus_chunks` (Task 1), `EscalationResidueCandidateSource` (Task 3), `pair_signals`, `CombinedRelationProposer` (A/cartographer).
- Produces: `scripts/curator_readiness.py` with a `main()` and a `format_report(rep) -> str` helper (unit-testable without argparse); `scripts/curator_serve.py` gains `--source {labeled-set,escalation-residue}` (default `labeled-set`) and `--corpus PATH...` (default the three `corpus_node*.json`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_readiness_cli.py
"""format_report renders the readiness gauge for the CLI."""
from gin.curator.readiness import ReadinessReport, ReadinessTarget
from scripts.curator_readiness import format_report


def test_format_report_shows_counts_and_verdict():
    rep = ReadinessReport(new_issue_frame=3, new_agree=12, new_unrelated=15,
                          target=ReadinessTarget(20, 20, 20), ready=False)
    out = format_report(rep)
    assert "issue_frame 3/20" in out
    assert "agree 12/20" in out
    assert "unrelated 15/20" in out
    assert "READY: False" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_readiness_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.curator_readiness'`

- [ ] **Step 3: Write `scripts/curator_readiness.py`**

```python
"""Print sub-project B's labeling readiness without launching the server.

    venv/Scripts/python.exe scripts/curator_readiness.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gin.curator.readiness import ReadinessReport, ReadinessTarget, readiness
from gin.curator.store import Store

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def format_report(rep: ReadinessReport) -> str:
    t = rep.target
    return (
        f"issue_frame {rep.new_issue_frame}/{t.issue_frame}\n"
        f"agree       {rep.new_agree}/{t.agree}\n"
        f"unrelated   {rep.new_unrelated}/{t.unrelated}\n"
        f"READY: {rep.ready}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="curator labeling readiness for sub-project B")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--issue-frame-target", type=int, default=20)
    ap.add_argument("--agree-target", type=int, default=20)
    ap.add_argument("--unrelated-target", type=int, default=20)
    args = ap.parse_args()
    target = ReadinessTarget(args.issue_frame_target, args.agree_target, args.unrelated_target)
    print(format_report(readiness(Store(args.log), target)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_readiness_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Extend `scripts/curator_serve.py` with the residue source**

Edit `scripts/curator_serve.py`. Add imports (next to the existing curator imports):

```python
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.residue import EscalationResidueCandidateSource
```

Add the new CLI args inside `main()` (after the existing `--no-seed` argument):

```python
    ap.add_argument("--source", choices=["labeled-set", "escalation-residue"],
                    default="labeled-set", help="candidate source")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("corpus_node1.json"), Path("corpus_node2.json"),
                             Path("corpus_node3.json")],
                    help="corpus_node*.json exports for the escalation-residue source")
```

Replace the existing source construction:

```python
    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded
    source = OfflineCandidateSource(labeled_set.chunks())
```

with source selection:

```python
    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded
    if args.source == "escalation-residue":
        source = EscalationResidueCandidateSource(load_corpus_chunks(args.corpus))
        print(f"escalation-residue source over {len(source.chunks())} corpus chunks")
    else:
        source = OfflineCandidateSource(labeled_set.chunks())
```

(`OfflineCandidateSource` and `labeled_set` are already imported in the file from sub-project A's Task 7; do not duplicate those imports.)

- [ ] **Step 6: Verify the launcher imports cleanly and the readiness CLI runs on the seeded log**

Run:
```bash
PYTHONPATH=. venv/Scripts/python.exe -c "import scripts.curator_serve, scripts.curator_readiness; print('launcher + readiness import OK')"
PYTHONPATH=. venv/Scripts/python.exe scripts/curator_readiness.py
```
Expected: `launcher + readiness import OK`, then a readiness report over `data/curator/labels.jsonl` showing `issue_frame 0/20` (the store holds only the 4 bar issue_frame seeds), with `agree`/`unrelated` reflecting the disjoint labeled_set seed controls, `READY: False`.

- [ ] **Step 7: Manual smoke — residue source over the real corpus**

Run:
```bash
PYTHONPATH=. venv/Scripts/python.exe scripts/curator_serve.py --source escalation-residue --no-seed --port 8601
```
Expected: prints `escalation-residue source over N corpus chunks` (N ≈ 130+) then `curator UI: http://127.0.0.1:8601/curator/`. Open it: confirm the pair panels show real twonode/news chunks (e.g. `n1_doc_*` vs `n2_doc_*`), and the progress line shows the readiness summary (`issue_frame 0/20 · agree …/20 · unrelated …/20 · not ready`). Label one residue pair as `contradicts`/`issue_frame`, reload, confirm `issue_frame` increments to `1`. Stop with Ctrl-C. (No commit for the manual step; record the observation in the task report.)

- [ ] **Step 8: Commit**

```bash
git add scripts/curator_readiness.py scripts/curator_serve.py tests/test_curator_readiness_cli.py
git commit -m "Curator: readiness CLI + launcher --source escalation-residue (B0, task 5)."
```

---

## Task 6: Documentation

**Files:**
- Modify: `README.md`
- Modify: `architecture.md`

- [ ] **Step 1: Extend the README "Curator labeling tool" subsection**

In `README.md`, under the existing `### Curator labeling tool (framing corpus)` subsection (added in sub-project A), add a short paragraph: to grow the issue_frame training set for the bi-encoder, launch with the residue source —
```bash
venv/Scripts/python.exe scripts/curator_serve.py --source escalation-residue
```
— which surfaces the escalation residue (cross-outlet, not-same-story, cosine ≥ floor) over `corpus_node*.json` instead of the 18 labeled_set chunks; and check progress toward B with `venv/Scripts/python.exe scripts/curator_readiness.py` (per-class new-label counts vs a target of 20 each, excluding the fixed escalation-bar pairs). Note the readiness gauge trains no model — it only counts.

- [ ] **Step 2: Add a note to `architecture.md`**

In `architecture.md`, in the **Curator tier** paragraph added in sub-project A, append one sentence: the bi-encoder detector (B) is gated on labels — the 4 issue_frame pairs are the escalation eval set, so B0 (`gin/curator/residue.py` + `gin/curator/readiness.py`) makes the residue labelable and reports a no-model class-count readiness gauge (`GET /curator/readiness`, `scripts/curator_readiness.py`) that excludes the fixed bar pairs; B unblocks when the gauge is green.

- [ ] **Step 3: Commit**

```bash
git add README.md architecture.md
git commit -m "Docs: B0 residue source + readiness gauge (unblocks bi-encoder)."
```

---

## Self-Review Notes

**Spec coverage:** every spec section maps to a task — falsifiable-claim bars: residue reachability + cosine-sort (Task 3), chunk-id convention (Task 1), bar-exclusion + gauge correctness (Task 2), no-model (Tasks 2 constraint), additivity (Task 4 Step 6 full-suite gate), surfacing (Task 4 endpoint + Task 5 CLI). Scope decisions: reuse escalation_candidates (Task 3), offline corpus JSON (Task 1), id normalization (Task 1), configurable class-count readiness (Task 2), bar pairs from default_calibration_sets (Task 2), dual surfacing endpoint+script (Tasks 4/5), no-model/no-corpus-expansion (Global Constraints + out-of-scope). Data flow + launcher `--source` (Task 5). Error handling: missing file / missing position / missing doc_id (Task 1); empty residue and seeded-only store (covered by Task 2/3 behavior — `pairs()` returns `[]`, gauge returns zeros). Testing tiers 1–4 (Tasks 1–4), manual smoke (Task 5 Step 7). Docs (Task 6).

**Additive-only:** only `gin/curator/app.py`, `scripts/curator_serve.py`, `README.md`, `architecture.md` are modified; everything else is new. `gin/curator/` imports only from `gin/cartographer/` and sub-project A's own modules; nothing outside imports `gin.curator` except the two scripts. Task 4 Step 6 runs the full suite to prove zero regression.

**Type/interface consistency:** `ReadinessTarget(issue_frame, agree, unrelated)` and `ReadinessReport(new_issue_frame, new_agree, new_unrelated, target, ready)` field names are identical across Tasks 2, 4 (endpoint dict), and 5 (`format_report`). `readiness(store, target)` signature matches its callers in Task 4 (`readiness(store, readiness_target)`) and Task 5 (`readiness(Store(args.log), target)`). `EscalationResidueCandidateSource(chunks, *, proposer, cos_floor, max_candidates)` (Task 3) matches its Task 5 construction (`EscalationResidueCandidateSource(load_corpus_chunks(args.corpus))`). `load_corpus_chunks(paths)` (Task 1) returns `list[LabeledChunk]` consumed by Task 3/5. The `pair_key` used in Task 2's exclusion is A's existing `gin.curator.models.pair_key`, identical to how the store keys pairs. Endpoint JSON keys (`new_issue_frame`, `new_agree`, `new_unrelated`, `target.{issue_frame,agree,unrelated}`, `ready`) match the page's `loadNext` reads (Task 4 Step 4) and the CLI (Task 5).

**Placeholder scan:** no TBD/TODO; every code step is complete and grounded — `create_curator_app`'s current signature/body, the exact `loadNext` block, `escalation_candidates`'s signature + defaults (`0.30`/`400`), `default_calibration_sets()`'s return shape, `wire_same_story`'s signature, and the `corpus_node*.json` structure (`documents[].chunks[].position`/`text`) were all read from source before writing this plan.
