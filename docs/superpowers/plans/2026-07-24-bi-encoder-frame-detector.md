# Bi-Encoder Frame Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a learned `FrameJudge` that reproduces the curator's `issue_frame` stance from 80 labeled pairs, replacing the LLM escalation judges that failed at every capability tier.

**Architecture:** Frozen `all-MiniLM-L6-v2` embeddings feed order-invariant pair features `[|a-b|, a*b, (a+b)/2]` into a small scikit-learn head. A stage-0 linear probe gates the whole approach: if the frozen geometry has no recoverable stance axis, the plan stops there and that null result is the deliverable. Everything lives in a new `gin/frames/` package so the existing `gin.cartographer` → no-`gin.curator` layering invariant survives.

**Tech Stack:** Python 3.12, scikit-learn 1.9, numpy 2.5, sentence-transformers, PyYAML, pytest, joblib.

## Global Constraints

- **Layering:** `gin/frames/` may import `gin.curator` and `gin.cartographer`. Neither may import `gin.frames`. `gin.cartographer` must never import `gin.curator`.
- **DB-free:** no task may introduce a Postgres dependency. Bar text comes from `data/synthetic/news_corpus.yaml`, not `chunks_from_db`.
- **Model-free tests:** every test must pass without downloading a model. Inject `encode_fn` stubs. The only model-touching code is `ChunkEncoder`'s lazy default branch, which is never exercised in tests.
- **Count from the fold:** always `Store.gold()`, never raw JSONL lines. 104 lines = 102 pairs.
- **Exact expected counts** after Task 1: 80 examples — DIVERGENT 27, AGREE 17, RELATED_UNTYPED 15, UNRELATED 21; drops `{"schema": 11, "bar_chunk": 11}`, `text_unresolved` absent.
- **Probe thresholds:** pass ≥ 0.65, inconclusive 0.55–0.65, fail < 0.55 (LOO balanced accuracy, DIVERGENT-vs-rest, binary chance 0.50).
- **Decision rule:** bar all-green AND LOO 4-way balanced accuracy ≥ 0.50 → success; 0.40–0.50 → success with small-data caveat; < 0.40 → suspect, not shipped as a win.
- **Determinism:** every estimator takes an explicit `random_state`/seed. Dataset order is sorted by `pair_key` so LOO folds are reproducible.
- Run all commands from the repo root with the venv active. Prefix with `PYTHONPATH=.` if `pip install -e .` has not been run.

## File Structure

| File | Responsibility |
|------|----------------|
| `gin/frames/__init__.py` | Package marker, no logic |
| `gin/frames/backfill.py` | One-shot: tag the 7 pre-`relation_class` seed contradicts by register |
| `gin/frames/labels.py` | `FrameClass` enum, relation→class map, 4→3 judge collapse, `bar_chunk_ids()` |
| `gin/frames/dataset.py` | Text index (3 sources) + the three-filter pipeline → `DatasetReport` |
| `gin/frames/encoder.py` | `ChunkEncoder` (frozen, cached, injectable) + symmetric `pair_features` |
| `gin/frames/probe.py` | Stage-0 gate: LOO logistic regression on DIVERGENT-vs-rest |
| `gin/frames/head.py` | `Manifest`, estimator construction, train/save/load with mismatch guard |
| `gin/frames/judge.py` | `BiEncoderFrameJudge` implementing the `FrameJudge` contract |
| `gin/frames/eval.py` | Bar metrics, LOO report across seeds, baseline table, decision rule |
| `scripts/frames_backfill.py` | CLI for Task 1 |
| `scripts/frames_probe.py` | CLI for Task 5 |
| `scripts/frames_train.py` | CLI for Task 6 |
| `scripts/frames_eval.py` | CLI for Task 8 |
| `tests/test_frames_*.py` | One test module per `gin/frames/` module |

---

### Task 1: Backfill the 7 pre-`relation_class` seed labels

Seven `contradicts` rows were seeded before `relation_class` existed. Five are canonical `issue_frame` (institutional-vs-grassroots framing, housing); two are securities-fraud pairs that are propositional, not framing, and must stay excluded as `story`. The store is append-only with a latest-wins fold, so this is an append of superseding records — never an in-place edit.

**Files:**
- Create: `gin/frames/__init__.py`
- Create: `gin/frames/backfill.py`
- Create: `scripts/frames_backfill.py`
- Test: `tests/test_frames_backfill.py`

**Interfaces:**
- Consumes: `gin.curator.store.Store`, `gin.curator.models.LabelRecord`, `gin.curator.models.pair_key`, `gin.cartographer.models.Relation`
- Produces: `SEED_CLASS_BACKFILL: dict[tuple[str, str], str]`, `backfill_seed_classes(store: Store, *, curator: str = "backfill") -> int` returning the number of records appended

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_backfill.py`:

```python
"""Backfill of pre-relation_class seed contradicts, by register."""
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.store import Store
from gin.frames.backfill import SEED_CLASS_BACKFILL, backfill_seed_classes


def _rec(src, dst, relation_class=None, rid="seed-1"):
    return LabelRecord(
        id=rid, src_chunk_id=src, dst_chunk_id=dst,
        relation=Relation.CONTRADICTS, relation_class=relation_class,
        rationale="", curator="seed", ts="2026-07-17T00:00:00Z",
    )


def test_backfill_map_covers_seven_pairs_five_issue_frame():
    assert len(SEED_CLASS_BACKFILL) == 7
    assert sum(1 for v in SEED_CLASS_BACKFILL.values() if v == "issue_frame") == 5
    assert sum(1 for v in SEED_CLASS_BACKFILL.values() if v == "story") == 2


def test_backfill_map_keys_are_sorted_pair_keys():
    for key in SEED_CLASS_BACKFILL:
        assert key == pair_key(*key)


def test_appends_superseding_record_with_class(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0"))
    assert backfill_seed_classes(store) == 1
    current = store.fold_current()[pair_key("inst_em:0", "grass_em:0")]
    assert current.relation_class == "issue_frame"
    assert current.supersedes == "seed-1"
    assert current.relation is Relation.CONTRADICTS


def test_securities_pairs_tagged_story(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("disc_nw_pr:0", "disc_nw_complaint:0", rid="seed-2"))
    backfill_seed_classes(store)
    assert store.fold_current()[pair_key("disc_nw_pr:0", "disc_nw_complaint:0")].relation_class == "story"


def test_is_idempotent(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0"))
    assert backfill_seed_classes(store) == 1
    assert backfill_seed_classes(store) == 0
    assert len(store.read_log()) == 2


def test_ignores_pairs_already_classified(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0", relation_class="story"))
    assert backfill_seed_classes(store) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames'`

- [ ] **Step 3: Create the package marker**

Create `gin/frames/__init__.py`:

```python
"""Learned frame detector (sub-project B): curator labels -> FrameJudge.

Composes above gin.curator (labels) and gin.cartographer (models, eval bar).
Neither may import this package.
"""
```

- [ ] **Step 4: Write the backfill module**

Create `gin/frames/backfill.py`:

```python
"""One-shot: classify the 7 seed contradicts that predate relation_class.

Five are framing divergences over a shared issue (institutional vs grassroots,
landlord vs tenant) and are canonical issue_frame. Two are securities-fraud
pairs — propositional contradictions that NLI already types upstream — so they
are tagged `story` and stay out of the frame training set.

The store is append-only: this appends superseding records rather than editing,
so the original seed judgments remain auditable in the log.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.store import Store

# pair_key (sorted) -> relation_class
SEED_CLASS_BACKFILL: dict[tuple[str, str], str] = {
    pair_key("inst_em:0", "grass_em:0"): "issue_frame",
    pair_key("inst_wf:0", "grass_wf:0"): "issue_frame",
    pair_key("inst_wa:0", "grass_wa:0"): "issue_frame",
    pair_key("hf_af_staff:0", "hf_af_tenants:0"): "issue_frame",
    pair_key("hf_kc_inspection:0", "hf_kc_tenants:0"): "issue_frame",
    pair_key("disc_nw_pr:0", "disc_nw_complaint:0"): "story",
    pair_key("disc_mer_pr:0", "disc_mer_complaint:0"): "story",
}

RATIONALE = {
    "issue_frame": "backfill: framing divergence over a shared issue (pre-relation_class seed)",
    "story": "backfill: propositional contradiction, NLI-typed register (pre-relation_class seed)",
}


def backfill_seed_classes(store: Store, *, curator: str = "backfill") -> int:
    """Append superseding records tagging untyped seed contradicts. Idempotent."""
    current = store.fold_current()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    appended = 0
    for key, relation_class in sorted(SEED_CLASS_BACKFILL.items()):
        rec = current.get(key)
        if rec is None:
            continue
        if rec.relation is not Relation.CONTRADICTS or rec.relation_class is not None:
            continue
        store.append(
            LabelRecord(
                id=str(uuid.uuid4()),
                src_chunk_id=rec.src_chunk_id,
                dst_chunk_id=rec.dst_chunk_id,
                relation=Relation.CONTRADICTS,
                relation_class=relation_class,
                rationale=RATIONALE[relation_class],
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_backfill.py -v`
Expected: PASS — 6 passed

- [ ] **Step 6: Write the CLI**

Create `scripts/frames_backfill.py`:

```python
"""Tag the 7 pre-relation_class seed contradicts by register.

    python scripts/frames_backfill.py

Idempotent: re-running appends nothing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gin.curator.store import Store
from gin.frames.backfill import backfill_seed_classes

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill seed relation_class by register")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    store = Store(args.log)
    n = backfill_seed_classes(store)
    print(f"appended {n} superseding record(s) to {args.log}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the backfill against the real label log**

Run: `python scripts/frames_backfill.py`
Expected: `appended 7 superseding record(s) to data/curator/labels.jsonl`

Verify the fold now reports 31 issue_frame and no untyped contradicts left:

Run:
```bash
python -c "from pathlib import Path; from gin.curator.store import Store; from gin.cartographer.models import Relation; g=Store(Path('data/curator/labels.jsonl')).gold(); print('issue_frame', sum(1 for _,_,r,c in g if r is Relation.CONTRADICTS and c=='issue_frame')); print('untyped', sum(1 for _,_,r,c in g if r is Relation.CONTRADICTS and c is None))"
```
Expected: `issue_frame 31` and `untyped 0`.

Note the two different counts in play: **31** is the raw fold with no bar
exclusion; **27** is DIVERGENT after all three dataset filters (Task 3). Both
are correct — do not "fix" one to match the other.

- [ ] **Step 8: Commit**

```bash
git add gin/frames/__init__.py gin/frames/backfill.py scripts/frames_backfill.py tests/test_frames_backfill.py data/curator/labels.jsonl
git commit -m "Frames: backfill pre-relation_class seed contradicts by register"
```

---

### Task 2: Label schema and bar chunk blocklist

**Files:**
- Create: `gin/frames/labels.py`
- Test: `tests/test_frames_labels.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.Relation`, `gin.cartographer.escalation_eval.default_calibration_sets`
- Produces: `FrameClass` (str enum: `DIVERGENT`, `AGREE`, `RELATED_UNTYPED`, `UNRELATED`), `TRAINING_CLASSES: tuple[FrameClass, ...]`, `JUDGE_LABEL: dict[FrameClass, str]`, `frame_class_for(relation: Relation, relation_class: str | None) -> FrameClass | None`, `bar_chunk_ids() -> frozenset[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_labels.py`:

```python
"""4-way frame schema: relation -> training class, and the bar blocklist."""
from gin.cartographer.models import Relation
from gin.frames.labels import (
    JUDGE_LABEL,
    TRAINING_CLASSES,
    FrameClass,
    bar_chunk_ids,
    frame_class_for,
)


def test_issue_frame_contradicts_is_divergent():
    assert frame_class_for(Relation.CONTRADICTS, "issue_frame") is FrameClass.DIVERGENT


def test_story_contradicts_is_excluded():
    # NLI owns propositional conflict upstream; escalation never sees these.
    assert frame_class_for(Relation.CONTRADICTS, "story") is None


def test_untyped_contradicts_is_excluded():
    assert frame_class_for(Relation.CONTRADICTS, None) is None


def test_plain_relations_map_directly():
    assert frame_class_for(Relation.CORROBORATES, None) is FrameClass.AGREE
    assert frame_class_for(Relation.RELATED_UNTYPED, None) is FrameClass.RELATED_UNTYPED
    assert frame_class_for(Relation.UNRELATED, None) is FrameClass.UNRELATED


def test_supersedes_is_not_a_training_class():
    assert frame_class_for(Relation.SUPERSEDES, None) is None


def test_judge_collapse_covers_every_training_class():
    assert set(JUDGE_LABEL) == set(TRAINING_CLASSES)
    assert set(JUDGE_LABEL.values()) == {"DIVERGENT", "AGREE", "UNRELATED"}


def test_related_untyped_collapses_to_unrelated():
    # The 4th class sharpens the DIVERGENT boundary in training; it is never emitted.
    assert JUDGE_LABEL[FrameClass.RELATED_UNTYPED] == "UNRELATED"
    assert JUDGE_LABEL[FrameClass.DIVERGENT] == "DIVERGENT"


def test_bar_has_21_distinct_chunks():
    ids = bar_chunk_ids()
    assert len(ids) == 21
    assert "n1_doc_005:1" in ids
    assert "inflation_bureau_report:0" in ids


def test_bar_chunk_ids_is_cached():
    assert bar_chunk_ids() is bar_chunk_ids()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.labels'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/labels.py`:

```python
"""The 4-way training schema and the escalation-bar chunk blocklist.

DIVERGENT is issue_frame ONLY. story-class contradicts are excluded: NLI already
types propositional conflict upstream (combined.py, p_contra 0.899 on the legal
register), so the escalation judge never meets those pairs in production, and
mixing them dilutes the stance axis this detector exists to learn.

RELATED_UNTYPED is kept as a first-class training label because "topically
related, no typed relation" is exactly the hard negative every LLM judge
collapsed on. It is never emitted — inference folds it into UNRELATED.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.models import Relation


class FrameClass(str, Enum):
    DIVERGENT = "DIVERGENT"
    AGREE = "AGREE"
    RELATED_UNTYPED = "RELATED_UNTYPED"
    UNRELATED = "UNRELATED"


TRAINING_CLASSES: tuple[FrameClass, ...] = (
    FrameClass.DIVERGENT,
    FrameClass.AGREE,
    FrameClass.RELATED_UNTYPED,
    FrameClass.UNRELATED,
)

_LABEL_MAP: dict[tuple[Relation, Optional[str]], FrameClass] = {
    (Relation.CONTRADICTS, "issue_frame"): FrameClass.DIVERGENT,
    (Relation.CORROBORATES, None): FrameClass.AGREE,
    (Relation.RELATED_UNTYPED, None): FrameClass.RELATED_UNTYPED,
    (Relation.UNRELATED, None): FrameClass.UNRELATED,
}

# 4-way training class -> the 3-label FrameJudge contract.
JUDGE_LABEL: dict[FrameClass, str] = {
    FrameClass.DIVERGENT: "DIVERGENT",
    FrameClass.AGREE: "AGREE",
    FrameClass.RELATED_UNTYPED: "UNRELATED",
    FrameClass.UNRELATED: "UNRELATED",
}


def frame_class_for(relation: Relation, relation_class: Optional[str]) -> Optional[FrameClass]:
    """Training class for a labeled pair, or None if the pair is not trainable."""
    return _LABEL_MAP.get((relation, relation_class))


@lru_cache(maxsize=1)
def bar_chunk_ids() -> frozenset[str]:
    """Every chunk id appearing anywhere in the escalation bar.

    Chunk-level, not pair-level: labeling the residue drew from the same n1/n2
    corpus the bar was built from, so 9 bar chunks entered the label pool in
    different pairings. readiness.py filters exact bar PAIRS, so that reuse
    passed the gauge invisibly. Training on them would make a bar score
    partly unearned.
    """
    ids: set[str] = set()
    for group in default_calibration_sets().values():
        for src, dst, _register in group:
            ids.add(src)
            ids.add(dst)
    return frozenset(ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_labels.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add gin/frames/labels.py tests/test_frames_labels.py
git commit -m "Frames: 4-way label schema + chunk-level bar blocklist"
```

---

### Task 3: Dataset assembly

**Files:**
- Create: `gin/frames/dataset.py`
- Test: `tests/test_frames_dataset.py`

**Interfaces:**
- Consumes: `gin.frames.labels.{FrameClass, TRAINING_CLASSES, bar_chunk_ids, frame_class_for}`, `gin.curator.store.Store`, `gin.curator.corpus_json.load_corpus_chunks`, `gin.cartographer.labeled_set.chunks`
- Produces: `FrameExample` (frozen dataclass: `src_chunk_id`, `dst_chunk_id`, `src_text`, `dst_text`, `label: FrameClass`), `DatasetReport` (frozen dataclass: `examples: list[FrameExample]`, `drops: dict[str, int]`, property `counts: dict[str, int]`), `news_corpus_chunks(path: Path = NEWS_CORPUS) -> dict[str, str]`, `default_text_index() -> dict[str, str]`, `build_dataset(store: Store, text_index: dict[str, str] | None = None) -> DatasetReport`, constants `DEFAULT_LABELS`, `NEWS_CORPUS`, `CORPUS_NODES`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_dataset.py`:

```python
"""Dataset assembly: fold -> schema -> bar exclusion -> text resolution."""
import pytest

from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord
from gin.curator.store import Store
from gin.frames.dataset import (
    DEFAULT_LABELS,
    build_dataset,
    default_text_index,
    news_corpus_chunks,
)
from gin.frames.labels import FrameClass


def _rec(src, dst, relation, ts, relation_class=None):
    return LabelRecord(
        id=f"{src}|{dst}", src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator="t", ts=ts,
    )


def _text(*ids):
    return {i: f"text of {i}" for i in ids}


# build_dataset hard-errors on an empty class, so filter tests need a base of
# all four classes; the pair under test is added on top and its drop asserted.
_BASE = [
    ("base_a:0", "base_b:0", Relation.CONTRADICTS, "issue_frame"),
    ("base_c:0", "base_d:0", Relation.CORROBORATES, None),
    ("base_e:0", "base_f:0", Relation.RELATED_UNTYPED, None),
    ("base_g:0", "base_h:0", Relation.UNRELATED, None),
]


def _store_with_base(tmp_path, *extra):
    """Store holding one pair of every class, plus any extra rows."""
    store = Store(tmp_path / "l.jsonl")
    for i, (s, d, rel, cls) in enumerate(list(_BASE) + list(extra)):
        store.append(_rec(s, d, rel, f"2026-01-01T00:00:{i:02d}Z", relation_class=cls))
    return store


def _base_text():
    ids = [x for row in _BASE for x in row[:2]]
    return _text(*ids)


def test_news_corpus_supplies_21_chunks():
    assert len(news_corpus_chunks()) == 21


def test_default_index_resolves_every_bar_chunk():
    # Without the news YAML, 10 bar chunks resolve only via Postgres.
    from gin.frames.labels import bar_chunk_ids

    index = default_text_index()
    assert not (bar_chunk_ids() - set(index))


def test_real_label_log_yields_expected_counts():
    # Regression guard: if the label log drifts, this names the drift rather
    # than silently retraining on different data.
    report = build_dataset(Store(DEFAULT_LABELS))
    assert len(report.examples) == 80
    assert report.counts == {
        "DIVERGENT": 27, "AGREE": 17, "RELATED_UNTYPED": 15, "UNRELATED": 21,
    }
    assert report.drops == {"schema": 11, "bar_chunk": 11}


def test_bar_chunk_pair_is_dropped_and_counted(tmp_path):
    # n1_doc_005:1 is a real escalation-bar chunk reached via residue labeling.
    store = _store_with_base(
        tmp_path, ("n1_doc_005:1", "free_chunk:0", Relation.CORROBORATES, None)
    )
    text = _base_text() | _text("n1_doc_005:1", "free_chunk:0")
    report = build_dataset(store, text_index=text)
    assert report.drops["bar_chunk"] == 1
    assert len(report.examples) == 4
    assert all("n1_doc_005:1" not in (e.src_chunk_id, e.dst_chunk_id) for e in report.examples)


def test_unresolvable_text_is_dropped_and_counted(tmp_path):
    store = _store_with_base(tmp_path, ("solo:0", "ghost:0", Relation.UNRELATED, None))
    report = build_dataset(store, text_index=_base_text() | _text("solo:0"))
    assert report.drops["text_unresolved"] == 1
    assert len(report.examples) == 4


def test_story_contradicts_dropped_on_schema(tmp_path):
    store = _store_with_base(tmp_path, ("s1:0", "s2:0", Relation.CONTRADICTS, "story"))
    report = build_dataset(store, text_index=_base_text() | _text("s1:0", "s2:0"))
    assert report.drops["schema"] == 1
    assert report.counts["DIVERGENT"] == 1  # only the base issue_frame pair


def test_empty_class_is_a_hard_error(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "b:0", Relation.CORROBORATES, "2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="empty after filtering"):
        build_dataset(store, text_index=_text("a:0", "b:0"))


def test_examples_are_sorted_for_deterministic_folds(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    pairs = [("z:0", "y:0", Relation.CORROBORATES), ("a:0", "b:0", Relation.UNRELATED),
             ("m:0", "n:0", Relation.RELATED_UNTYPED), ("c:0", "d:0", Relation.CONTRADICTS)]
    for i, (s, d, rel) in enumerate(pairs):
        store.append(_rec(s, d, rel, f"2026-01-01T00:00:0{i}Z",
                          relation_class="issue_frame" if rel is Relation.CONTRADICTS else None))
    ids = [e.src_chunk_id for e in
           build_dataset(store, text_index=_text(*[x for p in pairs for x in p[:2]])).examples]
    assert ids == sorted(ids)


def test_label_and_text_are_carried_through(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    for s, d, rel, cls in [("a:0", "b:0", Relation.CONTRADICTS, "issue_frame"),
                           ("c:0", "d:0", Relation.CORROBORATES, None),
                           ("e:0", "f:0", Relation.RELATED_UNTYPED, None),
                           ("g:0", "h:0", Relation.UNRELATED, None)]:
        store.append(_rec(s, d, rel, "2026-01-01T00:00:00Z", relation_class=cls))
    report = build_dataset(store, text_index=_text(*[f"{c}:0" for c in "abcdefgh"]))
    first = report.examples[0]
    assert first.src_chunk_id == "a:0"
    assert first.src_text == "text of a:0"
    assert first.label is FrameClass.DIVERGENT
    assert report.counts["AGREE"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.dataset'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/dataset.py`:

```python
"""Training-set assembly from the curator label store.

Three ordered filters, each drop counted by reason and surfaced — never silent:

  1. schema          — relation/relation_class not in the 4-way map
  2. bar_chunk       — either endpoint appears anywhere in the escalation bar
  3. text_unresolved — no text available for an endpoint

Rows come from Store.gold(), the latest-wins FOLD of the append-only log, never
from raw JSONL lines: 104 lines currently fold to 102 unique pairs, so counting
lines double-counts relabeled pairs and trains on stale labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from gin.cartographer.labeled_set import chunks as labeled_set_chunks
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.models import pair_key
from gin.curator.store import Store

from .labels import TRAINING_CLASSES, FrameClass, bar_chunk_ids, frame_class_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "data" / "curator" / "labels.jsonl"
NEWS_CORPUS = REPO_ROOT / "data" / "synthetic" / "news_corpus.yaml"
CORPUS_NODES = tuple(REPO_ROOT / f"corpus_node{i}.json" for i in (1, 2, 3, 4))


@dataclass(frozen=True)
class FrameExample:
    src_chunk_id: str
    dst_chunk_id: str
    src_text: str
    dst_text: str
    label: FrameClass


@dataclass(frozen=True)
class DatasetReport:
    examples: list[FrameExample]
    drops: dict[str, int]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(e.label.value for e in self.examples))


def news_corpus_chunks(path: Path = NEWS_CORPUS) -> dict[str, str]:
    """Chunk texts from the synthetic news corpus.

    Ten escalation-bar chunks (inflation_*, labor_*, wage_*, export_*, school_*,
    transit_*) live here and nowhere else offline. Reading the YAML directly is
    what lets the bar be scored without Postgres.
    """
    if not path.is_file():
        raise FileNotFoundError(f"news corpus not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    index: dict[str, str] = {}
    for doc in data.get("documents", []):
        doc_id = doc["id"]
        for position, text in enumerate(doc.get("chunks", [])):
            index[f"{doc_id}:{position}"] = text
    return index


def default_text_index() -> dict[str, str]:
    """Union of the three offline text sources (236 chunks)."""
    index = {c.chunk_id: c.text for c in labeled_set_chunks()}
    for chunk in load_corpus_chunks(CORPUS_NODES):
        index[chunk.chunk_id] = chunk.text
    index.update(news_corpus_chunks())
    return index


def build_dataset(store: Store, text_index: Optional[dict[str, str]] = None) -> DatasetReport:
    """Fold the label log into trainable examples, counting every drop."""
    text = default_text_index() if text_index is None else text_index
    bar = bar_chunk_ids()
    drops: Counter[str] = Counter()
    examples: list[FrameExample] = []

    # Sorted so leave-one-out folds are reproducible run to run.
    for src, dst, relation, relation_class in sorted(
        store.gold(), key=lambda row: pair_key(row[0], row[1])
    ):
        label = frame_class_for(relation, relation_class)
        if label is None:
            drops["schema"] += 1
            continue
        if src in bar or dst in bar:
            drops["bar_chunk"] += 1
            continue
        if src not in text or dst not in text:
            drops["text_unresolved"] += 1
            continue
        examples.append(FrameExample(src, dst, text[src], text[dst], label))

    report = DatasetReport(examples, dict(drops))
    if not examples:
        raise ValueError(f"no trainable examples after filtering (drops: {report.drops})")
    empty = [c.value for c in TRAINING_CLASSES if report.counts.get(c.value, 0) == 0]
    if empty:
        raise ValueError(f"class(es) empty after filtering: {', '.join(empty)}")
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_dataset.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add gin/frames/dataset.py tests/test_frames_dataset.py
git commit -m "Frames: dataset assembly with counted drops and DB-free text index"
```

---

### Task 4: Encoder and symmetric pair features

Order invariance is the whole point of the feature design: all three blocks are symmetric in `a`/`b`, so `judge(a, b) == judge(b, a)` holds identically and `direction_flip_count = 0` is free rather than trained for. Every LLM judge in the sweep flipped on 3–7 of 14 pairs.

**Files:**
- Create: `gin/frames/encoder.py`
- Test: `tests/test_frames_encoder.py`

**Interfaces:**
- Consumes: `gin.cartographer.combined.DEFAULT_EMBED_MODEL`, `gin.frames.dataset.FrameExample`
- Produces: `ChunkEncoder(model_name: str = DEFAULT_EMBED_MODEL, encode_fn: Callable[[str], Sequence[float]] | None = None)` with `.encode(text) -> np.ndarray` and `.model_name`; `pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray`; `feature_matrix(examples: list[FrameExample], encoder: ChunkEncoder) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_encoder.py`:

```python
"""Frozen embeddings + order-invariant pair features."""
import numpy as np

from gin.frames.dataset import FrameExample
from gin.frames.encoder import ChunkEncoder, feature_matrix, pair_features
from gin.frames.labels import FrameClass


def _stub(dim=4):
    """Deterministic pseudo-embedding, no model download."""
    def encode(text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=dim)
        return vec / np.linalg.norm(vec)
    return encode


def test_pair_features_are_symmetric():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 4.0])
    assert np.allclose(pair_features(a, b), pair_features(b, a))


def test_pair_features_have_three_blocks():
    a, b = np.ones(5), np.zeros(5)
    assert pair_features(a, b).shape == (15,)


def test_pair_feature_blocks_are_abs_diff_product_mean():
    a, b = np.array([3.0, 1.0]), np.array([1.0, 5.0])
    got = pair_features(a, b)
    assert np.allclose(got[:2], [2.0, 4.0])     # |a-b|
    assert np.allclose(got[2:4], [3.0, 5.0])    # a*b
    assert np.allclose(got[4:], [2.0, 3.0])     # (a+b)/2


def test_encoder_caches_per_text():
    calls = []

    def counting(text):
        calls.append(text)
        return [1.0, 0.0]

    enc = ChunkEncoder(encode_fn=counting)
    enc.encode("same")
    enc.encode("same")
    enc.encode("other")
    assert calls == ["same", "other"]


def test_encoder_returns_float_array():
    enc = ChunkEncoder(encode_fn=lambda t: [1, 0, 0])
    vec = enc.encode("x")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64


def test_feature_matrix_shapes_and_labels():
    examples = [
        FrameExample("a:0", "b:0", "alpha", "beta", FrameClass.DIVERGENT),
        FrameExample("c:0", "d:0", "gamma", "delta", FrameClass.AGREE),
    ]
    X, y = feature_matrix(examples, ChunkEncoder(encode_fn=_stub(4)))
    assert X.shape == (2, 12)
    assert list(y) == ["DIVERGENT", "AGREE"]


def test_feature_matrix_is_row_order_invariant_per_example():
    enc = ChunkEncoder(encode_fn=_stub(4))
    forward = FrameExample("a:0", "b:0", "alpha", "beta", FrameClass.AGREE)
    reversed_ = FrameExample("b:0", "a:0", "beta", "alpha", FrameClass.AGREE)
    Xf, _ = feature_matrix([forward], enc)
    Xr, _ = feature_matrix([reversed_], enc)
    assert np.allclose(Xf, Xr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_encoder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.encoder'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/encoder.py`:

```python
"""Frozen sentence embeddings and the order-invariant pair representation.

The encoder is never fine-tuned, so embeddings stay precomputable and shared
with the cheap pipeline (same model as combined.py).

pair_features is symmetric in a/b by construction. That is a design commitment,
not an implementation detail: it makes judge(a,b) == judge(b,a) an identity, so
direction_flip_count = 0 without training for it. Every model in the 2026-07-13
sweep flipped on 3-7 of 14 pairs.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from gin.cartographer.combined import DEFAULT_EMBED_MODEL

from .dataset import FrameExample


class ChunkEncoder:
    """Lazily-loaded frozen encoder with a per-text cache.

    Pass ``encode_fn`` to run model-free (tests, CI).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        encode_fn: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> None:
        self.model_name = model_name
        self._encode_fn = encode_fn
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if self._encode_fn is not None:
            vec = np.asarray(self._encode_fn(text), dtype=np.float64)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            vec = np.asarray(
                self._model.encode([text], normalize_embeddings=True)[0],
                dtype=np.float64,
            )
        self._cache[text] = vec
        return vec


def pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric pair representation: [|a-b|, a*b, (a+b)/2]."""
    return np.concatenate([np.abs(a - b), a * b, (a + b) / 2.0])


def feature_matrix(
    examples: list[FrameExample], encoder: ChunkEncoder
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for the given examples; y holds FrameClass *values* as strings."""
    X = np.vstack(
        [
            pair_features(encoder.encode(e.src_text), encoder.encode(e.dst_text))
            for e in examples
        ]
    )
    y = np.array([e.label.value for e in examples])
    return X, y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_encoder.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add gin/frames/encoder.py tests/test_frames_encoder.py
git commit -m "Frames: frozen encoder + symmetric pair features"
```

---

### Task 5: Stage-0 probe gate

This task can end the sub-project. If the frozen geometry has no recoverable stance axis, that is a measured finding and the deliverable — not a reason to add capacity. `combined.py`'s cosine bands for divergent (0.134–0.552) and corroborate (0.490–0.727) overlap badly, so the failure branch is live.

**Files:**
- Create: `gin/frames/probe.py`
- Create: `scripts/frames_probe.py`
- Test: `tests/test_frames_probe.py`

**Interfaces:**
- Consumes: `gin.frames.labels.FrameClass`
- Produces: constants `PROBE_PASS = 0.65`, `PROBE_FLOOR = 0.55`; `ProbeResult` (frozen dataclass: `balanced_accuracy`, `baseline`, `n`, `n_positive`, `verdict`, property `passed`); `divergent_vs_rest(y: np.ndarray) -> np.ndarray`; `run_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> ProbeResult`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_probe.py`:

```python
"""Stage-0 gate: is DIVERGENT linearly recoverable from frozen embeddings?"""
import numpy as np

from gin.frames.probe import (
    PROBE_FLOOR,
    PROBE_PASS,
    divergent_vs_rest,
    run_probe,
)


def test_divergent_vs_rest_is_binary():
    y = np.array(["DIVERGENT", "AGREE", "UNRELATED", "RELATED_UNTYPED", "DIVERGENT"])
    assert list(divergent_vs_rest(y)) == [1, 0, 0, 0, 1]


def _separable(n=40, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    y = np.where(np.arange(n) % 2 == 0, "DIVERGENT", "AGREE")
    X[y == "DIVERGENT"] += 6.0  # plainly separable
    return X, y


def test_separable_data_passes():
    result = run_probe(*_separable())
    assert result.balanced_accuracy >= PROBE_PASS
    assert result.verdict == "pass"
    assert result.passed is True


def test_noise_does_not_pass():
    # Pure noise sits near 0.50 but bounces; the load-bearing claim is that it
    # never clears the gate, not that it lands in a specific band.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 6))
    y = np.where(np.arange(60) % 2 == 0, "DIVERGENT", "AGREE")
    result = run_probe(X, y)
    assert result.balanced_accuracy < PROBE_PASS
    assert result.verdict != "pass"
    assert result.passed is False


def test_verdict_bands_are_contiguous():
    assert PROBE_FLOOR < PROBE_PASS


def test_reports_counts_and_baseline():
    X, y = _separable()
    result = run_probe(X, y)
    assert result.n == 40
    assert result.n_positive == 20
    assert 0.3 <= result.baseline <= 0.7  # stratified-random sits near chance


def test_is_deterministic_for_a_seed():
    X, y = _separable()
    assert run_probe(X, y, seed=0).balanced_accuracy == run_probe(X, y, seed=0).balanced_accuracy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.probe'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/probe.py`:

```python
"""Stage-0 gate: linear separability of the stance axis in frozen embeddings.

Doubles as the linear baseline. If a logistic regression already clears the bar,
that IS the shipped model and no MLP is built.

Thresholds are fixed before the number is seen so the gate cannot be
renegotiated after the fact. DIVERGENT-vs-rest is binary, so chance balanced
accuracy is 0.50.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut

from .labels import FrameClass

PROBE_PASS = 0.65
PROBE_FLOOR = 0.55
_BASELINE_TRIALS = 200


@dataclass(frozen=True)
class ProbeResult:
    balanced_accuracy: float
    baseline: float
    n: int
    n_positive: int
    verdict: str  # "pass" | "inconclusive" | "fail"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def divergent_vs_rest(y: np.ndarray) -> np.ndarray:
    """Collapse the 4-way label vector to the binary stance axis."""
    return (y == FrameClass.DIVERGENT.value).astype(int)


def run_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> ProbeResult:
    """Leave-one-out logistic regression on DIVERGENT-vs-rest."""
    target = divergent_vs_rest(y)
    predictions = np.empty_like(target)
    for train_idx, test_idx in LeaveOneOut().split(X):
        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        )
        clf.fit(X[train_idx], target[train_idx])
        predictions[test_idx] = clf.predict(X[test_idx])

    accuracy = float(balanced_accuracy_score(target, predictions))

    rng = np.random.default_rng(seed)
    baseline = float(
        np.mean(
            [
                balanced_accuracy_score(target, rng.permutation(target))
                for _ in range(_BASELINE_TRIALS)
            ]
        )
    )

    if accuracy >= PROBE_PASS:
        verdict = "pass"
    elif accuracy < PROBE_FLOOR:
        verdict = "fail"
    else:
        verdict = "inconclusive"

    return ProbeResult(accuracy, baseline, len(target), int(target.sum()), verdict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_probe.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Write the CLI**

Create `scripts/frames_probe.py`:

```python
"""Stage-0 gate: can a linear model recover the stance axis?

    python scripts/frames_probe.py

Loads the real encoder (downloads on first run). Exit code 0 on pass or
inconclusive, 1 on fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder, feature_matrix
from gin.frames.probe import PROBE_FLOOR, PROBE_PASS, run_probe


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen-geometry separability probe")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = build_dataset(Store(args.log))
    print(f"dataset: {len(report.examples)} examples {report.counts}")
    print(f"drops:   {report.drops}")

    X, y = feature_matrix(report.examples, ChunkEncoder())
    result = run_probe(X, y, seed=args.seed)

    print(f"\nDIVERGENT-vs-rest, leave-one-out over n={result.n} ({result.n_positive} positive)")
    print(f"  balanced accuracy : {result.balanced_accuracy:.3f}")
    print(f"  stratified random : {result.baseline:.3f}")
    print(f"  bands             : fail < {PROBE_FLOOR} <= inconclusive < {PROBE_PASS} <= pass")
    print(f"  VERDICT           : {result.verdict.upper()}")
    if result.verdict == "fail":
        print("\nThe frozen geometry has no recoverable stance axis. Do not add")
        print("capacity to rescue this — escalate to encoder fine-tuning under a")
        print("separate spec. This null result is the deliverable.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the probe for real and record the number**

Run: `python scripts/frames_probe.py`
Expected: dataset line reads `80 examples {'DIVERGENT': 27, 'AGREE': 17, 'RELATED_UNTYPED': 15, 'UNRELATED': 21}`, followed by a verdict. **Record the balanced accuracy — it decides whether Tasks 6–8 proceed as planned.** If the verdict is FAIL, stop and report rather than continuing.

- [ ] **Step 7: Commit**

```bash
git add gin/frames/probe.py scripts/frames_probe.py tests/test_frames_probe.py
git commit -m "Frames: stage-0 linear separability probe gate"
```

---

### Task 6: Head training, artifact, and manifest

**Files:**
- Create: `gin/frames/head.py`
- Create: `scripts/frames_train.py`
- Test: `tests/test_frames_head.py`

**Interfaces:**
- Consumes: `gin.frames.labels.TRAINING_CLASSES`
- Produces: `HEAD_KINDS: tuple[str, ...]`, `Manifest` (frozen dataclass with `to_json()`/`from_json()`), `build_estimator(kind: str, seed: int)`, `train_head(X, y, kind="linear", seed=0)`, `save_head(directory: Path, model, manifest: Manifest) -> None`, `load_head(directory: Path, *, expect_encoder: str | None = None, expect_dim: int | None = None) -> tuple[object, Manifest]`, `HEAD_FILENAME`, `MANIFEST_FILENAME`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_head.py`:

```python
"""Head training, persistence, and the manifest compatibility gate."""
import numpy as np
import pytest

from gin.frames.head import (
    HEAD_KINDS,
    Manifest,
    build_estimator,
    load_head,
    save_head,
    train_head,
)


def _data(n=40, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    labels = np.array(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"] * (n // 4))
    for offset, name in enumerate(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"]):
        X[labels == name] += offset * 5.0
    return X, labels


def _manifest(dim=6, kind="linear"):
    return Manifest(
        encoder_model="stub-encoder", feature_dim=dim,
        classes=["AGREE", "DIVERGENT", "RELATED_UNTYPED", "UNRELATED"],
        kind=kind, seed=0, n_train=40,
        class_counts={"DIVERGENT": 10, "AGREE": 10, "RELATED_UNTYPED": 10, "UNRELATED": 10},
        git_sha="abc1234", created_utc="2026-07-24T00:00:00Z",
    )


def test_both_head_kinds_are_supported():
    assert HEAD_KINDS == ("linear", "mlp")
    for kind in HEAD_KINDS:
        assert build_estimator(kind, seed=0) is not None


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown head kind"):
        build_estimator("transformer", seed=0)


def test_trains_and_predicts_all_four_classes():
    X, y = _data()
    model = train_head(X, y, kind="linear", seed=0)
    assert set(model.classes_) == {"DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"}


def test_round_trip_reproduces_identical_predictions(tmp_path):
    X, y = _data()
    model = train_head(X, y, kind="linear", seed=0)
    before = model.predict(X)
    save_head(tmp_path, model, _manifest())
    loaded, manifest = load_head(tmp_path)
    assert list(loaded.predict(X)) == list(before)
    assert manifest.encoder_model == "stub-encoder"


def test_manifest_json_round_trip():
    m = _manifest()
    assert Manifest.from_json(m.to_json()) == m


def test_encoder_mismatch_is_a_hard_error(tmp_path):
    X, y = _data()
    save_head(tmp_path, train_head(X, y, seed=0), _manifest())
    with pytest.raises(ValueError, match="encoder mismatch"):
        load_head(tmp_path, expect_encoder="different-encoder")


def test_feature_dim_mismatch_is_a_hard_error(tmp_path):
    X, y = _data()
    save_head(tmp_path, train_head(X, y, seed=0), _manifest())
    with pytest.raises(ValueError, match="feature dim mismatch"):
        load_head(tmp_path, expect_dim=999)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_head(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_head.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.head'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/head.py`:

```python
"""The pair-head: a small scikit-learn estimator plus a gating manifest.

Deliberately not a hand-written torch module. At 80 rows a training loop is pure
surface area for bugs; sklearn gives deterministic fits, LeaveOneOut, and
class_weight="balanced" for free.

Capacity is a liability here, so "linear" is the default and "mlp" exists only
for the case where linear provably underfits.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

HEAD_KINDS: tuple[str, ...] = ("linear", "mlp")
HEAD_FILENAME = "head.joblib"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class Manifest:
    encoder_model: str
    feature_dim: int
    classes: list[str]
    kind: str
    seed: int
    n_train: int
    class_counts: dict[str, int]
    git_sha: str
    created_utc: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Manifest":
        return cls(**data)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_estimator(kind: str, seed: int):
    if kind == "linear":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,), alpha=1.0, max_iter=5000,
            early_stopping=False, random_state=seed,
        )
    raise ValueError(f"unknown head kind: {kind!r} (expected one of {HEAD_KINDS})")


def train_head(X: np.ndarray, y: np.ndarray, kind: str = "linear", seed: int = 0):
    model = build_estimator(kind, seed)
    model.fit(X, y)
    return model


def save_head(directory: Path, model, manifest: Manifest) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / HEAD_FILENAME)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_json(), indent=2) + "\n", encoding="utf-8"
    )


def load_head(
    directory: Path,
    *,
    expect_encoder: Optional[str] = None,
    expect_dim: Optional[int] = None,
) -> tuple[object, Manifest]:
    """Load head + manifest. Incompatibility is a hard error, never a fallback."""
    directory = Path(directory)
    head_path = directory / HEAD_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    if not head_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"no trained head in {directory}")
    manifest = Manifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
    if expect_encoder is not None and manifest.encoder_model != expect_encoder:
        raise ValueError(
            f"encoder mismatch: head trained on {manifest.encoder_model!r}, "
            f"caller supplied {expect_encoder!r}"
        )
    if expect_dim is not None and manifest.feature_dim != expect_dim:
        raise ValueError(
            f"feature dim mismatch: head expects {manifest.feature_dim}, got {expect_dim}"
        )
    return joblib.load(head_path), manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_head.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Write the CLI**

Create `scripts/frames_train.py`:

```python
"""Train the pair-head and write head.joblib + manifest.json.

    python scripts/frames_train.py --kind linear
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder, feature_matrix
from gin.frames.head import HEAD_KINDS, Manifest, git_sha, save_head, train_head

DEFAULT_OUT = Path("data/frames")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the bi-encoder pair-head")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kind", choices=HEAD_KINDS, default="linear")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = build_dataset(Store(args.log))
    encoder = ChunkEncoder()
    X, y = feature_matrix(report.examples, encoder)
    model = train_head(X, y, kind=args.kind, seed=args.seed)

    manifest = Manifest(
        encoder_model=encoder.model_name,
        feature_dim=int(X.shape[1]),
        classes=sorted(set(y.tolist())),
        kind=args.kind,
        seed=args.seed,
        n_train=len(report.examples),
        class_counts=report.counts,
        git_sha=git_sha(),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    save_head(args.out, model, manifest)
    print(f"trained {args.kind} head on {manifest.n_train} rows {manifest.class_counts}")
    print(f"wrote {args.out}/head.joblib + manifest.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Gitignore the model artifact, keep the manifest**

Append to `.gitignore`:

```
data/frames/*.joblib
```

- [ ] **Step 7: Train for real**

Run: `python scripts/frames_train.py --kind linear`
Expected: `trained linear head on 80 rows {'DIVERGENT': 27, 'AGREE': 17, 'RELATED_UNTYPED': 15, 'UNRELATED': 21}`

- [ ] **Step 8: Commit**

```bash
git add gin/frames/head.py scripts/frames_train.py tests/test_frames_head.py .gitignore data/frames/manifest.json
git commit -m "Frames: pair-head training with manifest-gated artifacts"
```

---

### Task 7: The FrameJudge adapter

**Files:**
- Create: `gin/frames/judge.py`
- Test: `tests/test_frames_judge.py`

**Interfaces:**
- Consumes: `gin.frames.encoder.{ChunkEncoder, pair_features}`, `gin.frames.labels.{FrameClass, JUDGE_LABEL}`, `gin.frames.head.load_head`
- Produces: `BiEncoderFrameJudge(model, encoder: ChunkEncoder)` — callable `(a_text: str, b_text: str) -> str` returning one of `DIVERGENT`/`AGREE`/`UNRELATED`; `load_judge(directory: Path, encoder: ChunkEncoder | None = None) -> BiEncoderFrameJudge`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_judge.py`:

```python
"""BiEncoderFrameJudge: FrameJudge contract + structural order invariance."""
import numpy as np

from gin.frames.encoder import ChunkEncoder
from gin.frames.judge import BiEncoderFrameJudge


class _StubModel:
    """Predicts by a deterministic function of the feature vector."""

    def __init__(self, label):
        self._label = label

    def predict(self, X):
        return np.array([self._label] * X.shape[0])


def _stub_encoder(dim=4):
    def encode(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=dim)
        return vec / np.linalg.norm(vec)
    return ChunkEncoder(encode_fn=encode)


def test_emits_the_three_contract_labels():
    for internal, expected in [
        ("DIVERGENT", "DIVERGENT"),
        ("AGREE", "AGREE"),
        ("UNRELATED", "UNRELATED"),
        ("RELATED_UNTYPED", "UNRELATED"),
    ]:
        judge = BiEncoderFrameJudge(_StubModel(internal), _stub_encoder())
        assert judge("alpha text", "beta text") == expected


def test_related_untyped_is_never_emitted():
    judge = BiEncoderFrameJudge(_StubModel("RELATED_UNTYPED"), _stub_encoder())
    assert judge("a", "b") not in {"RELATED_UNTYPED"}


def test_order_invariance_is_structural():
    # Not a trained property: the pair features are symmetric, so this is an
    # identity. direction_flip_count = 0 follows by construction.
    class _Echo:
        def predict(self, X):
            return np.array(["DIVERGENT" if X[0].sum() > 0 else "AGREE"])

    judge = BiEncoderFrameJudge(_Echo(), _stub_encoder())
    for a, b in [("one", "two"), ("longer text here", "x"), ("same", "same")]:
        assert judge(a, b) == judge(b, a)


def test_is_callable_as_a_frame_judge():
    from gin.cartographer.frame_judge import FrameJudge  # noqa: F401  (protocol alias)

    judge = BiEncoderFrameJudge(_StubModel("AGREE"), _stub_encoder())
    assert callable(judge)
    assert isinstance(judge("a", "b"), str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.judge'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/judge.py`:

```python
"""Drop-in FrameJudge backed by the trained pair-head.

Satisfies the same (a_text, b_text) -> {DIVERGENT, AGREE, UNRELATED} contract
the LLM judges used, so evaluate_escalation_judge scores it unchanged and the
comparison against the 2026-07-13 sweep is apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .encoder import ChunkEncoder, pair_features
from .head import load_head
from .labels import JUDGE_LABEL, FrameClass


class BiEncoderFrameJudge:
    """Frozen embeddings -> symmetric pair features -> head -> 3-way label."""

    def __init__(self, model, encoder: ChunkEncoder) -> None:
        self.model = model
        self.encoder = encoder

    def __call__(self, a_text: str, b_text: str) -> str:
        features = pair_features(
            self.encoder.encode(a_text), self.encoder.encode(b_text)
        ).reshape(1, -1)
        predicted = str(self.model.predict(features)[0])
        return JUDGE_LABEL[FrameClass(predicted)]


def load_judge(directory: Path, encoder: Optional[ChunkEncoder] = None) -> BiEncoderFrameJudge:
    """Load a trained head, verifying it matches the encoder it was trained on."""
    encoder = ChunkEncoder() if encoder is None else encoder
    model, _manifest = load_head(directory, expect_encoder=encoder.model_name)
    return BiEncoderFrameJudge(model, encoder)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_judge.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add gin/frames/judge.py tests/test_frames_judge.py
git commit -m "Frames: BiEncoderFrameJudge with structural order invariance"
```

---

### Task 8: Evaluation — bar, cross-validation, baselines, decision

**Files:**
- Create: `gin/frames/eval.py`
- Create: `scripts/frames_eval.py`
- Test: `tests/test_frames_eval.py`

**Interfaces:**
- Consumes: `gin.cartographer.escalation_eval.{default_calibration_sets, evaluate_escalation_judge}`, `gin.frames.dataset.default_text_index`, `gin.frames.head.{build_estimator, train_head}`, `gin.frames.judge.BiEncoderFrameJudge`
- Produces: `BAR_METRIC_KEYS`, `PUBLISHED_BASELINES`, `bar_metrics(judge, text_index=None, both_directions=True) -> dict`, `bar_all_green(metrics: dict) -> bool`, `loo_report(X, y, kind="linear", seeds=(0,1,2,3,4)) -> dict`, `decide(bar: dict, loo_mean: float) -> str` returning `"success"`/`"success_caveated"`/`"suspect"`/`"bar_failed"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frames_eval.py`:

```python
"""Bar scoring, LOO across seeds, and the pre-registered decision rule."""
import numpy as np
import pytest

from gin.frames.eval import (
    BAR_METRIC_KEYS,
    PUBLISHED_BASELINES,
    bar_all_green,
    bar_metrics,
    decide,
    loo_report,
)


def _green():
    return {"issue_frame_recall": 1.0, "class_c_discrimination": 1.0,
            "unrelated_discrimination": 1.0, "direction_flip_count": 0}


def test_bar_metric_keys_match_the_spec_bar():
    assert BAR_METRIC_KEYS == (
        "issue_frame_recall", "class_c_discrimination",
        "unrelated_discrimination", "direction_flip_count",
    )


def test_all_green_requires_every_metric():
    assert bar_all_green(_green()) is True
    for key, bad in [("issue_frame_recall", 0.75), ("class_c_discrimination", 0.9),
                     ("unrelated_discrimination", 0.5), ("direction_flip_count", 1)]:
        metrics = _green() | {key: bad}
        assert bar_all_green(metrics) is False


def test_none_metric_is_not_green():
    assert bar_all_green(_green() | {"issue_frame_recall": None}) is False


def test_published_baselines_include_the_failed_judges():
    names = {row["model"] for row in PUBLISHED_BASELINES}
    assert "Qwen2.5-14B dense" in names
    assert "Opus 4.8" in names
    opus = next(r for r in PUBLISHED_BASELINES if r["model"] == "Opus 4.8")
    assert opus["issue_frame_recall"] == 0.00


def test_decision_rule_bands():
    green = _green()
    assert decide(green, 0.62) == "success"
    assert decide(green, 0.50) == "success"
    assert decide(green, 0.45) == "success_caveated"
    assert decide(green, 0.30) == "suspect"
    assert decide(_green() | {"direction_flip_count": 2}, 0.9) == "bar_failed"


def test_loo_report_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    labels = np.array(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"] * 10)
    for offset, name in enumerate(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"]):
        X[labels == name] += offset * 5.0
    report = loo_report(X, labels, kind="linear", seeds=(0, 1))
    assert set(report) == {"balanced_accuracy_mean", "balanced_accuracy_spread",
                           "per_seed", "per_class_recall", "n"}
    assert report["n"] == 40
    assert len(report["per_seed"]) == 2
    assert report["balanced_accuracy_mean"] > 0.9  # separable by construction


def test_bar_metrics_runs_db_free_with_a_stub_judge():
    # Proves the bar is scorable without Postgres.
    metrics = bar_metrics(lambda a, b: "DIVERGENT")
    assert metrics["issue_frame_recall"] == 1.0        # constant judge catches all gold
    assert metrics["class_c_discrimination"] == 0.0    # and fails every control
    assert metrics["issue_frame_scorable_count"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frames_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.frames.eval'`

- [ ] **Step 3: Write the implementation**

Create `gin/frames/eval.py`:

```python
"""Measurement: the pre-registered bar, honest cross-validation, baselines.

The bar stays the headline gate so the comparison with the 2026-07-13 judge
sweep is apples-to-apples. But it is 14 pairs, 4 of them issue_frame, so a pass
can be luck — hence LOO alongside, and a decision rule fixed BEFORE the numbers
are seen. Precedent: calibration.leave_one_out reported 0.69 against 0.875
in-sample, and the honest number was the valuable one.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneOut

from gin.cartographer.escalation_eval import (
    default_calibration_sets,
    evaluate_escalation_judge,
)

from .dataset import default_text_index
from .head import train_head
from .labels import TRAINING_CLASSES

BAR_METRIC_KEYS: tuple[str, ...] = (
    "issue_frame_recall",
    "class_c_discrimination",
    "unrelated_discrimination",
    "direction_flip_count",
)

# Measured 2026-07-13 (data/eval_runs/). Reported alongside every result.
PUBLISHED_BASELINES: tuple[dict, ...] = (
    {"model": "Mistral-7B dense", "issue_frame_recall": 0.50,
     "class_c_discrimination": 0.67, "unrelated_discrimination": 0.25,
     "direction_flip_count": 7},
    {"model": "Qwen3.6-14B-A3B MoE", "issue_frame_recall": 0.25,
     "class_c_discrimination": 0.50, "unrelated_discrimination": 0.50,
     "direction_flip_count": 7},
    {"model": "Qwen2.5-14B dense", "issue_frame_recall": 0.50,
     "class_c_discrimination": 0.33, "unrelated_discrimination": 1.00,
     "direction_flip_count": 3},
    {"model": "Opus 4.8", "issue_frame_recall": 0.00,
     "class_c_discrimination": 0.67, "unrelated_discrimination": 1.00,
     "direction_flip_count": 3},
)

LOO_SUCCESS = 0.50
LOO_SUSPECT = 0.40


def bar_metrics(
    judge: Callable[[str, str], str],
    text_index: Optional[dict[str, str]] = None,
    both_directions: bool = True,
) -> dict:
    """Score a judge on the fixed escalation bar, without touching Postgres."""
    text = default_text_index() if text_index is None else text_index
    sets = default_calibration_sets()
    return evaluate_escalation_judge(
        judge,
        text,
        issue_frame_pairs=sets["issue_frame"],
        corroboration_pairs=sets["corroboration"],
        unrelated_pairs=sets["unrelated"],
        labeled_pairs=None,
        both_directions=both_directions,
    )


def bar_all_green(metrics: dict) -> bool:
    """1.0 on all three discrimination metrics and zero direction flips."""
    for key in ("issue_frame_recall", "class_c_discrimination", "unrelated_discrimination"):
        value = metrics.get(key)
        if value is None or value < 1.0:
            return False
    return metrics.get("direction_flip_count") == 0


def loo_report(
    X: np.ndarray,
    y: np.ndarray,
    kind: str = "linear",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Leave-one-out 4-way balanced accuracy, averaged across seeds.

    A single-seed number at this sample size is not trustworthy, so spread is
    reported and a result quoted from one seed is treated as unreported.
    """
    per_seed: list[float] = []
    last_predictions = None
    for seed in seeds:
        # Collected as a list, not np.empty_like(y): y is a fixed-width unicode
        # array, so assigning into it silently truncates "RELATED_UNTYPED"
        # whenever the held-out split happens to lack that class.
        held_out: list[str] = []
        for train_idx, test_idx in LeaveOneOut().split(X):
            model = train_head(X[train_idx], y[train_idx], kind=kind, seed=seed)
            held_out.append(str(model.predict(X[test_idx])[0]))
        predictions = np.array(held_out)
        per_seed.append(float(balanced_accuracy_score(y, predictions)))
        last_predictions = predictions

    class_names = [c.value for c in TRAINING_CLASSES]
    recalls = recall_score(
        y, last_predictions, labels=class_names, average=None, zero_division=0
    )
    return {
        "n": int(len(y)),
        "per_seed": per_seed,
        "balanced_accuracy_mean": float(np.mean(per_seed)),
        "balanced_accuracy_spread": float(np.max(per_seed) - np.min(per_seed)),
        "per_class_recall": {n: float(r) for n, r in zip(class_names, recalls)},
    }


def decide(bar: dict, loo_mean: float) -> str:
    """The rule, fixed in advance so it cannot be renegotiated after the fact."""
    if not bar_all_green(bar):
        return "bar_failed"
    if loo_mean >= LOO_SUCCESS:
        return "success"
    if loo_mean >= LOO_SUSPECT:
        return "success_caveated"
    return "suspect"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frames_eval.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Write the CLI**

Create `scripts/frames_eval.py`:

```python
"""Score the trained head: bar + leave-one-out + baseline table.

    python scripts/frames_eval.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gin.curator.store import Store
from gin.frames.dataset import DEFAULT_LABELS, build_dataset
from gin.frames.encoder import ChunkEncoder, feature_matrix
from gin.frames.eval import (
    BAR_METRIC_KEYS,
    PUBLISHED_BASELINES,
    bar_all_green,
    bar_metrics,
    decide,
    loo_report,
)
from gin.frames.judge import load_judge

DEFAULT_HEAD = Path("data/frames")
DEFAULT_OUT = Path("data/eval_runs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the bi-encoder frame detector")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    ap.add_argument("--kind", default="linear")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    encoder = ChunkEncoder()
    report = build_dataset(Store(args.log))
    X, y = feature_matrix(report.examples, encoder)

    judge = load_judge(args.head, encoder=encoder)
    bar = bar_metrics(judge)
    loo = loo_report(X, y, kind=args.kind)
    verdict = decide(bar, loo["balanced_accuracy_mean"])

    print("=== escalation bar ===")
    for key in BAR_METRIC_KEYS:
        print(f"  {key:28s} {bar.get(key)}")
    print(f"  ALL GREEN: {bar_all_green(bar)}")

    print("\n=== leave-one-out (honest generalization) ===")
    print(f"  n                      {loo['n']}")
    print(f"  balanced acc (mean)    {loo['balanced_accuracy_mean']:.3f}")
    print(f"  spread across seeds    {loo['balanced_accuracy_spread']:.3f}")
    for name, value in loo["per_class_recall"].items():
        print(f"  recall {name:18s} {value:.3f}")

    print("\n=== baselines (2026-07-13 sweep) ===")
    for row in PUBLISHED_BASELINES:
        print(f"  {row['model']:22s} recall {row['issue_frame_recall']:.2f}  "
              f"class_c {row['class_c_discrimination']:.2f}  "
              f"unrel {row['unrelated_discrimination']:.2f}  "
              f"flips {row['direction_flip_count']}")

    print(f"\nVERDICT: {verdict}")
    if verdict == "suspect":
        print("Bar is green but cross-validation is at chance. Report as overfit")
        print("or lucky — do NOT ship this as a win.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bar": {k: bar.get(k) for k in BAR_METRIC_KEYS},
        "bar_all_green": bar_all_green(bar),
        "loo": loo,
        "baselines": list(PUBLISHED_BASELINES),
        "verdict": verdict,
        "dataset_counts": report.counts,
        "dataset_drops": report.drops,
    }
    (run_dir / "frame_detector_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {run_dir}/frame_detector_metrics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full evaluation**

Run: `python scripts/frames_eval.py`
Expected: a bar block, a LOO block, the baseline table, and a verdict of `success`, `success_caveated`, `suspect`, or `bar_failed`. **Record all four bar metrics and the LOO mean.** Whatever the verdict, report it as measured — a `suspect` or `bar_failed` outcome is a real result, not a failure to be fixed by retuning.

- [ ] **Step 7: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: all prior tests still pass, plus the new `tests/test_frames_*.py` modules. No test may require Postgres or a model download.

- [ ] **Step 8: Verify the layering invariant holds**

Run:
```bash
python -c "import subprocess,sys; out=subprocess.run(['git','grep','-n','from gin.frames\|import gin.frames','--','gin/cartographer','gin/curator'],capture_output=True,text=True).stdout; print(out or 'clean: no gin.frames imports in cartographer/curator'); sys.exit(1 if out else 0)"
```
Expected: `clean: no gin.frames imports in cartographer/curator`

Also confirm cartographer still does not import curator:

Run:
```bash
python -c "import subprocess,sys; out=subprocess.run(['git','grep','-n','gin.curator','--','gin/cartographer'],capture_output=True,text=True).stdout; print(out or 'clean: cartographer does not import curator'); sys.exit(1 if out else 0)"
```
Expected: `clean: cartographer does not import curator`

- [ ] **Step 9: Commit**

```bash
git add gin/frames/eval.py scripts/frames_eval.py tests/test_frames_eval.py
git commit -m "Frames: bar + LOO + baseline evaluation with pre-registered decision rule"
```

---

## Post-Implementation

Update `docs/architecture.md` with the `gin/frames/` layer and record the measured result (bar metrics, LOO mean and spread, verdict) in the spec's own results section. If the probe failed at Task 5, the writeup is the deliverable and Tasks 6–8 are not attempted — say so plainly rather than reaching for a bigger model.
