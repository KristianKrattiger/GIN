# Two-Curator Labeling Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a second labeler (a friend) run their own instance of the curator UI against a distinct slice of the corpus, with their labels correctly attributed, and give both of you a way to compute inter-rater agreement on an overlap set and a written guide so labels are consistent.

**Architecture:** No changes to `gin/curator/` internals. The label store (`Store`) already supports multi-curator, multi-session history via its append-only, latest-wins-by-`(ts, idx)` fold (`gin/curator/store.py:41-49`) and the `curator` field on every `LabelRecord`. The only real gap is that `scripts/curator_serve.py` currently hardcodes `curator="kristian"` and has no CLI flag for it (`scripts/curator_serve.py:65`) — that's a one-line-of-behavior fix. Topic-based assignment is already free: `--corpus` already accepts a list of `corpus_node*.json` files (`scripts/curator_serve.py:37-39`), so assigning by node file *is* the topic split, no new filtering code needed. The remaining work is a small standalone merge/agreement-check script plus two docs (labeling guide, assignment/setup instructions).

**Tech Stack:** Python 3.12, pytest, existing `gin.curator.store.Store` / `gin.curator.models.LabelRecord`.

## Global Constraints

- No changes to any file under `gin/curator/` (per spec's out-of-scope section) — all new logic lives in `scripts/`.
- Every new script function must be independently testable without a running FastAPI app (follow the existing pattern in `scripts/curator_readiness.py` / `tests/test_curator_readiness_cli.py`).
- Label store writes stay append-only; nothing may rewrite or delete lines in an existing `labels.jsonl`.
- Run the venv at `venv/Scripts/python.exe` for all commands (per this machine's setup — `python` on PATH does not have `gin` installed).

---

### Task 1: `--curator` CLI flag on `curator_serve.py`

**Files:**
- Modify: `scripts/curator_serve.py:29-73`
- Test: `tests/test_curator_serve_cli.py` (new)

**Interfaces:**
- Produces: `parse_args(argv: list[str] | None = None) -> argparse.Namespace` — extracted from `main()`, with a `.curator: str` attribute (default `"kristian"`, so your own existing usage is unaffected).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curator_serve_cli.py
"""--curator lets a second labeler stamp their own name on every LabelRecord."""
from scripts.curator_serve import parse_args


def test_default_curator_is_kristian():
    args = parse_args([])
    assert args.curator == "kristian"


def test_curator_flag_overrides_default():
    args = parse_args(["--curator", "alex"])
    assert args.curator == "alex"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_serve_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_args'`

- [ ] **Step 3: Extract `parse_args` and add the flag**

In `scripts/curator_serve.py`, replace the body of `main()` from the start through `args = ap.parse_args()` with:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GIN curator labeling app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--no-seed", action="store_true", help="skip seeding existing gold")
    ap.add_argument("--source", choices=["labeled-set", "escalation-residue"],
                    default="labeled-set", help="candidate source")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("corpus_node1.json"), Path("corpus_node2.json"),
                             Path("corpus_node3.json"), Path("corpus_node4.json")],
                    help="corpus_node*.json exports for the escalation-residue source")
    ap.add_argument("--curator", default="kristian",
                    help="name stamped on every LabelRecord this instance writes")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
```

Then update every remaining reference to the old local `args` in `main()` (they're unchanged in name, so no further edits needed there), and change the `create_curator_app(...)` call's `curator="kristian"` to `curator=args.curator`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_serve_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full curator test suite to check nothing broke**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_app.py tests/test_curator_serve_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/curator_serve.py tests/test_curator_serve_cli.py
git commit -m "curator_serve: add --curator flag for multi-labeler attribution"
```

---

### Task 2: Merge & agreement-check script

**Files:**
- Create: `scripts/curator_merge_check.py`
- Test: `tests/test_curator_merge_check.py`

**Interfaces:**
- Consumes: `gin.curator.store.Store` (`gin/curator/store.py`), `gin.curator.models.LabelRecord` / `pair_key` (`gin/curator/models.py`).
- Produces:
  - `merge_logs(paths: list[Path], out_path: Path) -> None`
  - `AgreementResult` — dataclass with `agree: int`, `disagree: int`, `disagreements: list[tuple[tuple[str, str], LabelRecord, LabelRecord]]`
  - `check_agreement(fold_a: dict[tuple[str, str], LabelRecord], fold_b: dict[tuple[str, str], LabelRecord], keys: set[tuple[str, str]]) -> AgreementResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_curator_merge_check.py
"""Merging two curators' label logs and measuring agreement on shared pairs."""
from pathlib import Path

from gin.curator.models import LabelRecord
from gin.curator.store import Store
from scripts.curator_merge_check import AgreementResult, check_agreement, merge_logs


def _rec(id_, src, dst, relation, curator, ts, relation_class=None):
    return LabelRecord(
        id=id_, src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator=curator, ts=ts,
    )


def test_merge_logs_concatenates_both_files(tmp_path: Path):
    from gin.cartographer.models import Relation

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    store_a = Store(a)
    store_b = Store(b)
    store_a.append(_rec("1", "x:0", "y:0", Relation.CONTRADICTS, "kristian", "2026-07-20T00:00:00Z"))
    store_b.append(_rec("2", "p:0", "q:0", Relation.UNRELATED, "alex", "2026-07-20T00:00:01Z"))

    out = tmp_path / "merged.jsonl"
    merge_logs([a, b], out)

    merged = Store(out).fold_current()
    assert len(merged) == 2
    assert merged[("x:0", "y:0")].curator == "kristian"
    assert merged[("p:0", "q:0")].curator == "alex"


def test_check_agreement_counts_matches_and_mismatches():
    from gin.cartographer.models import Relation

    fold_a = {
        ("x:0", "y:0"): _rec("1", "x:0", "y:0", Relation.CONTRADICTS, "kristian", "t1", "issue_frame"),
        ("p:0", "q:0"): _rec("2", "p:0", "q:0", Relation.UNRELATED, "kristian", "t2"),
    }
    fold_b = {
        ("x:0", "y:0"): _rec("3", "x:0", "y:0", Relation.CONTRADICTS, "alex", "t3", "issue_frame"),
        ("p:0", "q:0"): _rec("4", "p:0", "q:0", Relation.RELATED_UNTYPED, "alex", "t4"),
    }
    result = check_agreement(fold_a, fold_b, {("x:0", "y:0"), ("p:0", "q:0")})

    assert isinstance(result, AgreementResult)
    assert result.agree == 1
    assert result.disagree == 1
    assert len(result.disagreements) == 1
    assert result.disagreements[0][0] == ("p:0", "q:0")


def test_check_agreement_skips_keys_missing_from_either_fold():
    from gin.cartographer.models import Relation

    fold_a = {("x:0", "y:0"): _rec("1", "x:0", "y:0", Relation.CONTRADICTS, "kristian", "t1")}
    fold_b: dict = {}
    result = check_agreement(fold_a, fold_b, {("x:0", "y:0")})
    assert result.agree == 0
    assert result.disagree == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_merge_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.curator_merge_check'`

- [ ] **Step 3: Implement the script**

```python
# scripts/curator_merge_check.py
"""Merge two curators' label logs and measure agreement on a shared overlap set.

Concatenation is a safe merge because Store.fold_current() folds latest-wins
by (ts, idx) over the whole read log, regardless of which file a line came
from (gin/curator/store.py:41-49) — no reconciliation logic needed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gin.curator.models import LabelRecord


def merge_logs(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            out.write(path.read_text(encoding="utf-8"))


@dataclass
class AgreementResult:
    agree: int
    disagree: int
    disagreements: list[tuple[tuple[str, str], LabelRecord, LabelRecord]] = field(default_factory=list)


def check_agreement(
    fold_a: dict[tuple[str, str], LabelRecord],
    fold_b: dict[tuple[str, str], LabelRecord],
    keys: set[tuple[str, str]],
) -> AgreementResult:
    agree = 0
    disagree = 0
    disagreements: list[tuple[tuple[str, str], LabelRecord, LabelRecord]] = []
    for key in keys:
        rec_a = fold_a.get(key)
        rec_b = fold_b.get(key)
        if rec_a is None or rec_b is None:
            continue
        if (rec_a.relation, rec_a.relation_class) == (rec_b.relation, rec_b.relation_class):
            agree += 1
        else:
            disagree += 1
            disagreements.append((key, rec_a, rec_b))
    return AgreementResult(agree=agree, disagree=disagree, disagreements=disagreements)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge curator label logs and check overlap agreement")
    ap.add_argument("logs", type=Path, nargs="+", help="labels.jsonl files to merge")
    ap.add_argument("--out", type=Path, required=True, help="merged output path")
    args = ap.parse_args()

    merge_logs(args.logs, args.out)
    print(f"merged {len(args.logs)} logs into {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_merge_check.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/curator_merge_check.py tests/test_curator_merge_check.py
git commit -m "curator: add merge + overlap agreement-check script for multi-labeler use"
```

---

### Task 3: Labeling guide doc

**Files:**
- Create: `docs/curator-labeling-guide.md`

**Interfaces:** None (doc only).

- [ ] **Step 1: Write the guide, using real examples pulled from `data/curator/labels.jsonl`**

```markdown
# Curator labeling guide

Five relations, two `relation_class` values seen so far. When unsure, prefer
`related_untyped` over forcing a typed relation — it's a valid, storeable
answer, not a cop-out (see `gin/cartographer/models.py:14-26`).

## contradicts

Two chunks describe the same underlying event/topic but frame or diverge on
it. Split into two `relation_class` flavors:

- **issue_frame** — same topic, different institutional-vs-grassroots framing
  of the *issue itself* (not a specific breaking story):
  - "emissions: institutional reduction-target framing vs grassroots treaty
    demands"
  - "wildfire: acreage / suppression metrics vs air quality in low-income
    zones"
  - "rezoning: parcel/FAR/density technical framing vs displacement/
    right-to-return organizing framing"

- **story** — same breaking story, sharing a lede/anchor fact, but the two
  chunks diverge on a specific reported number:
  - "hospital treatment count and arrest count diverge after shared lede"
  - "turnout percentage diverges after shared margin"

If it's same-topic-divergent-numbers *without* a shared lede/anchor, it's
probably `issue_frame` or `related_untyped`, not `story` — `story` requires
that shared anchor.

## corroborates

Same stance, same topic — one chunk backs up or gives a concrete instance of
the other's claim:
- "Snippet B provides specific empirical evidence (historical US warming
  data) that corroborates the broader claim"
- "Snippet B provides a specific empirical example of a record-breaking
  extreme heat event"

## supersedes

One chunk is a later, corrected/updated version of the same claim (rare in
the corpus so far — no worked example yet; if you hit one, add it here).

## related_untyped

Same broad topic, but the two chunks can't be directly logically linked
(different metrics, different aspects) — genuinely related, but not
`corroborates`/`contradicts`:
- "Both snippets discuss climate warming temperature metrics, but they
  cannot be directly linked logically"
- "Both snippets provide data regarding record-breaking warm temperatures,
  but they are independent metrics"

## unrelated

Different topics entirely — this is what the relatedness gate should reject
outright:
- "Different types of facts, statistics that are unrelated to elderly,
  low-income populations..."

## The hardest borderline: related_untyped vs unrelated vs contradicts

Same-topic-different-metric pairs *feel* related but often aren't a typed
relation. Ask in order:
1. Do the two chunks discuss the same real-world topic at all? If no →
   `unrelated`.
2. If yes: can you draw a direct logical line between them (one confirms,
   extends, or conflicts with the other's specific claim)? If no → 
   `related_untyped`.
3. If yes, and the conflict is framing/stance rather than the same
   number/fact → `contradicts` (pick `issue_frame` or `story` per above).
```

- [ ] **Step 2: Commit**

```bash
git add docs/curator-labeling-guide.md
git commit -m "docs: curator labeling guide with worked examples from existing labels"
```

---

### Task 4: Assignment & friend setup doc

**Files:**
- Create: `docs/curator-collab-setup.md`

**Interfaces:** None (doc only). Depends on Task 1 (`--curator` flag) and Task 2 (`curator_merge_check.py`) existing.

- [ ] **Step 1: Write the setup + assignment doc**

```markdown
# Curator collaboration setup

## One-time setup (friend's machine)

1. Clone/pull the repo and set up the venv per the repo's normal instructions.
2. Copy the current `data/curator/labels.jsonl` to their machine (so they see
   existing labels, not an empty corpus) — e.g. `scp` it over, or just hand
   them the file. They keep their own copy; it is **not** a live-shared file
   between the two of you (avoids concurrent-write races).
3. Read `docs/curator-labeling-guide.md` first.

## Launching

Friend's instance, labeling `corpus_node2.json` + `corpus_node3.json`:

```
venv/Scripts/python.exe scripts/curator_serve.py \
  --curator alex \
  --source escalation-residue \
  --corpus corpus_node2.json corpus_node3.json \
  --log data/curator/labels.alex.jsonl \
  --port 8601
```

Your instance keeps doing what it already does (fixture set + node4), just
naming yourself explicitly:

```
venv/Scripts/python.exe scripts/curator_serve.py --curator kristian
```

Assignment:
- **You**: `--source labeled-set` (fixture: clim/disc/grass/hf/inst) plus
  `corpus_node4.json` (node4 issue_frame corpus).
- **Friend**: `corpus_node1.json` + `corpus_node2.json` + `corpus_node3.json`.
- **Overlap set**: friend additionally runs a throwaway pass over
  `corpus_node1.json` alone (already substantially labeled by you) into a
  *separate* log, `--log data/curator/labels.alex.overlap.jsonl`, without
  looking at your existing `n1_*` labels first. This is what
  `curator_merge_check.py` measures agreement over.

Each `--corpus` file is a full topic/node boundary already — no new
filtering code was needed to split by topic (`scripts/curator_serve.py`
already takes `--corpus` as a list).

## Merging

Once both of you have a batch done:

```
venv/Scripts/python.exe scripts/curator_merge_check.py \
  data/curator/labels.jsonl data/curator/labels.alex.jsonl \
  --out data/curator/labels.merged.jsonl
```

Then in a Python shell, compute the overlap-set agreement:

```python
from pathlib import Path
from gin.curator.store import Store

mine = Store(Path("data/curator/labels.jsonl")).fold_current()
overlap = Store(Path("data/curator/labels.alex.overlap.jsonl")).fold_current()
overlap_keys = set(overlap.keys())

from scripts.curator_merge_check import check_agreement
result = check_agreement(mine, overlap, overlap_keys)
print(result.agree, result.disagree)
for key, rec_a, rec_b in result.disagreements:
    print(key, rec_a.relation, rec_a.relation_class, "vs", rec_b.relation, rec_b.relation_class)
```

## Reconciliation

For each disagreement, you (the seed curator) make the tie-break call and
append **one more** `LabelRecord` via the curator UI with `supersedes`
pointing at whichever of the two records it overrides — never edit
`labels.jsonl` by hand. Then re-run `curator_merge_check.py` to fold the
final merged log.
```

- [ ] **Step 2: Commit**

```bash
git add docs/curator-collab-setup.md
git commit -m "docs: curator collaboration setup and assignment for second labeler"
```

---

## Self-Review Notes

- **Spec coverage:** §1 setup/identity → Task 1 + Task 4 setup section. §2 assignment/overlap → Task 4 assignment section. §3 guide/merge → Task 3 (guide) + Task 2 (merge/agreement code) + Task 4 (merge/reconciliation steps).
- **Placeholder scan:** none — every step has literal file contents or commands.
- **Type consistency:** `check_agreement` takes `dict[tuple[str, str], LabelRecord]` in both Task 2's tests and Task 4's usage example; `AgreementResult.disagreements` entries are `(key, rec_a, rec_b)` tuples consistently in both places.
