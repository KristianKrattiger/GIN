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
  *separate* log, without looking at your existing `n1_*` labels first.
  Use `--no-seed` to exclude auto-seeded fixture records (measures only real
  inter-rater agreement, not trivial matches on seed records):

  ```
  venv/Scripts/python.exe scripts/curator_serve.py \
    --curator alex \
    --source escalation-residue \
    --corpus corpus_node1.json \
    --log data/curator/labels.alex.overlap.jsonl \
    --no-seed \
    --port 8602
  ```

  This is what `curator_merge_check.py` measures agreement over.

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
