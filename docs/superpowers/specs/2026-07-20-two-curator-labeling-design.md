# Two-curator labeling workflow

Date: 2026-07-20
Status: approved (pending spec self-review + user sign-off)

## Context

The framing corpus (`data/curator/labels.jsonl`) is currently labeled solely by
one curator (`curator: "seed"` on all 75 records). We want a friend to join as
a second labeler to speed up coverage of the remaining backlog. The label
store (`gin/curator/store.py`) is already append-only with latest-wins folding
by `(ts, idx)` and carries a per-record `curator` field — both of which exist
specifically to support exactly this kind of multi-labeler, multi-session
history without ever mutating a record in place.

Given there are exactly two labelers, this design deliberately avoids adding
any assignment/locking infrastructure (hash-partitioning, a shared task-queue
server) — that engineering only pays for itself at 3+ concurrent labelers.
Everything below is process convention plus one new doc; no code changes to
`gin/curator/`.

## 1. Setup & identity

- Friend clones/pulls the repo and gets a copy of the current
  `data/curator/labels.jsonl` (so they see the 75 existing labels, not an
  empty corpus).
- They launch their own instance: `python scripts/curator_serve.py
  --curator=<friend-name>`.
- The existing `--curator` flag (`gin/curator/app.py:46`) stamps their name on
  every `LabelRecord` they write — no schema or code change needed, this field
  is already there and simply always been `"seed"` until now.
- They work against their own local copy of `labels.jsonl`, not a live-shared
  file, to avoid concurrent-write races on the same file from two processes.

## 2. Assignment: topic split + overlap

Current corpus spans these source-topic prefixes (chunk-id prefix before the
first `_`/`:`), with current label counts:

| topic | labels |
|---|---|
| n1 | 30 |
| inst | 23 |
| hf | 19 |
| disc | 18 |
| clim | 18 |
| grass | 16 |
| n2 | 16 |
| incident | 6 |
| wf | 2 |
| election | 2 |

Split:
- **You** keep working the topics you're already deep in (candidate: `inst`,
  `clim`, `disc`, `n1`).
- **Friend** takes a disjoint primary set (candidate: `hf`, `n2`, `grass`,
  plus the thinner `incident`/`wf`/`election` topics to build coverage there).
  Exact split is finalized when the plan is written, based on the actual
  remaining (unlabeled) backlog per topic, not just current label counts.
- **Overlap set**: friend also blind-relabels one topic you've already fully
  labeled (candidate: `grass`, 16 existing labels) — blind meaning they don't
  see your existing labels for that topic first. This produces an
  inter-rater agreement measurement before trusting their labels on new
  ground.

## 3. Labeling guide & merge

**Guide doc** (`docs/curator-labeling-guide.md`, produced as part of the
implementation plan, not this spec): defines the 5 `Relation` values
(`contradicts`, `corroborates`, `supersedes`, `related_untyped`, `unrelated`)
and the two `relation_class` values seen in the corpus so far (`issue_frame`,
`story`), each illustrated with 2-3 real examples pulled from existing
`rationale` fields already in `labels.jsonl` (e.g. the `issue_frame` vs
`story` distinction, and the `related_untyped`-vs-`unrelated`-vs-`contradicts`
borderline case for same-topic-different-metric pairs). No guide content is
invented from scratch — it's extracted and organized from the seed curator's
own past rationale.

**Merge**: after each labeling pass, concatenate your `labels.jsonl` and
friend's `labels.jsonl` into one file (order between the two files doesn't
matter — `Store.fold_current()` sorts globally by `(ts, idx)`). Run
`fold_current()` on the merged file, and separately on each individual file
restricted to the overlap-set pairs, to compute agreement.

**Reconciliation**: for overlap disagreements, the seed curator (you) makes
the tie-break call and appends one more `LabelRecord` with `supersedes`
pointing at the record it overrides — never an in-place edit, consistent with
the store's existing history-preserving design.

## Out of scope

- Any code change to `gin/curator/` (assignment partitioning, locking, a
  shared server). Revisit only if a 3rd+ labeler joins.
- Automated agreement-scoring tooling beyond a manual `fold_current()` diff —
  can be added later if overlap-checking becomes a recurring need.
