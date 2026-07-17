# Curator UI + Label Store (Design)

**Date:** 2026-07-17
**Status:** approved design, pre-implementation
**Phase:** Forward layer — sub-project **A** of the framing-corpus track. The shared spine that unblocks sub-project **B** (bi-encoder frame detector) and sub-project **C** (larger-set recalibration of the cheap Cartographer pipeline). See `escalation_judge_model_sweep` finding: issue_frame is "curation-only by nature," so a curator-in-the-loop tool is the honest home for it, not a workaround.

---

## Why this exists (the two items it serves)

Two open items both blocked on the *same* missing input — a labeled framing corpus at scale:

- **Item 1 (Cartographer cheap pipeline)** is built and passing in-sample (`gin/cartographer/combined.py`: precision 0.875 / recall 1.0 / F1 0.933 on 13 core pairs) but its honest `leave_one_out()` number is 0.69 (`gin/cartographer/calibration.py`). The named cause is "13 pairs too few." Its remaining work is *more labels + recalibrate* (sub-project C).
- **Item 2 (issue_frame residue)** is closed for off-the-shelf LLM judges — 7B → Opus 4.8 frontier all fail, and the failure is signal, not scale: `contradicts` here encodes a curatorial stance no general judge reproduces. The decided forward path is a **bi-encoder trained on curator labels** (sub-project B), not a purpose-trained generative LLM.

Today the labels are ~33 hand-edited tuples in `gin/cartographer/labeled_set.py` plus YAML fixtures loaded by `gin/cartographer/gold_edges.py`. This sub-project builds the durable, growing, reproducible label substrate both B and C consume.

**This sub-project is infrastructure.** Its bar is correctness and reproducibility of the labeling substrate — **not** clearing the escalation bar (that is B's job) and **not** improving any Cartographer metric (that is C's job).

## Falsifiable claim

A local FastAPI labeling tool writes curator judgments to a durable, git-trackable, append-only store whose folded view can (a) exactly reproduce today's gold when seeded and (b) be consumed by B and C in the shape they already expect — without touching any currently-passing Cartographer eval or calibration code.

| Metric | Bar |
|---|---|
| Store round-trip | `fold_current(log)` yields latest-wins per unordered pair; a relabel and an adjudication record both correctly supersede the prior label |
| **Seed regression guard** | after seeding the store from `labeled_set.py` + gold YAML, for **every** seeded pair the folded `relation` equals the source's relation (proves the store faithfully represents the existing gold before growing it) — the invariant is the pair→relation mapping, not tuple-shape identity, since the two sources carry different metadata (`register` vs `relation_class`) |
| Ordering | on a fixture, signal-disagreement pairs and mid-band-cosine pairs rank strictly above easy/obvious pairs |
| Reader shape | `gin.curator.store.gold()` returns `(src_chunk_id, dst_chunk_id, relation, relation_class)` tuples that a caller can drop into the same consumption path as `labeled_set.gold()` |
| End-to-end | labeling N pairs through the app (`TestClient`) appends N records to `data/curator/labels.jsonl`, and `fold_current()` reflects all N |
| Existing suite | every current test stays green with **zero** modifications; `labeled_set.py` / `gold_edges.py` / `calibration.py` are byte-for-byte unchanged |
| New runtime dependencies | none (FastAPI + Starlette already in the federation stack) |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-17, with rationale)

1. **Pluggable candidate source, offline default.** A `CandidateSource` protocol; the default `OfflineCandidateSource` reads a DB-free chunk set (fixtures + corpus JSON exports) so labeling runs with no node and no Postgres up, matching the repo's DB-free-first pattern (`gin/eval/edge_robustness.py`, `edge_degradation.py`, and `calibration.default_samples()` all avoid the DB deliberately). A `PostgresCandidateSource` over `gin/cartographer/scan.py`'s live residue is a later adapter behind the same protocol — not built here.

2. **Append-only JSONL event log, not a flat gold file or a DB table.** Every label / relabel / adjudication is an immutable record carrying `curator`, UTC `ts`, `relation`, `relation_class`, `rationale`, the pair, and an optional `supersedes` id. "Current gold" is derived by folding the log latest-wins per *unordered* pair. This keeps the disputed `inst_em`↔`clim_pledges` adjudication as recorded history (a later record supersedes an earlier one) rather than a silent overwrite, mirrors the Bookkeeper's `Provenance` ethos (`gin/bookkeeper/models.py`), and stays git-diffable for reproducible eval snapshots. A flat YAML file loses history; a Postgres table cuts against the offline-default choice and isn't git-diffable.

3. **Full relation vocab + `relation_class` refinement + free-text rationale.** The curator assigns one `Relation` from the existing enum (`contradicts`, `corroborates`, `supersedes`, `related_untyped`, `unrelated`); when `contradicts`, also a `relation_class` (`story` | `issue_frame`); plus an optional rationale. `relation_class` is the exact target the bi-encoder (B) must learn — a flat relation would give B no target class. The rationale keeps the curatorial "why" legible (GIN's whole ethos; the disputed-pair adjudication is precisely the case where the recorded reason matters). `supersedes` is representable for completeness but is not the focus of this track.

4. **FastAPI + one no-build-step HTML/JS page.** One new local-only router on the existing FastAPI stack (already serving the federation layer), serving a single vanilla-JS/HTMX page plus two JSON endpoints. No heavy frontend framework, no build step, full layout control, git-friendly. A browser gives clean side-by-side pair reading and keyboard shortcuts; Streamlit/Gradio would add a heavy dependency and a separate process; a TUI makes side-by-side chunk reading clunky.

5. **Seed + static hard-cases-first ordering, no retraining loop.** Seeding imports the ~33 existing labels as done. The unlabeled backlog is ranked by a *static* informativeness score computed from the cheap pipeline's own signals: signal disagreements first (NLI-says-contradict vs cosine-says-corroborate), then mid-band cosine, then the escalation residue (not-same-story, cosine ≥ floor). An active-learning loop that retrains/rescores the bi-encoder after each batch is deferred to a later iteration (it can't exist before B does, and it couples the UI to a training loop).

6. **Auto-derived sentence anchors, no manual span selection in v1.** Anchors are populated via the existing `gin.cartographer.scan.sentence_anchor()` at save time, matching how `escalate_proposals` already anchors (`gin/cartographer/escalation.py:88-90`). Manual token-offset span selection in the browser is deferred — anchors are optional in `EdgeProposal` and verified downstream by the Bookkeeper anyway, so v1 loses nothing structural by auto-deriving them.

7. **Additive now — do not rewrite `labeled_set.py` / `gold_edges.py`.** The store ships as a *new, separate* labeled source. The existing hardcoded sets stay frozen so the currently-passing combined-detector and calibration evals do not move. This sub-project exposes `gin.curator.store.gold()` in the existing `(src, dst, relation, relation_class)` shape so **C** (recalibration) can consume `store ∪ labeled_set` and is the natural place to unify the loaders — that refactor serves C's goal, not A's, and doing it here would needlessly perturb passing evals. **B** reads the same reader, filtering `relation_class == issue_frame` for its target class.

## Architecture — new `gin/curator/` package + one FastAPI router

Sibling to `gin/cartographer/`, one clear job per module.

| Module | Responsibility |
|---|---|
| `gin/curator/models.py` | `LabelRecord` frozen dataclass (`id`, `src_chunk_id`, `dst_chunk_id`, `relation: Relation`, `relation_class: Optional[str]`, `rationale: str`, `curator: str`, `ts: str` UTC ISO, `supersedes: Optional[str]`). Reuses `gin.cartographer.models.Relation` — no parallel vocab. Pair key is the *unordered* `frozenset({src, dst})` so A↔B and B↔A fold together. |
| `gin/curator/store.py` | `append(record)` (one JSON line to `data/curator/labels.jsonl`), `read_log()` (all records in order), `fold_current()` (latest-`ts`-wins per unordered pair → dict keyed by pair), and `gold()` (the folded view as `(src, dst, relation, relation_class)` tuples — the reader B and C consume). Path is injectable for tests. |
| `gin/curator/candidates.py` | `CandidateSource` protocol (`chunks()`, `pairs()`); `OfflineCandidateSource` reading a DB-free chunk set; `order_backlog(pairs, signals, already_labeled)` computing the static hard-cases-first ranking and excluding folded/seeded pairs. `PostgresCandidateSource` is a documented stub, not implemented. |
| `gin/curator/signals.py` | `pair_signals(a_text, b_text)` → `{cosine, nli_p_contra, same_story, cheap_verdict}` by reusing a single shared `CombinedRelationProposer` (`gin/cartographer/combined.py`) — for display and for the ordering heuristic. No new model code; wraps what the detector already computes. |
| `gin/curator/seed.py` | One-time importer: reads `labeled_set.py` gold + `gold_edges.load_all_gold_contradicts()` and appends them as seed `LabelRecord`s (`curator="seed"`). Metadata gap is handled explicitly: `gold_edges` pairs carry their YAML `relation_class`; `labeled_set` pairs have no such field, so their `contradicts` pairs seed with `relation_class=None` (unknown story-vs-issue_frame), left for the curator to refine later — the importer never *guesses* a class. Idempotent (skips pairs already present with the same relation). |
| `gin/curator/app.py` | `build_curator_router()` → FastAPI `APIRouter`. `GET /curator/` serves the single HTML/JS page; `GET /curator/next?n=` returns the next ordered batch of unlabeled pairs with their text + signals; `POST /curator/label` validates and appends a `LabelRecord`. Local-only; not mounted on the federation app by default. |
| `scripts/curator_serve.py` | Launches a uvicorn app mounting only the curator router, on localhost, pointed at a chosen offline chunk set + label log. |

New data:
- `data/curator/labels.jsonl` — the git-tracked event log (created on first append).
- Offline chunk set — reuse existing corpus JSON exports (`corpus_node*.json`) and/or the fixture texts already embedded in `labeled_set.py`; `OfflineCandidateSource` takes the chunk set as a constructor argument, no new fixture invented here.

Tests: `tests/test_curator_store.py`, `tests/test_curator_candidates.py`, `tests/test_curator_app.py`.

## Data flow

1. `scripts/curator_serve.py` builds an `OfflineCandidateSource` over a chunk set and a `Store` over `data/curator/labels.jsonl`, then serves `build_curator_router()` on localhost.
2. On load, the page calls `GET /curator/next` → the router folds the log, asks the candidate source for unlabeled pairs, computes `pair_signals` for the top slice, orders them hard-cases-first, and returns a batch (pair ids, both texts, signals, cheap verdict).
3. The curator reads chunk A and chunk B side by side, sees the signal readout, and picks a relation (number key). If `contradicts`, a `relation_class` toggle (`story`/`issue_frame`) appears; an optional rationale box is always available.
4. Enter → `POST /curator/label` with the pair + relation (+ class + rationale). The router auto-derives sentence anchors, builds a `LabelRecord` with a fresh id and UTC ts, and `store.append()`s one JSON line.
5. The page advances to the next pair in the batch; when the batch drains it re-fetches. Progress/remaining counter reflects folded-vs-total.
6. A relabel of an already-labeled pair appends a new record carrying `supersedes` = the prior record's id; `fold_current()` surfaces the newer one. The disputed-pair adjudication is exactly this path.

## Error handling

- **`POST /curator/label` with `contradicts` but no `relation_class`:** 422 — the class is required for `contradicts` (it's the bi-encoder's target); the page enforces this too but the router is the source of truth.
- **Label for a pair not in the candidate source:** accepted and stored anyway (the store is the source of truth; the candidate source only drives *ordering*, not admissibility) — recorded with whatever ids the client sent.
- **Empty/missing `labels.jsonl`:** `read_log()` returns `[]`, `fold_current()` returns `{}` — first append creates the file.
- **Malformed line in the log:** `read_log()` fails loudly with the offending line number rather than silently skipping — a corrupted append must not silently drop labels from the folded gold.
- **Concurrent appends:** out of scope — single-curator, single-process tool (see §Out of scope). Appends are one-line writes; no locking designed.

## Testing — three tiers

1. **Store unit (`tests/test_curator_store.py`):** append→read round-trip; `fold_current()` latest-wins on a relabel; unordered-pair folding (A↔B and B↔A collapse); the **seed regression guard** — seed from the existing gold, fold, assert exact reproduction of `labeled_set.gold()` on overlapping pairs; malformed-line loud failure.
2. **Candidates unit (`tests/test_curator_candidates.py`):** `OfflineCandidateSource.pairs()` enumerates the right unlabeled pairs; `order_backlog` ranks a hand-built signal-disagreement pair and a mid-band pair strictly above an obvious high-cosine corroboration; already-folded pairs are excluded. Signals are injected as fixtures (no real embedding/NLI model in the unit test), matching how the federation tests inject fakes.
3. **App (`tests/test_curator_app.py`, `TestClient`):** `GET /curator/next` returns ordered pairs with text + signals (signals stubbed via an injected `signals` callable); `POST /curator/label` appends exactly one record and it appears in the folded view; `contradicts`-without-class → 422; a second label on the same pair supersedes the first in the folded view.

`signals.py`'s real embedding+NLI path is exercised only when a model is actually present; the automated tiers inject fake signal functions so the suite stays model-free and fast, consistent with the rest of the codebase.

## Out of scope (later, in likely order)

1. **Bi-encoder training/detection** — sub-project B; consumes this store's `gold()` filtered to `relation_class == issue_frame`.
2. **Larger-set recalibration + loader unification** — sub-project C; consumes `store ∪ labeled_set`, and is where `labeled_set.py`/`gold_edges.py` get rewritten as readers over the store (scope decision 7).
3. **Active-learning retraining loop** — surfacing the bi-encoder's highest-uncertainty pairs after each batch (needs B first; scope decision 5).
4. **`PostgresCandidateSource`** — labeling the live node's real escalation residue over pgvector (scope decision 1).
5. **Manual anchor span-selection in the browser** — v1 auto-derives sentence anchors (scope decision 6).
6. **Multi-curator / auth / concurrent-write safety** — this is a single-researcher local tool; inter-curator disagreement, if it ever matters, is itself GIN-shaped content but not built here.
7. **Corpus growth** — ingesting more source documents to widen register diversity is a separate corpus-building concern; A labels over whatever chunk set it's pointed at.

## Documentation updates shipped with implementation

- `README.md`: a "curator labeling tool" subsection — how to launch `scripts/curator_serve.py`, the `labels.jsonl` record shape, and how `store.gold()` feeds later work.
- `architecture.md`: note the curator/label-store as the framing-corpus spine feeding the (still-open) bi-encoder frame detector and the cheap-pipeline recalibration, and that it operationalizes the `escalation_judge_model_sweep` conclusion (issue_frame is curation-only by nature).

## New dependencies

None — FastAPI + Starlette (`TestClient`, `APIRouter`, static HTML response) are already in the federation stack; stdlib `json`/`uuid`/`datetime` cover the store; `gin.cartographer` supplies `Relation`, `CombinedRelationProposer`, and `sentence_anchor`.
