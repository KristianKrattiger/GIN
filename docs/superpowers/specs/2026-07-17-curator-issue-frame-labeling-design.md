# Curator issue_frame Labeling Productivity (B0) — Design

**Date:** 2026-07-17
**Status:** approved design, pre-implementation
**Phase:** Framing-corpus track, sub-project **B0** — the labeling-productivity layer that unblocks sub-project **B** (bi-encoder frame detector). B0 trains no model; it makes issue_frame labeling reachable and its progress measurable. See [[framing_corpus_curator]] and the sub-project A spec (`2026-07-17-curator-ui-label-store-design.md`).

---

## Why this exists

B (bi-encoder frame detector) is **gated on more labels** (decided 2026-07-17). The census is decisive: the escalation bar is 4 issue_frame + 6 corroboration + 4 unrelated pairs, and the *only* issue_frame-tagged data anywhere (curator store + `gold_edges`) is exactly those **4 pairs** — they **are** the eval set. There is zero held-out issue_frame training data, so B cannot train or validate honestly yet.

Two gaps block "just label more":
1. Sub-project A's launcher points at `labeled_set.chunks()` (18 chunks), which does **not contain** the twonode/news corpus where the issue_frame residue lives — so labeling as-is can never grow the issue_frame set.
2. "Gated on labels" has no measure — nothing says how close B is to trainable.

B0 closes both: a residue candidate source over the full corpus, and a class-count readiness gauge. When the gauge is green, B unblocks.

## Falsifiable claim

The curator can label the real issue_frame residue over the full corpus, and a cheap (no-model) readiness gauge reports honest per-class progress toward a documented target — excluding the fixed escalation-bar pairs so the bar's own data never counts as progress.

| Metric | Bar |
|---|---|
| Residue reachability | `EscalationResidueCandidateSource.pairs()` over `corpus_node*.json` surfaces the not-same-story, cosine≥floor residue (the ~91–338 candidate pool), not the full O(n²) pair space |
| chunk-id convention | loaded corpus chunk ids are `f"{doc_id}:{position}"` (e.g. `n1_doc_005:2`), matching the gold/bar/store convention exactly — verified on a fixture |
| Bar-exclusion | a store seeded with only the 4 escalation-bar issue_frame pairs reports `new_issue_frame == 0` (the bar's own pairs are excluded from progress) |
| Gauge correctness | `readiness()` counts new (non-bar) issue_frame/agree/unrelated labels correctly and flips `ready` true only when all three meet the target |
| No model | B0 imports/trains no embedding-head or classifier; the gauge is pure counting over the folded store |
| Additivity | sub-project A's modules and tests stay green with zero modifications except the two files B0 explicitly extends (`app.py`, `scripts/curator_serve.py`) |
| New runtime deps | none |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-17, with rationale)

1. **Reuse `escalation_candidates`, don't re-implement the residue.** `gin/cartographer/escalation.py::escalation_candidates(pairs, proposer, *, cos_floor=0.30, max_candidates=400)` is the already-measured residue definition (not same-story, cosine ≥ floor, cosine-sorted). B0's candidate source wraps it rather than writing a parallel filter — so what the curator labels stays aligned with what the escalation bar actually tests. This requires `proposer.same_story` wired, which `gin/cartographer/scan.py::wire_same_story(proposer, chunks)` provides from the corpus texts.

2. **Offline corpus JSON, DB-free.** Chunks load from `corpus_node*.json` exports, matching sub-project A's offline-default posture and the repo's DB-free-first pattern. A live-Postgres residue source is deferred (same posture as A's deferred `PostgresCandidateSource`).

3. **chunk-id normalization to `{doc_id}:{position}`.** `corpus_node*.json` stores chunk ids as `n1_doc_005_c002`, but the gold, the escalation bar, and the curator store all use `n1_doc_005:2`. The loader builds `LabeledChunk(f"{doc_id}:{position}", text)` from the document's `doc_id` and each chunk's `position`. Without this, residue-labeled pairs and the bar's pairs would never match by key, and the readiness gauge would silently miscount.

4. **Readiness = configurable class-count thresholds, no model training.** The gate was chosen precisely to avoid building B prematurely, so the gauge must not train anything. It counts curator-store labels mapped to the 3-way frame vocabulary and compares each count to a target. Exact counting rule (to remove ambiguity):
   - `new_issue_frame` = store labels with `relation == contradicts` **and** `relation_class == "issue_frame"`, pair ∉ bar. (None-class contradicts — the `labeled_set` framing-divergence seeds — are **not** counted here; they are a different flavor and the target class we are short on is specifically the anchor-less issue_frame residue.)
   - `new_agree` = store `corroborates`, pair ∉ bar. `new_unrelated` = store `unrelated`, pair ∉ bar. These **do** include the disjoint `labeled_set` corroborates/unrelated seeds (10 AGREE, 16 UNRELATED) — those are legitimate training controls disjoint from the bar, so on day one the gauge honestly shows e.g. `agree 10/20, unrelated 16/20, issue_frame 0/20`; the binding constraint is issue_frame.

   Only the fixed escalation-bar 14 pairs are excluded. `ReadinessTarget` defaults to `issue_frame=20, agree=20, unrelated=20`, documented rationale: enough for a small head plus an internal held-out val split disjoint from the bar. It is a parameter, adjustable via CLI/constructor — not a hard-coded truth.

5. **Bar pairs come from `default_calibration_sets()`, DRY.** The 14 excluded pairs are read from `gin/cartographer/escalation_eval.default_calibration_sets()` (issue_frame / corroboration / unrelated), the same source the bar itself uses — so if the bar's definition ever changes, the exclusion follows automatically.

6. **Surfaced two ways: `GET /curator/readiness` (page progress line) + `scripts/curator_readiness.py`.** See progress while labeling and check it from the CLI without a server.

7. **B0 builds no model and expands no corpus.** Bi-encoder training/eval (B proper), corpus expansion, and active-learning retraining are all out of scope.

## Architecture — new `gin/curator/` modules + two extensions

| Module | Responsibility |
|---|---|
| `gin/curator/corpus_json.py` (new) | `load_corpus_chunks(paths: Iterable[Path]) -> list[LabeledChunk]` — flatten each `corpus_node*.json`'s `documents[].chunks[]`, building `LabeledChunk(f"{doc_id}:{position}", text)`. Deduplicates by chunk id. No DB. |
| `gin/curator/residue.py` (new) | `EscalationResidueCandidateSource` implementing A's `CandidateSource` protocol (`chunks()`, `pairs()`). Constructor takes the loaded chunks (+ optional injected proposer for tests, `cos_floor`, `max_candidates`); builds a `CombinedRelationProposer`, calls `wire_same_story(proposer, chunks)`, and `pairs()` returns `escalation_candidates(all_pairs, proposer, cos_floor=..., max_candidates=...)`. |
| `gin/curator/readiness.py` (new) | `ReadinessTarget(issue_frame=20, agree=20, unrelated=20)` frozen dataclass; `ReadinessReport(new_issue_frame, new_agree, new_unrelated, target, ready)`; `readiness(store: Store, target: ReadinessTarget = ...) -> ReadinessReport`. Excludes bar pairs (via `default_calibration_sets()` → `pair_key`), counts new labels, sets `ready` iff all three ≥ target. |
| `scripts/curator_readiness.py` (new) | CLI: load the store at `--log` (default `data/curator/labels.jsonl`), print the `ReadinessReport` (per-class current/target + verdict). |
| `gin/curator/app.py` (extend) | Add `GET /curator/readiness` returning the `ReadinessReport` as JSON; the served page's progress line fetches it and renders `issue_frame N/T · agree M/T · unrelated K/T · ready?`. Existing endpoints unchanged. `create_curator_app` gains an optional `readiness_target` param (defaults to the standard target). |
| `scripts/curator_serve.py` (extend) | Add `--source {labeled-set,escalation-residue}` (default `labeled-set`, unchanged behavior) and `--corpus PATH...` (default the three `corpus_node*.json`). When `escalation-residue`, build `EscalationResidueCandidateSource(load_corpus_chunks(corpus))` instead of the `labeled_set` source. |

## Data flow

1. `venv/Scripts/python.exe scripts/curator_serve.py --source escalation-residue --corpus corpus_node1.json corpus_node2.json corpus_node3.json`.
2. `load_corpus_chunks` flattens + normalizes ids → `EscalationResidueCandidateSource` builds a `CombinedRelationProposer`, wires `same_story`, and `pairs()` yields the residue (not same-story, cosine ≥ floor, cosine-sorted).
3. `GET /curator/next` scores those residue pairs' signals and applies A's `order_backlog` (hard-cases-first) — so within the residue, signal-disagreement and mid-band pairs come first.
4. The curator labels each as contradicts (+ `issue_frame`) / corroborates / unrelated; `Store` appends.
5. The page progress line calls `GET /curator/readiness`, which folds the store, drops the 14 bar pairs, and returns per-class new counts + `ready`. `scripts/curator_readiness.py` prints the same without a server.
6. When `ready` is true (all three classes ≥ target beyond the bar), B is unblocked — its own spec→plan→SDD cycle follows.

## Error handling

- **Missing/malformed `corpus_node*.json`:** `load_corpus_chunks` raises with the offending path; a missing file in the `--corpus` list is a hard error, not a silent skip (labeling over a truncated corpus would distort the residue).
- **A chunk missing `position` or its document missing `doc_id`:** raises — id normalization must be exact, so a malformed record fails loudly rather than producing an unmatchable id.
- **Empty residue (no pair clears the floor):** `pairs()` returns `[]`; `/curator/next` returns an empty batch; the page shows "remaining 0" — a valid state, not an error.
- **Store with only the seeded bar pairs:** `readiness()` returns all-zero new counts and `ready=false` — the expected day-one state.
- **`escalation-residue` selected with no `--corpus` resolvable:** launcher exits with a clear message rather than serving an empty source.

## Testing

1. **`tests/test_curator_corpus_json.py`:** on a tiny 2-document fixture, `load_corpus_chunks` flattens chunks and normalizes ids (`n1_doc_005_c002` → `n1_doc_005:2`); a record missing `position`/`doc_id` raises; duplicate ids dedupe.
2. **`tests/test_curator_residue.py`:** `EscalationResidueCandidateSource.pairs()` with an **injected** proposer (fake `same_story` + `embed_cos`, model-free) excludes same-story pairs, drops pairs below `cos_floor`, and returns cosine-sorted candidates — asserting it delegates to `escalation_candidates` (same result as calling it directly).
3. **`tests/test_curator_readiness.py`:** a store seeded with only the 4 bar issue_frame pairs → `new_issue_frame == 0`; adding a non-bar issue_frame label → count 1; `ready` is false below target and true at/above it for all three classes; bar corroboration/unrelated pairs are likewise excluded.
4. **`tests/test_curator_app.py` (extend):** `GET /curator/readiness` returns the report shape (`new_issue_frame`, `new_agree`, `new_unrelated`, `target`, `ready`) over a tmp store.
5. **Manual smoke:** launch with `--source escalation-residue`, confirm the page surfaces real twonode/news residue pairs and the progress line shows the readiness counts; `scripts/curator_readiness.py` prints a matching report.

Model-backed paths (the real `CombinedRelationProposer` inside the residue source) are exercised only in the manual smoke; the automated tiers inject fakes so the suite stays model-free — consistent with sub-project A.

## Out of scope (later, in likely order)

1. **Bi-encoder training + escalation-bar eval** — sub-project B; unblocks when the readiness gauge is green.
2. **Corpus expansion** — ingesting more source documents rich in opposing-frame-same-issue material, if the existing corpus's residue proves too thin to reach the target.
3. **Active-learning retraining ordering** — surfacing the bi-encoder's highest-uncertainty residue pairs (needs B to exist).
4. **Live-Postgres residue source** — labeling the running node's real residue over pgvector.

## New dependencies

None — reuses `gin.cartographer` (`escalation_candidates`, `wire_same_story`, `CombinedRelationProposer`, `default_calibration_sets`), sub-project A's `Store`/`CandidateSource`/`order_backlog`/`app`, stdlib `json`/`pathlib`, and FastAPI (already present).
