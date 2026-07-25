# Design: Cheap-Pipeline Recalibration (sub-project C)

**Date:** 2026-07-25
**Status:** Approved (design), pending implementation plan
**Sub-project:** C — consumes A/B0's label store; independent of B's detector

## Problem

`CombinedRelationProposer`'s thresholds are calibrated by grid search over
**39 baked sample tuples** hardcoded in `gin/cartographer/calibration.py`
(`_MEASURED`), measured over `labeled_set.py`'s 33 gold pairs plus 6
`related_untyped` samples pulled from a 2026-07-12 scan run. The module's own
comment says "regenerate if the labeled set changes." It has not been
regenerated. The curator store now holds **178 labeled pairs**.

Three concrete defects follow:

1. **The calibration set is stale and small.** 39 samples, hand-transcribed,
   while 133 non-eval labeled pairs sit unused in the store.
2. **The persisted artifact does not match the code.**
   `data/cartographer_thresholds.json` records
   `"leave_one_out_accuracy": 0.923`; recomputing `leave_one_out(default_samples())`
   today yields **0.897**. Something moved after 2026-07-13 without the file
   being regenerated, and nothing detected it.
3. **Calibration and evaluation overlap.** The 39 samples derive from
   `labeled_set`, which is also the set `scan_eval`/`evaluation` measure
   against. Reported accuracy is partly in-sample.

Sub-project A deliberately deferred this: it declined to rewrite
`labeled_set.py`/`gold_edges.py` "so passing evals don't move." That deferral
was correct then and the constraint still binds — this design honors it rather
than discarding it.

## Goal

Recalibrate the cheap pipeline against the full curated corpus, with an honest
held-out number, and make regeneration a command rather than a hand-edit — while
leaving every pre-registered evaluation surface provably frozen.

**Not in scope:** the encoder/representation question sub-project B surfaced
(whether epistemic relation structure is recoverable from a different encoder).
That is a separate research probe with a different risk profile and gets its own
spec. Also out of scope: any change to `gin/frames/`, and any new relation type.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope of "C" | Recalibration, not the encoder probe | Original-C is fully unblocked and improves code already in production; the encoder question is speculative and deserves separate framing. |
| Loader unification | **Split** — eval surfaces frozen, calibration reads the store | The bar is pre-registered and must not move; calibration should track new labels. Full unification would put a pre-registered eval one bad label away from drifting across a 15-file blast radius. |
| Train/eval split | Calibrate on store **minus** the 45 eval pairs | `labeled_set` + `gold_edges` pairs are what `scan_eval`/`evaluation` measure against. Excluding them makes the reported number held-out rather than a restatement. Leaves 133 calibration pairs. |
| Sample persistence | Generated JSON data file + manifest | Keeps the deliberate "calibration reproduces without models" property while making regeneration a command. A generated Python literal invites accidental hand-edits; on-the-fly computation would force models into calibration tests. |
| Disputed pair | Curator adjudicates; run reports sensitivity on the **held-out score** | `inst_em:0 ↔ clim_pledges:0` is a curatorial call, not the implementer's. It is a `labeled_set` member, hence an eval pair excluded from calibration, so flipping it moves the held-out number rather than the thresholds. The run reports it both ways so the cost of the decision is visible. |

## Current state (measured 2026-07-25)

| | value |
|---|---|
| Baked calibration samples | 39 |
| Calibrated thresholds | gate 0.140, ceiling 0.486, contra 0.686 |
| LOO accuracy (recomputed) | **0.897** |
| LOO class_c_discrimination | 1.000 |
| Persisted `thresholds.json` claim | 0.923 — **stale, does not reproduce** |
| Store folded pairs | 178 |
| Eval pairs (`labeled_set` ∪ `gold_edges`) | 45 id-keys, **40 distinct offline-measurable** — see note |
| Available calibration pairs | **133** |

**Note on 45 vs 40.** `gold_edges` and `labeled_set` name the same 5 pairs under
two chunk-id schemes (`disc_northwind_complaint:0` *is* `disc_nw_complaint:0`;
likewise `disc_meridian_*`, `hf_alderflats_*`, `hf_kestrel_*`, `wf_multi_*`). The
store was seeded from both and holds both copies, but only the short-form ones
resolve to offline text. So `eval_pair_keys()` is 45 id-keys while the held-out
set is **40 distinct pairs**. The leakage guarantee is unaffected: long-form
copies drop as `text_unresolved` and their short-form twins as `eval_pair`, so
neither can become a calibration sample. This is the same id-aliasing hazard
sub-project B hit from the other direction, and it is why guards in this codebase
should compare text rather than chunk ids wherever that is possible.

Calibration-pair class mix: `related_untyped` 62, `corroborates` 26,
`contradicts` 22, `unrelated` 21, `supersedes` 2 (the last excluded — not a
classifier output).

## Architecture

```
data/curator/labels.jsonl
        │  Store.gold()  → 178 folded pairs
        ▼
  exclude the 45 eval pairs (labeled_set ∪ gold_edges)  ── frozen, never calibrated on
        │
        ▼  133 pairs
  scripts/regen_calibration_samples.py
        │  real embed + NLI + same_story, once
        ▼
  data/calibration/samples.json   {manifest, samples[]}
        │
        ▼  model-free from here down
  calibration.default_samples() ── manifest gate (model ids must match combined.py)
        │
        ├──► calibrate()      → Thresholds
        └──► leave_one_out()  → honest accuracy
        │
        ▼
  data/cartographer_thresholds.json  (consumed by combined.load_thresholds)
```

### Frozen surfaces — the load-bearing constraint

These do **not** change and gain an explicit guard:

- `gin/cartographer/gold_edges.py` and `escalation_eval.default_calibration_sets()`
  keep their current definitions. A new test pins the escalation bar's exact 14
  pairs **by chunk id**, so any future edit that would move the bar fails loudly.
- `scan_eval` / `evaluation` continue to measure against `gold_edges`. They are
  the cheap pipeline's held-out eval and must not become store readers.
- `labeled_set.py` keeps serving chunk text — `gin/curator/text_index.py`
  depends on it, and sub-project B's whole DB-free path runs through that.

Only **sample generation** reads the curator store. This is the split the
brainstorming settled on: single-source-of-truth where it helps, nowhere it
endangers a pre-registered eval.

### Component 1 — sample generator (`scripts/regen_calibration_samples.py`)

Reads the store, drops the 45 eval pairs and any `supersedes` rows, resolves
text via `gin.curator.text_index.default_text_index()`, and for each remaining
pair computes:

- `cos` — `CombinedRelationProposer.embedding_cosine`
- `p_contra` — max-direction NLI, the same scorer `combined.py` uses
- `same_story` — `relatedness.make_same_story` over the corpus texts, matching
  what the classifier receives at scan time

Writes `data/calibration/samples.json`:

```json
{
  "manifest": {
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "nli_model": "cross-encoder/nli-deberta-v3-xsmall",
    "n_samples": 133,
    "class_counts": {"related_untyped": 62, "corroborates": 26,
                     "contradicts": 22, "unrelated": 21},
    "excluded_eval_pairs": 45,
    "git_sha": "…",
    "created_utc": "…"
  },
  "samples": [{"cos": 0.39, "p_contra": 0.068,
               "same_story": false, "relation": "contradicts"}]
}
```

Network/model-bound, run deliberately — never during tests or calibration.

### Component 2 — sample loading (`calibration.default_samples()`)

Reads the JSON file. **Manifest gate:** if `embed_model` or `nli_model` disagree
with `combined.DEFAULT_EMBED_MODEL` / `DEFAULT_NLI_MODEL`, raise — a sample file
measured with different models must never silently calibrate the live pipeline.
This mirrors the `head_sha256` gate `gin/frames/head.py` already uses, and it is
the direct fix for defect 2 above.

`_MEASURED` is deleted, not kept as a fallback: a silent fallback to 39 stale
samples is precisely the failure being removed.

### Component 3 — recalibration CLI (`scripts/recalibrate_cheap_pipeline.py`)

Loads samples, calibrates, computes LOO, and prints a comparison against the
current baseline. Writes `data/cartographer_thresholds.json` including the
provenance fields the current file lacks (`n_samples`, model ids, git sha,
UTC) so a stale artifact is detectable next time.

**Sensitivity line.** `inst_em:0 ↔ clim_pledges:0` is a member of
`labeled_set` gold, so it is one of the 45 **eval** pairs and is excluded from
calibration — flipping it cannot move the thresholds. What it moves is the
**held-out score**. The CLI therefore reports held-out accuracy computed twice:
once with the pair as `corroborates` (its current store label) and once as
`contradicts`. This changes no label; it prices the open curatorial decision so
the curator can adjudicate on evidence.

## Success criteria

Pre-registered, stated before the numbers are seen:

- **Primary:** LOO accuracy over the 133 calibration pairs, reported against the
  current **0.897 / 39 samples** baseline, together with
  `class_c_discrimination` (currently 1.000).
- **Held-out:** the recalibrated thresholds scored on the 45 frozen eval pairs
  via the existing `scan_eval`/`evaluation` path — the number that was
  previously partly in-sample and now is not.
- **Invariant:** the escalation bar's 14 pairs are byte-identical before and
  after; the pinning test proves it.

**Report the number whichever direction it moves.** More calibration data
reducing accuracy is a live outcome, not a failure — it happened to AGREE in
sub-project B when hard negatives were added, and it would mean the 39 baked
samples were flattering. A drop with a wider, more representative sample is more
trustworthy than the 0.897 it replaces, and the writeup must say so rather than
reaching for the older number.

## Failure modes

| Condition | Handling |
|-----------|----------|
| `samples.json` missing | Hard error naming the regen command — never fall back to baked samples |
| Manifest model ids mismatch | Hard error; a differently-measured sample set must not calibrate the live pipeline |
| A class empties after eval-pair exclusion | Hard error — never grid-search on a degenerate set |
| Chunk text unresolvable during regen | Pair dropped, counted by reason, surfaced in the run summary |
| Escalation bar would change | Pinning test fails |
| Thresholds file written without provenance | Not possible — fields are required by the writer |

## Testing

- **Bar pinning** — the 14 escalation-bar pairs asserted by exact chunk id.
- **Eval-pair exclusion** — no pair in `labeled_set ∪ gold_edges` appears in the
  generated sample set; asserted against the real store (regression guard on the
  133/45 split).
- **Manifest gate** — mismatched model ids raise; matching ids load.
- **Schema round-trip** — samples survive write→read identically.
- **Calibration determinism** — same samples produce the same thresholds.
- **Existing calibration tests** — repointed at a small committed fixture sample
  file rather than the live one, so they stay model-free and stable.
- Everything except `regen_calibration_samples.py` runs without models.

## Open questions

None blocking. The disputed pair is deliberately left open and priced rather
than decided; adjudicating it is a curator action that needs no code change.
