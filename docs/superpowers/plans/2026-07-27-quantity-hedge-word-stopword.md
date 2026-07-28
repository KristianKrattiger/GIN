# Strip "roughly" From Measure Tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a single unstripped hedge-adverb ("roughly") from spuriously aligning two unrelated quantities in the same-story stance channel.

**Architecture:** One word added to a stopword frozenset in `gin/cartographer/quantity.py`. No other production code changes — the function that reads this set (`_content`, feeding `_measure_tokens`) has exactly one call site, so the change cannot reach scope extraction, `unit_class`, or revision/`as_of` detection.

**Tech Stack:** Python, pytest. Model-free throughout except the final measurement task, which uses the real `sentence-transformers`/`cross-encoder` models already used elsewhere in this repo.

## Global Constraints

- `data/cartographer_thresholds.json` must be byte-identical after this work.
- No new tunable threshold introduced (in particular, `ALIGN_FLOOR` is not touched).
- Only `"roughly"` is added to `_STOPWORDS` — no other hedge word.
- Spec: `docs/superpowers/specs/2026-07-27-quantity-hedge-word-stopword-design.md`.

---

### Task 1: Add `"roughly"` to `_STOPWORDS`

**Files:**
- Modify: `gin/cartographer/quantity.py:73-79` (`_STOPWORDS`)
- Test: `tests/test_cartographer_quantity.py`

**Interfaces:**
- Consumes: `_STOPWORDS: frozenset[str]` (module-private, read only by `_content()`), `extract_mentions(text: str) -> list[QuantityMention]`, `stance_for(a_text: str, b_text: str) -> Optional[str]` (all pre-existing, unchanged signatures).
- Produces: no new public interface. `stance_for` on the two specific sentences named in Step 1 below now returns `quantity.UNALIGNED` instead of `"conflict"` — this is the fact Task 2 and Task 3 build on.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cartographer_quantity.py`, add this test directly after `test_extracts_speed_area_and_thousands_separators` (currently ending at line 54, immediately before `test_extracts_a_date_as_its_own_unit_class` at line 57):

```python
def test_roughly_is_not_a_measure_token():
    m = _only("Officials said the disruption delayed commuters by roughly "
               "45 minutes during the morning rush.")
    assert "roughly" not in m.measure
```

Add this test directly after `test_unaligned_when_both_state_quantities_that_do_not_align` (currently ending at line 189, immediately before `test_none_when_either_text_states_no_quantity` at line 192):

```python
def test_roughly_does_not_align_two_unrelated_quantities():
    # Real node5 corpus sentences (n5_doc_023, n5_doc_024). Before this fix,
    # the shared hedge-adverb "roughly" was the ENTIRE measure overlap
    # (Jaccard 1/18 ~= 0.056, just above ALIGN_FLOOR), so an unrelated
    # dockworker headcount and a transit delay in minutes spuriously aligned
    # as a "conflict". Both sentences still state a quantity, they just no
    # longer share a token -- same semantics as the UNALIGNED test above,
    # not the None test below (neither is quantity-free).
    dockworkers = ("Organizers said the action is part of a coordinated "
                   "national walkout involving roughly 3,200 dockworkers "
                   "at ports across the country.")
    transit_delay = ("Officials said the disruption delayed commuters by "
                      "roughly 45 minutes during the morning rush.")
    assert stance_for(dockworkers, transit_delay) is UNALIGNED
```

Both `_only` and `UNALIGNED` are already imported/defined earlier in this file (`_only` at line 13, `UNALIGNED` imported at line 5) — no new imports needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_quantity.py -v -k "roughly"`

Expected: **2 failed**.
- `test_roughly_is_not_a_measure_token`: fails with `AssertionError` — `"roughly"` IS currently in `m.measure`.
- `test_roughly_does_not_align_two_unrelated_quantities`: fails — `stance_for` currently returns `"conflict"`, not `UNALIGNED`.

- [ ] **Step 3: Add `"roughly"` to `_STOPWORDS`**

In `gin/cartographer/quantity.py`, replace (lines 73-79):

```python
_STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with without as is are was were
be been being has have had said say says it its this that these those they
their there then than not no nor so if while during after before over under
about up out off down more most less least new newly than which who whom whose
will would can could may might must shall should do does did done also very
""".split())
```

with:

```python
_STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with without as is are was were
be been being has have had said say says it its this that these those they
their there then than not no nor so if while during after before over under
about up out off down more most less least new newly than which who whom whose
will would can could may might must shall should do does did done also very
roughly
""".split())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_quantity.py -v`

Expected: **all passing** (33 tests: 31 pre-existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add gin/cartographer/quantity.py tests/test_cartographer_quantity.py
git commit -m "quantity.py: strip roughly as a hedge-adverb from measure tokens"
```

---

### Task 2: Update the pinned false-positive set and confirm no regression

**Files:**
- Modify: `tests/test_cartographer_stance_node5.py`

**Interfaces:**
- Consumes: Task 1's `_STOPWORDS` change (already committed; no code from Task 1 is called directly by this task beyond what the existing test file already imports).
- Produces: an updated pinned false-positive set that Task 3's real-model run is expected to match.

- [ ] **Step 1: Update the pinned false-positive test**

In `tests/test_cartographer_stance_node5.py`, the function `test_the_two_residual_false_positives_are_the_pre_registered_ones` currently asserts (near the end of the function):

```python
    assert false_positives == {
        frozenset(("n5_doc_023:0", "n5_doc_024:0")): "stance",
        frozenset(("n5_doc_023:0", "n5_doc_026:0")): "band",
    }
```

Replace with:

```python
    assert false_positives == {
        frozenset(("n5_doc_023:0", "n5_doc_026:0")): "band",
    }
```

Also update this function's docstring, which currently reads:

```python
    """Both are cross-event, and each illustrates a different known cost.

    ``n5_doc_023 <-> 024`` fires through the **stance** channel and needs BOTH
    known weaknesses at once: stage 1's union anchor passes it ("Union Yard" in
    a transit report against "the union local" in a port-strike report), and at
    ALIGN_FLOOR 0.05 two unrelated counts clear the measure-overlap test. It is
    the concrete instance of the low-floor hazard the plan pre-registered, and
    either fix removes it.

    ``n5_doc_023 <-> 026`` fires through the **band** channel with
    ``stance=None`` -- the pre-stance branch, reached because one side states no
    quantity. That is the deliberate price of the None-versus-UNALIGNED split:
    keeping the None path is what preserves the three gold contradicts pairs
    that contradict qualitatively, and the same path necessarily preserves the
    degenerate branch for quantity-free pairs. Recorded as a measured cost of
    that decision, not a defect.

    Pinned by name and channel so a change that swaps one false positive for a
    different one cannot hide behind an unchanged total.
    """
```

Replace with:

```python
    """The one remaining false positive is cross-event and unrelated to the
    stance mechanism itself.

    ``n5_doc_023 <-> 024`` was fixed in sub-project G (`docs/superpowers/specs/
    2026-07-27-quantity-hedge-word-stopword-design.md`): the shared hedge-word
    "roughly" was the entire measure overlap between an unrelated dockworker
    headcount and a transit delay in minutes. Stripping it from
    ``_STOPWORDS`` moves this pair's stance to ``UNALIGNED``, which correctly
    abstains.

    ``n5_doc_023 <-> 026`` fires through the **band** channel with
    ``stance=None`` -- the pre-stance branch, reached because one side states no
    quantity. That is the deliberate price of the None-versus-UNALIGNED split:
    keeping the None path is what preserves the three gold contradicts pairs
    that contradict qualitatively, and the same path necessarily preserves the
    degenerate branch for quantity-free pairs. This is a stage-1 defect (the
    union/Union anchor collision), not a stance-layer one, and stays an
    accepted, documented cost -- sub-project G's spec deliberately does not
    address it.

    Pinned by name and channel so a change that swaps this false positive for
    a different one cannot hide behind an unchanged total.
    """
```

- [ ] **Step 2: Run this test file to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_stance_node5.py -v`

Expected: **3 passed**. In particular, `test_the_two_residual_false_positives_are_the_pre_registered_ones` passes with the single remaining pair.

- [ ] **Step 3: Run the full suite**

Run: `venv/Scripts/python.exe -m pytest -q`

Expected: **746 passed, 16 skipped** (baseline was 744 passed / 16 skipped before Task 1's 2 new test cases). If the actual number differs, reconcile the difference before proceeding.

- [ ] **Step 4: Confirm `data/cartographer_thresholds.json` is untouched**

Run: `git status --short data/cartographer_thresholds.json`

Expected: **no output** (empty).

- [ ] **Step 5: Confirm the 14-pair bar pin, 45-pair eval set, and scan gold eval are unaffected**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_eval_pairs.py tests/test_cartographer_scan_gold.py tests/test_scan_precision.py -v`

Expected: **26 passed** (7 + 8 + 11, matching the counts already established during sub-project F).

- [ ] **Step 6: Commit**

```bash
git add tests/test_cartographer_stance_node5.py
git commit -m "test_cartographer_stance_node5: re-pin the false-positive set after the roughly fix"
```

---

### Task 3: Measure the real-model numbers and record the spec's Results

**Files:**
- Modify: `docs/superpowers/specs/2026-07-27-quantity-hedge-word-stopword-design.md` (append a `## Results` section, update the `**Status:**` line)

**Interfaces:** none — this task consumes Tasks 1 and 2, and produces the finished spec document. Nothing downstream depends on this task's output within this plan.

- [ ] **Step 1: Run the end-to-end node5 measurement**

Run: `venv/Scripts/python.exe scripts/eval_node5_stance.py`

Predicted (stated as a prediction to confirm by reading the actual output, not assumed): `P_all` 0.857 → **0.923** (tp 12, fp 1). `P` (within-event) stays **1.000**, `R` stays **1.000** — neither residual false positive was ever within-event. The "false positives by channel" section is predicted to show only `n5_doc_023:0<->n5_doc_026:0` (`band` or `nli` depending on which channel reaches it first with real models — record whichever the script actually prints) remaining.

- [ ] **Step 2: Run the held-out-40 calibration score**

Run: `venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py --score-only`

Predicted: unchanged at **0.725** — none of the 9 same-story pairs in that 40-pair set were measured (during sub-project F) to have a stance value affected by this change, but this must be checked directly, not inferred. Record the actual number regardless of which way the prediction lands.

- [ ] **Step 3: Update the spec document**

In `docs/superpowers/specs/2026-07-27-quantity-hedge-word-stopword-design.md`, change:
```
**Status:** Proposed
```
to:
```
**Status:** IMPLEMENTED and measured 2026-07-27.
```

Append a `## Results (measured 2026-07-27)` section reporting, plainly and
without rounding up: the actual `eval_node5_stance.py` output (`P`/`P_all`/`R`,
the false-positive list by channel), the actual `--score-only` number
compared to 0.725, the actual full-suite count from Task 2, and — if any
actual number differs from this plan's predictions — the difference and the
most likely reason. Match the reporting style already used in this track's
other specs (report every number whichever way it moves; do not silently
adjust a prediction to match a result without saying so).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-27-quantity-hedge-word-stopword-design.md
git commit -m "Results: roughly stopword fix removes 023<->024, measured on node5 and the held-out-40 score"
```
