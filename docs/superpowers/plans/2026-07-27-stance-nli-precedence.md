# Stance-NLI Precedence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the fact-aligned stance channel disagrees with a firing NLI channel on a same-story pair, make the stance channel win instead of NLI.

**Architecture:** One function (`classify_relation` in `gin/cartographer/combined.py`) gains two local booleans and one new early return in its existing `same_story` branch. No other function's signature, control flow, or call sites change.

**Tech Stack:** Python, pytest. Real models (`sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-xsmall`) only for the measurement task; everything else is model-free via injected `cos`/`p_contra`.

## Global Constraints

- `data/cartographer_thresholds.json` must be byte-identical after this work — no threshold value changes.
- No new tunable threshold or confidence-blend introduced anywhere.
- The veto applies only when `same_story is True` — never when `same_story` is `False` or `None`.
- `stance is None` must continue to reproduce pre-2026-07-26 behavior exactly (the existing byte-for-byte contract).
- Spec: `docs/superpowers/specs/2026-07-27-stance-nli-precedence-design.md`.

---

### Task 1: Implement the disagreement veto in `classify_relation`

**Files:**
- Modify: `gin/cartographer/combined.py:78-142` (function body + docstring), `gin/cartographer/combined.py:268-273` (inline comment in `type_relation`)
- Test: `tests/test_cartographer_stance_branch.py`

**Interfaces:**
- Consumes: `classify_relation(cos: float, p_contra: float, t: Thresholds, *, same_story: Optional[bool] = None, stance: Optional[str] = None) -> tuple[Relation, str]` — signature unchanged.
- Produces: same signature. New reachable outcome `(Relation.RELATED_UNTYPED, "abstain")` from a precondition (`same_story=True`, `stance` decisive and not `"conflict"`, `p_contra >= contra_threshold`) that previously produced `(Relation.CONTRADICTS, "nli")`. No other caller (`CombinedRelationProposer.type_relation`, `.assess_pair`) needs any change — both already thread `stance` and `p_contra` through unconditionally.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cartographer_stance_branch.py`, replace this function (currently at lines 61–65):

```python
def test_nli_still_outranks_the_stance_branch():
    # The NLI channel owns the legal/securities register it was calibrated on;
    # stance evidence does not override a confident propositional contradiction.
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance="partial")
    assert (relation, channel) == (Relation.CONTRADICTS, "nli")
```

with:

```python
def test_stance_disagreement_overrules_a_firing_nli():
    # Measured 2026-07-27: on the 24 node5 labels, every pair where NLI fires
    # and stance disagrees is wrong (a supersedes and a corroborates, both
    # scored CONTRADICTS by NLI alone); every pair where they agree is right.
    # The stance channel now wins on disagreement instead of NLI.
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance="partial")
    assert (relation, channel) == (Relation.RELATED_UNTYPED, "abstain")


from gin.cartographer.quantity import UNALIGNED


@pytest.mark.parametrize("stance", ["revision", "partial", "agreement", UNALIGNED])
def test_any_disagreeing_stance_overrules_a_firing_nli(stance):
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance=stance)
    assert (relation, channel) == (Relation.RELATED_UNTYPED, "abstain")


def test_agreeing_stance_leaves_nli_priority_untouched():
    # stance == "conflict" agrees with a firing NLI, so the veto never
    # applies -- channel stays "nli", not "stance", matching today's
    # attribution exactly.
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance="conflict")
    assert (relation, channel) == (Relation.CONTRADICTS, "nli")
```

(The `UNALIGNED` import is placed at module level here, right before its first use, matching this file's existing convention of importing near point-of-use rather than clustering everything at the top — see the `CombinedRelationProposer` import already placed mid-file.)

- [ ] **Step 2: Run tests to verify the expected failures**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_stance_branch.py -v`

Expected: **5 failed, 20 passed** (25 collected). The 5 failures are `test_stance_disagreement_overrules_a_firing_nli` and all 4 parametrized cases of `test_any_disagreeing_stance_overrules_a_firing_nli` — each currently gets `(CONTRADICTS, "nli")` from the unmodified code but now asserts `(RELATED_UNTYPED, "abstain")`. `test_agreeing_stance_leaves_nli_priority_untouched` is expected to **already pass** — it pins behavior the old code already has; treat an unexpected failure there as a sign the test itself is wrong, not a reason to change the implementation to match.

- [ ] **Step 3: Implement the veto**

In `gin/cartographer/combined.py`, replace the function body (currently lines 128–142):

```python
    if cos < t.gate_floor:
        return Relation.UNRELATED, "gate"
    if p_contra >= t.contra_threshold and same_story is not False:
        return Relation.CONTRADICTS, "nli"
    if same_story:
        if stance is None:
            return Relation.CONTRADICTS, "band"
        if stance == "conflict":
            return Relation.CONTRADICTS, "stance"
        if stance == "agreement" and cos >= t.corroborate_ceiling:
            return Relation.CORROBORATES, "band"
        return Relation.RELATED_UNTYPED, "abstain"
    if cos >= t.corroborate_ceiling:
        return Relation.CORROBORATES, "band"
    return Relation.RELATED_UNTYPED, "band"
```

with:

```python
    if cos < t.gate_floor:
        return Relation.UNRELATED, "gate"
    nli_contradicts = p_contra >= t.contra_threshold and same_story is not False
    stance_disagrees = same_story and stance is not None and stance != "conflict"
    if nli_contradicts and not stance_disagrees:
        return Relation.CONTRADICTS, "nli"
    if same_story:
        if stance is None:
            return Relation.CONTRADICTS, "band"
        if stance == "conflict":
            return Relation.CONTRADICTS, "stance"
        if nli_contradicts:
            return Relation.RELATED_UNTYPED, "abstain"
        if stance == "agreement" and cos >= t.corroborate_ceiling:
            return Relation.CORROBORATES, "band"
        return Relation.RELATED_UNTYPED, "abstain"
    if cos >= t.corroborate_ceiling:
        return Relation.CORROBORATES, "band"
    return Relation.RELATED_UNTYPED, "band"
```

- [ ] **Step 4: Correct the now-stale docstring**

In the same function, the docstring currently ends with (immediately before the closing `"""`):

```python
    Why the arm needed evidence: measured on the 24 node5 labels, the
    unconditional "same_story -> CONTRADICTS" scored precision 12/24. The NLI
    channel cannot replace it (its two highest p_contra in that set are a
    corroborates and a supersedes), so the evidence has to be per-fact. The
    fallback is ABSTENTION rather than corroboration: a wrong CONTRADICTS edge
    costs a knowledge graph more than a missing one.
    """
```

Replace with:

```python
    Why the arm needed evidence: measured on the 24 node5 labels, the
    unconditional "same_story -> CONTRADICTS" scored precision 12/24. The NLI
    channel cannot replace it (its two highest p_contra in that set are a
    corroborates and a supersedes), so the evidence has to be per-fact. The
    fallback is ABSTENTION rather than corroboration: a wrong CONTRADICTS edge
    costs a knowledge graph more than a missing one.

    When ``same_story`` is True and ``stance`` renders a decisive verdict that
    disagrees with a firing NLI channel (stance is not None and not
    "conflict", while p_contra >= contra_threshold), the stance channel wins
    and the pair abstains to RELATED_UNTYPED instead of typing CONTRADICTS on
    the NLI channel. Measured 2026-07-27 on the 24 node5 labels: every pair
    where NLI fires and stance disagrees is wrong (a supersedes and a
    corroborates, both scored CONTRADICTS by NLI alone); every pair where NLI
    fires and stance agrees (stance == "conflict") is right. Agreement with a
    firing NLI, or stance is None, leaves this function's prior behavior
    exactly as it was -- the veto only ever fires on genuine disagreement.
    """
```

- [ ] **Step 5: Correct the now-stale inline comment in `type_relation`**

Still in `gin/cartographer/combined.py`, `type_relation` currently has:

```python
        p_contra = self._p_contra(a_text, b_text)
        # Computed whenever `story` is true, even though classify_relation may
        # have the NLI channel decide the pair first (p_contra >= contra_thresh
        # takes priority over the stance arm below). In that case ev["stance"]
        # still gets set on a channel: "nli" result -- the stance evidence was
        # examined but discarded, not consulted for the decision.
        stance = (
```

Replace the comment with:

```python
        p_contra = self._p_contra(a_text, b_text)
        # Computed whenever `story` is true. When stance is None or agrees
        # with NLI (stance == "conflict"), NLI keeps priority and this may
        # still return channel "nli" -- ev["stance"] is recorded whenever
        # stance is not None, even on an "nli" result, since it was examined
        # even when not decisive. When stance disagrees (not None, not
        # "conflict") it overrules a firing NLI, and classify_relation
        # returns channel "abstain" instead.
        stance = (
```

- [ ] **Step 6: Run tests to verify they now pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_stance_branch.py -v`

Expected: **25 passed**.

- [ ] **Step 7: Commit**

```bash
git add gin/cartographer/combined.py tests/test_cartographer_stance_branch.py
git commit -m "classify_relation: stance channel overrules NLI on same-story disagreement"
```

---

### Task 2: Confirm no regression on the frozen surfaces

**Files:** none modified — this task only runs and records; there is nothing to commit.

**Interfaces:**
- Consumes: the `classify_relation` change from Task 1.
- Produces: a pass/fail confirmation consumed by Task 3 (Task 3 assumes this task's checks are green before recording headline numbers).

- [ ] **Step 1: Run the full suite**

Run: `venv/Scripts/python.exe -m pytest -q`

Expected: **742 passed, 16 skipped** (baseline was 737 passed / 16 skipped before Task 1's 5 new test cases; if the actual number differs from 742, reconcile the difference against Task 1's test count before proceeding — do not proceed past a discrepancy without understanding it).

If anything outside `tests/test_cartographer_stance_branch.py` fails — for
example a node1–4 or 45-pair-set contradiction that was previously caught
only by NLI and turns out to have a disagreeing, non-`None` stance —
**investigate before reverting**, per the spec's Failure modes table: if
stance is right and NLI was catching it for the wrong reason, that is this
fix working correctly elsewhere; if stance is wrong, that is new information
about the stance channel's coverage outside node5 and needs its own writeup,
not a silent special case.

- [ ] **Step 2: Confirm `data/cartographer_thresholds.json` is untouched**

Run: `git status --short data/cartographer_thresholds.json`

Expected: **no output** (empty). The spec requires this file byte-identical;
no task in this plan writes it, and this step confirms that rather than
assuming it.

- [ ] **Step 3: Explicitly confirm the 14-pair escalation bar pin**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_eval_pairs.py -v`

Expected: **7 passed** (unchanged from before Task 1 — this file was confirmed during design to never set `same_story` or `stance`, so it is structurally unreachable by this change; this step exists to confirm that structural argument, not because the mechanism suggests risk).

- [ ] **Step 4: Explicitly confirm the isolated stance-arm test is unaffected**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_stance_node5.py -v`

Expected: **3 passed**, with the same pinned false-positive set as before
(`test_the_two_residual_false_positives_are_the_pre_registered_ones` must
still report exactly `n5_doc_023:0<->n5_doc_024:0` as `"stance"` and
`n5_doc_023:0<->n5_doc_026:0` as `"band"`). This test injects `p_contra=0.05`
for every pair, always below `contra_threshold`, so `nli_contradicts` is
always `False` and Task 1's new branch is never reached — if this test's
result changed at all, stop and re-examine the reasoning in the spec's
"Failure modes" table before continuing to Task 3.

---

### Task 3: Measure the real-model numbers and record the spec's Results

**Files:**
- Modify: `docs/superpowers/specs/2026-07-27-stance-nli-precedence-design.md` (append a `## Results` section, update the `**Status:**` line)

**Interfaces:** none — this task consumes Task 1's implementation and Task 2's green confirmation, and produces the finished spec document. Nothing downstream depends on this task's output within this plan.

- [ ] **Step 1: Run the end-to-end node5 measurement**

Run: `venv/Scripts/python.exe scripts/eval_node5_stance.py`

Predicted (stated as a prediction to confirm by reading the actual output, not assumed): within-event `P` 0.857 → **1.000** (tp 12, fp 0), `P_all` 0.750 → **0.857** (tp 12, fp 2), `R` stays **1.000**. Dev/held-out split predicted to both reach `P` 1.000 (currently 0.900 / 0.750). The "false positives by channel" section is predicted to show only `n5_doc_023:0<->n5_doc_024:0` (stance) and `n5_doc_023:0<->n5_doc_026:0` (nli) remaining.

- [ ] **Step 2: Run the held-out-40 calibration score**

Run: `venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py --score-only`

Predicted: unchanged at **0.725** (the post-E baseline) — none of the 9 same-story pairs in that 40-pair set had a stance value among the disagreement cases this change acts on, per the spec's reasoning. Record the actual number regardless of which way the prediction lands.

- [ ] **Step 3: Update the spec document**

In `docs/superpowers/specs/2026-07-27-stance-nli-precedence-design.md`:

Change the status line from:
```
**Status:** Proposed
```
to:
```
**Status:** IMPLEMENTED and measured 2026-07-27.
```

Append a `## Results (measured 2026-07-27)` section reporting, plainly and
without rounding up: the actual `eval_node5_stance.py` output (P/P_all/R,
dev/held-out split, the false-positive list by channel), the actual
`--score-only` number compared to 0.725, the actual full-suite count from
Task 2, and — if any actual number differs from this plan's predictions —
the difference and the most likely reason, matching the style already used
throughout this track's specs (report every number whichever way it moves;
do not silently adjust a prediction to match a result without saying so).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-27-stance-nli-precedence-design.md
git commit -m "Results: stance channel overrules NLI, measured on node5 and the held-out-40 score"
```
