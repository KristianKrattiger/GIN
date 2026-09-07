# GIN_14_Assay Vertical — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GIN_14_Assay` to the GIN series — one conceptual-register document naming the portable constraint core shared by GIN and Receipts — and wire it into the series index and backlinks.

**Architecture:** This is a documentation change, not code. Task 1 writes the new document with its full prose supplied here verbatim. Tasks 2–4 are small edits to existing files (the Reader index, two Related lines, the Receipts README Lineage section). Each task ends with a grep-based verification and a commit. There are no unit tests; the "test" for a prose file is that the required structural elements and cross-links are present and that Obsidian wikilinks resolve to real files.

**Tech Stack:** Markdown, Obsidian-style `[[wikilink]]` cross-references, git. Two repositories: `C:\Users\krist\Projects\gin\GIN` (Tasks 1–3) and `C:\Users\krist\Projects\receipts` (Task 4).

## Global Constraints

- **Register:** conceptual. The document may discuss problems; it must not present itself as a specification. Any request/response shape is marked *illustrative, not normative* — this phrasing appears in the doc at least once (§2).
- **Frontmatter fields, exact:** `tags`, `updated`, `version`, `status`, `register` — matching the shape in `docs/GIN_00_Reader.md` and `docs/GIN_12_Ecosystem_Licensing.md`.
- **`updated` value:** `2026-09-06` (verbatim).
- **`version` value:** `0.4-preliminary` (verbatim, matches the rest of the series).
- **`register` value:** `conceptual` (verbatim).
- **`status` value:** `working draft` (verbatim).
- **Every document ends with** a `## Related` line of `·`-separated wikilinks, then `## Back to Vault` and `[[HOME]]`.
- **Line length:** wrap prose at roughly 80 columns, matching the surrounding series files.
- **Commit trailer:** end every commit message with
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
- **Source of truth for content:** the design spec at `docs/superpowers/specs/2026-09-06-assay-vertical-design.md`. If this plan and the spec disagree, stop and ask.

---

### Task 1: Write `docs/GIN_14_Assay.md`

**Files:**
- Create: `C:\Users\krist\Projects\gin\GIN\docs\GIN_14_Assay.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a file at `docs/GIN_14_Assay.md` containing an H1 `# GIN 14 — Assay`, a thesis blockquote, nine `##`-level sections numbered `1 —` through `9 —`, a `## Related` line, and a `## Back to Vault` section. Later tasks link to it as `[[GIN_14_Assay]]`.

- [ ] **Step 1: Create the file with the exact content below**

Write this to `C:\Users\krist\Projects\gin\GIN\docs\GIN_14_Assay.md` verbatim:

````markdown
---
tags: [GIN, research, assay, grounding, refusal, portable]
updated: 2026-09-06
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN 14 — Assay

> The Assay is grounded confidence and loud refusal with everything situational
> removed. Hand it a set of documents and a query; it returns a cited answer, a
> divergence report, or a refusal — and never a fluent claim it cannot anchor.
> GIN is the assay office at scale; Receipts is a field assay of the live web;
> the Assay is the portable kit both are built from.

---

## 1 — Three verticals, one constraint

The GIN series has, until now, described one system. This document adds a second
and a third thing that are the same system at a different scale, and names what
all three share.

The shared claim is small: **every emitted claim traces to a span that provably
exists in the supplied corpus, and where nothing grounds an answer the system
marks the silence instead of filling it.** Grounded confidence, and loud refusal.
Everything else in the series is downstream of that sentence.

GIN is that claim scaled. Around the constraint it wraps federation
([[GIN_03_Node_Identity]]), a governing institution ([[GIN_10_Epistemic_Council]]),
constrained decoding that enforces grounding geometrically ([[GIN_04_SEAR]]), and a
tiered corpus that ranges from cold content-addressed blobs to hot vector search.
The wrapping is most of the engineering and all of the politics.

Receipts is that claim shipped thin. It points a fan of cloud browsers at one
vendor's marketing and at independent writing about it, lets a model propose which
claims corroborate or contradict which, and then runs a deterministic gate that
re-derives every quoted span from the bytes that were actually fetched and
discards anything it cannot find. No federation, no council, no local model — a
live-web claim ledger and nothing more.

Strip the situational layers from either one — the federation and governance from
GIN, the browsers and the live web from Receipts — and the same object is left in
both hands: a function from a corpus and a query to a grounded answer, a surfaced
disagreement, or a refusal. That object is the **Assay**. GIN is the assay office;
Receipts is a field assay; this document is about the kit.

## 2 — The operation

An assay takes a sample and a question and returns a reading or a rejection. The
Assay takes a set of documents and a query and returns one of three things:

- a **grounded answer** — a claim with every load-bearing span in it citing a
  location in the supplied corpus;
- a **divergence report** — two or more grounded claims that the corpus supports
  and that contradict each other, presented together with their citations rather
  than reconciled;
- a **refusal** — a statement that the corpus does not support an answer, carrying
  a reason and the spans that came closest.

Every response carries a **confidence**. Refusal is not a fourth outcome bolted
on; it is the band below a threshold the caller sets. A caller who wants only
near-certain answers sets the bar high and receives more refusals; a caller
triaging leads sets it low. The spans that were rejected for falling under the bar
stay attached to the refusal, because a refusal that shows its work — here are the
three passages we considered and the score each earned — is worth more than a bare
no, and it is what lets the guarantee be checked rather than trusted.

The response also carries an **audit line**: how many candidate claims were
proposed, how many admitted, how many denied and under which reason. Receipts
publishes exactly this line on every ledger, and it is the single thing that
converts "we verified our quotes" from an assurance into something a reader can
recount.

The request and response shapes in this section are *illustrative, not normative*.
This document is in the conceptual register; it draws the boundary, it does not
specify a wire format. A normative specification waits for an implementation.

## 3 — Agnostic to model, source, and task

The name's adjective was "agnostic" before it was "Assay", and three axes are what
it meant.

**Model.** The grounding check is post-hoc: generate a claim by whatever means,
then verify each span against the corpus by exact match. Nothing in that loop
needs control over decoding, so the Assay runs on a frontier API model, a local
open-weights model, or a model not yet released. This is the clean line between the
Assay and [[GIN_04_SEAR]], which enforces the same guarantee *during* generation
and therefore needs a model it can open up.

**Source.** The Assay never fetches. The caller hands it bytes — from a web crawl,
a database, an upload, a git repository, a disclosure packet — and the Assay treats
all of them the same way. Receipts owns cloud browsers because reading hostile
pages is its whole problem; the Assay has no opinion about where a document was
before it arrived.

**Task.** Question answering, claim checking, extraction, and cited summarisation
are one operation seen from different sides: a query and a corpus resolve to
grounded spans or to a refusal. The Assay does not have four modes for these; it
has one.

Domain-agnosticism is not a fourth axis because it is already a house rule. GIN
runs no per-region code; Receipts adds a domain with a JSON file and no engine
change. The Assay inherits that, and would be remarked on only if it broke it.

## 4 — The loud refusal

Half the thesis is the refusal, so it has a vocabulary. A refusal carries one of:

- `NO_GROUNDING` — nothing in the corpus speaks to the query at all;
- `BELOW_THRESHOLD` — the corpus speaks to it, but no candidate span clears the
  caller's confidence bar;
- `QUERY_UNGROUNDABLE` — the question is not the kind a corpus can answer: a
  prediction, an opinion, a counterfactual with no basis in the documents;
- `CORPUS_INSUFFICIENT` — the corpus is empty, or so thin that neither an answer
  nor an honest refusal can be built on it. This is the guard against a result
  that looks like a finding and is not; Receipts refuses a run before it spends
  money when a plan names only one side, for the same reason;
- `CONFLICTING_UNRESOLVABLE` — the grounded spans contradict each other and the
  caller asked for a single answer. By default this is not a refusal at all (see
  §5); it becomes one only when the caller has opted out of divergence reports.

The reason codes are a draft. What is not a draft is that a refusal names its
reason: an unexplained refusal is as opaque as an ungrounded answer, and opacity
on either side is the failure the Assay exists to prevent.

## 5 — Divergence is an outcome, not a failure

When two passages in the corpus are each well grounded and say opposite things,
three moves are available: pick one, average them, or refuse. All three destroy
information. Picking one hides a real citation; averaging invents a claim that no
passage makes; refusing throws away two findings that are individually sound.

The Assay takes the fourth option and returns both, cited, marked as in conflict.
This is [[GIN_02_Productive_Divergence|productive divergence]] at the smallest
possible scale — not a federation of nodes disagreeing across a network, just two
spans in one bundle of documents — and it is the reason the Assay is a faithful
core for Receipts rather than a cousin of it. Receipts' entire divergent section
is this outcome.

A caller who genuinely needs one answer can ask for convergent-only behaviour and
receive `CONFLICTING_UNRESOLVABLE` instead. That is a legitimate request — some
consumers cannot act on a pair — but it is the exception, and the default is to
surface the disagreement.

What the Assay does not do is judge the disagreement. Whether a contradiction is a
legitimate difference of situated perspective or a sign that one source is simply
wrong is the validity question, and it belongs to [[GIN_07_Governance_Validity]]
and the institution that holds it. The Assay reports that the corpus disagrees
with itself; it does not rule on the merits.

## 6 — What the Assay is not

The Assay does not fetch, crawl, or render. It does not federate, sync, or hold a
network identity. It has no council, no promotion gate, no quarantine register, no
notion of a claim graduating from provisional to established. It does not persist
anything between calls. It does not govern divergence.

Every one of those is deliberate, and together they are the definition. GIN is the
Assay plus the machinery that makes grounded reasoning survivable at global scale
and accountable to someone. Receipts is the Assay plus the machinery for reading
sources that fight back. The Assay is what is common to both once the machinery is
removed — and a thing that portable is a thing a third party can pick up without
adopting a federation or a browser fleet.

## 7 — Who it is for

Three readers.

An **agent developer** who wants their tool to cite or decline, and has no
interest in running nodes. Today the options are a full retrieval stack they must
trust or a constrained-decoding setup they must host. The Assay is the guarantee
without either.

**Receipts itself**, rebuilt. The propose–admit–render pipeline in Receipts is the
Assay with a browser fan bolted to its front and a claim-ledger renderer bolted to
its back. Factoring the middle out and depending on it would shrink Receipts to
the parts that are actually about the live web.

A **GIN node**, for the paths where constrained decoding is too heavy or the base
model is not one SEAR can open. The Assay and SEAR enforce the same contract; a
node can hold both and choose per query.

## 8 — SEAR and the Assay are one contract

[[GIN_04_SEAR]] constrains a model so that it can only emit spans that occur in the
corpus; grounding is true by the time a token exists. The Assay lets the model
emit anything and then deletes what does not verify; grounding is true by the time
a response is returned. The enforcement point is different — during decoding
versus after it — and the cost profile is different, but the guarantee delivered
to the reader is identical: no claim without a span.

So the Assay is not a lightweight SEAR. It is the contract stated independently of
how it is kept, with SEAR as the enforcement that needs an open model and post-hoc
verification as the enforcement that does not. Naming the contract on its own is
what lets the two implementations be compared instead of conflated.

## 9 — Open seams

Held open, in the series' habit:

- **What is a span, when the answer is synthesised.** Exact-substring
  verification is clean for extractive answers and unclear for a sentence that
  fairly paraphrases three passages. Receipts sidesteps this by only ever quoting;
  a general Assay has to decide whether paraphrase is admissible and, if so, how it
  is checked.
- **Confidence across models.** A caller-set threshold assumes the confidence
  number means the same thing from one model to the next. It does not, without
  calibration, and calibration is work the conceptual register cannot wave away.
- **Divergence detection may need a model.** Deciding that two spans contradict is
  itself an inference. If it takes a model call, the detection step is not
  model-agnostic even though the verification step is, and the doc should not claim
  otherwise.
- **`CORPUS_INSUFFICIENT` is a judgement.** The line between "thin but answerable"
  and "too thin to be honest" is not a measurement. Receipts draws it with a
  single-role guard; a general Assay needs a defensible rule, and any rule it picks
  is a small act of governance in a thing that claims to have none.

## Related

[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_04_SEAR]] · [[GIN_07_Governance_Validity]] · [[GIN_09_Agentic_Layer]] · [[GIN_STRAT_00_Strategy_Register]]

## Back to Vault

[[HOME]]
````

- [ ] **Step 2: Verify structure is present**

Run from `C:\Users\krist\Projects\gin\GIN`:

```bash
grep -nE '^(# GIN 14 — Assay|## [1-9] —|## Related|## Back to Vault)' docs/GIN_14_Assay.md
```

Expected: 12 lines — the H1, sections `1 —` through `9 —` (nine lines), `## Related`, `## Back to Vault`.

- [ ] **Step 3: Verify frontmatter and the non-normative disclaimer**

Run:

```bash
grep -nE 'register: conceptual|updated: 2026-09-06|version: 0.4-preliminary|illustrative, not normative' docs/GIN_14_Assay.md
```

Expected: 4 matches — three frontmatter fields and the §2 disclaimer.

- [ ] **Step 4: Verify no placeholder text**

Run:

```bash
grep -nE 'TBD|TODO|FIXME|XXX|\bfill in\b|lorem ipsum' docs/GIN_14_Assay.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 5: Verify every outbound wikilink resolves to a real file**

Run from `C:\Users\krist\Projects\gin\GIN`:

```bash
for L in GIN_00_Reader GIN_02_Productive_Divergence GIN_03_Node_Identity GIN_04_SEAR GIN_07_Governance_Validity GIN_09_Agentic_Layer GIN_10_Epistemic_Council GIN_STRAT_00_Strategy_Register; do
  test -f "docs/$L.md" && echo "ok  $L" || echo "MISSING  $L"
done
```

Expected: eight `ok` lines, no `MISSING`. (`[[HOME]]` is a vault-level note outside `docs/` and is not checked here — it is the standard footer used by every file in `docs/`.)

- [ ] **Step 6: Commit**

```bash
git add docs/GIN_14_Assay.md
git commit -m "$(cat <<'EOF'
GIN_14: Assay — the portable constraint vertical

Names the core shared by GIN and Receipts: grounded confidence and loud
refusal with the federation, governance, and live-web machinery removed.
Documents + query in; cited answer, divergence report, or refusal out.
Conceptual register; the API shape is illustrative, not normative.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire `GIN_14_Assay` into `docs/GIN_00_Reader.md`

**Files:**
- Modify: `C:\Users\krist\Projects\gin\GIN\docs\GIN_00_Reader.md` (document-set table, ~line 61; "How to read this set", ~line 73)

**Interfaces:**
- Consumes: `[[GIN_14_Assay]]` from Task 1.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Add the table row**

In `docs/GIN_00_Reader.md`, find the row for `[[GIN_13_Temporal_Sensor_Grounding]]` in the "document set" table. Immediately **after** that row and **before** the `[[GIN_ENG_00_Engineering_Register]]` row, insert:

```markdown
| [[GIN_14_Assay]] | The portable constraint core — corpus + query → cited answer / divergence report / refusal; agnostic to model, source, and task | conceptual |
```

- [ ] **Step 2: Add a sentence to "How to read this set"**

In the same file, in the "How to read this set" section, find the sentence that ends `The mechanism papers (03–06) each show how one component expresses the principle.` Immediately after that sentence, in the same paragraph, insert:

```markdown
[[GIN_14_Assay]] runs the other way from the mechanism papers: it is the constraint with the architecture taken off — the portable core that GIN scales and Receipts ships thin — and can be read any time after [[GIN_04_SEAR]].
```

- [ ] **Step 3: Verify both edits landed**

Run from `C:\Users\krist\Projects\gin\GIN`:

```bash
grep -nE 'GIN_14_Assay' docs/GIN_00_Reader.md
```

Expected: 2 matches — one in the table, one in the "How to read this set" paragraph.

- [ ] **Step 4: Verify the table row is well-formed and correctly placed**

Run:

```bash
grep -nE 'GIN_13_Temporal_Sensor_Grounding|GIN_14_Assay|GIN_ENG_00_Engineering_Register' docs/GIN_00_Reader.md | head -5
```

Expected: the three line numbers are consecutive and in this order (13, 14, ENG_00), confirming GIN_14 sits between them.

- [ ] **Step 5: Commit**

```bash
git add docs/GIN_00_Reader.md
git commit -m "$(cat <<'EOF'
GIN_00: index GIN_14_Assay in the document set and reading order

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `[[GIN_14_Assay]]` backlinks to SEAR and Governance

**Files:**
- Modify: `C:\Users\krist\Projects\gin\GIN\docs\GIN_04_SEAR.md` (Related line, currently line 76)
- Modify: `C:\Users\krist\Projects\gin\GIN\docs\GIN_07_Governance_Validity.md` (Related line, currently line 78)

**Interfaces:**
- Consumes: `[[GIN_14_Assay]]` from Task 1.
- Produces: nothing.

- [ ] **Step 1: Edit the SEAR Related line**

In `docs/GIN_04_SEAR.md`, the Related line currently reads exactly:

```markdown
[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_03_Node_Identity]] · [[GIN_07_Governance_Validity]] · [[GIN_09_Agentic_Layer]] · [[GIN_13_Temporal_Sensor_Grounding]] · [[CALICHE_INDEX]]
```

Replace it with (insert `[[GIN_14_Assay]]` before `[[CALICHE_INDEX]]`):

```markdown
[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_03_Node_Identity]] · [[GIN_07_Governance_Validity]] · [[GIN_09_Agentic_Layer]] · [[GIN_13_Temporal_Sensor_Grounding]] · [[GIN_14_Assay]] · [[CALICHE_INDEX]]
```

- [ ] **Step 2: Edit the Governance Related line**

In `docs/GIN_07_Governance_Validity.md`, the Related line currently reads exactly:

```markdown
[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_04_SEAR]] · [[GIN_09_Agentic_Layer]] · [[GIN_10_Epistemic_Council]] · [[GIN_08_Adversarial_Analysis]]
```

Replace it with (append `[[GIN_14_Assay]]` at the end):

```markdown
[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_04_SEAR]] · [[GIN_09_Agentic_Layer]] · [[GIN_10_Epistemic_Council]] · [[GIN_08_Adversarial_Analysis]] · [[GIN_14_Assay]]
```

- [ ] **Step 3: Verify both Related lines**

Run from `C:\Users\krist\Projects\gin\GIN`:

```bash
grep -nE 'GIN_14_Assay' docs/GIN_04_SEAR.md docs/GIN_07_Governance_Validity.md
```

Expected: exactly one match in each file, both on the `## Related` line (the line following `## Related`).

- [ ] **Step 4: Verify nothing else in those files changed**

Run:

```bash
git diff --stat docs/GIN_04_SEAR.md docs/GIN_07_Governance_Validity.md
```

Expected: each file shows `1 insertion(+), 1 deletion(-)` (the single Related line rewritten).

- [ ] **Step 5: Commit**

```bash
git add docs/GIN_04_SEAR.md docs/GIN_07_Governance_Validity.md
git commit -m "$(cat <<'EOF'
GIN_04, GIN_07: backlink GIN_14_Assay from SEAR and Governance

SEAR and post-hoc verification are two enforcements of one contract; the
validity question the Assay defers to lives in GIN_07.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Point the Receipts README Lineage section at `GIN_14_Assay`

**Files:**
- Modify: `C:\Users\krist\Projects\receipts\README.md` (the `## Lineage` section)

**Interfaces:**
- Consumes: the concept name `GIN_14_Assay` (no wikilink — the Receipts repo is not an Obsidian vault; reference it as plain text, the way the section already refers to `GIN` in bold without a link).
- Produces: nothing.

- [ ] **Step 1: Read the current Lineage section**

Run from `C:\Users\krist\Projects\receipts`:

```bash
grep -nA 30 '^## Lineage' README.md
```

Note the three carried-over bullet points ("Productive divergence", "Exact attribution by construction", "Layer separation") and the paragraph that begins `Deliberately *not* ported:`.

- [ ] **Step 2: Insert the Assay pointer**

In `README.md`, immediately **after** the paragraph that begins `Deliberately *not* ported:` (the paragraph ending `...the class it failed on.`) and **before** the `Full design:` paragraph, insert this new paragraph:

```markdown
Those three ideas have a name in the series now. `GIN_14_Assay` factors them out
as the **Assay** — the same constraint stated without the federation: a set of
documents and a query in; a cited answer, a divergence report, or a refusal out.
Receipts is a field instance of that contract, pointed at the live web and the
sources that refuse automation. GIN is the contract scaled and governed.
```

- [ ] **Step 3: Verify the edit landed in the right place**

Run:

```bash
grep -nE 'GIN_14_Assay|Deliberately \*not\* ported|^Full design:' README.md
```

Expected: three matches, in this order — `Deliberately *not* ported`, then `GIN_14_Assay`, then `Full design:`.

- [ ] **Step 4: Verify the rest of the README is unchanged**

Run:

```bash
git diff --stat README.md
```

Expected: `1 file changed, 6 insertions(+)` (the new paragraph plus its surrounding blank line), no deletions.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): name the ported constraint — GIN_14_Assay

The three ideas in Lineage are the Assay: the GIN/Receipts constraint
stated without the federation. Receipts is a field instance of it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**1. Spec coverage.**

| Spec element | Task |
|---|---|
| New file `docs/GIN_14_Assay.md`, conceptual register | Task 1 |
| Frontmatter shape + exact values | Task 1, Global Constraints + Step 3 |
| Thesis blockquote (assay office / field assay / kit) | Task 1, Step 1 |
| §1 subtractive map (spine; GIN wraps; Receipts thins; Assay is residue) | Task 1 §1 |
| §2 three outcomes + graded confidence + caller-set threshold + near-miss spans + audit line + non-normative disclaimer | Task 1 §2 |
| §3 model / source / task agnosticism; domain inherited | Task 1 §3 |
| §4 five refusal reason codes | Task 1 §4 |
| §5 divergence as outcome #3; convergent-only → `CONFLICTING_UNRESOLVABLE`; validity deferred to GIN_07 | Task 1 §5 |
| §6 what it is not (no fetch/federation/council/promotion/persistence/divergence-governance) | Task 1 §6 |
| §7 three readers (agent developer, Receipts rebuilt, GIN node) | Task 1 §7 |
| §8 SEAR and Assay are one contract, two enforcement points | Task 1 §8 |
| §9 four open seams (span-when-synthesised, calibration, divergence-detection needs a model, `CORPUS_INSUFFICIENT` is a judgement) | Task 1 §9 |
| Related line + Back to Vault | Task 1, Step 1 + Step 2 |
| Ripple: `GIN_00_Reader` table row + reading-order sentence | Task 2 |
| Ripple: `[[GIN_14_Assay]]` in `GIN_04_SEAR` Related | Task 3 |
| Ripple: `[[GIN_14_Assay]]` in `GIN_07_Governance_Validity` Related | Task 3 |
| Ripple: Receipts README Lineage pointer | Task 4 |
| Out of scope: no software spec, no new register, no engineering/strategy edits, no ruling on divergence legitimacy | honored — no such task exists |

No gaps.

**2. Placeholder scan.** The reason codes and API shape are described in the plan as "a draft" / "illustrative" — this is deliberate content from the spec (the doc is conceptual and says so), not a plan placeholder. No `TBD`/`TODO`/"implement later"/"add error handling"/"similar to Task N" anywhere. Task 1 carries the full prose; Tasks 2–4 carry exact before/after text. Clean.

**3. Type consistency.** The cross-reference name is `GIN_14_Assay` in every task (filename `docs/GIN_14_Assay.md`, wikilink `[[GIN_14_Assay]]`, plain-text `GIN_14_Assay` in the non-vault Receipts repo). The five refusal codes are spelled identically wherever they appear (`NO_GROUNDING`, `BELOW_THRESHOLD`, `QUERY_UNGROUNDABLE`, `CORPUS_INSUFFICIENT`, `CONFLICTING_UNRESOLVABLE`). Section numbering `1 —` … `9 —` matches between the Task 1 content and the Step 2 verification grep. The `## Related` / `## Back to Vault` / `[[HOME]]` footer matches the series convention verified in `GIN_04_SEAR.md` and `GIN_07_Governance_Validity.md`.
