# Node6 collaborator setup — machine, architecture, and authoring workflow

This is the one-stop doc for authoring the node6 corpus on your own machine.
It deliberately does **not** describe how the detector works internally — the
value of your corpus is that it was written blind to the implementation. If
you want the internals afterwards, read `architecture.md`; not before.

## 1. Machine setup

Prerequisites: git, Python 3.12+, ~5 GB free disk (models download on first
model-dependent run; pure authoring needs none of that).

Kristian must add your GitHub account as a collaborator on the repo first.

```bash
git clone https://github.com/KristianKrattiger/GIN.git
cd GIN
```

Create the venv and install. On Windows, use WSL (Ubuntu) — the ML stack is
only exercised under Linux here; native Windows Python crashes in the
generation path:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Sanity check (no models, no network, a few seconds):

```bash
./venv/bin/python -m pytest tests/test_curator_node5_build.py tests/test_cartographer_same_story.py -q
```

Both files passing means the manifest builder and the corpus tooling work on
your machine. If you later run the full suite: ~25 collection errors from
missing `fastapi`/`uvicorn` are known and harmless unless you want to run the
labeling UI (see §6, which installs them).

## 2. GIN in five sentences (what your corpus feeds)

GIN ingests news-like documents into **corpus nodes** (`corpus_node1..5.json`
at the repo root; yours becomes `corpus_node6.json`). The **Cartographer**
proposes typed relations between text chunks — *contradicts*, *corroborates*,
*supersedes*, *unrelated* — in two stages: a cheap stage-1 relatedness gate
over the whole pair space, then an expensive stage-2 relation typer on the
survivors. The **Curator** is a local web app where a human labels proposed
pairs, producing the gold data every threshold and eval in the project rests
on. Synthetic corpora like yours exist because they come with a known answer
key: you author the relationships on purpose, someone else labels them blind,
and the gap between the two measures both the human and the machine. Node6's
specific job is to be the first corpus whose prose the detector's author had
no hand in.

## 3. Authoring the manifest

Your entire deliverable is one YAML file:

```
data/curator/node6_events.yaml
```

Format (identical to node5's — see `data/curator/node5_events.yaml` for a
full real example):

```yaml
- event: riverbend_levee_breach          # snake_case, unique per event
  domain: incident                       # incident | election | policy | ... (your call)
  shared_lede: "RIVERBEND — Crews worked overnight after a levee breach east of town Thursday."
  reports:
    - outlet: CentralWire                # each outlet appears once per event
      published: "2026-08-03T06:10Z"     # ISO timestamp; order updates by time
      chunks:
        - "RIVERBEND — Crews worked overnight after a levee breach east of town Thursday. <3-5 more sentences in your own words...>"
    - outlet: MetroDaily
      published: "2026-08-03T07:40Z"
      chunks:
        - "RIVERBEND — Crews worked overnight after a levee breach east of town Thursday. <...>"
  intent:                                # YOUR ANSWER KEY — one entry per report pair
    - pair: [CentralWire, MetroDaily]
      kind: update                       # conflict | corroboration | update | compatible_partial
      varied_fact: displaced_count       # what differs; null for corroboration
```

Rules the builder enforces (`gin/curator/node5_build.py` validation):

- every event needs `event`, `domain`, `shared_lede`, `reports`, `intent`
- every report needs `outlet`, `published`, and at least one chunk
- the `shared_lede` must open every report of its event, verbatim
- the `intent` matrix must cover pairs using that event's real outlet names,
  with `kind` one of the four above

Composition targets:

| kind | target pairs | notes |
|---|---|---|
| `update` | ~12 | the scarce class — later report revises an earlier figure |
| `conflict` | ~8 | flat factual disagreement, usually numeric |
| `corroboration` | ~6 | include 1–2 "different measurements of the same thing" |
| `compatible_partial` | 2–4 | overlapping but not fully aligned coverage |

Plus the **confusable events**: 3–4 events that reuse surface vocabulary
across event boundaries (a proper name in one event that is an ordinary word
in another, the same number in two unrelated stories, shared boilerplate
phrasing). These need no `intent` entries — cross-event pairs are implicit
negatives. Keep a private note of which events these are.

Everything fictional: invented towns, outlets, officials. Write in your own
register; do not mimic node1–5's style.

## 4. Build and validate the corpus

```bash
./venv/bin/python scripts/build_node6.py
```

This validates the manifest, prints the pair inventory (check it against the
table above), and writes `corpus_node6.json` at the repo root. It fails
loudly on a malformed manifest or a too-thin composition — fix and re-run
until it prints the inventory and a `wrote N docs` line.

## 5. Handoff — the blind protocol

Order matters here:

1. Commit and push `corpus_node6.json` **only** (or send the file directly).
2. Do **not** share `node6_events.yaml` yet — the `intent` matrix is the
   answer key, and Kristian labels blind against the corpus alone.
3. After Kristian's labels are in, push the manifest; the surfacing gate and
   agreement scoring run against it.
4. Disagreements between your intent and the blind labels get adjudicated
   pair-by-pair (see `docs/curator-collab-setup.md`, "Reconciliation") —
   sometimes the label is wrong, sometimes the prose didn't say what you
   meant it to. Both outcomes are data.

## 6. Optional: running the curator labeling app

Only needed if you also take a labeling shift on the other nodes (see
`docs/curator-collab-setup.md` for assignments and
`docs/curator-labeling-guide.md` for how to label). Requires the web extras
and a one-time model download (a few GB, several minutes on first launch):

```bash
./venv/bin/pip install fastapi uvicorn
./venv/bin/python scripts/curator_serve.py \
  --curator <your-name> \
  --source same-story \
  --corpus corpus_node2.json corpus_node3.json \
  --log data/curator/labels.<your-name>.jsonl \
  --port 8601
```

Then open http://127.0.0.1:8601/curator/ and label. Your log file is yours
alone — merging happens on Kristian's side via
`scripts/curator_merge_check.py`.
