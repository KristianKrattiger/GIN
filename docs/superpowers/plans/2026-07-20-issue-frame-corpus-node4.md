# issue_frame Corpus (node4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `corpus_node4.json` of real fetched pro/con citations across 10 proposition-level contested topics so genuine `issue_frame` pairs exist in the corpus and surface to the curator UI.

**Architecture:** A two-stage reproducible pipeline. Stage 1 (by hand): research + fetch real opposing sources into a reviewable YAML manifest, with the source list approved before sentence extraction. Stage 2 (deterministic): a pure builder turns the manifest into `corpus_node4.json` with computed provenance. A model-backed surfacing verifier is a hard gate: every topic's thesis pair must reach the curator backlog before the corpus is done. Pure logic lives in importable `gin/curator/` modules; `scripts/` holds thin CLIs (matching the existing `curator_serve.py` / `curator_readiness.py` split — this refines the spec's script-only phrasing for testability).

**Tech Stack:** Python 3.14, pytest, PyYAML 6, `gin.cartographer` (SentenceTransformer + HF NLI via `CombinedRelationProposer`), `gin.curator`.

## Global Constraints

- Provenance recipe (verbatim): `global_id = "gid_" + hashlib.sha256(f"{source}|{author}|{date}".encode()).hexdigest()[:16]`.
- Known-answer anchor: `source="Indigenous Environmental Network: Frontline Communities Demand Real Climate Solutions"`, `author="Indigenous Environmental Network"`, `date="2023-12"` → `gid_f5842fdb72d6327a`.
- Corpus schema per doc: `{doc_id, global_id, source, url, node, metadata:{author,category,date,domain,type,stance,topic}, chunks:[{chunk_id, position, text}]}`; top level `{node_id, documents:[...]}`.
- `node_id = "node_4_contested"`; `node` field per doc = `"node_4_contested"`.
- Doc ids `n4_doc_00X` (1-indexed, zero-padded to 3); chunk ids `n4_doc_00X_c000` (zero-padded to 3); `position` stored as a **string** (`"0"`).
- Pro before con: `n4_doc_001` = topic-1 pro, `n4_doc_002` = topic-1 con, etc. (pro = odd id, con = even id).
- Chunk sizing: 3–5 chunks/doc, one claim each, ~20–30 words (matches node1–3: median 5 chunks/doc, 23 words/chunk).
- `stance ∈ {"pro","con"}`; `domain ∈ {"climate_policy","energy_policy","fiscal_policy"}`; `type ∈ {"opinion","advocacy","analysis"}`.
- Chunks are single attributed claim sentences (no long copyrighted passages).
- All unit tests DB-free and llama_cpp-free; the real-model surfacing run is script-driven, not CI.
- 10 fixed topics: carbon_tax, nuclear_power, carbon_offsets, degrowth_vs_growth, fossil_divestment, solar_geoengineering, renewables_grid, gas_bridge_fuel, climate_spending, border_carbon_adjustment.

---

### Task 1: Manifest→corpus builder

**Files:**
- Create: `gin/curator/node4_build.py`
- Create: `scripts/build_node4.py`
- Test: `tests/test_curator_node4_build.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `compute_global_id(source: str, author: str, date: str) -> str`
  - `build_node4(manifest: list[dict]) -> dict` — manifest entries → corpus JSON dict. Each entry has keys `topic, stance, source, author, date, url, domain, type, chunks` (`chunks: list[str]`). Raises `ValueError` on: missing key, bad `stance`, a topic not appearing exactly twice as one pro + one con, non-adjacent topic pair, or a `global_id` collision.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_curator_node4_build.py
"""Deterministic manifest -> corpus_node4 builder."""
import pytest

from gin.curator.node4_build import build_node4, compute_global_id


def _entry(topic, stance, n=2, **over):
    e = {
        "topic": topic, "stance": stance,
        "source": f"{topic} {stance} source", "author": f"{topic} author",
        "date": "2023-05", "url": f"https://example.org/{topic}/{stance}",
        "domain": "climate_policy", "type": "opinion",
        "chunks": [f"{topic} {stance} claim {i}" for i in range(n)],
    }
    e.update(over)
    return e


def _pair(topic):
    return [_entry(topic, "pro"), _entry(topic, "con")]


def test_global_id_matches_known_anchor():
    got = compute_global_id(
        "Indigenous Environmental Network: Frontline Communities Demand Real Climate Solutions",
        "Indigenous Environmental Network", "2023-12",
    )
    assert got == "gid_f5842fdb72d6327a"


def test_builds_node_id_and_doc_ids_pro_then_con():
    out = build_node4(_pair("carbon_tax") + _pair("nuclear_power"))
    assert out["node_id"] == "node_4_contested"
    docs = out["documents"]
    assert [d["doc_id"] for d in docs] == [
        "n4_doc_001", "n4_doc_002", "n4_doc_003", "n4_doc_004",
    ]
    assert docs[0]["metadata"]["stance"] == "pro"
    assert docs[1]["metadata"]["stance"] == "con"
    assert docs[0]["metadata"]["topic"] == "carbon_tax"
    assert docs[0]["node"] == "node_4_contested"


def test_chunk_ids_and_string_positions():
    out = build_node4(_pair("carbon_tax"))
    chunks = out["documents"][0]["chunks"]
    assert [c["chunk_id"] for c in chunks] == ["n4_doc_001_c000", "n4_doc_001_c001"]
    assert [c["position"] for c in chunks] == ["0", "1"]
    assert chunks[0]["text"] == "carbon_tax pro claim 0"


def test_computes_global_id_per_doc():
    out = build_node4(_pair("carbon_tax"))
    d = out["documents"][0]
    assert d["global_id"] == compute_global_id(d["source"], d["metadata"]["author"], d["metadata"]["date"])


def test_missing_key_raises():
    bad = _pair("carbon_tax")
    del bad[0]["url"]
    with pytest.raises(ValueError, match="url"):
        build_node4(bad)


def test_bad_stance_raises():
    with pytest.raises(ValueError, match="stance"):
        build_node4([_entry("carbon_tax", "maybe"), _entry("carbon_tax", "con")])


def test_topic_not_exactly_pro_con_raises():
    with pytest.raises(ValueError, match="carbon_tax"):
        build_node4([_entry("carbon_tax", "pro"), _entry("carbon_tax", "pro")])


def test_non_adjacent_topic_raises():
    manifest = [_entry("carbon_tax", "pro"), _entry("nuclear_power", "pro"),
                _entry("carbon_tax", "con"), _entry("nuclear_power", "con")]
    with pytest.raises(ValueError, match="adjacent"):
        build_node4(manifest)


def test_global_id_collision_raises():
    dup = _pair("carbon_tax")
    dup[1]["source"] = dup[0]["source"]
    dup[1]["author"] = dup[0]["author"]
    dup[1]["date"] = dup[0]["date"]
    with pytest.raises(ValueError, match="global_id"):
        build_node4(dup)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_curator_node4_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.node4_build'`

- [ ] **Step 3: Write the builder**

```python
# gin/curator/node4_build.py
"""Deterministic builder: source manifest -> corpus_node4.json dict.

Pure and network-free. The manifest (data/curator/node4_sources.yaml) is the
reviewable artifact; this module turns it into the same schema node1-3 use, so
load_corpus_chunks and the whole curator path work unchanged.
"""
from __future__ import annotations

import hashlib

NODE_ID = "node_4_contested"
_REQUIRED = ("topic", "stance", "source", "author", "date", "url", "domain", "type", "chunks")


def compute_global_id(source: str, author: str, date: str) -> str:
    digest = hashlib.sha256(f"{source}|{author}|{date}".encode()).hexdigest()
    return "gid_" + digest[:16]


def _validate(manifest: list[dict]) -> None:
    for i, e in enumerate(manifest):
        for key in _REQUIRED:
            if key not in e:
                raise ValueError(f"manifest entry {i} missing required key {key!r}")
        if e["stance"] not in {"pro", "con"}:
            raise ValueError(f"manifest entry {i} bad stance {e['stance']!r} (pro|con)")
        if not e["chunks"]:
            raise ValueError(f"manifest entry {i} ({e['topic']}) has no chunks")
    # Each topic appears exactly twice: one pro, one con.
    by_topic: dict[str, list[str]] = {}
    for e in manifest:
        by_topic.setdefault(e["topic"], []).append(e["stance"])
    for topic, stances in by_topic.items():
        if sorted(stances) != ["con", "pro"]:
            raise ValueError(f"topic {topic!r} must appear exactly once pro and once con, got {stances}")
    # Topic pairs must be adjacent (entries 2k, 2k+1 share a topic).
    for k in range(0, len(manifest), 2):
        if manifest[k]["topic"] != manifest[k + 1]["topic"]:
            raise ValueError(
                f"topic pair not adjacent at entries {k},{k + 1}: "
                f"{manifest[k]['topic']!r} vs {manifest[k + 1]['topic']!r}"
            )


def build_node4(manifest: list[dict]) -> dict:
    if len(manifest) % 2 != 0:
        raise ValueError(f"manifest must be pro/con pairs (even length), got {len(manifest)}")
    _validate(manifest)
    documents = []
    seen_gids: dict[str, str] = {}
    for idx, e in enumerate(manifest, start=1):
        doc_id = f"n4_doc_{idx:03d}"
        gid = compute_global_id(e["source"], e["author"], e["date"])
        if gid in seen_gids:
            raise ValueError(
                f"global_id collision {gid} between {seen_gids[gid]} and {doc_id} "
                f"(identical source|author|date)"
            )
        seen_gids[gid] = doc_id
        chunks = [
            {"chunk_id": f"{doc_id}_c{j:03d}", "position": str(j), "text": text}
            for j, text in enumerate(e["chunks"])
        ]
        documents.append({
            "doc_id": doc_id,
            "global_id": gid,
            "source": e["source"],
            "url": e["url"],
            "node": NODE_ID,
            "metadata": {
                "author": e["author"],
                "category": e["topic"],
                "date": e["date"],
                "domain": e["domain"],
                "type": e["type"],
                "stance": e["stance"],
                "topic": e["topic"],
            },
            "chunks": chunks,
        })
    return {"node_id": NODE_ID, "documents": documents}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_curator_node4_build.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Write the thin CLI**

```python
# scripts/build_node4.py
"""Build corpus_node4.json from the approved source manifest.

    venv/Scripts/python.exe scripts/build_node4.py \
        --manifest data/curator/node4_sources.yaml --out corpus_node4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gin.curator.node4_build import build_node4


def main() -> None:
    ap = argparse.ArgumentParser(description="Build corpus_node4.json from manifest")
    ap.add_argument("--manifest", type=Path, default=Path("data/curator/node4_sources.yaml"))
    ap.add_argument("--out", type=Path, default=Path("corpus_node4.json"))
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    corpus = build_node4(manifest)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(corpus['documents'])} docs to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add gin/curator/node4_build.py scripts/build_node4.py tests/test_curator_node4_build.py
git commit -m "Curator: node4 manifest->corpus builder (issue_frame corpus)"
```

---

### Task 2: Surfacing verifier

**Files:**
- Create: `gin/curator/node4_verify.py`
- Create: `scripts/verify_node4_surfacing.py`
- Test: `tests/test_curator_node4_verify.py`

**Interfaces:**
- Consumes: `EscalationResidueCandidateSource` (`gin/curator/residue.py`), `CombinedRelationProposer` (`gin/cartographer/combined.py`), `load_corpus_chunks` (`gin/curator/corpus_json.py`), `pair_key` (`gin/curator/models.py`).
- Produces:
  - `TopicResult` dataclass: `topic: str`, `pro_key: tuple[str, str]` (the thesis `pair_key`), `passed: bool`, `rank: int | None` (index in surfaced backlog, `None` if absent).
  - `intended_thesis_pairs(documents: list[dict]) -> dict[str, tuple[str, str]]` — per topic, the `pair_key` of the two docs' position-0 (thesis) chunks, normalized to `{doc_id}:0`. Pure, no models.
  - `verify_surfacing(chunks: list[LabeledChunk], documents: list[dict], proposer: CombinedRelationProposer) -> list[TopicResult]` — builds the residue source, marks each thesis pair PASS if present in `source.pairs()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_curator_node4_verify.py
"""node4 surfacing verifier: thesis-pair identification + PASS/SINK over a
model-free proposer (same injection pattern as test_curator_residue)."""
from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk
from gin.curator.models import pair_key
from gin.curator.node4_verify import intended_thesis_pairs, verify_surfacing

DOCS = [
    {"doc_id": "n4_doc_001", "metadata": {"topic": "carbon_tax", "stance": "pro"},
     "chunks": [{"position": "0", "text": "carbon tax pro thesis"},
                {"position": "1", "text": "carbon tax pro support"}]},
    {"doc_id": "n4_doc_002", "metadata": {"topic": "carbon_tax", "stance": "con"},
     "chunks": [{"position": "0", "text": "carbon tax con thesis"},
                {"position": "1", "text": "carbon tax con support"}]},
]


def test_intended_thesis_pairs_uses_position_zero():
    got = intended_thesis_pairs(DOCS)
    assert got == {"carbon_tax": pair_key("n4_doc_001:0", "n4_doc_002:0")}


def _proposer(cos_map):
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos_map.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: False,
    )


def test_pass_when_thesis_pair_surfaces():
    chunks = [LabeledChunk("n4_doc_001:0", "carbon tax pro thesis"),
              LabeledChunk("n4_doc_002:0", "carbon tax con thesis")]
    cos = {frozenset({"carbon tax pro thesis", "carbon tax con thesis"}): 0.40}
    results = verify_surfacing(chunks, DOCS, _proposer(cos))
    assert len(results) == 1
    assert results[0].topic == "carbon_tax"
    assert results[0].passed is True
    assert results[0].rank == 0


def test_sink_when_below_floor():
    chunks = [LabeledChunk("n4_doc_001:0", "carbon tax pro thesis"),
              LabeledChunk("n4_doc_002:0", "carbon tax con thesis")]
    cos = {frozenset({"carbon tax pro thesis", "carbon tax con thesis"}): 0.05}
    results = verify_surfacing(chunks, DOCS, _proposer(cos))
    assert results[0].passed is False
    assert results[0].rank is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_curator_node4_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.node4_verify'`

- [ ] **Step 3: Write the verifier logic**

```python
# gin/curator/node4_verify.py
"""Hard-gate verifier: do node4's thesis pairs reach the curator backlog?

A genuinely-opposed pair whose cosine is below the residue floor (or that reads
same-story) never surfaces; this catches that at build time so sources can be
sharpened before a human labels. Model-free under test via an injected proposer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk

from .models import pair_key
from .residue import EscalationResidueCandidateSource


@dataclass(frozen=True)
class TopicResult:
    topic: str
    pro_key: tuple[str, str]
    passed: bool
    rank: Optional[int]


def intended_thesis_pairs(documents: list[dict]) -> dict[str, tuple[str, str]]:
    """Per topic, the pair_key of each side's position-0 (thesis) chunk."""
    thesis_by_topic: dict[str, dict[str, str]] = {}
    for doc in documents:
        topic = doc["metadata"]["topic"]
        stance = doc["metadata"]["stance"]
        zero = next(c for c in doc["chunks"] if str(c["position"]) == "0")
        cid = f"{doc['doc_id']}:{zero['position']}"
        thesis_by_topic.setdefault(topic, {})[stance] = cid
    out: dict[str, tuple[str, str]] = {}
    for topic, sides in thesis_by_topic.items():
        out[topic] = pair_key(sides["pro"], sides["con"])
    return out


def verify_surfacing(
    chunks: list[LabeledChunk],
    documents: list[dict],
    proposer: CombinedRelationProposer,
) -> list[TopicResult]:
    source = EscalationResidueCandidateSource(chunks, proposer=proposer)
    surfaced = [pair_key(a.chunk_id, b.chunk_id) for a, b in source.pairs()]
    rank_of = {key: i for i, key in enumerate(surfaced)}
    results = []
    for topic, key in intended_thesis_pairs(documents).items():
        rank = rank_of.get(key)
        results.append(TopicResult(topic, key, rank is not None, rank))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_curator_node4_verify.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the real-model CLI**

```python
# scripts/verify_node4_surfacing.py
"""Hard gate: every node4 topic's thesis pair must reach the curator backlog.

    venv/Scripts/python.exe scripts/verify_node4_surfacing.py \
        --corpus corpus_node1.json corpus_node2.json corpus_node3.json corpus_node4.json

Loads the real SentenceTransformer + NLI proposer (no llama_cpp). Exit 0 iff all
node4 thesis pairs PASS; exit 1 lists the sinkers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node4_verify import verify_surfacing


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify node4 issue_frame pairs surface")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("corpus_node1.json"), Path("corpus_node2.json"),
                             Path("corpus_node3.json"), Path("corpus_node4.json")])
    ap.add_argument("--node4", type=Path, default=Path("corpus_node4.json"))
    args = ap.parse_args()

    chunks = load_corpus_chunks(args.corpus)
    node4_docs = json.loads(args.node4.read_text(encoding="utf-8"))["documents"]
    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded

    results = verify_surfacing(chunks, node4_docs, proposer)
    sinks = [r for r in results if not r.passed]
    for r in sorted(results, key=lambda r: (r.passed, r.topic)):
        mark = f"PASS rank={r.rank}" if r.passed else "SINK"
        print(f"{'✓' if r.passed else '✗'} {r.topic:<24} {mark}")
    print(f"\n{len(results) - len(sinks)}/{len(results)} thesis pairs surfaced")
    if sinks:
        print("HARD GATE FAILED — sharpen sources for: " + ", ".join(r.topic for r in sinks))
        sys.exit(1)
    print("HARD GATE PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add gin/curator/node4_verify.py scripts/verify_node4_surfacing.py tests/test_curator_node4_verify.py
git commit -m "Curator: node4 surfacing verifier + hard-gate CLI"
```

---

### Task 3: Research the 10 topic pairs → approved source list

**Files:**
- Create: `data/curator/node4_sources.yaml` (source list only at this stage; `chunks` left empty per entry)

**Interfaces:**
- Consumes: nothing (content-gathering).
- Produces: the manifest file consumed by Task 1's `build_node4` and Task 4's build step. This is a data/content task, not a code task — validation replaces unit tests.

- [ ] **Step 1: Research one real source per topic-stance.** For each of the 10 fixed topics (carbon_tax, nuclear_power, carbon_offsets, degrowth_vs_growth, fossil_divestment, solar_geoengineering, renewables_grid, gas_bridge_fuel, climate_spending, border_carbon_adjustment), use WebSearch + WebFetch to find one credible source arguing **pro** and one arguing **con** of the *same proposition* (not merely the same topic). Record real `source` (title), `author`/org, `date` (YYYY-MM), `url`, `domain` (climate_policy | energy_policy | fiscal_policy), `type` (opinion | advocacy | analysis). Prefer sources whose thesis sentence is a direct, contradictory claim on the shared proposition — that is what will clear the surfacing gate.

- [ ] **Step 2: Write the source list to the manifest** with `chunks: []` placeholders, ordered pro-then-con, topic pairs adjacent:

```yaml
# data/curator/node4_sources.yaml
- topic: carbon_tax
  stance: pro
  source: "<real article title>"
  author: "<real author/org>"
  date: "2023-11"
  url: "<real url>"
  domain: fiscal_policy
  type: opinion
  chunks: []
- topic: carbon_tax
  stance: con
  source: "<real article title>"
  author: "<real author/org>"
  date: "2023-09"
  url: "<real url>"
  domain: fiscal_policy
  type: opinion
  chunks: []
# ... 8 more topic pairs, same shape ...
```

- [ ] **Step 3: HUMAN GATE — present the source list for approval.** Show the user the 20 sources (topic, stance, title, author, url) in a table. Do **not** proceed to extraction until the user approves. If the user rejects a source, find a replacement and re-present. (This is the "approve source list first" decision from the spec.)

- [ ] **Step 4: Commit the approved source list**

```bash
git add data/curator/node4_sources.yaml
git commit -m "Curator: node4 approved source list (10 contested topics, chunks pending)"
```

---

### Task 4: Extract chunks, build corpus, wire launcher, pass hard gate

**Files:**
- Modify: `data/curator/node4_sources.yaml` (fill `chunks`)
- Create: `corpus_node4.json` (generated)
- Modify: `scripts/curator_serve.py:38-41` (add node4 to default `--corpus`)
- Test: `tests/test_curator_node4_corpus.py` (loader-compat over the generated file)

**Interfaces:**
- Consumes: `build_node4` (Task 1), `verify_surfacing` CLI (Task 2), the approved manifest (Task 3), `load_corpus_chunks`.
- Produces: `corpus_node4.json` passing the hard gate; launcher default includes it.

- [ ] **Step 1: Extract claim sentences into the manifest.** For each source, pull 3–5 single-claim sentences (~20–30 words, real text, verbatim or minimally trimmed) from the fetched article and fill its `chunks:` list. Make the **first chunk (position 0) the thesis** — the sentence that most directly asserts the pro/con position — because the verifier keys on position-0 for the thesis pair.

- [ ] **Step 2: Build the corpus**

Run: `python scripts/build_node4.py --manifest data/curator/node4_sources.yaml --out corpus_node4.json`
Expected: `wrote 20 docs to corpus_node4.json`

- [ ] **Step 3: Write the loader-compat test**

```python
# tests/test_curator_node4_corpus.py
"""corpus_node4.json loads and normalizes like node1-3, with the expected shape."""
import json
from pathlib import Path

from gin.curator.corpus_json import load_corpus_chunks

CORPUS = Path("corpus_node4.json")


def test_loads_and_normalizes():
    chunks = load_corpus_chunks([CORPUS])
    assert chunks, "node4 produced no chunks"
    for c in chunks:
        assert c.chunk_id.startswith("n4_doc_")
        assert ":" in c.chunk_id  # normalized to {doc_id}:{position}
        assert c.text.strip()


def test_twenty_docs_ten_topics_pro_con():
    docs = json.loads(CORPUS.read_text(encoding="utf-8"))["documents"]
    assert len(docs) == 20
    topics = {}
    for d in docs:
        topics.setdefault(d["metadata"]["topic"], []).append(d["metadata"]["stance"])
    assert len(topics) == 10
    for topic, stances in topics.items():
        assert sorted(stances) == ["con", "pro"], topic
```

- [ ] **Step 4: Run the loader-compat test**

Run: `python -m pytest tests/test_curator_node4_corpus.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the surfacing hard gate**

Run: `python scripts/verify_node4_surfacing.py`
Expected: `10/10 thesis pairs surfaced` then `HARD GATE PASSED` (exit 0).
If any topic SINKs: return to Task 4 Step 1, pick sharper (more directly contradictory) thesis sentences for the sinking topic(s), rebuild (Step 2), and re-run the gate. Repeat until 10/10. Do **not** modify residue/candidate ranking logic — adapt the sources.

- [ ] **Step 6: Wire the launcher default.** In `scripts/curator_serve.py`, change the `--corpus` default:

```python
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("corpus_node1.json"), Path("corpus_node2.json"),
                             Path("corpus_node3.json"), Path("corpus_node4.json")],
                    help="corpus_node*.json exports for the escalation-residue source")
```

- [ ] **Step 7: Run the full curator suite to confirm no regressions**

Run: `python -m pytest tests/ -k curator -v`
Expected: PASS (existing curator tests + the three new node4 test files all green)

- [ ] **Step 8: Commit**

```bash
git add data/curator/node4_sources.yaml corpus_node4.json scripts/curator_serve.py tests/test_curator_node4_corpus.py
git commit -m "Curator: node4 issue_frame corpus built, surfacing gate passed, launcher wired"
```

---

## Notes for the implementer

- **Why position-0 = thesis:** the verifier (Task 2) identifies each topic's intended issue_frame pair as the two docs' position-0 chunks. Task 4 Step 1 must therefore put the most directly-opposed claim first in each doc, or a genuine pair can SINK purely from chunk ordering.
- **Surfacing intuition:** the residue keeps pairs that are *not same-story* and *cosine ≥ 0.30* (`DEFAULT_ESCALATION_COS_FLOOR`). Proposition-level pro/con sentences share vocabulary (high cosine, clears the floor) but are different stories (clears same-story) — that is the design's whole bet. A SINK almost always means the two sentences were too topically distant (paraphrased away the shared terms) or accidentally same-story.
- **No llama_cpp anywhere here:** the proposer's cosine + NLI are SentenceTransformer/HF and run on Windows Python. The llama_cpp/WSL constraint only applies to *generation*, which this plan never touches.
- **Parked (out of scope):** the header-refresh bug and the same-issue candidate guard are not addressed here.
