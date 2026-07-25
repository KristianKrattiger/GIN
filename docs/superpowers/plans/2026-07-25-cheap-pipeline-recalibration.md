# Cheap-Pipeline Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate `CombinedRelationProposer`'s thresholds against the 133 non-eval pairs in the curator store instead of 39 hardcoded sample tuples, with an honest held-out number and provenance that makes staleness detectable.

**Architecture:** A model-bound generator reads the curator store and measures `(cos, p_contra, same_story)` for every labeled pair, writing `data/calibration/samples.json` with a manifest. The 45 frozen eval pairs land in a separate `eval_samples` array that `calibrate()` never sees, which is what makes the held-out score computable without models. Everything downstream — loading, calibration, leave-one-out — stays model-free and reads that file behind a manifest gate. Every pre-registered evaluation surface is left untouched and gains a pinning test.

**Tech Stack:** Python 3.12, sentence-transformers, `cross-encoder/nli-deberta-v3-xsmall`, numpy, pytest.

## Global Constraints

- **Layering:** `gin.cartographer` must **never** import `gin.curator`. `gin.curator` may import `gin.cartographer`. Nothing may import `gin.frames`. Scripts may import anything.
- **Frozen eval surfaces:** `gold_edges.py`, `escalation_eval.py`, `scan_eval.py`, `evaluation.py`, and `labeled_set.py` keep their current behavior. No task may change which pairs they yield.
- **The escalation bar is 14 pairs** (4 issue_frame + 6 corroboration + 4 unrelated) and must be byte-identical before and after this work.
- **Model-free by default:** only `scripts/regen_calibration_samples.py` may load embed/NLI models. Every test must pass without a model download.
- **Exact expected counts:** store folds to **178** pairs; eval set is **45** pairs (`labeled_set` ∪ `gold_edges`); calibration set is **133** pairs. Class mix of the 133: `related_untyped` 62, `corroborates` 26, `contradicts` 22, `unrelated` 21, plus 2 `supersedes` rows that are excluded (not a classifier output), so **131 usable samples**.
- **Current baseline to beat or report against:** 39 baked samples, thresholds gate 0.140 / ceiling 0.486 / contra 0.686, LOO accuracy **0.897**, LOO `class_c_discrimination` **1.000**.
- **Model ids:** embed `sentence-transformers/all-MiniLM-L6-v2`, NLI `cross-encoder/nli-deberta-v3-xsmall` (read from `combined.DEFAULT_EMBED_MODEL` / `DEFAULT_NLI_MODEL`, never hardcoded in new code).
- **Pair identity in `gin/cartographer/`** uses `frozenset((src, dst))`, matching the existing `gold_edges.gold_contradicts_keys()` convention. Do **not** import `gin.curator.models.pair_key` into cartographer — that would break layering.
- Run all commands from the repo root using `venv/Scripts/python.exe`. Scripts that import through `gin.curator.store` need the repo-root `sys.path` prelude used by `scripts/frames_probe.py`.

## File Structure

| File | Responsibility |
|------|----------------|
| `gin/cartographer/eval_pairs.py` | The frozen eval-pair set (`labeled_set` ∪ `gold_edges`) as `frozenset` keys |
| `gin/cartographer/calibration_samples.py` | `Sample`, `EvalSample`, `SampleManifest`, JSON read/write, manifest gate |
| `gin/cartographer/calibration.py` | Modified: `default_samples()` reads the file; `_MEASURED` deleted |
| `gin/curator/calibration_export.py` | Store → calibration rows + held-out eval rows; signal computation injected |
| `scripts/regen_calibration_samples.py` | Model-bound CLI producing `data/calibration/samples.json` |
| `scripts/recalibrate_cheap_pipeline.py` | Calibrate + LOO + held-out + sensitivity; writes thresholds with provenance |
| `tests/test_cartographer_eval_pairs.py` | Eval-set counts **and** the escalation-bar pinning test |
| `tests/test_cartographer_calibration_samples.py` | Schema round-trip, manifest gate, missing-file error |
| `tests/test_curator_calibration_export.py` | Exclusion rules, counts, drop reasons |
| `tests/fixtures/calibration_samples_fixture.json` | Small committed sample file for the existing calibration tests |

---

### Task 1: Frozen eval-pair set and the escalation-bar pin

The escalation bar is a pre-registered evaluation. This task makes it impossible to change it silently, and defines the 45-pair set that calibration must never train on.

**Files:**
- Create: `gin/cartographer/eval_pairs.py`
- Test: `tests/test_cartographer_eval_pairs.py`

**Interfaces:**
- Consumes: `gin.cartographer.labeled_set.gold()`, `gin.cartographer.gold_edges.gold_pairs()`, `gin.cartographer.escalation_eval.default_calibration_sets()`
- Produces: `eval_pair_keys() -> frozenset[frozenset[str]]` (each inner frozenset is `{src_chunk_id, dst_chunk_id}`), `BAR_PAIR_IDS: tuple[tuple[str, str, str], ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cartographer_eval_pairs.py`:

```python
"""The frozen eval surfaces: the 45-pair eval set and the 14-pair bar pin."""
from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.eval_pairs import BAR_PAIR_IDS, eval_pair_keys


def test_eval_set_is_45_pairs():
    # labeled_set gold (33) union gold_edges pairs; calibration must never
    # train on any of these or its reported accuracy is a restatement.
    assert len(eval_pair_keys()) == 45


def test_eval_keys_are_unordered_pairs():
    for key in eval_pair_keys():
        assert isinstance(key, frozenset)
        assert len(key) == 2


def test_known_eval_members():
    assert frozenset(("inst_em:0", "clim_pledges:0")) in eval_pair_keys()


def test_eval_pair_keys_is_cached():
    assert eval_pair_keys() is eval_pair_keys()


def test_bar_is_exactly_14_pairs():
    sets = default_calibration_sets()
    assert len(sets["issue_frame"]) == 4
    assert len(sets["corroboration"]) == 6
    assert len(sets["unrelated"]) == 4


def test_bar_pairs_are_pinned_by_chunk_id():
    # The escalation bar is pre-registered. If this fails, some change moved a
    # pre-registered eval — revert it rather than updating this expectation.
    sets = default_calibration_sets()
    live = tuple(
        (src, dst, register)
        for group in ("issue_frame", "corroboration", "unrelated")
        for src, dst, register in sets[group]
    )
    assert live == BAR_PAIR_IDS


def test_every_bar_pair_is_in_the_eval_set_or_is_a_control():
    # The 4 issue_frame bar pairs come from curated gold_edges, so they must be
    # inside the eval set; controls are separate tuples and need not be.
    keys = eval_pair_keys()
    for src, dst, _register in default_calibration_sets()["issue_frame"]:
        assert frozenset((src, dst)) in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_eval_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.cartographer.eval_pairs'`

- [ ] **Step 3: Write the implementation**

Create `gin/cartographer/eval_pairs.py`:

```python
"""The frozen evaluation surfaces of the cheap pipeline.

Two things live here, both about what must NOT move:

``eval_pair_keys()`` is the set of pairs that scan_eval/evaluation measure
against. Calibration must exclude them, or the accuracy it reports is partly a
restatement of its own training data.

``BAR_PAIR_IDS`` pins the escalation bar. The bar is pre-registered — four LLM
judges and one learned detector have been scored on exactly these 14 pairs — so
a silent change to it would invalidate every published comparison. The pinning
test is the guard.

Pair identity is ``frozenset((src, dst))``, matching gold_edges'
``gold_contradicts_keys()``. The curator package defines an alternative ``pair_key``
helper, but cartographer deliberately avoids importing it to maintain layering
boundaries: cartographer must never depend on curator.
"""
from __future__ import annotations

from functools import lru_cache

from .gold_edges import gold_pairs
from .labeled_set import gold as labeled_set_gold

# Captured from default_calibration_sets() — issue_frame, then corroboration,
# then unrelated, in list order. Regenerate ONLY when deliberately changing the
# bar, which invalidates prior published comparisons.
BAR_PAIR_IDS: tuple[tuple[str, str, str], ...] = (
    # issue_frame (4)
    ('n1_doc_005:2', 'n2_doc_001:4', 'twonode'),
    ('n1_doc_005:1', 'n2_doc_001:1', 'twonode'),
    ('n1_doc_008:0', 'n2_doc_005:1', 'twonode'),
    ('n1_doc_009:0', 'n2_doc_008:2', 'twonode'),
    # corroboration (6)
    ('n1_doc_008:0', 'n1_doc_008:2', 'twonode'),
    ('labor_bureau_report:0', 'labor_independent_survey:0', 'news'),
    ('wage_bureau_report:0', 'wage_independent_survey:0', 'news'),
    ('inflation_bureau_report:0', 'inflation_independent_survey:0', 'news'),
    ('export_trade_report:0', 'export_independent_review:0', 'news'),
    ('n1_doc_002:0', 'n1_doc_006:2', 'twonode'),
    # unrelated (4)
    ('n1_doc_008:0', 'n2_doc_008:2', 'twonode'),
    ('n1_doc_009:0', 'n2_doc_005:1', 'twonode'),
    ('n1_doc_008:0', 'n1_doc_009:0', 'twonode'),
    ('transit_authority_update:0', 'school_district_report:0', 'news'),
)


@lru_cache(maxsize=1)
def eval_pair_keys() -> frozenset[frozenset[str]]:
    """Pairs the cheap pipeline is EVALUATED on; calibration must exclude them."""
    keys: set[frozenset[str]] = set()
    for src, dst, _relation, _register in labeled_set_gold():
        keys.add(frozenset((src, dst)))
    for pair in gold_pairs():
        keys.add(frozenset((pair.src_chunk_id, pair.dst_chunk_id)))
    return frozenset(keys)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_eval_pairs.py -v`
Expected: PASS — 7 passed

If `test_eval_set_is_45_pairs` fails with a different number, STOP and report — the store or the loaders changed and the plan's counts need revisiting.

- [ ] **Step 5: Verify layering**

Run:
```bash
venv/Scripts/python.exe -c "import subprocess,sys; out=subprocess.run(['git','grep','-nE',r'^\s*(from|import)\s+gin\.curator','--','gin/cartographer'],capture_output=True,text=True).stdout; print(out or 'clean: cartographer does not import curator'); sys.exit(1 if out else 0)"
```
Expected: `clean: cartographer does not import curator`

- [ ] **Step 6: Commit**

```bash
git add gin/cartographer/eval_pairs.py tests/test_cartographer_eval_pairs.py
git commit -m "Cartographer: frozen eval-pair set + escalation-bar pinning test"
```

---

### Task 2: Sample schema, file I/O, and the manifest gate

**Files:**
- Create: `gin/cartographer/calibration_samples.py`
- Test: `tests/test_cartographer_calibration_samples.py`

**Interfaces:**
- Consumes: `gin.cartographer.models.Relation`
- Produces: `Sample` (frozen dataclass: `cos: float`, `p_contra: float`, `relation: Relation`, `same_story: bool = False`), `EvalSample` (frozen dataclass: `src: str`, `dst: str`, plus the same four fields), `SampleManifest` (frozen dataclass with `to_json()`/`from_json()`), `DEFAULT_SAMPLES_PATH: Path`, `write_samples(path, manifest, samples, eval_samples) -> None`, `load_samples(path=None, *, expect_embed_model=None, expect_nli_model=None) -> tuple[list[Sample], SampleManifest]`, `load_eval_samples(path=None) -> list[EvalSample]`

**Why two arrays in one file.** The 45 eval pairs are measured in the same run with the same models, but land in a separate `eval_samples` array that `calibrate()` never sees. That is what makes the held-out score and the disputed-pair sensitivity computable **model-free**, without ever letting an eval pair influence a threshold.

Note: `Sample` moves here from `calibration.py`. Task 4 re-exports it so existing importers keep working.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cartographer_calibration_samples.py`:

```python
"""Calibration sample file: schema round-trip and the manifest compatibility gate."""
import json

import pytest

from gin.cartographer.calibration_samples import (
    EvalSample,
    Sample,
    SampleManifest,
    load_eval_samples,
    load_samples,
    write_samples,
)
from gin.cartographer.models import Relation


def _eval_samples():
    return [
        EvalSample(src="inst_em:0", dst="clim_pledges:0", cos=0.52,
                   p_contra=0.93, relation=Relation.CORROBORATES, same_story=True),
    ]


def _manifest(embed="embed-x", nli="nli-y"):
    return SampleManifest(
        embed_model=embed, nli_model=nli, n_samples=2,
        class_counts={"contradicts": 1, "unrelated": 1},
        excluded_eval_pairs=45, git_sha="abc1234",
        created_utc="2026-07-25T00:00:00Z",
    )


def _samples():
    return [
        Sample(cos=0.39, p_contra=0.068, relation=Relation.CONTRADICTS, same_story=True),
        Sample(cos=0.10, p_contra=0.010, relation=Relation.UNRELATED, same_story=False),
    ]


def test_round_trip_preserves_samples(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, manifest = load_samples(path)
    assert loaded == _samples()
    assert manifest == _manifest()


def test_eval_samples_round_trip_and_carry_ids(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    assert load_eval_samples(path) == _eval_samples()


def test_load_samples_never_returns_eval_samples(tmp_path):
    # The whole point of the split: calibrate() must not see an eval pair.
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, _ = load_samples(path)
    assert len(loaded) == 2
    assert all(not hasattr(s, "src") for s in loaded)


def test_round_trip_goes_through_real_json(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"manifest", "samples", "eval_samples"}
    assert raw["samples"][0]["relation"] == "contradicts"


def test_embed_model_mismatch_is_a_hard_error(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    with pytest.raises(ValueError, match="embed model mismatch"):
        load_samples(path, expect_embed_model="different-embed")


def test_nli_model_mismatch_is_a_hard_error(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    with pytest.raises(ValueError, match="NLI model mismatch"):
        load_samples(path, expect_nli_model="different-nli")


def test_matching_models_load_cleanly(tmp_path):
    path = tmp_path / "samples.json"
    write_samples(path, _manifest(), _samples(), _eval_samples())
    loaded, _ = load_samples(path, expect_embed_model="embed-x", expect_nli_model="nli-y")
    assert len(loaded) == 2


def test_missing_file_names_the_regen_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="regen_calibration_samples"):
        load_samples(tmp_path / "absent.json")


def test_manifest_json_round_trip():
    m = _manifest()
    assert SampleManifest.from_json(json.loads(json.dumps(m.to_json()))) == m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_calibration_samples.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.cartographer.calibration_samples'`

- [ ] **Step 3: Write the implementation**

Create `gin/cartographer/calibration_samples.py`:

```python
"""Calibration samples as a generated data file rather than a baked literal.

Measuring (cos, p_contra, same_story) needs embed + NLI models. Calibration
itself must not — the codebase deliberately keeps threshold search and its tests
model-free. So measurement happens once, deliberately, via
scripts/regen_calibration_samples.py, and everything downstream reads this file.

The manifest gate is the point: a sample file measured with different models
must never silently calibrate the live pipeline. That failure already happened
once in the other direction — data/cartographer_thresholds.json recorded an
accuracy the code no longer reproduced, and nothing detected it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .models import Relation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES_PATH = REPO_ROOT / "data" / "calibration" / "samples.json"
REGEN_COMMAND = "venv/Scripts/python.exe scripts/regen_calibration_samples.py"


@dataclass(frozen=True)
class Sample:
    cos: float
    p_contra: float
    relation: Relation
    # Stage-1 same-story signal: does the pair share >= 2 corpus-rare tokens?
    # Calibration feeds the classifier the signal it receives at scan time.
    same_story: bool = False


@dataclass(frozen=True)
class EvalSample:
    """A held-out eval pair, measured but NEVER calibrated on.

    Carries chunk ids because the held-out score and the disputed-pair
    sensitivity both need to identify specific pairs.
    """

    src: str
    dst: str
    cos: float
    p_contra: float
    relation: Relation
    same_story: bool = False


@dataclass(frozen=True)
class SampleManifest:
    embed_model: str
    nli_model: str
    n_samples: int
    class_counts: dict[str, int]
    excluded_eval_pairs: int
    git_sha: str
    created_utc: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "SampleManifest":
        return cls(**data)


def write_samples(
    path: Path,
    manifest: SampleManifest,
    samples: list[Sample],
    eval_samples: list[EvalSample],
) -> None:
    """Write calibration samples and held-out eval samples to one file.

    Both arrays are measured in the same run with the same models, but only
    ``samples`` is ever handed to calibrate(). Keeping them in one file means
    they cannot drift apart in provenance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest.to_json(),
        "samples": [
            {
                "cos": s.cos,
                "p_contra": s.p_contra,
                "same_story": s.same_story,
                "relation": s.relation.value,
            }
            for s in samples
        ],
        "eval_samples": [
            {
                "src": e.src,
                "dst": e.dst,
                "cos": e.cos,
                "p_contra": e.p_contra,
                "same_story": e.same_story,
                "relation": e.relation.value,
            }
            for e in eval_samples
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_samples(
    path: Optional[Path] = None,
    *,
    expect_embed_model: Optional[str] = None,
    expect_nli_model: Optional[str] = None,
) -> tuple[list[Sample], SampleManifest]:
    """Load samples + manifest. Model mismatch is a hard error, never a fallback."""
    path = DEFAULT_SAMPLES_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no calibration samples at {path}; generate them with: {REGEN_COMMAND}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = SampleManifest.from_json(payload["manifest"])
    if expect_embed_model is not None and manifest.embed_model != expect_embed_model:
        raise ValueError(
            f"embed model mismatch: samples measured with {manifest.embed_model!r}, "
            f"pipeline uses {expect_embed_model!r}"
        )
    if expect_nli_model is not None and manifest.nli_model != expect_nli_model:
        raise ValueError(
            f"NLI model mismatch: samples measured with {manifest.nli_model!r}, "
            f"pipeline uses {expect_nli_model!r}"
        )
    samples = [
        Sample(
            cos=row["cos"],
            p_contra=row["p_contra"],
            relation=Relation(row["relation"]),
            same_story=row["same_story"],
        )
        for row in payload["samples"]
    ]
    return samples, manifest


def load_eval_samples(path: Optional[Path] = None) -> list[EvalSample]:
    """The held-out eval pairs. Never pass these to calibrate()."""
    path = DEFAULT_SAMPLES_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no calibration samples at {path}; generate them with: {REGEN_COMMAND}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalSample(
            src=row["src"],
            dst=row["dst"],
            cos=row["cos"],
            p_contra=row["p_contra"],
            relation=Relation(row["relation"]),
            same_story=row["same_story"],
        )
        for row in payload.get("eval_samples", [])
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_calibration_samples.py -v`
Expected: PASS — 10 passed

- [ ] **Step 5: Commit**

```bash
git add gin/cartographer/calibration_samples.py tests/test_cartographer_calibration_samples.py
git commit -m "Cartographer: calibration sample file schema + manifest gate"
```

---

### Task 3: Store export — labels to unmeasured sample rows

Signal computation is injected so this whole module is testable without models. The real scorer is wired in Task 4's script.

**Files:**
- Create: `gin/curator/calibration_export.py`
- Test: `tests/test_curator_calibration_export.py`

**Interfaces:**
- Consumes: `gin.curator.store.Store`, `gin.curator.text_index.default_text_index`, `gin.cartographer.eval_pairs.eval_pair_keys`, `gin.cartographer.models.Relation`
- Produces: `SignalsFn = Callable[[str, str], tuple[float, float, bool]]`, `ExportReport` (frozen dataclass: `rows: list[dict]`, `eval_rows: list[dict]`, `drops: dict[str, int]`, property `class_counts`), `export_calibration_rows(store, signals_fn, text_index=None) -> ExportReport`

Each calibration row is `{"cos": float, "p_contra": float, "same_story": bool, "relation": str}`. Each eval row adds `"src"` and `"dst"`. Eval pairs are **measured but kept separate** — they are what makes the held-out score computable model-free, and they must never reach `calibrate()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_curator_calibration_export.py`:

```python
"""Store -> calibration rows: eval pairs excluded, drops counted."""
import pytest

from gin.cartographer.models import Relation
from gin.curator.calibration_export import export_calibration_rows
from gin.curator.models import LabelRecord
from gin.curator.store import Store


def _rec(src, dst, relation, ts, relation_class=None):
    return LabelRecord(
        id=f"{src}|{dst}", src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator="t", ts=ts,
    )


def _signals(a_text, b_text):
    """Deterministic stand-in for embed + NLI; no models."""
    return (0.5, 0.25, True)


def _text(*ids):
    return {i: f"text of {i}" for i in ids}


def test_excludes_eval_pairs(tmp_path):
    # inst_em:0 <-> clim_pledges:0 is a labeled_set gold member, so it is an
    # eval pair and must never reach calibration.
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("inst_em:0", "clim_pledges:0", Relation.CORROBORATES, "2026-01-01T00:00:00Z"))
    store.append(_rec("free_a:0", "free_b:0", Relation.UNRELATED, "2026-01-01T00:00:01Z"))
    report = export_calibration_rows(
        store, _signals, text_index=_text("inst_em:0", "clim_pledges:0", "free_a:0", "free_b:0")
    )
    assert report.drops["eval_pair"] == 1
    assert len(report.rows) == 1
    assert report.rows[0]["relation"] == "unrelated"
    # Measured, but held out — never a calibration row.
    assert len(report.eval_rows) == 1
    assert report.eval_rows[0]["src"] == "inst_em:0"


def test_supersedes_rows_are_dropped(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "b:0", Relation.SUPERSEDES, "2026-01-01T00:00:00Z"))
    store.append(_rec("c:0", "d:0", Relation.CORROBORATES, "2026-01-01T00:00:01Z"))
    report = export_calibration_rows(store, _signals, text_index=_text("a:0", "b:0", "c:0", "d:0"))
    assert report.drops["not_a_classifier_output"] == 1
    assert len(report.rows) == 1


def test_unresolvable_text_is_dropped_and_counted(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "ghost:0", Relation.UNRELATED, "2026-01-01T00:00:00Z"))
    store.append(_rec("c:0", "d:0", Relation.CORROBORATES, "2026-01-01T00:00:01Z"))
    report = export_calibration_rows(store, _signals, text_index=_text("a:0", "c:0", "d:0"))
    assert report.drops["text_unresolved"] == 1
    assert len(report.rows) == 1


def test_rows_carry_the_injected_signals(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "b:0", Relation.CONTRADICTS, "2026-01-01T00:00:00Z"))
    report = export_calibration_rows(store, _signals, text_index=_text("a:0", "b:0"))
    assert report.rows[0] == {
        "cos": 0.5, "p_contra": 0.25, "same_story": True, "relation": "contradicts",
    }


def test_empty_result_is_a_hard_error(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("inst_em:0", "clim_pledges:0", Relation.CORROBORATES, "2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="no calibration rows"):
        export_calibration_rows(store, _signals, text_index=_text("inst_em:0", "clim_pledges:0"))


def test_real_store_yields_expected_counts():
    # Regression guard on the 133/45 split. If the label log changes, this names
    # the drift rather than silently recalibrating on a different set.
    from pathlib import Path

    from gin.frames.dataset import DEFAULT_LABELS

    report = export_calibration_rows(Store(Path(DEFAULT_LABELS)), _signals)
    assert report.drops["eval_pair"] == 45
    assert report.drops["not_a_classifier_output"] == 2
    assert len(report.rows) == 131
    assert len(report.eval_rows) == 45
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_calibration_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.curator.calibration_export'`

- [ ] **Step 3: Write the implementation**

Create `gin/curator/calibration_export.py`:

```python
"""Export curator labels as unmeasured calibration rows.

Lives in gin.curator because it reads the label store; gin.cartographer may not
import gin.curator, so the cartographer-side code only handles schema and I/O.

Signal computation is injected rather than imported, so every test here runs
without embed or NLI models. The real scorer is wired in
scripts/regen_calibration_samples.py.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from gin.cartographer.eval_pairs import eval_pair_keys
from gin.cartographer.models import Relation

from .store import Store
from .text_index import default_text_index

# (a_text, b_text) -> (cos, p_contra, same_story)
SignalsFn = Callable[[str, str], tuple[float, float, bool]]

# Relations the threshold classifier can emit. SUPERSEDES is a graph relation,
# not a detector output, so it is not a calibration target.
_CLASSIFIER_RELATIONS = frozenset(
    {Relation.CONTRADICTS, Relation.CORROBORATES, Relation.RELATED_UNTYPED, Relation.UNRELATED}
)


@dataclass(frozen=True)
class ExportReport:
    rows: list[dict]        # calibration rows
    eval_rows: list[dict]   # held-out rows, measured but never calibrated on
    drops: dict[str, int]

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(r["relation"] for r in self.rows))


def export_calibration_rows(
    store: Store,
    signals_fn: SignalsFn,
    text_index: Optional[dict[str, str]] = None,
) -> ExportReport:
    """Fold the store into calibration rows, excluding every eval pair."""
    text = default_text_index() if text_index is None else text_index
    eval_keys = eval_pair_keys()
    drops: Counter[str] = Counter()
    rows: list[dict] = []
    eval_rows: list[dict] = []

    for src, dst, relation, _relation_class in sorted(
        store.gold(), key=lambda row: tuple(sorted((row[0], row[1])))
    ):
        if relation not in _CLASSIFIER_RELATIONS:
            drops["not_a_classifier_output"] += 1
            continue
        if src not in text or dst not in text:
            drops["text_unresolved"] += 1
            continue
        cos, p_contra, same_story = signals_fn(text[src], text[dst])
        measured = {
            "cos": float(cos),
            "p_contra": float(p_contra),
            "same_story": bool(same_story),
            "relation": relation.value,
        }
        if frozenset((src, dst)) in eval_keys:
            # Measured for the held-out score, then kept out of calibration.
            drops["eval_pair"] += 1
            eval_rows.append({"src": src, "dst": dst, **measured})
            continue
        rows.append(measured)

    if not rows:
        raise ValueError(f"no calibration rows after filtering (drops: {dict(drops)})")
    return ExportReport(rows, eval_rows, dict(drops))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_curator_calibration_export.py -v`
Expected: PASS — 6 passed

If `test_real_store_yields_expected_counts` reports different numbers, STOP and report both the expected and actual values — do not edit the expectation to match.

- [ ] **Step 5: Commit**

```bash
git add gin/curator/calibration_export.py tests/test_curator_calibration_export.py
git commit -m "Curator: export non-eval labels as calibration rows"
```

---

### Task 4: Generator script, and switch `default_samples()` to the file

This task deletes `_MEASURED`. There is deliberately no fallback — a silent fallback to 39 stale samples is the exact failure being removed.

**Files:**
- Create: `scripts/regen_calibration_samples.py`
- Create: `tests/fixtures/calibration_samples_fixture.json`
- Modify: `gin/cartographer/calibration.py` (delete `_MEASURED`, rewrite `default_samples()`, re-export `Sample`)
- Modify: `tests/test_cartographer_calibration.py` (point at the fixture)

**Interfaces:**
- Consumes: `gin.cartographer.calibration_samples.{Sample, SampleManifest, write_samples, load_samples, DEFAULT_SAMPLES_PATH}`, `gin.curator.calibration_export.export_calibration_rows`, `gin.cartographer.combined.{CombinedRelationProposer, DEFAULT_EMBED_MODEL, DEFAULT_NLI_MODEL}`, `gin.cartographer.relatedness.make_same_story`
- Produces: `calibration.default_samples() -> list[Sample]` (unchanged signature, new source), `calibration.Sample` re-exported

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cartographer_calibration.py`:

```python
def test_default_samples_reads_the_generated_file(tmp_path, monkeypatch):
    # default_samples() must read the generated file, not a baked literal.
    import json

    from gin.cartographer import calibration
    from gin.cartographer.combined import DEFAULT_EMBED_MODEL, DEFAULT_NLI_MODEL

    payload = {
        "manifest": {
            "embed_model": DEFAULT_EMBED_MODEL, "nli_model": DEFAULT_NLI_MODEL,
            "n_samples": 2, "class_counts": {"contradicts": 1, "unrelated": 1},
            "excluded_eval_pairs": 45, "git_sha": "test",
            "created_utc": "2026-07-25T00:00:00Z",
        },
        "samples": [
            {"cos": 0.9, "p_contra": 0.9, "same_story": True, "relation": "contradicts"},
            {"cos": 0.01, "p_contra": 0.01, "same_story": False, "relation": "unrelated"},
        ],
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(calibration, "DEFAULT_SAMPLES_PATH", path)
    samples = calibration.default_samples()
    assert len(samples) == 2
    assert samples[0].cos == 0.9


def test_baked_measured_literal_is_gone():
    from gin.cartographer import calibration

    assert not hasattr(calibration, "_MEASURED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_calibration.py -v`
Expected: FAIL — `test_baked_measured_literal_is_gone` fails because `_MEASURED` still exists, and `test_default_samples_reads_the_generated_file` fails because `calibration.DEFAULT_SAMPLES_PATH` does not exist.

- [ ] **Step 3: Rewrite the sample source in `calibration.py`**

In `gin/cartographer/calibration.py`: delete the entire `_MEASURED` list and the local `Sample` dataclass, then replace `default_samples()`. The imports at the top become:

```python
from .calibration_samples import (  # noqa: F401  (Sample re-exported for importers)
    DEFAULT_SAMPLES_PATH,
    Sample,
    load_samples,
)
from .combined import DEFAULT_EMBED_MODEL, DEFAULT_NLI_MODEL, Thresholds, classify_relation
from .models import Relation
```

and `default_samples()` becomes:

```python
def default_samples() -> list[Sample]:
    """Calibration samples from the generated file.

    Previously a baked 39-tuple literal measured over labeled_set. That set was
    stale (the store holds far more labels) and overlapped the evaluation set,
    making reported accuracy partly in-sample. There is deliberately no fallback
    to the old literal: silently calibrating on stale samples is the failure
    being removed.
    """
    samples, _manifest = load_samples(
        DEFAULT_SAMPLES_PATH,
        expect_embed_model=DEFAULT_EMBED_MODEL,
        expect_nli_model=DEFAULT_NLI_MODEL,
    )
    return samples
```

Keep the existing `Thresholds` import behavior intact — `calibration.py` already imports `classify_relation` and `Thresholds` from `combined`; only add the two model-id names.

- [ ] **Step 4: Point the existing calibration tests at a fixture**

**Do this from the commit BEFORE the Step 3 edit**, so `default_samples()` still returns the baked 39. If you already applied Step 3, run `git stash` first and `git stash pop` after.

Run:
```bash
venv/Scripts/python.exe -c "
from gin.cartographer.calibration import default_samples
from gin.cartographer.calibration_samples import SampleManifest, write_samples
from gin.cartographer.combined import DEFAULT_EMBED_MODEL, DEFAULT_NLI_MODEL
from collections import Counter
s = default_samples()
m = SampleManifest(embed_model=DEFAULT_EMBED_MODEL, nli_model=DEFAULT_NLI_MODEL,
                   n_samples=len(s), class_counts=dict(Counter(x.relation.value for x in s)),
                   excluded_eval_pairs=0, git_sha='baked-39', created_utc='2026-07-13T00:00:00Z')
write_samples('tests/fixtures/calibration_samples_fixture.json', m, s, [])
print('wrote fixture with', len(s), 'samples')
"
```
Expected: `wrote fixture with 39 samples`

`write_samples` reads `.cos`/`.p_contra`/`.relation`/`.same_story` by attribute, so the old `Sample` class works here even before Step 3 swaps it out.

Then in `tests/test_cartographer_calibration.py`, any test that called `default_samples()` for its threshold expectations loads the fixture instead:

```python
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "calibration_samples_fixture.json"


def _fixture_samples():
    from gin.cartographer.calibration_samples import load_samples

    samples, _ = load_samples(FIXTURE)
    return samples
```

This keeps the historical threshold expectations (gate 0.140 / ceiling 0.486 / contra 0.686, LOO 0.897) meaningful and model-free — they now assert against a pinned fixture rather than whatever the live corpus happens to be.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_calibration.py -v`
Expected: PASS — including the two new tests, with the historical threshold expectations unchanged against the fixture.

- [ ] **Step 6: Write the generator script**

Create `scripts/regen_calibration_samples.py`:

```python
"""Measure calibration samples from the curator store. Model-bound; run rarely.

    venv/Scripts/python.exe scripts/regen_calibration_samples.py

Loads embed + NLI models, so this is the ONE place in the calibration path that
touches a model. Everything downstream reads the file it writes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.cartographer.calibration_samples import (
    DEFAULT_SAMPLES_PATH,
    EvalSample,
    Sample,
    SampleManifest,
    write_samples,
)
from gin.cartographer.combined import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_NLI_MODEL,
    CombinedRelationProposer,
)
from gin.cartographer.models import Relation
from gin.cartographer.relatedness import make_same_story
from gin.curator.calibration_export import export_calibration_rows
from gin.curator.store import Store
from gin.curator.text_index import default_text_index

DEFAULT_LABELS = ROOT / "data" / "curator" / "labels.jsonl"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate calibration samples")
    ap.add_argument("--log", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_SAMPLES_PATH)
    args = ap.parse_args()

    text = default_text_index()
    proposer = CombinedRelationProposer()
    same_story = make_same_story(list(text.values()))

    def signals(a_text: str, b_text: str) -> tuple[float, float, bool]:
        return (
            proposer.embedding_cosine(a_text, b_text),
            proposer._p_contra(a_text, b_text),  # noqa: SLF001 - same scorer the classifier uses
            same_story(a_text, b_text),
        )

    report = export_calibration_rows(Store(args.log), signals, text_index=text)
    samples = [
        Sample(
            cos=r["cos"], p_contra=r["p_contra"],
            relation=Relation(r["relation"]), same_story=r["same_story"],
        )
        for r in report.rows
    ]
    eval_samples = [
        EvalSample(
            src=r["src"], dst=r["dst"], cos=r["cos"], p_contra=r["p_contra"],
            relation=Relation(r["relation"]), same_story=r["same_story"],
        )
        for r in report.eval_rows
    ]
    manifest = SampleManifest(
        embed_model=DEFAULT_EMBED_MODEL,
        nli_model=DEFAULT_NLI_MODEL,
        n_samples=len(samples),
        class_counts=report.class_counts,
        excluded_eval_pairs=report.drops.get("eval_pair", 0),
        git_sha=git_sha(),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_samples(args.out, manifest, samples, eval_samples)
    print(f"measured {len(samples)} calibration samples {report.class_counts}")
    print(f"measured {len(eval_samples)} held-out eval samples")
    print(f"drops: {report.drops}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Generate the real sample file**

Run: `venv/Scripts/python.exe scripts/regen_calibration_samples.py`
Expected: `measured 131 calibration samples {...}`, `measured 45 held-out eval samples`, `drops: {'eval_pair': 45, 'not_a_classifier_output': 2}`, and a written `data/calibration/samples.json`.

This loads models and scores 131 pairs through a cross-encoder; allow several minutes.

- [ ] **Step 8: Run the full suite**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass. No test may require a model download.

- [ ] **Step 9: Commit**

```bash
git add gin/cartographer/calibration.py gin/cartographer/calibration_samples.py scripts/regen_calibration_samples.py tests/test_cartographer_calibration.py tests/fixtures/calibration_samples_fixture.json data/calibration/samples.json
git commit -m "Calibration: read generated samples, delete the baked 39-tuple literal"
```

---

### Task 5: Recalibration CLI with held-out score, sensitivity, and provenance

**Files:**
- Create: `scripts/recalibrate_cheap_pipeline.py`
- Modify: `data/cartographer_thresholds.json` (regenerated with provenance)

**Interfaces:**
- Consumes: `gin.cartographer.calibration.{calibrate, leave_one_out}`, `gin.cartographer.combined.{Thresholds, classify_relation, DEFAULT_EMBED_MODEL, DEFAULT_NLI_MODEL}`, `gin.cartographer.calibration_samples.{load_samples, load_eval_samples, EvalSample}`, `gin.cartographer.models.Relation`
- Produces: `data/cartographer_thresholds.json` with fields `gate_floor`, `corroborate_ceiling`, `contra_threshold`, `n_samples`, `leave_one_out_accuracy`, `leave_one_out_class_c_discrimination`, `embed_model`, `nli_model`, `git_sha`, `created_utc`

- [ ] **Step 1: Write the script**

Create `scripts/recalibrate_cheap_pipeline.py`:

```python
"""Recalibrate the cheap pipeline from the generated samples.

    venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py

Model-free: reads data/calibration/samples.json, grid-searches thresholds,
reports leave-one-out, and writes thresholds with provenance so a stale artifact
is detectable next time.

Pre-registered: report the number whichever way it moves. More calibration data
reducing accuracy is a real outcome, not a failure — it would mean the previous
39 baked samples were flattering.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from gin.cartographer.calibration import calibrate, leave_one_out
from gin.cartographer.calibration_samples import (
    DEFAULT_SAMPLES_PATH,
    EvalSample,
    load_eval_samples,
    load_samples,
)
from gin.cartographer.combined import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_NLI_MODEL,
    Thresholds,
    classify_relation,
)
from gin.cartographer.models import Relation

THRESHOLDS_PATH = ROOT / "data" / "cartographer_thresholds.json"
DISPUTED_PAIR = {"inst_em:0", "clim_pledges:0"}


def _score_held_out(eval_samples: list[EvalSample], t: Thresholds) -> float:
    """Fraction of held-out eval pairs the thresholds classify correctly."""
    if not eval_samples:
        return float("nan")
    correct = sum(
        classify_relation(e.cos, e.p_contra, t, same_story=e.same_story)[0] == e.relation
        for e in eval_samples
    )
    return correct / len(eval_samples)

# Measured 2026-07-25 on the 39 baked samples, before this recalibration.
BASELINE = {"n_samples": 39, "gate_floor": 0.140, "corroborate_ceiling": 0.486,
            "contra_threshold": 0.686, "loo_accuracy": 0.897, "loo_class_c": 1.000}


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Recalibrate cheap-pipeline thresholds")
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    ap.add_argument("--out", type=Path, default=THRESHOLDS_PATH)
    ap.add_argument("--write", action="store_true",
                    help="write the thresholds file (default: report only)")
    args = ap.parse_args()

    samples, manifest = load_samples(
        args.samples,
        expect_embed_model=DEFAULT_EMBED_MODEL,
        expect_nli_model=DEFAULT_NLI_MODEL,
    )
    thresholds = calibrate(samples)
    loo = leave_one_out(samples)

    print(f"samples: {len(samples)} {manifest.class_counts}")
    print(f"excluded eval pairs: {manifest.excluded_eval_pairs}")
    print()
    print("=== thresholds ===")
    print(f"  {'':22s} {'baseline(39)':>14s} {'recalibrated':>14s}")
    for name, base_key in (("gate_floor", "gate_floor"),
                           ("corroborate_ceiling", "corroborate_ceiling"),
                           ("contra_threshold", "contra_threshold")):
        print(f"  {name:22s} {BASELINE[base_key]:14.3f} {getattr(thresholds, name):14.3f}")
    print()
    print("=== leave-one-out ===")
    print(f"  {'accuracy':22s} {BASELINE['loo_accuracy']:14.3f} {loo.accuracy:14.3f}")
    cc = loo.class_c_discrimination
    cc_s = f"{cc:14.3f}" if cc is not None else f"{'n/a':>14s}"
    print(f"  {'class_c_discrimination':22s} {BASELINE['loo_class_c']:14.3f} {cc_s}")
    prec, rec = loo.contradicts_precision, loo.contradicts_recall
    print(f"  contradicts precision  {prec if prec is None else round(prec, 3)}")
    print(f"  contradicts recall     {rec if rec is None else round(rec, 3)}")

    eval_samples = load_eval_samples(args.samples)
    held_out = _score_held_out(eval_samples, thresholds)
    print()
    print("=== held-out (45 eval pairs, never calibrated on) ===")
    print(f"  accuracy               {held_out:14.3f}")

    print()
    print("=== disputed pair sensitivity ===")
    print("  inst_em:0 <-> clim_pledges:0 is a labeled_set member, hence an EVAL")
    print("  pair excluded from calibration — flipping it cannot move the")
    print("  thresholds. It moves the held-out score only.")
    flipped = [
        replace(e, relation=Relation.CONTRADICTS)
        if {e.src, e.dst} == DISPUTED_PAIR
        else e
        for e in eval_samples
    ]
    if flipped == eval_samples:
        print("  (pair not present in the eval samples — nothing to flip)")
    else:
        alt = _score_held_out(flipped, thresholds)
        print(f"  held-out as corroborates (current)  {held_out:.3f}")
        print(f"  held-out as contradicts             {alt:.3f}")
        print(f"  cost of adjudicating it to contradicts: {alt - held_out:+.3f}")

    if args.write:
        payload = {
            "gate_floor": thresholds.gate_floor,
            "corroborate_ceiling": thresholds.corroborate_ceiling,
            "contra_threshold": thresholds.contra_threshold,
            "n_samples": len(samples),
            "leave_one_out_accuracy": round(loo.accuracy, 4),
            "leave_one_out_class_c_discrimination": None if cc is None else round(cc, 4),
            "held_out_accuracy": round(held_out, 4),
            "held_out_n": len(eval_samples),
            "embed_model": manifest.embed_model,
            "nli_model": manifest.nli_model,
            "git_sha": git_sha(),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    else:
        print("\n(report only — pass --write to update the thresholds file)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run report-only and record the numbers**

Run: `venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py`
Expected: a thresholds comparison table, a leave-one-out block, a held-out block over the 45 eval pairs, and the disputed-pair sensitivity showing held-out accuracy both ways. **Record every number.** The result is the deliverable whichever direction it moves — do not re-run with different settings to find a better one.

- [ ] **Step 3: Write the thresholds file**

Run: `venv/Scripts/python.exe scripts/recalibrate_cheap_pipeline.py --write`
Expected: `wrote data/cartographer_thresholds.json`

- [ ] **Step 4: Confirm the escalation bar did not move**

Run: `venv/Scripts/python.exe -m pytest tests/test_cartographer_eval_pairs.py -v`
Expected: PASS — the bar pin still holds.

- [ ] **Step 5: Run the full suite**

Run: `venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all pass.

Some `scan_eval`/`evaluation` tests read the live thresholds file. If any now fail, that is the **held-out effect of recalibration** and is a real result — report the before/after numbers rather than reverting the thresholds or loosening the test.

- [ ] **Step 6: Commit**

```bash
git add scripts/recalibrate_cheap_pipeline.py data/cartographer_thresholds.json
git commit -m "Calibration: recalibrate from the curated corpus with provenance"
```

---

## Post-Implementation

Record the measured outcome in the spec's own results section and in `architecture.md`: sample count, old vs new thresholds, LOO accuracy and `class_c_discrimination` against the 0.897 / 1.000 baseline, and the held-out effect on `scan_eval`/`evaluation`. If accuracy fell, say so plainly and state that a wider, non-overlapping sample is the more trustworthy number — do not present the superseded 0.897 as the headline.
