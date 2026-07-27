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
from dataclasses import MISSING, asdict, dataclass, fields
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
    # Per-fact quantity stance (quantity.stance_for): "conflict" | "revision" |
    # "partial" | "agreement" | "unaligned", or None. None and "unaligned" are
    # NOT interchangeable -- None means the channel had no quantitative claim to
    # judge and classify_relation falls through to its pre-stance branch, while
    # "unaligned" means it looked and found no shared fact and the branch
    # abstains. Defaulted so the committed 39-sample fixture, whose rows predate
    # this field, keeps loading.
    stance: Optional[str] = None


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
    # See Sample.stance. None vs "unaligned" is a real distinction, not a
    # missing value -- preserve it through serialization.
    stance: Optional[str] = None


@dataclass(frozen=True)
class SampleManifest:
    embed_model: str
    nli_model: str
    n_samples: int
    class_counts: dict[str, int]
    excluded_eval_pairs: int
    git_sha: str
    created_utc: str
    # Stage-1 same-story parameters actually used to build make_same_story()
    # for this run. same_story determined the entire recalibration outcome
    # (11 same-story rows under the generator's settings, 3 under scan-like
    # settings) and is both corpus- and parameter-dependent, so it belongs in
    # the manifest gate alongside the model ids. Defaulted so the committed
    # 39-sample fixture, whose manifest predates these fields, keeps loading.
    same_story_corpus_size: int = 0
    story_floor: int = 0
    df_ceiling: int | None = None
    require_anchor: bool = True
    # Identity of the stance provider these rows were measured with. stance now
    # decides the same-story arm outright, so a sample file measured under a
    # different provider must not silently calibrate the live pipeline -- the
    # same reasoning as the model ids and the same_story parameters above.
    # "none" means the rows predate the stance channel.
    stance_provider: str = "none"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "SampleManifest":
        valid_fields = {f.name for f in fields(cls)}
        unexpected = set(data) - valid_fields
        missing = {f.name for f in fields(cls) if f.default is MISSING} - set(data)
        if unexpected or missing:
            problems = []
            if unexpected:
                problems.append(f"unexpected key(s) {sorted(unexpected)}")
            if missing:
                problems.append(f"missing required key(s) {sorted(missing)}")
            raise ValueError(
                f"calibration samples manifest schema mismatch: {'; '.join(problems)}; "
                f"regenerate with: {REGEN_COMMAND}"
            )
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
                "stance": s.stance,
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
                "stance": e.stance,
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
            stance=row.get("stance"),
        )
        for row in payload["samples"]
    ]
    if not samples:
        raise ValueError(
            f"calibration samples array is empty; regenerate with: {REGEN_COMMAND}"
        )
    if len(samples) != manifest.n_samples:
        raise ValueError(
            f"samples count ({len(samples)}) does not match manifest "
            f"({manifest.n_samples}); file may be truncated or corrupted"
        )
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
            stance=row.get("stance"),
        )
        for row in payload.get("eval_samples", [])
    ]
