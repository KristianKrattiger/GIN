"""Cross-encoder sweep for the question sub-project B left open.

B's handoff asks whether framing divergence is recoverable from a frozen encoder
*at all*, before more curation effort is spent. Answering it takes more than
re-running the stage-0 probe under a different model name, because the aggregate
DIVERGENT-vs-rest number does not measure framing.

Of the 24 DIVERGENT rows in the current 102-row training set, 22 are ``n4_doc_*``
proposition-level policy pro/con -- the class B measured at 22/22 leave-one-out
recall -- and 2 are framing divergence. An encoder that gains nothing whatsoever
on framing can still post a high aggregate on node4 alone. That is precisely how
0.939 was produced, and a sweep that reported only the aggregate would reproduce
the same misreading once per candidate model.

So each encoder is scored three ways, and only the last two bear on the question:

  1. aggregate -- DIVERGENT-vs-rest LOO balanced accuracy, kept for continuity
     with the published 0.939 and for nothing else
  2. by origin -- the same held-out predictions split node4 / framing
  3. bar       -- issue_frame recall on the 4 held-out ``n1<->n2`` bar pairs

Sample sizes, stated here rather than in a footnote. The corpus holds 31
issue_frame gold pairs: 22 node4, 4 in the escalation bar, 3 ``inst_*<->grass_*``
and 2 ``hf_*``. The 3 inst/grass rows are byte-identical aliases of bar chunks
and are dropped by ``build_dataset``'s ``bar_text_alias`` guard, so **2 framing
rows are trainable**. Metric (2) is therefore a screen, not a verdict -- see
``ORIGIN_SAMPLE_FLOOR`` -- and the 4 bar pairs in (3) are the only clean framing
measurement the corpus currently has. Reading a verdict off 2 rows is the error
this module is built to prevent, so the report labels it in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut

from .dataset import FrameExample
from .encoder import ChunkEncoder, feature_matrix
from .eval import bar_metrics
from .head import train_head
from .judge import BiEncoderFrameJudge
from .labels import FrameClass
from .probe import divergent_vs_rest

# Candidates span training objective, not just capacity. If framing divergence is
# missing from MiniLM because the model is small, mpnet recovers it; if it is
# missing because contrastive STS objectives encode topic rather than stance,
# only the NLI-trained encoder has a reason to carry it. A sweep that varied size
# alone could not tell those two explanations apart.
#
# NOTE on e5/bge: both are trained with "query:"/"passage:" prefixes. The sweep
# deliberately does not add them -- pair_features is symmetric by construction,
# so there is no query/passage role to assign, and inventing one would break the
# direction-invariance that makes direction_flip_count == 0 an identity. These
# two therefore run slightly off their intended protocol, which is a real caveat
# on a null result from them specifically and is reported as such.
CANDIDATE_ENCODERS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",      # incumbent, for continuity
    "sentence-transformers/all-mpnet-base-v2",     # same objective, more capacity
    "sentence-transformers/nli-mpnet-base-v2",     # NLI objective -- stance-adjacent
    "intfloat/e5-large-v2",                        # different objective, prefix-trained
    "BAAI/bge-large-en-v1.5",                      # different objective, prefix-trained
)

NODE4_PREFIX = "n4_doc_"
NODE4_ORIGIN = "node4_policy"
FRAMING_ORIGIN = "framing"

# Below this many rows an origin's recall is reported but carries no verdict.
# Set from the data (2 trainable framing rows), not from a rule of thumb: the
# point is that no threshold choice makes 2 rows decisive.
ORIGIN_SAMPLE_FLOOR = 5


def divergent_origin(src_chunk_id: str, dst_chunk_id: str) -> str:
    """Which phenomenon a DIVERGENT row is an example of.

    node4 pairs are proposition-level policy pro/con generated for sub-project B;
    everything else labeled issue_frame is cross-document framing divergence.
    The split is by corpus of origin because that is what distinguishes the two
    phenomena -- node4 was built as one document set, and B's 22/22-vs-0/5 result
    is exactly this partition.
    """
    if src_chunk_id.startswith(NODE4_PREFIX) and dst_chunk_id.startswith(NODE4_PREFIX):
        return NODE4_ORIGIN
    return FRAMING_ORIGIN


@dataclass(frozen=True)
class OriginRecall:
    origin: str
    n: int
    n_recovered: int
    chunk_pairs: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.n_recovered / self.n if self.n else float("nan")

    @property
    def decisive(self) -> bool:
        """False when the origin has too few rows to support a conclusion."""
        return self.n >= ORIGIN_SAMPLE_FLOOR


@dataclass(frozen=True)
class EncoderResult:
    model_name: str
    aggregate_balanced_accuracy: float
    by_origin: list[OriginRecall]
    bar: Optional[dict]
    error: Optional[str] = None

    @property
    def framing_recall(self) -> float:
        for row in self.by_origin:
            if row.origin == FRAMING_ORIGIN:
                return row.recall
        return float("nan")

    @property
    def bar_issue_frame_recall(self) -> Optional[float]:
        return None if self.bar is None else self.bar.get("issue_frame_recall")


def _loo_predictions(X: np.ndarray, target: np.ndarray, seed: int = 0) -> np.ndarray:
    """Held-out DIVERGENT-vs-rest predictions, one per row.

    Same estimator configuration as ``probe.run_probe`` on purpose: the aggregate
    computed from these predictions must equal the published probe number, or the
    origin split would be describing a different model than the headline does.
    A test pins that equality.
    """
    predictions = np.empty_like(target)
    for train_idx, test_idx in LeaveOneOut().split(X):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
        clf.fit(X[train_idx], target[train_idx])
        predictions[test_idx] = clf.predict(X[test_idx])
    return predictions


def recall_by_origin(
    examples: list[FrameExample], target: np.ndarray, predictions: np.ndarray
) -> list[OriginRecall]:
    """Split held-out DIVERGENT recall by the phenomenon each row exemplifies."""
    buckets: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        if target[index] != 1:
            continue
        origin = divergent_origin(example.src_chunk_id, example.dst_chunk_id)
        buckets.setdefault(origin, []).append(index)

    rows: list[OriginRecall] = []
    for origin in (NODE4_ORIGIN, FRAMING_ORIGIN):
        indices = buckets.get(origin, [])
        if not indices:
            continue
        rows.append(
            OriginRecall(
                origin=origin,
                n=len(indices),
                n_recovered=int(sum(1 for i in indices if predictions[i] == 1)),
                chunk_pairs=[
                    f"{examples[i].src_chunk_id} <-> {examples[i].dst_chunk_id}"
                    for i in indices
                    if predictions[i] != 1
                ],
            )
        )
    return rows


def sweep_encoder(
    examples: list[FrameExample],
    encoder: ChunkEncoder,
    *,
    seed: int = 0,
    score_bar: bool = True,
) -> EncoderResult:
    """Score one frozen encoder on all three metrics.

    The bar is scored with a head trained on every row, which is sound precisely
    because ``build_dataset`` drops bar chunks *and* their text aliases -- the 4
    issue_frame bar pairs cannot be in ``examples``.
    """
    X, y = feature_matrix(examples, encoder)
    target = divergent_vs_rest(y)
    predictions = _loo_predictions(X, target, seed=seed)

    aggregate = float(balanced_accuracy_score(target, predictions))
    by_origin = recall_by_origin(examples, target, predictions)

    bar = None
    if score_bar:
        judge = BiEncoderFrameJudge(train_head(X, y, kind="linear", seed=seed), encoder)
        bar = bar_metrics(judge)

    return EncoderResult(
        model_name=encoder.model_name,
        aggregate_balanced_accuracy=aggregate,
        by_origin=by_origin,
        bar=bar,
    )


def format_result(result: EncoderResult) -> str:
    """Human-readable block for one encoder, with the caveats attached in place."""
    lines = [f"=== {result.model_name} ==="]
    if result.error:
        lines.append(f"  ERROR: {result.error}")
        return "\n".join(lines)

    lines.append(
        f"  aggregate DIVERGENT-vs-rest LOO : {result.aggregate_balanced_accuracy:.3f}"
        "   (carried by node4 -- not the answer)"
    )
    for row in result.by_origin:
        note = "" if row.decisive else f"   [n<{ORIGIN_SAMPLE_FLOOR}, screen only]"
        lines.append(
            f"  {row.origin:<14} recall           : {row.recall:.3f} "
            f"({row.n_recovered}/{row.n}){note}"
        )
        for pair in row.chunk_pairs:
            lines.append(f"      missed: {pair}")

    if result.bar is not None:
        lines.append(
            f"  bar issue_frame recall (n=4)    : {result.bar.get('issue_frame_recall'):.3f}"
            "   <- the clean framing measurement"
        )
        lines.append(
            f"  bar class_c / unrelated / flips : "
            f"{result.bar.get('class_c_discrimination'):.3f} / "
            f"{result.bar.get('unrelated_discrimination'):.3f} / "
            f"{result.bar.get('direction_flip_count')}"
        )
    return "\n".join(lines)


def verdict(results: list[EncoderResult]) -> str:
    """The reading the sweep licenses, decided by the bar rather than the aggregate.

    Fixed before any number is seen, in the same spirit as the stage-0 gate: if
    no candidate encoder recovers a single held-out bar issue_frame pair, the
    frozen-encoder path is exhausted for framing and fine-tuning becomes the live
    question rather than a deferred fallback. Any encoder that does recover pairs
    makes an encoder swap the cheaper move than more curation.
    """
    scored = [r for r in results if r.error is None and r.bar_issue_frame_recall is not None]
    if not scored:
        return "no_measurement"
    best = max(scored, key=lambda r: r.bar_issue_frame_recall or 0.0)
    if (best.bar_issue_frame_recall or 0.0) > 0.0:
        return f"framing_recoverable:{best.model_name}"
    return "framing_not_recoverable_frozen"
