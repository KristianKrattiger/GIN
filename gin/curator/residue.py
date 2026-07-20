"""Candidate source that surfaces the issue_frame residue for labeling.

Reuses cartographer.escalation.escalation_candidates for the residue FILTER
(not same-story, cosine >= floor) so what the curator labels stays aligned with
what the escalation bar tests — then re-ranks the result before capping.

Ranking (corrected 2026-07-20 from real corpus_node4 evidence): real opposed
pairs on one proposition share dense vocabulary, so issue_frame lives at HIGH
cosine, not the mid band the module originally assumed. A strong NLI
contradiction is the sharpest instance, but framing/efficacy divergences (which
no cheap signal reliably flags — the reason the LLM escalation judge exists) are
also high-cosine. So the residue floats NLI contradictions first (by strength),
then ranks the rest cosine-descending. High-cosine AGREE pairs surface here too,
which is wanted — the readiness gauge also needs agree labels. Implements
sub-project A's CandidateSource protocol.

Bounded NLI (2026-07-20, cost fix): NLI is a per-pair cross-encoder call, not a
lookup — running it over every residue pair took ~7 minutes on the 197-chunk
node4 corpus, which is long enough to hang the curator UI's first page load
(``/curator/next`` calls ``source.pairs()`` synchronously). A human curator only
ever looks at the front of the ranked list, so NLI is consulted only for the
``nli_rank_limit`` highest-cosine residue pairs; everything past that limit is
ranked by cosine alone and never pays for a cross-encoder call.
``CombinedRelationProposer`` also memoizes ``nli_p_contra`` per unordered text
pair (mirroring its embedding cache), so a repeated pair — e.g. across multiple
``EscalationResidueCandidateSource`` instances sharing one proposer — is free.

Tension with the typer, recorded honestly: every residue pair is, by
construction, not-same-story — and ``combined.classify_relation`` deliberately
BLOCKS the NLI channel when ``same_story is False`` (a documented cross-topic
numeric-claim artifact — see ``combined.py``). So this ranking consults NLI on
exactly the population the typer distrusts for typing. That is a deliberate,
human-approved choice: here NLI is a ranking HINT that helps a human adjudicator
triage, not an automated edge-type decision — every surfaced pair still gets a
human label. One consequence: ``gin/curator/signals.py`` (via ``type_relation``)
reports ``nli_p_contra: None`` for these same not-same-story pairs, so the
curator's signal panel will show no NLI value even for pairs this ranking moved
to the top on the strength of one.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.escalation import (
    DEFAULT_ESCALATION_COS_FLOOR,
    DEFAULT_MAX_CANDIDATES,
    escalation_candidates,
)
from gin.cartographer.models import LabeledChunk
from gin.cartographer.scan import wire_same_story

from .models import pair_key

DEFAULT_NLI_RANK_LIMIT = 400


class EscalationResidueCandidateSource:
    """A.CandidateSource over the escalation residue of a corpus."""

    # pairs() already returns the evidence-based ranking (NLI contradictions
    # first, then cosine-descending) — app.next_pairs must not re-sort it.
    pre_ranked = True

    def __init__(
        self,
        chunks: list[LabeledChunk],
        *,
        proposer: Optional[CombinedRelationProposer] = None,
        cos_floor: float = DEFAULT_ESCALATION_COS_FLOOR,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        nli_rank_limit: int = DEFAULT_NLI_RANK_LIMIT,
    ) -> None:
        self._chunks = list(chunks)
        if proposer is None:
            proposer = CombinedRelationProposer()
        # escalation_candidates needs the stage-1 same-story provider. This
        # no-ops when the proposer already has one (an injected fake), so it is
        # safe to call unconditionally.
        wire_same_story(proposer, self._chunks)
        self._proposer = proposer
        self._cos_floor = cos_floor
        self._max_candidates = max_candidates
        self._nli_rank_limit = nli_rank_limit
        self._pairs: Optional[list[tuple[LabeledChunk, LabeledChunk]]] = None

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        # Static for a given corpus + proposer — compute once.
        if self._pairs is not None:
            return self._pairs
        all_pairs = list(combinations(self._chunks, 2))
        # Full residue (pass an unbounded cap so the cosine-desc slice inside
        # escalation_candidates doesn't drop the moderate-cosine band before we
        # re-rank).
        residue = escalation_candidates(
            all_pairs,
            self._proposer,
            cos_floor=self._cos_floor,
            max_candidates=len(all_pairs),
        )

        def _tie(pair: tuple[LabeledChunk, LabeledChunk]) -> tuple[str, str]:
            return pair_key(pair[0].chunk_id, pair[1].chunk_id)

        def _cos(pair: tuple[LabeledChunk, LabeledChunk]) -> float:
            return self._proposer.embedding_cosine(pair[0].text, pair[1].text)

        # Cosine-descending first (deterministic ties by chunk-id pair) — this
        # is the ordering NLI is bounded against, so only the pairs a curator
        # would plausibly reach get a cross-encoder call at all.
        residue_by_cos = sorted(residue, key=lambda p: (-_cos(p), _tie(p)))
        head = residue_by_cos[: self._nli_rank_limit]
        tail = residue_by_cos[self._nli_rank_limit :]

        contra_threshold = self._proposer.thresholds.contra_threshold
        # A high p_contra only counts as issue_frame evidence when the pair is
        # topically close enough for a shared proposition to be plausible.
        # Cross-topic pairs that merely share numeric/economic vocabulary score
        # very high p_contra — the artifact classify_relation story-gates NLI
        # off for, and every residue pair is by construction in that
        # story-gated population. Measured in this corpus: monetary-policy x
        # climate pairs at cos 0.32-0.36 scoring p_contra 0.87-0.93, which
        # without this floor would rank ABOVE every genuine issue_frame pair.
        # Reuses the calibrated corroborate_ceiling — the same "these are
        # topically close" boundary classify_relation uses.
        contra_cos_floor = self._proposer.thresholds.corroborate_ceiling
        contras: list[tuple[float, tuple[LabeledChunk, LabeledChunk]]] = []
        rest_head: list[tuple[LabeledChunk, LabeledChunk]] = []
        for pair in head:
            if _cos(pair) < contra_cos_floor:
                rest_head.append(pair)  # too far apart to trust an NLI contradiction
                continue
            p_contra = self._proposer.nli_p_contra(pair[0].text, pair[1].text)
            if p_contra >= contra_threshold:
                contras.append((p_contra, pair))
            else:
                rest_head.append(pair)
        contras.sort(key=lambda item: (-item[0], _tie(item[1])))
        contra_pairs = [pair for _p, pair in contras]

        # Everything else — head pairs that didn't clear the NLI bar, plus the
        # whole tail (NLI never consulted) — ranked by cosine alone.
        remaining = rest_head + tail
        remaining.sort(key=lambda p: (-_cos(p), _tie(p)))

        ordered = contra_pairs + remaining
        self._pairs = ordered[: self._max_candidates]
        return self._pairs
