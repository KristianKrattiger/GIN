"""
sear.processor
--------------
Hand-rolled LogitsProcessor implementing cursor-based copy constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .corpus import Corpus

NEG_INF = -1e30

BOUNDARY, IN_SPAN, IN_CONNECTIVE, IN_CITE = "BOUNDARY", "IN_SPAN", "IN_CONNECTIVE", "IN_CITE"


@dataclass
class Segment:
    token_ids: list[int]
    sources: list[tuple[int, int, int]]
    kind: str
    guidance: str = ""


class ExtractiveCopyConstraint:
    def __init__(
        self,
        corpus: Corpus,
        prompt_len: int,
        eos_id: int,
        delim_id: int,
        min_span_len: int = 3,
        *,
        connective_starts: Optional[frozenset[int]] = None,
        connective_continuations: Optional[dict[int, frozenset[int]]] = None,
        connective_phrases: Optional[dict[int, list[int]]] = None,
        max_connective_len: int = 6,
        cite_ids: Optional[dict[int, int]] = None,
        cite_sequences_by_doc: Optional[dict[int, list[int]]] = None,
        cite_continuations: Optional[dict[int, frozenset[int]]] = None,
        close_on_doc_divergence: bool = False,
        required_doc_groups: Optional[list[frozenset[int]]] = None,
        focus_doc_indices: Optional[frozenset[int]] = None,
        reject_ambiguous_spans: bool = False,
        allow_shared_prefix: bool = True,
        span_must_start_at_sentence: bool = False,
        require_cite_after_extract: bool = False,
        stop_when_groups_satisfied: bool = False,
        block_eos_until_groups_satisfied: bool = False,
        force_connective_ids: Optional[frozenset[int]] = None,
        preferred_starts: Optional[set[tuple[int, int]]] = None,
        forbidden_starts: Optional[set[tuple[int, int]]] = None,
        divergence_starts: Optional[dict[int, set[int]]] = None,
        require_divergence_after_first: bool = False,
        span_must_close_at_sentence_end: bool = False,
        divergence_sentence_ends: Optional[dict[int, dict[int, int]]] = None,
        ranked_sentence_starts: Optional[list[tuple[int, int, float]]] = None,
    ):
        self.corpus = corpus
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.delim_id = delim_id
        self.min_span_len = min_span_len
        self.connective_starts = connective_starts or frozenset()
        self.connective_continuations = connective_continuations or {}
        self.connective_phrases = connective_phrases or {}
        self.max_connective_len = max_connective_len
        self.cite_ids = cite_ids or {}
        self.cite_sequences_by_doc = cite_sequences_by_doc or {}
        self.cite_continuations = cite_continuations or {}
        self.close_on_doc_divergence = close_on_doc_divergence
        self.required_doc_groups = required_doc_groups or []
        self.focus_doc_indices = focus_doc_indices
        self.reject_ambiguous_spans = reject_ambiguous_spans
        self.allow_shared_prefix = allow_shared_prefix
        self.span_must_start_at_sentence = span_must_start_at_sentence
        self.require_cite_after_extract = require_cite_after_extract
        self.stop_when_groups_satisfied = stop_when_groups_satisfied
        self.block_eos_until_groups_satisfied = block_eos_until_groups_satisfied
        self.force_connective_ids = force_connective_ids or frozenset()
        self.preferred_starts = preferred_starts or set()
        self.forbidden_starts = forbidden_starts or set()
        self.divergence_starts = divergence_starts or {}
        self.require_divergence_after_first = require_divergence_after_first
        self.span_must_close_at_sentence_end = span_must_close_at_sentence_end
        self.divergence_sentence_ends = divergence_sentence_ends or {}
        self.ranked_sentence_starts = ranked_sentence_starts or []

        self.structural = {eos_id, delim_id}
        self.structural.update(self.cite_ids.keys())

        self.mode = BOUNDARY
        self.cursors: list[tuple[int, int]] = []
        self.span_start: list[tuple[int, int]] = []
        self._span_start_docs: set[int] = set()
        self.span_len = 0
        self.segments: list[Segment] = []
        self._cur_tokens: list[int] = []
        self._seen = prompt_len
        self._used_positions: set[tuple[int, int]] = set()
        self._has_closed_extract = False
        self._connective_phrase: list[int] = []
        self._connective_pos = 0
        self._connective_len = 0
        self._pending_cite_docs: set[int] = set()
        self._quoted_docs: set[int] = set()
        self._cite_gate_active = False
        self._cite_accum: list[int] = []
        self._span_anchor: tuple[int, int] | None = None
        self._current_span_guidance: str = ""

    def _unquoted_docs_in_first_unsatisfied_group(self) -> set[int]:
        for group in self.required_doc_groups:
            if not group <= self._quoted_docs:
                return set(group - self._quoted_docs)
        return set()

    def _doc_start_permitted(self, doc_idx: int) -> bool:
        if self.focus_doc_indices is not None and doc_idx not in self.focus_doc_indices:
            return False
        if self.required_doc_groups and not self._groups_satisfied():
            unquoted = self._unquoted_docs_in_first_unsatisfied_group()
            if unquoted and doc_idx not in unquoted:
                return False
        return True

    def _position_start_permitted(self, doc: int, pos: int) -> bool:
        if not self._doc_start_permitted(doc):
            return False
        if (doc, pos) in self.forbidden_starts:
            return False
        if self.span_must_start_at_sentence and self.corpus.sentence_starts:
            if (doc, pos) not in self.corpus.sentence_starts:
                return False
        if self.require_divergence_after_first and self.divergence_starts:
            unquoted = self._unquoted_docs_in_first_unsatisfied_group()
            if unquoted and doc in unquoted:
                div = self.divergence_starts.get(doc, set())
                if div and pos not in div:
                    return False
        if (
            self.preferred_starts
            and not self._has_closed_extract
            and not self.require_divergence_after_first
        ):
            if (doc, pos) not in self.preferred_starts:
                return False
        return True

    def _resolve_ambiguous_starts(
        self,
        occurrences: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Narrow multi-doc token matches to steered unquoted docs instead of rejecting."""
        if not occurrences:
            return occurrences
        if self.allow_shared_prefix:
            return occurrences
        docs = {d for d, _ in occurrences}
        if len(docs) <= 1:
            return occurrences
        if not self.reject_ambiguous_spans:
            return occurrences

        unquoted = self._unquoted_docs_in_first_unsatisfied_group()
        if self.required_doc_groups and unquoted:
            filtered = [(d, p) for d, p in occurrences if d in unquoted]
        else:
            filtered = list(occurrences)

        if not filtered:
            return []

        sub_docs = {d for d, _ in filtered}
        if len(sub_docs) <= 1:
            return filtered

        preferred = [
            (d, p) for d, p in filtered if (d, p) in self.preferred_starts
        ]
        if preferred:
            pref_docs = {d for d, _ in preferred}
            if len(pref_docs) == 1:
                target = min(pref_docs)
                return [(d, p) for d, p in preferred if d == target]
            return [min(preferred, key=lambda dp: (dp[0], dp[1]))]

        return [(d, p) for d, p in filtered if d == min(sub_docs)]

    def _permitted_starts_for_token(self, tok: int) -> list[tuple[int, int]]:
        occurrences = [
            (d, p)
            for (d, p) in self.corpus.start_index.get(tok, [])
            if (d, p) not in self._used_positions and self._position_start_permitted(d, p)
        ]
        return self._resolve_ambiguous_starts(occurrences)

    def _corpus_starts_allowed(self) -> set[int]:
        allowed: set[int] = set()
        for t in self.corpus.start_index:
            if self._permitted_starts_for_token(t):
                allowed.add(t)
        return allowed

    def _connective_eligible(self) -> bool:
        if not self._has_closed_extract or self._cite_gate_active:
            return False
        if self.block_eos_until_groups_satisfied and not self._groups_satisfied():
            return False
        return True

    def _pending_cite_sequences(self) -> list[tuple[int, list[int]]]:
        return [
            (doc, seq)
            for doc in self._pending_cite_docs
            if (seq := self.cite_sequences_by_doc.get(doc))
        ]

    def _cite_start_tokens_for_pending(self) -> set[int]:
        if not self._pending_cite_docs:
            return set()
        starts: set[int] = set()
        for _doc, seq in self._pending_cite_sequences():
            if seq:
                starts.add(seq[0])
        return starts

    def _cite_next_tokens_for_pending(self, prefix: list[int]) -> set[int]:
        nxt: set[int] = set()
        plen = len(prefix)
        for _doc, seq in self._pending_cite_sequences():
            if len(seq) > plen and seq[:plen] == prefix:
                nxt.add(seq[plen])
        return nxt

    def _matching_cite_docs(self, tokens: list[int]) -> list[int]:
        return [
            doc for doc, seq in self._pending_cite_sequences()
            if seq == tokens
        ]

    def _cite_tokens_allowed(self) -> set[int]:
        return self._cite_start_tokens_for_pending()

    def _eos_permitted(self) -> bool:
        if self._cite_gate_active and self._cite_start_tokens_for_pending():
            return False
        if self.block_eos_until_groups_satisfied and not self._groups_satisfied():
            return False
        return True

    def _cursor_at_sentence_end(self, doc: int, pos: int) -> bool:
        if (doc, pos) in self.corpus.sentence_ends:
            return True
        return self.corpus.continuation(doc, pos) is None

    def _span_close_permitted(self) -> bool:
        if self.span_len < self.min_span_len or not self.cursors:
            return False
        for doc, pos in self.cursors:
            if self._span_anchor and self.divergence_sentence_ends:
                anchor_doc, anchor_start = self._span_anchor
                if doc == anchor_doc:
                    required_end = self.divergence_sentence_ends.get(doc, {}).get(
                        anchor_start
                    )
                    if required_end is not None and pos < required_end:
                        return False
            if self.span_must_close_at_sentence_end:
                if not self._cursor_at_sentence_end(doc, pos):
                    return False
        return True

    def _groups_satisfied(self) -> bool:
        if not self.required_doc_groups:
            return False
        return all(group <= self._quoted_docs for group in self.required_doc_groups)

    def _allowed(self) -> set[int]:
        if self.mode == IN_CITE:
            nxt = self._cite_next_tokens_for_pending(self._cite_accum)
            if nxt:
                return nxt
            return {self.eos_id}

        if self.mode == BOUNDARY:
            if self.stop_when_groups_satisfied and self._groups_satisfied():
                return {self.eos_id}
            if self._cite_gate_active and self._pending_cite_docs:
                allowed = self._cite_start_tokens_for_pending()
                if not allowed:
                    return {self.eos_id}
                return allowed
            allowed = self._corpus_starts_allowed()
            if self._eos_permitted():
                allowed.add(self.eos_id)
            if self._connective_eligible():
                allowed.update(self.connective_starts)
                allowed.update(self._cite_start_tokens_for_pending())
            return allowed

        if self.mode == IN_CONNECTIVE:
            allowed = self._corpus_starts_allowed()
            if self._connective_pos < len(self._connective_phrase):
                allowed.add(self._connective_phrase[self._connective_pos])
            return allowed

        allowed: set[int] = set()
        for (d, p) in self.cursors:
            c = self.corpus.continuation(d, p)
            if c is not None and (d, p + 1) not in self._used_positions:
                allowed.add(c)
        if self.span_len >= self.min_span_len and self._span_close_permitted():
            allowed.add(self.delim_id)
            if self._eos_permitted():
                allowed.add(self.eos_id)
        if not allowed:
            if self._eos_permitted():
                allowed.add(self.eos_id)
        return allowed

    def _start_connective(self, tok: int) -> None:
        phrase = self.connective_phrases.get(tok, [tok])
        self.segments.append(Segment([tok], [], "connective"))
        if len(phrase) > 1:
            self._connective_phrase = phrase
            self._connective_pos = 1
            self._connective_len = 1
            self.mode = IN_CONNECTIVE
        else:
            self._reset_connective()

    def _reset_cite(self) -> None:
        self._cite_accum = []
        self.mode = BOUNDARY

    def _finish_cite(self, tokens: list[int], doc: int) -> None:
        self.segments.append(Segment(tokens, [(doc, -1, -1)], "cite"))
        self._pending_cite_docs.discard(doc)
        if not self._pending_cite_docs:
            self._cite_gate_active = False
        self._reset_cite()

    def _try_emit_cite(self, tok: int) -> bool:
        if not self._pending_cite_docs:
            return False
        if not self._cite_gate_active and not self._connective_eligible():
            if tok not in self.cite_ids or self.cite_ids[tok] not in self._pending_cite_docs:
                return False
        if tok not in self._cite_start_tokens_for_pending():
            if tok in self.cite_ids and self.cite_ids[tok] in self._pending_cite_docs:
                self._finish_cite([tok], self.cite_ids[tok])
                return True
            return False
        matching = [
            (doc, seq) for doc, seq in self._pending_cite_sequences()
            if seq and seq[0] == tok
        ]
        if not matching:
            return False
        if len(matching) == 1 and len(matching[0][1]) == 1:
            self._finish_cite([tok], matching[0][0])
            return True
        self._cite_accum = [tok]
        self.mode = IN_CITE
        return True

    def _reset_connective(self) -> None:
        self._connective_phrase = []
        self._connective_pos = 0
        self._connective_len = 0
        self.mode = BOUNDARY

    def _append_connective_token(self, tok: int) -> None:
        self.segments.append(Segment([tok], [], "connective"))
        self._connective_len += 1

    def _maybe_auto_close_on_divergence(self) -> None:
        if not self.close_on_doc_divergence:
            return
        if self.span_len < self.min_span_len:
            return
        if len(self._span_start_docs) <= 1:
            return
        current_docs = {d for d, _ in self.cursors}
        if current_docs and len(current_docs) < len(self._span_start_docs):
            self._close_span()

    def _begin_span(self, tok: int, cursors: list[tuple[int, int]]) -> None:
        self.cursors = cursors
        self.span_start = list(self.cursors)
        self._span_start_docs = {d for d, _ in self.cursors}
        self._span_anchor = cursors[0] if len(cursors) == 1 else None
        self.span_len = 1
        self._cur_tokens = [tok]
        self.mode = IN_SPAN
        guidance = ""
        for d, p in cursors:
            if p in self.divergence_starts.get(d, set()):
                guidance = "divergence-steered"
                break
            if (d, p) in self.preferred_starts:
                guidance = "steered"
        self._current_span_guidance = guidance

    def _try_start_span(self, tok: int) -> bool:
        cursors = self._permitted_starts_for_token(tok)
        if not cursors:
            return False
        self._begin_span(tok, cursors)
        return True

    def _consume(self, tok: int) -> None:
        if self.mode == IN_CITE:
            if tok in self._cite_next_tokens_for_pending(self._cite_accum):
                self._cite_accum.append(tok)
                matches = self._matching_cite_docs(self._cite_accum)
                if matches:
                    self._finish_cite(list(self._cite_accum), matches[0])
            return

        if self.mode == BOUNDARY:
            if tok == self.eos_id:
                return
            if tok == self.delim_id:
                self.segments.append(Segment([tok], [], "connective"))
                return
            if self._try_emit_cite(tok):
                return
            if self._connective_eligible() and tok in self.connective_starts:
                if tok in self.force_connective_ids:
                    self._start_connective(tok)
                    return
                if not self._permitted_starts_for_token(tok):
                    self._start_connective(tok)
                    return
            if self._try_start_span(tok):
                return
            if self._connective_eligible() and tok in self.connective_starts:
                self._start_connective(tok)
            return

        if self.mode == IN_CONNECTIVE:
            if (
                self._connective_pos < len(self._connective_phrase)
                and tok == self._connective_phrase[self._connective_pos]
            ):
                self._append_connective_token(tok)
                self._connective_pos += 1
                if self._connective_pos >= len(self._connective_phrase):
                    self._reset_connective()
                return
            if self._try_start_span(tok):
                self._reset_connective()
            return

        if tok in self.structural:
            self._close_span()
            if tok == self.delim_id:
                self.segments.append(Segment([tok], [], "connective"))
            self.mode = BOUNDARY
            return

        new_cursors = []
        for (d, p) in self.cursors:
            if self.corpus.continuation(d, p) == tok:
                if (d, p + 1) not in self._used_positions:
                    new_cursors.append((d, p + 1))
        self.cursors = new_cursors
        self.span_len += 1
        self._cur_tokens.append(tok)
        self._maybe_auto_close_on_divergence()
        if self.mode == IN_SPAN and self.span_len > 0 and not self.cursors:
            self._close_span()
            self.mode = BOUNDARY

    def _close_span(self) -> None:
        if self.span_len == 0:
            return
        sources = []
        for (d, end_pos) in self.cursors:
            start = end_pos - (self.span_len - 1)
            sources.append((d, start, end_pos + 1))
            for pos in range(start, end_pos + 1):
                self._used_positions.add((d, pos))
            self._quoted_docs.add(d)
        self.segments.append(
            Segment(
                list(self._cur_tokens),
                sources,
                "extract",
                guidance=self._current_span_guidance,
            )
        )
        self._has_closed_extract = True
        self.allow_shared_prefix = False
        self._pending_cite_docs = {d for d, _, _ in sources}
        if self.require_cite_after_extract and self._pending_cite_docs:
            self._cite_gate_active = True
        self.span_len = 0
        self._cur_tokens = []
        self._span_start_docs = set()
        self._span_anchor = None
        self._current_span_guidance = ""
        self.mode = BOUNDARY

    def __call__(self, input_ids, scores):
        ids = list(input_ids)
        for i in range(self._seen, len(ids)):
            self._consume(ids[i])
        self._seen = len(ids)

        allowed = self._allowed()
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.full(scores.shape, NEG_INF, dtype=np.float32)
        idx = np.fromiter((t for t in allowed if t < scores.shape[0]), dtype=np.int64)
        mask[idx] = scores[idx]
        return mask

    def finalize(self):
        if self.mode == IN_SPAN:
            self._close_span()
        return self.segments

    def _doc_label(self, doc_idx: int) -> str:
        if doc_idx < len(self.corpus.doc_meta) and self.corpus.doc_meta[doc_idx]:
            meta = self.corpus.doc_meta[doc_idx]
            outlet = meta.get("outlet")
            if outlet:
                return str(outlet)
        if doc_idx < len(self.corpus.doc_names):
            return self.corpus.doc_names[doc_idx]
        return str(doc_idx)

    def render(self, detok: Callable[[list[int]], str]) -> str:
        out: list[str] = []
        last_extract_idx: int | None = None

        for seg in self.segments:
            if seg.kind == "cite":
                doc = seg.sources[0][0] if seg.sources else -1
                marker = f"[{doc + 1}]"
                if last_extract_idx is not None:
                    line = out[last_extract_idx]
                    out[last_extract_idx] = line.replace("  <-", f"{marker}  <-", 1)
                continue
            if seg.kind == "connective":
                if seg.token_ids == [self.delim_id]:
                    out.append("  |  ")
                else:
                    out.append(detok(seg.token_ids))
                continue
            text = detok(seg.token_ids)
            srcs = ", ".join(
                f"{self._doc_label(d)}[{s}:{e}]" for (d, s, e) in seg.sources
            ) or "UNATTRIBUTED"
            tag = "AMBIGUOUS" if len(seg.sources) > 1 else "EXACT"
            guidance_tag = f" [{seg.guidance}]" if seg.guidance else ""
            out.append(f'"{text}"  <- {tag}: {srcs}{guidance_tag}')
            last_extract_idx = len(out) - 1

        return "\n".join(out)

    def groups_satisfied(self) -> bool:
        return self._groups_satisfied()

    def quoted_docs(self) -> set[int]:
        return set(self._quoted_docs)

    def eligible_bias_tokens(self) -> dict[int, float]:
        """Tokens to nudge when currently allowed (for BiasedGINLogitsProcessor)."""
        bias: dict[int, float] = {}
        allowed = self._allowed()
        for tok in self._cite_start_tokens_for_pending():
            if tok in allowed:
                bias[tok] = 2.0
        if self._connective_eligible():
            for tok in self.connective_starts:
                if tok in allowed:
                    bias[tok] = 1.5
        if not self._has_closed_extract and self.preferred_starts:
            ranked_map = {
                (d, p): s for d, p, s in self.ranked_sentence_starts
            }
            for doc, pos in self.preferred_starts:
                if doc < len(self.corpus.docs) and pos < len(self.corpus.docs[doc]):
                    tok = self.corpus.docs[doc][pos]
                    if tok in allowed:
                        score = ranked_map.get((doc, pos), 0.0)
                        bias[tok] = max(bias.get(tok, 0.0), 3.0 + score)
        return bias

    def group_satisfaction_status(self) -> list[tuple[frozenset[int], bool]]:
        return [
            (group, group <= self._quoted_docs)
            for group in self.required_doc_groups
        ]
