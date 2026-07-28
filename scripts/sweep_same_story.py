"""Sweep the same-story predicate's design space. WRITES NOTHING.

    venv/Scripts/python.exe scripts/sweep_same_story.py

This exists because the stage-1 anchor fix was WITHDRAWN (2026-07-26): requiring
the anchor to be entity-grade in both texts fixes the cross-event false positives
on node5 but regresses two pre-registered gold contradicts pairs, because
anchor_tokens treats sentence-initial capitalisation as carrying no entity signal
and "Northwind Systems reported..." puts the real entity first in its sentence.
See docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md.

Rather than leave that design space in prose, this reproduces it. Every cell is
scored against BOTH corpora that constrain the predicate -- optimising node5's
cross-event false positives alone is precisely what produced the withdrawn fix.

Anchor modes:
  union      (anchor(a) | anchor(b)) & rare   -- shipped until 2026-07-28
  inter      (anchor(a) & anchor(b)) & rare   -- WITHDRAWN: costs the legal pairs
  inter_cap  as inter, but a sentence-initial capitalised word counts as
             entity-grade when the NEXT word is also capitalised, so
             "Northwind Systems" qualifies and "Combined reservoir" does not
  mixed      entity-grade on one side, capitalised-or-entity-grade on the other
             -- THE SHIPPED BEHAVIOUR (variant D, 2026-07-28), measured here
             with the production capitalized_tokens so the row is the real
             predicate

Columns:
  n5_in    node5 within-event pairs still firing     (19 = no real story lost)
  n5_fp    node5 cross-event pairs still firing      (0 = target)
  gold_c   gold contradicts pairs firing             (4 = the shipped baseline;
           the other 3 are cross-story climate framing and correctly never fire)
  gold_fp  gold NON-contradicts pairs firing         (0 everywhere measured so
           far -- no headroom to gain here, only to break)

The df corpus is default_text_index() ALONE. It already contains node5 since
c039edd; adding node5 texts on top -- as verify_node5_surfacing.py and
curator_serve.py did until df_corpus_texts() -- doubles node5's document
frequencies and MASKS cross-event false positives. That reported union at 0/5
when the truth is 4/5.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.cartographer.labeled_set import chunks as gold_chunks
from gin.cartographer.labeled_set import gold
from gin.cartographer.models import Relation
from gin.cartographer.relatedness import (
    CALENDAR_WORDS,
    _doc_freq,
    _rare_df_ceiling,
    anchor_tokens,
    capitalized_tokens,
)
from gin.corpus.relevance import _norm_tokens, _normalize_token
from gin.curator.node5_labels import node5_pairs, node5_texts
from gin.curator.text_index import default_text_index

FLOORS = (2, 3, 4)
CEILINGS = (4, 6, 7, 9, 12)

_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?]\s*$")


def _scan(text: str):
    """(normalized_token, entity_grade, capitalized, sentence_initial_name).

    Mirrors anchor_tokens' own tests so the sweep measures the real predicate,
    and adds the two extra signals the alternative modes need.
    ``sentence_initial_name`` is the inter_cap signal: a capitalised word that
    OPENS its sentence and is followed by another capitalised word, i.e. the
    first token of a multi-word proper name ("Northwind Systems"), not
    boilerplate ("Combined reservoir").
    """
    matches = list(_WORD.finditer(text))
    for i, m in enumerate(matches):
        word = m.group(0)
        before = text[: m.start()]
        sentence_initial = (
            not before.strip()
            or bool(_SENTENCE_END.search(before))
            or before.endswith("\n")
        )
        capitalized = word[0].isupper()
        entity_grade = (
            (word.isdigit() and len(word) >= 2)
            or (len(word) > 2 and word.isupper())
            or (capitalized and not word.isupper() and not sentence_initial)
        )
        nxt = matches[i + 1].group(0) if i + 1 < len(matches) else ""
        next_capitalized = bool(nxt) and nxt[0].isupper() and not nxt.isupper()
        yield (
            _normalize_token(word.lower()),
            entity_grade,
            capitalized,
            sentence_initial and capitalized and next_capitalized,
        )


def _anchors_with_initial_names(text: str) -> set[str]:
    return {
        tok for tok, eg, _cap, initial_name in _scan(text)
        if (eg or initial_name) and tok not in CALENDAR_WORDS
    }


MODES = {
    "union": lambda a, b, rare: bool((anchor_tokens(a) | anchor_tokens(b)) & rare),
    "inter": lambda a, b, rare: bool((anchor_tokens(a) & anchor_tokens(b)) & rare),
    "inter_cap": lambda a, b, rare: bool(
        (_anchors_with_initial_names(a) & _anchors_with_initial_names(b)) & rare
    ),
    # Shipped (variant D): capitalized_tokens is cap-or-entity-grade, so a
    # story figure like '11' corroborates an anchor even though digits are
    # never capitalised.
    "mixed": lambda a, b, rare: bool(
        ((anchor_tokens(a) & capitalized_tokens(b))
         | (capitalized_tokens(a) & anchor_tokens(b)))
        & rare
    ),
}


def main() -> int:
    index = default_text_index()
    df = _doc_freq(list(index.values()))
    n5_text = node5_texts()
    n5 = node5_pairs()

    gold_text = {c.chunk_id: c.text for c in gold_chunks()}
    gold_rows = [
        (src, dst, relation)
        for src, dst, relation, _register in gold()
        if src in gold_text and dst in gold_text
    ]

    print(f"df corpus: default_text_index() alone, {len(index)} docs "
          f"(natural ceiling {_rare_df_ceiling(len(index))})")
    print(f"node5: {len(n5)} labels; gold: {len(gold_rows)} resolvable pairs\n")
    print(f"{'mode':>10} {'floor':>6} {'ceil':>5} "
          f"{'n5_in':>6} {'n5_fp':>6} {'gold_c':>7} {'gold_fp':>8}")

    def fires(mode, a, b, floor, ceiling):
        rare = {
            t for t in (_norm_tokens(a) & _norm_tokens(b))
            if df.get(t, 0) <= ceiling
        }
        return len(rare) >= floor and MODES[mode](a, b, rare)

    for mode in MODES:
        for floor in FLOORS:
            for ceiling in CEILINGS:
                n5_in = sum(
                    fires(mode, n5_text[p.src], n5_text[p.dst], floor, ceiling)
                    for p in n5 if p.within_event
                )
                n5_fp = sum(
                    fires(mode, n5_text[p.src], n5_text[p.dst], floor, ceiling)
                    for p in n5 if not p.within_event
                )
                gold_c = gold_fp = 0
                for src, dst, relation in gold_rows:
                    if not fires(mode, gold_text[src], gold_text[dst], floor, ceiling):
                        continue
                    if relation is Relation.CONTRADICTS:
                        gold_c += 1
                    else:
                        gold_fp += 1
                print(f"{mode:>10} {floor:>6} {ceiling:>5} "
                      f"{n5_in:>6} {n5_fp:>6} {gold_c:>7} {gold_fp:>8}")

    print("\nThe shipped cell is  mixed / floor 2 / ceil 9  (variant D, 2026-07-28;")
    print("union / floor 2 / ceil 9 before that).")
    print("Read every candidate against BOTH gold_c and n5_fp: the withdrawn fix")
    print("reached n5_fp 0 by dropping gold_c from 4 to 2, which is why scoring")
    print("node5 alone is not enough to justify an anchor change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
