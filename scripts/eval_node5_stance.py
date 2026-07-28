"""Score the stance-gated CONTRADICTS branch on the 24 node5 curator labels.

    venv/Scripts/python.exe scripts/eval_node5_stance.py

WRITES NOTHING. This is the reproducible artifact behind the numbers in the
spec's Results section.

The metric is three numbers, reported together and never traded against one
another, because 12/19 is a PRECISION figure and a rule that can abstain would
otherwise look better simply by emitting fewer edges:

  P      of the within-event pairs typed CONTRADICTS, the fraction labeled
         contradicts                                   (0.632 at ebceb46)
  R      of the 12 labeled contradicts, the fraction typed CONTRADICTS
                                                        (1.000 at ebceb46)
  P_all  P over all 24 pairs, so stage-1 false positives count against
         stage 2                                        (0.500 at ebceb46)

Pre-registered bar: P and P_all both strictly improve, at R >= 0.75.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import Relation
from gin.cartographer.quantity import evidence_for
from gin.cartographer.relatedness import make_same_story
from gin.curator.node5_labels import (
    BASELINE_P,
    BASELINE_P_ALL,
    BASELINE_R,
    node5_pairs,
    node5_texts,
    score,
)
from gin.curator.text_index import default_text_index


def main() -> int:
    texts = node5_texts()
    proposer = CombinedRelationProposer()
    # default_text_index() ALONE, and this matters. It already contains node5
    # since CORPUS_NODES registered it (c039edd), so adding node5's texts on top
    # -- as verify_node5_surfacing.py and curator_serve.py did before
    # df_corpus_texts() -- doubles node5's document frequencies, lifts its
    # tokens above the rare ceiling, and MASKS cross-event false positives.
    # Measured both ways during sub-project F: the doubled corpus (312 docs,
    # ceiling 10) reported P_all 0.857 with 0 cross-event false positives,
    # while the real corpus (274 docs, ceiling 9) reported P_all 0.750 with 2.
    # Reporting the flattering number would be an artifact of a known bug, so
    # this script never inherited it.
    proposer.same_story = make_same_story(list(default_text_index().values()))

    # (pair, typed_contradicts, evidence_dict) once, reused by every report.
    rows = []
    for pair in node5_pairs():
        typed, ev = proposer.type_relation(texts[pair.src], texts[pair.dst])
        rows.append((pair, typed, ev))

    within = [(p, t, e) for p, t, e in rows if p.within_event]
    cross = [(p, t, e) for p, t, e in rows if not p.within_event]
    print(f"{len(within)} within-event pairs, {len(cross)} cross-event pairs\n")

    print("=== per pair ===")
    for pair, typed, ev in rows:
        is_contra = typed is Relation.CONTRADICTS
        mark = "ok " if is_contra == pair.gold_contradicts else "MISS"
        held = "H" if pair.held_out else ("d" if pair.within_event else "x")
        facts = ""
        if ev.get("stance"):
            e = evidence_for(texts[pair.src], texts[pair.dst])
            # e.first() walks STANCE_PRECEDENCE via StanceEvidence.bucket, so
            # the ordering is not re-hardcoded a third time here.
            bucket = e.first()
            if bucket:
                x, y = bucket[0]
                facts = f"  [{x.value:g} vs {y.value:g} {x.unit_class}]"
            facts += f" stance={ev['stance']}"
        print(f"  {mark} {held} {pair.event:<28} gold={pair.relation.value:<12} "
              f"typed={typed.value:<16} ch={ev['channel']:<8}{facts}")

    def typed_rows(subset):
        return [(p, t is Relation.CONTRADICTS) for p, t, _e in subset]

    s = score(typed_rows(within))
    s_all = score(typed_rows(rows))
    print("\n=== pre-registered metric ===")
    print(f"  {'':8s} {'baseline':>9s} {'measured':>9s}")
    print(f"  {'P':8s} {BASELINE_P:9.3f} {s.precision:9.3f}   "
          f"(tp {s.tp} fp {s.fp} fn {s.fn})")
    print(f"  {'R':8s} {BASELINE_R:9.3f} {s.recall:9.3f}")
    print(f"  {'P_all':8s} {BASELINE_P_ALL:9.3f} {s_all.precision:9.3f}   "
          f"(tp {s_all.tp} fp {s_all.fp})")
    passed = (
        s.precision > BASELINE_P
        and s_all.precision > BASELINE_P_ALL
        and s.recall >= 0.75
    )
    print(f"\n  pre-registered bar: {'PASS' if passed else 'FAIL'}"
          f"  (P and P_all both improve, R >= 0.75)")

    dev = [row for row in within if not row[0].held_out]
    held = [row for row in within if row[0].held_out]
    ds, hs = score(typed_rows(dev)), score(typed_rows(held))
    print("\n=== over-fitting control (the split was named before measuring) ===")
    print(f"  development ({len(dev)} pairs, 7 events)   "
          f"P {ds.precision:.3f}  R {ds.recall:.3f}")
    print(f"  held out    ({len(held)} pairs, 3 events)   "
          f"P {hs.precision:.3f}  R {hs.recall:.3f}")
    print(f"  gap in P: {hs.precision - ds.precision:+.3f}")
    print("  CAVEAT: the planning session's exploratory sweep included these")
    print("  events, so this is a weaker independent check than the named split")
    print("  implies. The alignment floor was still selected on development only.")

    print("\n=== false positives by channel (the attribution that matters) ===")
    by_channel: Counter[str] = Counter()
    for pair, typed, ev in rows:
        if typed is Relation.CONTRADICTS and not pair.gold_contradicts:
            by_channel[ev["channel"]] += 1
            print(f"  {ev['channel']:<8} gold={pair.relation.value:<12} "
                  f"p_contra={ev.get('p_contra', 0.0):.3f} cos={ev['cos']:.3f} "
                  f"stance={ev.get('stance')}  {pair.src} <-> {pair.dst}")
    print(f"  totals: {dict(by_channel)}")
    print("  The stance channel now overrules a firing NLI on same-story pairs")
    print("  when stance disagrees decisively (not None, not conflict), so a")
    print("  firing NLI is no longer automatically right. NLI's two highest")
    print("  p_contra in this set -- a corroborates (0.983) and a supersedes")
    print("  (0.980) -- were exactly the disagreeing-stance pairs; the veto now")
    print("  fixes both, so neither shows up here. The last stance=None false")
    print("  positive (n5_doc_023 <-> 026, the path the veto cannot reach) was")
    print("  removed upstream by the variant-D anchor test (2026-07-28): the")
    print("  pair no longer counts as same-story, so nothing types it.")

    print("\n=== 4-way confusion (reported, NOT gated) ===")
    matrix = Counter((p.relation.value, t.value) for p, t, _e in rows)
    for (gold, typed), n in sorted(matrix.items()):
        print(f"  gold {gold:<14} -> typed {typed:<16} {n}")
    print("  n5_doc_036 <-> 037 (corroborates, scopes differ) is expected to")
    print("  abstain rather than corroborate: an incorrect 4-way answer that is")
    print("  nonetheless the right CONTRADICTS decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
