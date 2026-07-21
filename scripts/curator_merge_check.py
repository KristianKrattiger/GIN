"""Merge two curators' label logs and measure agreement on a shared overlap set.

Concatenation is a safe merge because Store.fold_current() folds latest-wins
by (ts, idx) over the whole read log, regardless of which file a line came
from (gin/curator/store.py:41-49) — no reconciliation logic needed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gin.curator.models import LabelRecord


def merge_logs(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            out.write(path.read_text(encoding="utf-8"))


@dataclass
class AgreementResult:
    agree: int
    disagree: int
    disagreements: list[tuple[tuple[str, str], LabelRecord, LabelRecord]] = field(default_factory=list)


def check_agreement(
    fold_a: dict[tuple[str, str], LabelRecord],
    fold_b: dict[tuple[str, str], LabelRecord],
    keys: set[tuple[str, str]],
) -> AgreementResult:
    agree = 0
    disagree = 0
    disagreements: list[tuple[tuple[str, str], LabelRecord, LabelRecord]] = []
    for key in keys:
        rec_a = fold_a.get(key)
        rec_b = fold_b.get(key)
        if rec_a is None or rec_b is None:
            continue
        if (rec_a.relation, rec_a.relation_class) == (rec_b.relation, rec_b.relation_class):
            agree += 1
        else:
            disagree += 1
            disagreements.append((key, rec_a, rec_b))
    return AgreementResult(agree=agree, disagree=disagree, disagreements=disagreements)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge curator label logs and check overlap agreement")
    ap.add_argument("logs", type=Path, nargs="+", help="labels.jsonl files to merge")
    ap.add_argument("--out", type=Path, required=True, help="merged output path")
    args = ap.parse_args()

    merge_logs(args.logs, args.out)
    print(f"merged {len(args.logs)} logs into {args.out}")


if __name__ == "__main__":
    main()
