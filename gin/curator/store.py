"""Append-only JSONL label store. Source of truth for the framing corpus.

The current gold is DERIVED by folding the log latest-wins per pair; a relabel
or adjudication is a new record superseding an earlier one, never an in-place
edit — so labeling history (including contested-then-adjudicated pairs) survives
and the file stays git-diffable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from gin.cartographer.models import Relation

from .models import LabelRecord, pair_key


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, rec: LabelRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")

    def read_log(self) -> list[LabelRecord]:
        if not self.path.is_file():
            return []
        records: list[LabelRecord] = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(LabelRecord.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"{self.path}: malformed record on line {lineno}: {exc}") from exc
        return records

    def fold_current(self) -> dict[tuple[str, str], LabelRecord]:
        current: dict[tuple[str, str], tuple[str, int, LabelRecord]] = {}
        for idx, rec in enumerate(self.read_log()):
            key = pair_key(rec.src_chunk_id, rec.dst_chunk_id)
            stamp = (rec.ts, idx)
            prev = current.get(key)
            if prev is None or stamp >= (prev[0], prev[1]):
                current[key] = (rec.ts, idx, rec)
        return {key: value[2] for key, value in current.items()}

    def gold(self) -> list[tuple[str, str, Relation, Optional[str]]]:
        return [
            (r.src_chunk_id, r.dst_chunk_id, r.relation, r.relation_class)
            for r in self.fold_current().values()
        ]
