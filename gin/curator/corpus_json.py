"""Load corpus_node*.json exports into LabeledChunks, DB-free.

Normalizes chunk ids to the {doc_id}:{position} convention the gold, the
escalation bar, and the curator store all use — the JSON stores them as
n1_doc_005_c002, which would never match those keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from gin.cartographer.models import LabeledChunk


def load_corpus_chunks(paths: Iterable[Union[Path, str]]) -> list[LabeledChunk]:
    chunks: dict[str, LabeledChunk] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"corpus file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level JSON must be an object, got {type(data).__name__}")
        for doc in data.get("documents", []):
            if "doc_id" not in doc:
                raise ValueError(f"{path}: document missing 'doc_id'")
            doc_id = doc["doc_id"]
            for ch in doc.get("chunks", []):
                if "position" not in ch:
                    raise ValueError(f"{path}: chunk in {doc_id} missing 'position'")
                if "text" not in ch:
                    raise ValueError(f"{path}: chunk {doc_id}:{ch['position']} missing 'text'")
                cid = f"{doc_id}:{ch['position']}"
                if cid not in chunks:
                    chunks[cid] = LabeledChunk(cid, ch["text"])
    return list(chunks.values())
