"""Content-addressed retrieval manifest for synthesis provenance."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import SynthesisBundle


@dataclass(frozen=True)
class RetrievalEntry:
    chunk_id: str
    outlet: str
    title: str
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float


@dataclass(frozen=True)
class RetrievalManifest:
    query: str
    query_hash: str
    synthesis_mode: str
    edge_types: list[str]
    entries: list[RetrievalEntry]
    manifest_hash: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_hash": self.query_hash,
            "synthesis_mode": self.synthesis_mode,
            "edge_types": self.edge_types,
            "entries": [asdict(e) for e in self.entries],
            "manifest_hash": self.manifest_hash,
        }


def retrieval_manifests_dir(base_dir: Path | None = None) -> Path:
    base = base_dir or (Path(__file__).resolve().parents[2] / "data" / "retrieval_manifests")
    return base


def _canonical_payload(manifest_dict: dict) -> str:
    payload = {k: v for k, v in manifest_dict.items() if k != "manifest_hash"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_retrieval_manifest(query: str, bundle: SynthesisBundle) -> RetrievalManifest:
    entries = [
        RetrievalEntry(
            chunk_id=hit.chunk_id,
            outlet=hit.outlet,
            title=hit.title,
            dense_rank=hit.dense_rank,
            sparse_rank=hit.sparse_rank,
            rrf_score=hit.rrf_score,
        )
        for hit in bundle.hits
    ]
    edge_types = sorted({e.edge_type for e in bundle.edges})
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

    partial = {
        "query": query,
        "query_hash": query_hash,
        "synthesis_mode": bundle.mode,
        "edge_types": edge_types,
        "entries": [asdict(e) for e in entries],
    }
    manifest_hash = hashlib.sha256(
        _canonical_payload({**partial, "manifest_hash": ""}).encode("utf-8")
    ).hexdigest()

    return RetrievalManifest(
        query=query,
        query_hash=query_hash,
        synthesis_mode=bundle.mode,
        edge_types=edge_types,
        entries=entries,
        manifest_hash=manifest_hash,
    )


def write_retrieval_manifest(
    manifest: RetrievalManifest,
    *,
    base_dir: Path | None = None,
) -> Path:
    root = retrieval_manifests_dir(base_dir)
    target_path = root / manifest.manifest_hash[:2] / f"{manifest.manifest_hash}.json"
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)
    return target_path
