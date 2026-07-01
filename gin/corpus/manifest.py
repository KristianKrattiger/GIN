"""Versioned manifest utilities for corpus ingestion snapshots."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestVersionInfo:
    version: int
    created_at: str
    source: str = "corpus_manager"


@dataclass(frozen=True)
class ManifestDocumentEntry:
    doc_id: str
    version: int
    storage_path: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    info: ManifestVersionInfo
    documents: list[ManifestDocumentEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": asdict(self.info),
            "documents": [asdict(doc) for doc in self.documents],
        }


def manifests_dir(base_dir: Path | None = None) -> Path:
    base = base_dir or (Path(__file__).resolve().parents[2] / "data" / "manifests")
    return base


def manifest_path_for_version(version: int, *, base_dir: Path | None = None) -> Path:
    return manifests_dir(base_dir) / f"manifest_v{version}.json"


def _parse_version_from_name(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("manifest_v"):
        return None
    raw = stem.removeprefix("manifest_v")
    return int(raw) if raw.isdigit() else None


def latest_manifest(base_dir: Path | None = None) -> Path | None:
    root = manifests_dir(base_dir)
    if not root.exists():
        return None
    best: tuple[int, Path] | None = None
    for path in root.glob("manifest_v*.json"):
        v = _parse_version_from_name(path)
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, path)
    return best[1] if best else None


def next_manifest_version(base_dir: Path | None = None) -> int:
    current = latest_manifest(base_dir)
    if current is None:
        return 1
    version = _parse_version_from_name(current)
    if version is None:
        raise ValueError(f"Cannot parse manifest version from {current}")
    return version + 1


def _manifest_from_dict(data: dict[str, Any]) -> Manifest:
    info_raw = data.get("info", {})
    docs_raw = data.get("documents", [])
    info = ManifestVersionInfo(
        version=int(info_raw["version"]),
        created_at=str(info_raw["created_at"]),
        source=str(info_raw.get("source", "corpus_manager")),
    )
    docs = [
        ManifestDocumentEntry(
            doc_id=str(item["doc_id"]),
            version=int(item["version"]),
            storage_path=str(item["storage_path"]),
            content_hash=str(item["content_hash"]),
            metadata=dict(item.get("metadata", {})),
        )
        for item in docs_raw
    ]
    return Manifest(info=info, documents=docs)


def load_manifest(
    path_or_version: str | int | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> Manifest:
    if isinstance(path_or_version, int):
        path = manifest_path_for_version(path_or_version, base_dir=base_dir)
    elif path_or_version in (None, "latest"):
        latest = latest_manifest(base_dir=base_dir)
        if latest is None:
            raise FileNotFoundError("No manifests found")
        path = latest
    else:
        path = Path(path_or_version)
        if not path.is_absolute():
            path = manifests_dir(base_dir) / path
    data = json.loads(path.read_text(encoding="utf-8"))
    return _manifest_from_dict(data)


def resolve_entry(
    doc_id: str,
    version: int,
    *,
    base_dir: Path | None = None,
) -> ManifestDocumentEntry:
    manifest = load_manifest(version, base_dir=base_dir)
    for entry in manifest.documents:
        if entry.doc_id == doc_id:
            return entry
    raise KeyError(f"doc_id {doc_id!r} not found in manifest version {version}")


def write_manifest_atomic(
    manifest: Manifest,
    target_path: Path,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    tmp_path.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp_path, target_path)
    return target_path


def build_manifest(
    *,
    version: int,
    documents: list[ManifestDocumentEntry],
    source: str = "corpus_manager",
) -> Manifest:
    return Manifest(
        info=ManifestVersionInfo(
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            source=source,
        ),
        documents=documents,
    )
