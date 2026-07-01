"""Corpus ingestion manager with immutable storage and versioned manifests."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .manifest import (
    Manifest,
    ManifestDocumentEntry,
    build_manifest,
    manifest_path_for_version,
    manifests_dir,
    next_manifest_version,
    write_manifest_atomic,
)


@dataclass
class SourceDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    input_path: str = ""


@dataclass
class IngestResult:
    manifest: Manifest
    manifest_path: Path
    docs_seen: int
    docs_written: int
    deduped: int


class StoreBackend(Protocol):
    def staged_write_blob(self, staging_dir: Path, content: bytes) -> tuple[str, Path, bool]:
        """Write blob in staging and return (hash, staged_path, created)."""

    def commit_staged_blob(self, staged_path: Path, content_hash: str) -> tuple[Path, bool]:
        """Move staged blob into immutable store location and report if created."""

    def read_blob(self, content_hash: str) -> bytes:
        """Read immutable blob content by hash."""


class FileStoreBackend:
    def __init__(self, store_root: Path):
        self.store_root = store_root

    def _target_path(self, content_hash: str) -> Path:
        return self.store_root / content_hash[:2] / content_hash

    def staged_write_blob(self, staging_dir: Path, content: bytes) -> tuple[str, Path, bool]:
        content_hash = hashlib.sha256(content).hexdigest()
        staged_path = staging_dir / content_hash
        created = False
        if not staged_path.exists():
            staged_path.write_bytes(content)
            created = True
        return content_hash, staged_path, created

    def commit_staged_blob(self, staged_path: Path, content_hash: str) -> tuple[Path, bool]:
        target = self._target_path(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target, False
        os.replace(staged_path, target)
        return target, True

    def read_blob(self, content_hash: str) -> bytes:
        return self._target_path(content_hash).read_bytes()


class CorpusManager:
    def __init__(
        self,
        *,
        store_root: Path | None = None,
        manifests_root: Path | None = None,
        backend: StoreBackend | None = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.store_root = store_root or (project_root / "data" / "corpus_store")
        self.manifests_root = manifests_root or manifests_dir()
        self.backend = backend or FileStoreBackend(self.store_root)

    def ingest_directory(
        self,
        input_dir: Path,
        *,
        file_format: str = "auto",
        metadata_defaults: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> IngestResult:
        docs = self._load_documents(input_dir, file_format=file_format, metadata_defaults=metadata_defaults)
        version = next_manifest_version(self.manifests_root)
        entries: list[ManifestDocumentEntry] = []
        docs_written = 0
        deduped = 0
        staged: list[tuple[SourceDocument, str, Path, bool]] = []

        staging_dir = self.manifests_root / ".staging" / f"v{version}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            for doc in docs:
                content = doc.text.encode("utf-8")
                content_hash, staged_path, created = self.backend.staged_write_blob(staging_dir, content)
                if not created:
                    deduped += 1
                staged.append((doc, content_hash, staged_path, created))

            moved_paths: list[Path] = []
            for doc, content_hash, staged_path, _created in staged:
                doc_id = self._doc_id_for(doc, content_hash)
                if not dry_run:
                    final_path, created = self.backend.commit_staged_blob(staged_path, content_hash)
                    if created:
                        moved_paths.append(final_path)
                else:
                    final_path = self.store_root / content_hash[:2] / content_hash
                docs_written += 1
                entry = ManifestDocumentEntry(
                    doc_id=doc_id,
                    version=version,
                    storage_path=str(final_path.resolve()),
                    content_hash=content_hash,
                    metadata=self._normalized_metadata(doc.metadata, doc.input_path),
                )
                entries.append(entry)

            manifest = build_manifest(version=version, documents=entries)
            manifest_path = manifest_path_for_version(version, base_dir=self.manifests_root)
            if not dry_run:
                write_manifest_atomic(manifest, manifest_path)
            return IngestResult(
                manifest=manifest,
                manifest_path=manifest_path,
                docs_seen=len(docs),
                docs_written=docs_written,
                deduped=deduped,
            )
        except Exception:
            for path in moved_paths if "moved_paths" in locals() else []:
                if path.exists():
                    path.unlink()
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def load_texts_from_manifest(self, *, version: int | None = None) -> list[tuple[str, str]]:
        from .manifest import load_manifest

        manifest = load_manifest(version if version is not None else "latest", base_dir=self.manifests_root)
        docs: list[tuple[str, str]] = []
        for entry in manifest.documents:
            blob = self.backend.read_blob(entry.content_hash)
            docs.append((entry.doc_id, blob.decode("utf-8")))
        return docs

    def _normalized_metadata(self, metadata: dict[str, Any], input_path: str) -> dict[str, Any]:
        out = dict(metadata)
        out.setdefault("input_path", input_path)
        out.setdefault("title", "")
        out.setdefault("authors", [])
        out.setdefault("source", "")
        out.setdefault("published_at", "")
        return out

    def _doc_id_for(self, doc: SourceDocument, content_hash: str) -> str:
        md = self._normalized_metadata(doc.metadata, doc.input_path)
        title = str(md.get("title", "")).strip()
        authors = md.get("authors", [])
        if not isinstance(authors, list):
            authors = [str(authors)]
        source = str(md.get("source", "")).strip() or str(md.get("source_uri", "")).strip()
        seed = f"{title}|{','.join(str(a).strip() for a in authors)}|{source}"
        if title or authors or source:
            return hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return content_hash

    def _load_documents(
        self,
        input_dir: Path,
        *,
        file_format: str,
        metadata_defaults: dict[str, Any] | None,
    ) -> list[SourceDocument]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        defaults = metadata_defaults or {}
        docs: list[SourceDocument] = []
        paths = sorted(p for p in input_dir.iterdir() if p.is_file())
        for path in paths:
            suffix = path.suffix.lower()
            use_jsonl = file_format == "jsonl" or (file_format == "auto" and suffix == ".jsonl")
            use_txt = file_format == "txt" or (file_format == "auto" and suffix in {".txt", ".md"})
            if use_jsonl:
                docs.extend(self._load_jsonl(path, defaults))
            elif use_txt:
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                docs.append(
                    SourceDocument(
                        text=text,
                        metadata={"title": path.stem, **defaults},
                        input_path=str(path),
                    )
                )
        return docs

    def _load_jsonl(self, path: Path, defaults: dict[str, Any]) -> list[SourceDocument]:
        docs: list[SourceDocument] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("body") or item.get("text") or item.get("content") or "").strip()
            if not text:
                continue
            metadata = dict(defaults)
            metadata.update({
                "title": item.get("title", ""),
                "authors": item.get("authors", []),
                "source": item.get("source", item.get("source_uri", "")),
                "source_uri": item.get("source_uri", ""),
                "published_at": item.get("published_at", ""),
            })
            docs.append(SourceDocument(text=text, metadata=metadata, input_path=str(path)))
        return docs
