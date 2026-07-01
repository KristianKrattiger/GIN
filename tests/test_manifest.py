from __future__ import annotations

from pathlib import Path

from gin.corpus.manifest import (
    ManifestDocumentEntry,
    build_manifest,
    latest_manifest,
    load_manifest,
    manifest_path_for_version,
    resolve_entry,
    write_manifest_atomic,
)


def test_manifest_roundtrip_and_latest(tmp_path: Path):
    root = tmp_path / "manifests"
    m1 = build_manifest(
        version=1,
        documents=[
            ManifestDocumentEntry(
                doc_id="doc-a",
                version=1,
                storage_path="/tmp/a",
                content_hash="hash-a",
                metadata={"title": "A"},
            )
        ],
    )
    m2 = build_manifest(
        version=2,
        documents=[
            ManifestDocumentEntry(
                doc_id="doc-b",
                version=2,
                storage_path="/tmp/b",
                content_hash="hash-b",
                metadata={"title": "B"},
            )
        ],
    )
    write_manifest_atomic(m1, manifest_path_for_version(1, base_dir=root))
    write_manifest_atomic(m2, manifest_path_for_version(2, base_dir=root))

    latest = latest_manifest(root)
    assert latest is not None
    assert latest.name == "manifest_v2.json"

    loaded = load_manifest(2, base_dir=root)
    assert loaded.info.version == 2
    assert loaded.documents[0].doc_id == "doc-b"


def test_old_manifests_remain_resolvable(tmp_path: Path):
    root = tmp_path / "manifests"
    m1 = build_manifest(
        version=1,
        documents=[
            ManifestDocumentEntry(
                doc_id="doc-a",
                version=1,
                storage_path="/tmp/a",
                content_hash="hash-a",
                metadata={},
            )
        ],
    )
    m2 = build_manifest(
        version=2,
        documents=[
            ManifestDocumentEntry(
                doc_id="doc-a",
                version=2,
                storage_path="/tmp/a2",
                content_hash="hash-a2",
                metadata={},
            )
        ],
    )
    write_manifest_atomic(m1, manifest_path_for_version(1, base_dir=root))
    write_manifest_atomic(m2, manifest_path_for_version(2, base_dir=root))

    old = resolve_entry("doc-a", 1, base_dir=root)
    new = resolve_entry("doc-a", 2, base_dir=root)
    assert old.storage_path == "/tmp/a"
    assert new.storage_path == "/tmp/a2"
