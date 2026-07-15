"""Measure the Merkle anchor sync bar against two live node processes.

Prereqs (three terminals):
    bash scripts/federation_db_setup.sh                        # once
    python scripts/node_serve.py --config config/node_b.yaml    # terminal 1
    python scripts/node_serve.py --config config/node_a.yaml    # terminal 2
    python scripts/eval_anchor_sync.py                          # terminal 3

Bar (spec): 0 diff between node A's cache of B and B's ground truth after
convergence; a no-op cycle transfers O(1) bytes; a cycle following a
single-chunk mutation transfers far less than a full-corpus resync would.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import psycopg

from gin.federation.schema import AnchorLeaf, AnchorLeavesResponse, AnchorSyncStats

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def _ground_truth(database_url: str) -> dict[str, dict]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT c.chunk_id, c.content_hash, d.outlet, d.title "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
        ).fetchall()
    return {r[0]: {"content_hash": r[1], "outlet": r[2], "title": r[3]} for r in rows}


def _peer_anchors(database_url: str, peer_node_id: str) -> dict[str, dict]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT chunk_id, content_hash, outlet, title FROM peer_anchors "
            "WHERE peer_node_id = %s", (peer_node_id,),
        ).fetchall()
    return {r[0]: {"content_hash": r[1], "outlet": r[2], "title": r[3]} for r in rows}


def _diff(local: dict, remote: dict) -> dict:
    return {
        "added": [cid for cid in remote if cid not in local],
        "changed": [cid for cid in remote if cid in local and local[cid] != remote[cid]],
        "removed": [cid for cid in local if cid not in remote],
    }


def _diff_is_empty(diff: dict) -> bool:
    return not diff["added"] and not diff["changed"] and not diff["removed"]


def _wait_for_convergence(node_a_db: str, node_b_db: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    diff = {"added": ["<not yet checked>"], "changed": [], "removed": []}
    while time.monotonic() < deadline:
        diff = _diff(_peer_anchors(node_a_db, "node_b"), _ground_truth(node_b_db))
        if _diff_is_empty(diff):
            return diff
        time.sleep(1.0)
    return diff


def _sync_stats(node_a_url: str, secret: str) -> AnchorSyncStats:
    r = httpx.get(
        f"{node_a_url}/v1/federated/anchors/sync_stats",
        headers={"Authorization": f"Bearer {secret}"}, timeout=10.0,
    )
    r.raise_for_status()
    return AnchorSyncStats.model_validate(r.json())


def _full_corpus_bytes(truth: dict[str, dict]) -> int:
    leaves = [AnchorLeaf(chunk_id=cid, **fields) for cid, fields in truth.items()]
    resp = AnchorLeavesResponse(node_id="node_b", bucket_index=-1, leaves=leaves)
    return len(resp.model_dump_json().encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-url", default="http://127.0.0.1:8471")
    parser.add_argument("--node-a-db", default="postgresql://gin:gin@localhost:5432/gin_node_a")
    parser.add_argument("--node-b-db", default="postgresql://gin:gin@localhost:5432/gin_node_b")
    parser.add_argument("--secret", default="dev-federation-secret")
    parser.add_argument("--converge-timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("[1/4] waiting for initial convergence (node A's cache of B vs. B's ground truth)")
    initial_diff = _wait_for_convergence(args.node_a_db, args.node_b_db, args.converge_timeout)
    correctness_pass = _diff_is_empty(initial_diff)
    print(f"    diff: {initial_diff}")

    print("[2/4] recording a no-op cycle's byte count")
    no_op_bytes = None
    deadline = time.monotonic() + args.converge_timeout
    while time.monotonic() < deadline:
        stats = _sync_stats(args.node_a_url, args.secret)
        if stats.last_root_matched:
            no_op_bytes = stats.last_cycle_bytes
            break
        time.sleep(1.0)
    print(f"    no-op cycle bytes: {no_op_bytes}")

    print("[3/4] mutating one chunk in node B directly, then waiting for the next sync")
    with psycopg.connect(args.node_b_db) as conn:
        mutated_chunk_id, original_hash = conn.execute(
            "SELECT chunk_id, content_hash FROM chunks LIMIT 1"
        ).fetchone()
        conn.execute(
            "UPDATE chunks SET content_hash = %s WHERE chunk_id = %s",
            (original_hash + "-eval-mutated", mutated_chunk_id),
        )
        conn.commit()

    try:
        mutation_bytes = None
        cycles_before = _sync_stats(args.node_a_url, args.secret).cycles_run
        deadline = time.monotonic() + args.converge_timeout
        while time.monotonic() < deadline:
            stats = _sync_stats(args.node_a_url, args.secret)
            if stats.cycles_run > cycles_before and not stats.last_root_matched:
                mutation_bytes = stats.last_cycle_bytes
                break
            time.sleep(1.0)
        print(f"    mutation cycle bytes: {mutation_bytes}")

        post_mutation_diff = _wait_for_convergence(
            args.node_a_db, args.node_b_db, args.converge_timeout
        )
        correctness_pass_after_mutation = _diff_is_empty(post_mutation_diff)
        full_corpus_bytes = _full_corpus_bytes(_ground_truth(args.node_b_db))
    finally:
        print("[4/4] reverting the mutation")
        with psycopg.connect(args.node_b_db) as conn:
            conn.execute(
                "UPDATE chunks SET content_hash = %s WHERE chunk_id = %s",
                (original_hash, mutated_chunk_id),
            )
            conn.commit()

    bandwidth_pass = (
        no_op_bytes is not None and no_op_bytes < 1000
        and mutation_bytes is not None and mutation_bytes < full_corpus_bytes / 4
    )

    metrics = {
        "initial_convergence_diff": initial_diff,
        "correctness_pass": correctness_pass,
        "no_op_cycle_bytes": no_op_bytes,
        "mutation_cycle_bytes": mutation_bytes,
        "full_corpus_bytes_reference": full_corpus_bytes,
        "post_mutation_diff": post_mutation_diff,
        "correctness_pass_after_mutation": correctness_pass_after_mutation,
        "bandwidth_pass": bandwidth_pass,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "anchor_sync_metrics.json"
    artifact.write_text(
        json.dumps(
            {"run_id": ts, "generated_at": datetime.now(timezone.utc).isoformat(), **metrics},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
