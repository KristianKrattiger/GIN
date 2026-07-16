"""Measure the peer-selection bar against three live node processes.

Prereqs (four terminals):
    bash scripts/federation_db_setup.sh                        # once
    python scripts/node_serve.py --config config/node_b.yaml    # terminal 1
    python scripts/node_serve.py --config config/node_c.yaml    # terminal 2
    python scripts/node_serve.py --config config/node_a.yaml    # terminal 3
    python scripts/eval_peer_selection.py                       # terminal 4

Node A must have completed at least one anchor-sync cycle with each peer so its
summary cache is populated (the driver polls A's sync_stats and sleeps first).

Bar (spec): selection_precision_at_1 1.0; avg_peers_tried ~1.0;
routing_false_positives 0; routed_fabrication_rate 0.0; honest_refusal_rate 1.0.
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

from gin.federation.eval import verify_claims_in_db
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.selection_eval import (
    SelectionOutcome,
    compute_selection_metrics,
    load_selection_queryset,
)

DEFAULT_OUT = ROOT / "data" / "eval_runs"
DB_FOR_NODE = {
    "node_b": "postgresql://gin:gin@localhost:5432/gin_node_b",
    "node_c": "postgresql://gin:gin@localhost:5432/gin_node_c",
}


def _await_summaries(node_a_url: str, headers: dict, timeout_s: float) -> None:
    """Wait until A has run enough sync cycles to have cached peer summaries."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = httpx.get(f"{node_a_url}/v1/federated/anchors/sync_stats", headers=headers, timeout=10.0)
        if r.status_code == 200 and r.json().get("cycles_run", 0) >= 1:
            time.sleep(2.0)  # let the summary fetch that follows the mismatch land
            return
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-url", default="http://127.0.0.1:8471")
    parser.add_argument("--queryset", default=str(ROOT / "data" / "eval" / "queryset_peer_selection.yaml"))
    parser.add_argument("--secret", default="dev-federation-secret")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--summary-wait", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--gated-peer", default=None,
        help="If set, report how many outcomes still reached this peer_node_id "
             "(expected 0 when that peer is trust-gated in the target node's config)",
    )
    args = parser.parse_args()

    queries = load_selection_queryset(args.queryset)
    headers = {"Authorization": f"Bearer {args.secret}"}
    _await_summaries(args.node_a_url, headers, args.summary_wait)

    outcomes: list[SelectionOutcome] = []
    with httpx.Client(timeout=args.timeout) as client:
        for q in queries:
            fq = FederatedQuery(query=q.query, origin_node="eval_driver", hop_count=0)
            r = client.post(f"{args.node_a_url}/v1/federated/query", headers=headers, json=fq.model_dump())
            r.raise_for_status()
            resp = FederatedResponse.model_validate(r.json())

            if resp.refusal is not None:
                attempted = list(resp.refusal.peer_reasons.keys())
                outcomes.append(SelectionOutcome(
                    id=q.id, federation_class=q.federation_class, refused=True,
                    routed=bool(resp.refusal.peer_reasons), peers_attempted=attempted,
                    refusal_reasons={resp.refusal.node_id: resp.refusal.reason, **resp.refusal.peer_reasons},
                ))
            else:
                routed = resp.federation is not None
                attempted = list(resp.federation.peers_attempted) if routed else []
                verified = (
                    verify_claims_in_db(resp.answer.claims, DB_FOR_NODE[resp.answer.node_id])
                    if routed and resp.answer.node_id in DB_FOR_NODE else None
                )
                outcomes.append(SelectionOutcome(
                    id=q.id, federation_class=q.federation_class, refused=False,
                    routed=routed, source_node=resp.answer.node_id,
                    peers_attempted=attempted, attribution_verified=verified,
                ))
            o = outcomes[-1]
            print(f"[{q.id}] class={q.federation_class} routed={o.routed} "
                  f"source={o.source_node!r} attempted={o.peers_attempted} verified={o.attribution_verified}")

    metrics = compute_selection_metrics(outcomes, gated_peer=args.gated_peer)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "peer_selection_metrics.json"
    artifact.write_text(json.dumps(
        {"run_id": ts, "generated_at": datetime.now(timezone.utc).isoformat(),
         "node_a_url": args.node_a_url, "queryset": str(args.queryset), **metrics},
        indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_query"}, indent=2))
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
