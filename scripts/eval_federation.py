"""Measure the Federation v1 bar against two live node processes.

Prereqs (three terminals):
    bash scripts/federation_db_setup.sh                      # once
    python scripts/node_serve.py --config config/node_b.yaml # terminal 1
    python scripts/node_serve.py --config config/node_a.yaml # terminal 2
    python scripts/eval_federation.py                        # terminal 3

Bar (spec): routing_false_positives 0; routing_recall 1.0;
routed_fabrication_rate 0.0; honest_refusal_rate 1.0.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ssl

import httpx

from gin.federation.eval import (
    QueryOutcome,
    compute_metrics,
    load_federation_queryset,
    verify_claims_in_db,
)
from gin.federation.schema import FederatedQuery, FederatedResponse

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def _driver_ssl_context(trust_cert: str, driver_cert: str, driver_key: str) -> ssl.SSLContext:
    """Node A's server only accepts client certs it has pinned (node_b) — the
    driver authenticates as node_b, trusting node_a's own cert as the server
    identity. Same pinned-cert pattern as HttpPeerClient."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=trust_cert)
    ctx.load_cert_chain(certfile=driver_cert, keyfile=driver_key)
    return ctx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-url", default="https://127.0.0.1:8471")
    parser.add_argument(
        "--node-b-db",
        default="postgresql://gin:gin@localhost:5432/gin_node_b",
        help="verification DB for routed answers (the answering node's)",
    )
    parser.add_argument(
        "--queryset", default=str(ROOT / "data" / "eval" / "queryset_federation.yaml")
    )
    parser.add_argument("--trust-cert", default=str(ROOT / "certs" / "node_a" / "cert.pem"),
                        help="node_a's own cert, to validate it as the server")
    parser.add_argument("--driver-cert", default=str(ROOT / "certs" / "node_b" / "cert.pem"),
                        help="a cert node_a has pinned as a peer, presented as the driver's own identity")
    parser.add_argument("--driver-key", default=str(ROOT / "certs" / "node_b" / "key.pem"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    queries = load_federation_queryset(args.queryset)
    ssl_context = _driver_ssl_context(args.trust_cert, args.driver_cert, args.driver_key)
    outcomes: list[QueryOutcome] = []

    with httpx.Client(timeout=args.timeout, verify=ssl_context) as client:
        for q in queries:
            fq = FederatedQuery(query=q.query, origin_node="eval_driver", hop_count=0)
            r = client.post(
                f"{args.node_a_url}/v1/federated/query",
                json=fq.model_dump(),
            )
            r.raise_for_status()
            resp = FederatedResponse.model_validate(r.json())

            if resp.refusal is not None:
                # Delegation happened iff the peer contributed a reason.
                routed = bool(resp.refusal.peer_reasons)
                outcome = QueryOutcome(
                    id=q.id,
                    federation_class=q.federation_class,
                    refused=True,
                    routed=routed,
                    refusal_reasons={
                        resp.refusal.node_id: resp.refusal.reason,
                        **resp.refusal.peer_reasons,
                    },
                )
            else:
                routed = resp.federation is not None
                verified = (
                    verify_claims_in_db(resp.answer.claims, args.node_b_db)
                    if routed
                    else None
                )
                outcome = QueryOutcome(
                    id=q.id,
                    federation_class=q.federation_class,
                    refused=False,
                    routed=routed,
                    source_node=resp.answer.node_id,
                    attribution_verified=verified,
                )
            print(
                f"[{q.id}] class={q.federation_class} refused={outcome.refused} "
                f"routed={outcome.routed} source={outcome.source_node!r} "
                f"verified={outcome.attribution_verified}"
            )
            outcomes.append(outcome)

    metrics = compute_metrics(outcomes)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "federation_metrics.json"
    artifact.write_text(
        json.dumps(
            {
                "run_id": ts,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "node_a_url": args.node_a_url,
                "queryset": str(args.queryset),
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {k: v for k, v in metrics.items() if k != "per_query"}
    print(json.dumps(summary, indent=2))
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
