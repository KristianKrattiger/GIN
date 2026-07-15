"""One anchor-sync cycle: compare cached root to the peer's, and if they
differ, drill down to just the mismatched buckets. Bandwidth is the point —
matched buckets are never fetched.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .anchor_store import PeerAnchorStore
from .anchor_tree import NUM_BUCKETS, all_bucket_hashes, root_hash
from .client import PeerClient
from .config import PeerConfig
from .peer_summary_store import PeerSummaryStore
from .schema import AnchorSyncStats

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    root_matched: bool
    buckets_synced: int
    bytes_transferred: int


def _response_bytes(model) -> int:
    return len(model.model_dump_json().encode("utf-8"))


def sync_once(
    peer: PeerConfig, peer_client: PeerClient, store: PeerAnchorStore
) -> SyncStats:
    root_resp = peer_client.get_anchor_root(peer)
    bytes_transferred = _response_bytes(root_resp)
    local_rows = store.all_rows(peer.node_id)
    local_root = root_hash(all_bucket_hashes(local_rows))

    if local_root == root_resp.root_hash:
        return SyncStats(root_matched=True, buckets_synced=0, bytes_transferred=bytes_transferred)

    buckets_resp = peer_client.get_anchor_buckets(peer)
    bytes_transferred += _response_bytes(buckets_resp)
    local_bucket_hashes = all_bucket_hashes(local_rows)
    mismatched = [
        i for i in range(NUM_BUCKETS)
        if local_bucket_hashes[i] != buckets_resp.bucket_hashes[i]
    ]

    for i in mismatched:
        leaves_resp = peer_client.get_anchor_bucket(peer, i)
        bytes_transferred += _response_bytes(leaves_resp)
        store.replace_bucket(peer.node_id, i, leaves_resp.leaves)

    return SyncStats(
        root_matched=False, buckets_synced=len(mismatched), bytes_transferred=bytes_transferred
    )


async def run_forever(
    peer: PeerConfig,
    peer_client: PeerClient,
    store: PeerAnchorStore,
    interval_s: float,
    stats: AnchorSyncStats,
    summary_store: Optional[PeerSummaryStore] = None,
) -> None:
    """One sync_once() per interval, forever, until cancelled. When the peer's
    anchor root changed this cycle, its routing summary is assumed stale too and
    refetched. Any failure is logged and skipped — background maintenance must
    never affect query answering."""
    while True:
        try:
            result = await asyncio.to_thread(sync_once, peer, peer_client, store)
            stats.cycles_run += 1
            stats.last_root_matched = result.root_matched
            stats.last_cycle_buckets_synced = result.buckets_synced
            stats.last_cycle_bytes = result.bytes_transferred
            if summary_store is not None and not result.root_matched:
                summary = await asyncio.to_thread(peer_client.get_summary, peer)
                summary_store.set(peer.node_id, summary)
        except Exception:
            logger.exception("anchor sync with %s failed", peer.node_id)
        await asyncio.sleep(interval_s)
