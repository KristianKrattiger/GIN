"""Canonical graph state maintained by the Bookkeeper.

Holds admitted edges, dedups by identity, and answers the one structural question
admission needs: would adding this ordering edge create a cycle? Symmetric
relations (contradicts/corroborates) carry no ordering, so only their identity is
tracked; ordering relations (supersedes) also maintain an adjacency for cycle
detection.
"""
from __future__ import annotations

from gin.cartographer.models import Relation

from .models import ORDERING_RELATIONS, AdmittedEdge


def edge_key(src: str, dst: str, relation: Relation) -> tuple:
    """Identity of an edge. Symmetric relations are direction-independent."""
    if relation in (Relation.CONTRADICTS, Relation.CORROBORATES):
        return (relation, frozenset({src, dst}))
    return (relation, src, dst)


class GraphState:
    def __init__(self) -> None:
        self._edges: dict[tuple, AdmittedEdge] = {}
        # relation -> src -> {dst}, for ordering relations only.
        self._adj: dict[Relation, dict[str, set[str]]] = {}

    def __len__(self) -> int:
        return len(self._edges)

    def edges(self) -> list[AdmittedEdge]:
        return list(self._edges.values())

    def contains(self, src: str, dst: str, relation: Relation) -> bool:
        return edge_key(src, dst, relation) in self._edges

    def _reachable(self, relation: Relation, start: str, target: str) -> bool:
        """DFS over admitted ordering edges: is target reachable from start?"""
        adj = self._adj.get(relation, {})
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj.get(node, ()))
        return False

    def would_create_cycle(self, src: str, dst: str, relation: Relation) -> bool:
        """True if adding src->dst (ordering) closes a cycle: dst already reaches src."""
        if relation not in ORDERING_RELATIONS:
            return False
        if src == dst:
            return True
        return self._reachable(relation, dst, src)

    def add(self, edge: AdmittedEdge) -> None:
        """Insert an admitted edge. Caller (Bookkeeper) has already validated it."""
        self._edges[edge_key(edge.src_chunk_id, edge.dst_chunk_id, edge.relation)] = edge
        if edge.relation in ORDERING_RELATIONS:
            self._adj.setdefault(edge.relation, {}).setdefault(
                edge.src_chunk_id, set()
            ).add(edge.dst_chunk_id)
