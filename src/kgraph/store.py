"""Graph store: typed nodes/edges, versioned snapshots with
time-travel reads, and entity resolution (merge/split) with an audit
trail.

Snapshots are full deep copies of the node/edge maps. That is
deliberately simple: at enterprise scale this would be a delta log
or a copy-on-write store, but full copies make time-travel reads
trivially correct and auditable, and the corpus here is small. The
trade-off is documented in docs/architecture.md.

Merge/split are the entity-resolution primitives: when two nodes turn
out to be the same real-world entity, ``merge_nodes`` keeps one id,
rewires every edge, and records the dropped id as an alias; when one
node turns out to be two, ``split_node`` moves a chosen set of edges
to a new node. Both operations append to the audit log.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from kgraph.governance import AuditLog
from kgraph.models import Edge, Node, Snapshot


@dataclass
class MergeResult:
    kept_id: str
    dropped_id: str
    edges_rewired: int
    edges_dropped_as_duplicates: int


class GraphStore:
    """In-memory typed graph with snapshots and entity resolution."""

    def __init__(self, audit: AuditLog | None = None):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._snapshots: list[Snapshot] = []
        self._next_version = 1
        self._next_edge = 1
        self.audit = audit or AuditLog()

    # -- basic mutation -------------------------------------------------

    def add_node(self, node: Node, actor: str = "system") -> Node:
        if node.node_id in self.nodes:
            raise ValueError(f"node already exists: {node.node_id}")
        self.nodes[node.node_id] = node
        self.audit.append(actor, "node.add",
                          f"{node.node_type}:{node.node_id} ({node.label})")
        return node

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                 actor: str = "system", attributes: dict | None = None) -> Edge:
        if source_id not in self.nodes:
            raise ValueError(f"unknown source node: {source_id}")
        if target_id not in self.nodes:
            raise ValueError(f"unknown target node: {target_id}")
        edge_id = f"edge-{self._next_edge:04d}"
        self._next_edge += 1
        edge = Edge(edge_id=edge_id, source_id=source_id, target_id=target_id,
                    edge_type=edge_type, attributes=dict(attributes or {}))
        self.edges[edge_id] = edge
        self.audit.append(actor, "edge.add",
                          f"{source_id} -[{edge_type}]-> {target_id}")
        return edge

    def remove_node(self, node_id: str, actor: str = "system") -> Node:
        node = self.nodes.pop(node_id, None)
        if node is None:
            raise KeyError(f"unknown node: {node_id}")
        dropped = [eid for eid, e in self.edges.items()
                   if e.source_id == node_id or e.target_id == node_id]
        for eid in dropped:
            del self.edges[eid]
        self.audit.append(actor, "node.remove",
                          f"{node.node_id} ({node.label}); "
                          f"{len(dropped)} incident edges removed")
        return node

    # -- entity resolution ----------------------------------------------

    def merge_nodes(self, keep_id: str, drop_id: str, actor: str,
                    reason: str) -> MergeResult:
        """Merge ``drop_id`` into ``keep_id``: rewire every incident
        edge to ``keep_id``, drop edges that would become duplicates
        or self-loops, keep the dropped label as an alias."""
        if keep_id not in self.nodes:
            raise KeyError(f"unknown node: {keep_id}")
        if drop_id not in self.nodes:
            raise KeyError(f"unknown node: {drop_id}")
        if keep_id == drop_id:
            raise ValueError("cannot merge a node into itself")
        kept = self.nodes[keep_id]
        dropped = self.nodes[drop_id]
        kept.aliases.append(dropped.label)
        kept.aliases.extend(dropped.aliases)

        rewired = 0
        duplicates = 0
        existing = {(e.source_id, e.edge_type, e.target_id)
                    for e in self.edges.values()}
        for edge in list(self.edges.values()):
            if edge.source_id != drop_id and edge.target_id != drop_id:
                continue
            new_src = keep_id if edge.source_id == drop_id else edge.source_id
            new_tgt = keep_id if edge.target_id == drop_id else edge.target_id
            if new_src == new_tgt:
                del self.edges[edge.edge_id]
                duplicates += 1
                continue
            if (new_src, edge.edge_type, new_tgt) in existing:
                del self.edges[edge.edge_id]
                duplicates += 1
                continue
            edge.source_id, edge.target_id = new_src, new_tgt
            existing.add((new_src, edge.edge_type, new_tgt))
            rewired += 1
        del self.nodes[drop_id]
        self.audit.append(actor, "node.merge",
                          f"{drop_id} -> {keep_id}; reason: {reason}; "
                          f"{rewired} edges rewired, {duplicates} dropped")
        return MergeResult(kept_id=keep_id, dropped_id=drop_id,
                           edges_rewired=rewired,
                           edges_dropped_as_duplicates=duplicates)

    def split_node(self, node_id: str, new_node: Node,
                   edge_ids_to_move: list[str], actor: str,
                   reason: str) -> Node:
        """Split ``node_id``: move the listed edges to ``new_node``.
        The caller decides which edges belong to the new entity; the
        store only enforces that the edges exist and touch the node
        being split."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node: {node_id}")
        if new_node.node_id in self.nodes:
            raise ValueError(f"node already exists: {new_node.node_id}")
        moved = 0
        for edge_id in edge_ids_to_move:
            edge = self.edges.get(edge_id)
            if edge is None:
                raise KeyError(f"unknown edge: {edge_id}")
            if edge.source_id == node_id:
                edge.source_id = new_node.node_id
                moved += 1
            elif edge.target_id == node_id:
                edge.target_id = new_node.node_id
                moved += 1
            else:
                raise ValueError(f"edge {edge_id} does not touch {node_id}")
        self.nodes[new_node.node_id] = new_node
        self.audit.append(actor, "node.split",
                          f"{node_id} -> {node_id} + {new_node.node_id}; "
                          f"reason: {reason}; {moved} edges moved")
        return new_node

    # -- snapshots / time travel ----------------------------------------

    def snapshot(self, label: str, actor: str = "system") -> Snapshot:
        snap = Snapshot(version=self._next_version, label=label,
                        nodes=copy.deepcopy(self.nodes),
                        edges=copy.deepcopy(self.edges))
        self._snapshots.append(snap)
        self._next_version += 1
        self.audit.append(actor, "snapshot.create",
                          f"v{snap.version} '{label}' "
                          f"({len(snap.nodes)} nodes, {len(snap.edges)} edges)")
        return snap

    def list_snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    def read_at(self, version: int) -> "GraphView":
        """Time-travel read: a read-only view of the graph as of a
        snapshot version. Mutations after the snapshot are invisible."""
        for snap in self._snapshots:
            if snap.version == version:
                return GraphView(snap.nodes, snap.edges)
        raise KeyError(f"no snapshot with version {version}")

    def current_view(self) -> "GraphView":
        return GraphView(self.nodes, self.edges)


class GraphView:
    """Read-only view over a node/edge map (live store or snapshot).
    Query code works against views, never against the mutable store,
    so a query can never observe a half-applied mutation."""

    def __init__(self, nodes: dict, edges: dict):
        self.nodes = nodes
        self.edges = edges

    def node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def find_by_label(self, label: str) -> Node | None:
        for node in self.nodes.values():
            if node.label == label or label in node.aliases:
                return node
        return None

    def out_edges(self, node_id: str, edge_types=None) -> list[Edge]:
        return [e for e in self.edges.values()
                if e.source_id == node_id
                and (edge_types is None or e.edge_type in edge_types)]

    def in_edges(self, node_id: str, edge_types=None) -> list[Edge]:
        return [e for e in self.edges.values()
                if e.target_id == node_id
                and (edge_types is None or e.edge_type in edge_types)]

    def neighbors(self, node_id: str, edge_types=None,
                  reverse: bool = False) -> list[Node]:
        edges = self.in_edges(node_id, edge_types) if reverse \
            else self.out_edges(node_id, edge_types)
        other = (e.source_id for e in edges) if reverse \
            else (e.target_id for e in edges)
        return [self.nodes[nid] for nid in other if nid in self.nodes]

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for node in self.nodes.values():
            by_type[node.node_type] = by_type.get(node.node_type, 0) + 1
        by_edge: dict[str, int] = {}
        for edge in self.edges.values():
            by_edge[edge.edge_type] = by_edge.get(edge.edge_type, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "nodes_by_type": by_type, "edges_by_type": by_edge}
