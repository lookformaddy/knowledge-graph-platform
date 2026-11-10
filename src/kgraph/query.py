"""Graph query engine: traversal, shortest path, communities,
centrality, subgraph extraction, and a small declarative query
language.

All queries run against a ``GraphView`` (live store or snapshot), so
the same code answers questions about the current graph and about
any point in history. Everything is deterministic: traversals visit
neighbours in sorted id order, label propagation breaks ties by the
smallest label, and PageRank starts from a uniform vector.

The query language is one line:

    Person -[WORKS_ON*1..2]-> Project
    Risk <-[EXPOSES*1..1]- Product

i.e. ``<NodeType> -[<EdgeType>*<min>..<max>]-> <NodeType>`` for a
forward hop, or ``<NodeType> <-[<EdgeType>*<min>..<max>]- <NodeType>``
for a reverse hop. ``parse_query`` turns that string into a
``GraphQuery``; ``run_query`` executes it.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from kgraph.models import EDGE_TYPES, NODE_TYPES
from kgraph.store import GraphView

_QUERY_RE = re.compile(
    r"^\s*(?P<from_type>\w+)\s*"
    r"(?P<arrow_f>-|\u003c-)\[(?P<edge_type>[A-Z_]+)"
    r"(?:\*(?P<min_hops>\d+)\.\.(?P<max_hops>\d+))?\]"
    r"(?P<arrow_t>->|-)\s*(?P<to_type>\w+)\s*$"
)


@dataclass
class GraphQuery:
    """Structured query: start node type, edge types, hop range,
    target node type, direction."""

    from_type: str
    edge_types: list
    min_hops: int = 1
    max_hops: int = 1
    to_type: str | None = None
    reverse: bool = False
    start_label: str | None = None

    def __post_init__(self):
        if self.from_type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {self.from_type!r}")
        if self.to_type is not None and self.to_type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {self.to_type!r}")
        for edge_type in self.edge_types:
            if edge_type not in EDGE_TYPES:
                raise ValueError(f"unknown edge type: {edge_type!r}")
        if not 0 <= self.min_hops <= self.max_hops:
            raise ValueError("hop range must satisfy 0 <= min <= max")


def parse_query(text: str) -> GraphQuery:
    """Parse one line of the query language (see module docstring)."""
    match = _QUERY_RE.match(text)
    if not match:
        raise ValueError(
            "query must look like: Person -[WORKS_ON*1..2]-> Project "
            f"(got {text!r})")
    reverse = match.group("arrow_f") == "<-"
    min_hops = int(match.group("min_hops") or 1)
    max_hops = int(match.group("max_hops") or min_hops)
    return GraphQuery(
        from_type=match.group("from_type"),
        edge_types=[match.group("edge_type")],
        min_hops=min_hops, max_hops=max_hops,
        to_type=match.group("to_type"), reverse=reverse)


def k_hop_neighborhood(view: GraphView, start_id: str,
                       edge_types=None, max_hops: int = 1,
                       reverse: bool = False) -> dict:
    """Breadth-first neighbourhood: {node_id: hop distance} for every
    node reachable within ``max_hops``. Deterministic (sorted order)."""
    if start_id not in view.nodes:
        raise KeyError(f"unknown node: {start_id}")
    distances = {start_id: 0}
    frontier = [start_id]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        for node_id in sorted(frontier):
            for nbr in view.neighbors(node_id, edge_types, reverse=reverse):
                if nbr.node_id not in distances:
                    distances[nbr.node_id] = hop
                    next_frontier.append(nbr.node_id)
        frontier = next_frontier
    return distances


def run_query(view: GraphView, query: GraphQuery) -> list:
    """Execute a structured query. Returns the matching target nodes
    (sorted by label), starting from every node of ``from_type``
    (or just ``start_label`` when given)."""
    starts = [n for n in view.nodes.values()
              if n.node_type == query.from_type
              and (query.start_label is None or n.label == query.start_label)]
    found: dict[str, object] = {}
    for start in starts:
        distances = k_hop_neighborhood(view, start.node_id, query.edge_types,
                                       query.max_hops, query.reverse)
        for node_id, hops in distances.items():
            if node_id == start.node_id:
                continue
            if hops < query.min_hops:
                continue
            node = view.nodes[node_id]
            if query.to_type is not None and node.node_type != query.to_type:
                continue
            found[node_id] = node
    return sorted(found.values(), key=lambda n: n.label)


def shortest_path(view: GraphView, source_id: str, target_id: str,
                  edge_types=None) -> list | None:
    """Breadth-first shortest path; returns the list of node ids, or
    None when no path exists. Ties break toward the smallest node id
    because neighbours are visited in sorted order."""
    if source_id not in view.nodes:
        raise KeyError(f"unknown node: {source_id}")
    if target_id not in view.nodes:
        raise KeyError(f"unknown node: {target_id}")
    if source_id == target_id:
        return [source_id]
    prev: dict[str, str | None] = {source_id: None}
    queue = deque([source_id])
    while queue:
        current = queue.popleft()
        for nbr in sorted(view.neighbors(current, edge_types),
                          key=lambda n: n.node_id):
            if nbr.node_id in prev:
                continue
            prev[nbr.node_id] = current
            if nbr.node_id == target_id:
                path = [target_id]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            queue.append(nbr.node_id)
    return None


def label_propagation_communities(view: GraphView,
                                 max_iterations: int = 100) -> dict:
    """Community detection by label propagation on the undirected
    graph. Each node starts with its own label; each iteration a node
    adopts the most common label among its neighbours (ties broken by
    the smallest label, so the result is deterministic). Returns
    {community_label: [node ids]}."""
    labels = {nid: nid for nid in view.nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in view.edges.values():
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    for _ in range(max_iterations):
        changed = False
        for node_id in sorted(view.nodes):
            neighbour_labels = Counter(labels[n] for n in adjacency[node_id])
            if not neighbour_labels:
                continue
            top_count = max(neighbour_labels.values())
            best = min(label for label, count in neighbour_labels.items()
                       if count == top_count)
            if labels[node_id] != best:
                labels[node_id] = best
                changed = True
        if not changed:
            break
    communities: dict[str, list[str]] = defaultdict(list)
    for node_id, label in labels.items():
        communities[label].append(node_id)
    return {label: sorted(members)
            for label, members in sorted(communities.items())}


def degree_centrality(view: GraphView) -> dict:
    """Degree centrality (in + out degree, normalized by n - 1)."""
    n = len(view.nodes)
    if n <= 1:
        return {nid: 0.0 for nid in view.nodes}
    scores = {}
    for node_id in view.nodes:
        degree = len(view.out_edges(node_id)) + len(view.in_edges(node_id))
        scores[node_id] = degree / (n - 1)
    return scores


def pagerank(view: GraphView, damping: float = 0.85,
             iterations: int = 100, tolerance: float = 1e-6) -> dict:
    """PageRank over directed edges, uniform start, deterministic.
    Dangling nodes (no out-edges) distribute their rank uniformly."""
    n = len(view.nodes)
    if n == 0:
        return {}
    ids = sorted(view.nodes)
    rank = {nid: 1.0 / n for nid in ids}
    out_degree = {nid: len(view.out_edges(nid)) for nid in ids}
    for _ in range(iterations):
        new_rank = {}
        dangling_sum = sum(rank[nid] for nid in ids if out_degree[nid] == 0)
        for nid in ids:
            incoming = sum(rank[src.source_id] / out_degree[src.source_id]
                           for src in view.in_edges(nid)
                           if out_degree[src.source_id] > 0)
            new_rank[nid] = ((1 - damping) / n
                             + damping * (incoming + dangling_sum / n))
        delta = max(abs(new_rank[nid] - rank[nid]) for nid in ids)
        rank = new_rank
        if delta < tolerance:
            break
    return rank


def extract_subgraph(view: GraphView, node_ids) -> GraphView:
    """Induced subgraph: the given nodes plus every edge whose
    endpoints are both in the set."""
    wanted = set(node_ids)
    nodes = {nid: view.nodes[nid] for nid in wanted if nid in view.nodes}
    edges = {eid: e for eid, e in view.edges.items()
             if e.source_id in wanted and e.target_id in wanted}
    return GraphView(nodes, edges)
