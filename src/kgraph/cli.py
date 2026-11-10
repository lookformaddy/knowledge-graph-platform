"""CLI for the knowledge graph platform: build the graph from the
synthetic corpus, query it, run the eval harness, inspect snapshots
and the audit log. Thin wrapper over the stdlib core — no second
source of truth.
"""
from __future__ import annotations

import argparse
import sys

from kgraph.data import DOCUMENTS, build_canonical_graph_data
from kgraph.eval import evaluate_extraction, evaluate_qa
from kgraph.extractors import extract_corpus
from kgraph.governance import AuditLog, can_read, redact
from kgraph.query import (GraphQuery, label_propagation_communities,
                          pagerank, parse_query, run_query, shortest_path)
from kgraph.store import GraphStore


def build_store() -> GraphStore:
    """Build the canonical graph from the synthetic corpus: extract
    entities/relations from every document, load them into the store,
    and snapshot the result."""
    store = GraphStore(AuditLog())
    results = extract_corpus(DOCUMENTS)
    nodes, edges = build_canonical_graph_data()
    for node in nodes.values():
        store.add_node(node, actor="ingest")
    for edge in edges.values():
        store.add_edge(edge.source_id, edge.target_id, edge.edge_type,
                       actor="ingest")
    store.snapshot("canonical-corpus-v1", actor="ingest")
    return store


def _print_stats(store: GraphStore) -> None:
    stats = store.current_view().stats()
    print(f"nodes: {stats['nodes']}  edges: {stats['edges']}")
    print("nodes by type:")
    for node_type, count in sorted(stats["nodes_by_type"].items()):
        print(f"  {node_type}: {count}")
    print("edges by type:")
    for edge_type, count in sorted(stats["edges_by_type"].items()):
        print(f"  {edge_type}: {count}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kgraph",
                                     description="Enterprise knowledge graph platform")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="print graph statistics")
    q = sub.add_parser("query", help="run a query-language line, e.g. "
                       "'Person -[WORKS_ON*1..2]-> Project'")
    q.add_argument("query")
    q.add_argument("--start", default=None,
                   help="start node label (restricts the query)")
    q.add_argument("--role", default="admin", help="role for RBAC/redaction")
    p = sub.add_parser("path", help="shortest path between two node labels")
    p.add_argument("source")
    p.add_argument("target")
    sub.add_parser("eval", help="extraction + golden QA eval")
    sub.add_parser("communities", help="label-propagation communities")
    sub.add_parser("centrality", help="PageRank top nodes")
    sub.add_parser("audit", help="print the audit log and verify its chain")

    args = parser.parse_args(argv)
    store = build_store()
    view = store.current_view()

    if args.command == "stats":
        _print_stats(store)
    elif args.command == "query":
        query = parse_query(args.query)
        if args.start:
            query.start_label = args.start
        for node in run_query(view, query):
            if can_read(args.role, node.node_type):
                print(f"{node.node_type}: {redact(node, args.role).label}")
    elif args.command == "path":
        source = view.find_by_label(args.source)
        target = view.find_by_label(args.target)
        if source is None or target is None:
            print("unknown node label", file=sys.stderr)
            return 2
        path = shortest_path(view, source.node_id, target.node_id)
        if path is None:
            print("no path")
        else:
            print(" -> ".join(view.nodes[nid].label for nid in path))
    elif args.command == "eval":
        extraction = evaluate_extraction(extract_corpus(DOCUMENTS))
        qa = evaluate_qa(view)
        p, r, f1 = extraction.micro_entities
        print(f"extraction entities:  P={p:.3f} R={r:.3f} F1={f1:.3f}")
        p, r, f1 = extraction.micro_relations
        print(f"extraction relations: P={p:.3f} R={r:.3f} F1={f1:.3f}")
        print(f"golden QA accuracy: {qa.accuracy:.3f} "
              f"({sum(1 for q in qa.questions if q.correct)}/{len(qa.questions)})")
        for q in qa.questions:
            mark = "PASS" if q.correct else "FAIL"
            print(f"  [{mark}] {q.question_id}: {q.question}")
            if not q.correct:
                print(f"        expected {sorted(q.expected)} got {sorted(q.got)}")
    elif args.command == "communities":
        for label, members in label_propagation_communities(view).items():
            names = [view.nodes[nid].label for nid in members]
            print(f"community {label[:12]}: {', '.join(sorted(names))}")
    elif args.command == "centrality":
        ranks = pagerank(view)
        for node_id, score in sorted(ranks.items(), key=lambda kv: -kv[1])[:5]:
            print(f"{score:.4f}  {view.nodes[node_id].label} "
                  f"({view.nodes[node_id].node_type})")
    elif args.command == "audit":
        ok, reason = store.audit.verify_chain()
        print(reason)
        for entry in store.audit.entries:
            print(f"  #{entry.seq} {entry.actor} {entry.action}: {entry.detail}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
