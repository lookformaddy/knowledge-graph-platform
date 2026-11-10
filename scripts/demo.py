"""End-to-end demo: build the graph from the synthetic corpus, run the
extraction + golden QA eval, answer multi-hop questions, show
communities and centrality, merge a duplicate entity, snapshot, and
verify the audit chain — all offline, no network, no API keys.

Expected outcome:
  1. Graph builds: 17 nodes, 20 edges.
  2. Extraction eval: P/R/F1 = 1.000 for entities and relations.
  3. Golden QA: 5/5 questions answered exactly.
  4. A duplicate "Alice C." node is merged into Alice Chen; the audit
     log records it and the chain still verifies.
  5. A second snapshot is taken; the v1 snapshot still shows the
     pre-merge graph (time travel).
Reports are written to ./kg-reports/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kgraph.cli import build_store  # noqa: E402
from kgraph.data import DOCUMENTS  # noqa: E402
from kgraph.eval import evaluate_extraction, evaluate_qa  # noqa: E402
from kgraph.extractors import extract_corpus  # noqa: E402
from kgraph.models import Node  # noqa: E402
from kgraph.query import (GraphQuery, label_propagation_communities,  # noqa: E402
                          pagerank, run_query, shortest_path)  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "kg-reports"
    out_dir.mkdir(exist_ok=True)

    store = build_store()
    view = store.current_view()
    stats = view.stats()
    print(f"Graph built: {stats['nodes']} nodes, {stats['edges']} edges")
    for node_type, count in sorted(stats["nodes_by_type"].items()):
        print(f"  {node_type}: {count}")

    extraction = evaluate_extraction(extract_corpus(DOCUMENTS))
    p, r, f1 = extraction.micro_entities
    print(f"\nExtraction entities:  P={p:.3f} R={r:.3f} F1={f1:.3f}")
    p, r, f1 = extraction.micro_relations
    print(f"Extraction relations: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    qa = evaluate_qa(view)
    print(f"\nGolden QA: {qa.accuracy:.3f} "
          f"({sum(1 for q in qa.questions if q.correct)}/{len(qa.questions)})")
    for q in qa.questions:
        mark = "PASS" if q.correct else "FAIL"
        print(f"  [{mark}] {q.question_id}: {q.question}")
        print(f"        -> {', '.join(sorted(q.got))}")

    # A two-hop dependency question, answered live.
    ekg = view.find_by_label("Enterprise Knowledge Graph")
    deps = run_query(view, GraphQuery(
        from_type="Project", edge_types=["DEPENDS_ON"],
        min_hops=1, max_hops=2, to_type=None,
        start_label="Enterprise Knowledge Graph"))
    print("\nWhat does Enterprise Knowledge Graph transitively depend on?")
    print(f"  -> {', '.join(n.label for n in deps)}")

    carla = view.find_by_label("Carla Ruiz")
    risk = view.find_by_label("PII Exposure in Search Logs")
    path = shortest_path(view, carla.node_id, risk.node_id)
    print("\nShortest path: Carla Ruiz -> PII Exposure in Search Logs")
    print("  " + " -> ".join(view.nodes[nid].label for nid in path))

    communities = label_propagation_communities(view)
    print(f"\nCommunities: {len(communities)} "
          f"(largest: {max(len(m) for m in communities.values())} nodes)")

    ranks = pagerank(view)
    top = sorted(ranks.items(), key=lambda kv: -kv[1])[:3]
    print("Top PageRank nodes:")
    for node_id, score in top:
        print(f"  {score:.4f}  {view.nodes[node_id].label}")

    # Entity resolution: a duplicate record for Alice Chen arrives.
    dup = Node(node_id="person-alice-c", node_type="Person",
               label="Alice C.")
    store.add_node(dup, actor="demo")
    store.add_edge("person-alice-c", "project-fraud-signals", "WORKS_ON",
                   actor="demo")
    result = store.merge_nodes("person-alice-chen", "person-alice-c",
                               actor="demo", reason="duplicate record")
    print(f"\nMerged duplicate 'Alice C.' into Alice Chen: "
          f"{result.edges_rewired} edge(s) rewired, "
          f"{result.edges_dropped_as_duplicates} duplicate(s) dropped")
    kept = store.current_view().node("person-alice-chen")
    print(f"  aliases now: {kept.aliases}")

    snap = store.snapshot("post-merge", actor="demo")
    print(f"\nSnapshot v{snap.version} '{snap.label}' taken "
          f"({len(snap.nodes)} nodes, {len(snap.edges)} edges)")
    v1 = store.read_at(1)
    print(f"Time travel: v1 still has {v1.stats()['nodes']} nodes "
          f"(pre-merge graph)")

    ok, reason = store.audit.verify_chain()
    print(f"\nAudit log: {reason}")
    print(f"  entries: {len(store.audit.entries)}")

    report = {
        "nodes": stats["nodes"], "edges": stats["edges"],
        "extraction_entities_f1": extraction.micro_entities[2],
        "extraction_relations_f1": extraction.micro_relations[2],
        "qa_accuracy": qa.accuracy,
        "communities": len(communities),
        "audit_valid": ok,
    }
    (out_dir / "demo-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_dir / 'demo-report.json'}")
    print("Demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
