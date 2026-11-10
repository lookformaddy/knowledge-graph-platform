"""FastAPI surface for the knowledge graph platform.

Thin HTTP wrapper over the pure-stdlib core: it owns one GraphStore
built from the synthetic corpus, and translates domain errors into
HTTP responses (ValueError/KeyError/PermissionError -> 400/404/403).
All extraction, query, governance, and eval logic lives in the core,
so the API adds no second source of truth.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kgraph import __version__
from kgraph.cli import build_store
from kgraph.eval import evaluate_extraction, evaluate_qa
from kgraph.extractors import extract_corpus
from kgraph.data import DOCUMENTS
from kgraph.governance import can_read, enforce_read, redact
from kgraph.query import (label_propagation_communities,
                          pagerank, parse_query, run_query, shortest_path)

ACTOR = os.environ.get("KGRAPH_ACTOR", "api")

app = FastAPI(title="Knowledge Graph Platform", version=__version__)

store = build_store()


class QueryRequest(BaseModel):
    query: str
    start_label: str | None = None
    role: str = "admin"


class MergeRequest(BaseModel):
    keep_id: str
    drop_id: str
    reason: str


@app.get("/health")
def health() -> dict:
    stats = store.current_view().stats()
    return {"status": "ok", "nodes": stats["nodes"], "edges": stats["edges"]}


@app.get("/stats")
def stats() -> dict:
    return store.current_view().stats()


@app.get("/nodes")
def list_nodes(role: str = "admin") -> dict:
    nodes = []
    for node in sorted(store.current_view().nodes.values(),
                       key=lambda n: n.label):
        if can_read(role, node.node_type):
            nodes.append(redact(node, role).to_dict())
    return {"nodes": nodes}


@app.get("/nodes/{node_id}")
def get_node(node_id: str, role: str = "admin") -> dict:
    node = store.current_view().node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node: {node_id}")
    try:
        return enforce_read(role, node).to_dict()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/query")
def run_graph_query(req: QueryRequest) -> dict:
    try:
        query = parse_query(req.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.start_label:
        query.start_label = req.start_label
    view = store.current_view()
    nodes = [redact(n, req.role).to_dict() for n in run_query(view, query)
             if can_read(req.role, n.node_type)]
    return {"results": nodes}


@app.get("/nodes/{node_id}/neighbors")
def neighbors(node_id: str, max_hops: int = 1, role: str = "admin") -> dict:
    from kgraph.query import k_hop_neighborhood
    view = store.current_view()
    if view.node(node_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown node: {node_id}")
    distances = k_hop_neighborhood(view, node_id, max_hops=max_hops)
    nodes = []
    for nid, hops in sorted(distances.items(), key=lambda kv: (kv[1], kv[0])):
        node = view.nodes[nid]
        if can_read(role, node.node_type):
            nodes.append({**redact(node, role).to_dict(), "hops": hops})
    return {"nodes": nodes}


@app.get("/shortest-path")
def get_shortest_path(source: str, target: str) -> dict:
    view = store.current_view()
    src = view.find_by_label(source)
    tgt = view.find_by_label(target)
    if src is None or tgt is None:
        raise HTTPException(status_code=404, detail="unknown node label")
    path = shortest_path(view, src.node_id, tgt.node_id)
    if path is None:
        raise HTTPException(status_code=404, detail="no path")
    return {"path": [view.nodes[nid].label for nid in path]}


@app.get("/communities")
def communities() -> dict:
    view = store.current_view()
    return {"communities": {
        label: [view.nodes[nid].label for nid in members]
        for label, members in label_propagation_communities(view).items()
    }}


@app.get("/centrality")
def centrality() -> dict:
    view = store.current_view()
    ranks = pagerank(view)
    return {"pagerank": [
        {"node_id": nid, "label": view.nodes[nid].label, "score": score}
        for nid, score in sorted(ranks.items(), key=lambda kv: -kv[1])
    ]}


@app.post("/nodes/merge")
def merge_nodes(req: MergeRequest) -> dict:
    try:
        result = store.merge_nodes(req.keep_id, req.drop_id,
                                   actor=ACTOR, reason=req.reason)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"kept_id": result.kept_id, "dropped_id": result.dropped_id,
            "edges_rewired": result.edges_rewired,
            "edges_dropped_as_duplicates": result.edges_dropped_as_duplicates}


@app.get("/snapshots")
def list_snapshots() -> dict:
    return {"snapshots": [
        {"version": s.version, "label": s.label,
         "nodes": len(s.nodes), "edges": len(s.edges)}
        for s in store.list_snapshots()
    ]}


@app.get("/snapshots/{version}/stats")
def snapshot_stats(version: int) -> dict:
    try:
        return store.read_at(version).stats()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/eval")
def run_eval() -> dict:
    extraction = evaluate_extraction(extract_corpus(DOCUMENTS))
    qa = evaluate_qa(store.current_view())
    return {"extraction": extraction.to_dict(), "qa": qa.to_dict()}


@app.get("/audit")
def audit_log() -> dict:
    ok, reason = store.audit.verify_chain()
    return {"valid": ok, "reason": reason,
            "entries": store.audit.to_dicts()}
