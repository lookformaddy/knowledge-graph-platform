"""API tests (run in the venv where FastAPI is installed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import kgraph.service as service
from kgraph.cli import build_store


@pytest.fixture
def client():
    service.store = build_store()
    return TestClient(service.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["nodes"] == 17
    assert resp.json()["edges"] == 20


def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["nodes_by_type"]["Person"] == 5


def test_list_nodes_redacts_pii_for_analyst(client):
    resp = client.get("/nodes", params={"role": "analyst"})
    assert resp.status_code == 200
    alice = next(n for n in resp.json()["nodes"]
                 if n["label"] == "Alice Chen")
    assert alice["attributes"]["email"] == "***"


def test_list_nodes_viewer_cannot_see_persons(client):
    resp = client.get("/nodes", params={"role": "viewer"})
    labels = {n["label"] for n in resp.json()["nodes"]}
    assert "Alice Chen" not in labels
    assert "Semantic Layer" in labels


def test_get_node_404(client):
    assert client.get("/nodes/no-such-node").status_code == 404


def test_get_node_forbidden_type(client):
    resp = client.get("/nodes/risk-pii-exposure", params={"role": "viewer"})
    assert resp.status_code == 403


def test_query_endpoint(client):
    resp = client.post("/query", json={
        "query": "Person -[WORKS_ON*1..1]-> Project",
        "start_label": "Alice Chen"})
    assert resp.status_code == 200
    assert {n["label"] for n in resp.json()["results"]} == {
        "Semantic Layer", "Enterprise Knowledge Graph"}


def test_query_bad_syntax_400(client):
    resp = client.post("/query", json={"query": "not a query"})
    assert resp.status_code == 400


def test_neighbors_endpoint(client):
    resp = client.get("/nodes/project-knowledge-graph/neighbors",
                      params={"max_hops": 2})
    assert resp.status_code == 200
    labels = {n["label"] for n in resp.json()["nodes"]}
    assert "Semantic Layer" in labels
    assert "Atlas" in labels


def test_shortest_path_endpoint(client):
    resp = client.get("/shortest-path",
                      params={"source": "Carla Ruiz",
                              "target": "PII Exposure in Search Logs"})
    assert resp.status_code == 200
    assert resp.json()["path"] == [
        "Carla Ruiz", "Northstar", "PII Exposure in Search Logs"]


def test_shortest_path_no_path_404(client):
    resp = client.get("/shortest-path",
                      params={"source": "Alice Chen",
                              "target": "PII Exposure in Search Logs"})
    assert resp.status_code == 404


def test_communities_endpoint(client):
    resp = client.get("/communities")
    assert resp.status_code == 200
    members = [n for group in resp.json()["communities"].values()
               for n in group]
    assert len(members) == 17


def test_centrality_endpoint(client):
    resp = client.get("/centrality")
    assert resp.status_code == 200
    ranks = resp.json()["pagerank"]
    assert len(ranks) == 17
    assert abs(sum(r["score"] for r in ranks) - 1.0) < 1e-6


def test_eval_endpoint(client):
    resp = client.get("/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction"]["micro_entities"] == [1.0, 1.0, 1.0]
    assert body["extraction"]["micro_relations"] == [1.0, 1.0, 1.0]
    assert body["qa"]["accuracy"] == 1.0


def test_audit_endpoint(client):
    resp = client.get("/audit")
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
    assert len(resp.json()["entries"]) > 0


def test_snapshots_endpoint(client):
    resp = client.get("/snapshots")
    assert resp.status_code == 200
    assert len(resp.json()["snapshots"]) == 1
    assert resp.json()["snapshots"][0]["version"] == 1
    stats = client.get("/snapshots/1/stats")
    assert stats.status_code == 200
    assert stats.json()["nodes"] == 17
    assert client.get("/snapshots/99/stats").status_code == 404


def test_merge_endpoint(client):
    # Add a duplicate node first via the store, then merge via the API.
    from kgraph.models import Node
    service.store.add_node(Node(node_id="person-alice-c",
                                node_type="Person", label="Alice C."),
                           actor="test")
    service.store.add_edge("person-alice-c", "project-fraud-signals",
                           "WORKS_ON", actor="test")
    resp = client.post("/nodes/merge", json={
        "keep_id": "person-alice-chen", "drop_id": "person-alice-c",
        "reason": "duplicate record"})
    assert resp.status_code == 200
    assert resp.json()["edges_rewired"] == 1
    assert client.get("/nodes/person-alice-c").status_code == 404
