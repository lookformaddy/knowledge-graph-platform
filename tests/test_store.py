"""Store tests: mutation guards, snapshots/time travel, merge/split."""
import pytest

from kgraph.cli import build_store
from kgraph.data import build_canonical_graph_data
from kgraph.models import Edge, Node
from kgraph.store import GraphStore


@pytest.fixture
def store():
    return build_store()


def test_canonical_graph_shape(store):
    view = store.current_view()
    assert view.stats()["nodes"] == 17
    assert view.stats()["edges"] == 20
    assert view.stats()["nodes_by_type"]["Person"] == 5
    assert view.stats()["nodes_by_type"]["Risk"] == 3


def test_add_duplicate_node_rejected(store):
    node = Node(node_id="person-alice-chen", node_type="Person",
                label="Alice Chen")
    with pytest.raises(ValueError):
        store.add_node(node)


def test_add_edge_with_unknown_endpoint_rejected(store):
    with pytest.raises(ValueError):
        store.add_edge("person-alice-chen", "no-such-node", "WORKS_ON")


def test_invalid_node_type_rejected():
    with pytest.raises(ValueError):
        Node(node_id="x", node_type="Alien", label="X")


def test_invalid_edge_type_rejected():
    with pytest.raises(ValueError):
        Edge(edge_id="e1", source_id="a", target_id="b", edge_type="LOVES")


def test_snapshot_and_time_travel(store):
    snap = store.snapshot("before-merge", actor="test")
    assert snap.version == 2  # v1 was created by build_store
    store.remove_node("risk-vendor-lockin", actor="test")
    assert store.current_view().node("risk-vendor-lockin") is None
    old = store.read_at(snap.version)
    assert old.node("risk-vendor-lockin") is not None
    assert old.stats()["nodes"] == 17
    assert store.current_view().stats()["nodes"] == 16


def test_read_at_unknown_version_raises(store):
    with pytest.raises(KeyError):
        store.read_at(999)


def test_merge_rewires_edges_and_keeps_alias(store):
    nodes, _edges = build_canonical_graph_data()
    dup = Node(node_id="person-alice-c", node_type="Person",
               label="Alice C.", attributes={})
    store.add_node(dup, actor="test")
    store.add_edge("person-alice-c", "project-fraud-signals", "WORKS_ON",
                   actor="test")
    result = store.merge_nodes("person-alice-chen", "person-alice-c",
                               actor="test", reason="duplicate record")
    assert result.kept_id == "person-alice-chen"
    assert result.edges_rewired == 1
    kept = store.current_view().node("person-alice-chen")
    assert "Alice C." in kept.aliases
    assert store.current_view().node("person-alice-c") is None
    # The rewired edge now points at the kept node.
    targets = {e.target_id for e in
               store.current_view().out_edges("person-alice-chen",
                                              ["WORKS_ON"])}
    assert "project-fraud-signals" in targets


def test_merge_drops_duplicate_edges(store):
    dup = Node(node_id="person-alice-c", node_type="Person", label="Alice C.")
    store.add_node(dup, actor="test")
    store.add_edge("person-alice-c", "project-semantic-layer", "WORKS_ON",
                   actor="test")
    result = store.merge_nodes("person-alice-chen", "person-alice-c",
                               actor="test", reason="duplicate record")
    assert result.edges_dropped_as_duplicates == 1
    works_on = store.current_view().out_edges("person-alice-chen",
                                              ["WORKS_ON"])
    assert len({e.target_id for e in works_on}) == len(works_on)


def test_merge_same_node_rejected(store):
    with pytest.raises(ValueError):
        store.merge_nodes("person-alice-chen", "person-alice-chen",
                          actor="test", reason="no-op")


def test_split_moves_edges(store):
    new_node = Node(node_id="project-semantic-layer-v2", node_type="Project",
                    label="Semantic Layer v2")
    # Move the DEPENDS_ON edge from Semantic Layer to the new node.
    edge_id = next(e.edge_id for e in store.current_view().edges.values()
                   if e.source_id == "project-semantic-layer"
                   and e.edge_type == "DEPENDS_ON")
    store.split_node("project-semantic-layer", new_node, [edge_id],
                     actor="test", reason="fork")
    view = store.current_view()
    assert view.node("project-semantic-layer-v2") is not None
    assert view.edges[edge_id].source_id == "project-semantic-layer-v2"
    assert view.node("project-semantic-layer") is not None


def test_split_edge_not_touching_node_rejected(store):
    new_node = Node(node_id="project-x", node_type="Project", label="X")
    edge_id = next(e.edge_id for e in store.current_view().edges.values()
                   if e.source_id == "person-dev-patel")
    with pytest.raises(ValueError):
        store.split_node("project-semantic-layer", new_node, [edge_id],
                         actor="test", reason="bad split")


def test_remove_node_drops_incident_edges(store):
    before = store.current_view().stats()["edges"]
    store.remove_node("product-atlas", actor="test")
    after = store.current_view().stats()["edges"]
    # Atlas had: owned by Carla, depended on by Semantic Layer,
    # exposes Vendor Lock-in -> 3 incident edges.
    assert after == before - 3
