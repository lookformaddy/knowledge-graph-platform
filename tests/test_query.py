"""Query engine tests: DSL parsing, traversal, paths, communities,
centrality, subgraphs."""
import pytest

from kgraph.cli import build_store
from kgraph.query import (GraphQuery, extract_subgraph,
                          k_hop_neighborhood, label_propagation_communities,
                          pagerank, parse_query, run_query, shortest_path,
                          degree_centrality)


@pytest.fixture
def view():
    return build_store().current_view()


def test_parse_query_forward():
    q = parse_query("Person -[WORKS_ON*1..2]-> Project")
    assert q.from_type == "Person"
    assert q.edge_types == ["WORKS_ON"]
    assert (q.min_hops, q.max_hops) == (1, 2)
    assert q.to_type == "Project"
    assert q.reverse is False


def test_parse_query_reverse():
    q = parse_query("Risk <-[EXPOSES*1..1]- Product")
    assert q.reverse is True
    assert q.from_type == "Risk"
    assert q.to_type == "Product"


def test_parse_query_defaults_to_single_hop():
    q = parse_query("Person -[LEADS]-> Project")
    assert (q.min_hops, q.max_hops) == (1, 1)


def test_parse_query_rejects_garbage():
    with pytest.raises(ValueError):
        parse_query("Person works on Project")
    with pytest.raises(ValueError):
        parse_query("Alien -[WORKS_ON]-> Project")
    with pytest.raises(ValueError):
        parse_query("Person -[HUGS]-> Project")


def test_run_query_works_on(view):
    q = GraphQuery(from_type="Person", edge_types=["WORKS_ON"],
                   to_type="Project", start_label="Alice Chen")
    assert {n.label for n in run_query(view, q)} == {
        "Semantic Layer", "Enterprise Knowledge Graph"}


def test_run_query_two_hop_depends_on(view):
    q = GraphQuery(from_type="Project", edge_types=["DEPENDS_ON"],
                   min_hops=1, max_hops=2, to_type="Product",
                   start_label="Enterprise Knowledge Graph")
    assert {n.label for n in run_query(view, q)} == {"Atlas"}


def test_run_query_dsl_string(view):
    q = parse_query("Person -[WORKS_ON*1..1]-> Project")
    q.start_label = "Dev Patel"
    assert {n.label for n in run_query(view, q)} == {
        "Fraud Signals", "Enterprise Knowledge Graph"}


def test_k_hop_neighborhood_distances(view):
    node = view.find_by_label("Enterprise Knowledge Graph")
    dists = k_hop_neighborhood(view, node.node_id, ["DEPENDS_ON"],
                               max_hops=2)
    labels = {view.nodes[nid].label: d for nid, d in dists.items()}
    assert labels["Semantic Layer"] == 1
    assert labels["Atlas"] == 2
    assert labels["Enterprise Knowledge Graph"] == 0


def test_k_hop_unknown_node_raises(view):
    with pytest.raises(KeyError):
        k_hop_neighborhood(view, "no-such-node", max_hops=1)


def test_shortest_path(view):
    src = view.find_by_label("Carla Ruiz")
    tgt = view.find_by_label("PII Exposure in Search Logs")
    path = shortest_path(view, src.node_id, tgt.node_id)
    assert [view.nodes[nid].label for nid in path] == [
        "Carla Ruiz", "Northstar", "PII Exposure in Search Logs"]


def test_shortest_path_none_when_disconnected(view):
    src = view.find_by_label("Alice Chen")
    tgt = view.find_by_label("PII Exposure in Search Logs")
    assert shortest_path(view, src.node_id, tgt.node_id) is None


def test_shortest_path_same_node(view):
    node = view.find_by_label("Atlas")
    assert shortest_path(view, node.node_id, node.node_id) == [node.node_id]


def test_communities_partition_all_nodes(view):
    communities = label_propagation_communities(view)
    members = [nid for group in communities.values() for nid in group]
    assert sorted(members) == sorted(view.nodes)
    assert all(len(group) > 0 for group in communities.values())


def test_degree_centrality_range(view):
    scores = degree_centrality(view)
    assert set(scores) == set(view.nodes)
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    # Semantic Layer is the most connected project in the corpus.
    top = max(scores, key=scores.get)
    assert view.nodes[top].label in {"Semantic Layer", "Fraud Signals",
                                     "Enterprise Knowledge Graph"}


def test_pagerank_sums_to_one(view):
    ranks = pagerank(view)
    assert abs(sum(ranks.values()) - 1.0) < 1e-6
    assert all(score > 0 for score in ranks.values())


def test_extract_subgraph(view):
    wanted = {"person-alice-chen", "project-semantic-layer",
              "project-knowledge-graph"}
    sub = extract_subgraph(view, wanted)
    assert set(sub.nodes) == wanted
    # Alice -> Semantic Layer, Alice -> Knowledge Graph, and
    # Knowledge Graph -> Semantic Layer all survive.
    assert len(sub.edges) == 3
    assert all(e.source_id in wanted and e.target_id in wanted
               for e in sub.edges.values())
