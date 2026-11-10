"""Synthetic enterprise corpus and ground truth.

Every person, product, project, and risk here is invented. The corpus
is small on purpose: it is the deterministic ground truth the
extractors and the golden QA set are scored against, so extraction
precision/recall and multi-hop QA answers are reproducible numbers,
not vibes. No real data, no employer data, no network.
"""
from __future__ import annotations

# Canonical entity registry: (node_id, node_type, label, attributes).
ENTITIES = [
    ("person-alice-chen", "Person", "Alice Chen",
     {"role": "Staff Engineer", "email": "alice.chen@example-corp.com",
      "department": "Platform Engineering"}),
    ("person-ben-okafor", "Person", "Ben Okafor",
     {"role": "Engineering Manager", "email": "ben.okafor@example-corp.com",
      "department": "Platform Engineering"}),
    ("person-carla-ruiz", "Person", "Carla Ruiz",
     {"role": "Product Manager", "email": "carla.ruiz@example-corp.com",
      "department": "Product"}),
    ("person-dev-patel", "Person", "Dev Patel",
     {"role": "Data Scientist", "email": "dev.patel@example-corp.com",
      "department": "Data Science"}),
    ("person-erin-novak", "Person", "Erin Novak",
     {"role": "Risk Analyst", "email": "erin.novak@example-corp.com",
      "department": "Risk & Compliance"}),
    ("org-platform-eng", "Organization", "Platform Engineering",
     {"region": "North America", "headcount_band": "50-100"}),
    ("org-data-science", "Organization", "Data Science",
     {"region": "North America", "headcount_band": "20-50"}),
    ("org-risk-compliance", "Organization", "Risk & Compliance",
     {"region": "Europe", "headcount_band": "10-20"}),
    ("product-northstar", "Product", "Northstar",
     {"category": "Search", "tier": "Tier 1"}),
    ("product-atlas", "Product", "Atlas",
     {"category": "Analytics", "tier": "Tier 2"}),
    ("product-meridian", "Product", "Meridian",
     {"category": "Billing", "tier": "Tier 1"}),
    ("project-semantic-layer", "Project", "Semantic Layer",
     {"code": "PRJ-1042", "status": "active", "budget_band": "$1M-$5M"}),
    ("project-knowledge-graph", "Project", "Enterprise Knowledge Graph",
     {"code": "PRJ-1087", "status": "active", "budget_band": "$1M-$5M"}),
    ("project-fraud-signals", "Project", "Fraud Signals",
     {"code": "PRJ-1119", "status": "active", "budget_band": "$500K-$1M"}),
    ("risk-pii-exposure", "Risk", "PII Exposure in Search Logs",
     {"severity": "high", "status": "open"}),
    ("risk-model-drift", "Risk", "Model Drift in Fraud Scoring",
     {"severity": "medium", "status": "open"}),
    ("risk-vendor-lockin", "Risk", "Vendor Lock-in on Analytics Store",
     {"severity": "low", "status": "mitigated"}),
]

# Canonical relations: (source_id, edge_type, target_id).
RELATIONS = [
    ("person-alice-chen", "WORKS_ON", "project-semantic-layer"),
    ("person-alice-chen", "WORKS_ON", "project-knowledge-graph"),
    ("person-ben-okafor", "LEADS", "project-semantic-layer"),
    ("person-ben-okafor", "REPORTS_TO", "person-carla-ruiz"),
    ("person-carla-ruiz", "OWNS", "product-northstar"),
    ("person-carla-ruiz", "OWNS", "product-atlas"),
    ("person-dev-patel", "WORKS_ON", "project-fraud-signals"),
    ("person-dev-patel", "WORKS_ON", "project-knowledge-graph"),
    ("person-erin-novak", "WORKS_ON", "project-fraud-signals"),
    ("person-erin-novak", "MITIGATES", "risk-pii-exposure"),
    ("person-alice-chen", "MEMBER_OF", "org-platform-eng"),
    ("person-ben-okafor", "MEMBER_OF", "org-platform-eng"),
    ("person-dev-patel", "MEMBER_OF", "org-data-science"),
    ("person-erin-novak", "MEMBER_OF", "org-risk-compliance"),
    ("project-semantic-layer", "DEPENDS_ON", "product-atlas"),
    ("project-knowledge-graph", "DEPENDS_ON", "project-semantic-layer"),
    ("project-fraud-signals", "DEPENDS_ON", "product-meridian"),
    ("project-fraud-signals", "EXPOSES", "risk-model-drift"),
    ("product-northstar", "EXPOSES", "risk-pii-exposure"),
    ("product-atlas", "EXPOSES", "risk-vendor-lockin"),
]

# Documents the extractors read. Each document mentions entities and
# relations in plain sentences; the ground truth below is what a
# correct extraction must recover from them.
DOCUMENTS = [
    {
        "doc_id": "doc-001",
        "title": "Semantic Layer staffing note",
        "text": (
            "Alice Chen works on Semantic Layer (PRJ-1042) and on "
            "Enterprise Knowledge Graph (PRJ-1087). Ben Okafor leads "
            "Semantic Layer. Alice Chen is a member of Platform "
            "Engineering."
        ),
    },
    {
        "doc_id": "doc-002",
        "title": "Product ownership update",
        "text": (
            "Carla Ruiz owns Northstar and owns Atlas. Ben Okafor "
            "reports to Carla Ruiz. Ben Okafor is a member of Platform "
            "Engineering."
        ),
    },
    {
        "doc_id": "doc-003",
        "title": "Fraud Signals project brief",
        "text": (
            "Dev Patel works on Fraud Signals (PRJ-1119) and on "
            "Enterprise Knowledge Graph (PRJ-1087). Erin Novak works on "
            "Fraud Signals (PRJ-1119). Fraud Signals depends on Meridian "
            "and exposes Model Drift in Fraud Scoring. Dev Patel is a "
            "member of Data Science. Erin Novak is a member of Risk & "
            "Compliance."
        ),
    },
    {
        "doc_id": "doc-004",
        "title": "Dependency and risk register",
        "text": (
            "Semantic Layer depends on Atlas. Enterprise Knowledge Graph "
            "depends on Semantic Layer. Northstar exposes PII Exposure "
            "in Search Logs. Atlas exposes Vendor Lock-in on Analytics "
            "Store. Erin Novak mitigates PII Exposure in Search Logs."
        ),
    },
    {
        "doc_id": "doc-005",
        "title": "Distractor memo (no new entities)",
        "text": (
            "The quarterly review covered roadmap priorities and hiring "
            "plans. No decisions were recorded in this memo."
        ),
    },
]

# Ground truth: what a perfect extraction recovers, per document.
# Entity ground truth is the set of entity labels mentioned.
GROUND_TRUTH_ENTITIES = {
    "doc-001": {"Alice Chen", "Semantic Layer", "Enterprise Knowledge Graph",
                "Ben Okafor", "Platform Engineering"},
    "doc-002": {"Carla Ruiz", "Northstar", "Atlas", "Ben Okafor",
                "Platform Engineering"},
    "doc-003": {"Dev Patel", "Fraud Signals", "Enterprise Knowledge Graph",
                "Erin Novak", "Meridian", "Model Drift in Fraud Scoring",
                "Data Science", "Risk & Compliance"},
    "doc-004": {"Semantic Layer", "Atlas", "Enterprise Knowledge Graph",
                "Northstar", "PII Exposure in Search Logs",
                "Vendor Lock-in on Analytics Store", "Erin Novak"},
    "doc-005": set(),
}

# Relation ground truth: (source label, edge type, target label).
GROUND_TRUTH_RELATIONS = {
    "doc-001": {
        ("Alice Chen", "WORKS_ON", "Semantic Layer"),
        ("Alice Chen", "WORKS_ON", "Enterprise Knowledge Graph"),
        ("Ben Okafor", "LEADS", "Semantic Layer"),
        ("Alice Chen", "MEMBER_OF", "Platform Engineering"),
    },
    "doc-002": {
        ("Carla Ruiz", "OWNS", "Northstar"),
        ("Carla Ruiz", "OWNS", "Atlas"),
        ("Ben Okafor", "REPORTS_TO", "Carla Ruiz"),
        ("Ben Okafor", "MEMBER_OF", "Platform Engineering"),
    },
    "doc-003": {
        ("Dev Patel", "WORKS_ON", "Fraud Signals"),
        ("Dev Patel", "WORKS_ON", "Enterprise Knowledge Graph"),
        ("Erin Novak", "WORKS_ON", "Fraud Signals"),
        ("Fraud Signals", "DEPENDS_ON", "Meridian"),
        ("Fraud Signals", "EXPOSES", "Model Drift in Fraud Scoring"),
        ("Dev Patel", "MEMBER_OF", "Data Science"),
        ("Erin Novak", "MEMBER_OF", "Risk & Compliance"),
    },
    "doc-004": {
        ("Semantic Layer", "DEPENDS_ON", "Atlas"),
        ("Enterprise Knowledge Graph", "DEPENDS_ON", "Semantic Layer"),
        ("Northstar", "EXPOSES", "PII Exposure in Search Logs"),
        ("Atlas", "EXPOSES", "Vendor Lock-in on Analytics Store"),
        ("Erin Novak", "MITIGATES", "PII Exposure in Search Logs"),
    },
    "doc-005": set(),
}

# Golden QA set: multi-hop questions answered by graph traversal.
# ``answer`` is the exact set of node labels a correct traversal
# returns; ``hops`` is the traversal depth the question needs.
GOLDEN_QUESTIONS = [
    {
        "question_id": "q1",
        "question": "Which projects does Alice Chen work on?",
        "start_label": "Alice Chen",
        "edge_types": ["WORKS_ON"],
        "max_hops": 1,
        "answer": {"Semantic Layer", "Enterprise Knowledge Graph"},
    },
    {
        "question_id": "q2",
        "question": "Which products does Carla Ruiz own, and which risks do they expose?",
        "start_label": "Carla Ruiz",
        "edge_types": ["OWNS", "EXPOSES"],
        "max_hops": 2,
        "answer": {"Northstar", "Atlas", "PII Exposure in Search Logs",
                   "Vendor Lock-in on Analytics Store"},
    },
    {
        "question_id": "q3",
        "question": "What does Enterprise Knowledge Graph transitively depend on?",
        "start_label": "Enterprise Knowledge Graph",
        "edge_types": ["DEPENDS_ON"],
        "max_hops": 2,
        "answer": {"Semantic Layer", "Atlas"},
    },
    {
        "question_id": "q4",
        "question": "Who works on a project that exposes Model Drift in Fraud Scoring?",
        "start_label": "Model Drift in Fraud Scoring",
        "edge_types": ["EXPOSES", "WORKS_ON"],
        "max_hops": 2,
        "reverse": True,
        "to_type": "Person",
        "answer": {"Dev Patel", "Erin Novak"},
    },
    {
        "question_id": "q5",
        "question": "Which risks is Erin Novak connected to (works on or mitigates)?",
        "start_label": "Erin Novak",
        "edge_types": ["WORKS_ON", "EXPOSES", "MITIGATES"],
        "max_hops": 2,
        "answer": {"Fraud Signals", "Model Drift in Fraud Scoring",
                   "PII Exposure in Search Logs"},
    },
]


def build_canonical_graph_data():
    """Return (nodes, edges) for the full canonical graph."""
    from kgraph.models import Edge, Node

    nodes = {n[0]: Node(node_id=n[0], node_type=n[1], label=n[2],
                        attributes=dict(n[3])) for n in ENTITIES}
    edges = {}
    for i, (src, etype, tgt) in enumerate(RELATIONS):
        edge_id = f"edge-{i + 1:03d}"
        edges[edge_id] = Edge(edge_id=edge_id, source_id=src,
                              target_id=tgt, edge_type=etype)
    return nodes, edges
