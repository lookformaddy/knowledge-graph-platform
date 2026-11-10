"""Core data model for the knowledge graph platform.

Pure stdlib dataclasses. A graph is a set of typed nodes and typed
edges; every mutation is recorded in a hash-chained audit log and can
be snapshotted for time-travel reads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


# Entity (node) types the platform understands. The set is fixed by
# design: a typed schema is what makes RBAC, redaction, and query
# validation possible, instead of a free-form property graph.
NODE_TYPES = ("Person", "Organization", "Product", "Project", "Risk")
EDGE_TYPES = (
    "WORKS_ON",
    "LEADS",
    "OWNS",
    "DEPENDS_ON",
    "REPORTS_TO",
    "EXPOSES",
    "MITIGATES",
    "SUPPLIES",
    "MEMBER_OF",
)


@dataclass
class Node:
    """A typed entity. ``attributes`` holds free-form properties;
    attributes named in SENSITIVE_ATTRIBUTES are PII and redacted for
    roles without ``pii:read`` (see governance.py)."""

    node_id: str
    node_type: str
    label: str
    attributes: dict = field(default_factory=dict)
    aliases: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {self.node_type!r}")
        if not self.node_id:
            raise ValueError("node_id must be non-empty")

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "attributes": dict(self.attributes),
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            label=data["label"],
            attributes=dict(data.get("attributes", {})),
            aliases=list(data.get("aliases", [])),
        )


@dataclass
class Edge:
    """A typed, directed relationship between two nodes."""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    attributes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {self.edge_type!r}")
        if not self.edge_id:
            raise ValueError("edge_id must be non-empty")

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["edge_type"],
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class AuditEntry:
    """One entry in the hash-chained audit log. ``entry_hash`` chains
    to the previous entry, so tampering with any entry breaks every
    later hash (verified by AuditLog.verify_chain)."""

    seq: int
    actor: str
    action: str
    detail: str
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps(
            {
                "seq": self.seq,
                "actor": self.actor,
                "action": self.action,
                "detail": self.detail,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "actor": self.actor,
            "action": self.action,
            "detail": self.detail,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


@dataclass
class Snapshot:
    """An immutable, versioned copy of the graph at a point in time."""

    version: int
    label: str
    nodes: dict
    edges: dict

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "label": self.label,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
        }
