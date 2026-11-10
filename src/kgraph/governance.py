"""Governance: role-based access, PII redaction, and the hash-chained
audit log.

Access model: a role is a set of permissions. Permissions are checked
per node/edge *type* (not per instance) — "analysts may read Project
and Product nodes but not Risk nodes" — which keeps the policy small
enough to review in one screen. PII lives in node attributes named in
``SENSITIVE_ATTRIBUTES``; ``redact`` replaces those values with
``***`` for any role without ``pii:read``.

The audit log is append-only and hash-chained: each entry's hash
covers the previous entry's hash, so editing or deleting any entry
breaks verification of every later entry (``verify_chain``).
"""
from __future__ import annotations

import copy

from kgraph.models import AuditEntry, Node

# Attribute names treated as PII wherever they appear on a node.
SENSITIVE_ATTRIBUTES = {"email", "ssn", "phone", "salary", "home_address"}
REDACTED = "***"

# role -> permissions. "read:<Type>" grants reading nodes of that
# type; "read:edges" grants reading edges; "pii:read" grants seeing
# sensitive attributes unredacted; "write" grants mutation.
ROLE_PERMISSIONS = {
    "admin": {"read:*", "write", "pii:read"},
    "analyst": {"read:Person", "read:Organization", "read:Product",
                "read:Project", "read:edges"},
    "auditor": {"read:*", "read:edges", "pii:read"},
    "viewer": {"read:Product", "read:Project", "read:edges"},
}


def can_read(role: str, node_type: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, set())
    return "read:*" in perms or f"read:{node_type}" in perms


def can_read_edges(role: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, set())
    return "read:*" in perms or "read:edges" in perms


def can_write(role: str) -> bool:
    return "write" in ROLE_PERMISSIONS.get(role, set())


def can_read_pii(role: str) -> bool:
    return "pii:read" in ROLE_PERMISSIONS.get(role, set())


def redact(node: Node, role: str) -> Node:
    """Return a copy of ``node`` with sensitive attributes replaced,
    unless the role may read PII. The original is never mutated."""
    if can_read_pii(role):
        return copy.deepcopy(node)
    redacted = copy.deepcopy(node)
    for key in SENSITIVE_ATTRIBUTES:
        if key in redacted.attributes:
            redacted.attributes[key] = REDACTED
    return redacted


def enforce_read(role: str, node: Node) -> Node:
    """Raise PermissionError when the role may not read this node type,
    otherwise return the (possibly redacted) node."""
    if not can_read(role, node.node_type):
        raise PermissionError(
            f"role {role!r} may not read nodes of type {node.node_type!r}")
    return redact(node, role)


class AuditLog:
    """Append-only, hash-chained audit log."""

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def append(self, actor: str, action: str, detail: str) -> AuditEntry:
        prev_hash = self.entries[-1].entry_hash if self.entries else "GENESIS"
        entry = AuditEntry(seq=len(self.entries) + 1, actor=actor,
                           action=action, detail=detail, prev_hash=prev_hash)
        entry.entry_hash = entry.compute_hash()
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute every hash and every link. Returns (ok, reason)."""
        prev = "GENESIS"
        for entry in self.entries:
            if entry.prev_hash != prev:
                return False, f"entry {entry.seq}: prev_hash mismatch"
            if entry.entry_hash != entry.compute_hash():
                return False, f"entry {entry.seq}: hash mismatch"
            prev = entry.entry_hash
        return True, f"chain valid: {len(self.entries)} entries"

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]
