"""Governance tests: RBAC, PII redaction, hash-chained audit log."""
import pytest

from kgraph.governance import (AuditLog, can_read, can_write, enforce_read,
                               redact)
from kgraph.models import Node


def _person():
    return Node(node_id="person-alice-chen", node_type="Person",
                label="Alice Chen",
                attributes={"email": "alice.chen@example-corp.com",
                            "role": "Staff Engineer"})


def test_admin_reads_everything():
    assert can_read("admin", "Risk") is True
    assert can_write("admin") is True


def test_viewer_cannot_read_risk_or_person():
    assert can_read("viewer", "Risk") is False
    assert can_read("viewer", "Person") is False
    assert can_read("viewer", "Project") is True


def test_analyst_cannot_write():
    assert can_write("analyst") is False


def test_unknown_role_reads_nothing():
    assert can_read("nobody", "Project") is False


def test_redaction_hides_email_for_analyst():
    node = redact(_person(), "analyst")
    assert node.attributes["email"] == "***"
    assert node.attributes["role"] == "Staff Engineer"
    # Original untouched.
    assert _person().attributes["email"] == "alice.chen@example-corp.com"


def test_redaction_keeps_email_for_admin_and_auditor():
    for role in ("admin", "auditor"):
        node = redact(_person(), role)
        assert node.attributes["email"] == "alice.chen@example-corp.com"


def test_enforce_read_raises_for_forbidden_type():
    risk = Node(node_id="risk-x", node_type="Risk", label="X")
    with pytest.raises(PermissionError):
        enforce_read("viewer", risk)


def test_enforce_read_redacts_for_allowed_type():
    node = enforce_read("analyst", _person())
    assert node.attributes["email"] == "***"


def test_audit_chain_valid():
    log = AuditLog()
    log.append("alice", "node.add", "Person:person-alice-chen")
    log.append("bob", "edge.add", "a -[WORKS_ON]-> b")
    ok, reason = log.verify_chain()
    assert ok is True
    assert "2 entries" in reason


def test_audit_chain_detects_tampering():
    log = AuditLog()
    log.append("alice", "node.add", "Person:person-alice-chen")
    log.append("bob", "edge.add", "a -[WORKS_ON]-> b")
    log.entries[0].detail = "Person:person-mallory"
    ok, reason = log.verify_chain()
    assert ok is False
    assert "hash mismatch" in reason


def test_audit_chain_detects_deletion():
    log = AuditLog()
    log.append("alice", "node.add", "x")
    log.append("bob", "edge.add", "y")
    log.append("carol", "snapshot.create", "z")
    del log.entries[1]
    ok, _reason = log.verify_chain()
    assert ok is False


def test_audit_empty_chain_valid():
    ok, _reason = AuditLog().verify_chain()
    assert ok is True
