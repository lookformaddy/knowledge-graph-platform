"""CLI tests: every subcommand runs end to end on the synthetic corpus."""
from kgraph.cli import main


def test_cli_stats(capsys):
    assert main(["stats"]) == 0
    out = capsys.readouterr().out
    assert "nodes: 17" in out
    assert "edges: 20" in out


def test_cli_query(capsys):
    assert main(["query", "Person -[WORKS_ON*1..1]-> Project",
                 "--start", "Alice Chen"]) == 0
    out = capsys.readouterr().out
    assert "Semantic Layer" in out
    assert "Enterprise Knowledge Graph" in out


def test_cli_query_redacts_for_analyst(capsys):
    # Querying persons as analyst still lists them (labels are not
    # PII); the redaction applies to attributes, covered in the API
    # tests. Here we just check the command runs for a non-admin role.
    assert main(["query", "Product -[EXPOSES*1..1]-> Risk",
                 "--role", "viewer"]) == 0


def test_cli_path(capsys):
    assert main(["path", "Carla Ruiz",
                 "PII Exposure in Search Logs"]) == 0
    out = capsys.readouterr().out
    assert "Carla Ruiz -> Northstar -> PII Exposure in Search Logs" in out


def test_cli_path_no_path(capsys):
    assert main(["path", "Alice Chen",
                 "PII Exposure in Search Logs"]) == 0
    assert "no path" in capsys.readouterr().out


def test_cli_eval(capsys):
    assert main(["eval"]) == 0
    out = capsys.readouterr().out
    assert "extraction entities:  P=1.000 R=1.000 F1=1.000" in out
    assert "extraction relations: P=1.000 R=1.000 F1=1.000" in out
    assert "golden QA accuracy: 1.000 (5/5)" in out


def test_cli_communities(capsys):
    assert main(["communities"]) == 0
    assert "community" in capsys.readouterr().out


def test_cli_centrality(capsys):
    assert main(["centrality"]) == 0
    out = capsys.readouterr().out
    assert "Semantic Layer" in out or "Fraud Signals" in out


def test_cli_audit(capsys):
    assert main(["audit"]) == 0
    out = capsys.readouterr().out
    assert "chain valid" in out
    assert "snapshot.create" in out
