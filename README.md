# Knowledge Graph Platform

Enterprise knowledge graph: deterministic entity/relation extraction
from documents, a typed graph store with versioned snapshots and
time-travel reads, entity resolution (merge/split) with a hash-chained
audit trail, multi-hop query with a small declarative language,
role-based governance with PII redaction, and a golden-set eval
harness — with a CLI, a FastAPI service, and a demo that runs end to
end with no network and no API keys.

## The enterprise problem

Every enterprise has the same knowledge problem: who works on what,
what depends on what, and which risks attach to which products — and
that knowledge lives in scattered documents, not in a queryable
system. A knowledge graph turns documents into a typed, governed,
auditable graph you can ask multi-hop questions of: *"Which risks
does this person's work expose, transitively?"* This platform answers
that question with deterministic extraction (no LLM calls, so the
graph is reproducible), versioned snapshots (so you can ask what the
graph looked like last quarter), and governance (so a viewer never
sees PII and every mutation is audit-logged).

## What it does

| Capability | How |
|---|---|
| **Extraction** | Dictionary + pattern extractors over documents; pluggable `Extractor` protocol (swap in an ML/LLM extractor without touching the store) |
| **Graph store** | Typed nodes (Person, Organization, Product, Project, Risk) and typed edges; full-copy versioned snapshots with time-travel reads |
| **Entity resolution** | Merge (rewire edges, keep alias, drop duplicates) and split (move chosen edges to a new node), both audit-logged |
| **Query** | k-hop neighbourhood, shortest path, label-propagation communities, degree centrality, PageRank, induced subgraphs, plus a one-line query language: `Person -[WORKS_ON*1..2]-> Project` |
| **Governance** | Role-based read access per node type, PII redaction on sensitive attributes, hash-chained audit log with chain verification |
| **Eval** | Extraction precision/recall/F1 vs synthetic ground truth; golden-set multi-hop QA (exact-set match) |

## Architecture overview

```
synthetic docs (data.py) ──► extractors ──► GraphStore
        (pattern/dictionary)      │  nodes / edges
                                  │  snapshots (time travel)
                                  │  merge / split
                                  ▼
                          query engine ──► traversal, shortest path,
                          communities, PageRank, subgraph, DSL
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        governance        eval harness          audit log
        RBAC + PII        P/R/F1 + golden QA    hash-chained
              │                   │                   │
              └──────────► CLI · FastAPI · demo.py ◄──┘
```

- **Core is pure stdlib.** Extraction, store, query, governance, and
  eval have no dependencies, so the platform runs anywhere — including
  inside the CI job that tests it.
- **Extraction is deterministic.** The same documents always produce
  the same graph, which is what makes precision/recall a stable
  regression metric instead of a vibe.
- **Snapshots are full copies.** Simple and trivially correct for
  time-travel reads; the trade-off (memory at scale) is documented in
  `docs/architecture.md`, where the delta-log alternative is sketched.
- **The audit log is hash-chained.** Each entry's hash covers the
  previous entry, so editing or deleting any entry breaks verification
  of every later entry.
- **Governance is fail-closed.** An unknown role reads nothing; a
  role that may not read a node type gets a 403, never a partial view.

See [docs/architecture.md](docs/architecture.md) for the full design,
including the diagram and the trade-offs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Graph statistics (17 nodes, 20 edges from the synthetic corpus)
PYTHONPATH=src python3 -m kgraph.cli stats

# Multi-hop query: which projects does Alice Chen work on?
PYTHONPATH=src python3 -m kgraph.cli query \
  "Person -[WORKS_ON*1..1]-> Project" --start "Alice Chen"

# Shortest path from a person to a risk
PYTHONPATH=src python3 -m kgraph.cli path "Carla Ruiz" \
  "PII Exposure in Search Logs"

# Extraction + golden QA eval (both should be 1.000 on this corpus)
PYTHONPATH=src python3 -m kgraph.cli eval

# Communities, centrality, audit log
PYTHONPATH=src python3 -m kgraph.cli communities
PYTHONPATH=src python3 -m kgraph.cli centrality
PYTHONPATH=src python3 -m kgraph.cli audit

# Full end-to-end demo (report in ./kg-reports/)
make demo

# Tests (plain pytest passes from the project root, with or without FastAPI)
make test
```

## API service

```bash
uvicorn kgraph.service:app --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + node/edge counts |
| GET | `/stats` | Nodes/edges by type |
| GET | `/nodes?role=` | List nodes (RBAC + PII redaction applied) |
| GET | `/nodes/{id}?role=` | One node (403 when the role may not read its type) |
| POST | `/query` | Run a query-language line, optional `start_label` |
| GET | `/nodes/{id}/neighbors?max_hops=` | k-hop neighbourhood with hop distances |
| GET | `/shortest-path?source=&target=` | Shortest path between two node labels |
| GET | `/communities` | Label-propagation communities |
| GET | `/centrality` | PageRank over all nodes |
| POST | `/nodes/merge` | Merge a duplicate node into a kept node |
| GET | `/snapshots` | Snapshot versions |
| GET | `/snapshots/{version}/stats` | Time-travel read of a snapshot |
| GET | `/eval` | Extraction + golden QA eval |
| GET | `/audit` | Audit log + chain verification |

## Key design decisions and trade-offs

1. **Deterministic extractors, no LLM calls.** Reproducible, cheap,
   auditable — the properties an eval harness needs. The price is
   recall on free-form prose: the pattern extractor only fires on
   sentence shapes it knows. The mitigation is the pluggable
   `Extractor` protocol — an LLM extractor can be added behind the
   same interface, with its output scored by the same eval.
2. **Typed schema, not a free-form property graph.** A fixed set of
   node and edge types is what makes RBAC, redaction, and query
   validation possible. The price is that new entity kinds need a
   schema change — deliberate, since a new kind is also a new
   governance decision.
3. **Full-copy snapshots.** Time-travel reads are trivially correct.
   The price is memory proportional to (versions × graph size); at
   enterprise scale this becomes a delta log or copy-on-write store
   behind the same `snapshot`/`read_at` interface.
4. **Hash-chained audit log.** Tamper-evident without a database or a
   signature service. The price is that verification is O(n) over the
   log — fine for an audit trail, wrong for a hot path.
5. **Synthetic data only.** All people, products, projects, and risks
   are invented. No real data, no employer data, no network.
6. **FastAPI is optional.** The core runs on stdlib alone; the service
   and its tests auto-skip when FastAPI is absent, so `make test` works
   on a bare Python install and the full suite runs in CI.

## What this demonstrates

Deterministic NLP-free extraction (dictionary + patterns) · typed
graph modelling · versioned snapshots and time-travel reads ·
entity resolution with audit trails · multi-hop query, shortest path,
community detection, centrality · a declarative graph query language ·
RBAC and PII redaction · hash-chained tamper-evident audit logging ·
golden-set QA and extraction P/R/F1 evaluation · CLI + FastAPI service
· CI-ready pytest suite.

## Project layout

```
src/kgraph/
  models.py      dataclasses: Node, Edge, Snapshot, AuditEntry
  data.py        synthetic corpus, ground truth, golden QA questions
  extractors.py  Extractor protocol, dictionary + pattern extractors
  store.py       GraphStore: snapshots, merge/split, GraphView
  query.py       traversal, shortest path, communities, PageRank,
                 subgraph, query-language parser
  governance.py  roles, RBAC, PII redaction, hash-chained AuditLog
  eval.py        extraction P/R/F1 + golden QA harness
  cli.py         `kgraph` CLI
  service.py     FastAPI service
tests/           pytest suite (API tests auto-skip without FastAPI)
scripts/demo.py  end-to-end demo over the synthetic corpus
docs/            architecture + CI definition (see below)
```

## CI

`docs/github-actions-ci.yml` is the CI definition. It ships under
`docs/` (not `.github/workflows/`) because the GitHub App used to push
this repo lacks the Workflows permission; activate it by moving it to
`.github/workflows/ci.yml`.
