# Architecture — Knowledge Graph Platform

## Diagram

```
                          ┌─────────────────────┐
                          │  Synthetic corpus   │
                          │  data.py            │
                          │  5 docs, 17 nodes,  │
                          │  20 relations,      │
                          │  ground truth +     │
                          │  5 golden questions │
                          └─────────┬───────────┘
                                    │ documents
                                    ▼
┌──────────────┐   extract()    ┌──────────────────┐
│ Extractor    │ ─────────────► │  GraphStore      │
│ (protocol)   │  Extraction-   │  store.py        │
│              │  Result        │  nodes / edges   │
│ dictionary + │                │  snapshots       │
│ pattern      │                │  merge / split   │
└──────────────┘                └──────┬───────────┘
                                       │ GraphView
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │ query.py     │   │ governance.py│   │ eval.py      │
           │ k-hop, path, │   │ RBAC, PII    │   │ P/R/F1,      │
           │ communities, │   │ redaction,   │   │ golden QA    │
           │ PageRank,    │   │ hash-chained │   │              │
           │ DSL parser   │   │ audit log    │   │              │
           └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                  │                  │                  │
                  └──────────────────┴──────────────────┘
                                     │
                              ┌──────┴───────┐
                              ▼              ▼
                        ┌──────────┐  ┌────────────┐  ┌───────────┐
                        │ CLI      │  │ FastAPI    │  │ demo.py   │
                        │ cli.py   │  │ service.py │  │ scripts/  │
                        └──────────┘  └────────────┘  └───────────┘
```

## Components

### Data model (`models.py`, `data.py`)

Five node types (`Person`, `Organization`, `Product`, `Project`,
`Risk`) and nine edge types (`WORKS_ON`, `LEADS`, `OWNS`,
`DEPENDS_ON`, `REPORTS_TO`, `EXPOSES`, `MITIGATES`, `SUPPLIES`,
`MEMBER_OF`) are fixed in code. The corpus is 5 synthetic documents
mentioning 17 entities and 20 relations, with per-document ground
truth for entities and relation triples, plus 5 golden multi-hop
questions. Everything is invented; the corpus is small on purpose so
extraction scores and QA answers are exact, reproducible numbers.

### Extractors (`extractors.py`)

`Extractor` is a protocol with one method, `extract(doc_id, text)
-> ExtractionResult`, so a new extractor plugs in without touching
the store, query engine, or eval. Two ship:

- **DictionaryExtractor** — exact, case-insensitive matching of
  canonical labels (plus project codes like `PRJ-1042`, which resolve
  to their project). Unknown strings are never invented as entities:
  a hallucinated node poisons every downstream query.
- **PatternExtractor** — dictionary entities plus relation patterns
  over sentence clauses. Sentences are split on "and" into clauses;
  the subject (and verb, for "and on B") carries across the split, so
  "works on A and on B", "owns X and owns Y", and "depends on M and
  exposes R" each yield the right relations. A relation is only
  emitted when both endpoints were extracted in the same document.

### Store (`store.py`)

`GraphStore` holds the mutable node/edge maps plus an `AuditLog`.
`add_node`/`add_edge` validate types and endpoints fail-closed.
`snapshot(label)` deep-copies the maps into a versioned `Snapshot`;
`read_at(version)` returns a read-only `GraphView` of that version,
so queries about the past can never see later mutations. All query
code works against `GraphView`, never the mutable store.

`merge_nodes(keep, drop, actor, reason)` rewires every incident edge
to the kept node, drops edges that would become duplicates or
self-loops, keeps the dropped label as an alias, and audit-logs the
reason. `split_node` moves a caller-chosen set of edges to a new
node (the caller decides the split; the store enforces that the edges
exist and touch the node being split).

### Query engine (`query.py`)

- **k-hop neighbourhood** — BFS with hop distances, deterministic
  (sorted visit order).
- **Shortest path** — BFS; ties break toward the smallest node id.
- **Communities** — label propagation on the undirected graph, ties
  broken by the smallest label, so results are deterministic.
- **Centrality** — degree centrality and PageRank (damping 0.85,
  uniform start, dangling nodes distribute uniformly).
- **Subgraph extraction** — induced subgraph on a node set.
- **Query language** — one line,
  `Person -[WORKS_ON*1..2]-> Project` (forward) or
  `Risk <-[EXPOSES*1..1]- Product` (reverse). `parse_query` validates
  node/edge types and hop ranges; `run_query` executes against a view
  and returns target nodes sorted by label.

### Governance (`governance.py`)

Roles are permission sets: `admin` (everything), `analyst` (read
all types except none — but no PII, no write), `auditor` (read
everything incl. PII, no write), `viewer` (Product/Project only).
`can_read` is checked per node type; `redact` replaces attributes
named in `SENSITIVE_ATTRIBUTES` (`email`, `ssn`, `phone`, `salary`,
`home_address`) with `***` for roles without `pii:read`, on a copy —
the stored node is never mutated. Unknown roles read nothing
(fail-closed).

`AuditLog` is append-only and hash-chained: each entry's SHA-256
covers its own fields plus the previous entry's hash. `verify_chain`
recomputes every hash and link; editing or deleting any entry fails
verification from that point on.

### Eval (`eval.py`)

- **Extraction eval** — per-document and micro-averaged
  precision/recall/F1 for entities and relation triples vs ground
  truth. Predictions are tagged with the doc id so identical labels
  in different documents do not collapse.
- **Golden QA** — each question is a traversal from its start node
  over its edge types; the answer is the exact set of labels a
  correct traversal returns. Exact-set match: a question scores 1.0
  only with no missing and no extra labels.

On the shipped corpus both evals score 1.000 — the corpus is built so
the deterministic extractor recovers ground truth exactly, which
means any future regression in either number is a real bug, not noise.

### Surfaces (`cli.py`, `service.py`, `scripts/demo.py`)

The CLI (`stats`, `query`, `path`, `eval`, `communities`,
`centrality`, `audit`) and the FastAPI service (see README for the
endpoint table) are thin wrappers over the core — no second source of
truth. The demo builds the graph, runs both evals, answers a
transitive-dependency question and a shortest path, shows communities
and PageRank, merges a duplicate entity, snapshots, time-travels back
to v1, verifies the audit chain, and writes `kg-reports/demo-report.json`.

## Design decisions and trade-offs

1. **Deterministic extractors instead of LLM extraction.**
   Reproducible, cheap, auditable — the properties an eval harness
   needs. The price is recall on free-form prose: the pattern
   extractor only fires on sentence shapes it knows. An LLM extractor
   can be added later behind the same `Extractor` protocol, with its
   output scored by the same eval.
2. **Typed schema instead of a free-form property graph.** A fixed
   type set is what makes RBAC, redaction, and query validation
   possible. The price is that a new entity kind needs a schema
   change — deliberate, since a new kind is also a new governance
   decision.
3. **Full-copy snapshots.** Time-travel reads are trivially correct.
   The price is memory proportional to (versions × graph size). At
   enterprise scale this becomes a delta log or copy-on-write store
   behind the same `snapshot`/`read_at` interface; the query code
   would not change, because it only ever sees `GraphView`.
4. **Hash-chained audit log.** Tamper-evident without a database or
   signature service. The price is O(n) verification — fine for an
   audit trail, wrong for a hot path.
5. **Stdlib-only core.** The platform runs inside the CI job that
   tests it, with no dependency installation. FastAPI is an optional
   surface; its tests auto-skip without it.
6. **Synthetic data only.** All people, products, projects, and risks
   are invented. No real data, no employer data, no network calls
   anywhere in the pipeline.

## Failure modes this design anticipates

- **An extractor hallucinates a node** — dictionary matching only
  ever returns canonical labels; unknown strings produce nothing.
- **A merge silently loses edges** — duplicates and self-loops are
  counted and reported in the `MergeResult`, and the merge is
  audit-logged with its reason.
- **A query sees a half-applied mutation** — queries run against
  `GraphView`, a stable snapshot of the maps.
- **A role sees PII it should not** — redaction is applied on every
  read path (CLI and API) from the same `enforce_read`/`redact`
  functions; unknown roles read nothing.
- **An audit entry is edited or deleted** — the hash chain breaks at
  the tampered entry and `verify_chain` reports it.
- **A snapshot silently changes** — snapshots are deep copies taken
  at creation time; later mutations cannot reach them.
- **An extraction regression ships unnoticed** — the corpus is built
  for exact recovery, so any drop below 1.000 P/R/F1 is a failing
  test, not a judgement call.
