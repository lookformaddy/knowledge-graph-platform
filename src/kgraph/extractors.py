"""Deterministic entity/relation extraction from documents.

No external LLM calls, no network. Extractors implement the
``Extractor`` protocol, so a dictionary/pattern extractor can be
swapped for an ML or LLM-backed one without touching the store,
query engine, or eval harness. What ships here:

- ``DictionaryExtractor`` — finds canonical entity labels (and their
  project codes) by exact, case-insensitive matching.
- ``PatternExtractor`` — relation extraction from a small set of
  explicit sentence patterns ("X works on Y", "X leads Y", ...).

Determinism is the point: the same documents always produce the same
graph, which is what makes extraction precision/recall a stable
regression metric (see eval.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from kgraph.data import ENTITIES, GROUND_TRUTH_ENTITIES, GROUND_TRUTH_RELATIONS
from kgraph.models import Edge, Node

# label -> (node_id, node_type); built from the canonical registry.
LABEL_INDEX = {label: (node_id, node_type)
               for node_id, node_type, label, _attrs in ENTITIES}
CODE_INDEX = {attrs["code"]: label
              for _id, _t, label, attrs in ENTITIES if "code" in attrs}

# Relation sentence patterns: (regex, edge_type, verb phrase). Groups are
# (subject label, object label). A sentence may join two clauses with
# "and" — "owns X and owns Y", "works on A and on B", "depends on M
# and exposes R" — so extraction splits sentences into clauses and
# carries the subject (and verb, for "and on B") across the split.
_RELATION_PATTERNS = [
    (re.compile(r"(.+?)\s+works on\s+(.+?)(?:\.|$)"), "WORKS_ON", "works on"),
    (re.compile(r"(.+?)\s+leads\s+(.+?)(?:\.|$)"), "LEADS", "leads"),
    (re.compile(r"(.+?)\s+owns\s+(.+?)(?:\.|$)"), "OWNS", "owns"),
    (re.compile(r"(.+?)\s+reports to\s+(.+?)(?:\.|$)"), "REPORTS_TO", "reports to"),
    (re.compile(r"(.+?)\s+depends on\s+(.+?)(?:\.|$)"), "DEPENDS_ON", "depends on"),
    (re.compile(r"(.+?)\s+exposes\s+(.+?)(?:\.|$)"), "EXPOSES", "exposes"),
    (re.compile(r"(.+?)\s+mitigates\s+(.+?)(?:\.|$)"), "MITIGATES", "mitigates"),
    (re.compile(r"(.+?)\s+is a member of\s+(.+?)(?:\.|$)"), "MEMBER_OF", "is a member of"),
]


def _clean_label(text: str) -> str:
    """Strip a trailing project code in parens and whitespace."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", text.strip()).strip()


@dataclass
class ExtractionResult:
    """Everything one extractor run recovered from one document."""

    doc_id: str
    entities: list = field(default_factory=list)  # list[Node]
    relations: list = field(default_factory=list)  # list[Edge] (ids are local)

    def entity_labels(self) -> set:
        return {n.label for n in self.entities}

    def relation_triples(self) -> set:
        return {(self._label(e.source_id), e.edge_type,
                 self._label(e.target_id)) for e in self.relations}

    def _label(self, node_id: str) -> str:
        for n in self.entities:
            if n.node_id == node_id:
                return n.label
        return node_id


class Extractor(Protocol):
    """Pluggable extractor interface. Implementations must be
    deterministic: same input documents -> same ExtractionResult."""

    def extract(self, doc_id: str, text: str) -> ExtractionResult: ...


class DictionaryExtractor:
    """Finds canonical entity labels by exact match, plus project
    codes (PRJ-1042) which resolve to their project label. Unknown
    strings are never invented as entities — an extractor that
    hallucinates nodes is worse than one that misses them, because a
    hallucinated node poisons every downstream query."""

    def __init__(self, entities=None):
        self.entities = entities if entities is not None else ENTITIES

    def extract(self, doc_id: str, text: str) -> ExtractionResult:
        found: dict[str, Node] = {}
        lowered = text.lower()
        for node_id, node_type, label, attrs in self.entities:
            if label.lower() in lowered:
                found[node_id] = Node(node_id=node_id, node_type=node_type,
                                      label=label, attributes=dict(attrs))
        # Project codes resolve to the project they name.
        for code, label in CODE_INDEX.items():
            if code.lower() in lowered:
                node_id, node_type = LABEL_INDEX[label]
                attrs = dict(next(a for _i, _t, l, a in self.entities
                                  if l == label))
                found[node_id] = Node(node_id=node_id, node_type=node_type,
                                      label=label, attributes=attrs)
        return ExtractionResult(doc_id=doc_id,
                                entities=sorted(found.values(),
                                                key=lambda n: n.label))


class PatternExtractor:
    """Dictionary entities + relation patterns. Runs the dictionary
    pass first, then matches relation sentences against the entities
    actually found in the document, so a relation is only emitted
    when both endpoints were extracted."""

    def __init__(self, entities=None):
        self._dict = DictionaryExtractor(entities)

    def extract(self, doc_id: str, text: str) -> ExtractionResult:
        result = self._dict.extract(doc_id, text)
        label_to_id = {n.label: n.node_id for n in result.entities}
        relations: list[Edge] = []
        seen: set[tuple[str, str, str]] = set()

        def emit(subj: str, edge_type: str, obj: str) -> None:
            if subj in label_to_id and obj in label_to_id:
                key = (label_to_id[subj], edge_type, label_to_id[obj])
                if key not in seen:
                    seen.add(key)
                    relations.append(Edge(
                        edge_id=f"{doc_id}-rel-{len(relations) + 1:02d}",
                        source_id=key[0], target_id=key[2],
                        edge_type=edge_type,
                        attributes={"source_doc": doc_id}))

        def match_clause(clause: str):
            for pattern, edge_type, verb in _RELATION_PATTERNS:
                match = pattern.search(clause)
                if match:
                    return (_clean_label(match.group(1)), edge_type,
                            _clean_label(match.group(2)), verb)
            return None

        for sentence in re.split(r"\.\s*", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            pending_subj: str | None = None
            pending_verb: str | None = None
            for clause in re.split(r"\s+and\s+", sentence):
                clause = clause.strip()
                if not clause:
                    continue
                found = match_clause(clause)
                if found is None and pending_subj is not None:
                    # Subject-less continuation: "and on B" reuses the
                    # previous verb; "and exposes R" brings its own.
                    trial = (f"{pending_subj} {pending_verb} "
                             f"{clause[3:].strip()}"
                             if clause.startswith("on ")
                             else f"{pending_subj} {clause}")
                    found = match_clause(trial)
                if found is not None:
                    subj, edge_type, obj, verb = found
                    emit(subj, edge_type, obj)
                    pending_subj, pending_verb = subj, verb
        result.relations = relations
        return result


def extract_corpus(documents, extractor=None) -> list[ExtractionResult]:
    """Run an extractor over a list of {doc_id, text} documents."""
    extractor = extractor or PatternExtractor()
    return [extractor.extract(d["doc_id"], d["text"]) for d in documents]


def ground_truth(doc_id: str):
    """Ground-truth entity labels and relation triples for a document."""
    return (set(GROUND_TRUTH_ENTITIES.get(doc_id, set())),
            set(GROUND_TRUTH_RELATIONS.get(doc_id, set())))
