"""Evaluation: extraction quality and golden-set QA over the graph.

Two scores, both deterministic and reproducible from the synthetic
corpus:

1. Extraction precision/recall/F1 — the extractor's recovered
   entities and relation triples vs the ground truth in data.py,
   per document and micro-averaged over the corpus.
2. Golden QA accuracy — each question in GOLDEN_QUESTIONS is a
   multi-hop traversal from a start node; the answer is the exact set
   of labels a correct traversal returns. Exact-set match: a question
   scores 1.0 only when the traversal returns exactly the expected
   labels (no missing, no extra).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kgraph.data import (DOCUMENTS, GOLDEN_QUESTIONS,
                         GROUND_TRUTH_ENTITIES, GROUND_TRUTH_RELATIONS)
from kgraph.extractors import ExtractionResult, extract_corpus
from kgraph.query import GraphQuery, run_query
from kgraph.store import GraphView


def precision_recall_f1(predicted: set, truth: set) -> tuple[float, float, float]:
    if not predicted and not truth:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@dataclass
class ExtractionEval:
    per_doc: dict  # doc_id -> {"entities": (p, r, f1), "relations": (p, r, f1)}
    micro_entities: tuple  # (precision, recall, f1)
    micro_relations: tuple

    def to_dict(self) -> dict:
        return {
            "per_doc": {
                doc_id: {"entities": list(v["entities"]),
                         "relations": list(v["relations"])}
                for doc_id, v in self.per_doc.items()
            },
            "micro_entities": list(self.micro_entities),
            "micro_relations": list(self.micro_relations),
        }


def evaluate_extraction(results: list[ExtractionResult]) -> ExtractionEval:
    per_doc = {}
    pred_entities: set = set()
    true_entities: set = set()
    pred_relations: set = set()
    true_relations: set = set()
    for result in results:
        # Tag predictions with the doc so identical labels in different
        # docs do not collapse into one another.
        pred_e = {(result.doc_id, label) for label in result.entity_labels()}
        true_e = {(result.doc_id, label)
                  for label in GROUND_TRUTH_ENTITIES.get(result.doc_id, set())}
        pred_r = {(result.doc_id, *triple)
                  for triple in result.relation_triples()}
        true_r = {(result.doc_id, *triple)
                  for triple in GROUND_TRUTH_RELATIONS.get(result.doc_id, set())}
        per_doc[result.doc_id] = {
            "entities": precision_recall_f1(pred_e, true_e),
            "relations": precision_recall_f1(pred_r, true_r),
        }
        pred_entities |= pred_e
        true_entities |= true_e
        pred_relations |= pred_r
        true_relations |= true_r
    return ExtractionEval(
        per_doc=per_doc,
        micro_entities=precision_recall_f1(pred_entities, true_entities),
        micro_relations=precision_recall_f1(pred_relations, true_relations),
    )


@dataclass
class QuestionResult:
    question_id: str
    question: str
    expected: set
    got: set
    correct: bool


@dataclass
class QaEval:
    questions: list  # list[QuestionResult]
    accuracy: float

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "questions": [
                {"question_id": q.question_id, "question": q.question,
                 "expected": sorted(q.expected), "got": sorted(q.got),
                 "correct": q.correct}
                for q in self.questions
            ],
        }


def evaluate_qa(view: GraphView,
                questions=None) -> QaEval:
    """Run the golden questions against a graph view. Each question
    is a traversal from its start node over its edge types."""
    questions = questions if questions is not None else GOLDEN_QUESTIONS
    results = []
    for q in questions:
        start = view.find_by_label(q["start_label"])
        if start is None:
            got: set = set()
        else:
            query = GraphQuery(
                from_type=start.node_type, edge_types=list(q["edge_types"]),
                min_hops=1, max_hops=q["max_hops"],
                to_type=q.get("to_type"), reverse=q.get("reverse", False),
                start_label=q["start_label"])
            got = {n.label for n in run_query(view, query)}
        expected = set(q["answer"])
        results.append(QuestionResult(
            question_id=q["question_id"], question=q["question"],
            expected=expected, got=got, correct=got == expected))
    accuracy = (sum(1 for r in results if r.correct) / len(results)
                if results else 0.0)
    return QaEval(questions=results, accuracy=accuracy)


def full_eval(view: GraphView, documents=None) -> dict:
    """Extraction eval over the corpus + golden QA over the graph."""
    documents = documents if documents is not None else DOCUMENTS
    extraction = evaluate_extraction(extract_corpus(documents))
    qa = evaluate_qa(view)
    return {"extraction": extraction.to_dict(), "qa": qa.to_dict()}
