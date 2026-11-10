"""Eval tests: extraction precision/recall and golden QA accuracy."""
from kgraph.cli import build_store
from kgraph.data import DOCUMENTS
from kgraph.eval import (evaluate_extraction, evaluate_qa,
                         precision_recall_f1)
from kgraph.extractors import extract_corpus


def test_precision_recall_f1_basics():
    assert precision_recall_f1({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)
    assert precision_recall_f1(set(), set()) == (1.0, 1.0, 1.0)
    p, r, f1 = precision_recall_f1({"a", "b"}, {"a", "c"})
    assert (p, r, f1) == (0.5, 0.5, 0.5)
    assert precision_recall_f1(set(), {"a"}) == (0.0, 0.0, 0.0)


def test_extraction_eval_is_perfect_on_synthetic_corpus():
    # The corpus is built so the deterministic extractor recovers the
    # ground truth exactly; any regression here is a real bug.
    evaluation = evaluate_extraction(extract_corpus(DOCUMENTS))
    p, r, f1 = evaluation.micro_entities
    assert (p, r, f1) == (1.0, 1.0, 1.0)
    p, r, f1 = evaluation.micro_relations
    assert (p, r, f1) == (1.0, 1.0, 1.0)
    for doc_id, scores in evaluation.per_doc.items():
        assert scores["entities"] == (1.0, 1.0, 1.0), doc_id
        assert scores["relations"] == (1.0, 1.0, 1.0), doc_id


def test_golden_qa_all_pass_on_canonical_graph():
    view = build_store().current_view()
    qa = evaluate_qa(view)
    assert qa.accuracy == 1.0
    assert all(q.correct for q in qa.questions)
    assert len(qa.questions) == 5


def test_golden_qa_fails_when_graph_is_wrong():
    store = build_store()
    # Removing the risk node breaks q2 (Carla's products' risks).
    store.remove_node("risk-pii-exposure", actor="test")
    qa = evaluate_qa(store.current_view())
    assert qa.accuracy < 1.0
    failed = {q.question_id for q in qa.questions if not q.correct}
    assert "q2" in failed
