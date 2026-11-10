"""Extractor tests: dictionary entities, pattern relations, and the
"and"-clause handling in the synthetic corpus."""
from kgraph.data import DOCUMENTS, GROUND_TRUTH_ENTITIES, GROUND_TRUTH_RELATIONS
from kgraph.extractors import (DictionaryExtractor, PatternExtractor,
                               extract_corpus, ground_truth)


def test_dictionary_extractor_finds_all_labels_in_doc1():
    doc = DOCUMENTS[0]
    result = DictionaryExtractor().extract(doc["doc_id"], doc["text"])
    assert result.entity_labels() == GROUND_TRUTH_ENTITIES["doc-001"]


def test_dictionary_extractor_empty_doc():
    doc = DOCUMENTS[4]
    result = DictionaryExtractor().extract(doc["doc_id"], doc["text"])
    assert result.entity_labels() == set()


def test_dictionary_extractor_resolves_project_codes():
    result = DictionaryExtractor().extract(
        "t", "The Semantic Layer (PRJ-1042) review is due.")
    assert "Semantic Layer" in result.entity_labels()


def test_dictionary_extractor_never_invents_entities():
    result = DictionaryExtractor().extract(
        "t", "Zebra Quokka and the Quantum Banana met for lunch.")
    assert result.entity_labels() == set()


def test_pattern_extractor_recovers_doc1_relations():
    doc = DOCUMENTS[0]
    result = PatternExtractor().extract(doc["doc_id"], doc["text"])
    assert result.relation_triples() == GROUND_TRUTH_RELATIONS["doc-001"]


def test_pattern_extractor_recovers_doc2_relations():
    doc = DOCUMENTS[1]
    result = PatternExtractor().extract(doc["doc_id"], doc["text"])
    assert result.relation_triples() == GROUND_TRUTH_RELATIONS["doc-002"]


def test_pattern_extractor_recovers_doc3_relations():
    doc = DOCUMENTS[2]
    result = PatternExtractor().extract(doc["doc_id"], doc["text"])
    assert result.relation_triples() == GROUND_TRUTH_RELATIONS["doc-003"]


def test_pattern_extractor_recovers_doc4_relations():
    doc = DOCUMENTS[3]
    result = PatternExtractor().extract(doc["doc_id"], doc["text"])
    assert result.relation_triples() == GROUND_TRUTH_RELATIONS["doc-004"]


def test_extract_corpus_covers_all_docs():
    results = extract_corpus(DOCUMENTS)
    assert len(results) == 5
    assert all(r.doc_id for r in results)


def test_ground_truth_helper():
    entities, relations = ground_truth("doc-002")
    assert entities == GROUND_TRUTH_ENTITIES["doc-002"]
    assert relations == GROUND_TRUTH_RELATIONS["doc-002"]
