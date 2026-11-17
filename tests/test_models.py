#!/usr/bin/env python3
"""
Unit tests for statistical sequence labeling models (HMM, MEMM, CRF)
and evaluation metrics in src/models/.
"""

import pytest
import numpy as np

from src.models.evaluation import (
    extract_entities,
    compute_entity_metrics,
    evaluate_predictions,
    TARGET_ENTITY_TYPES
)
from src.models.hmm import HiddenMarkovModel
from src.models.memm import MaximumEntropyMarkovModel
from src.models.crf import LinearChainCRF


# -----------------------------------------------------------------------------
# 1. Evaluation Metric Tests
# -----------------------------------------------------------------------------

def test_extract_entities_standard():
    tags = ["O", "B-DISEASE", "I-DISEASE", "O", "B-MEDICATION", "O", "B-PATIENT_NAME", "I-PATIENT_NAME"]
    entities = extract_entities(tags)
    assert len(entities) == 3
    assert ("DISEASE", 1, 3) in entities
    assert ("MEDICATION", 4, 5) in entities
    assert ("PATIENT_NAME", 6, 8) in entities


def test_extract_entities_adjacent_same_type():
    tags = ["B-DISEASE", "I-DISEASE", "B-DISEASE", "O"]
    entities = extract_entities(tags)
    assert len(entities) == 2
    assert ("DISEASE", 0, 2) in entities
    assert ("DISEASE", 2, 3) in entities


def test_extract_entities_ill_formed_orphan_inside():
    # Orphan I- tag should start a span gracefully
    tags = ["O", "I-DISEASE", "I-DISEASE", "O"]
    entities = extract_entities(tags)
    assert len(entities) == 1
    assert ("DISEASE", 1, 3) in entities


def test_extract_entities_type_switch_inside_keeps_every_token():
    # I- of a different type closes the open entity and starts a new one (conlleval)
    tags = ["B-DISEASE", "I-CHEMICAL", "I-CHEMICAL", "O"]
    assert extract_entities(tags) == [("DISEASE", 0, 1), ("CHEMICAL", 1, 3)]


def test_compute_entity_metrics_perfect():
    gold = [["B-DISEASE", "I-DISEASE", "O", "O", "B-MEDICATION"]]
    pred = [["B-DISEASE", "I-DISEASE", "O", "O", "B-MEDICATION"]]
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["per_entity"]["DISEASE"]["f1"] == 1.0
    assert metrics["per_entity"]["MEDICATION"]["f1"] == 1.0
    assert metrics["micro"]["f1"] == 1.0
    assert metrics["macro"]["f1"] == 1.0
    # In perfect 2-class slice, all 4 classes average: (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5
    assert metrics["macro"]["macro_all_f1"] == 0.5


def test_macro_active_vs_macro_all_distinction():
    # Only DISEASE and MEDICATION in gold
    gold = [["B-DISEASE", "O", "B-MEDICATION"]]
    # Prediction has 1 false positive PATIENT_NAME
    pred = [["B-DISEASE", "B-PATIENT_NAME", "B-MEDICATION"]]
    metrics = compute_entity_metrics(gold, pred)

    # Active classes (support > 0): DISEASE (f1=1.0), MEDICATION (f1=1.0)
    assert metrics["macro"]["f1"] == 1.0
    assert metrics["macro"]["active_classes"] == ["DISEASE", "MEDICATION"]

    # Full 4-class schema: DISEASE (1.0), MEDICATION (1.0), PATIENT_NAME (0.0), CHEMICAL (0.0)
    assert metrics["macro"]["macro_all_f1"] == 0.5
    assert metrics["per_entity"]["PATIENT_NAME"]["fp"] == 1


def test_compute_entity_metrics_partial_match_penalized():
    # Exact-match requires span boundaries to match exactly
    gold = [["O", "B-DISEASE", "I-DISEASE"]]
    pred = [["O", "B-DISEASE", "O"]]  # partial span
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["per_entity"]["DISEASE"]["precision"] == 0.0
    assert metrics["per_entity"]["DISEASE"]["recall"] == 0.0
    assert metrics["per_entity"]["DISEASE"]["f1"] == 0.0


def test_evaluate_predictions_overall():
    gold_seqs = [["O", "B-DISEASE", "I-DISEASE", "O"]]
    pred_seqs = [["O", "B-DISEASE", "I-DISEASE", "O"]]
    res = evaluate_predictions(gold_seqs, pred_seqs)
    assert res["token_metrics"]["token_accuracy"] == 1.0
    assert res["entity_metrics"]["micro"]["f1"] == 1.0


# -----------------------------------------------------------------------------
# 2. Hidden Markov Model (HMM) Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def toy_dataset():
    sents = [
        ["Patient", "John", "Doe", "has", "pneumonia", "."],
        ["Jane", "Smith", "was", "prescribed", "aspirin", "for", "fever", "."],
        ["Ibuprofen", "relieved", "the", "headache", "in", "Bob", "Taylor", "."]
    ]
    tags = [
        ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-DISEASE", "O"],
        ["B-PATIENT_NAME", "I-PATIENT_NAME", "O", "O", "B-MEDICATION", "O", "B-DISEASE", "O"],
        ["B-MEDICATION", "O", "O", "B-DISEASE", "O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O"]
    ]
    return sents, tags


def test_hmm_fit_and_predict(toy_dataset):
    sents, tags = toy_dataset
    hmm = HiddenMarkovModel(alpha_trans=0.1, alpha_emit=0.1, alpha_init=0.1, min_word_freq=1)
    hmm.fit(sents, tags)

    preds = hmm.predict(sents)
    assert len(preds) == len(sents)
    for sent, pred in zip(sents, preds):
        assert len(pred) == len(sent)
        # Verify valid BIO transitions in predicted sequence
        for i, tag in enumerate(pred):
            if tag.startswith("I-"):
                assert i > 0, "I- tag cannot be first token"
                prev_tag = pred[i - 1]
                entity = tag.split("-")[1]
                assert prev_tag in (f"B-{entity}", f"I-{entity}"), f"Invalid transition: {prev_tag} -> {tag}"


def test_hmm_unseen_token_handling(toy_dataset):
    sents, tags = toy_dataset
    hmm = HiddenMarkovModel(min_word_freq=1)
    hmm.fit(sents, tags)

    # Test completely unseen sentence
    unseen_sent = [["Alice", "Brown", "received", "metformin", "for", "diabetes", "."]]
    preds = hmm.predict(unseen_sent)
    assert len(preds[0]) == len(unseen_sent[0])
    for tag in preds[0]:
        assert tag in hmm.states


def test_hmm_parameter_matrices_include_start_and_tokens(toy_dataset):
    sents, tags = toy_dataset
    hmm = HiddenMarkovModel(min_word_freq=1)
    hmm.fit(sents, tags)
    tokens = sents[0]
    mats = hmm.parameter_matrices(tokens)
    assert mats["transition"]["rows"][0] == "START"
    assert mats["transition"]["rows"][1:] == hmm.states
    assert mats["transition"]["cols"] == hmm.states
    assert len(mats["transition"]["scores"]) == len(hmm.states) + 1
    assert mats["emission"]["rows"] == tokens
    assert mats["emission"]["cols"] == hmm.states
    assert len(mats["emission"]["scores"]) == len(tokens)
    assert len(mats["emission"]["sources"]) == len(tokens)


# -----------------------------------------------------------------------------
# 3. Maximum Entropy Markov Model (MEMM) Tests
# -----------------------------------------------------------------------------

def test_memm_fit_and_predict(toy_dataset):
    sents, tags = toy_dataset
    memm = MaximumEntropyMarkovModel(C=1.0, max_iter=50)
    memm.fit(sents, tags)

    preds = memm.predict(sents)
    assert len(preds) == len(sents)
    for sent, pred in zip(sents, preds):
        assert len(pred) == len(sent)
        for i, tag in enumerate(pred):
            if tag.startswith("I-"):
                assert i > 0
                prev_tag = pred[i - 1]
                entity = tag.split("-")[1]
                assert prev_tag in (f"B-{entity}", f"I-{entity}")


def test_memm_weight_decomposition(toy_dataset):
    sents, tags = toy_dataset
    memm = MaximumEntropyMarkovModel(C=1.0, max_iter=50)
    memm.fit(sents, tags)

    # Verify W_obs and trans_logits matrices are populated
    assert memm.W_obs is not None
    assert memm.trans_logits is not None
    assert memm.b is not None
    assert memm.trans_logits.shape == (len(memm.states) + 1, len(memm.states))


# -----------------------------------------------------------------------------
# 4. Conditional Random Field (CRF) Tests
# -----------------------------------------------------------------------------

def test_crf_fit_and_predict(toy_dataset):
    sents, tags = toy_dataset
    crf = LinearChainCRF(c1=0.1, c2=0.1, max_iterations=50)
    crf.fit(sents, tags)

    preds = crf.predict(sents)
    assert len(preds) == len(sents)
    for sent, pred in zip(sents, preds):
        assert len(pred) == len(sent)
        for i, tag in enumerate(pred):
            if tag.startswith("I-"):
                assert i > 0
                prev_tag = pred[i - 1]
                entity = tag.split("-")[1]
                assert prev_tag in (f"B-{entity}", f"I-{entity}")
