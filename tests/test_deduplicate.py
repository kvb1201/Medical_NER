#!/usr/bin/env python3
"""
Unit tests for duplicate detection, near-duplicate auditing, and filtering
in src/data/deduplicate.py.
"""

import pytest
from src.data.deduplicate import (
    normalize_text,
    compute_token_jaccard,
    compute_conservative_similarity,
    detect_exact_duplicates,
    detect_near_duplicates,
    filter_cross_source_duplicates
)

def test_text_normalization():
    # Whitespace, punctuation, case, and Unicode NFKC normalization
    tokens = ["  Patient", "John", "Doe", "was", "admitted", "...", " "]
    norm = normalize_text(tokens)
    assert norm == "patient john doe was admitted"

    raw_str = "\tPatient  has   severe   hypertension. \n"
    assert normalize_text(raw_str) == "patient has severe hypertension"

def test_conservative_similarity():
    tokens1 = ["Patient", "John", "Doe", "was", "prescribed", "metformin", "."]
    tokens2 = ["Patient", "Jane", "Doe", "was", "prescribed", "metformin", "."]
    # Identical except for first name (John vs Jane)
    sim = compute_conservative_similarity(tokens1, tokens2)
    assert sim >= 0.85

    tokens3 = ["Metformin", "decreases", "hepatic", "glucose", "production", "."]
    sim_unrelated = compute_conservative_similarity(tokens1, tokens3)
    assert sim_unrelated < 0.50

def test_exact_duplicate_detection():
    orig_records = [
        {"document_id": "orig_001", "tokens": ["Patient", "John", "Doe", "has", "pneumonia", "."], "ner_tags": ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-DISEASE", "O"]},
        {"document_id": "orig_002", "tokens": ["The", "rat", "was", "given", "aspirin", "."], "ner_tags": ["O", "O", "O", "O", "B-MEDICATION", "O"]}
    ]
    synth_records = [
        {"document_id": "syn_001", "tokens": ["patient", "john", "doe", "has", "pneumonia"], "ner_tags": ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-DISEASE"]}
    ]
    dups = detect_exact_duplicates(orig_records, synth_records)
    assert len(dups) == 1
    assert dups[0]["original_doc_id"] == "orig_001"
    assert dups[0]["synthetic_doc_id"] == "syn_001"
    assert dups[0]["is_confirmed_leakage"] is True

def test_shared_entity_vocabulary_not_removed():
    # Requirement 8: Shared entity vocabulary without shared syntactic structure is NOT leakage.
    orig_records = [
        {
            "document_id": "orig_pubmed_123",
            "tokens": ["Pharmacokinetics", "of", "metformin", "in", "patients", "with", "type", "2", "diabetes", "mellitus", "."],
            "ner_tags": ["O", "O", "B-MEDICATION", "O", "O", "O", "B-DISEASE", "I-DISEASE", "I-DISEASE", "I-DISEASE", "O"]
        }
    ]
    synth_records = [
        {
            "document_id": "syn_clinical_456",
            "tokens": ["Patient", "John", "Doe", "was", "prescribed", "metformin", "for", "type", "2", "diabetes", "mellitus", "."],
            "ner_tags": ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "O", "B-MEDICATION", "O", "B-DISEASE", "I-DISEASE", "I-DISEASE", "I-DISEASE", "O"]
        }
    ]
    near_dups = detect_near_duplicates(orig_records, synth_records, threshold=0.85)
    # The PubMed sentence describes pharmacokinetics with distinct non-entity syntax; it must NOT be confirmed leakage
    confirmed = [d for d in near_dups if d["is_confirmed_leakage"]]
    assert len(confirmed) == 0

def test_fixed_test_benchmark_immutability():
    # Requirement 6: Original test partition is fixed. Duplicates cannot remove test originals.
    orig_test = [
        {"document_id": "test_orig_001", "tokens": ["Patient", "has", "asthma", "."], "ner_tags": ["O", "O", "B-DISEASE", "O"]}
    ]
    synth_test = [
        {"document_id": "test_syn_001", "tokens": ["Patient", "has", "asthma", "."], "ner_tags": ["O", "O", "B-DISEASE", "O"]}
    ]
    # Attempting to filter test records with confirmed duplicates raises RuntimeError
    with pytest.raises(RuntimeError, match="Test partition duplicate detected"):
        filter_cross_source_duplicates(orig_test, synth_test, split_name="test")

def test_train_validation_removal():
    # Requirement 5: In train, confirmed duplicates are filtered from original partition
    orig_train = [
        {"document_id": "train_orig_001", "tokens": ["Patient", "John", "Doe", "admitted", "for", "asthma", "."], "ner_tags": ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "O", "B-DISEASE", "O"]},
        {"document_id": "train_orig_002", "tokens": ["Effect", "of", "aspirin", "on", "headache", "."], "ner_tags": ["O", "O", "B-MEDICATION", "O", "B-DISEASE", "O"]}
    ]
    synth_train = [
        {"document_id": "train_syn_001", "tokens": ["Patient", "John", "Doe", "admitted", "for", "asthma", "."], "ner_tags": ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "O", "B-DISEASE", "O"]}
    ]
    filtered_orig, removed_orig, audit_log = filter_cross_source_duplicates(orig_train, synth_train, split_name="train")
    assert len(filtered_orig) == 1
    assert filtered_orig[0]["document_id"] == "train_orig_002"
    assert len(removed_orig) == 1
    assert removed_orig[0]["document_id"] == "train_orig_001"
    assert len(audit_log) == 1
    assert audit_log[0]["action_taken"] == "REMOVED_ORIGINAL_TRAIN"

