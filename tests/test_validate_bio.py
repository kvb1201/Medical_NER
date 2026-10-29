#!/usr/bin/env python3
"""Unit tests for BIO validation logic in src/data/validate_bio.py."""

import pytest
from src.data.validate_bio import (
    validate_record_bio,
    audit_split_integrity,
    check_cross_split_leakage,
    DatasetValidationError
)

def test_valid_record():
    tokens = ["Patient", "John", "Doe", "has", "pneumonia", "."]
    tags = ["O", "B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-DISEASE", "O"]
    res = validate_record_bio(tokens, tags, record_id="test_01")
    assert res["valid"] is True
    assert res["error"] is None

def test_length_mismatch_raises_error():
    tokens = ["Patient", "has", "hypertension"]
    tags = ["O", "O"]
    with pytest.raises(DatasetValidationError, match="length mismatch"):
        validate_record_bio(tokens, tags, record_id="test_err_len")

def test_empty_tokens_raises_error():
    tokens = []
    tags = []
    with pytest.raises(DatasetValidationError, match="empty token sequence"):
        validate_record_bio(tokens, tags, record_id="test_err_empty")

def test_invalid_bio_transition_raises_error():
    # I-DISEASE without preceding B-DISEASE
    tokens = ["Patient", "has", "heart", "failure", "."]
    tags = ["O", "O", "I-DISEASE", "I-DISEASE", "O"]
    with pytest.raises(DatasetValidationError, match="Invalid BIO transition"):
        validate_record_bio(tokens, tags, record_id="test_err_bio")

def test_incompatible_continuation_tag():
    # I-MEDICATION following B-DISEASE
    tokens = ["severe", "infection", "treated"]
    tags = ["B-DISEASE", "I-MEDICATION", "O"]
    with pytest.raises(DatasetValidationError, match="Invalid BIO transition"):
        validate_record_bio(tokens, tags, record_id="test_err_tag_mismatch")

def test_invalid_tag_syntax():
    tokens = ["Aspirin", "was", "given"]
    tags = ["MEDICATION", "O", "O"]  # missing B- prefix
    with pytest.raises(DatasetValidationError, match="invalid tag syntax"):
        validate_record_bio(tokens, tags, record_id="test_err_syntax")

def test_audit_split_integrity():
    records = [
        {"tokens": ["John", "Doe", "takes", "aspirin", "."], "ner_tags": ["B-PATIENT_NAME", "I-PATIENT_NAME", "O", "B-MEDICATION", "O"], "document_id": "doc1"},
        {"tokens": ["Hypertension", "is", "common", "."], "ner_tags": ["B-DISEASE", "O", "O", "O"], "document_id": "doc2"}
    ]
    res = audit_split_integrity(records, split_name="test_split", required_entity_types={"DISEASE", "MEDICATION", "PATIENT_NAME"})
    assert res["total_records"] == 2
    assert res["entity_mention_counts"]["DISEASE"] == 1
    assert res["entity_mention_counts"]["MEDICATION"] == 1
    assert res["entity_mention_counts"]["PATIENT_NAME"] == 1

def test_cross_split_leakage_detection():
    split_a = [{"tokens": ["A", "."], "ner_tags": ["O", "O"], "document_id": "doc_1", "template_id": "tmpl_1"}]
    split_b = [{"tokens": ["B", "."], "ner_tags": ["O", "O"], "document_id": "doc_1", "template_id": "tmpl_2"}]
    
    # doc_1 overlaps between split_a and split_b
    with pytest.raises(DatasetValidationError, match="Document leakage detected"):
        check_cross_split_leakage({"split_a": split_a, "split_b": split_b}, strict=True)

