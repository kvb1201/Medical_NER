#!/usr/bin/env python3
"""Unit tests for synthetic patient augmentation in src/data/generate_synthetic.py."""

import pytest
from src.data.generate_synthetic import (
    generate_synthetic_split,
    generate_sentence_from_template,
    TEMPLATE_ALLOCATION,
    FICTIONAL_NAMES
)

def test_template_generation_exact_alignment():
    tmpl_id = "admission_dx_01"
    name = ["Jane", "Doe"]
    disease = ["diabetes", "mellitus"]
    medication = ["metformin"]
    
    tokens, tags = generate_sentence_from_template(tmpl_id, name, disease, medication)
    assert len(tokens) == len(tags)
    
    # Check patient name BIO tags
    p_indices = [idx for idx, t in enumerate(tags) if "PATIENT_NAME" in t]
    assert len(p_indices) == 2
    assert tags[p_indices[0]] == "B-PATIENT_NAME"
    assert tags[p_indices[1]] == "I-PATIENT_NAME"
    assert tokens[p_indices[0]] == "Jane"
    assert tokens[p_indices[1]] == "Doe"

def test_split_template_disjointness():
    train_tmpls = set(TEMPLATE_ALLOCATION["train"])
    valid_tmpls = set(TEMPLATE_ALLOCATION["valid"])
    test_tmpls = set(TEMPLATE_ALLOCATION["test"])
    
    assert len(train_tmpls.intersection(valid_tmpls)) == 0, "Train and Valid share templates!"
    assert len(train_tmpls.intersection(test_tmpls)) == 0, "Train and Test share templates!"
    assert len(valid_tmpls.intersection(test_tmpls)) == 0, "Valid and Test share templates!"

def test_fictional_names_disjointness():
    # Flatten names
    train_names = set(" ".join(n) for n in FICTIONAL_NAMES["train"])
    valid_names = set(" ".join(n) for n in FICTIONAL_NAMES["valid"])
    test_names = set(" ".join(n) for n in FICTIONAL_NAMES["test"])
    
    assert len(train_names.intersection(valid_names)) == 0, "Train and Valid share fictional names!"
    assert len(train_names.intersection(test_names)) == 0, "Train and Test share fictional names!"
    assert len(valid_names.intersection(test_names)) == 0, "Valid and Test share fictional names!"

def test_deterministic_generation():
    recs1 = generate_synthetic_split("train", count=10, random_seed=42)
    recs2 = generate_synthetic_split("train", count=10, random_seed=42)
    
    for r1, r2 in zip(recs1, recs2):
        assert r1["tokens"] == r2["tokens"]
        assert r1["ner_tags"] == r2["ner_tags"]
        assert r1["template_id"] == r2["template_id"]

def test_provenance_metadata():
    recs = generate_synthetic_split("train", count=5, random_seed=123)
    for r in recs:
        assert r["source"] == "synthetic"
        assert r["random_seed"] == 123
        assert r["generation_version"] == "2.0.0"
        assert "template_id" in r
        assert "document_id" in r

