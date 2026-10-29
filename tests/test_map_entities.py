#!/usr/bin/env python3
"""Unit tests for Chemical-to-Medication mapping logic in src/data/map_entities.py."""

import pytest
from src.data.map_entities import EntityMapper

@pytest.fixture
def mapper():
    return EntityMapper(retain_unmapped_chemical=True)

def test_disease_mapping(mapper):
    record = {
        "tokens": ["Patient", "has", "hypertension", "and", "heart", "failure", "."],
        "raw_tags": ["O", "O", "B-Disease", "O", "B-Disease", "I-Disease", "O"]
    }
    mapped = mapper.map_record(record)
    assert mapped["ner_tags"] == [
        "O", "O", "B-DISEASE", "O", "B-DISEASE", "I-DISEASE", "O"
    ]

def test_curated_medication_mapping(mapper):
    record = {
        "tokens": ["Administered", "aspirin", "and", "morphine", "."],
        "raw_tags": ["O", "B-Chemical", "O", "B-Chemical", "O"]
    }
    mapped = mapper.map_record(record)
    assert mapped["ner_tags"] == ["O", "B-MEDICATION", "O", "B-MEDICATION", "O"]

def test_pharmacological_suffix_mapping(mapper):
    # metoprolol (-olol), enalapril (-pril), atorvastatin (-statin)
    record = {
        "tokens": ["Prescribed", "metoprolol", "and", "enalapril", "."],
        "raw_tags": ["O", "B-Chemical", "O", "B-Chemical", "O"]
    }
    mapped = mapper.map_record(record)
    assert mapped["ner_tags"] == ["O", "B-MEDICATION", "O", "B-MEDICATION", "O"]

def test_non_medication_exclusion(mapper):
    # creatinine (metabolite) and calcium (electrolyte) must be retained as CHEMICAL
    record = {
        "tokens": ["Serum", "creatinine", "and", "calcium", "levels", "."],
        "raw_tags": ["O", "B-Chemical", "O", "B-Chemical", "O", "O"]
    }
    mapped = mapper.map_record(record)
    assert mapped["ner_tags"] == ["O", "B-CHEMICAL", "O", "B-CHEMICAL", "O", "O"]

def test_multi_word_chemical_preservation(mapper):
    record = {
        "tokens": ["Patient", "takes", "alpha", "-", "methyldopa", "."],
        "raw_tags": ["O", "O", "B-Chemical", "I-Chemical", "I-Chemical", "O"]
    }
    mapped = mapper.map_record(record)
    # methyldopa is a known drug (-dopa)
    assert mapped["ner_tags"] == ["O", "O", "B-MEDICATION", "I-MEDICATION", "I-MEDICATION", "O"]

