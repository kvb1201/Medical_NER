#!/usr/bin/env python3
"""Integration tests verifying the generated datasets, labels, and reports."""

import os
import json
import pytest

PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"

@pytest.fixture
def label2id():
    with open(os.path.join(PROCESSED_DIR, "label2id.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def test_label_mappings_exist(label2id):
    assert "O" in label2id
    assert label2id["O"] == 0
    assert "B-DISEASE" in label2id
    assert "I-DISEASE" in label2id
    assert "B-MEDICATION" in label2id
    assert "I-MEDICATION" in label2id
    assert "B-PATIENT_NAME" in label2id
    assert "I-PATIENT_NAME" in label2id

def test_augmented_files_integrity(label2id):
    valid_tags = set(label2id.keys())
    
    paths = [os.path.join(PROCESSED_DIR, f"bc5cdr_augmented_{s}.jsonl") for s in ["train", "valid", "test"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        # The JSONL splits are gitignored, so a fresh clone does not have them yet.
        pytest.skip(
            f"Dataset not built ({', '.join(missing)} missing); "
            "run PYTHONPATH=. python3 src/data/prepare_dataset.py first."
        )

    for split, file_path in zip(["train", "valid", "test"], paths):

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                count += 1
                rec = json.loads(line)
                assert "tokens" in rec
                assert "ner_tags" in rec
                assert "source" in rec
                assert "document_id" in rec
                
                tokens = rec["tokens"]
                tags = rec["ner_tags"]
                
                assert len(tokens) == len(tags), f"Length mismatch in {split} line {idx}"
                assert len(tokens) > 0, f"Empty tokens in {split} line {idx}"
                
                # Check all tags belong to label vocabulary
                for t in tags:
                    assert t in valid_tags, f"Unknown tag '{t}' in {split} line {idx}"
                    
        assert count > 0

def test_reports_generated():
    assert os.path.exists(os.path.join(REPORTS_DIR, "dataset_audit_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "label_distribution.csv"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "split_statistics.json"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "medication_mapping_audit.csv"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "near_duplicate_audit.csv"))

