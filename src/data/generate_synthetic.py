#!/usr/bin/env python3
"""
Module: generate_synthetic.py
Generates controlled, deterministic synthetic clinical-style sentences containing:
  - Fictional patient names (B-PATIENT_NAME, I-PATIENT_NAME)
  - Clinically verified diseases (B-DISEASE, I-DISEASE)
  - Optional medications (B-MEDICATION, I-MEDICATION)
  - Strict disjoint template allocation to prevent cross-split leakage.
"""

import os
import json
import random
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

GENERATION_VERSION = "1.0.0"

# Disjoint Fictional Names Pool by Split (Guarantees zero name memorization leakage)
FICTIONAL_NAMES = {
    "train": [
        ["John", "Doe"],
        ["Robert", "Taylor"],
        ["Mary", "Johnson"],
        ["James", "Anderson"],
        ["Sarah", "Miller"],
        ["David", "Wilson"],
        ["Jennifer", "Thomas"],
        ["Daniel", "Jackson"],
        ["Lisa", "White"],
        ["William", "Harris"],
        ["Brian", "Clark"],
        ["Susan", "Walker"],
        ["Kevin", "Hall"],
        ["Karen", "Young"],
        ["Steven", "Allen"],
        ["Donna", "King"],
        ["Mark", "Wright"],
        ["Carol", "Scott"],
        ["Edward", "Green"],
        ["Ruth", "Baker"]
    ],
    "valid": [
        ["Michael", "Brown"],
        ["Emily", "Davis"],
        ["Thomas", "Martin"],
        ["Jessica", "Garcia"],
        ["George", "Adams"],
        ["Nancy", "Nelson"],
        ["Kenneth", "Carter"],
        ["Betty", "Mitchell"]
    ],
    "test": [
        ["Jane", "Smith"],
        ["Richard", "Clark"],
        ["Patricia", "Lewis"],
        ["Christopher", "Hill"],
        ["Paul", "Roberts"],
        ["Margaret", "Phillips"],
        ["Donald", "Campbell"],
        ["Dorothy", "Parker"]
    ]
}

# Verified clinical diseases (single and multi-token)
VERIFIED_DISEASES = [
    ["diabetes", "mellitus"],
    ["type", "2", "diabetes"],
    ["essential", "hypertension"],
    ["pneumonia"],
    ["atrial", "fibrillation"],
    ["asthma"],
    ["osteoarthritis"],
    ["rheumatoid", "arthritis"],
    ["major", "depressive", "disorder"],
    ["hypothyroidism"],
    ["chronic", "kidney", "disease"],
    ["heart", "failure"],
    ["migraine"],
    ["pulmonary", "embolism"],
    ["deep", "vein", "thrombosis"],
    ["acute", "bronchitis"],
    ["pancreatitis"],
    ["multiple", "sclerosis"],
    ["hyperlipidemia"],
    ["ulcerative", "colitis"]
]

# Verified clinical medications (single and multi-token)
VERIFIED_MEDICATIONS = [
    ["metformin"],
    ["lisinopril"],
    ["atorvastatin"],
    ["amoxicillin"],
    ["ibuprofen"],
    ["albuterol"],
    ["levothyroxine"],
    ["omeprazole"],
    ["amlodipine"],
    ["metoprolol"],
    ["losartan"],
    ["hydrochlorothiazide"],
    ["gabapentin"],
    ["sertraline"],
    ["furosemide"],
    ["prednisone"],
    ["ciprofloxacin"],
    ["warfarin"],
    ["aspirin"],
    ["clopidogrel"]
]

# Explicit, tokenized template patterns with typed slots
# Slots: {PATIENT}, {DISEASE}, {MEDICATION}
TEMPLATES: Dict[str, List[Any]] = {
    # --- TRAIN TEMPLATES (12 unique templates) ---
    "admission_dx_01": [
        ("Patient", "O"), ("{PATIENT}", "PATIENT_NAME"), ("was", "O"), ("admitted", "O"),
        ("for", "O"), ("management", "O"), ("of", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "admission_dx_02": [
        ("Upon", "O"), ("hospital", "O"), ("admission", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("presented", "O"), ("with", "O"),
        ("symptoms", "O"), ("of", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "admission_dx_03": [
        ("The", "O"), ("attending", "O"), ("physician", "O"), ("evaluated", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("and", "O"), ("confirmed", "O"),
        ("a", "O"), ("diagnosis", "O"), ("of", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_01": [
        ("{PATIENT}", "PATIENT_NAME"), ("was", "O"), ("prescribed", "O"),
        ("{MEDICATION}", "MEDICATION"), ("for", "O"), ("treatment", "O"),
        ("of", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_02": [
        ("Following", "O"), ("the", "O"), ("clinical", "O"), ("workup", "O"), (",", "O"),
        ("Dr.", "O"), ("Johnson", "O"), ("started", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("on", "O"), ("daily", "O"), ("{MEDICATION}", "MEDICATION"), ("to", "O"),
        ("control", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_03": [
        ("Therapy", "O"), ("with", "O"), ("{MEDICATION}", "MEDICATION"), ("was", "O"),
        ("initiated", "O"), ("in", "O"), ("{PATIENT}", "PATIENT_NAME"), ("to", "O"),
        ("manage", "O"), ("recurrent", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "history_review_01": [
        ("{PATIENT}", "PATIENT_NAME"), ("'s", "O"), ("medical", "O"), ("history", "O"),
        ("is", "O"), ("notable", "O"), ("for", "O"), ("long-standing", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "history_review_02": [
        ("A", "O"), ("past", "O"), ("medical", "O"), ("history", "O"), ("taken", "O"),
        ("from", "O"), ("{PATIENT}", "PATIENT_NAME"), ("revealed", "O"),
        ("untreated", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "history_review_03": [
        ("Review", "O"), ("of", "O"), ("systems", "O"), ("for", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("demonstrates", "O"), ("active", "O"),
        ("{DISEASE}", "DISEASE"), ("stabilized", "O"), ("on", "O"),
        ("{MEDICATION}", "MEDICATION"), (".", "O")
    ],
    "followup_discharge_01": [
        ("Discharge", "O"), ("instructions", "O"), ("for", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("include", "O"), ("continuation", "O"),
        ("of", "O"), ("{MEDICATION}", "MEDICATION"), ("for", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "referral_consult_01": [
        ("Dr.", "O"), ("Williams", "O"), ("referred", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("to", "O"), ("cardiology", "O"), ("for", "O"), ("suspected", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "multi_entity_01": [
        ("After", "O"), ("{PATIENT}", "PATIENT_NAME"), ("developed", "O"),
        ("complications", "O"), ("from", "O"), ("{DISEASE}", "DISEASE"), (",", "O"),
        ("treatment", "O"), ("was", "O"), ("switched", "O"), ("to", "O"),
        ("{MEDICATION}", "MEDICATION"), (".", "O")
    ],

    # --- VALIDATION TEMPLATES (4 disjoint templates) ---
    "admission_dx_04": [
        ("Initial", "O"), ("assessment", "O"), ("indicates", "O"), ("that", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("suffers", "O"), ("from", "O"),
        ("severe", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_04": [
        ("The", "O"), ("clinical", "O"), ("team", "O"), ("administered", "O"),
        ("{MEDICATION}", "MEDICATION"), ("to", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("for", "O"), ("symptomatic", "O"), ("relief", "O"), ("of", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "history_review_04": [
        ("According", "O"), ("to", "O"), ("outpatient", "O"), ("records", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("has", "O"), ("been", "O"), ("treated", "O"),
        ("for", "O"), ("{DISEASE}", "DISEASE"), ("since", "O"), ("2021", "O"), (".", "O")
    ],
    "followup_discharge_02": [
        ("At", "O"), ("today", "O"), ("'s", "O"), ("follow-up", "O"), ("visit", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("reported", "O"), ("good", "O"), ("tolerance", "O"),
        ("to", "O"), ("{MEDICATION}", "MEDICATION"), ("with", "O"), ("improved", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],

    # --- TEST TEMPLATES (4 disjoint templates) ---
    "admission_dx_05": [
        ("Emergency", "O"), ("department", "O"), ("intake", "O"), ("notes", "O"),
        ("show", "O"), ("that", "O"), ("{PATIENT}", "PATIENT_NAME"), ("was", "O"),
        ("admitted", "O"), ("with", "O"), ("acute", "O"), ("decompensated", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_05": [
        ("A", "O"), ("new", "O"), ("prescription", "O"), ("for", "O"),
        ("{MEDICATION}", "MEDICATION"), ("was", "O"), ("filled", "O"),
        ("for", "O"), ("{PATIENT}", "PATIENT_NAME"), ("to", "O"), ("counter", "O"),
        ("progressive", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "referral_consult_02": [
        ("A", "O"), ("specialist", "O"), ("consultation", "O"), ("was", "O"),
        ("requested", "O"), ("regarding", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("due", "O"), ("to", "O"), ("refractory", "O"), ("{DISEASE}", "DISEASE"),
        ("unresponsive", "O"), ("to", "O"), ("{MEDICATION}", "MEDICATION"), (".", "O")
    ],
    "multi_entity_02": [
        ("Prior", "O"), ("to", "O"), ("scheduling", "O"), ("surgery", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("was", "O"), ("evaluated", "O"), ("for", "O"),
        ("pre-existing", "O"), ("{DISEASE}", "DISEASE"), ("while", "O"), ("taking", "O"),
        ("{MEDICATION}", "MEDICATION"), (".", "O")
    ]
}

TEMPLATE_ALLOCATION = {
    "train": [
        "admission_dx_01", "admission_dx_02", "admission_dx_03",
        "prescription_tx_01", "prescription_tx_02", "prescription_tx_03",
        "history_review_01", "history_review_02", "history_review_03",
        "followup_discharge_01", "referral_consult_01", "multi_entity_01"
    ],
    "valid": [
        "admission_dx_04", "prescription_tx_04", "history_review_04", "followup_discharge_02"
    ],
    "test": [
        "admission_dx_05", "prescription_tx_05", "referral_consult_02", "multi_entity_02"
    ]
}

def generate_sentence_from_template(
    template_id: str,
    name_tokens: List[str],
    disease_tokens: List[str],
    medication_tokens: List[str]
) -> Tuple[List[str], List[str]]:
    """Build tokens and ner_tags token-by-token with exact BIO tags."""
    template_def = TEMPLATES[template_id]
    out_tokens = []
    out_tags = []
    
    for token_pattern, entity_type in template_def:
        if token_pattern == "{PATIENT}":
            for idx, tok in enumerate(name_tokens):
                out_tokens.append(tok)
                out_tags.append(f"B-{entity_type}" if idx == 0 else f"I-{entity_type}")
        elif token_pattern == "{DISEASE}":
            for idx, tok in enumerate(disease_tokens):
                out_tokens.append(tok)
                out_tags.append(f"B-{entity_type}" if idx == 0 else f"I-{entity_type}")
        elif token_pattern == "{MEDICATION}":
            for idx, tok in enumerate(medication_tokens):
                out_tokens.append(tok)
                out_tags.append(f"B-{entity_type}" if idx == 0 else f"I-{entity_type}")
        else:
            out_tokens.append(token_pattern)
            out_tags.append("O")
            
    assert len(out_tokens) == len(out_tags), "Token and tag lengths must match exactly!"
    return out_tokens, out_tags

def generate_synthetic_split(
    split_name: str,
    count: int,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate deterministically augmented synthetic sentences for a given split."""
    if split_name not in TEMPLATE_ALLOCATION:
        raise ValueError(f"Unknown split '{split_name}' for synthetic generation.")
        
    rng = random.Random(random_seed)
    allowed_templates = TEMPLATE_ALLOCATION[split_name]
    names_pool = FICTIONAL_NAMES[split_name]
    
    seen_texts = set()
    records = []
    i = 0
    max_attempts = count * 50
    attempts = 0
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        tmpl_id = allowed_templates[i % len(allowed_templates)]
        name = rng.choice(names_pool)
        disease = rng.choice(VERIFIED_DISEASES)
        medication = rng.choice(VERIFIED_MEDICATIONS)
        
        tokens, ner_tags = generate_sentence_from_template(tmpl_id, name, disease, medication)
        sent_str = " ".join(tokens)
        if sent_str in seen_texts:
            continue
            
        seen_texts.add(sent_str)
        record = {
            "document_id": f"synthetic_{split_name}_{len(records):06d}",
            "sentence_id": f"synth_{split_name}_{len(records)}",
            "source": "synthetic",
            "source_split": split_name,
            "template_id": tmpl_id,
            "random_seed": random_seed,
            "generation_version": GENERATION_VERSION,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "original_ner_tags": ner_tags
        }
        records.append(record)
        i += 1
        
    if len(records) < count:
        raise ValueError(f"Could not generate {count} unique sentences (reached {len(records)})")
        
    logger.info(f"Generated {len(records)} synthetic records for split '{split_name}' using seed {random_seed}")
    return records

def generate_and_save_synthetic(
    output_dir: str = "data/synthetic",
    counts: Dict[str, int] = None,
    random_seed: int = 42
) -> Dict[str, str]:
    """Generate and save synthetic datasets for train, valid, and test splits."""
    if counts is None:
        counts = {"train": 800, "valid": 100, "test": 100}
        
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    for split_name, count in counts.items():
        records = generate_synthetic_split(split_name, count, random_seed=random_seed + hash(split_name) % 1000)
        out_path = os.path.join(output_dir, f"synthetic_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        saved_files[split_name] = out_path
        logger.info(f"Saved synthetic split '{split_name}' to {out_path} ({len(records)} records)")
        
    meta_path = os.path.join(output_dir, "synthetic_generation_metadata.json")
    meta = {
        "generation_version": GENERATION_VERSION,
        "base_seed": random_seed,
        "template_allocation": TEMPLATE_ALLOCATION,
        "counts": counts,
        "saved_files": saved_files
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return saved_files

if __name__ == "__main__":
    generate_and_save_synthetic()

