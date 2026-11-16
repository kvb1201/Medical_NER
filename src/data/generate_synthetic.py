#!/usr/bin/env python3
"""
Module: generate_synthetic.py
Generates controlled, deterministic synthetic clinical-style sentences containing:
  - Fictional patient names (B-PATIENT_NAME, I-PATIENT_NAME)
  - Clinically verified diseases (B-DISEASE, I-DISEASE)
  - Optional medications (B-MEDICATION, I-MEDICATION)
  - Strict disjoint template allocation and patient name pools to prevent cross-split leakage.
  - Active collision avoidance against original test sentences (Requirement 6).
"""

import os
import json
import random
import zlib
import logging
from typing import Dict, List, Tuple, Any, Set
from src.data.deduplicate import normalize_text

logger = logging.getLogger(__name__)

GENERATION_VERSION = "2.0.0"

# Disjoint Fictional Names Pool by Split (Guarantees zero name memorization leakage)
FICTIONAL_NAMES = {
    "train": [
        ["John", "Doe"], ["Robert", "Taylor"], ["Mary", "Johnson"], ["James", "Anderson"],
        ["Sarah", "Miller"], ["David", "Wilson"], ["Jennifer", "Thomas"], ["Daniel", "Jackson"],
        ["Lisa", "White"], ["William", "Harris"], ["Brian", "Clark"], ["Susan", "Walker"],
        ["Kevin", "Hall"], ["Karen", "Young"], ["Steven", "Allen"], ["Donna", "King"],
        ["Mark", "Wright"], ["Carol", "Scott"], ["Edward", "Green"], ["Ruth", "Baker"],
        ["Jason", "Adams"], ["Laura", "Nelson"], ["Gary", "Hill"], ["Sharon", "Ramirez"],
        ["Timothy", "Campbell"], ["Cynthia", "Mitchell"], ["Jeffrey", "Roberts"], ["Amy", "Carter"],
        ["Ryan", "Phillips"], ["Angela", "Evans"], ["Jacob", "Turner"], ["Melissa", "Torres"],
        ["Gary", "Parker"], ["Brenda", "Collins"], ["Nicholas", "Edwards"], ["Emma", "Stewart"],
        ["Eric", "Flores"], ["Olivia", "Morris"], ["Stephen", "Nguyen"], ["Rebecca", "Murphy"],
        ["Larry", "Rivera"], ["Deborah", "Cook"], ["Justin", "Rogers"], ["Amanda", "Morgan"],
        ["Scott", "Peterson"], ["Stephanie", "Cooper"], ["Brandon", "Reed"], ["Kathleen", "Bailey"],
        ["Benjamin", "Bell"], ["Shirley", "Gomez"], ["Jonathan", "Ward"], ["Nicole", "Cox"],
        ["Samuel", "Diaz"], ["Christine", "Richardson"], ["Frank", "Wood"], ["Samantha", "Watson"],
        ["Raymond", "Brooks"], ["Katherine", "Bennett"], ["Gregory", "Gray"], ["Debra", "James"]
    ],
    "valid": [
        ["Michael", "Brown"], ["Emily", "Davis"], ["Thomas", "Martin"], ["Jessica", "Garcia"],
        ["George", "Perez"], ["Nancy", "Nelson-Cole"], ["Kenneth", "Foster"], ["Betty", "Simmons"],
        ["Douglas", "Bryant"], ["Sandra", "Alexander"], ["Dennis", "Russell"], ["Ashley", "Griffin"],
        ["Jerry", "Hayes"], ["Kimberly", "Myers"], ["Alexander", "Ford"], ["Victoria", "Hamilton"],
        ["Patrick", "Graham"], ["Carolyn", "Sullivan"], ["Peter", "Wallace"], ["Martha", "Woods"],
        ["Harold", "Cole"], ["Michelle", "West"], ["Henry", "Jordan"], ["Kelly", "Owens"],
        ["Carl", "Reynolds"], ["Heather", "Fisher"], ["Arthur", "Ellis"], ["Teresa", "Harrison"],
        ["Ryan", "Gibson"], ["Ann", "McDonald"]
    ],
    "test": [
        ["Jane", "Smith"], ["Richard", "Black"], ["Patricia", "Lewis"], ["Christopher", "Knight"],
        ["Paul", "Cruz"], ["Margaret", "Marshall"], ["Donald", "Ortiz"], ["Dorothy", "Gomez-Ruiz"],
        ["Walter", "Freeman"], ["Helen", "Wells"], ["Keith", "Webb"], ["Joyce", "Simpson"],
        ["Lawrence", "Stevens"], ["Diane", "Tucker"], ["Roger", "Porter"], ["Alice", "Hunter"],
        ["Albert", "Hicks"], ["Julie", "Crawford"], ["Terry", "Henry"], ["Doris", "Boyd"],
        ["Gerald", "Mason"], ["Gloria", "Morales"], ["Joe", "Kennedy"], ["Evelyn", "Warren"],
        ["Sean", "Dixon"], ["Cheryl", "Ramos"], ["Austin", "Reyes"], ["Mildred", "Burns"],
        ["Jesse", "Gordon"], ["Joan", "Shaw"]
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
    ["ulcerative", "colitis"],
    ["peptic", "ulcer", "disease"],
    ["congestive", "heart", "failure"],
    ["chronic", "obstructive", "pulmonary", "disease"],
    ["ischemic", "stroke"],
    ["sepsis"],
    ["diabetic", "ketoacidosis"],
    ["hyperkalemia"],
    ["acute", "renal", "failure"],
    ["atopic", "dermatitis"],
    ["celiac", "disease"],
    ["epilepsy"],
    ["gout"],
    ["cirrhosis"],
    ["endocarditis"],
    ["osteoporosis"],
    ["diverticulitis"],
    ["cellulitis"],
    ["glaucoma"],
    ["peripheral", "neuropathy"],
    ["cholecystitis"]
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
    ["clopidogrel"],
    ["insulin", "glargine"],
    ["doxycycline"],
    ["simvastatin"],
    ["pantoprazole"],
    ["carvedilol"],
    ["escitalopram"],
    ["tramadol"],
    ["montelukast"],
    ["allopurinol"],
    ["duloxetine"],
    ["fluoxetine"],
    ["meloxicam"],
    ["spironolactone"],
    ["rosuvastatin"],
    ["famotidine"],
    ["clonazepam"],
    ["azithromycin"],
    ["apixaban"],
    ["venlafaxine"],
    ["propranolol"]
]

# Explicit, tokenized template patterns with typed slots
# Slots: {PATIENT}, {DISEASE}, {MEDICATION}
TEMPLATES: Dict[str, List[Any]] = {
    # --- TRAIN TEMPLATES (16 disjoint templates) ---
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
    "admission_dx_04_tr": [
        ("Inpatient", "O"), ("evaluation", "O"), ("of", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("revealed", "O"), ("acute", "O"), ("exacerbation", "O"), ("of", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
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
    "prescription_tx_04_tr": [
        ("The", "O"), ("dosage", "O"), ("of", "O"), ("{MEDICATION}", "MEDICATION"),
        ("was", "O"), ("adjusted", "O"), ("for", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("due", "O"), ("to", "O"), ("persistent", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
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
    "history_review_04_tr": [
        ("Electronic", "O"), ("health", "O"), ("records", "O"), ("for", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("document", "O"), ("a", "O"),
        ("chronic", "O"), ("history", "O"), ("of", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "followup_discharge_01": [
        ("Discharge", "O"), ("instructions", "O"), ("for", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("include", "O"), ("continuation", "O"),
        ("of", "O"), ("{MEDICATION}", "MEDICATION"), ("for", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "followup_discharge_02_tr": [
        ("During", "O"), ("outpatient", "O"), ("follow-up", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("reported", "O"), ("adherence", "O"),
        ("to", "O"), ("{MEDICATION}", "MEDICATION"), ("for", "O"),
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

    # --- VALIDATION TEMPLATES (6 disjoint templates) ---
    "admission_dx_01_va": [
        ("Initial", "O"), ("assessment", "O"), ("indicates", "O"), ("that", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("suffers", "O"), ("from", "O"),
        ("severe", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "admission_dx_02_va": [
        ("Physical", "O"), ("examination", "O"), ("of", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("was", "O"), ("consistent", "O"), ("with", "O"), ("acute", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_01_va": [
        ("The", "O"), ("clinical", "O"), ("team", "O"), ("administered", "O"),
        ("{MEDICATION}", "MEDICATION"), ("to", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("for", "O"), ("symptomatic", "O"), ("relief", "O"), ("of", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_02_va": [
        ("Maintenance", "O"), ("therapy", "O"), ("with", "O"), ("{MEDICATION}", "MEDICATION"),
        ("was", "O"), ("ordered", "O"), ("for", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("to", "O"), ("prevent", "O"), ("{DISEASE}", "DISEASE"), ("recurrence", "O"), (".", "O")
    ],
    "history_review_01_va": [
        ("According", "O"), ("to", "O"), ("outpatient", "O"), ("records", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("has", "O"), ("been", "O"), ("treated", "O"),
        ("for", "O"), ("{DISEASE}", "DISEASE"), ("since", "O"), ("2021", "O"), (".", "O")
    ],
    "followup_discharge_01_va": [
        ("At", "O"), ("today", "O"), ("'s", "O"), ("follow-up", "O"), ("visit", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("reported", "O"), ("good", "O"), ("tolerance", "O"),
        ("to", "O"), ("{MEDICATION}", "MEDICATION"), ("with", "O"), ("improved", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],

    # --- TEST TEMPLATES (6 disjoint templates) ---
    "admission_dx_01_te": [
        ("Emergency", "O"), ("department", "O"), ("intake", "O"), ("notes", "O"),
        ("show", "O"), ("that", "O"), ("{PATIENT}", "PATIENT_NAME"), ("was", "O"),
        ("admitted", "O"), ("with", "O"), ("acute", "O"), ("decompensated", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "admission_dx_02_te": [
        ("Urgent", "O"), ("hospital", "O"), ("intake", "O"), ("documented", "O"),
        ("that", "O"), ("{PATIENT}", "PATIENT_NAME"), ("presented", "O"),
        ("with", "O"), ("critical", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_01_te": [
        ("A", "O"), ("new", "O"), ("prescription", "O"), ("for", "O"),
        ("{MEDICATION}", "MEDICATION"), ("was", "O"), ("filled", "O"),
        ("for", "O"), ("{PATIENT}", "PATIENT_NAME"), ("to", "O"), ("counter", "O"),
        ("progressive", "O"), ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "prescription_tx_02_te": [
        ("The", "O"), ("attending", "O"), ("consultant", "O"), ("prescribed", "O"),
        ("oral", "O"), ("{MEDICATION}", "MEDICATION"), ("for", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("targeting", "O"), ("refractory", "O"),
        ("{DISEASE}", "DISEASE"), (".", "O")
    ],
    "referral_consult_01_te": [
        ("A", "O"), ("specialist", "O"), ("consultation", "O"), ("was", "O"),
        ("requested", "O"), ("regarding", "O"), ("{PATIENT}", "PATIENT_NAME"),
        ("due", "O"), ("to", "O"), ("refractory", "O"), ("{DISEASE}", "DISEASE"),
        ("unresponsive", "O"), ("to", "O"), ("{MEDICATION}", "MEDICATION"), (".", "O")
    ],
    "multi_entity_01_te": [
        ("Prior", "O"), ("to", "O"), ("scheduling", "O"), ("surgery", "O"), (",", "O"),
        ("{PATIENT}", "PATIENT_NAME"), ("was", "O"), ("evaluated", "O"), ("for", "O"),
        ("pre-existing", "O"), ("{DISEASE}", "DISEASE"), ("while", "O"), ("taking", "O"),
        ("{MEDICATION}", "MEDICATION"), (".", "O")
    ]
}

TEMPLATE_ALLOCATION = {
    "train": [
        "admission_dx_01", "admission_dx_02", "admission_dx_03", "admission_dx_04_tr",
        "prescription_tx_01", "prescription_tx_02", "prescription_tx_03", "prescription_tx_04_tr",
        "history_review_01", "history_review_02", "history_review_03", "history_review_04_tr",
        "followup_discharge_01", "followup_discharge_02_tr", "referral_consult_01", "multi_entity_01"
    ],
    "valid": [
        "admission_dx_01_va", "admission_dx_02_va",
        "prescription_tx_01_va", "prescription_tx_02_va",
        "history_review_01_va", "followup_discharge_01_va"
    ],
    "test": [
        "admission_dx_01_te", "admission_dx_02_te",
        "prescription_tx_01_te", "prescription_tx_02_te",
        "referral_consult_01_te", "multi_entity_01_te"
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
    random_seed: int = 42,
    disallowed_sentences: Set[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate deterministically augmented synthetic sentences for a given split.
    Prevents duplicates intra-split and cross-source (Requirement 6 & 7).
    """
    if split_name not in TEMPLATE_ALLOCATION:
        raise ValueError(f"Unknown split '{split_name}' for synthetic generation.")

    rng = random.Random(random_seed)
    allowed_templates = TEMPLATE_ALLOCATION[split_name]
    names_pool = FICTIONAL_NAMES[split_name]

    disallowed = disallowed_sentences or set()
    seen_texts = set()
    records = []
    i = 0
    max_attempts = count * 100
    attempts = 0

    while len(records) < count and attempts < max_attempts:
        attempts += 1
        tmpl_id = allowed_templates[i % len(allowed_templates)]
        name = rng.choice(names_pool)
        disease = rng.choice(VERIFIED_DISEASES)
        medication = rng.choice(VERIFIED_MEDICATIONS)

        tokens, ner_tags = generate_sentence_from_template(tmpl_id, name, disease, medication)
        norm_text = normalize_text(tokens)

        # Check intra-split uniqueness and cross-source disallowed sentences
        if norm_text in seen_texts or norm_text in disallowed:
            # Rejection sampling: regenerate another sample without modifying fixed benchmarks
            i += 1
            continue

        seen_texts.add(norm_text)
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

def derive_split_seed(base_seed: int, split_name: str) -> int:
    """Return a per-split seed that is identical on every run and machine.

    Python's built-in hash() of a str is salted per process (PYTHONHASHSEED), so it
    must not be used here; CRC32 of the split name is stable.
    """
    return base_seed + zlib.crc32(split_name.encode("utf-8")) % 1000

def generate_and_save_synthetic(
    output_dir: str = "data/synthetic",
    counts: Dict[str, int] = None,
    random_seed: int = 42,
    disallowed_by_split: Dict[str, Set[str]] = None
) -> Dict[str, str]:
    """Generate and save synthetic datasets for train, valid, and test splits."""
    if counts is None:
        counts = {"train": 4000, "valid": 500, "test": 500}

    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    disallowed_map = disallowed_by_split or {}

    # Track all synthetic sentences across splits to enforce cross-split synthetic uniqueness
    global_seen_synthetic = set()

    for split_name, count in counts.items():
        split_disallowed = set(disallowed_map.get(split_name, set()))
        split_disallowed.update(global_seen_synthetic)

        records = generate_synthetic_split(
            split_name,
            count,
            random_seed=derive_split_seed(random_seed, split_name),
            disallowed_sentences=split_disallowed
        )

        for r in records:
            global_seen_synthetic.add(normalize_text(r["tokens"]))

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
