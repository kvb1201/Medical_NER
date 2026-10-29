#!/usr/bin/env python3
"""
Medical NER Dataset Audit and Preparation Script
Dataset: tner/bc5cdr (BioCreative V Chemical-Disease Relation)
Target Models: HMM, MEMM, CRF
Target Schema:
  - B-DISEASE, I-DISEASE
  - B-MEDICATION, I-MEDICATION
  - B-PATIENT_NAME, I-PATIENT_NAME (Documented limitation: 0 in PubMed literature)
  - O
"""

import os
import json
import random
from collections import Counter
from huggingface_hub import hf_hub_download

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

REPO_ID = "tner/bc5cdr"

# Original label mapping in tner/bc5cdr:
# {"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}
ORIGINAL_ID2TAG = {
    0: "O",
    1: "B-Chemical",
    2: "B-Disease",
    3: "I-Disease",
    4: "I-Chemical"
}

# Semantic mapping to target project schema
SEMANTIC_TAG_MAP = {
    "O": "O",
    "B-Chemical": "B-MEDICATION",
    "I-Chemical": "I-MEDICATION",
    "B-Disease": "B-DISEASE",
    "I-Disease": "I-DISEASE"
}

def load_split(split_name: str):
    """Download and parse a JSON split from the Hugging Face repository."""
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"dataset/{split_name}.json",
        repo_type="dataset"
    )
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def audit_split(split_name: str, records: list):
    """Perform structural and semantic validation on records."""
    total_records = len(records)
    total_tokens = 0
    token_lengths = []
    tag_counter = Counter()
    entity_mention_counter = Counter()
    
    length_mismatches = 0
    empty_records = 0
    invalid_bio_transitions = 0
    
    unique_sentences = set()
    duplicate_count = 0
    
    sentences_with_med = 0
    sentences_with_dis = 0
    sentences_with_any = 0
    
    processed_records = []
    
    for r in records:
        tokens = r["tokens"]
        raw_tags = r["tags"]
        n_toks = len(tokens)
        total_tokens += n_toks
        token_lengths.append(n_toks)
        
        if n_toks == 0:
            empty_records += 1
            continue
            
        if n_toks != len(raw_tags):
            length_mismatches += 1
            
        # Map tags to standardized names
        target_tags = []
        prev_tag = "O"
        has_med = False
        has_dis = False
        
        for t_idx in raw_tags:
            orig_tag = ORIGINAL_ID2TAG.get(t_idx, "O")
            target_tag = SEMANTIC_TAG_MAP[orig_tag]
            target_tags.append(target_tag)
            tag_counter[target_tag] += 1
            
            if target_tag.startswith("B-"):
                entity_mention_counter[target_tag[2:]] += 1
                if target_tag == "B-MEDICATION":
                    has_med = True
                elif target_tag == "B-DISEASE":
                    has_dis = True
            elif target_tag.startswith("I-"):
                ent_type = target_tag[2:]
                if prev_tag not in (f"B-{ent_type}", f"I-{ent_type}"):
                    invalid_bio_transitions += 1
                    
            prev_tag = target_tag
            
        if has_med:
            sentences_with_med += 1
        if has_dis:
            sentences_with_dis += 1
        if has_med or has_dis:
            sentences_with_any += 1
            
        sent_text = " ".join(tokens)
        if sent_text in unique_sentences:
            duplicate_count += 1
        else:
            unique_sentences.add(sent_text)
            
        processed_records.append({
            "tokens": tokens,
            "ner_tags": target_tags
        })
        
    return {
        "split_name": split_name,
        "total_records": total_records,
        "unique_records": len(unique_sentences),
        "duplicates": duplicate_count,
        "total_tokens": total_tokens,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "median_tokens": sorted(token_lengths)[len(token_lengths)//2] if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "length_mismatches": length_mismatches,
        "empty_records": empty_records,
        "invalid_bio_transitions": invalid_bio_transitions,
        "tag_counts": dict(tag_counter),
        "entity_mentions": dict(entity_mention_counter),
        "sentences_with_med": sentences_with_med,
        "sentences_with_dis": sentences_with_dis,
        "sentences_with_any": sentences_with_any,
        "pct_sentences_with_any": (sentences_with_any / total_records * 100) if total_records > 0 else 0,
        "processed_records": processed_records
    }

def main():
    print("=" * 70)
    print("AUDIT & PREPARATION: tner/bc5cdr for HMM / MEMM / CRF Evaluation")
    print(f"Repository: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)
    
    splits = ["train", "valid", "test"]
    audit_results = {}
    
    for s in splits:
        print(f"\n[+] Loading and auditing split: '{s}'...")
        raw_records = load_split(s)
        res = audit_split(s, raw_records)
        audit_results[s] = res
        
        print(f"    - Sentence count: {res['total_records']}")
        print(f"    - Unique sentences: {res['unique_records']} (Duplicates: {res['duplicates']})")
        print(f"    - Token range: [{res['min_tokens']}, {res['max_tokens']}] (median: {res['median_tokens']})")
        print(f"    - Token-label alignment errors: {res['length_mismatches']}")
        print(f"    - Invalid BIO transitions: {res['invalid_bio_transitions']}")
        print(f"    - Entity mentions: {res['entity_mentions']}")
        print(f"    - Sentences with entities: {res['sentences_with_any']} ({res['pct_sentences_with_any']:.1f}%)")
        print(f"    - Sentences with MEDICATION: {res['sentences_with_med']} ({res['sentences_with_med']/res['total_records']*100:.1f}%)")
        print(f"    - Sentences with DISEASE:    {res['sentences_with_dis']} ({res['sentences_with_dis']/res['total_records']*100:.1f}%)")

    # Check cross-split sentence overlap
    train_texts = set(" ".join(r["tokens"]) for r in audit_results["train"]["processed_records"])
    valid_texts = set(" ".join(r["tokens"]) for r in audit_results["valid"]["processed_records"])
    test_texts = set(" ".join(r["tokens"]) for r in audit_results["test"]["processed_records"])
    
    print("\n[+] Cross-split sentence overlap analysis:")
    print(f"    - Train & Validation overlap: {len(train_texts.intersection(valid_texts))} sentences")
    print(f"    - Train & Test overlap:       {len(train_texts.intersection(test_texts))} sentences")
    print(f"    - Valid & Test overlap:       {len(valid_texts.intersection(test_texts))} sentences")
    
    # Save prepared dataset in standard statistical NER format (JSONL)
    out_dir = "data/bc5cdr_prepared"
    os.makedirs(out_dir, exist_ok=True)
    
    for s in splits:
        out_path = os.path.join(out_dir, f"{s}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in audit_results[s]["processed_records"]:
                f.write(json.dumps(rec) + "\n")
        print(f"[+] Saved standardized {s} split to {out_path} ({len(audit_results[s]['processed_records'])} sentences)")
        
    # Save a comprehensive audit summary report
    summary_path = os.path.join(out_dir, "dataset_audit_report.json")
    clean_summary = {
        s: {k: v for k, v in audit_results[s].items() if k != "processed_records"}
        for s in splits
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"[+] Saved audit metrics to {summary_path}")
    print("\nValidation completed successfully.")

if __name__ == "__main__":
    main()

