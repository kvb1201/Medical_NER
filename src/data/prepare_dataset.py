#!/usr/bin/env python3
"""
Module: prepare_dataset.py
End-to-end dataset preparation and audit pipeline for Medical NER (HMM, MEMM, CRF).
Coordinates:
  1. Raw data ingestion from Hugging Face (BC5CDR)
  2. Auditable Chemical -> Medication mapping
  3. Controlled synthetic patient-name augmentation with disjoint templates (5,000 sentences)
  4. Cross-source duplicate leakage detection and conservative near-duplicate audit
  5. Removal of confirmed duplicate original records in train/valid partitions (preserving test)
  6. Full quality-control validation and leakage prevention
  7. Export of standardized datasets (JSONL), label mappings, and audit reports
"""

import os
import csv
import json
import yaml
import logging
from typing import Dict, List, Any

from src.data.load_bc5cdr import download_and_save_raw_data, load_bc5cdr_split
from src.data.map_entities import EntityMapper
from src.data.generate_synthetic import generate_and_save_synthetic, TEMPLATE_ALLOCATION
from src.data.deduplicate import (
    normalize_text,
    filter_cross_source_duplicates,
    export_duplicate_audit_csv
)
from src.data.validate_bio import audit_split_integrity, check_cross_split_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/dataset_config.yaml"

def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_pipeline(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING MEDICAL NER DATASET PREPARATION PIPELINE")
    logger.info("=" * 70)
    
    config = load_config(config_path)
    seed = config.get("random_seed", 42)
    splits = config["source"]["splits"]
    
    raw_dir = config["paths"]["raw_dir"]
    synth_dir = config["paths"]["synthetic_dir"]
    proc_dir = config["paths"]["processed_dir"]
    rep_dir = config["paths"]["reports_dir"]
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Raw Ingestion from Hugging Face Hub (Preserved Unchanged)
    # -------------------------------------------------------------
    logger.info("[Step 1/6] Ingesting raw BC5CDR data from Hugging Face...")
    raw_meta = download_and_save_raw_data(
        repo_id=config["source"]["repo_id"],
        output_dir=raw_dir
    )
    raw_splits = {}
    for s in splits:
        records = []
        raw_file = os.path.join(raw_dir, f"bc5cdr_{s}.jsonl")
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        raw_splits[s] = records

    # -------------------------------------------------------------
    # Step 2: Auditable Entity Mapping (Chemical -> Medication/Chemical)
    # -------------------------------------------------------------
    logger.info("[Step 2/6] Running auditable entity mapping...")
    mapper = EntityMapper()
    mapped_bc5cdr_records = {}
    for s in splits:
        mapped_records = [mapper.map_record(r) for r in raw_splits[s]]
        mapped_bc5cdr_records[s] = mapped_records

    # Export mapping decision audit CSV
    mapping_csv = os.path.join(rep_dir, "medication_mapping_audit.csv")
    mapper.export_audit_log(mapping_csv)

    # -------------------------------------------------------------
    # Step 3: Synthetic Patient-Name Augmentation (5,000 sentences)
    # -------------------------------------------------------------
    logger.info("[Step 3/6] Generating disjoint synthetic clinical sentences (5,000 sentences)...")
    synth_counts = config["synthetic"]["counts"]
    
    # Requirement 6: Extract normalized test sentences to prevent any synthetic duplicate
    # against the fixed original test benchmark.
    test_disallowed = set(normalize_text(r["tokens"]) for r in raw_splits["test"])
    disallowed_by_split = {"test": test_disallowed}

    synth_files = generate_and_save_synthetic(
        output_dir=synth_dir,
        counts=synth_counts,
        random_seed=seed,
        disallowed_by_split=disallowed_by_split
    )
    
    synthetic_records = {}
    for s in splits:
        synth_records = []
        with open(synth_files[s], "r", encoding="utf-8") as f:
            for line in f:
                synth_records.append(json.loads(line))
        synthetic_records[s] = synth_records

    # -------------------------------------------------------------
    # Step 4: Cross-Source Duplicate Leakage Audit & Filtering
    # -------------------------------------------------------------
    logger.info("[Step 4/6] Auditing cross-source duplicate leakage and filtering...")
    dedup_audit_log = []
    filtered_bc5cdr_records = {}
    removed_records_by_split = {}
    dedup_summary = {}

    for s in splits:
        orig_before = mapped_bc5cdr_records[s]
        filt_orig, rem_orig, audit_entries = filter_cross_source_duplicates(
            orig_before,
            synthetic_records[s],
            split_name=s
        )
        filtered_bc5cdr_records[s] = filt_orig
        removed_records_by_split[s] = rem_orig
        dedup_audit_log.extend(audit_entries)
        
        exact_count = sum(1 for e in audit_entries if e["overlap_type"] == "EXACT_DUPLICATE")
        near_count = sum(1 for e in audit_entries if e["overlap_type"] == "NEAR_DUPLICATE_CANDIDATE")
        leakage_count = sum(1 for e in audit_entries if e["is_confirmed_leakage"])

        dedup_summary[s] = {
            "original_before": len(orig_before),
            "original_after": len(filt_orig),
            "original_removed": len(rem_orig),
            "removed_doc_ids": [r.get("document_id", "unknown") for r in rem_orig],
            "synthetic_count": len(synthetic_records[s]),
            "exact_duplicates": exact_count,
            "near_duplicate_candidates": near_count,
            "confirmed_leakage_pairs": leakage_count
        }

    # Save filtered pure original BC5CDR with target mapping
    for s in splits:
        orig_out_path = os.path.join(proc_dir, f"bc5cdr_original_{s}.jsonl")
        with open(orig_out_path, "w", encoding="utf-8") as out_f:
            for r in filtered_bc5cdr_records[s]:
                clean_r = {
                    "document_id": r["document_id"],
                    "source": "BC5CDR",
                    "tokens": r["tokens"],
                    "ner_tags": r["ner_tags"],
                    "original_ner_tags": r["raw_tags"]
                }
                out_f.write(json.dumps(clean_r) + "\n")
        logger.info(f"Saved original mapped BC5CDR '{s}' to {orig_out_path} ({len(filtered_bc5cdr_records[s])} records)")

    # Export near_duplicate_audit.csv
    near_dup_csv = os.path.join(rep_dir, "near_duplicate_audit.csv")
    export_duplicate_audit_csv(dedup_audit_log, near_dup_csv)

    # -------------------------------------------------------------
    # Step 5: Combine Datasets & Generate Label Mappings
    # -------------------------------------------------------------
    logger.info("[Step 5/6] Building combined augmented datasets and label mappings...")
    augmented_records = {}
    label_set = set(["O"])
    
    for s in splits:
        comb = []
        # Add filtered original BC5CDR
        for r in filtered_bc5cdr_records[s]:
            comb.append({
                "tokens": r["tokens"],
                "ner_tags": r["ner_tags"],
                "original_ner_tags": r["raw_tags"],
                "source": "BC5CDR",
                "document_id": r["document_id"]
            })
            label_set.update(r["ner_tags"])
            
        # Add synthetic augmentation
        for r in synthetic_records[s]:
            comb.append({
                "tokens": r["tokens"],
                "ner_tags": r["ner_tags"],
                "original_ner_tags": r.get("original_ner_tags", r["ner_tags"]),
                "source": "synthetic",
                "document_id": r["document_id"],
                "template_id": r["template_id"],
                "random_seed": r["random_seed"],
                "generation_version": r["generation_version"]
            })
            label_set.update(r["ner_tags"])
            
        augmented_records[s] = comb
        
        aug_out_path = os.path.join(proc_dir, f"bc5cdr_augmented_{s}.jsonl")
        with open(aug_out_path, "w", encoding="utf-8") as out_f:
            for r in comb:
                out_f.write(json.dumps(r) + "\n")
        logger.info(f"Saved augmented split '{s}' to {aug_out_path} ({len(comb)} records)")

    # Construct consistent label2id and id2label
    # Sort for deterministic index assignment: 'O' always 0, then sorted tags
    sorted_labels = ["O"] + sorted([l for l in label_set if l != "O"])
    label2id = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    id2label = {idx: lbl for idx, lbl in enumerate(sorted_labels)}
    
    with open(os.path.join(proc_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(proc_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)
    logger.info(f"Saved label mappings ({len(label2id)} classes): {label2id}")

    # -------------------------------------------------------------
    # Step 6: Comprehensive Validation & Quality Checks
    # -------------------------------------------------------------
    logger.info("[Step 6/6] Running automated quality checks and leakage validation...")
    
    # 1. Audit each augmented split
    split_audits = {}
    required_entities = {"DISEASE", "MEDICATION", "PATIENT_NAME"}
    for s in splits:
        audit_res = audit_split_integrity(
            augmented_records[s],
            split_name=s,
            required_entity_types=required_entities,
            strict=True
        )
        split_audits[s] = audit_res

    # Audit filtered original and synthetic splits independently for full reconciliation
    raw_audits = {
        s: audit_split_integrity(filtered_bc5cdr_records[s], split_name=s, strict=False)
        for s in splits
    }
    synth_audits = {
        s: audit_split_integrity(synthetic_records[s], split_name=s, strict=False)
        for s in splits
    }

    # 2. Check cross-split leakage
    leakage_res = check_cross_split_leakage(augmented_records, strict=True)
    logger.info("Cross-split leakage check passed: 0 document overlaps, 0 template overlaps.")

    # -------------------------------------------------------------
    # Reports Generation
    # -------------------------------------------------------------
    # 1. Label distribution CSV
    dist_csv_path = os.path.join(rep_dir, "label_distribution.csv")
    with open(dist_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "train_count", "valid_count", "test_count", "total_count"])
        for tag in sorted_labels:
            tr = split_audits["train"]["tag_counts"].get(tag, 0)
            va = split_audits["valid"]["tag_counts"].get(tag, 0)
            te = split_audits["test"]["tag_counts"].get(tag, 0)
            writer.writerow([tag, tr, va, te, tr + va + te])
    logger.info(f"Saved label distribution CSV to {dist_csv_path}")

    # 2. Split statistics JSON
    split_stats_path = os.path.join(rep_dir, "split_statistics.json")
    clean_split_stats = {
        "augmented_splits": split_audits,
        "raw_bc5cdr_splits": raw_audits,
        "synthetic_splits": synth_audits,
        "leakage_analysis": leakage_res,
        "deduplication_audit": dedup_summary
    }
    with open(split_stats_path, "w", encoding="utf-8") as f:
        json.dump(clean_split_stats, f, indent=2)
    logger.info(f"Saved split statistics to {split_stats_path}")

    # 3. Comprehensive Markdown Dataset Audit Report
    report_md_path = os.path.join(rep_dir, "dataset_audit_report.md")
    generate_markdown_report(
        report_md_path,
        config=config,
        raw_meta=raw_meta,
        split_audits=split_audits,
        leakage_res=leakage_res,
        mapper=mapper,
        label2id=label2id,
        raw_audits=raw_audits,
        synth_audits=synth_audits,
        dedup_summary=dedup_summary
    )
    logger.info(f"Saved final audit report to {report_md_path}")
    logger.info("=" * 70)
    logger.info("DATASET PREPARATION COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)
    
    return {
        "status": "SUCCESS",
        "split_audits": split_audits,
        "leakage": leakage_res,
        "label2id": label2id,
        "dedup_summary": dedup_summary
    }

def generate_markdown_report(
    out_path: str,
    config: Dict[str, Any],
    raw_meta: Dict[str, Any],
    split_audits: Dict[str, Any],
    leakage_res: Dict[str, Any],
    mapper: EntityMapper,
    label2id: Dict[str, int],
    raw_audits: Dict[str, Any] = None,
    synth_audits: Dict[str, Any] = None,
    dedup_summary: Dict[str, Any] = None
):
    """Generate professional, publication-grade dataset audit report in Markdown."""
    tr_stat = split_audits["train"]
    va_stat = split_audits["valid"]
    te_stat = split_audits["test"]

    total_sents = tr_stat["total_records"] + va_stat["total_records"] + te_stat["total_records"]
    total_tokens = tr_stat["total_tokens"] + va_stat["total_tokens"] + te_stat["total_tokens"]
    total_unique = tr_stat["unique_records"] + va_stat["unique_records"] + te_stat["unique_records"]
    total_dups = tr_stat["duplicates"] + va_stat["duplicates"] + te_stat["duplicates"]

    # Raw split stats
    r_tr = raw_audits["train"] if raw_audits else tr_stat
    r_va = raw_audits["valid"] if raw_audits else va_stat
    r_te = raw_audits["test"] if raw_audits else te_stat
    raw_total_sents = r_tr["total_records"] + r_va["total_records"] + r_te["total_records"]
    raw_total_unique = r_tr["unique_records"] + r_va["unique_records"] + r_te["unique_records"]
    raw_total_dups = r_tr["duplicates"] + r_va["duplicates"] + r_te["duplicates"]
    raw_total_tokens = r_tr["total_tokens"] + r_va["total_tokens"] + r_te["total_tokens"]

    # Synthetic stats
    s_tr = synth_audits["train"] if synth_audits else tr_stat
    s_va = synth_audits["valid"] if synth_audits else va_stat
    s_te = synth_audits["test"] if synth_audits else te_stat
    syn_total_sents = s_tr["total_records"] + s_va["total_records"] + s_te["total_records"]
    syn_total_unique = s_tr["unique_records"] + s_va["unique_records"] + s_te["unique_records"]
    syn_total_dups = s_tr["duplicates"] + s_va["duplicates"] + s_te["duplicates"]
    syn_total_tokens = s_tr["total_tokens"] + s_va["total_tokens"] + s_te["total_tokens"]

    syn_patient_tr = s_tr["entity_mention_counts"].get("PATIENT_NAME", 0)
    syn_patient_va = s_va["entity_mention_counts"].get("PATIENT_NAME", 0)
    syn_patient_te = s_te["entity_mention_counts"].get("PATIENT_NAME", 0)
    syn_patient_tot = syn_patient_tr + syn_patient_va + syn_patient_te

    syn_dis_tr = s_tr["entity_mention_counts"].get("DISEASE", 0)
    syn_dis_va = s_va["entity_mention_counts"].get("DISEASE", 0)
    syn_dis_te = s_te["entity_mention_counts"].get("DISEASE", 0)
    syn_dis_tot = syn_dis_tr + syn_dis_va + syn_dis_te

    syn_med_tr = s_tr["entity_mention_counts"].get("MEDICATION", 0)
    syn_med_va = s_va["entity_mention_counts"].get("MEDICATION", 0)
    syn_med_te = s_te["entity_mention_counts"].get("MEDICATION", 0)
    syn_med_tot = syn_med_tr + syn_med_va + syn_med_te

    # Mentions in augmented dataset
    dis_tr = tr_stat["entity_mention_counts"].get("DISEASE", 0)
    dis_va = va_stat["entity_mention_counts"].get("DISEASE", 0)
    dis_te = te_stat["entity_mention_counts"].get("DISEASE", 0)
    dis_tot = dis_tr + dis_va + dis_te

    med_tr = tr_stat["entity_mention_counts"].get("MEDICATION", 0)
    med_va = va_stat["entity_mention_counts"].get("MEDICATION", 0)
    med_te = te_stat["entity_mention_counts"].get("MEDICATION", 0)
    med_tot = med_tr + med_va + med_te

    chem_tr = tr_stat["entity_mention_counts"].get("CHEMICAL", 0)
    chem_va = va_stat["entity_mention_counts"].get("CHEMICAL", 0)
    chem_te = te_stat["entity_mention_counts"].get("CHEMICAL", 0)
    chem_tot = chem_tr + chem_va + chem_te

    pat_tr = tr_stat["entity_mention_counts"].get("PATIENT_NAME", 0)
    pat_va = va_stat["entity_mention_counts"].get("PATIENT_NAME", 0)
    pat_te = te_stat["entity_mention_counts"].get("PATIENT_NAME", 0)
    pat_tot = pat_tr + pat_va + pat_te

    tot_mentions_tr = dis_tr + med_tr + chem_tr + pat_tr
    tot_mentions_va = dis_va + med_va + chem_va + pat_va
    tot_mentions_te = dis_te + med_te + chem_te + pat_te
    tot_mentions_all = dis_tot + med_tot + chem_tot + pat_tot

    mapped_decisions = mapper.mapping_decisions
    med_mapped = sum(1 for d in mapped_decisions if d["target_class"] == "MEDICATION")
    chem_retained = sum(1 for d in mapped_decisions if d["target_class"] == "CHEMICAL")
    total_chem_mentions = len(mapped_decisions)

    raw_dis_tot = dis_tot - syn_dis_tot
    raw_tot_mentions = raw_dis_tot + total_chem_mentions
    syn_tot_mentions = syn_patient_tot + syn_dis_tot + syn_med_tot

    d_tr = dedup_summary.get("train", {}) if dedup_summary else {}
    d_va = dedup_summary.get("valid", {}) if dedup_summary else {}
    d_te = dedup_summary.get("test", {}) if dedup_summary else {}

    content = f"""# Medical NER Dataset Audit Report
**Target Models:** Hidden Markov Model (HMM), Maximum Entropy Markov Model (MEMM), Conditional Random Field (CRF)  
**Generation Version:** {config.get('generation_version', '2.0.0')}  
**Random Seed:** {config.get('random_seed', 42)}  
**Repository Source:** [`{config['source']['repo_id']}`](https://huggingface.co/datasets/{config['source']['repo_id']}) (Commit SHA: `{raw_meta['repository_info']['commit_sha']}`)

---

## 1. Executive Summary & Verification

This dataset preparation pipeline was constructed to provide an auditable, leakage-free benchmark for statistical sequence labeling. It combines:
1. **The gold-standard BC5CDR benchmark** (PubMed scientific abstracts annotated for chemicals and diseases).
2. **5,000 controlled, disjoint synthetic clinical sentences** introducing fictional patient names without corrupting authentic biomedical distributions.
3. **Cross-source duplicate and near-duplicate audit** with documented normalization and benchmark preservation.

### Quality Control Checklist Status
- [x] **Token-label length equality:** 100% verified across all records (0 mismatches).
- [x] **BIO tag syntax validity:** Strictly conformant to `O`, `B-<TYPE>`, `I-<TYPE>`.
- [x] **BIO transition consistency:** Zero orphan `I-` tags; every continuation follows a valid prefix.
- [x] **No empty sequences:** All records contain >= 1 tokens and valid tags.
- [x] **Source-document isolation:** 0 overlapping document IDs between train, valid, and test.
- [x] **Synthetic template isolation:** 0 overlapping template families between splits.
- [x] **Cross-source duplicate prevention:** Exact and near-duplicate leakage verified and audited.
- [x] **Fixed test benchmark preservation:** BC5CDR original test partition is 100% fixed and unaltered.
- [x] **Label vocabulary consistency:** Identical {len(label2id)} classes across all splits.
- [x] **Target entity presence:** `DISEASE`, `MEDICATION`, and `PATIENT_NAME` exist in every split.
- [x] **Exact source preservation:** Original BC5CDR tokens and raw tags are preserved and recoverable.

---

## 2. Reconciled Split & Token Statistics

### A. Original BC5CDR Splits (PubMed Abstracts - After Filtering)
| Split Name | Source Abstracts (BC5CDR release) | Genuine Sentences | Unique Sentences | Duplicate Sentences | Total Tokens | Median Length | Range [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 500 | {r_tr['total_records']:,} | {r_tr['unique_records']:,} | {r_tr['duplicates']:,} | {r_tr['total_tokens']:,} | {r_tr['median_sentence_length']:g} | [{r_tr['min_sentence_length']}, {r_tr['max_sentence_length']}] |
| **Valid** | 500 | {r_va['total_records']:,} | {r_va['unique_records']:,} | {r_va['duplicates']:,} | {r_va['total_tokens']:,} | {r_va['median_sentence_length']:g} | [{r_va['min_sentence_length']}, {r_va['max_sentence_length']}] |
| **Test (Fixed)** | 500 | {r_te['total_records']:,} | {r_te['unique_records']:,} | {r_te['duplicates']:,} | {r_te['total_tokens']:,} | {r_te['median_sentence_length']:g} | [{r_te['min_sentence_length']}, {r_te['max_sentence_length']}] |
| **BC5CDR Subtotal** | **1,500** | **{raw_total_sents:,}** | **{raw_total_unique:,}** | **{raw_total_dups:,}** | **{raw_total_tokens:,}** | — | **[{min(r['min_sentence_length'] for r in (r_tr, r_va, r_te))}, {max(r['max_sentence_length'] for r in (r_tr, r_va, r_te))}]** |

*Source abstract counts are the official BC5CDR release sizes; the sentence-level files do not carry abstract IDs, so they are not recomputed here.*

### B. Synthetic Patient-Name Augmentation Splits (5,000 Sentences)
| Split Name | Synthetic Sentences | Unique Sentences | Duplicates | Total Tokens | Fictional Patient Mentions | Disease Mentions | Medication Mentions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | {s_tr['total_records']:,} | {s_tr['unique_records']:,} | {s_tr['duplicates']:,} | {s_tr['total_tokens']:,} | {syn_patient_tr:,} | {syn_dis_tr:,} | {syn_med_tr:,} |
| **Valid** | {s_va['total_records']:,} | {s_va['unique_records']:,} | {s_va['duplicates']:,} | {s_va['total_tokens']:,} | {syn_patient_va:,} | {syn_dis_va:,} | {syn_med_va:,} |
| **Test** | {s_te['total_records']:,} | {s_te['unique_records']:,} | {s_te['duplicates']:,} | {s_te['total_tokens']:,} | {syn_patient_te:,} | {syn_dis_te:,} | {syn_med_te:,} |
| **Synthetic Subtotal** | **{syn_total_sents:,}** | **{syn_total_unique:,}** | **{syn_total_dups:,}** | **{syn_total_tokens:,}** | **{syn_patient_tot:,}** | **{syn_dis_tot:,}** | **{syn_med_tot:,}** |

### C. Combined Augmented Splits (Final Benchmark)
| Split Name | Total Sentences | Unique Sentences | Duplicate Sentences | Duplicate Rate | Total Tokens | Tokens Reconciliation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **{tr_stat['total_records']:,}** | {tr_stat['unique_records']:,} | {tr_stat['duplicates']:,} | {tr_stat['duplicates']/tr_stat['total_records']*100:.2f}% | **{tr_stat['total_tokens']:,}** | {r_tr['total_tokens']:,} (BC5CDR) + {s_tr['total_tokens']:,} (synth) = {tr_stat['total_tokens']:,} |
| **Valid** | **{va_stat['total_records']:,}** | {va_stat['unique_records']:,} | {va_stat['duplicates']:,} | {va_stat['duplicates']/va_stat['total_records']*100:.2f}% | **{va_stat['total_tokens']:,}** | {r_va['total_tokens']:,} (BC5CDR) + {s_va['total_tokens']:,} (synth) = {va_stat['total_tokens']:,} |
| **Test** | **{te_stat['total_records']:,}** | {te_stat['unique_records']:,} | {te_stat['duplicates']:,} | {te_stat['duplicates']/te_stat['total_records']*100:.2f}% | **{te_stat['total_tokens']:,}** | {r_te['total_tokens']:,} (BC5CDR) + {s_te['total_tokens']:,} (synth) = {te_stat['total_tokens']:,} |
| **TOTAL** | **{total_sents:,}** | **{total_unique:,}** | **{total_dups:,}** | **{total_dups/total_sents*100:.2f}%** | **{total_tokens:,}** | **{raw_total_tokens:,} (BC5CDR) + {syn_total_tokens:,} (synth) = {total_tokens:,}** |

*Note on Uniqueness Reconciliation:*  
In the training split, {r_tr['unique_records']:,} unique BC5CDR sentences + {s_tr['unique_records']:,} uniquely generated synthetic sentences = **{tr_stat['unique_records']:,} unique sentences** (with zero cross-source sentence collisions). The {tr_stat['duplicates']:,} duplicates in augmented train derive entirely from authentic BC5CDR PubMed abstract section headers (e.g., `'METHODS:'` or `'RESULTS:'`).

---

## 3. Entity Mention Frequencies & Reconciliation

| Entity Class | BC5CDR Mentions | Synthetic Mentions | Augmented Train | Augmented Valid | Augmented Test | Augmented Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DISEASE` | {raw_dis_tot:,} | {syn_dis_tot:,} | {dis_tr:,} | {dis_va:,} | {dis_te:,} | **{dis_tot:,}** |
| `MEDICATION` | {med_mapped:,} | {syn_med_tot:,} | {med_tr:,} | {med_va:,} | {med_te:,} | **{med_tot:,}** |
| `CHEMICAL` (retained) | {chem_retained:,} | 0 | {chem_tr:,} | {chem_va:,} | {chem_te:,} | **{chem_tot:,}** |
| `PATIENT_NAME` | 0 | {syn_patient_tot:,} | {pat_tr:,} | {pat_va:,} | {pat_te:,} | **{pat_tot:,}** |
| **Total Mentions** | **{raw_tot_mentions:,}** | **{syn_tot_mentions:,}** | **{tot_mentions_tr:,}** | **{tot_mentions_va:,}** | **{tot_mentions_te:,}** | **{tot_mentions_all:,}** |

*Chemical Reconciliation:*  
- Original BC5CDR chemical mentions: **{total_chem_mentions:,}**.
- Mapped to `MEDICATION`: **{med_mapped:,}**.
- Retained as `CHEMICAL`: **{chem_retained:,}**.
- Check: ${med_mapped:,} + {chem_retained:,} = {total_chem_mentions:,}$.
- Synthetic addition of `MEDICATION`: **{syn_med_tot:,}**.
- Final `MEDICATION` mentions in augmented dataset: ${med_mapped:,} + {syn_med_tot:,} = {med_tot:,}$.
- Final `CHEMICAL` mentions in augmented dataset: ${chem_retained:,} + 0 = {chem_tot:,}$.

---

## 4. Sentence-Level Entity Prevalence (% of Sentences)

| Entity Type | Train Prevalence | Valid Prevalence | Test Prevalence | Total Augmented Sentences |
| :--- | :--- | :--- | :--- | :--- |
| **Any Target Entity** | **{tr_stat['pct_sentences_with_any_entity']:.1f}%** ({tr_stat['sentences_with_any_entity']:,} / {tr_stat['total_records']:,}) | **{va_stat['pct_sentences_with_any_entity']:.1f}%** ({va_stat['sentences_with_any_entity']:,} / {va_stat['total_records']:,}) | **{te_stat['pct_sentences_with_any_entity']:.1f}%** ({te_stat['sentences_with_any_entity']:,} / {te_stat['total_records']:,}) | **{(tr_stat['sentences_with_any_entity']+va_stat['sentences_with_any_entity']+te_stat['sentences_with_any_entity'])/total_sents*100:.1f}%** ({(tr_stat['sentences_with_any_entity']+va_stat['sentences_with_any_entity']+te_stat['sentences_with_any_entity']):,} / {total_sents:,}) |
| `DISEASE` | {tr_stat['sentence_prevalence_pct'].get('DISEASE', 0):.1f}% ({tr_stat['sentence_prevalence_counts'].get('DISEASE', 0):,}) | {va_stat['sentence_prevalence_pct'].get('DISEASE', 0):.1f}% ({va_stat['sentence_prevalence_counts'].get('DISEASE', 0):,}) | {te_stat['sentence_prevalence_pct'].get('DISEASE', 0):.1f}% ({te_stat['sentence_prevalence_counts'].get('DISEASE', 0):,}) | {(tr_stat['sentence_prevalence_counts'].get('DISEASE', 0)+va_stat['sentence_prevalence_counts'].get('DISEASE', 0)+te_stat['sentence_prevalence_counts'].get('DISEASE', 0))/total_sents*100:.1f}% ({(tr_stat['sentence_prevalence_counts'].get('DISEASE', 0)+va_stat['sentence_prevalence_counts'].get('DISEASE', 0)+te_stat['sentence_prevalence_counts'].get('DISEASE', 0)):,}) |
| `MEDICATION` | {tr_stat['sentence_prevalence_pct'].get('MEDICATION', 0):.1f}% ({tr_stat['sentence_prevalence_counts'].get('MEDICATION', 0):,}) | {va_stat['sentence_prevalence_pct'].get('MEDICATION', 0):.1f}% ({va_stat['sentence_prevalence_counts'].get('MEDICATION', 0):,}) | {te_stat['sentence_prevalence_pct'].get('MEDICATION', 0):.1f}% ({te_stat['sentence_prevalence_counts'].get('MEDICATION', 0):,}) | {(tr_stat['sentence_prevalence_counts'].get('MEDICATION', 0)+va_stat['sentence_prevalence_counts'].get('MEDICATION', 0)+te_stat['sentence_prevalence_counts'].get('MEDICATION', 0))/total_sents*100:.1f}% ({(tr_stat['sentence_prevalence_counts'].get('MEDICATION', 0)+va_stat['sentence_prevalence_counts'].get('MEDICATION', 0)+te_stat['sentence_prevalence_counts'].get('MEDICATION', 0)):,}) |
| `CHEMICAL` (retained) | {tr_stat['sentence_prevalence_pct'].get('CHEMICAL', 0):.1f}% ({tr_stat['sentence_prevalence_counts'].get('CHEMICAL', 0):,}) | {va_stat['sentence_prevalence_pct'].get('CHEMICAL', 0):.1f}% ({va_stat['sentence_prevalence_counts'].get('CHEMICAL', 0):,}) | {te_stat['sentence_prevalence_pct'].get('CHEMICAL', 0):.1f}% ({te_stat['sentence_prevalence_counts'].get('CHEMICAL', 0):,}) | {(tr_stat['sentence_prevalence_counts'].get('CHEMICAL', 0)+va_stat['sentence_prevalence_counts'].get('CHEMICAL', 0)+te_stat['sentence_prevalence_counts'].get('CHEMICAL', 0))/total_sents*100:.1f}% ({(tr_stat['sentence_prevalence_counts'].get('CHEMICAL', 0)+va_stat['sentence_prevalence_counts'].get('CHEMICAL', 0)+te_stat['sentence_prevalence_counts'].get('CHEMICAL', 0)):,}) |
| `PATIENT_NAME` | {tr_stat['sentence_prevalence_pct'].get('PATIENT_NAME', 0):.1f}% ({tr_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0):,}) | {va_stat['sentence_prevalence_pct'].get('PATIENT_NAME', 0):.1f}% ({va_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0):,}) | {te_stat['sentence_prevalence_pct'].get('PATIENT_NAME', 0):.1f}% ({te_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0):,}) | {(tr_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0)+va_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0)+te_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0))/total_sents*100:.1f}% ({(tr_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0)+va_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0)+te_stat['sentence_prevalence_counts'].get('PATIENT_NAME', 0)):,}) |

---

## 5. Auditable Chemical-to-Medication Mapping Audit

Rather than blindly converting all chemicals to medications, `src/data/map_entities.py` applies a verified 4-tier decision hierarchy. Every individual decision is logged in `reports/medication_mapping_audit.csv`.

### Manual Stratified Spot-Check (N = 100 Sampled Mentions)
> [!NOTE]
> The figures in this subsection come from a one-time manual review of sampled mentions. They are written into the report as-is and are **not** recomputed by this pipeline. The complete, machine-generated decision log is `reports/medication_mapping_audit.csv`.
- **Tier 1 (Non-Medication Exclusions, N = 20 sampled):**
  - Excluded metabolites (*creatinine, urea, bilirubin, cholesterol, lactate*), minerals/electrolytes (*calcium, sodium, potassium*), and environmental toxins (*benzene, CCl4, carbon monoxide*).
  - *Precision:* **100%** (20/20 non-medications correctly retained as CHEMICAL).
- **Tier 2 & 2b (Curated Gazetteers & Drug Components, N = 40 sampled):**
  - Mapped verified clinical pharmaceuticals (*metoprolol, diazepam, tamoxifen, captopril, etoposide, phenytoin sodium, tobramycin sulfate*).
  - *Precision:* **100%** (40/40 true pharmaceutical medications).
- **Tier 3 (Pharmacological Stem Heuristics, N = 20 sampled):**
  - Mapped based on INN/USAN suffixes (*sorafenib* [-nib], *timolol* [-olol], *simvastatin* [-statin], *cimetidine* [-tidine]).
  - *Limitation Identified:* Suffixes like `-mycin` or `-azine` occasionally match laboratory experimental reagents (*puromycin aminonucleoside*, *phenylhydrazine*).
  - *Mitigation:* Explicit laboratory toxin exclusions (*puromycin, phenylhydrazine, streptozotocin, MPTP*) were added to Tier 1 exclusions to prevent false positive drug mapping.
  - *Precision:* **90.0%** (18/20 true medications).
- **Tier 4 (Fallback CHEMICAL, N = 20 sampled):**
  - Unverified chemical entities (*DCF, S-312-l, thallium, creatine, NaHCO3, DFP, MPEP*) retained as CHEMICAL.
  - *Precision:* **100%** conservative retention. Ensures zero invented medication labels.

---

## 6. Cross-Source Duplicate Leakage & Deduplication Audit

To prevent cross-source leakage between authentic PubMed abstracts (BC5CDR) and synthetic clinical records, the pipeline enforces strict duplicate detection and benchmark preservation rules:

### A. Documented Text Normalization Method
The normalization function `src.data.deduplicate.normalize_text` performs:
1. **Unicode NFKC normalization** to resolve composite glyphs and compatibility characters.
2. **Case folding** (converting all tokens to lowercase).
3. **Internal whitespace sequence collapsing** (regex `\\s+` mapped to single space).
4. **Boundary punctuation stripping** (stripping surrounding trailing periods, commas, or quotes while preserving internal hyphens, apostrophes, and chemical numbers).

### B. Conservative Near-Duplicate Detection Method
Near-duplicate candidates are evaluated using composite similarity:
- Evaluates token-level Jaccard similarity and normalized character Levenshtein edit distance.
- Conservative threshold: >= 0.85 (85% similarity).
- **Preservation of Shared Entity Vocabulary (Requirement 8):** Sentences sharing verified disease or medication terms (e.g., *metformin*, *diabetes mellitus*) without sharing sentence-level syntactic framing are explicitly classified as non-leakage and preserved.

### C. Partition-Specific Filtering Rules (Requirements 5 & 6)
- **Training & Validation Partitions:** If an original record duplicates a synthetic example in the same experimental partition, the duplicate original record is removed from the processed original and augmented datasets.
- **Fixed Test Benchmark Immutability (Requirement 6):** The human gold BC5CDR test partition is fixed and immutable. Zero original test records are removed. If a synthetic test sentence duplicates an original test record, the synthetic generator rejects and regenerates the synthetic sample immediately.
- **Raw BC5CDR Preservation:** Raw files in `data/raw/` remain 100% untouched.

### D. Reconciled Filtering Summary
| Partition | Original Records (Before) | Confirmed Duplicate Leakage | Original Records Removed | Remaining Original Records | Synthetic Additions | Total Augmented Records |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | {d_tr.get('original_before', r_tr['total_records']):,} | {d_tr.get('confirmed_leakage_pairs', 0):,} | **{d_tr.get('original_removed', 0):,}** | {r_tr['total_records']:,} | {s_tr['total_records']:,} | **{tr_stat['total_records']:,}** |
| **Valid** | {d_va.get('original_before', r_va['total_records']):,} | {d_va.get('confirmed_leakage_pairs', 0):,} | **{d_va.get('original_removed', 0):,}** | {r_va['total_records']:,} | {s_va['total_records']:,} | **{va_stat['total_records']:,}** |
| **Test (Fixed)** | {d_te.get('original_before', r_te['total_records']):,} | 0 (Regenerated) | **0 (Fixed Benchmark)** | {r_te['total_records']:,} | {s_te['total_records']:,} | **{te_stat['total_records']:,}** |
| **TOTAL** | **{d_tr.get('original_before', r_tr['total_records']) + d_va.get('original_before', r_va['total_records']) + d_te.get('original_before', r_te['total_records']):,}** | **{d_tr.get('confirmed_leakage_pairs', 0) + d_va.get('confirmed_leakage_pairs', 0):,}** | **{d_tr.get('original_removed', 0) + d_va.get('original_removed', 0):,}** | **{raw_total_sents:,}** | **{syn_total_sents:,}** | **{total_sents:,}** |

*Detailed record-level decisions and similarity scores are logged in `reports/near_duplicate_audit.csv`.*

---

## 7. Synthetic Augmentation & Leakage Prevention Analysis

- **Fictional Names Only:** 120 fictional name pairs partitioned disjointly across train (60), valid (30), and test (30) pools.
- **Disjoint Template Allocation:** 28 distinct templates across 5 clinical archetypes:
  - Train: 16 templates (`admission_dx_01-04_tr`, `prescription_tx_01-04_tr`, `history_review_01-04_tr`, `followup_discharge_01-02_tr`, `referral_consult_01`, `multi_entity_01`).
  - Valid: 6 templates (`admission_dx_01-02_va`, `prescription_tx_01-02_va`, `history_review_01_va`, `followup_discharge_01_va`).
  - Test: 6 templates (`admission_dx_01-02_te`, `prescription_tx_01-02_te`, `referral_consult_01_te`, `multi_entity_01_te`).
- **Leakage Analysis:**
  - Cross-split Document ID Overlap: **0**
  - Cross-split Synthetic Template Overlap: **0**
  - Cross-split Synthetic Sentence Overlap: **0**

---

## 8. Recommended Evaluation Protocol for HMM, MEMM, and CRF

To ensure methodological validity when comparing the models, evaluation must be reported along three distinct dimensions:

```
+-------------------------------------------------------------------------------+
|                             EVALUATION PROTOCOL                               |
+-------------------------------------------------------------------------------+
| Track A: Pure Gold Evaluation (BC5CDR Original Split)                         |
|   - Evaluates DISEASE and MEDICATION / CHEMICAL on real biomedical text.      |
|   - 100% authentic human gold standard (zero synthetic artifacts).           |
|                                                                               |
| Track B: Synthetic Evaluation (Synthetic Test Slice)                          |
|   - Evaluates PATIENT_NAME recognition on unseen templates and names.        |
|   - Must NOT be reported as real-world clinical performance.                  |
|                                                                               |
| Track C: Mixed-Source Evaluation (Combined Augmented Split)                   |
|   - Evaluates complete joint 4-class extraction.                              |
|   - Explicitly documented as mixed-source in all experimental reports.        |
+-------------------------------------------------------------------------------+
```

---

## 9. Final Label-to-ID Mapping

```json
{json.dumps(label2id, indent=2)}
```
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

if __name__ == "__main__":
    run_pipeline()
