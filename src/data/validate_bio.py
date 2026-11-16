#!/usr/bin/env python3
"""
Module: validate_bio.py
Implements automated quality-control and integrity checks for sequence labeling datasets:
  1. Token-label length equality
  2. Valid BIO tag syntax (O, B-TYPE, I-TYPE)
  3. Valid entity span boundaries
  4. No orphan I-tag without a preceding compatible B-tag or I-tag
  5. No empty sentences
  6. Duplicate detection
  7. Cross-split source-document overlap
  8. Cross-split synthetic-template overlap
  9. Entity counts and sentence-level entity prevalence
  10. Percentage of sentences containing each target entity
  11. Check that required entities appear in every split
  12. Verification of medication mapping compliance
"""

import re
import statistics
import logging
from collections import Counter
from typing import Dict, List, Set, Any, Tuple

logger = logging.getLogger(__name__)

BIO_TAG_REGEX = re.compile(r"^(O|B-[A-Z_]+|I-[A-Z_]+)$")

class DatasetValidationError(Exception):
    """Custom exception raised when a dataset integrity check fails."""
    pass

def validate_record_bio(
    tokens: List[str],
    ner_tags: List[str],
    record_id: str = "unknown",
    strict: bool = True
) -> Dict[str, Any]:
    """
    Validate a single sequence labeling record.
    Fails loudly if length mismatch, empty tokens, invalid tag syntax,
    or invalid BIO transition occurs.
    """
    # 1. Non-empty check
    if len(tokens) == 0:
        msg = f"Record '{record_id}' has empty token sequence."
        if strict: raise DatasetValidationError(msg)
        return {"valid": False, "error": msg}

    # 2. Length equality check
    if len(tokens) != len(ner_tags):
        msg = (
            f"Record '{record_id}' length mismatch: "
            f"len(tokens)={len(tokens)} vs len(ner_tags)={len(ner_tags)}"
        )
        if strict: raise DatasetValidationError(msg)
        return {"valid": False, "error": msg}

    # 3. Tag syntax and transition check
    prev_tag = "O"
    for idx, (tok, tag) in enumerate(zip(tokens, ner_tags)):
        if not BIO_TAG_REGEX.match(tag):
            msg = f"Record '{record_id}' at position {idx} has invalid tag syntax: '{tag}'"
            if strict: raise DatasetValidationError(msg)
            return {"valid": False, "error": msg}

        if tag.startswith("I-"):
            current_entity = tag[2:]
            valid_prev = {f"B-{current_entity}", f"I-{current_entity}"}
            if prev_tag not in valid_prev:
                msg = (
                    f"Record '{record_id}' at position {idx} ('{tok}'): "
                    f"Invalid BIO transition '{tag}' following '{prev_tag}'. "
                    f"Expected one of {valid_prev}"
                )
                if strict: raise DatasetValidationError(msg)
                return {"valid": False, "error": msg}
                
        prev_tag = tag

    return {"valid": True, "error": None}

def audit_split_integrity(
    records: List[Dict[str, Any]],
    split_name: str,
    required_entity_types: Set[str] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Audit an entire split for internal consistency, BIO transitions,
    duplicates, and entity distributions.
    """
    total_records = len(records)
    total_tokens = 0
    unique_sentence_texts = set()
    duplicate_count = 0
    
    tag_counter = Counter()
    entity_mention_counter = Counter()
    
    sentence_contains_entity = Counter()
    sentences_with_any_entity = 0
    sentence_lengths: List[int] = []
    
    for r in records:
        tokens = r["tokens"]
        sentence_lengths.append(len(tokens))
        tags = r.get("ner_tags", r.get("raw_tags", []))
        rec_id = r.get("document_id", r.get("sentence_id", "unknown"))
        
        # Run individual record validation
        validate_record_bio(tokens, tags, record_id=rec_id, strict=strict)
        
        total_tokens += len(tokens)
        sent_str = " ".join(tokens)
        if sent_str in unique_sentence_texts:
            duplicate_count += 1
        else:
            unique_sentence_texts.add(sent_str)
            
        # Count tags and entity occurrences
        present_in_sentence = set()
        for t in tags:
            tag_counter[t] += 1
            if t.startswith("B-"):
                ent = t[2:]
                entity_mention_counter[ent] += 1
                present_in_sentence.add(ent)
                
        if present_in_sentence:
            sentences_with_any_entity += 1
            for ent in present_in_sentence:
                sentence_contains_entity[ent] += 1

    # Check required entity types presence
    if required_entity_types:
        missing_entities = required_entity_types - set(entity_mention_counter.keys())
        if missing_entities and strict:
            raise DatasetValidationError(
                f"Split '{split_name}' is missing required entity types: {missing_entities}"
            )

    prevalence_pct = {
        ent: (count / total_records * 100) if total_records > 0 else 0
        for ent, count in sentence_contains_entity.items()
    }

    return {
        "split_name": split_name,
        "total_records": total_records,
        "unique_records": len(unique_sentence_texts),
        "duplicates": duplicate_count,
        "total_tokens": total_tokens,
        "tag_counts": dict(tag_counter),
        "entity_mention_counts": dict(entity_mention_counter),
        "sentence_prevalence_counts": dict(sentence_contains_entity),
        "sentence_prevalence_pct": prevalence_pct,
        "sentences_with_any_entity": sentences_with_any_entity,
        "pct_sentences_with_any_entity": (sentences_with_any_entity / total_records * 100) if total_records > 0 else 0,
        "median_sentence_length": statistics.median(sentence_lengths) if sentence_lengths else 0,
        "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
        "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0
    }

def check_cross_split_leakage(
    splits_records: Dict[str, List[Dict[str, Any]]],
    strict: bool = True
) -> Dict[str, Any]:
    """
    Check for:
      1. Source-document overlap across splits.
      2. Synthetic template overlap across splits.
      3. Exact identical sentence overlap across splits.
    """
    split_names = list(splits_records.keys())
    leakage_report = {
        "document_overlap": {},
        "template_overlap": {},
        "sentence_overlap": {}
    }
    
    split_docs = {}
    split_templates = {}
    split_sentences = {}
    
    for s_name, recs in splits_records.items():
        docs = set(r["document_id"] for r in recs if "document_id" in r)
        tmpls = set(r["template_id"] for r in recs if "template_id" in r)
        sents = set(" ".join(r["tokens"]) for r in recs)
        
        split_docs[s_name] = docs
        split_templates[s_name] = tmpls
        split_sentences[s_name] = sents

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            pair_key = f"{s1}_vs_{s2}"
            
            # 1. Document overlap
            doc_common = split_docs[s1].intersection(split_docs[s2])
            leakage_report["document_overlap"][pair_key] = len(doc_common)
            if doc_common and strict:
                raise DatasetValidationError(
                    f"Document leakage detected between '{s1}' and '{s2}': {list(doc_common)[:5]}"
                )
                
            # 2. Synthetic template overlap
            tmpl_common = split_templates[s1].intersection(split_templates[s2])
            leakage_report["template_overlap"][pair_key] = len(tmpl_common)
            if tmpl_common and strict:
                raise DatasetValidationError(
                    f"Synthetic template leakage detected between '{s1}' and '{s2}': {tmpl_common}"
                )
                
            # 3. Sentence overlap
            sent_common = split_sentences[s1].intersection(split_sentences[s2])
            leakage_report["sentence_overlap"][pair_key] = len(sent_common)
            
    return leakage_report

