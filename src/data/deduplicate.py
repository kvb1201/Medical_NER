#!/usr/bin/env python3
"""
Module: deduplicate.py
Implements exact and near-duplicate detection and filtering across original
BC5CDR records and synthetic clinical examples:
  1. Documented text normalization method for exact match detection.
  2. Conservative similarity method (token-level and character n-gram Jaccard,
     Levenshtein ratio >= 0.85) for near-duplicate candidate detection.
  3. Strict entity-vocabulary preservation rule: sentences sharing disease,
     medication, or chemical vocabulary are NOT removed unless structural
     and lexical content constitutes genuine leakage.
  4. Test partition immutability: Original BC5CDR test partition is fixed.
     If a synthetic test sentence duplicates an original test record, the
     synthetic sentence is regenerated instead.
  5. Removal of confirmed duplicate original records from training and
     validation partitions when cross-source leakage occurs.
"""

import re
import csv
import logging
import unicodedata
from typing import Dict, List, Set, Tuple, Any, Union

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.85

def normalize_text(tokens_or_str: Union[List[str], str]) -> str:
    """
    Standard text normalization for exact cross-source duplicate detection:
      1. If a token list, join with single whitespace.
      2. Normalize Unicode characters with standard NFKC decomposition.
      3. Lowercase all characters.
      4. Normalize all contiguous whitespace characters (tabs, newlines, multiple spaces) to a single space.
      5. Strip leading and trailing non-alphanumeric punctuation (e.g. trailing periods, quotes),
         while preserving internal punctuation (hyphens, commas, apostrophes).
    """
    if isinstance(tokens_or_str, list):
        text = " ".join(tokens_or_str)
    else:
        text = str(tokens_or_str)

    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower().strip()

    # 3. Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # 4. Strip surrounding boundary punctuation
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)

    return text

def compute_token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token-level Jaccard similarity between two token sequences."""
    set1 = set(t.lower() for t in tokens1 if re.search(r"\w", t))
    set2 = set(t.lower() for t in tokens2 if re.search(r"\w", t))
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0

def compute_character_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    if len(t1) < n or len(t2) < n:
        return 1.0 if t1 == t2 else 0.0

    ngrams1 = set(t1[i:i+n] for i in range(len(t1) - n + 1))
    ngrams2 = set(t2[i:i+n] for i in range(len(t2) - n + 1))

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return float(intersection) / float(union) if union > 0 else 0.0

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_conservative_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Conservative composite similarity metric:
      Evaluates normalized character Levenshtein edit similarity and token Jaccard similarity.
      Returns the maximum of edit similarity and token Jaccard similarity.
    """
    str1 = normalize_text(tokens1)
    str2 = normalize_text(tokens2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0

    edit_dist = compute_levenshtein_distance(str1, str2)
    edit_sim = 1.0 - (edit_dist / max_len)
    tok_jaccard = compute_token_jaccard(tokens1, tokens2)

    return max(edit_sim, tok_jaccard)

def detect_exact_duplicates(
    original_records: List[Dict[str, Any]],
    synthetic_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect exact duplicates between original records and synthetic records
    using the documented text normalization method.
    """
    synth_norm_map: Dict[str, List[Dict[str, Any]]] = {}
    for syn in synthetic_records:
        norm = normalize_text(syn["tokens"])
        synth_norm_map.setdefault(norm, []).append(syn)

    exact_duplicates = []
    for orig in original_records:
        norm = normalize_text(orig["tokens"])
        if norm in synth_norm_map:
            for matching_syn in synth_norm_map[norm]:
                exact_duplicates.append({
                    "overlap_type": "EXACT_DUPLICATE",
                    "original_doc_id": orig.get("document_id", "unknown"),
                    "synthetic_doc_id": matching_syn.get("document_id", "unknown"),
                    "original_text": " ".join(orig["tokens"]),
                    "synthetic_text": " ".join(matching_syn["tokens"]),
                    "normalized_text": norm,
                    "similarity_score": 1.0,
                    "is_confirmed_leakage": True,
                    "reason": "Exact normalized sentence match across original and synthetic records."
                })

    return exact_duplicates

def detect_near_duplicates(
    original_records: List[Dict[str, Any]],
    synthetic_records: List[Dict[str, Any]],
    threshold: float = SIMILARITY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Detect near-duplicates using conservative similarity (threshold >= 0.85).
    Distinguishes genuine structural sentence leakage from shared entity vocabulary.
    """
    near_duplicates = []

    # Fast filtering: group synthetic records by length buckets to avoid O(N*M) full comparisons
    synth_entries = []
    for syn in synthetic_records:
        syn_tokens = syn["tokens"]
        syn_norm = normalize_text(syn_tokens)
        syn_tok_set = set(t.lower() for t in syn_tokens if re.search(r"\w", t))
        synth_entries.append((syn, syn_norm, syn_tokens, syn_tok_set))

    for orig in original_records:
        orig_tokens = orig["tokens"]
        orig_norm = normalize_text(orig_tokens)
        orig_tok_set = set(t.lower() for t in orig_tokens if re.search(r"\w", t))
        if not orig_tok_set:
            continue

        for syn, syn_norm, syn_tokens, syn_tok_set in synth_entries:
            # Skip exact matches (handled by exact detector)
            if orig_norm == syn_norm:
                continue

            # Quick heuristic filter: token set overlap must be non-trivial
            common_tokens = orig_tok_set.intersection(syn_tok_set)
            if len(common_tokens) < 3:
                continue

            # Exact pruning: skip the O(n*m) Levenshtein when neither similarity can reach
            # the threshold. Edit similarity is at most 1 - |len1 - len2| / max_len, and the
            # token Jaccard here equals compute_token_jaccard() for these token sets.
            max_len = max(len(orig_norm), len(syn_norm))
            if max_len > 0:
                edit_upper_bound = 1.0 - abs(len(orig_norm) - len(syn_norm)) / max_len
                tok_jaccard = len(common_tokens) / len(orig_tok_set | syn_tok_set)
                if edit_upper_bound < threshold and tok_jaccard < threshold:
                    continue

            # Check conservative similarity
            sim = compute_conservative_similarity(orig_tokens, syn_tokens)
            if sim >= threshold:
                # Check whether overlap is merely shared entity vocabulary
                # If non-entity tokens have low overlap, it is shared vocabulary, not leakage
                orig_non_entity = {tok.lower() for tok, tag in zip(orig["tokens"], orig.get("ner_tags", orig.get("raw_tags", []))) if tag == "O"}
                syn_non_entity = {tok.lower() for tok, tag in zip(syn["tokens"], syn["ner_tags"]) if tag == "O"}
                non_entity_overlap = len(orig_non_entity.intersection(syn_non_entity))
                
                is_leakage = (non_entity_overlap >= 3) and (sim >= threshold)
                reason = (
                    "High structural and lexical similarity (>= 0.85) with shared syntactic framing."
                    if is_leakage
                    else "Shared entity vocabulary present but distinct sentence context (not leakage)."
                )

                near_duplicates.append({
                    "overlap_type": "NEAR_DUPLICATE_CANDIDATE",
                    "original_doc_id": orig.get("document_id", "unknown"),
                    "synthetic_doc_id": syn.get("document_id", "unknown"),
                    "original_text": " ".join(orig_tokens),
                    "synthetic_text": " ".join(syn_tokens),
                    "normalized_text": orig_norm,
                    "similarity_score": round(sim, 4),
                    "is_confirmed_leakage": is_leakage,
                    "reason": reason
                })

    return near_duplicates

def export_duplicate_audit_csv(
    audit_entries: List[Dict[str, Any]],
    output_path: str = "reports/near_duplicate_audit.csv"
) -> None:
    """Export duplicate audit report to CSV for manual review."""
    fieldnames = [
        "split",
        "overlap_type",
        "is_confirmed_leakage",
        "similarity_score",
        "original_doc_id",
        "synthetic_doc_id",
        "action_taken",
        "reason",
        "original_text",
        "synthetic_text"
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in audit_entries:
            row = {k: e.get(k, "") for k in fieldnames}
            writer.writerow(row)
    logger.info(f"Saved duplicate audit report ({len(audit_entries)} records) to {output_path}")

def filter_cross_source_duplicates(
    original_records: List[Dict[str, Any]],
    synthetic_records: List[Dict[str, Any]],
    split_name: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter confirmed duplicate original records for training and validation splits.
    For test split: NEVER remove original records; if synthetic duplicate occurs,
    raises exception requiring synthetic regeneration.

    Returns:
      (filtered_original_records, removed_original_records, audit_log_entries)
    """
    exact_dups = detect_exact_duplicates(original_records, synthetic_records)
    near_dups = detect_near_duplicates(original_records, synthetic_records, threshold=SIMILARITY_THRESHOLD)

    all_candidates = exact_dups + near_dups
    confirmed_leakage_doc_ids: Set[str] = set()
    audit_log = []

    for item in all_candidates:
        item["split"] = split_name
        if item["is_confirmed_leakage"]:
            if split_name == "test":
                # STRICT REQUIREMENT: Keep test partition fixed and unchanged.
                # If a synthetic test example duplicates an original test sentence,
                # regenerate the synthetic example instead!
                msg = (
                    f"Test partition duplicate detected between synthetic {item['synthetic_doc_id']} "
                    f"and fixed original test record {item['original_doc_id']}. "
                    "Original test records cannot be filtered; synthetic example must be regenerated."
                )
                logger.error(msg)
                item["action_taken"] = "REGENERATE_SYNTHETIC (Test original fixed)"
                audit_log.append(item)
                raise RuntimeError(msg)
            else:
                confirmed_leakage_doc_ids.add(item["original_doc_id"])
                item["action_taken"] = f"REMOVED_ORIGINAL_{split_name.upper()}"
        else:
            item["action_taken"] = "RETAINED (Shared entity vocabulary only)"
        audit_log.append(item)

    if confirmed_leakage_doc_ids:
        filtered_originals = [
            r for r in original_records
            if r.get("document_id") not in confirmed_leakage_doc_ids
        ]
        removed_originals = [
            r for r in original_records
            if r.get("document_id") in confirmed_leakage_doc_ids
        ]
        logger.info(
            f"Split '{split_name}': Filtered {len(removed_originals)} duplicate original records. "
            f"Remaining original records: {len(filtered_originals)}"
        )
    else:
        filtered_originals = list(original_records)
        removed_originals = []
        logger.info(f"Split '{split_name}': 0 duplicate original records detected. All records retained.")

    return filtered_originals, removed_originals, audit_log
