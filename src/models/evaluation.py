#!/usr/bin/env python3
"""
Module: evaluation.py
Implements entity-level (span-level) and token-level sequence labeling evaluation
for Named Entity Recognition models (HMM, MEMM, CRF).

Supports CoNLL-style exact span matching:
  - DISEASE
  - MEDICATION
  - PATIENT_NAME
  - CHEMICAL
  - Micro-average and Macro-average metrics
"""

from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import numpy as np

TARGET_ENTITY_TYPES = ["DISEASE", "MEDICATION", "PATIENT_NAME", "CHEMICAL"]

def extract_entities(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Extract entity chunks from a BIO-encoded sequence of tags.
    Returns list of tuples: (entity_type, start_token_idx, end_token_idx)
    where start is inclusive and end is exclusive.
    """
    entities = []
    current_entity = None
    start_idx = -1

    for idx, tag in enumerate(tags):
        if tag == "O":
            if current_entity is not None:
                entities.append((current_entity, start_idx, idx))
                current_entity = None
                start_idx = -1
        elif tag.startswith("B-"):
            if current_entity is not None:
                entities.append((current_entity, start_idx, idx))
            current_entity = tag[2:]
            start_idx = idx
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_entity is not None and tag_type != current_entity:
                # Incompatible continuation (e.g. B-DISEASE I-CHEMICAL): close the open
                # entity and start a new one here, exactly as an orphan I- tag does
                # (CoNLL conlleval convention), so no token is silently dropped.
                entities.append((current_entity, start_idx, idx))
                current_entity = None
            if current_entity is None:
                current_entity = tag_type
                start_idx = idx
        else:
            if current_entity is not None:
                entities.append((current_entity, start_idx, idx))
                current_entity = None
                start_idx = -1

    if current_entity is not None:
        entities.append((current_entity, start_idx, len(tags)))

    return entities

def compute_entity_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    entity_types: List[str] = None
) -> Dict[str, Any]:
    """
    Compute exact-match entity-level precision, recall, and F1 score.
    Returns per-entity metrics as well as micro- and macro-averages.
    """
    if entity_types is None:
        entity_types = TARGET_ENTITY_TYPES

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for gold_tags, pred_tags in zip(y_true, y_pred):
        gold_entities = extract_entities(gold_tags)
        pred_entities = extract_entities(pred_tags)

        gold_set = set(gold_entities)
        pred_set = set(pred_entities)

        for ent in pred_set:
            ent_type = ent[0]
            if ent in gold_set:
                tp[ent_type] += 1
            else:
                fp[ent_type] += 1

        for ent in gold_set:
            ent_type = ent[0]
            if ent not in pred_set:
                fn[ent_type] += 1

    per_entity = {}
    macro_p = []
    macro_r = []
    macro_f1 = []

    for ent_type in entity_types:
        ent_tp = tp[ent_type]
        ent_fp = fp[ent_type]
        ent_fn = fn[ent_type]
        support = ent_tp + ent_fn

        prec = float(ent_tp) / (ent_tp + ent_fp) if (ent_tp + ent_fp) > 0 else 0.0
        rec = float(ent_tp) / (ent_tp + ent_fn) if (ent_tp + ent_fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_entity[ent_type] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": ent_tp,
            "fp": ent_fp,
            "fn": ent_fn,
            "support": support
        }

    # Active benchmark classes (classes with gold support > 0 in this evaluation partition)
    active_classes = [c for c in entity_types if per_entity[c]["support"] > 0]
    eval_classes = active_classes if len(active_classes) > 0 else entity_types

    active_p = [per_entity[c]["precision"] for c in eval_classes]
    active_r = [per_entity[c]["recall"] for c in eval_classes]
    active_f1 = [per_entity[c]["f1"] for c in eval_classes]

    # Full schema classes (all target entity types, including inactive classes with support = 0)
    all_p = [per_entity[c]["precision"] for c in entity_types]
    all_r = [per_entity[c]["recall"] for c in entity_types]
    all_f1 = [per_entity[c]["f1"] for c in entity_types]

    total_tp = sum(tp[k] for k in entity_types)
    total_fp = sum(fp[k] for k in entity_types)
    total_fn = sum(fn[k] for k in entity_types)
    total_support = total_tp + total_fn

    micro_prec = float(total_tp) / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = float(total_tp) / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2.0 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "per_entity": per_entity,
        "micro": {
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "f1": round(micro_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "support": total_support
        },
        "macro": {
            "precision": round(float(np.mean(active_p)), 4),
            "recall": round(float(np.mean(active_r)), 4),
            "f1": round(float(np.mean(active_f1)), 4),
            "active_classes": eval_classes,
            "macro_all_precision": round(float(np.mean(all_p)), 4),
            "macro_all_recall": round(float(np.mean(all_r)), 4),
            "macro_all_f1": round(float(np.mean(all_f1)), 4)
        }
    }

def compute_token_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> Dict[str, Any]:
    """Compute token-level accuracy and macro-averaged metrics."""
    correct = 0
    total = 0
    per_tag_tp = defaultdict(int)
    per_tag_fp = defaultdict(int)
    per_tag_fn = defaultdict(int)
    all_tags = set()

    for gold_tags, pred_tags in zip(y_true, y_pred):
        for g, p in zip(gold_tags, pred_tags):
            total += 1
            all_tags.add(g)
            all_tags.add(p)
            if g == p:
                correct += 1
                per_tag_tp[g] += 1
            else:
                per_tag_fp[p] += 1
                per_tag_fn[g] += 1

    accuracy = float(correct) / total if total > 0 else 0.0
    return {
        "token_accuracy": round(accuracy, 4),
        "total_tokens": total,
        "correct_tokens": correct
    }

def evaluate_predictions(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> Dict[str, Any]:
    """Comprehensive evaluation returning both entity-level and token-level metrics."""
    entity_res = compute_entity_metrics(y_true, y_pred)
    token_res = compute_token_metrics(y_true, y_pred)
    return {
        "entity_metrics": entity_res,
        "token_metrics": token_res
    }

