#!/usr/bin/env python3
"""
Module: load_bc5cdr.py
Acquires the official BC5CDR dataset from Hugging Face Hub (tner/bc5cdr),
validates its initial integrity, preserves raw tokens and gold annotations,
and saves raw split records with complete provenance metadata.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
from huggingface_hub import hf_hub_download, HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "tner/bc5cdr"
RAW_SPLIT_FILES = {
    "train": "dataset/train.json",
    "valid": "dataset/valid.json",
    "test": "dataset/test.json",
}
LABEL_FILE = "dataset/label.json"

def fetch_repo_metadata(repo_id: str = REPO_ID) -> Dict[str, Any]:
    """Retrieve repository metadata including commit SHA and file list."""
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        commit_sha = repo_info.sha
    except Exception as e:
        logger.warning(f"Could not retrieve repo commit sha via API: {e}")
        commit_sha = "unknown"
    return {
        "repo_id": repo_id,
        "commit_sha": commit_sha,
        "homepage": f"https://huggingface.co/datasets/{repo_id}",
    }

def load_raw_label_map(repo_id: str = REPO_ID) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Download and return raw integer-to-tag mapping from label.json."""
    label_path = hf_hub_download(repo_id=repo_id, filename=LABEL_FILE, repo_type="dataset")
    with open(label_path, "r", encoding="utf-8") as f:
        tag2id = json.load(f)
    id2tag = {v: k for k, v in tag2id.items()}
    logger.info(f"Loaded raw label mapping from {LABEL_FILE}: {tag2id}")
    return tag2id, id2tag

def load_bc5cdr_split(split_name: str, repo_id: str = REPO_ID, id2tag: Dict[int, str] = None) -> List[Dict[str, Any]]:
    """
    Download a single split from Hugging Face and convert integer tags to string BIO tags,
    preserving exact tokens and labels with provenance metadata.
    """
    if split_name not in RAW_SPLIT_FILES:
        raise ValueError(f"Unknown split name '{split_name}'. Expected one of {list(RAW_SPLIT_FILES.keys())}")
    
    if id2tag is None:
        _, id2tag = load_raw_label_map(repo_id)

    file_path = hf_hub_download(repo_id=repo_id, filename=RAW_SPLIT_FILES[split_name], repo_type="dataset")
    records = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tokens = item["tokens"]
            raw_int_tags = item["tags"]
            
            # Strict token-label alignment check at ingestion
            if len(tokens) != len(raw_int_tags):
                raise ValueError(
                    f"Integrity Error in {split_name} record {idx}: "
                    f"Token length ({len(tokens)}) != Tag length ({len(raw_int_tags)})"
                )
            
            string_tags = [id2tag[t] for t in raw_int_tags]
            
            record = {
                "document_id": f"BC5CDR_{split_name}_{idx:06d}",
                "sentence_id": f"{split_name}_{idx}",
                "source": "BC5CDR",
                "source_split": split_name,
                "tokens": tokens,
                "raw_tags": string_tags,
                "raw_int_tags": raw_int_tags
            }
            records.append(record)
            
    logger.info(f"Loaded split '{split_name}': {len(records)} sentences from {RAW_SPLIT_FILES[split_name]}")
    return records

def download_and_save_raw_data(output_dir: str = "data/raw", repo_id: str = REPO_ID) -> Dict[str, Any]:
    """Download all official splits, save raw records to disk, and return audit metadata."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = fetch_repo_metadata(repo_id)
    tag2id, id2tag = load_raw_label_map(repo_id)
    
    split_summaries = {}
    
    for split_name in RAW_SPLIT_FILES:
        records = load_bc5cdr_split(split_name, repo_id=repo_id, id2tag=id2tag)
        out_path = os.path.join(output_dir, f"bc5cdr_{split_name}.jsonl")
        
        total_tokens = sum(len(r["tokens"]) for r in records)
        unique_sentences = len(set(" ".join(r["tokens"]) for r in records))
        
        with open(out_path, "w", encoding="utf-8") as out_f:
            for r in records:
                out_f.write(json.dumps(r) + "\n")
                
        split_summaries[split_name] = {
            "sentence_count": len(records),
            "unique_sentence_count": unique_sentences,
            "total_tokens": total_tokens,
            "raw_file": out_path
        }
        logger.info(f"Saved raw split '{split_name}' to {out_path} ({len(records)} records)")
        
    meta_path = os.path.join(output_dir, "raw_provenance_metadata.json")
    full_meta = {
        "repository_info": metadata,
        "label_mapping": tag2id,
        "splits": split_summaries
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)
        
    logger.info(f"Saved raw dataset provenance to {meta_path}")
    return full_meta

if __name__ == "__main__":
    download_and_save_raw_data()

