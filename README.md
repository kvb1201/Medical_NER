# Medical Named Entity Recognition (HMM, MEMM, CRF)

A reproducible sequence labeling project comparing classical statistical models (**Hidden Markov Model**, **Maximum Entropy Markov Model**, and **Conditional Random Field**) on clinical and biomedical entities.

---

## Target Entity Schema

The project models 4 core medical entity types using BIO encoding (9 tag classes):
- `DISEASE`: Diseases, conditions, and diagnoses (mapped from BC5CDR and synthetic records)
- `MEDICATION`: Verified pharmacological drugs and therapeutics (mapped via 4-tier decision tree)
- `CHEMICAL`: Authentic non-therapeutic chemical entities and metabolites (preserved from BC5CDR)
- `PATIENT_NAME`: Fictional patient names (from controlled synthetic clinical sentences)
- `O`: Non-entity background tokens

---

## Directory Structure

```
nlp-project/
├── .gitignore                     # Comprehensive Git exclusion rules
├── README.md                      # Project documentation and reproduction guide
├── config/
│   └── dataset_config.yaml        # Dataset generation, mapping, and split configs
├── data/
│   ├── raw/                       # Downloaded BC5CDR corpus & provenance metadata
│   ├── synthetic/                 # Deterministically generated fictional clinical notes
│   └── processed/                 # Original & augmented BIO datasets, label mappings
├── reports/
│   ├── dataset_audit_report.md    # Formal publication-grade dataset audit report
│   ├── medication_mapping_audit.csv # Record-level audit log (15,935 chemical decisions)
│   ├── label_distribution.csv     # Token and mention frequencies per split
│   └── split_statistics.json      # Machine-readable split and token counts
├── src/
│   └── data/
│       ├── load_bc5cdr.py         # Hugging Face downloader & schema parser
│       ├── map_entities.py        # 4-tier auditable chemical-to-medication mapper
│       ├── generate_synthetic.py  # Disjoint clinical sentence generator
│       ├── validate_bio.py        # 12-point dataset integrity validator
│       └── prepare_dataset.py     # Main end-to-end dataset pipeline runner
└── tests/                         # Pytest test suite (21 unit and integration tests)
```

---

## Reproducing the Datasets

All raw, synthetic, and processed dataset files (`*.jsonl`, ~29 MB) can be reproduced deterministically in approximately 2 seconds using the pipeline runner:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the end-to-end data acquisition and preparation pipeline
PYTHONPATH=. python3 src/data/prepare_dataset.py
```

This pipeline automatically:
1. Ingests gold BC5CDR splits from Hugging Face Hub (`tner/bc5cdr`, commit `f68cdc7db924369241e7868656f583072acd4e90`).
2. Applies the auditable 4-tier Chemical-to-Medication classification hierarchy.
3. Generates 1,000 disjoint synthetic clinical sentences across train (800), valid (100), and test (100).
4. Assembles `bc5cdr_original_{split}.jsonl` and `bc5cdr_augmented_{split}.jsonl`.
5. Executes automated quality and leakage validation checks.
6. Regenerates the audit reports and metrics summaries in `reports/`.

---

## Running Automated Tests

Run the full pytest suite (21 automated integrity and quality checks):

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Git Tracking Note

Large generated and downloaded dataset files (`data/raw/*.jsonl`, `data/processed/*.jsonl`, `data/synthetic/*.jsonl`) are ignored via `.gitignore` to keep the repository lightweight. Dataset configurations (`config/dataset_config.yaml`), provenance metadata (`data/raw/raw_provenance_metadata.json`), label mappings (`data/processed/label2id.json`), and audit reports (`reports/`) are intentionally version-controlled.

