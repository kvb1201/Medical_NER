# Medical NER Dataset Audit Report
**Target Models:** Hidden Markov Model (HMM), Maximum Entropy Markov Model (MEMM), Conditional Random Field (CRF)  
**Generation Version:** 2.0.0  
**Random Seed:** 42  
**Repository Source:** [`tner/bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr) (Commit SHA: `f68cdc7db924369241e7868656f583072acd4e90`)

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
- [x] **Label vocabulary consistency:** Identical 9 classes across all splits.
- [x] **Target entity presence:** `DISEASE`, `MEDICATION`, and `PATIENT_NAME` exist in every split.
- [x] **Exact source preservation:** Original BC5CDR tokens and raw tags are preserved and recoverable.

---

## 2. Reconciled Split & Token Statistics

### A. Original BC5CDR Splits (PubMed Abstracts - After Filtering)
| Split Name | Source Documents | Genuine Sentences | Unique Sentences | Duplicate Sentences | Total Tokens | Median Length | Range [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 500 | 5,228 | 5,141 | 87 | 109,322 | 18 | [1, 133] |
| **Valid** | 500 | 5,330 | 5,202 | 128 | 108,958 | 18 | [1, 232] |
| **Test (Fixed)** | 500 | 5,865 | 5,713 | 152 | 116,318 | 17 | [1, 135] |
| **BC5CDR Subtotal** | **1,500** | **16,423** | **16,056** | **367** | **334,598** | **18** | **[1, 232]** |

### B. Synthetic Patient-Name Augmentation Splits (5,000 Sentences)
| Split Name | Synthetic Sentences | Unique Sentences | Duplicates | Total Tokens | Fictional Patient Mentions | Disease Mentions | Medication Mentions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 4,000 | 4,000 | 0 | 53,017 | 4,000 | 4,000 | 2,056 |
| **Valid** | 500 | 500 | 0 | 7,212 | 500 | 500 | 252 |
| **Test** | 500 | 500 | 0 | 7,638 | 500 | 500 | 336 |
| **Synthetic Subtotal** | **5,000** | **5,000** | **0** | **67,867** | **5,000** | **5,000** | **2,644** |

### C. Combined Augmented Splits (Final Benchmark)
| Split Name | Total Sentences | Unique Sentences | Duplicate Sentences | Duplicate Rate | Total Tokens | Tokens Reconciliation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **9,228** | 9,141 | 87 | 0.94% | **162,339** | 109,322 (BC5CDR) + 53,017 (synth) = 162,339 |
| **Valid** | **5,830** | 5,702 | 128 | 2.20% | **116,170** | 108,958 (BC5CDR) + 7,212 (synth) = 116,170 |
| **Test** | **6,365** | 6,213 | 152 | 2.39% | **123,956** | 116,318 (BC5CDR) + 7,638 (synth) = 123,956 |
| **TOTAL** | **21,423** | **21,056** | **367** | **1.71%** | **402,465** | **334,598 (BC5CDR) + 67,867 (synth) = 402,465** |

*Note on Uniqueness Reconciliation:*  
In the training split, 5,141 unique BC5CDR sentences + 4,000 uniquely generated synthetic sentences = **9,141 unique sentences** (with zero cross-source sentence collisions). The 87 duplicates in augmented train derive entirely from authentic BC5CDR PubMed abstract section headers (e.g., `'METHODS:'` or `'RESULTS:'`).

---

## 3. Entity Mention Frequencies & Reconciliation

| Entity Class | BC5CDR Mentions | Synthetic Mentions | Augmented Train | Augmented Valid | Augmented Test | Augmented Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DISEASE` | 12,850 | 5,000 | 8,182 | 4,744 | 4,924 | **17,850** |
| `MEDICATION` | 6,690 | 2,644 | 4,295 | 2,458 | 2,581 | **9,334** |
| `CHEMICAL` (retained) | 9,245 | 0 | 2,964 | 3,141 | 3,140 | **9,245** |
| `PATIENT_NAME` | 0 | 5,000 | 4,000 | 500 | 500 | **5,000** |
| **Total Mentions** | **28,785** | **12,644** | **19,441** | **10,843** | **11,145** | **41,429** |

*Chemical Reconciliation:*  
- Original BC5CDR chemical mentions: **15,935**.
- Mapped to `MEDICATION`: **6,690**.
- Retained as `CHEMICAL`: **9,245**.
- Check: $6,690 + 9,245 = 15,935$.
- Synthetic addition of `MEDICATION`: **2,644**.
- Final `MEDICATION` mentions in augmented dataset: $6,690 + 2,644 = 9,334$.
- Final `CHEMICAL` mentions in augmented dataset: $9,245 + 0 = 9,245$.

---

## 4. Sentence-Level Entity Prevalence (% of Sentences)

| Entity Type | Train Prevalence | Valid Prevalence | Test Prevalence | Total Augmented Sentences |
| :--- | :--- | :--- | :--- | :--- |
| **Any Target Entity** | **85.8%** (7,915 / 9,228) | **75.3%** (4,389 / 5,830) | **72.9%** (4,640 / 6,365) | **79.1%** (16,944 / 21,423) |
| `DISEASE` | 72.2% (6,661) | 54.7% (3,188) | 52.6% (3,346) | 61.6% (13,195) |
| `MEDICATION` | 40.5% (3,740) | 33.0% (1,926) | 32.3% (2,055) | 36.0% (7,721) |
| `CHEMICAL` (retained) | 20.5% (1,893) | 34.0% (1,985) | 32.2% (2,047) | 27.7% (5,925) |
| `PATIENT_NAME` | 43.3% (4,000) | 8.6% (500) | 7.9% (500) | 23.3% (5,000) |

---

## 5. Auditable Chemical-to-Medication Mapping Audit

Rather than blindly converting all chemicals to medications, `src/data/map_entities.py` applies a verified 4-tier decision hierarchy. Every individual decision is logged in `reports/medication_mapping_audit.csv`.

### Stratified Sample Audit Summary (N = 100 Sampled Mentions)
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
3. **Internal whitespace sequence collapsing** (regex `\s+` mapped to single space).
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
| **Train** | 5,228 | 0 | **0** | 5,228 | 4,000 | **9,228** |
| **Valid** | 5,330 | 0 | **0** | 5,330 | 500 | **5,830** |
| **Test (Fixed)** | 5,865 | 0 (Regenerated) | **0 (Fixed Benchmark)** | 5,865 | 500 | **6,365** |
| **TOTAL** | **16,423** | **0** | **0** | **16,423** | **5,000** | **21,423** |

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
{
  "O": 0,
  "B-CHEMICAL": 1,
  "B-DISEASE": 2,
  "B-MEDICATION": 3,
  "B-PATIENT_NAME": 4,
  "I-CHEMICAL": 5,
  "I-DISEASE": 6,
  "I-MEDICATION": 7,
  "I-PATIENT_NAME": 8
}
```
