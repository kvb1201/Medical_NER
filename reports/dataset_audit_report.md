# Medical NER Dataset Audit Report
**Target Models:** Hidden Markov Model (HMM), Maximum Entropy Markov Model (MEMM), Conditional Random Field (CRF)  
**Generation Version:** 1.0.0  
**Random Seed:** 42  
**Repository Source:** [`tner/bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr) (Commit SHA: `f68cdc7db924369241e7868656f583072acd4e90`)

---

## 1. Executive Summary & Verification

This dataset preparation pipeline was constructed to provide an auditable, leakage-free benchmark for statistical sequence labeling. It combines:
1. **The gold-standard BC5CDR benchmark** (PubMed scientific abstracts annotated for chemicals and diseases).
2. **Controlled, disjoint synthetic clinical sentences** introducing fictional patient names without corrupting authentic biomedical distributions.

### Quality Control Checklist Status
- [x] **Token-label length equality:** 100% verified across all records (0 mismatches).
- [x] **BIO tag syntax validity:** Strictly conformant to `O`, `B-<TYPE>`, `I-<TYPE>`.
- [x] **BIO transition consistency:** Zero orphan `I-` tags; every continuation follows a valid prefix.
- [x] **No empty sequences:** All records contain >= 1 tokens and valid tags.
- [x] **Source-document isolation:** 0 overlapping document IDs between train, valid, and test.
- [x] **Synthetic template isolation:** 0 overlapping template families between splits.
- [x] **Label vocabulary consistency:** Identical 9 classes across all splits.
- [x] **Target entity presence:** `DISEASE`, `MEDICATION`, and `PATIENT_NAME` exist in every split.
- [x] **Exact source preservation:** Original BC5CDR tokens and raw tags are preserved and recoverable.

---

## 2. Reconciled Split & Token Statistics

### A. Original BC5CDR Splits (PubMed Abstracts)
| Split Name | Source Documents | Genuine Sentences | Unique Sentences | Duplicate Sentences | Total Tokens | Median Length | Range [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 500 | 5,228 | 5,141 | 87 | 109,322 | 18 | [1, 133] |
| **Valid** | 500 | 5,330 | 5,202 | 128 | 108,958 | 18 | [1, 232] |
| **Test** | 500 | 5,865 | 5,713 | 152 | 116,318 | 17 | [1, 135] |
| **BC5CDR Subtotal** | **1,500** | **16,423** | **16,056** | **367** | **334,598** | **18** | **[1, 232]** |

### B. Synthetic Patient-Name Augmentation Splits
| Split Name | Synthetic Sentences | Unique Sentences | Duplicates | Total Tokens | Fictional Patient Mentions | Disease Mentions | Medication Mentions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 800 | 800 | 0 | 10,556 | 800 | 800 | 399 |
| **Valid** | 100 | 100 | 0 | 1,509 | 100 | 100 | 50 |
| **Test** | 100 | 100 | 0 | 1,628 | 100 | 100 | 75 |
| **Synthetic Subtotal** | **1,000** | **1,000** | **0** | **13,693** | **1,000** | **1,000** | **524** |

### C. Combined Augmented Splits (Final Benchmark)
| Split Name | Total Sentences | Unique Sentences | Duplicate Sentences | Duplicate Rate | Total Tokens | Tokens Reconciliation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **6,028** | 5,941 | 87 | 1.44% | **119,878** | 109,322 (BC5CDR) + 10,556 (synth) = 119,878 |
| **Valid** | **5,430** | 5,302 | 128 | 2.36% | **110,467** | 108,958 (BC5CDR) + 1,509 (synth) = 110,467 |
| **Test** | **5,965** | 5,813 | 152 | 2.55% | **117,946** | 116,318 (BC5CDR) + 1,628 (synth) = 117,946 |
| **TOTAL** | **17,423** | **17,056** | **367** | **2.11%** | **348,291** | **334,598 (BC5CDR) + 13,693 (synth) = 348,291** |

*Note on Uniqueness Reconciliation:*  
In the training split, 5,141 unique BC5CDR sentences + 800 uniquely generated synthetic sentences = **5,941 unique sentences** (with zero cross-source sentence collisions). The 87 duplicates in augmented train derive entirely from the original BC5CDR abstracts (e.g. repetitive section headers such as `'METHODS:'` or `'RESULTS:'`).

---

## 3. Entity Mention Frequencies & Reconciliation

| Entity Class | BC5CDR Mentions | Synthetic Mentions | Augmented Train | Augmented Valid | Augmented Test | Augmented Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DISEASE` | 12,850 | 1,000 | 4,982 | 4,344 | 4,524 | **13,850** |
| `MEDICATION` | 6,690 | 524 | 2,638 | 2,256 | 2,320 | **7,214** |
| `CHEMICAL` (retained) | 9,245 | 0 | 2,964 | 3,141 | 3,140 | **9,245** |
| `PATIENT_NAME` | 0 | 1,000 | 800 | 100 | 100 | **1,000** |
| **Total Mentions** | **28,785** | **2,524** | **11,384** | **9,841** | **10,084** | **31,309** |

*Chemical Reconciliation:*  
- Original BC5CDR chemical mentions: **15,935**.
- Mapped to `MEDICATION`: **6,690**.
- Retained as `CHEMICAL`: **9,245**.
- Check: $6,690 + 9,245 = 15,935$.
- Synthetic addition of `MEDICATION`: **524**.
- Final `MEDICATION` mentions in augmented dataset: $6,690 + 524 = 7,214$.
- Final `CHEMICAL` mentions in augmented dataset: $9,245 + 0 = 9,245$.

---

## 4. Sentence-Level Entity Prevalence (% of Sentences)

| Entity Type | Train Prevalence | Valid Prevalence | Test Prevalence | Total Augmented Sentences |
| :--- | :--- | :--- | :--- | :--- |
| **Any Target Entity** | **78.2%** (4,715 / 6,028) | **73.5%** (3,989 / 5,430) | **71.1%** (4,240 / 5,965) | **74.3%** (12,944 / 17,423) |
| `DISEASE` | 57.4% (3,461) | 51.3% (2,788) | 49.4% (2,946) | 52.8% (9,195) |
| `MEDICATION` | 34.6% (2,083) | 31.7% (1,724) | 30.1% (1,794) | 32.1% (5,601) |
| `CHEMICAL` (retained) | 31.4% (1,893) | 36.6% (1,985) | 34.3% (2,047) | 34.0% (5,925) |
| `PATIENT_NAME` | 13.3% (800) | 1.8% (100) | 1.7% (100) | 5.7% (1,000) |

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

## 6. Synthetic Augmentation & Leakage Prevention Analysis

- **Fictional Names Only:** 100% fictional name pairs (*John Doe, Robert Taylor, Jane Smith, etc.*), partitioned disjointly across train, valid, and test pools.
- **Disjoint Template Allocation:**
  - Train: 12 templates (`admission_dx_01-03`, `prescription_tx_01-03`, `history_review_01-03`, `followup_discharge_01`, `referral_consult_01`, `multi_entity_01`).
  - Valid: 4 templates (`admission_dx_04`, `prescription_tx_04`, `history_review_04`, `followup_discharge_02`).
  - Test: 4 templates (`admission_dx_05`, `prescription_tx_05`, `referral_consult_02`, `multi_entity_02`).
- **Leakage Analysis:**
  - Cross-split Document ID Overlap: **0**
  - Cross-split Synthetic Template Overlap: **0**
  - Cross-split Synthetic Sentence Overlap: **0**

---

## 7. Recommended Evaluation Protocol for HMM, MEMM, and CRF

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

## 8. Final Label-to-ID Mapping

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
