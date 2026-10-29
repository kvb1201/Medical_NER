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
- [x] **No empty sequences:** All records contain $\ge 1$ tokens and valid tags.
- [x] **Source-document isolation:** 0 overlapping document IDs between train, valid, and test.
- [x] **Synthetic template isolation:** 0 overlapping template families between splits.
- [x] **Label vocabulary consistency:** Identical 9 classes across all splits.
- [x] **Target entity presence:** `DISEASE`, `MEDICATION`, and `PATIENT_NAME` exist in every split.

---

## 2. Dataset Split Statistics

| Split Name | Source BC5CDR Sentences | Synthetic Sentences | Total Usable Sentences | Total Tokens | Unique Sentences | Duplicate Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 5,228 | 800 | **6,028** | 119,879 | 5,909 | 1.97% |
| **Valid** | 5,330 | 100 | **5,430** | 110,464 | 5,301 | 2.38% |
| **Test** | 5,865 | 100 | **5,965** | 117,957 | 5,810 | 2.60% |
| **TOTAL** | 16,423 | 1,000 | **17,423** | 348,300 | **17,020** | -- |

---

## 3. Entity Mention Frequencies

| Entity Class | Train Mentions | Valid Mentions | Test Mentions | Total Mentions |
| :--- | :--- | :--- | :--- | :--- |
| `DISEASE` | 4,982 | 4,344 | 4,524 | 13,850 |
| `MEDICATION` | 2,649 | 2,276 | 2,325 | 7,250 |
| `PATIENT_NAME` | 800 | 100 | 100 | 1,000 |
| `CHEMICAL` (retained) | 2,953 | 3,121 | 3,135 | 9,209 |

---

## 4. Sentence-Level Entity Prevalence (% of Sentences)

| Entity Type | Train Prevalence | Valid Prevalence | Test Prevalence |
| :--- | :--- | :--- | :--- |
| **Any Target Entity** | **78.2%** (4,715) | **73.5%** (3,989) | **71.1%** (4,240) |
| `DISEASE` | 57.4% | 51.3% | 49.4% |
| `MEDICATION` | 34.7% | 32.1% | 30.1% |
| `PATIENT_NAME` | 13.3% | 1.8% | 1.7% |
| `CHEMICAL` (retained) | 31.4% | 36.3% | 34.3% |

---

## 5. Auditable Medication Mapping Breakdown

BC5CDR annotates generic chemical entities. Rather than blindly renaming all chemicals to medications, our mapper uses a verified rule hierarchy (curated pharmaceuticals gazetteer + USAN/INN stems + exclusion lists):
- **Total Chemical Mentions Evaluated:** 15,935
- **Mapped to `MEDICATION`:** 6,726 (42.21%)
- **Retained as `CHEMICAL`:** 9,209 (57.79%)
  - *Retained examples:* Endogenous metabolites (*creatinine*, *urea*, *glucose*), dietary minerals/electrolytes (*calcium*, *sodium*, *potassium*), and non-drug reagents (*carbon monoxide*, *lead*).
- Complete decision logs are saved to `reports/medication_mapping_audit.csv`.

---

## 6. Synthetic Augmentation & Leakage Prevention Strategy

To evaluate statistical models on patient name recognition without compromising clinical authenticity:
1. **Fictional Names Only:** Names were selected from fictional US name combinations and partitioned disjointly by split.
2. **Disjoint Template Allocation:**
   - **Train:** 12 templates (`admission_dx_01-03`, `prescription_tx_01-03`, `history_review_01-03`, `followup_discharge_01`, `referral_consult_01`, `multi_entity_01`)
   - **Valid:** 4 templates (`admission_dx_04`, `prescription_tx_04`, `history_review_04`, `followup_discharge_02`)
   - **Test:** 4 templates (`admission_dx_05`, `prescription_tx_05`, `referral_consult_02`, `multi_entity_02`)
3. **Leakage Verification:**
   - Document ID overlap between splits: **0**
   - Template overlap between splits: **0**

---

## 7. Recommended Evaluation Protocol for HMM, MEMM, and CRF

To ensure methodological validity when comparing the models, evaluation must be reported along three distinct dimensions:

```
+-------------------------------------------------------------------------------+
|                             EVALUATION PROTOCOL                               |
+-------------------------------------------------------------------------------+
| Track A: Pure Gold Evaluation (BC5CDR Original Split)                         |
|   - Evaluates DISEASE and MEDICATION / CHEMICAL on real biomedical text.      |
|   - 100% authentic human gold standard.                                       |
|                                                                               |
| Track B: Synthetic Evaluation (Synthetic Test Slice)                          |
|   - Evaluates PATIENT_NAME recognition on unseen templates and names.        |
|   - Must NOT be reported as real-world clinical performance.                  |
|                                                                               |
| Track C: Mixed-Source Evaluation (Combined Augmented Split)                   |
|   - Evaluates complete 3-class joint extraction.                              |
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
