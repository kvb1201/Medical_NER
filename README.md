# Medical Named Entity Recognition (HMM, MEMM, CRF)

A rigorous, reproducible sequence-labeling project comparing three classical statistical sequence models—**Hidden Markov Models (HMM)**, **Maximum Entropy Markov Models (MEMM)**, and **Linear-Chain Conditional Random Fields (CRF)**—for Named Entity Recognition (NER) across authentic biomedical literature and structured clinical templates.

> [!IMPORTANT]
> **Classical Statistical NLP Scope:** This project deliberately investigates classical statistical sequence-labeling algorithms grounded in probabilistic graphical models and convex optimization. It does **not** employ transformer-based neural language models (such as BERT or RoBERTa).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features and Capabilities](#2-features-and-capabilities)
3. [Entity Schema and BIO Tagging](#3-entity-schema-and-bio-tagging)
4. [Dataset Sources and Composition](#4-dataset-sources-and-composition)
5. [Dataset Preparation Pipeline](#5-dataset-preparation-pipeline)
6. [Model Architectures and Mathematical Foundations](#6-model-architectures-and-mathematical-foundations)
7. [Model Training and Inference Workflow](#7-model-training-and-inference-workflow)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Experimental Results](#9-experimental-results)
10. [Limitations and Responsible Use](#10-limitations-and-responsible-use)
11. [Installation and Environment Setup](#11-installation-and-environment-setup)
12. [Reproducibility Commands](#12-reproducibility-commands)
13. [Repository Structure](#13-repository-structure)
14. [Provenance and Integrity Safeguards](#14-provenance-and-integrity-safeguards)
15. [Future Improvements](#15-future-improvements)

---

## 1. Project Overview

### Problem Statement & Clinical Motivation
Extracting clinical entities from medical and biomedical text is a core capability in clinical Natural Language Processing. Clinical decision support, pharmacovigilance (monitoring adverse drug reactions), cohort discovery, and medical record de-identification (HIPAA/PHI compliance) all depend on reliable token-level span classification. The target entities include:
- `DISEASE`: Pathology, clinical syndromes, signs, and symptoms (e.g., *hypertension*, *diabetic ketoacidosis*, *pulmonary embolism*).
- `MEDICATION`: Approved pharmaceuticals, therapeutics, and drug components (e.g., *metformin*, *lisinopril*, *amoxicillin*).
- `CHEMICAL`: Authentic chemical compounds, laboratory metabolites, and industrial reagents that are not administered as drugs (e.g., *carbon tetrachloride*, *hydrogen peroxide*).
- `PATIENT_NAME`: Personal names requiring detection and redaction for patient privacy compliance (e.g., *John Doe*, *Jane Smith*).

### What the System Does
The system takes raw, tokenized text sequences and assigns each token an aligned label in the Begin-Inside-Outside (**BIO**) tagging format:

$$\mathbf{x} = (x_1, x_2, \dots, x_T) \quad \longrightarrow \quad \hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_T)$$

```
Token:   Patient  John            Doe             presented  with  severe  pneumonia  and  was  prescribed  amoxicillin  .
Output:  O        B-PATIENT_NAME  I-PATIENT_NAME  O          O     O       B-DISEASE  O    O    O           B-MEDICATION O
```

### Why Compare HMM, MEMM, and CRF?
The three models represent the evolutionary milestones of classical statistical sequence modeling:
1. **Hidden Markov Model (HMM):** A **generative** model estimating the joint probability $P(X, Y)$ via Maximum Likelihood Estimation with smoothing. Fast and memory-efficient, but constrained by strict conditional independence assumptions.
2. **Maximum Entropy Markov Model (MEMM):** A **discriminative, locally normalized** conditional model $P(y_t \mid X, t, y_{t-1})$ using multinomial logistic regression with rich, overlapping observation features. However, it is fundamentally vulnerable to the **Label Bias Problem**.
3. **Linear-Chain Conditional Random Field (CRF):** A **discriminative, globally normalized** sequence model $P(Y \mid X)$ that resolves label bias via a global sequence partition function $Z(X)$, jointly optimizing transition potentials and observation features.

---

## 2. Features and Capabilities

The repository implements a production-grade, reproducible pipeline covering data acquisition, synthesis, auditing, modeling, and evaluation:

- **Automated Dataset Acquisition:** Reproducible download and schema validation of the gold-standard BioCreative V CDR corpus (`tner/bc5cdr`) from the Hugging Face Hub.
- **4-Tier Auditable Chemical-to-Medication Classification:** Resolves the conflation between generic chemical compounds and approved pharmacological drugs through an explicit 4-tier decision hierarchy with a 15,935-mention audit trail.
- **Deterministic Synthetic Clinical Augmentation:** Generates 5,000 synthetic sentences featuring fictional patient names across 28 disjoint template families and 120 disjoint fictional names, ensuring zero split leakage.
- **Leakage Prevention & Cross-Source Deduplication:** Normalizes text and computes conservative token/character similarities to detect and isolate cross-split duplicates while strictly preserving the authentic BC5CDR test benchmark.
- **12-Point BIO Dataset Integrity Suite:** Programmatically verifies sequence lengths, syntax conformity, continuation consistency, and cross-split disjointness.
- **Generative Sequence Modeling (HMM):** MLE transition and emission parameter estimation with Laplace smoothing and 8-signature sub-word morphological fallbacks for Out-of-Vocabulary (OOV) tokens.
- **Discriminative Sequence Modeling (MEMM):** Rich feature extraction (word forms, word shapes, prefixes, suffixes, context windows, bigrams) with exact linear weight decomposition ($W_{\text{obs}} + W_{\text{trans}}$) for sub-millisecond Viterbi decoding.
- **Global Sequence Modeling (CRF):** Linear-Chain CRF trained via L-BFGS optimization with elastic net ($L_1 + L_2$) regularization using identical observation features as MEMM for a strict apples-to-apples comparison.
- **Exact-Span CoNLL Evaluation Engine:** Chunks predicted and gold BIO tags into exact $(entity, start, end)$ spans. Supports token accuracy, micro-averaging, active-class macro-averaging, and fixed 4-class macro-averaging.
- **Zero-Retraining Artifact Evaluation:** Serialized `.joblib` model checkpoints and standalone prediction interfaces supporting deterministic reproduction.

---

## 3. Entity Schema and BIO Tagging

### Target Label Schema
The project models **4 clinical entity classes** plus non-entity background tokens using standard BIO encoding (9 unique tag classes):

| Tag | Entity Class | Definition | Clinical Example |
| :--- | :--- | :--- | :--- |
| `O` | Background | Non-entity token | *was*, *admitted*, *with* |
| `B-DISEASE` | Disease (Begin) | First token of a disease, symptom, or disorder | **hypertension**, **heart** |
| `I-DISEASE` | Disease (Inside) | Subsequent token of a multi-word disease | heart **failure** |
| `B-MEDICATION` | Medication (Begin) | First token of a therapeutic drug or medication | **metformin**, **insulin** |
| `I-MEDICATION` | Medication (Inside) | Subsequent token of a multi-word drug name | insulin **glargine** |
| `B-CHEMICAL` | Chemical (Begin) | First token of a non-therapeutic chemical substance | **potassium**, **carbon** |
| `I-CHEMICAL` | Chemical (Inside) | Subsequent token of a multi-word chemical entity | carbon **tetrachloride** |
| `B-PATIENT_NAME`| Patient Name (Begin) | First name or title of a fictional patient | **John**, **Jane** |
| `I-PATIENT_NAME`| Patient Name (Inside) | Surname or hyphenated surname of a fictional patient | **Smith**, **Nelson-Cole** |

### BIO Tag Alignment Example
In standard BIO encoding, multi-token entity spans begin with a `B-` prefix and continue with matching `I-` prefixes. Consecutive distinct entities of the same type are disambiguated by a new `B-` tag:

```
Sentence:   "Patient Robert Taylor was administered insulin glargine for diabetes mellitus ."
Tokens:     ['Patient', 'Robert', 'Taylor', 'was', 'administered', 'insulin', 'glargine', 'for', 'diabetes', 'mellitus', '.']
BIO Tags:   ['O', 'B-PATIENT_NAME', 'I-PATIENT_NAME', 'O', 'O', 'B-MEDICATION', 'I-MEDICATION', 'O', 'B-DISEASE', 'I-DISEASE', 'O']
```

### Distinction Between `MEDICATION` and `CHEMICAL`
Original BC5CDR annotations group all substances under a generic `Chemical` label. In clinical applications, treating industrial solvents, reagents, or toxic mutagens as medications is dangerous. `src/data/map_entities.py` implements an auditable 4-tier decision hierarchy:
1. **Tier 1 (Non-Medication Exclusions):** Explicitly retains known non-drug chemicals (e.g., *carbon monoxide*, *lead*, *cyanide*) as `CHEMICAL`.
2. **Tier 2 (Curated Drug Gazettes):** Maps verified FDA/EMA-approved pharmaceuticals (e.g., *naloxone*, *clonidine*, *ibuprofen*) to `MEDICATION`.
3. **Tier 3 (Pharmacological Suffix Heuristics):** Identifies active pharmaceutical ingredients via USAN/INN stems (e.g., `-olol` for beta-blockers, `-statin` for HMG-CoA reductase inhibitors, `-cillin` for penicillins).
4. **Tier 4 (Conservative Fallback):** Retains all ambiguous or unverified chemical entities as `CHEMICAL`.

The full decision log across all 15,935 chemical mentions is recorded in [`reports/medication_mapping_audit.csv`](reports/medication_mapping_audit.csv). Exactly **6,690 mentions (41.98%)** are classified as `MEDICATION`, and **9,245 mentions (58.02%)** are preserved as `CHEMICAL`.

---

## 4. Dataset Sources and Composition

### 1. Authentic Biomedical Literature (BC5CDR)
- **Source:** BioCreative V Chemical-Disease Relation corpus ([Li et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27161011/)).
- **Repository:** [`tner/bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr) (Commit SHA: `f68cdc7db924369241e7868656f583072acd4e90`).
- **Nature of Text:** Peer-reviewed PubMed scientific literature abstracts.
- **License:** Openly accessible for non-commercial scientific research under BioCreative terms.

### 2. Controlled Synthetic Clinical Augmentation
- **Motivation:** Authentic PubMed literature contains **zero patient personal names** due to peer-reviewed publication conventions. Clinical EHR systems, however, require robust detection of patient names alongside diagnoses and prescriptions for de-identification.
- **Generation:** 5,000 synthetic sentences generated deterministically via [`src/data/generate_synthetic.py`](src/data/generate_synthetic.py) using 28 clinical templates across distinct hospital encounters (admissions, outpatient reviews, discharges).
- **Leakage Prevention:** 120 fictional names are partitioned into strictly disjoint subsets (60 train, 30 valid, 30 test; 0 overlap). The 28 template families are similarly allocated disjointly across splits (16 train, 6 valid, 6 test; 0 overlap).

### 3. Verified Corpus Partition Statistics

All counts below were verified directly from the JSONL files:

| Partition / Split | Sentences | Unique Sentences | Duplicates | Total Tokens | `DISEASE` | `MEDICATION` | `CHEMICAL` | `PATIENT_NAME` | Total Mentions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC5CDR Raw Train** | 5,228 | 5,141 | 87 | 109,322 | 4,182 | 2,239 | 2,964 | 0 | 9,385 |
| **Synthetic Train** | 4,000 | 4,000 | 0 | 52,984 | 4,000 | 2,055 | 0 | 4,000 | 10,055 |
| **Augmented Train (Total)** | **9,228** | **9,141** | **87** | **162,306** | **8,182** | **4,294** | **2,964** | **4,000** | **19,440** |
| **BC5CDR Raw Valid** | 5,330 | 5,202 | 128 | 108,958 | 4,244 | 2,206 | 3,141 | 0 | 9,591 |
| **Synthetic Valid** | 500 | 500 | 0 | 7,238 | 500 | 252 | 0 | 500 | 1,252 |
| **Augmented Valid (Total)** | **5,830** | **5,702** | **128** | **116,196** | **4,744** | **2,458** | **3,141** | **500** | **10,843** |
| **BC5CDR Original Test (Fixed)** | 5,865 | 5,713 | 152 | 116,318 | 4,424 | 2,245 | 3,140 | 0 | 9,809 |
| **Synthetic Test** | 500 | 500 | 0 | 7,680 | 500 | 339 | 0 | 500 | 1,339 |
| **Augmented Test (Total)** | **6,365** | **6,213** | **152** | **123,998** | **4,924** | **2,584** | **3,140** | **500** | **11,148** |
| **GRAND TOTAL CORPUS** | **21,423** | **21,056** | **367** | **402,500** | **17,850** | **9,336** | **9,245** | **5,000** | **41,431** |

*Note on BC5CDR duplicates:* The 367 duplicate sentences within BC5CDR are verbatim repeated standard section headers and study phrases (e.g., `'METHODS:'` or `'RESULTS:'`). They are authentic to biomedical literature abstracts and are preserved in the gold data.

---

## 5. Dataset Preparation Pipeline

The complete end-to-end dataset pipeline is coordinated by [`src/data/prepare_dataset.py`](src/data/prepare_dataset.py) and executed via:

```bash
PYTHONPATH=. python3 src/data/prepare_dataset.py
```

```
                                      DATASET PREPARATION PIPELINE

  Hugging Face Hub (tner/bc5cdr)                Fictional Clinical Generator
               │                                             │
               ▼                                             ▼
  [1] Download & Parse Raw Data               [5] Generate 5,000 Sentences
      (src/data/load_bc5cdr.py)                   (src/data/generate_synthetic.py)
               │                                             │
               ▼                                             │
  [2] Auditable Chemical Mapping                             │
      (src/data/map_entities.py)                             │
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
                        [6] Deduplication & Overlap Audit
                            (src/data/deduplicate.py)
                                      │
                                      ▼
                        [7] 12-Point BIO Validator
                            (src/data/validate_bio.py)
                                      │
                                      ▼
                        [8] Assemble Standardized Datasets
                            • bc5cdr_original_{split}.jsonl
                            • bc5cdr_augmented_{split}.jsonl
                                      │
                                      ▼
                        [9] Export Audit Reports & Stats
                            • reports/dataset_audit_report.md
                            • reports/medication_mapping_audit.csv
                            • reports/split_statistics.json
```

### Protection of the BC5CDR Test Benchmark
The original BC5CDR test partition ([`data/processed/bc5cdr_original_test.jsonl`](data/processed/bc5cdr_original_test.jsonl)) is **100% immutable**:
- It contains exactly 5,865 sentences, 116,318 tokens, and 500 document IDs matching `data/raw/bc5cdr_test.jsonl` with **0 token mismatches**.
- Original gold tags are preserved in `original_ner_tags`.
- No synthetic records or patient names are introduced into this partition.

---

## 6. Model Architectures and Mathematical Foundations

### 1. Hidden Markov Model (HMM)
- **Mathematical Formulation:** Generative joint probability model factorized into initial, transition, and emission distributions:
  $$P(X, Y) = P(y_1) P(x_1 \mid y_1) \prod_{t=2}^T P(y_t \mid y_{t-1}) P(x_t \mid y_t)$$
- **Parameter Estimation:** Maximum Likelihood Estimation (MLE) with Laplace smoothing:
  $$P(y_j \mid y_i) = \frac{C(y_i, y_j) + \alpha_{\text{trans}}}{\sum_{k} C(y_i, y_k) + K \cdot \alpha_{\text{trans}}}$$
  $$P(x \mid y) = \frac{C(y, x) + \alpha_{\text{emit}}}{\sum_{w \in V} C(y, w) + (|V| + |S|) \cdot \alpha_{\text{emit}}}$$
- **Out-of-Vocabulary (OOV) Handling:** Words with training frequency $< 2$ are mapped during training and inference to 8 morphological signatures: `<UNK_TITLE>`, `<UNK_ALL_CAPS>`, `<UNK_NUM>`, `<UNK_DASH>`, `<UNK_SUFFIX_DISEASE>`, `<UNK_SUFFIX_DRUG>`, `<UNK_LOWER>`, and `<UNK>`.
- **Decoding:** Exact Viterbi trellis dynamic programming in log-space with strict BIO transition penalty masks ($-\infty$ assigned to illegal transitions such as `O -> I-*`).
- **Hyperparameters:** $\alpha_{\text{trans}} = 0.01$, $\alpha_{\text{emit}} = 0.001$, $\alpha_{\text{init}} = 0.01$, $\text{min\_word\_freq} = 2$.
- **Strengths & Weaknesses:** Trains in well under a second (**0.24 s** in the latest run). However, it cannot incorporate multi-token context or overlapping linguistic features and scores $F_1 = 0.0000$ on unseen patient names.

---

### 2. Maximum Entropy Markov Model (MEMM)
- **Mathematical Formulation:** Discriminative local conditional model parameterized by multinomial logistic regression:
  $$P(y_t \mid X, t, y_{t-1}) = \frac{\exp\left(\mathbf{w}_{y_t}^T \mathbf{f}(X, t, y_{t-1}) + b_{y_t}\right)}{\sum_{y'} \exp\left(\mathbf{w}_{y'}^T \mathbf{f}(X, t, y_{t-1}) + b_{y'}\right)}$$
- **Feature Representation (158,875 vectorized sparse features):**
  - Current word identity, lowercased word, word length.
  - Sub-word prefixes ($l \in \{2, 3\}$) and suffixes ($l \in \{2, 3, 4\}$).
  - Word shape (e.g., `Metformin` $\to$ `Xx+`, `COVID-19` $\to$ `XXXX-dd`).
  - Orthographic flags: `is_title`, `is_upper`, `is_digit`, `has_hyphen`.
  - Contextual window features: tokens and shapes for $w_{-2}, w_{-1}, w_{+1}, w_{+2}$.
  - Token bigrams: left bigram $(w_{-1}, w_0)$, right bigram $(w_0, w_{+1})$.
  - Transition feature: previous tag identity $y_{t-1}$.
- **Fast Viterbi Decoding via Decomposed Weights:**
  Logits are linear: $\mathbf{z}_t = \mathbf{W}_{\text{obs}} \mathbf{x}_{\text{obs}, t} + \mathbf{W}_{\text{trans}} \mathbf{e}_{y_{t-1}} + \mathbf{b}$. Observation projections $\mathbf{W}_{\text{obs}} \mathbf{X}_{\text{obs}}$ are precomputed per sentence in a single sparse matrix multiplication, reducing trellis recursion to $O(1)$ lookups.
- **Hyperparameters:** $C = 1.0$ ($L_2$ regularization), solver: `lbfgs`, $\text{max\_iter} = 100$, random seed: 42.
- **Label Bias Problem:** Because probabilities are normalized locally at each state ($\sum_{y'} P(y' \mid y_{t-1}, X) = 1$), states with low-entropy transitions dominate the trellis regardless of observation evidence, leading to suboptimal global sequences.

---

### 3. Linear-Chain Conditional Random Field (CRF)
- **Mathematical Formulation:** Discriminative globally normalized sequence model:
  $$P(Y \mid X) = \frac{1}{Z(X)} \exp\left(\sum_{t=1}^T \sum_k w_k f_k(y_{t-1}, y_t, X, t)\right)$$
  $$Z(X) = \sum_{Y'} \exp\left(\sum_{t=1}^T \sum_k w_k f_k(y'_{t-1}, y'_t, X, t)\right)$$
- **Global Normalization Advantage:** Resolves the label bias problem. Transition potentials $A_{y_{t-1}, y_t}$ and state observation potentials $P_{t, y_t}$ are globally balanced through the partition function $Z(X)$ computed via the forward-backward algorithm.
- **Feature Set:** Uses the exact same observation feature representation as MEMM (tokens, shapes, prefixes, suffixes, context windows, bigrams) to guarantee a strict, apples-to-apples comparison.
- **Hyperparameters:** $c_1 = 0.1$ ($L_1$ sparsity regularization), $c_2 = 0.1$ ($L_2$ smoothing regularization), algorithm: `lbfgs`, $\text{max\_iterations} = 100$, `all_possible_transitions = True`.

---

## 7. Model Training and Inference Workflow

### Training Procedure
All three models are trained on the combined augmented training set ([`data/processed/bc5cdr_augmented_train.jsonl`](data/processed/bc5cdr_augmented_train.jsonl), 9,228 sentences, 162,306 tokens):

```bash
# Train all three models and evaluate across all three tracks
PYTHONPATH=. python3 src/models/train_and_evaluate.py

# Train and evaluate a single model (e.g., CRF)
PYTHONPATH=. python3 src/models/train_and_evaluate.py --model crf
```

Model checkpoints are automatically serialized to `models/`:
- `models/hmm.joblib` (604 KB)
- `models/memm.joblib` (21 MB)
- `models/crf.joblib` (1.8 MB)

### Standalone Inference Interface
You can load any trained checkpoint and run inference directly in Python:

```python
import joblib

# 1. Load trained model artifact
model = joblib.load("models/crf.joblib")

# 2. Provide tokenized sentences (list of string lists)
sentences = [
    ["Patient", "John", "Doe", "was", "prescribed", "metformin", "for", "diabetes", "mellitus", "."]
]

# 3. Predict sequence of BIO tags
predictions = model.predict(sentences)

for token, tag in zip(sentences[0], predictions[0]):
    print(f"{token:15s} -> {tag}")
```

---

## 8. Evaluation Methodology

### 1. Exact-Span CoNLL Matching
In accordance with CoNLL/SemEval standards, an entity prediction is correct if and only if its **entity class**, **start token index**, and **end token index** match the gold annotation exactly:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

Partial overlaps receive no credit.

### 2. Aggregation Metrics
- **Micro-Average:** Computed by globally summing true positives, false positives, and false negatives across all classes:
  $$\text{Micro } F_1 = \frac{2 \cdot \sum_c TP_c}{2 \cdot \sum_c TP_c + \sum_c FP_c + \sum_c FN_c}$$
- **Active-Class Macro-Average:** Unweighted mean over entity classes that have gold support in that benchmark partition ($\text{support} > 0$). Evaluates the specific task defined by that slice.
- **Full 4-Class Macro-Average:** Unweighted mean across all 4 target schema classes ($N = 4$), strictly penalizing out-of-domain false positives with $F_1 = 0.0$.

### 3. Three Distinct Evaluation Tracks
- **Track A (Pure Gold Biomedical Benchmark):** Evaluates strictly on the original BC5CDR test partition (5,865 sentences, 116,318 tokens). Gold classes: `DISEASE`, `MEDICATION`, `CHEMICAL`.
- **Track B (Synthetic Clinical Name Benchmark):** Evaluates strictly on the synthetic test partition (500 sentences, 7,680 tokens). Gold classes: `PATIENT_NAME`, `DISEASE`, `MEDICATION`.
- **Track C (Combined Augmented Test Benchmark):** Joint evaluation over the entire augmented test partition (6,365 sentences, 123,998 tokens: exact concatenation of Track A + Track B). All 4 classes active.

> [!WARNING]
> **Track Composition Note:** Track C evaluates the exact union of Track A and Track B. Therefore, results from Track C are **not** independent of Tracks A and B. Across all three tracks, exactly **247,996 token evaluations** are executed over **123,998 test tokens**.

---

## 9. Experimental Results

The experimental metrics below are verified from [`reports/model_metrics.json`](reports/model_metrics.json) and [`reports/model_metrics.csv`](reports/model_metrics.csv):

### Executive Benchmark Summary

| Model | Architecture Type | Training Time | Track A Micro F1 | Track A Active Macro F1 | Track B Micro F1 | Track B Active Macro F1 | Track C Micro F1 | Track C 4-Class Macro F1 | Token Accuracy (Track C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HMM** | Generative (Joint MLE) | **0.24s** | 0.6838 | 0.6948 | 0.7548 | 0.6553 | 0.6916 | 0.5328 | 94.15% |
| **MEMM** | Discriminative (Local MaxEnt) | 63.07s | 0.7776 | 0.7905 | **0.9301** | **0.9399** | 0.7968 | 0.8317 | 96.22% |
| **CRF** | Discriminative (Global CRF) | 46.15s | **0.7905** | **0.8019** | 0.8883 | 0.9078 | **0.8029** | **0.8399** | **96.44%** |

---

### Detailed Performance by Evaluation Track

#### Track A: Pure Gold Evaluation (BC5CDR Original Test Partition, 5,865 Sentences, 116,318 Tokens)

| Model | Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **HMM** | `DISEASE` | 0.6729 | 0.6555 | **0.6641** | 4,424 | Active gold class |
| | `MEDICATION` | 0.9354 | 0.8383 | **0.8842** | 2,245 | Active gold class |
| | `CHEMICAL` | 0.8472 | 0.3920 | **0.5360** | 3,140 | Active gold class |
| | `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (4 false positives) |
| | **Micro-Average** | **0.7730** | **0.6130** | **0.6838** | **9,809** | Pooled span counts |
| | **Active Macro F1 (3-Class)** | **0.8185** | **0.6286** | **0.6948** | — | Mean of active classes |
| | **Full Macro F1 (4-Class)** | 0.6139 | 0.4715 | 0.5211 | — | Fixed $|C|=4$ schema |
| **MEMM** | `DISEASE` | 0.8103 | 0.7030 | **0.7528** | 4,424 | Active gold class |
| | `MEDICATION` | 0.9064 | 0.9234 | **0.9148** | 2,245 | Active gold class |
| | `CHEMICAL` | 0.8320 | 0.6102 | **0.7040** | 3,140 | Active gold class |
| | `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (22 false positives) |
| | **Micro-Average** | **0.8401** | **0.7237** | **0.7776** | **9,809** | Pooled span counts |
| | **Active Macro F1 (3-Class)** | **0.8496** | **0.7455** | **0.7905** | — | Mean of active classes |
| | **Full Macro F1 (4-Class)** | 0.6372 | 0.5591 | 0.5929 | — | Fixed $|C|=4$ schema |
| **CRF** | `DISEASE` | 0.8190 | 0.7344 | **0.7744** | 4,424 | Active gold class |
| | `MEDICATION` | 0.9174 | 0.9256 | **0.9215** | 2,245 | Active gold class |
| | `CHEMICAL` | 0.8388 | 0.6150 | **0.7097** | 3,140 | Active gold class |
| | `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (19 false positives) |
| | **Micro-Average** | **0.8486** | **0.7399** | **0.7905** | **9,809** | Pooled span counts |
| | **Active Macro F1 (3-Class)** | **0.8584** | **0.7583** | **0.8019** | — | Mean of active classes: $\frac{0.7744+0.9215+0.7097}{3}$ |
| | **Full Macro F1 (4-Class)** | 0.6438 | 0.5688 | 0.6014 | — | Fixed $|C|=4$ schema: $\frac{2.4056+0}{4}$ |

---

#### Track B: Synthetic Evaluation (Synthetic Test Partition, 500 Sentences, 7,680 Tokens)

| Model | Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **HMM** | `PATIENT_NAME` | 0.0000 | 0.0000 | **0.0000** | 500 | Unseen names fallback to OOV |
| | `DISEASE` | 0.9660 | 0.9660 | **0.9660** | 500 | Active gold class |
| | `MEDICATION` | 1.0000 | 1.0000 | **1.0000** | 339 | Active gold class |
| | `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (0 false positives) |
| | **Micro-Average** | **0.9797** | **0.6139** | **0.7548** | **1,339** | Pooled span counts |
| | **Active Macro F1 (3-Class)** | **0.6553** | **0.6553** | **0.6553** | — | Mean of active classes |
| **MEMM** | `PATIENT_NAME` | 1.0000 | 0.9160 | **0.9562** | 500 | Contextual features resolve names |
| | `DISEASE` | 0.8940 | 0.8940 | **0.8940** | 500 | Active gold class |
| | `MEDICATION` | 1.0000 | 0.9410 | **0.9696** | 339 | Active gold class |
| | `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (16 false positives) |
| | **Micro-Average** | **0.9466** | **0.9141** | **0.9301** | **1,339** | Pooled span counts |
| | **Active Macro F1 (3-Class)** | **0.9647** | **0.9170** | **0.9399** | — | Mean of active classes |
| **CRF** | `PATIENT_NAME` | 1.0000 | 0.9520 | **0.9754** | 500 | Global optimization of name spans |
| | `DISEASE` | 0.8540 | 0.8540 | **0.8540** | 500 | Active gold class |
| | `MEDICATION` | 1.0000 | 0.8083 | **0.8940** | 339 | Active gold class |
| | `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (61 false positives) |
| | **Micro-Average** | **0.8978** | **0.8790** | **0.8883** | **1,339** | Pooled span counts |
| | **Active Macro F1 (3-Class)** | **0.9513** | **0.8714** | **0.9078** | — | Mean of active classes |

---

#### Track C: Combined Augmented Test Benchmark (6,365 Sentences, 123,998 Tokens)

| Model | Class | Precision | Recall | F1 Score | Support (Gold Spans) | Token Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM** | `DISEASE` | 0.7033 | 0.6870 | **0.6951** | 4,924 | **94.15%** |
| | `MEDICATION` | 0.9447 | 0.8595 | **0.9001** | 2,584 | |
| | `PATIENT_NAME` | 0.0000 | 0.0000 | **0.0000** | 500 | |
| | `CHEMICAL` | 0.8472 | 0.3920 | **0.5360** | 3,140 | |
| | **Micro-Average** | **0.7931** | **0.6131** | **0.6916** | **11,148** | |
| | **Macro F1 (4-Class)** | 0.6238 | 0.4846 | 0.5328 | — | |
| **MEMM** | `DISEASE` | 0.8200 | 0.7224 | **0.7681** | 4,924 | **96.22%** |
| | `MEDICATION` | 0.9179 | 0.9257 | **0.9218** | 2,584 | |
| | `PATIENT_NAME` | 0.9542 | 0.9160 | **0.9347** | 500 | |
| | `CHEMICAL` | 0.8262 | 0.6102 | **0.7020** | 3,140 | |
| | **Micro-Average** | **0.8543** | **0.7466** | **0.7968** | **11,148** | |
| | **Macro F1 (4-Class)** | 0.8796 | 0.7936 | 0.8317 | — | |
| **CRF** | `DISEASE` | 0.8229 | 0.7465 | **0.7829** | 4,924 | **96.44%** |
| | `MEDICATION` | 0.9263 | 0.9102 | **0.9182** | 2,584 | |
| | `PATIENT_NAME` | 0.9616 | 0.9520 | **0.9568** | 500 | |
| | `CHEMICAL` | 0.8172 | 0.6150 | **0.7018** | 3,140 | |
| | **Micro-Average** | **0.8551** | **0.7566** | **0.8029** | **11,148** | |
| | **Macro F1 (4-Class)** | **0.8820** | **0.8059** | **0.8399** | — | |

---

### Interpretation of Model Comparison

1. **Why HMM Fails on `PATIENT_NAME` ($F_1 = 0.0000$):**
   The HMM emission distribution $P(x_t \mid y_t)$ assumes that token generation depends strictly on the current state. Because test patient names are strictly disjoint from training names, they map to `<UNK_TITLE>`. Without surrounding lexical context (e.g., *Patient*, *Mr.*, *was admitted*), the generative model cannot distinguish an unseen proper noun from non-entity capitalized words.
2. **Why MEMM and CRF Succeed on Names ($F_1 \ge 0.9347$):**
   By conditioning on observation features across a 5-token context window ($w_{-2} \ldots w_{+2}$), discriminative models learn that capitalized words preceded by *Patient* or followed by verbs like *presented* indicate patient names, regardless of vocabulary novelty.
3. **Why CRF Outperforms MEMM on Authentic Biomedical Text:**
   The Linear-Chain CRF achieves the highest Track A Micro $F_1$ (**0.7905** vs MEMM's **0.7776**) and Track C Micro $F_1$ (**0.8029** vs MEMM's **0.7968**). Global sequence normalization prevents the label bias problem, allowing transition penalties and observation potentials to optimize jointly over complete sentence paths.
4. **Why MEMM Leads on the Synthetic Track:**
   MEMM has the highest Track B Micro $F_1$ (**0.9301** vs CRF's **0.8883**). CRF is better on `PATIENT_NAME` ($0.9754$ vs $0.9562$), but MEMM scores higher on `DISEASE` ($0.8940$ vs $0.8540$) and `MEDICATION` ($0.9696$ vs $0.8940$), and CRF produces more spurious `CHEMICAL` spans on the clinical templates (61 vs 16 false positives). Track B contains only 500 template-generated sentences, so this gap should not be read as a general ranking of the two models.

---

## 10. Limitations and Responsible Use

- **Synthetic Data Non-Equivalence:** High performance on Track B (`PATIENT_NAME` $F_1 = 0.9754$) reflects accuracy on controlled clinical templates. It must **never** be cited as real-world de-identification accuracy on unredacted, unstructured clinical notes.
- **Biomedical Literature vs. Clinical Records:** Authentic BC5CDR text consists of published PubMed abstracts. It does not contain clinical shorthand, transcription errors, or doctor notes.
- **Chemical Mapping Uncertainty:** The 4-tier chemical-to-medication classification is rule- and gazetteer-based. While audited across 15,935 mentions, edge cases between active metabolites and pharmaceutical drugs remain subject to domain definition.
- **Vocabulary Bottlenecks:** Classical Markov models lack dense sub-word contextual embeddings (e.g., Byte-Pair Encoding in transformers), resulting in lower recall on rare, hyphenated chemical compounds.
- **Intended Use:** This repository is for academic research, education, and benchmark comparison. It is **not** certified for autonomous medical diagnosis or clinical decision-making.

---

## 11. Installation and Environment Setup

### Environment Requirements
- **OS:** macOS or Linux (tested on macOS Darwin 24.6.0 and Ubuntu Linux).
- **Python:** Python 3.10 to 3.14 (tested on Python 3.14).

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/kvb1201/Medical_NER.git
cd Medical_NER

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade package installer
pip install --upgrade pip

# 4. Install dependencies (versions tested with this project)
pip install -r requirements.txt
```

### Verify Environment
```bash
python3 -c "import sklearn, sklearn_crfsuite, joblib, yaml; print('Environment verified successfully.')"
```

---

## 12. Reproducibility Commands

### Quick Validation (Runs in ~2 seconds)
```bash
# Run the complete test suite (64 automated unit & integration tests)
PYTHONPATH=. pytest tests/ -v
```

### Evaluate Pre-Trained Models (no retraining; a few minutes on a laptop CPU)
```bash
# Evaluate existing serialized models and regenerate reports/
PYTHONPATH=. python3 src/models/train_and_evaluate.py --eval-only
```

### Full Pipeline Reproduction from Scratch
```bash
# 1. Download BC5CDR, generate synthetic notes, deduplicate, and assemble datasets
PYTHONPATH=. python3 src/data/prepare_dataset.py

# 2. Train HMM, MEMM, and CRF from scratch and evaluate all tracks (~6 minutes on a laptop CPU)
PYTHONPATH=. python3 src/models/train_and_evaluate.py

# 3. Inspect git status (verify no binaries are staged)
git status --short
```

> [!NOTE]
> The dataset build takes about 5 minutes and training about 6 minutes on a laptop CPU. With the fixed seeds, a fresh build reproduces the committed `reports/` exactly, apart from measured training and inference times.

### Run the Web Demo
The demo page tags a free-text note with all three trained models and shows where they agree. It needs the checkpoints in `models/` (built by the commands above).

```bash
PYTHONPATH=. python3 app/server.py            # then open http://127.0.0.1:8000
PYTHONPATH=. python3 app/server.py --port 8080  # optional: another port (or --host to change the interface)
```

- Paste a note (up to 4,000 characters) or click an example, then press **Extract entities** or <kbd>Ctrl</kbd>+<kbd>Enter</kbd>.
- **Model agreement** lists every entity any model found and which models found it; below it, each model's highlighted reading.
- Free text is tokenized to match the training data: BC5CDR-style hyphen splitting, titles such as `Dr.` and decimals kept whole. `Mr.`/`Mrs.`/`Ms.` never occur in training and are not passed to the models.
- The server listens on `127.0.0.1` only by default and is a research demo, **not** for clinical use.

---

## 13. Repository Structure

```
Medical_NER/
├── .gitignore                          # Comprehensive exclusion rules (ignores /models/, .venv, *.jsonl)
├── README.md                           # Master documentation, architecture guide, and reproduction manual
├── requirements.txt                    # Python dependencies (tested versions)
├── app/
│   ├── server.py                       # Local web demo server (HMM, MEMM, CRF side by side)
│   └── static/index.html               # Demo page: highlighting and model-agreement view
├── config/
│   └── dataset_config.yaml             # Provenance, split sizes, and template allocation configurations
├── data/
│   ├── raw/                            # BC5CDR raw downloads & provenance metadata (JSONL files gitignored)
│   │   └── raw_provenance_metadata.json
│   ├── synthetic/                      # Synthetic clinical note splits & metadata (JSONL files gitignored)
│   │   └── synthetic_generation_metadata.json
│   └── processed/                      # Original & augmented BIO datasets, label schemas (JSONL gitignored)
│       ├── id2label.json               # Integer-to-tag schema mapping
│       └── label2id.json               # Tag-to-integer schema mapping
├── models/                             # Serialized model checkpoints (gitignored via /models/)
│   ├── hmm.joblib                      # Fitted Hidden Markov Model artifact (604 KB)
│   ├── memm.joblib                     # Fitted Maximum Entropy Markov Model artifact (21 MB)
│   └── crf.joblib                      # Fitted Linear-Chain CRF artifact (1.8 MB)
├── reports/                            # Publication-grade evaluation reports and audit logs
│   ├── dataset_audit_report.md         # Comprehensive dataset provenance and split audit report
│   ├── medication_mapping_audit.csv    # Record-level audit log of all 15,935 chemical mapping decisions
│   ├── near_duplicate_audit.csv        # Cross-source near-duplicate audit log
│   ├── label_distribution.csv          # Mention frequencies across splits
│   ├── split_statistics.json           # Machine-readable token and sentence counts
│   ├── model_comparison_report.md      # Full model comparison report with exact track breakdown
│   ├── model_metrics.json              # Machine-readable evaluation metrics for HMM, MEMM, CRF
│   └── model_metrics.csv               # Tabular evaluation metrics per class and track
├── src/
│   ├── data/
│   │   ├── load_bc5cdr.py              # Hugging Face BC5CDR acquisition and parser
│   │   ├── map_entities.py             # 4-tier auditable chemical-to-medication mapper
│   │   ├── generate_synthetic.py       # Deterministic clinical sentence generator (v2.0)
│   │   ├── deduplicate.py              # Exact and near-duplicate leakage detection
│   │   ├── validate_bio.py             # 12-point BIO validation and integrity suite
│   │   └── prepare_dataset.py          # End-to-end dataset pipeline runner
│   ├── infer.py                        # Free-text tokenization and span extraction for the web demo
│   └── models/
│       ├── hmm.py                      # Supervised HMM with Laplace smoothing & Viterbi decoding
│       ├── memm.py                     # MEMM with dense feature extraction & decomposed Viterbi
│       ├── crf.py                      # Linear-Chain CRF wrapping sklearn-crfsuite (L-BFGS)
│       ├── evaluation.py               # Exact entity-span evaluation engine (Micro/Macro/Accuracy)
│       └── train_and_evaluate.py       # Unified training and 3-track evaluation pipeline
└── tests/                              # Pytest test suite (64 automated unit & integration tests)
    ├── test_deduplicate.py             # Tests text normalization, similarity, and duplicate filtering
    ├── test_generate_synthetic.py      # Tests deterministic generation, template & name disjointness
    ├── test_infer.py                   # Tests free-text tokenization, sentence splitting, and span offsets
    ├── test_map_entities.py            # Tests 4-tier chemical mapping and suffix rules
    ├── test_models.py                  # Tests HMM, MEMM, CRF fitting, inference, and exact span metrics
    ├── test_pipeline_integrity.py      # Tests configuration, augmented files, and report generation
    ├── test_reporting.py               # Tests that report conclusions are computed from the results
    ├── test_server.py                  # Tests demo server responses, including malformed requests
    └── test_validate_bio.py            # Tests sequence alignment, syntax, and cross-split leakage
```

---

## 14. Provenance and Integrity Safeguards

- **Deterministic Seeds:** Fixed random seeds (`seed = 42`) in `config/dataset_config.yaml`, `src/data/generate_synthetic.py`, and `src/models/train_and_evaluate.py` guarantee identical dataset generation, feature vectors, and model weights across runs.
- **Cross-Split Disjointness:** Verified zero patient-name overlap across splits (60 train, 30 valid, 30 test names) and zero template family overlap across splits (16 train, 6 valid, 6 test templates).
- **Benchmark Immutability:** The original BC5CDR test partition is strictly protected from synthetic additions or record removals.
- **Git Hygiene:** Large generated dataset files (`*.jsonl`) and binary model checkpoints (`/models/*.joblib`) are excluded from Git tracking via [`.gitignore`](.gitignore). Source code, tests, configuration schemas, and audit reports remain fully version-controlled.

---

## 15. Future Improvements

- **Independent Clinical Note Evaluation:** Benchmark models on authentic, de-identified clinical notes (e.g., MIMIC-III or i2b2/n2c2) to assess cross-domain transfer from biomedical literature to bedside documentation.
- **Sub-Word Tokenization & Contextual Embeddings:** Incorporate Byte-Pair Encoding (BPE) or WordPiece representations to reduce OOV rates on complex chemical nomenclature.
- **Neural Sequence Baselines:** Benchmark these statistical models against modern transformer architectures (such as BioBERT, ClinicalBERT, or PubMedBERT).
- **Confidence Calibration:** Implement posterior margin calibration or Platt scaling to provide probability confidence intervals alongside entity predictions.
- **Nested Entity Extraction:** Extend the tagger to handle overlapping clinical entities (e.g., *type 2 diabetes mellitus* containing both *type 2 diabetes* and *diabetes mellitus*).

---

## References

1. **BC5CDR Corpus:** J. Li et al., "BioCreative V CDR task corpus: a resource for chemical disease relation extraction," *Database*, vol. 2016, 2016. [PubMed: 27161011](https://pubmed.ncbi.nlm.nih.gov/27161011/).
2. **HMM for POS & NER:** D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft, Chapter 8: Sequence Labeling with HMMs and CRFs, 2024.
3. **MEMM:** A. McCallum, D. Freitag, and F. C. N. Pereira, "Maximum Entropy Markov Models for Information Extraction and Segmentation," in *Proc. ICML*, 2000, pp. 591–598.
4. **Linear-Chain CRF:** J. Lafferty, A. McCallum, and F. C. N. Pereira, "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data," in *Proc. ICML*, 2001, pp. 282–289.
5. **Exact-Span Evaluation:** E. F. Tjong Kim Sang and F. De Meulder, "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition," in *Proc. CoNLL*, 2003, pp. 142–147.
