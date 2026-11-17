# Medical NER Model Comparison Report: HMM vs. MEMM vs. CRF

**Evaluation Target:** 4-Entity Sequence Labeling (`DISEASE`, `MEDICATION`, `PATIENT_NAME`, `CHEMICAL`)  
**Training Corpus:** Combined Augmented Training Set (9,228 sentences: 5,228 BC5CDR + 4,000 Synthetic)  
**Validation Corpus:** Combined Augmented Validation Set (5,830 sentences: 5,330 BC5CDR + 500 Synthetic)  
**Evaluation Tracks:**
- **Track A (Pure Gold Biomedical Benchmark):** BC5CDR Original Test Slice (5,865 sentences, 116,318 tokens, 100% human gold standard)
- **Track B (Synthetic Clinical Name Benchmark):** Synthetic Test Slice (500 sentences, 6,411 tokens, fictional clinical EHR templates)
- **Track C (Combined Augmented Test Benchmark):** Joint Evaluation (6,365 sentences, 122,729 tokens: union of Track A + Track B)

### Corpus & Evaluation Token Accounting

| Evaluation Scope | Sentences | Tokens | Scope Description |
| :--- | :---: | :---: | :--- |
| **Track A (BC5CDR Original Test)** | 5,865 | 116,318 | Authentic PubMed literature abstracts |
| **Track B (Synthetic Clinical Test)** | 500 | 6,411 | Fictional clinical EHR templates |
| **Track C (Combined Augmented Test)** | 6,365 | 122,729 | Concatenation of Track A + Track B |
| **Token evaluations across Tracks A+B+C** | — | **245,458** | Track C re-evaluates the tokens of Tracks A and B |
| **Validation Partition** | 5,830 | 115,030 | Augmented validation split |
| **Training Partition** | 9,228 | 156,377 | Augmented training split |
| **Full Project Corpus (Train+Val+Test)** | **21,423** | **394,136** | All tokens across the augmented splits |

---

## 1. Executive Summary & Comparative Highlights

| Model | Architecture Type | Training Time | Track A F1 (BC5CDR) | Track B F1 (Synthetic) | Track C F1 (Combined) | Token Accuracy (Track C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM** | Generative (Joint MLE) | 0.11s | **0.6828** | **0.8752** | **0.7032** | 94.67% |
| **MEMM** | Discriminative (Local Cond. MaxEnt) | 9.85s | **0.7777** | **0.9592** | **0.7982** | 96.30% |
| **CRF** | Discriminative (Global Linear-Chain CRF) | 20.05s | **0.7889** | **0.9329** | **0.8051** | 96.49% |

---

## 2. Track A: Pure Gold Evaluation (BC5CDR Original Test Partition)
*Evaluates model generalization strictly on authentic PubMed scientific literature abstracts (5,865 sentences, 116,318 tokens).*

### HMM Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.6739 | 0.6564 | **0.6651** | 4,424 | Active gold class |
| `MEDICATION` | 0.9349 | 0.8383 | **0.8840** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8532 | 0.3943 | **0.5393** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (64 false positives) |
| **Micro-Average** | **0.7687** | **0.6141** | **0.6828** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8207** | **0.6297** | **0.6961** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6155 | 0.4722 | 0.5221 | — | All schema classes (incl. inactive) |

### MEMM Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.8071 | 0.7075 | **0.7540** | 4,424 | Active gold class |
| `MEDICATION` | 0.9083 | 0.9216 | **0.9149** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8350 | 0.6061 | **0.7023** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (21 false positives) |
| **Micro-Average** | **0.8399** | **0.7240** | **0.7777** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8501** | **0.7451** | **0.7904** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6376 | 0.5588 | 0.5928 | — | All schema classes (incl. inactive) |

### CRF Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.8134 | 0.7333 | **0.7713** | 4,424 | Active gold class |
| `MEDICATION` | 0.9208 | 0.9269 | **0.9239** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8346 | 0.6137 | **0.7073** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (20 false positives) |
| **Micro-Average** | **0.8455** | **0.7393** | **0.7889** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8563** | **0.7580** | **0.8008** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6422 | 0.5685 | 0.6006 | — | All schema classes (incl. inactive) |

---

## 3. Track B: Synthetic Evaluation (Synthetic Test Slice)
*Evaluates fictional `PATIENT_NAME` recognition alongside medical entities on unseen clinical EHR templates (500 sentences, 6,411 tokens).*

> [!WARNING]
> **Clinical Non-Equivalence Disclaimer:** Synthetic test performance demonstrates sequence labeling on controlled syntactic structures. It must **never** be cited as real-world clinical EHR extraction accuracy on unredacted medical notes.

### HMM Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.5120 | **0.6772** | 500 | Active gold class |
| `DISEASE` | 0.9820 | 0.9820 | **0.9820** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 1.0000 | **1.0000** | 172 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (0 false positives) |
| **Micro-Average** | **0.9903** | **0.7841** | **0.8752** | **1,172** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9940** | **0.8313** | **0.8864** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7455 | 0.6235 | 0.6648 | — | All schema classes (incl. inactive) |

### MEMM Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.9820 | **0.9909** | 500 | Active gold class |
| `DISEASE` | 0.9519 | 0.9500 | **0.9510** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 0.8837 | **0.9383** | 172 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (17 false positives) |
| **Micro-Average** | **0.9646** | **0.9539** | **0.9592** | **1,172** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9840** | **0.9386** | **0.9601** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7380 | 0.7039 | 0.7200 | — | All schema classes (incl. inactive) |

### CRF Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.9960 | **0.9980** | 500 | Active gold class |
| `DISEASE` | 0.9183 | 0.9220 | **0.9202** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 0.7733 | **0.8721** | 172 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (36 false positives) |
| **Micro-Average** | **0.9341** | **0.9317** | **0.9329** | **1,172** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9728** | **0.8971** | **0.9301** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7296 | 0.6728 | 0.6976 | — | All schema classes (incl. inactive) |

---

## 4. Track C: Combined Augmented Test Benchmark
*Evaluates joint 4-class extraction over the complete benchmark (6,365 sentences, 122,729 tokens: concatenation of Track A [116,318 tokens] + Track B [6,411 tokens]). Track C re-evaluates the combined test examples from Tracks A and B.*

### HMM Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.7060 | 0.6895 | **0.6976** | 4,924 |
| `MEDICATION` | 0.9400 | 0.8498 | **0.8927** | 2,417 |
| `PATIENT_NAME` | 0.8000 | 0.5120 | **0.6244** | 500 |
| `CHEMICAL` | 0.8532 | 0.3943 | **0.5393** | 3,140 |
| **Micro-Average** | **0.7921** | **0.6323** | **0.7032** | **10,981** |
| **Macro-Average (4-Class)** | **0.8248** | **0.6114** | **0.6885** | — |

### MEMM Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.8236 | 0.7321 | **0.7752** | 4,924 |
| `MEDICATION` | 0.9140 | 0.9189 | **0.9164** | 2,417 |
| `PATIENT_NAME` | 0.9590 | 0.9820 | **0.9704** | 500 |
| `CHEMICAL` | 0.8288 | 0.6061 | **0.7001** | 3,140 |
| **Micro-Average** | **0.8549** | **0.7486** | **0.7982** | **10,981** |
| **Macro-Average (4-Class)** | **0.8814** | **0.8098** | **0.8405** | — |

### CRF Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.8252 | 0.7524 | **0.7871** | 4,924 |
| `MEDICATION` | 0.9252 | 0.9160 | **0.9206** | 2,417 |
| `PATIENT_NAME` | 0.9614 | 0.9960 | **0.9784** | 500 |
| `CHEMICAL` | 0.8217 | 0.6137 | **0.7026** | 3,140 |
| **Micro-Average** | **0.8561** | **0.7599** | **0.8051** | **10,981** |
| **Macro-Average (4-Class)** | **0.8834** | **0.8195** | **0.8472** | — |

---

## 5. Architectural Comparison & In-Depth Discussion

### A. Hidden Markov Model (HMM)
- **Generative Formulation:** Models joint probability $P(X, Y) = \prod_t P(y_t \mid y_{t-1}) P(x_t \mid y_t)$.
- **Strengths:** Extremely fast parameter estimation (< 1s) via counting; minimal memory requirements.
- **Limitations:** Strict conditional independence assumption (tokens depend only on current state); cannot easily incorporate overlapping, multi-token, or sub-word features beyond morphological signatures.

### B. Maximum Entropy Markov Model (MEMM)
- **Discriminative Formulation:** Models local conditional probabilities $P(y_t \mid X, t, y_{t-1})$ using multinomial logistic regression.
- **Strengths:** Exploits arbitrary overlapping features (prefixes, suffixes, capitalization, context windows, word shapes).
- **Limitations:** Suffers from the **Label Bias Problem**: states with low entropy transition distributions dominate Viterbi decoding, as probability is locally normalized at each step rather than globally across the sequence.

### C. Linear-Chain Conditional Random Field (CRF)
- **Global Normalization:** Models $P(Y \mid X) = \frac{1}{Z(X)} \exp\left(\sum_t \sum_k w_k f_k(y_{t-1}, y_t, X, t)\right)$.
- **Strengths:** Resolves the label bias problem through global partition function normalization. Typically improves precision and recall over locally normalized models by balancing state and transition features jointly across the complete sentence.

### D. Empirical Findings (computed from this run)
No single model achieves the highest entity-level micro F1 on every evaluation track. Best micro F1 per track: Track A (BC5CDR): **CRF**; Track B (Synthetic): **MEMM**; Track C (Combined): **CRF**.

---

## 6. Reproducibility & Commands

```bash
# Run all model training, evaluation, and reporting
PYTHONPATH=. .venv/bin/python3 src/models/train_and_evaluate.py

# Run model test suite
PYTHONPATH=. .venv/bin/pytest tests/test_models.py -v
```
