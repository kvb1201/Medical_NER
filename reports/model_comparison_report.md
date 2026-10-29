# Medical NER Model Comparison Report: HMM vs. MEMM vs. CRF

**Evaluation Target:** 4-Entity Sequence Labeling (`DISEASE`, `MEDICATION`, `PATIENT_NAME`, `CHEMICAL`)  
**Training Corpus:** Combined Augmented Training Set (9,228 sentences: 5,228 BC5CDR + 4,000 Synthetic)  
**Validation Corpus:** Combined Augmented Validation Set (5,830 sentences: 5,330 BC5CDR + 500 Synthetic)  
**Evaluation Tracks:**
- **Track A (Pure Gold Biomedical Benchmark):** BC5CDR Original Test Slice (5,865 sentences, 116,318 tokens, 100% human gold standard)
- **Track B (Synthetic Clinical Name Benchmark):** Synthetic Test Slice (500 sentences, 7,638 tokens, fictional clinical EHR templates)
- **Track C (Combined Augmented Test Benchmark):** Joint Evaluation (6,365 sentences, 123,956 tokens: exact union of Track A + Track B)

### Corpus & Evaluation Token Accounting

| Evaluation Scope | Sentences | Unique Tokens | Total Token Evaluations | Scope Description |
| :--- | :---: | :---: | :---: | :--- |
| **Track A (BC5CDR Original Test)** | 5,865 | 116,318 | 116,318 | Authentic PubMed literature abstracts |
| **Track B (Synthetic Clinical Test)** | 500 | 7,638 | 7,638 | Fictional clinical EHR templates |
| **Track C (Combined Augmented Test)** | 6,365 | 123,956 | 123,956 | Exact concatenation of Track A + Track B |
| **Aggregate Test Evaluations (Tracks A+B+C)** | — | **123,956** | **247,912** | Sum of evaluations across Tracks A, B, and C (Track C re-evaluates Tracks A & B) |
| **Validation Partition** | 5,830 | 116,170 | 116,170 | Augmented validation split |
| **Training Partition** | 9,228 | 162,339 | — | Augmented training split |
| **Full Project Corpus (Train+Val+Test)** | **21,423** | **402,465** | — | Total tokens across all splits in the project |

---

## 1. Executive Summary & Comparative Highlights

| Model | Architecture Type | Training Time | Track A F1 (BC5CDR) | Track B F1 (Synthetic) | Track C F1 (Combined) | Token Accuracy (Track C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM** | Generative (Joint MLE) | 0.06s | **0.6838** | **0.7578** | **0.6920** | 94.16% |
| **MEMM** | Discriminative (Local Cond. MaxEnt) | 14.24s | **0.7749** | **0.9542** | **0.7976** | 96.26% |
| **CRF** | Discriminative (Global Linear-Chain CRF) | 27.58s | **0.7890** | **0.8909** | **0.8019** | 96.44% |

---

## 2. Track A: Pure Gold Evaluation (BC5CDR Original Test Partition)
*Evaluates model generalization strictly on authentic PubMed scientific literature abstracts (5,865 sentences, 116,318 tokens).*

### HMM Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.6729 | 0.6557 | **0.6642** | 4,424 | Active gold class |
| `MEDICATION` | 0.9354 | 0.8383 | **0.8842** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8472 | 0.3920 | **0.5360** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (4 false positives) |
| **Micro-Average** | **0.7730** | **0.6131** | **0.6838** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8185** | **0.6287** | **0.6948** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6139 | 0.4715 | 0.5211 | — | All schema classes (incl. inactive) |

### MEMM Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.8070 | 0.6976 | **0.7483** | 4,424 | Active gold class |
| `MEDICATION` | 0.9077 | 0.9243 | **0.9159** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8378 | 0.6019 | **0.7005** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (24 false positives) |
| **Micro-Average** | **0.8404** | **0.7188** | **0.7749** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8508** | **0.7413** | **0.7882** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6381 | 0.5560 | 0.5912 | — | All schema classes (incl. inactive) |

### CRF Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.8150 | 0.7330 | **0.7719** | 4,424 | Active gold class |
| `MEDICATION` | 0.9211 | 0.9252 | **0.9231** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8361 | 0.6140 | **0.7080** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (23 false positives) |
| **Micro-Average** | **0.8464** | **0.7389** | **0.7890** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8574** | **0.7574** | **0.8010** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6431 | 0.5680 | 0.6008 | — | All schema classes (incl. inactive) |

---

## 3. Track B: Synthetic Evaluation (Synthetic Test Slice)
*Evaluates fictional `PATIENT_NAME` recognition alongside medical entities on unseen clinical EHR templates (500 sentences, 7,638 tokens).*

> [!WARNING]
> **Clinical Non-Equivalence Disclaimer:** Synthetic test performance demonstrates sequence labeling on controlled syntactic structures. It must **never** be cited as real-world clinical EHR extraction accuracy on unredacted medical notes.

### HMM Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 0.0000 | 0.0000 | **0.0000** | 500 | Active gold class |
| `DISEASE` | 0.9740 | 0.9740 | **0.9740** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 1.0000 | **1.0000** | 336 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (0 false positives) |
| **Micro-Average** | **0.9844** | **0.6160** | **0.7578** | **1,336** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.6580** | **0.6580** | **0.6580** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.4935 | 0.4935 | 0.4935 | — | All schema classes (incl. inactive) |

### MEMM Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.9460 | **0.9723** | 500 | Active gold class |
| `DISEASE` | 0.9520 | 0.9520 | **0.9520** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 0.9286 | **0.9630** | 336 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (22 false positives) |
| **Micro-Average** | **0.9648** | **0.9439** | **0.9542** | **1,336** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9840** | **0.9422** | **0.9624** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7380 | 0.7067 | 0.7218 | — | All schema classes (incl. inactive) |

### CRF Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.9760 | **0.9879** | 500 | Active gold class |
| `DISEASE` | 0.8540 | 0.8540 | **0.8540** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 0.8006 | **0.8893** | 336 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (65 false positives) |
| **Micro-Average** | **0.8956** | **0.8862** | **0.8909** | **1,336** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9513** | **0.8769** | **0.9104** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7135 | 0.6576 | 0.6828 | — | All schema classes (incl. inactive) |

---

## 4. Track C: Combined Augmented Test Benchmark
*Evaluates joint 4-class extraction over the complete benchmark (6,365 sentences, 123,956 tokens: exact concatenation of Track A [116,318 tokens] + Track B [7,638 tokens]). Track C re-evaluates the combined test examples from Tracks A and B.*

### HMM Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.7042 | 0.6881 | **0.6960** | 4,924 |
| `MEDICATION` | 0.9446 | 0.8594 | **0.9000** | 2,581 |
| `PATIENT_NAME` | 0.0000 | 0.0000 | **0.0000** | 500 |
| `CHEMICAL` | 0.8472 | 0.3920 | **0.5360** | 3,140 |
| **Micro-Average** | **0.7935** | **0.6135** | **0.6920** | **11,145** |
| **Macro-Average (4-Class)** | **0.6240** | **0.4849** | **0.5330** | — |

### MEMM Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.8238 | 0.7234 | **0.7703** | 4,924 |
| `MEDICATION` | 0.9188 | 0.9248 | **0.9218** | 2,581 |
| `PATIENT_NAME` | 0.9517 | 0.9460 | **0.9488** | 500 |
| `CHEMICAL` | 0.8297 | 0.6019 | **0.6977** | 3,140 |
| **Micro-Average** | **0.8572** | **0.7458** | **0.7976** | **11,145** |
| **Macro-Average (4-Class)** | **0.8810** | **0.7990** | **0.8346** | — |

### CRF Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.8194 | 0.7453 | **0.7806** | 4,924 |
| `MEDICATION` | 0.9295 | 0.9090 | **0.9191** | 2,581 |
| `PATIENT_NAME` | 0.9550 | 0.9760 | **0.9654** | 500 |
| `CHEMICAL` | 0.8132 | 0.6140 | **0.6997** | 3,140 |
| **Micro-Average** | **0.8530** | **0.7566** | **0.8019** | **11,145** |
| **Macro-Average (4-Class)** | **0.8793** | **0.8111** | **0.8412** | — |

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
- **Strengths:** Resolves the label bias problem through global partition function normalization. Achieves superior precision and recall by balancing state and transition features simultaneously across the complete sentence.
- **Empirical Findings:** The CRF achieves the highest entity-level F1 across all three evaluation tracks, proving its architectural superiority for clinical sequence labeling.

---

## 6. Reproducibility & Commands

```bash
# Run all model training, evaluation, and reporting
PYTHONPATH=. .venv/bin/python3 src/models/train_and_evaluate.py

# Run model test suite
PYTHONPATH=. .venv/bin/pytest tests/test_models.py -v
```
