# Medical NER Model Comparison Report: HMM vs. MEMM vs. CRF

**Evaluation Target:** 4-Entity Sequence Labeling (`DISEASE`, `MEDICATION`, `PATIENT_NAME`, `CHEMICAL`)  
**Training Corpus:** Combined Augmented Training Set (9,228 sentences: 5,228 BC5CDR + 4,000 Synthetic)  
**Validation Corpus:** Combined Augmented Validation Set (5,830 sentences: 5,330 BC5CDR + 500 Synthetic)  
**Evaluation Tracks:**
- **Track A (Pure Gold Biomedical Benchmark):** BC5CDR Original Test Slice (5,865 sentences, 116,318 tokens, 100% human gold standard)
- **Track B (Synthetic Clinical Name Benchmark):** Synthetic Test Slice (500 sentences, 7,680 tokens, fictional clinical EHR templates)
- **Track C (Combined Augmented Test Benchmark):** Joint Evaluation (6,365 sentences, 123,998 tokens: union of Track A + Track B)

### Corpus & Evaluation Token Accounting

| Evaluation Scope | Sentences | Tokens | Scope Description |
| :--- | :---: | :---: | :--- |
| **Track A (BC5CDR Original Test)** | 5,865 | 116,318 | Authentic PubMed literature abstracts |
| **Track B (Synthetic Clinical Test)** | 500 | 7,680 | Fictional clinical EHR templates |
| **Track C (Combined Augmented Test)** | 6,365 | 123,998 | Concatenation of Track A + Track B |
| **Token evaluations across Tracks A+B+C** | — | **247,996** | Track C re-evaluates the tokens of Tracks A and B |
| **Validation Partition** | 5,830 | 116,196 | Augmented validation split |
| **Training Partition** | 9,228 | 162,306 | Augmented training split |
| **Full Project Corpus (Train+Val+Test)** | **21,423** | **402,500** | All tokens across the augmented splits |

---

## 1. Executive Summary & Comparative Highlights

| Model | Architecture Type | Training Time | Track A F1 (BC5CDR) | Track B F1 (Synthetic) | Track C F1 (Combined) | Token Accuracy (Track C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM** | Generative (Joint MLE) | 0.24s | **0.6838** | **0.7548** | **0.6916** | 94.15% |
| **MEMM** | Discriminative (Local Cond. MaxEnt) | 63.07s | **0.7776** | **0.9301** | **0.7968** | 96.22% |
| **CRF** | Discriminative (Global Linear-Chain CRF) | 46.15s | **0.7905** | **0.8883** | **0.8029** | 96.44% |

---

## 2. Track A: Pure Gold Evaluation (BC5CDR Original Test Partition)
*Evaluates model generalization strictly on authentic PubMed scientific literature abstracts (5,865 sentences, 116,318 tokens).*

### HMM Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.6729 | 0.6555 | **0.6641** | 4,424 | Active gold class |
| `MEDICATION` | 0.9354 | 0.8383 | **0.8842** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8472 | 0.3920 | **0.5360** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (4 false positives) |
| **Micro-Average** | **0.7730** | **0.6130** | **0.6838** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8185** | **0.6286** | **0.6948** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6139 | 0.4715 | 0.5211 | — | All schema classes (incl. inactive) |

### MEMM Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.8103 | 0.7030 | **0.7528** | 4,424 | Active gold class |
| `MEDICATION` | 0.9064 | 0.9234 | **0.9148** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8320 | 0.6102 | **0.7040** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (22 false positives) |
| **Micro-Average** | **0.8401** | **0.7237** | **0.7776** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8496** | **0.7455** | **0.7905** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6372 | 0.5591 | 0.5929 | — | All schema classes (incl. inactive) |

### CRF Performance on Pure BC5CDR Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `DISEASE` | 0.8190 | 0.7344 | **0.7744** | 4,424 | Active gold class |
| `MEDICATION` | 0.9174 | 0.9256 | **0.9215** | 2,245 | Active gold class |
| `CHEMICAL` | 0.8388 | 0.6150 | **0.7097** | 3,140 | Active gold class |
| `PATIENT_NAME` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (19 false positives) |
| **Micro-Average** | **0.8486** | **0.7399** | **0.7905** | **9,809** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.8584** | **0.7583** | **0.8019** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.6438 | 0.5688 | 0.6014 | — | All schema classes (incl. inactive) |

---

## 3. Track B: Synthetic Evaluation (Synthetic Test Slice)
*Evaluates fictional `PATIENT_NAME` recognition alongside medical entities on unseen clinical EHR templates (500 sentences, 7,680 tokens).*

> [!WARNING]
> **Clinical Non-Equivalence Disclaimer:** Synthetic test performance demonstrates sequence labeling on controlled syntactic structures. It must **never** be cited as real-world clinical EHR extraction accuracy on unredacted medical notes.

### HMM Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 0.0000 | 0.0000 | **0.0000** | 500 | Active gold class |
| `DISEASE` | 0.9660 | 0.9660 | **0.9660** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 1.0000 | **1.0000** | 339 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (0 false positives) |
| **Micro-Average** | **0.9797** | **0.6139** | **0.7548** | **1,339** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.6553** | **0.6553** | **0.6553** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.4915 | 0.4915 | 0.4915 | — | All schema classes (incl. inactive) |

### MEMM Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.9160 | **0.9562** | 500 | Active gold class |
| `DISEASE` | 0.8940 | 0.8940 | **0.8940** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 0.9410 | **0.9696** | 339 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (16 false positives) |
| **Micro-Average** | **0.9466** | **0.9141** | **0.9301** | **1,339** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9647** | **0.9170** | **0.9399** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7235 | 0.6877 | 0.7049 | — | All schema classes (incl. inactive) |

### CRF Performance on Synthetic Clinical Test Slice
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `PATIENT_NAME` | 1.0000 | 0.9520 | **0.9754** | 500 | Active gold class |
| `DISEASE` | 0.8540 | 0.8540 | **0.8540** | 500 | Active gold class |
| `MEDICATION` | 1.0000 | 0.8083 | **0.8940** | 339 | Active gold class |
| `CHEMICAL` | 0.0000 | 0.0000 | 0.0000 | 0 | Inactive class (61 false positives) |
| **Micro-Average** | **0.8978** | **0.8790** | **0.8883** | **1,339** | Pooled span counts |
| **Macro-Average (Active 3-Class)** | **0.9513** | **0.8714** | **0.9078** | — | Unweighted mean of active classes |
| **Macro-Average (Full 4-Class)** | 0.7135 | 0.6536 | 0.6808 | — | All schema classes (incl. inactive) |

---

## 4. Track C: Combined Augmented Test Benchmark
*Evaluates joint 4-class extraction over the complete benchmark (6,365 sentences, 123,998 tokens: concatenation of Track A [116,318 tokens] + Track B [7,680 tokens]). Track C re-evaluates the combined test examples from Tracks A and B.*

### HMM Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.7033 | 0.6870 | **0.6951** | 4,924 |
| `MEDICATION` | 0.9447 | 0.8595 | **0.9001** | 2,584 |
| `PATIENT_NAME` | 0.0000 | 0.0000 | **0.0000** | 500 |
| `CHEMICAL` | 0.8472 | 0.3920 | **0.5360** | 3,140 |
| **Micro-Average** | **0.7931** | **0.6131** | **0.6916** | **11,148** |
| **Macro-Average (4-Class)** | **0.6238** | **0.4846** | **0.5328** | — |

### MEMM Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.8200 | 0.7224 | **0.7681** | 4,924 |
| `MEDICATION` | 0.9179 | 0.9257 | **0.9218** | 2,584 |
| `PATIENT_NAME` | 0.9542 | 0.9160 | **0.9347** | 500 |
| `CHEMICAL` | 0.8262 | 0.6102 | **0.7020** | 3,140 |
| **Micro-Average** | **0.8543** | **0.7466** | **0.7968** | **11,148** |
| **Macro-Average (4-Class)** | **0.8796** | **0.7936** | **0.8317** | — |

### CRF Performance on Combined Benchmark
| Entity Class | Precision | Recall | F1 Score | Support (Gold Spans) |
| :--- | :---: | :---: | :---: | :---: |
| `DISEASE` | 0.8229 | 0.7465 | **0.7829** | 4,924 |
| `MEDICATION` | 0.9263 | 0.9102 | **0.9182** | 2,584 |
| `PATIENT_NAME` | 0.9616 | 0.9520 | **0.9568** | 500 |
| `CHEMICAL` | 0.8172 | 0.6150 | **0.7018** | 3,140 |
| **Micro-Average** | **0.8551** | **0.7566** | **0.8029** | **11,148** |
| **Macro-Average (4-Class)** | **0.8820** | **0.8059** | **0.8399** | — |

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
