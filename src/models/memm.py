#!/usr/bin/env python3
"""
Module: memm.py
Maximum Entropy Markov Model (MEMM) for Named Entity Recognition.

Implements:
  1. Rich morphological, orthographic, and context window feature extraction.
  2. Maximum Entropy (Multinomial Logistic Regression with L2 regularization) parameter estimation.
  3. Strict separation of transition features (prev_tag) and observation features for
     fast vectorized inference.
  4. Exact Viterbi decoding over conditional log-probability distributions with BIO transition constraints.
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy import sparse
from scipy.special import log_softmax
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.hmm import is_valid_bio_transition

logger = logging.getLogger(__name__)

def word_shape(text: str) -> str:
    """Map token to shape signature (e.g., 'Metformin' -> 'Xxxxx', '5-HT' -> 'd-XX')."""
    shape = re.sub(r"[A-Z]", "X", text)
    shape = re.sub(r"[a-z]", "x", shape)
    shape = re.sub(r"[0-9]", "d", shape)
    # Collapse consecutive identical characters
    return re.sub(r"(.)\1{2,}", r"\1\1", shape)

def extract_token_observation_features(tokens: List[str], i: int) -> Dict[str, Any]:
    """Extract observation features for token i (independent of previous tag)."""
    word = tokens[i]
    w_lower = word.lower()
    n = len(tokens)

    feats: Dict[str, Any] = {
        "bias": 1.0,
        "w": w_lower,
        "shape": word_shape(word),
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "has_digit": any(c.isdigit() for c in word),
        "has_hyphen": "-" in word,
        "is_punct": not re.search(r"\w", word),
        "is_first": i == 0,
        "is_last": i == n - 1
    }

    # Morphological prefixes and suffixes (crucial for medical stems)
    if len(w_lower) >= 2:
        feats["pref_2"] = w_lower[:2]
        feats["suff_2"] = w_lower[-2:]
    if len(w_lower) >= 3:
        feats["pref_3"] = w_lower[:3]
        feats["suff_3"] = w_lower[-3:]
    if len(w_lower) >= 4:
        feats["suff_4"] = w_lower[-4:]

    # Left context (window -1, -2)
    if i > 0:
        prev_w = tokens[i - 1]
        feats["w-1"] = prev_w.lower()
        feats["w-1:shape"] = word_shape(prev_w)
        feats["w-1:is_title"] = prev_w.istitle()
        feats["bi_left"] = f"{prev_w.lower()}_{w_lower}"
    else:
        feats["BOS"] = True

    if i > 1:
        feats["w-2"] = tokens[i - 2].lower()

    # Right context (window +1, +2)
    if i < n - 1:
        next_w = tokens[i + 1]
        feats["w+1"] = next_w.lower()
        feats["w+1:shape"] = word_shape(next_w)
        feats["w+1:is_title"] = next_w.istitle()
        feats["bi_right"] = f"{w_lower}_{next_w.lower()}"
    else:
        feats["EOS"] = True

    if i < n - 2:
        feats["w+2"] = tokens[i + 2].lower()

    return feats

class MaximumEntropyMarkovModel:
    """
    Maximum Entropy Markov Model (MEMM) for sequence labeling.
    Uses multinomial logistic regression with L2 regularization and Viterbi decoding.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 100, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state
        )
        self.vectorizer = DictVectorizer(sparse=True)

        self.states: List[str] = []
        self.state2id: Dict[str, int] = {}
        self.id2state: Dict[int, str] = {}

        # Decomposed linear model weights for O(1) inference per transition
        self.W_obs: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.trans_logits: Optional[np.ndarray] = None  # shape (num_prev_tags, num_states)
        self.prev_tag_index_map: Dict[str, int] = {}

    def fit(self, sentences: List[List[str]], tag_sequences: List[List[str]]) -> "MaximumEntropyMarkovModel":
        """
        Extract observation and transition features and train the Maximum Entropy classifier.
        """
        all_states = sorted(list(set(t for seq in tag_sequences for t in seq)))
        if "O" in all_states:
            all_states.remove("O")
            self.states = ["O"] + all_states
        else:
            self.states = all_states

        self.state2id = {s: i for i, s in enumerate(self.states)}
        self.id2state = {i: s for i, s in enumerate(self.states)}

        feature_dicts = []
        labels = []

        for sent, tags in zip(sentences, tag_sequences):
            n = len(sent)
            for i in range(n):
                prev_tag = tags[i - 1] if i > 0 else "__BOS__"
                feats = extract_token_observation_features(sent, i)
                feats["prev_tag"] = prev_tag
                feature_dicts.append(feats)
                labels.append(tags[i])

        logger.info(f"MEMM: Vectorizing {len(feature_dicts)} tokens...")
        X = self.vectorizer.fit_transform(feature_dicts)
        y = np.array([self.state2id[lbl] for lbl in labels], dtype=np.int32)

        logger.info(f"MEMM: Fitting LogisticRegression on matrix {X.shape} with {len(self.states)} classes...")
        self.classifier.fit(X, y)

        # Precompute decomposed weights for fast Viterbi decoding
        # classifier.coef_ has shape (n_classes, n_features)
        # classifier.intercept_ has shape (n_classes,)
        classes = self.classifier.classes_
        K = len(self.states)
        
        # Align class rows if classes_ is not strictly 0..K-1
        W = np.zeros((K, X.shape[1]), dtype=np.float32)
        b = np.zeros(K, dtype=np.float32)
        for row_idx, cls_id in enumerate(classes):
            W[cls_id, :] = self.classifier.coef_[row_idx]
            b[cls_id] = self.classifier.intercept_[row_idx]

        self.b = b

        # Partition feature indices into observation vs prev_tag
        prev_tags_list = ["__BOS__"] + self.states
        self.prev_tag_index_map = {tag: idx for idx, tag in enumerate(prev_tags_list)}
        
        # Precompute transition logits: shape (num_prev_tags, K)
        num_prev = len(prev_tags_list)
        self.trans_logits = np.zeros((num_prev, K), dtype=np.float32)

        feat_names = self.vectorizer.feature_names_
        prev_feat_indices = {}
        for idx, name in enumerate(feat_names):
            if name.startswith("prev_tag="):
                val = name[len("prev_tag="):]
                prev_feat_indices[val] = idx

        for val, p_idx in self.prev_tag_index_map.items():
            if val in prev_feat_indices:
                f_col = prev_feat_indices[val]
                self.trans_logits[p_idx, :] = W[:, f_col]

        # Zero out the prev_tag columns in W so W_obs represents strictly observation features
        for f_col in prev_feat_indices.values():
            W[:, f_col] = 0.0

        self.W_obs = W  # shape (K, n_features)
        logger.info("MEMM: Parameter estimation and weight decomposition completed successfully.")
        return self

    def decode_trellis(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Run Viterbi and return the real trellis used for decoding.

        ``scores[t][s]`` is the max path log-probability ending in state ``s``
        at token ``t``. Impossible cells are ``None``.
        """
        if not tokens:
            return {
                "kind": "viterbi",
                "unit": "log_probability",
                "tokens": [],
                "states": list(self.states),
                "scores": [],
                "path": [],
                "tags": [],
            }

        T = len(tokens)
        K = len(self.states)

        # 1. Extract observation features for all tokens in sentence and vectorize
        obs_dicts = [extract_token_observation_features(tokens, i) for i in range(T)]
        X_obs = self.vectorizer.transform(obs_dicts)  # sparse (T, n_features)

        # Compute observation logits: shape (T, K)
        # obs_logits = X_obs @ W_obs.T + b
        obs_logits = (X_obs.dot(self.W_obs.T)) + self.b  # shape (T, K)

        # Trellis for Viterbi
        V = np.full((T, K), -1e9, dtype=np.float32)
        bp = np.zeros((T, K), dtype=np.int32)

        # Step t = 0: prev_tag is __BOS__
        bos_idx = self.prev_tag_index_map["__BOS__"]
        first_logits = obs_logits[0] + self.trans_logits[bos_idx]
        first_log_probs = log_softmax(first_logits)

        for s in range(K):
            tag_name = self.id2state[s]
            if is_valid_bio_transition(None, tag_name):
                V[0, s] = first_log_probs[s]
            else:
                V[0, s] = -1e9

        # Recursion t = 1 to T - 1
        for t in range(1, T):
            t_obs = obs_logits[t]
            for prev_s in range(K):
                prev_tag = self.id2state[prev_s]
                prev_val = V[t - 1, prev_s]
                if prev_val <= -1e8:
                    continue

                prev_idx = self.prev_tag_index_map[prev_tag]
                cond_logits = t_obs + self.trans_logits[prev_idx]
                cond_log_probs = log_softmax(cond_logits)

                for curr_s in range(K):
                    curr_tag = self.id2state[curr_s]
                    if not is_valid_bio_transition(prev_tag, curr_tag):
                        continue

                    cand = prev_val + cond_log_probs[curr_s]
                    if cand > V[t, curr_s]:
                        V[t, curr_s] = cand
                        bp[t, curr_s] = prev_s

        # Termination
        best_last_state = int(np.argmax(V[T - 1]))

        # Backtracking
        best_path = [best_last_state]
        curr = best_last_state
        for t in range(T - 1, 0, -1):
            curr = bp[t, curr]
            best_path.append(curr)

        best_path.reverse()
        scores = []
        for t in range(T):
            row = []
            for s in range(K):
                val = float(V[t, s])
                row.append(None if val <= -1e8 else val)
            scores.append(row)
        path = [int(s) for s in best_path]
        return {
            "kind": "viterbi",
            "unit": "log_probability",
            "tokens": list(tokens),
            "states": list(self.states),
            "scores": scores,
            "path": path,
            "tags": [self.id2state[s] for s in path],
        }

    def decode_sentence(self, tokens: List[str]) -> List[str]:
        """
        Viterbi algorithm over conditional log-probabilities log P(y_t | X, t, y_{t-1}).
        Enforces strict BIO transition constraints.
        """
        return self.decode_trellis(tokens)["tags"]

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Decode BIO tag sequences for a batch of sentences."""
        return [self.decode_sentence(sent) for sent in sentences]

