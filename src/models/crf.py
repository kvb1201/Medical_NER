#!/usr/bin/env python3
"""
Module: crf.py
Linear-Chain Conditional Random Field (CRF) for Named Entity Recognition.

Implements:
  1. Global sequence-level normalized conditional distribution P(y | x).
  2. Identical linguistic, morphological, and context-window feature extraction as MEMM
     to ensure rigorous apples-to-apples baseline comparison.
  3. L-BFGS optimization with elastic net regularization (L1 + L2).
  4. Global Viterbi sequence decoding resolving the label bias problem.
"""

import re
import logging
from typing import List, Dict, Any
import sklearn_crfsuite

from src.models.memm import extract_token_observation_features

logger = logging.getLogger(__name__)

def sentence_to_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """Extract sequence of token observation features for CRF."""
    return [extract_token_observation_features(tokens, i) for i in range(len(tokens))]

class LinearChainCRF:
    """
    Linear-Chain Conditional Random Field model for sequence labeling.
    Uses L-BFGS parameter estimation with L1 and L2 regularization.
    """
    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        random_state: int = 42
    ):
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
            all_possible_states=True
        )
        self.states: List[str] = []

    def fit(self, sentences: List[List[str]], tag_sequences: List[List[str]]) -> "LinearChainCRF":
        """
        Extract sequence features and fit the CRF via L-BFGS optimization.
        """
        self.states = sorted(list(set(t for seq in tag_sequences for t in seq)))
        logger.info(f"CRF: Extracting sequence features for {len(sentences)} sentences...")
        X = [sentence_to_features(sent) for sent in sentences]
        y = tag_sequences

        logger.info(f"CRF: Training L-BFGS optimizer (c1={self.c1}, c2={self.c2}, max_iter={self.max_iterations})...")
        self.model.fit(X, y)
        logger.info("CRF: Model training completed successfully.")
        return self

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predict BIO tag sequences for a batch of sentences."""
        X = [sentence_to_features(sent) for sent in sentences]
        return self.model.predict(X)

    def decode_trellis(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Return the CRF Viterbi path plus real forward-backward tag marginals.

        sklearn-crfsuite does not expose the internal Viterbi cell scores, so
        ``scores[t][s]`` here is P(y_t = s | x) from ``predict_marginals``, not
        a Viterbi path score. ``path`` / ``tags`` still come from Viterbi
        (``predict``), so the highlighted path is the exact decoding used for NER.
        """
        if not tokens:
            states = list(self.states) if self.states else sorted(self.model.classes_)
            return {
                "kind": "marginals",
                "unit": "probability",
                "tokens": [],
                "states": states,
                "scores": [],
                "path": [],
                "tags": [],
            }

        feats = sentence_to_features(tokens)
        tags = [str(tag) for tag in self.model.predict([feats])[0]]
        marginals = self.model.predict_marginals([feats])[0]
        states = [str(s) for s in sorted(self.model.classes_)]
        state2id = {s: i for i, s in enumerate(states)}
        scores = [[float(token_marg.get(state, 0.0)) for state in states] for token_marg in marginals]
        path = [int(state2id[tag]) for tag in tags]
        return {
            "kind": "marginals",
            "unit": "probability",
            "tokens": list(tokens),
            "states": states,
            "scores": scores,
            "path": path,
            "tags": tags,
        }

