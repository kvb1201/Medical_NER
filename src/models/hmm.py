#!/usr/bin/env python3
"""
Module: hmm.py
First-Order Hidden Markov Model (HMM) for Named Entity Recognition.

Implements:
  1. Supervised Maximum Likelihood Estimation (MLE) of transition, emission,
     and initial state distributions.
  2. Additive (Laplace/Lidstone) smoothing.
  3. Morphological sub-word signature mapping for unseen tokens (OOV).
  4. Exact Viterbi decoding in log-probability space with BIO transition constraints.
"""

import math
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# BIO constraint checking helper
def is_valid_bio_transition(prev_tag: Optional[str], curr_tag: str) -> bool:
    """Check whether a transition prev_tag -> curr_tag obeys strict BIO grammar."""
    if curr_tag.startswith("I-"):
        curr_entity = curr_tag[2:]
        if prev_tag is None:
            # Cannot start sequence with an I- tag
            return False
        valid_prev = {f"B-{curr_entity}", f"I-{curr_entity}"}
        return prev_tag in valid_prev
    return True

def get_word_signature(word: str) -> str:
    """
    Map an out-of-vocabulary or rare word to a morphological signature
    to preserve capitalization, digits, and suffix clues for the HMM emission model.
    """
    if word.isupper():
        return "<UNK_ALL_CAPS>"
    elif word.istitle():
        return "<UNK_TITLE>"
    elif any(c.isdigit() for c in word):
        return "<UNK_NUM>"
    elif "-" in word:
        return "<UNK_DASH>"
    elif word.endswith(("itis", "osis", "emia", "oma", "pathy", "ia")):
        return "<UNK_SUFFIX_DISEASE>"
    elif word.endswith(("olol", "statin", "cillin", "mycin", "azole", "pril", "pine")):
        return "<UNK_SUFFIX_DRUG>"
    elif word.islower():
        return "<UNK_LOWER>"
    else:
        return "<UNK>"

class HiddenMarkovModel:
    """
    Supervised Hidden Markov Model for sequence labeling with Viterbi decoding.
    """
    def __init__(
        self,
        alpha_trans: float = 0.01,
        alpha_emit: float = 0.001,
        alpha_init: float = 0.01,
        min_word_freq: int = 2
    ):
        self.alpha_trans = alpha_trans
        self.alpha_emit = alpha_emit
        self.alpha_init = alpha_init
        self.min_word_freq = min_word_freq

        self.states: List[str] = []
        self.state2id: Dict[str, int] = {}
        self.id2state: Dict[int, str] = {}
        self.vocab: Dict[str, int] = {}
        self.signatures: List[str] = [
            "<UNK_ALL_CAPS>", "<UNK_TITLE>", "<UNK_NUM>", "<UNK_DASH>",
            "<UNK_SUFFIX_DISEASE>", "<UNK_SUFFIX_DRUG>", "<UNK_LOWER>", "<UNK>"
        ]

        # Log probability tables
        self.log_init: List[float] = []
        self.log_trans: List[List[float]] = []
        self.log_emit: Dict[str, List[float]] = {}
        self.log_emit_signatures: Dict[str, List[float]] = {}

    def fit(self, sentences: List[List[str]], tag_sequences: List[List[str]]) -> "HiddenMarkovModel":
        """
        Estimate HMM parameters using Maximum Likelihood Estimation with smoothing.
        Strictly learns only from the provided training split.
        """
        # 1. Establish state space
        unique_states = sorted(list(set(t for seq in tag_sequences for t in seq)))
        # Ensure 'O' is first if present
        if "O" in unique_states:
            unique_states.remove("O")
            self.states = ["O"] + unique_states
        else:
            self.states = unique_states

        self.state2id = {s: i for i, s in enumerate(self.states)}
        self.id2state = {i: s for i, s in enumerate(self.states)}
        K = len(self.states)

        # 2. Vocabulary count and rare word replacement
        raw_word_counts = Counter(w for sent in sentences for w in sent)
        self.vocab = {}
        for w, c in raw_word_counts.items():
            if c >= self.min_word_freq:
                self.vocab[w] = len(self.vocab)

        # 3. Accumulate frequency counts
        init_counts = [0] * K
        trans_counts = [[0] * K for _ in range(K)]
        emit_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * K)
        state_totals = [0] * K

        for sent, tags in zip(sentences, tag_sequences):
            if not sent:
                continue

            # Initial state
            first_state_id = self.state2id[tags[0]]
            init_counts[first_state_id] += 1

            for t, (tok, tag) in enumerate(zip(sent, tags)):
                state_id = self.state2id[tag]
                state_totals[state_id] += 1

                # Transition count
                if t > 0:
                    prev_state_id = self.state2id[tags[t - 1]]
                    trans_counts[prev_state_id][state_id] += 1

                # Emission count: register word or signature
                if tok in self.vocab:
                    emit_counts[tok][state_id] += 1
                else:
                    sig = get_word_signature(tok)
                    emit_counts[sig][state_id] += 1

        # 4. Compute smoothed log initial probabilities
        total_inits = sum(init_counts)
        self.log_init = [-float("inf")] * K
        for i in range(K):
            tag_name = self.id2state[i]
            # Enforce BIO constraint: initial token cannot be I- tag
            if not is_valid_bio_transition(None, tag_name):
                self.log_init[i] = -1e9
            else:
                prob = (init_counts[i] + self.alpha_init) / (total_inits + K * self.alpha_init)
                self.log_init[i] = math.log(prob)

        # 5. Compute smoothed log transition probabilities
        self.log_trans = [[-float("inf")] * K for _ in range(K)]
        for i in range(K):
            prev_tag = self.id2state[i]
            denom = sum(trans_counts[i]) + K * self.alpha_trans
            for j in range(K):
                curr_tag = self.id2state[j]
                # Enforce BIO transition constraint
                if not is_valid_bio_transition(prev_tag, curr_tag):
                    self.log_trans[i][j] = -1e9
                else:
                    prob = (trans_counts[i][j] + self.alpha_trans) / denom
                    self.log_trans[i][j] = math.log(prob)

        # 6. Compute smoothed log emission probabilities
        all_vocab_keys = list(self.vocab.keys()) + self.signatures
        V = len(all_vocab_keys)

        self.log_emit = {}
        for token in self.vocab:
            self.log_emit[token] = [0.0] * K
            for i in range(K):
                prob = (emit_counts[token][i] + self.alpha_emit) / (state_totals[i] + V * self.alpha_emit)
                self.log_emit[token][i] = math.log(prob)

        self.log_emit_signatures = {}
        for sig in self.signatures:
            self.log_emit_signatures[sig] = [0.0] * K
            for i in range(K):
                prob = (emit_counts[sig][i] + self.alpha_emit) / (state_totals[i] + V * self.alpha_emit)
                self.log_emit_signatures[sig][i] = math.log(prob)

        logger.info(
            f"Fitted HMM: {K} states, {len(self.vocab)} vocab items, "
            f"{len(self.signatures)} morphological signatures."
        )
        return self

    def _get_token_log_emissions(self, token: str) -> List[float]:
        """Retrieve log emissions for a token, falling back to signature if OOV."""
        if token in self.log_emit:
            return self.log_emit[token]
        sig = get_word_signature(token)
        if sig in self.log_emit_signatures:
            return self.log_emit_signatures[sig]
        return self.log_emit_signatures["<UNK>"]

    def decode_sentence(self, tokens: List[str]) -> List[str]:
        """
        Viterbi algorithm in log space to decode the most likely tag sequence
        for a sequence of tokens, respecting BIO constraints.
        """
        if not tokens:
            return []

        T = len(tokens)
        K = len(self.states)

        # Trellis: V[t][s] is max log prob of sequence ending in state s at time t
        V = [[-float("inf")] * K for _ in range(T)]
        bp = [[0] * K for _ in range(T)]

        # t = 0 (Initial step)
        first_emissions = self._get_token_log_emissions(tokens[0])
        for s in range(K):
            V[0][s] = self.log_init[s] + first_emissions[s]

        # t = 1 to T - 1 (Forward recursion)
        for t in range(1, T):
            emissions = self._get_token_log_emissions(tokens[t])
            for s in range(K):
                best_val = -float("inf")
                best_prev = 0
                for prev_s in range(K):
                    val = V[t - 1][prev_s] + self.log_trans[prev_s][s]
                    if val > best_val:
                        best_val = val
                        best_prev = prev_s
                V[t][s] = best_val + emissions[s]
                bp[t][s] = best_prev

        # Termination
        best_last_state = 0
        best_final_val = -float("inf")
        for s in range(K):
            if V[T - 1][s] > best_final_val:
                best_final_val = V[T - 1][s]
                best_last_state = s

        # Backtracking
        best_path = [best_last_state]
        curr = best_last_state
        for t in range(T - 1, 0, -1):
            curr = bp[t][curr]
            best_path.append(curr)

        best_path.reverse()
        return [self.id2state[s] for s in best_path]

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predict BIO tag sequences for a batch of sentences."""
        return [self.decode_sentence(sent) for sent in sentences]

