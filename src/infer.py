"""Turn raw clinical text into entity spans with a trained sequence tagger."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from src.models.evaluation import extract_entities

# Tokenization mirrors the training data so the models see familiar token shapes:
#   * BC5CDR splits most hyphenated words ("67 - year - old", "anti - inflammatory").
#   * The synthetic notes keep courtesy titles ("Dr.") and hyphenated surnames
#     ("Nelson-Cole") as single tokens.
# Alternatives are tried in order at each position.
_LETTERS = r"[^\W\d_]"  # any Unicode letter
_TOKEN = re.compile(
    r"(?<![^\W_])(?:Dr|Mr|Mrs|Ms|Prof)\."               # courtesy titles, kept with their period
    r"|(?<![^\W_])(?:e\.g|i\.e)\."                      # common abbreviations
    r"|\d+(?:[.,]\d+)+"                                 # decimals and grouped numbers: 2.5, 1,000
    r"|[A-Z]" + _LETTERS + r"*(?:-[A-Z]" + _LETTERS + r"*)+"  # hyphenated proper names: Nelson-Cole
    r"|[^\W_]+(?:'[^\W_]+)*"                            # words and numbers, internal apostrophes kept
    r"|[^\s\w]|_"                                       # any other single symbol, including "-"
)
_SENTENCE_END = {".", "!", "?"}
# "Dr." appears in the synthetic training notes (before clinicians' names) and is kept.
# These titles never appear in training; feeding them to the models cut patient-name
# recall on title-prefixed synthetic test sentences from ~0.92-0.95 to ~0.39-0.69.
_UNSEEN_TITLES = {"Mr.", "Mrs.", "Ms."}

MAX_CHARS = 4000

Token = Tuple[str, int, int]


def tokenize(text: str) -> List[Token]:
    """Split text into tokens and the character offsets they occupy."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def split_sentences(tokens: Sequence[Token]) -> List[List[Token]]:
    """Group tokens into sentences on . ! ? so decoding matches training sequences."""
    sentences: List[List[Token]] = []
    current: List[Token] = []
    for token in tokens:
        current.append(token)
        if token[0] in _SENTENCE_END:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def spans_from_tags(tokens: Sequence[Token], tags: Sequence[str]) -> List[Dict[str, Any]]:
    """Map a BIO tag sequence back onto character spans in the original text."""
    if len(tokens) != len(tags):
        raise ValueError(f"Token/tag length mismatch: {len(tokens)} tokens vs {len(tags)} tags")

    spans = []
    for entity_type, start, end in extract_entities(list(tags)):
        char_start = tokens[start][1]
        char_end = tokens[end - 1][2]
        spans.append({
            "text": None,  # filled by the caller from the source string
            "type": entity_type,
            "start": char_start,
            "end": char_end,
        })
    return spans


def extract_text(text: str, model: Any, model_name: str) -> Dict[str, Any]:
    """Tokenize ``text``, tag it with ``model.predict``, and return entity spans."""
    raw = text if text is not None else ""
    if len(raw) > MAX_CHARS:
        raise ValueError(f"Note is too long ({len(raw)} characters). Limit is {MAX_CHARS}.")

    stripped = raw.strip()
    if not stripped:
        return {"text": raw, "entities": [], "model": model_name}

    tokens = tokenize(raw)
    if not tokens:
        return {"text": raw, "entities": [], "model": model_name}

    sentences = split_sentences(tokens)
    # Patient courtesy titles never occur in the training data, so they are left out of
    # the model input (tagged "O"); character offsets still refer to the original text.
    kept = [[i for i, tok in enumerate(sent) if tok[0] not in _UNSEEN_TITLES] for sent in sentences]
    to_tag = [[sent[i][0] for i in idx] for sent, idx in zip(sentences, kept) if idx]
    predicted = iter(model.predict(to_tag) if to_tag else [])

    tag_sequences: List[List[str]] = []
    for sent, idx in zip(sentences, kept):
        tags = ["O"] * len(sent)
        if idx:
            sub_tags = next(predicted, None)
            if sub_tags is None or len(sub_tags) != len(idx):
                raise RuntimeError("Model output does not line up with the sentences it was given.")
            for i, tag in zip(idx, sub_tags):
                tags[i] = tag
        tag_sequences.append(tags)

    entities: List[Dict[str, Any]] = []
    for sent, tags in zip(sentences, tag_sequences):
        for span in spans_from_tags(sent, tags):
            span["text"] = raw[span["start"]:span["end"]]
            entities.append(span)

    return {"text": raw, "entities": entities, "model": model_name}
