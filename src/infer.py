"""Turn raw clinical text into entity spans with a trained sequence tagger."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from src.models.evaluation import extract_entities

# Words, including internal hyphens and apostrophes, or a single punctuation mark.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\s\w]", re.UNICODE)
_SENTENCE_END = {".", "!", "?"}

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
    tag_sequences = model.predict([[tok for tok, _, _ in sent] for sent in sentences])
    if len(tag_sequences) != len(sentences):
        raise RuntimeError("Model returned a different number of sentences than it was given.")

    entities: List[Dict[str, Any]] = []
    for sent, tags in zip(sentences, tag_sequences):
        for span in spans_from_tags(sent, tags):
            span["text"] = raw[span["start"]:span["end"]]
            entities.append(span)

    return {"text": raw, "entities": entities, "model": model_name}
