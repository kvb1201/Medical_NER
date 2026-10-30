"""Span alignment for free-text inference. Does not load a trained model."""

from src.infer import extract_text, split_sentences, tokenize


class _ScriptedModel:
    def __init__(self, tag_map):
        self.tag_map = tag_map

    def predict(self, sentences):
        sequences = []
        for sent in sentences:
            sequences.append([self.tag_map.get(tok, "O") for tok in sent])
        return sequences


def test_tokenize_keeps_character_offsets():
    text = "Started heparin."
    tokens = tokenize(text)
    assert [t[0] for t in tokens] == ["Started", "heparin", "."]
    assert text[tokens[1][1]:tokens[1][2]] == "heparin"


def test_hyphenated_token_stays_whole():
    tokens = tokenize("Nelson-Cole presented.")
    assert tokens[0][0] == "Nelson-Cole"


def test_split_sentences_on_period():
    tokens = tokenize("One. Two")
    sentences = split_sentences(tokens)
    assert [[t[0] for t in s] for s in sentences] == [["One", "."], ["Two"]]


def test_extract_text_maps_a_multi_token_span():
    text = "Patient Jane Smith has pulmonary embolism."
    model = _ScriptedModel({
        "Jane": "B-PATIENT_NAME",
        "Smith": "I-PATIENT_NAME",
        "pulmonary": "B-DISEASE",
        "embolism": "I-DISEASE",
    })
    result = extract_text(text, model, "CRF")
    assert result["entities"] == [
        {"text": "Jane Smith", "type": "PATIENT_NAME", "start": 8, "end": 18},
        {"text": "pulmonary embolism", "type": "DISEASE", "start": 23, "end": 41},
    ]


def test_blank_note_skips_the_model():
    class _Boom:
        def predict(self, sentences):
            raise AssertionError("model should not run")

    assert extract_text("   ", _Boom(), "HMM")["entities"] == []
