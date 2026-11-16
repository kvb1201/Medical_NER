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


def _sentences(text):
    return [[t[0] for t in sent] for sent in split_sentences(tokenize(text))]


def test_titles_and_decimals_do_not_end_a_sentence():
    assert _sentences("Dr. Smith gave 2.5 mg of warfarin.") == [
        ["Dr.", "Smith", "gave", "2.5", "mg", "of", "warfarin", "."]
    ]


def test_hyphenated_words_split_like_bc5cdr():
    assert [t[0] for t in tokenize("a 64-year-old man")] == ["a", "64", "-", "year", "-", "old", "man"]


def test_non_ascii_letters_stay_in_the_token():
    text = "Sjögren syndrome"
    tokens = tokenize(text)
    assert [t[0] for t in tokens] == ["Sjögren", "syndrome"]
    assert text[tokens[0][1]:tokens[0][2]] == "Sjögren"


def test_offsets_are_code_points_after_an_emoji():
    text = "\U0001F600 Patient has diabetes."
    model = _ScriptedModel({"diabetes": "B-DISEASE"})
    entity = extract_text(text, model, "CRF")["entities"][0]
    assert text[entity["start"]:entity["end"]] == "diabetes"


def test_unseen_patient_titles_are_not_sent_to_the_model():
    seen = []

    class _Recorder(_ScriptedModel):
        def predict(self, sentences):
            seen.extend(sentences)
            return super().predict(sentences)

    text = "Mr. John Doe has asthma."
    model = _Recorder({"John": "B-PATIENT_NAME", "Doe": "I-PATIENT_NAME"})
    entities = extract_text(text, model, "CRF")["entities"]
    assert seen == [["John", "Doe", "has", "asthma", "."]]
    assert [(e["text"], e["start"]) for e in entities] == [("John Doe", 4)]


def test_doctor_title_is_kept_because_it_appears_in_training():
    seen = []

    class _Recorder(_ScriptedModel):
        def predict(self, sentences):
            seen.extend(sentences)
            return super().predict(sentences)

    extract_text("Dr. Lee saw her.", _Recorder({}), "CRF")
    assert seen == [["Dr.", "Lee", "saw", "her", "."]]
