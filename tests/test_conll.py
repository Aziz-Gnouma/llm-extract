import pytest

from llmx.data.conll import bio_to_spans, sentence_to_record

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
FIELD_MAP = {"PER": "persons", "ORG": "organizations", "LOC": "locations"}


def test_bio_to_spans_iob2():
    tokens = ["Peter", "Blackburn", "visited", "New", "York"]
    tags = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    assert bio_to_spans(tokens, tags) == [("PER", "Peter Blackburn"), ("LOC", "New York")]


def test_bio_to_spans_iob1_and_adjacent_entities():
    # IOB1: span may start with I-; B- separates two adjacent entities of the same type
    tokens = ["Paris", "London", "Berlin"]
    tags = ["I-LOC", "B-LOC", "I-ORG"]
    assert bio_to_spans(tokens, tags) == [("LOC", "Paris"), ("LOC", "London"), ("ORG", "Berlin")]


def test_bio_to_spans_length_mismatch():
    with pytest.raises(ValueError):
        bio_to_spans(["a"], ["O", "O"])


def test_sentence_to_record_maps_fields_drops_misc_and_dedups():
    tokens = ["EU", "rejects", "German", "call", ",", "EU", "says"]
    tag_ids = [3, 0, 7, 0, 0, 3, 0]  # ORG, MISC, ORG
    rec = sentence_to_record("x-1", tokens, tag_ids, LABELS, FIELD_MAP)
    assert rec == {
        "id": "x-1",
        "text": "EU rejects German call , EU says",
        "entities": {"persons": [], "organizations": ["EU"], "locations": []},
    }


def test_docstart_is_skipped():
    assert sentence_to_record("x", ["-DOCSTART-"], [0], LABELS, FIELD_MAP) is None


def test_gold_values_are_substrings_of_text():
    tokens = ["Jean", "Dupont", "works", "at", "Air", "France", "in", "Lyon"]
    tag_ids = [1, 2, 0, 0, 3, 4, 0, 5]
    rec = sentence_to_record("x", tokens, tag_ids, LABELS, FIELD_MAP)
    for values in rec["entities"].values():
        for v in values:
            assert v in rec["text"]
