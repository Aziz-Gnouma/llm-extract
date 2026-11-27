from llmx.evaluation.errors import categorize_errors, summarize_errors
from llmx.inference.parsing import ParsedOutput

FIELDS = ["persons", "organizations", "locations"]
TEXT = "Ann Lee met the UN team in New York"
GOLD = {"persons": ["Ann Lee"], "organizations": ["UN"], "locations": ["New York"]}


def pred(**entities):
    return ParsedOutput(True, True, True, {f: entities.get(f, []) for f in FIELDS})


def categories(p):
    return sorted(e.category for e in categorize_errors(GOLD, p, TEXT, FIELDS))


def test_perfect_prediction_has_no_error():
    assert categories(pred(persons=["Ann Lee"], organizations=["UN"], locations=["New York"])) == []


def test_each_category():
    p = pred(
        persons=["Ann"],  # boundary (gold "Ann Lee")
        organizations=["Google"],  # hallucination (not in text)  + UN missed
        locations=["UN", "team"],  # wrong_field (UN is an ORG) + spurious ("team"); New York missed
    )
    assert categories(p) == [
        "boundary",
        "hallucination",
        "missed",
        "spurious",
        "wrong_field",
    ]
    # UN was predicted in another field -> counted once (wrong_field), not also as missed
    missed = [e.value for e in categorize_errors(GOLD, p, TEXT, FIELDS) if e.category == "missed"]
    assert missed == ["new york"]


def test_invalid_json_is_a_single_error():
    p = ParsedOutput(False, False, False, {f: [] for f in FIELDS})
    assert categories(p) == ["invalid_json"]


def test_summary_lists_every_category():
    summary = summarize_errors([categorize_errors(GOLD, pred(), TEXT, FIELDS)])
    assert summary["missed"] == 3
    assert set(summary) >= {"invalid_json", "hallucination", "wrong_field", "boundary", "spurious"}
