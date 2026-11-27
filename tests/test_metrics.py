import pytest

from llmx.evaluation.metrics import bootstrap_f1_ci, evaluate, field_counts
from llmx.inference.parsing import ParsedOutput

FIELDS = ["persons", "locations"]


def parsed(persons=(), locations=(), valid=True):
    return ParsedOutput(
        valid, valid, valid, {"persons": list(persons), "locations": list(locations)}
    )


def test_field_counts_normalises_case_and_spaces():
    c = field_counts(["New York", "Ann"], ["new  york", "Bob"])
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)
    assert c.precision == c.recall == c.f1 == 0.5


def test_hallucination_uses_whole_tokens():
    gold = {"persons": [], "locations": []}
    m = evaluate([gold], [parsed(locations=["UN"])], ["Under the sea"], FIELDS)
    assert m["hallucination_rate"] == 1.0


def test_evaluate_perfect_prediction():
    gold = {"persons": ["Ann"], "locations": ["Rome"]}
    m = evaluate([gold], [parsed(["Ann"], ["Rome"])], ["Ann is in Rome"], FIELDS)
    assert m["micro"]["f1"] == 1.0
    assert m["exact_match"] == 1.0
    assert m["hallucination_rate"] == 0.0


def test_evaluate_counts_hallucinations_and_invalid_json():
    golds = [{"persons": ["Ann"], "locations": []}, {"persons": [], "locations": ["Rome"]}]
    preds = [parsed(["Ann", "Zoe"]), parsed(valid=False)]
    texts = ["Ann sings", "Rome is big"]
    m = evaluate(golds, preds, texts, FIELDS)
    assert m["json_valid_rate"] == 0.5
    assert m["hallucination_rate"] == 0.5  # "zoe" not in text, "ann" is
    assert m["micro"]["precision"] == 0.5  # tp=1, fp=1
    assert m["micro"]["recall"] == 0.5  # tp=1, fn=1 (Rome missed)
    assert m["exact_match"] == 0.0


def test_empty_gold_and_empty_pred_is_exact_match():
    m = evaluate([{"persons": [], "locations": []}], [parsed()], ["nothing here"], FIELDS)
    assert m["exact_match"] == 1.0 and m["micro"]["f1"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate([{}], [], ["x"], FIELDS)


def test_bootstrap_ci_brackets_point_estimate():
    golds = [{"persons": ["Ann"], "locations": []}] * 20 + [
        {"persons": ["Bob"], "locations": []}
    ] * 20
    preds = [parsed(["Ann"])] * 20 + [parsed(["Max"])] * 20
    lo, hi = bootstrap_f1_ci(golds, preds, FIELDS, n_boot=300, seed=0)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert bootstrap_f1_ci(golds, preds, FIELDS, n_boot=300, seed=0) == (lo, hi)
