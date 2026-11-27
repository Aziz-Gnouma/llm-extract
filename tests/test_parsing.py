from llmx.inference.parsing import parse_prediction

FIELDS = ["persons", "organizations", "locations"]


def test_strict_valid_output():
    raw = '{"persons": ["Ann"], "organizations": [], "locations": ["Rome"]}'
    p = parse_prediction(raw, FIELDS)
    assert (p.is_valid_json, p.is_strict_json, p.is_valid_schema) == (True, True, True)
    assert p.entities == {"persons": ["Ann"], "organizations": [], "locations": ["Rome"]}


def test_code_fence_is_valid_but_not_strict():
    raw = '```json\n{"persons": [], "organizations": ["UN"], "locations": []}\n```'
    p = parse_prediction(raw, FIELDS)
    assert p.is_valid_json and not p.is_strict_json and p.is_valid_schema
    assert p.entities["organizations"] == ["UN"]


def test_missing_key_breaks_schema_but_keeps_values():
    p = parse_prediction('{"persons": ["Ann"]}', FIELDS)
    assert p.is_valid_json and not p.is_valid_schema
    assert p.entities == {"persons": ["Ann"], "organizations": [], "locations": []}


def test_wrong_value_type_breaks_schema():
    p = parse_prediction('{"persons": "Ann", "organizations": [1, "UN"], "locations": []}', FIELDS)
    assert not p.is_valid_schema
    assert p.entities == {"persons": [], "organizations": ["UN"], "locations": []}


def test_garbage_and_truncated_outputs():
    for raw in ["no json here", '{"persons": ["Ann"', "[1, 2]"]:
        p = parse_prediction(raw, FIELDS)
        assert not p.is_valid_json
        assert p.entities == {f: [] for f in FIELDS}
