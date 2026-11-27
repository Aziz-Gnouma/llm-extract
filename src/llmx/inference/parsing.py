"""Turn raw model text into a schema-checked prediction.

Three increasingly strict levels are reported separately, never silently "repaired":
* ``is_valid_json``  - a JSON object can be decoded starting at the first '{'
                       (tolerates prose / code fences around it);
* ``is_strict_json`` - the whole output, stripped, is exactly one JSON object;
* ``is_valid_schema`` - exactly the expected keys, each mapping to a list of strings.

Whatever fails validation contributes empty predictions, so it is scored as missed entities.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedOutput:
    is_valid_json: bool
    is_strict_json: bool
    is_valid_schema: bool
    entities: dict[str, list[str]] = field(default_factory=dict)


def _empty(fields: Sequence[str]) -> dict[str, list[str]]:
    return {f: [] for f in fields}


def _is_strict(raw: str) -> bool:
    try:
        return isinstance(json.loads(raw.strip()), dict)
    except json.JSONDecodeError:
        return False


def parse_prediction(raw: str, fields: Sequence[str]) -> ParsedOutput:
    start = raw.find("{")
    if start == -1:
        return ParsedOutput(False, False, False, _empty(fields))
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return ParsedOutput(False, False, False, _empty(fields))
    if not isinstance(obj, dict):
        return ParsedOutput(False, False, False, _empty(fields))

    schema_ok = set(obj) == set(fields)
    entities: dict[str, list[str]] = {}
    for f in fields:
        value = obj.get(f, [])
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            entities[f] = list(value)
        else:
            schema_ok = False
            entities[f] = (
                [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
            )

    return ParsedOutput(True, _is_strict(raw), schema_ok, entities)
