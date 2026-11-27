"""Error taxonomy for the critical-analysis section.

Every wrong or missing value gets exactly one category:

* ``invalid_json``  - the output could not be parsed at all
* ``hallucination`` - predicted value does not occur in the input text
* ``wrong_field``   - value is a gold entity, but of another field (e.g. ORG predicted as LOC)
* ``boundary``      - value overlaps a gold value of the same field (too long / too short)
* ``spurious``      - value is in the text but is not a gold entity
* ``missed``        - gold value not predicted and not explained by the categories above
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from llmx.evaluation.metrics import Entities, as_set, normalize_value, occurs_in
from llmx.inference.parsing import ParsedOutput

CATEGORIES = ("invalid_json", "hallucination", "wrong_field", "boundary", "spurious", "missed")


@dataclass(frozen=True)
class ErrorItem:
    category: str
    field: str | None
    value: str | None


def _overlaps(a: str, b: str) -> bool:
    return occurs_in(a, b) or occurs_in(b, a)


def categorize_errors(
    gold: Entities, pred: ParsedOutput, text: str, fields: Sequence[str]
) -> list[ErrorItem]:
    if not pred.is_valid_json:
        return [ErrorItem("invalid_json", None, None)]

    norm_text = normalize_value(text)
    gold_sets = {f: as_set(gold.get(f, [])) for f in fields}
    pred_sets = {f: as_set(pred.entities.get(f, [])) for f in fields}
    errors: list[ErrorItem] = []

    for f in fields:
        others_gold = set().union(*(gold_sets[o] for o in fields if o != f))
        others_pred = set().union(*(pred_sets[o] for o in fields if o != f))
        missed_gold = gold_sets[f] - pred_sets[f]
        extra_pred = pred_sets[f] - gold_sets[f]
        explained_gold: set[str] = set()

        for value in sorted(extra_pred):
            if not occurs_in(value, norm_text):
                errors.append(ErrorItem("hallucination", f, value))
            elif value in others_gold:
                errors.append(ErrorItem("wrong_field", f, value))
            else:
                partners = {g for g in missed_gold if _overlaps(value, g)}
                if partners:
                    explained_gold |= partners
                    errors.append(ErrorItem("boundary", f, value))
                else:
                    errors.append(ErrorItem("spurious", f, value))

        for value in sorted(missed_gold - explained_gold):
            if value not in others_pred:  # otherwise already counted as wrong_field
                errors.append(ErrorItem("missed", f, value))

    return errors


def summarize_errors(per_example: Sequence[Sequence[ErrorItem]]) -> dict[str, int]:
    counter = Counter(e.category for errs in per_example for e in errs)
    return {c: counter.get(c, 0) for c in CATEGORIES}
