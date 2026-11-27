"""Extraction metrics.

Values are compared as *sets* of normalised strings per field (case- and whitespace-
insensitive). Micro P/R/F1 pools all fields; a bootstrap CI on micro-F1 tells whether a
difference between two systems on a few hundred sentences is meaningful.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llmx.inference.parsing import ParsedOutput

Entities = Mapping[str, Sequence[str]]


def normalize_value(value: str) -> str:
    return " ".join(value.lower().split())


def as_set(values: Sequence[str]) -> set[str]:
    return {normalize_value(v) for v in values if v.strip()}


def occurs_in(value: str, text: str) -> bool:
    """Whole-token containment of normalised strings ('un' is not found in 'under')."""
    return f" {value} " in f" {text} "


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.tp + self.fn,
        }


def field_counts(gold: Sequence[str], pred: Sequence[str]) -> Counts:
    g, p = as_set(gold), as_set(pred)
    return Counts(tp=len(g & p), fp=len(p - g), fn=len(g - p))


def _micro_counts(gold: Entities, pred: Entities, fields: Sequence[str]) -> Counts:
    total = Counts()
    for f in fields:
        total.add(field_counts(gold.get(f, []), pred.get(f, [])))
    return total


def evaluate(
    golds: Sequence[Entities],
    preds: Sequence[ParsedOutput],
    texts: Sequence[str],
    fields: Sequence[str],
) -> dict[str, Any]:
    if not (len(golds) == len(preds) == len(texts)):
        raise ValueError("golds, preds and texts must have the same length")
    n = len(golds)
    if n == 0:
        raise ValueError("cannot evaluate an empty set")

    per_field = {f: Counts() for f in fields}
    exact = 0
    predicted_values = 0
    hallucinated = 0

    for gold, pred, text in zip(golds, preds, texts, strict=True):
        norm_text = normalize_value(text)
        all_match = True
        for f in fields:
            c = field_counts(gold.get(f, []), pred.entities.get(f, []))
            per_field[f].add(c)
            all_match &= c.fp == 0 and c.fn == 0
            for value in as_set(pred.entities.get(f, [])):
                predicted_values += 1
                hallucinated += not occurs_in(value, norm_text)
        exact += all_match

    micro = Counts()
    for c in per_field.values():
        micro.add(c)

    return {
        "n": n,
        "json_valid_rate": sum(p.is_valid_json for p in preds) / n,
        "strict_json_rate": sum(p.is_strict_json for p in preds) / n,
        "schema_valid_rate": sum(p.is_valid_schema for p in preds) / n,
        "exact_match": exact / n,
        "hallucination_rate": hallucinated / predicted_values if predicted_values else 0.0,
        "micro": micro.as_dict(),
        "per_field": {f: c.as_dict() for f, c in per_field.items()},
    }


def bootstrap_f1_ci(
    golds: Sequence[Entities],
    preds: Sequence[ParsedOutput],
    fields: Sequence[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of micro-F1, resampling sentences with replacement."""
    per_example = [_micro_counts(g, p.entities, fields) for g, p in zip(golds, preds, strict=True)]
    rng = random.Random(seed)
    n = len(per_example)
    scores = []
    for _ in range(n_boot):
        total = Counts()
        for _ in range(n):
            total.add(per_example[rng.randrange(n)])
        scores.append(total.f1)
    scores.sort()
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi
