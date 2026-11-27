"""Deterministic, leak-free train / val / test construction.

Policy (documented in the cahier des charges):
* test is sampled from CoNLL's official *test* split, val from *validation*, train from *train*;
* exact duplicates (after normalisation) are removed inside each split;
* any train/val sentence whose normalised text also appears in a later-priority split is
  dropped (priority: test > val > train), so the held-out sets are never seen in training.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from llmx.data.conll import Record


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate(records: Iterable[Record]) -> list[Record]:
    seen: set[str] = set()
    out: list[Record] = []
    for rec in records:
        key = normalize_text(rec["text"])
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out


def remove_overlap(records: Iterable[Record], reference: Iterable[Record]) -> list[Record]:
    banned = {normalize_text(r["text"]) for r in reference}
    return [r for r in records if normalize_text(r["text"]) not in banned]


def sample(records: Sequence[Record], n: int, seed: int) -> list[Record]:
    """Seeded sample without replacement; returns everything if fewer than ``n`` exist."""
    if n >= len(records):
        return list(records)
    return random.Random(seed).sample(list(records), n)


def build_splits(
    raw_train: Sequence[Record],
    raw_val: Sequence[Record],
    raw_test: Sequence[Record],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> dict[str, list[Record]]:
    test = sample(deduplicate(raw_test), n_test, seed)
    val = sample(remove_overlap(deduplicate(raw_val), test), n_val, seed)
    train = sample(remove_overlap(deduplicate(raw_train), [*test, *val]), n_train, seed)
    splits = {"train": train, "val": val, "test": test}
    assert_no_overlap(splits)
    return splits


def assert_no_overlap(splits: dict[str, Sequence[Record]]) -> None:
    names = list(splits)
    texts = {name: {normalize_text(r["text"]) for r in splits[name]} for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = texts[a] & texts[b]
            if shared:
                example = next(iter(shared))
                raise AssertionError(
                    f"{len(shared)} sentence(s) shared by {a}/{b}, e.g. {example!r}"
                )
