"""Convert token-level BIO annotations (CoNLL-2003) into JSON extraction records.

A record is a plain dict, so it serialises to JSONL as-is:

    {"id": "train-12", "text": "EU rejects German call ...",
     "entities": {"persons": [], "organizations": ["EU"], "locations": []}}

The text is the tokens joined by single spaces, and every entity is the space-joined
span of its tokens, so each gold value is guaranteed to be a substring of the text.
That invariant is what makes the hallucination metric meaningful.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

Record = dict

DOCSTART = "-DOCSTART-"


def bio_to_spans(tokens: Sequence[str], tags: Sequence[str]) -> list[tuple[str, str]]:
    """Return (entity_type, surface_text) spans.

    Accepts IOB2 and IOB1: an ``I-X`` that does not continue an ``X`` span opens a new one.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"{len(tokens)} tokens but {len(tags)} tags")

    spans: list[tuple[str, str]] = []
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_type is not None:
            spans.append((current_type, " ".join(current_tokens)))
        current_type, current_tokens = None, []

    for token, tag in zip(tokens, tags, strict=True):
        if tag == "O":
            flush()
            continue
        prefix, _, etype = tag.partition("-")
        if prefix == "B" or etype != current_type:
            flush()
            current_type = etype
        current_tokens.append(token)
    flush()
    return spans


def sentence_to_record(
    record_id: str,
    tokens: Sequence[str],
    tag_ids: Sequence[int],
    label_names: Sequence[str],
    field_map: Mapping[str, str],
) -> Record | None:
    """Build one extraction record, or None for document separators."""
    if not tokens or tokens[0] == DOCSTART:
        return None

    fields = list(dict.fromkeys(field_map.values()))
    entities: dict[str, list[str]] = {f: [] for f in fields}
    tags = [label_names[i] for i in tag_ids]

    for etype, surface in bio_to_spans(tokens, tags):
        key = field_map.get(etype)
        if key is not None and surface not in entities[key]:  # distinct values, stable order
            entities[key].append(surface)

    return {"id": record_id, "text": " ".join(tokens), "entities": entities}
