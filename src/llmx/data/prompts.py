"""Chat-message construction. The SAME prompt is used by the baseline and the fine-tuned
model, so the only difference between them is the LoRA adapter (fair comparison)."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence

from llmx.data.conll import Record

Message = dict[str, str]

FIELD_DEFINITIONS: Mapping[str, str] = {
    "persons": "names of people",
    "organizations": "companies, institutions, political parties, sports teams, agencies",
    "locations": "countries, cities, regions, geographic places",
}


def system_prompt(fields: Sequence[str]) -> str:
    keys = ", ".join(f'"{f}"' for f in fields)
    definitions = "\n".join(f"- {f}: {FIELD_DEFINITIONS.get(f, f)}" for f in fields)
    return (
        "You are an information extraction system. Extract the named entities from the "
        f"user's text and answer with ONE JSON object with exactly these keys: {keys}.\n"
        f"{definitions}\n"
        "Each value is a list of distinct strings copied verbatim from the text; "
        "use an empty list when there is none. Output the JSON object only."
    )


def target_json(entities: Mapping[str, Sequence[str]], fields: Sequence[str]) -> str:
    return json.dumps({f: list(entities.get(f, [])) for f in fields}, ensure_ascii=False)


def build_messages(
    text: str,
    fields: Sequence[str],
    shots: Sequence[Record] = (),
) -> list[Message]:
    """Prompt for one input text; ``shots`` are optional in-context examples (few-shot)."""
    messages: list[Message] = [{"role": "system", "content": system_prompt(fields)}]
    for shot in shots:
        messages.append({"role": "user", "content": shot["text"]})
        messages.append({"role": "assistant", "content": target_json(shot["entities"], fields)})
    messages.append({"role": "user", "content": text})
    return messages


def build_training_messages(record: Record, fields: Sequence[str]) -> list[Message]:
    """Prompt + gold answer (the assistant turn is the only part that is trained on)."""
    return [
        *build_messages(record["text"], fields),
        {"role": "assistant", "content": target_json(record["entities"], fields)},
    ]


def select_shots(train: Sequence[Record], k: int, seed: int) -> list[Record]:
    """Seeded choice of k train examples that contain at least one entity."""
    if k <= 0:
        return []
    candidates = [r for r in train if any(r["entities"].values())]
    return random.Random(seed).sample(candidates, min(k, len(candidates)))
