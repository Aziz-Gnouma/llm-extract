"""Download CoNLL-2003 from the Hugging Face Hub and convert it to extraction records."""

from __future__ import annotations

from llmx.config import DataConfig
from llmx.data.conll import Record, sentence_to_record

# Used when the dataset stores tags as plain ints (lhoestq/conll2003 does) instead of a
# ClassLabel. Order verified against the first rows: "EU"=3 (B-ORG), "German"=7 (B-MISC).
CONLL2003_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

_HF_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def _label_names(hf_split) -> list[str]:
    try:
        return list(hf_split.features["ner_tags"].feature.names)
    except (KeyError, AttributeError):
        return CONLL2003_LABELS


def load_conll_records(cfg: DataConfig) -> dict[str, list[Record]]:
    from datasets import load_dataset  # heavy import kept local

    ds = load_dataset(cfg.hf_dataset, revision=cfg.hf_revision)
    out: dict[str, list[Record]] = {}
    for ours, theirs in _HF_SPLITS.items():
        hf_split = ds[theirs]
        labels = _label_names(hf_split)
        records = []
        for i, row in enumerate(hf_split):
            if len(row["tokens"]) < cfg.min_tokens:
                continue
            rec = sentence_to_record(
                f"{ours}-{row.get('id', i)}", row["tokens"], row["ner_tags"], labels, cfg.field_map
            )
            if rec is not None:
                records.append(rec)
        out[ours] = records
    return out
