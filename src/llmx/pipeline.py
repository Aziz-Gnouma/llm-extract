"""Use cases. The notebook and the CLI both call these functions and nothing lower-level,
so every experiment is reproducible from the command line as well.

Artefacts:
    data/processed/{train,val,test}.jsonl + manifest.json   (hashes, counts, config)
    outputs/<run>/adapter/                                   (LoRA weights)
    reports/<run>/<system>_<split>/{predictions.jsonl, metrics.json}
    reports/<run>/train_report.json, config.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from llmx.config import ProjectConfig, set_seed
from llmx.data.conll import Record
from llmx.data.io import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from llmx.data.prompts import select_shots
from llmx.data.splits import build_splits
from llmx.evaluation.errors import categorize_errors, summarize_errors
from llmx.evaluation.guard import HoldoutGuard
from llmx.evaluation.metrics import bootstrap_f1_ci, evaluate
from llmx.inference.parsing import parse_prediction

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- data
class SplitMismatchError(RuntimeError):
    pass


def _files_match(cfg: ProjectConfig, manifest: dict[str, Any]) -> bool:
    paths = {name: cfg.processed_dir / f"{name}.jsonl" for name in SPLITS}
    return all(
        p.exists() and file_sha256(p) == manifest["splits"][name]["sha256"]
        for name, p in paths.items()
    )


def prepare_data(cfg: ProjectConfig, force: bool = False) -> dict[str, Any]:
    """Download, convert, split and freeze the dataset.

    The split files are not versioned (dataset licence), only ``manifest.json`` is. On a fresh
    clone the splits are regenerated and must reproduce the committed SHA-256 exactly, otherwise
    ``SplitMismatchError`` is raised. ``force`` rebuilds and overwrites the manifest.
    """
    manifest_path = cfg.processed_dir / "manifest.json"
    committed = read_json(manifest_path) if manifest_path.exists() and not force else None
    if committed is not None and _files_match(cfg, committed):
        return committed

    from llmx.data.loading import load_conll_records

    set_seed(cfg.seed)
    raw = load_conll_records(cfg.data)
    d = cfg.data
    splits = build_splits(
        raw["train"], raw["val"], raw["test"], d.n_train, d.n_val, d.n_test, cfg.seed
    )
    manifest: dict[str, Any] = {"source": d.hf_dataset, "revision": d.hf_revision, "splits": {}}
    for name, records in splits.items():
        path = cfg.processed_dir / f"{name}.jsonl"
        write_jsonl(path, records)
        manifest["splits"][name] = {
            "n": len(records),
            "sha256": file_sha256(path),
            "entity_counts": entity_counts(records, d.fields),
        }
    if committed is not None:
        if manifest != committed:
            raise SplitMismatchError(
                "Regenerated splits differ from the committed manifest.json "
                "(dataset revision, config or library behaviour changed). "
                "Use prepare_data(cfg, force=True) only if this change is intended."
            )
        return committed
    write_json(manifest_path, manifest)
    return manifest


def load_splits(cfg: ProjectConfig) -> dict[str, list[Record]]:
    return {name: read_jsonl(cfg.processed_dir / f"{name}.jsonl") for name in SPLITS}


def entity_counts(records: list[Record], fields: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in records:
        for f in fields:
            counts[f] += len(r["entities"].get(f, []))
        counts["sentences_without_entity"] += not any(r["entities"].values())
    return dict(counts)


# --------------------------------------------------------------------------- training
def run_training(cfg: ProjectConfig) -> dict[str, Any]:
    from llmx.modeling.loader import release_gpu_memory
    from llmx.training.sft import train

    set_seed(cfg.seed)
    splits = load_splits(cfg)
    train_records = splits["train"][: cfg.train.max_train_examples]  # None -> all
    report = train(cfg, train_records, splits["val"])
    write_json(cfg.run_reports_dir / "train_report.json", report)
    write_json(cfg.run_reports_dir / "config.json", cfg.to_dict())
    release_gpu_memory()
    return report


# --------------------------------------------------------------------------- evaluation
def score(
    records: list[Record], raw_outputs: list[str], fields: list[str], seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure scoring step, separated from generation so it can be re-run on saved outputs."""
    parsed = [parse_prediction(o, fields) for o in raw_outputs]
    golds = [r["entities"] for r in records]
    texts = [r["text"] for r in records]

    metrics = evaluate(golds, parsed, texts, fields)
    metrics["micro_f1_ci95"] = bootstrap_f1_ci(golds, parsed, fields, seed=seed)
    errors = [
        categorize_errors(g, p, t, fields) for g, p, t in zip(golds, parsed, texts, strict=True)
    ]
    metrics["error_summary"] = summarize_errors(errors)

    rows = [
        {
            "id": r["id"],
            "text": r["text"],
            "gold": r["entities"],
            "raw_output": raw,
            "pred": p.entities,
            "valid_json": p.is_valid_json,
            "errors": [e.__dict__ for e in errs],
        }
        for r, raw, p, errs in zip(records, raw_outputs, parsed, errors, strict=True)
    ]
    return metrics, rows


def run_evaluation(
    cfg: ProjectConfig,
    system: str,
    split: str = "val",
    k_shots: int = 0,
    use_adapter: bool = False,
    limit: int | None = None,
    force_test: bool = False,
) -> dict[str, Any]:
    """Evaluate one system: prompt-only (``use_adapter=False``) or fine-tuned.

    ``system`` is a free label used in file names, e.g. "zero_shot", "few_shot3", "qlora".
    """
    if split not in SPLITS or split == "train":
        raise ValueError("evaluate on 'val' (development) or 'test' (once, at the end)")
    guard = HoldoutGuard(Path(cfg.paths.reports_dir) / "test_locks")
    guard_tag = f"{cfg.run_name}__{system}"
    if split == "test":
        if limit is not None:
            raise ValueError("the test set is always evaluated in full")
        guard.check(guard_tag, force=force_test)  # fail fast, before loading the model

    from llmx.inference.generate import predict
    from llmx.modeling.loader import (
        load_base_model,
        load_finetuned,
        load_tokenizer,
        release_gpu_memory,
    )

    set_seed(cfg.seed)
    splits = load_splits(cfg)
    records = splits[split][:limit] if limit else splits[split]
    shots = select_shots(splits["train"], k_shots, cfg.seed)  # shots come from TRAIN only
    fields = cfg.data.fields

    tokenizer = load_tokenizer(cfg.model)
    model = (
        load_finetuned(cfg.model, cfg.adapter_dir) if use_adapter else load_base_model(cfg.model)
    )
    raw_outputs, sec_per_example = predict(model, tokenizer, records, fields, cfg.generation, shots)
    del model
    release_gpu_memory()
    if split == "test":  # lock only once predictions exist (a crash must not burn the test set)
        guard.acquire(guard_tag, force=force_test)

    metrics, rows = score(records, raw_outputs, fields, cfg.seed)
    metrics.update(
        {
            "system": system,
            "split": split,
            "k_shots": k_shots,
            "seconds_per_example": sec_per_example,
        }
    )
    out_dir = cfg.run_reports_dir / f"{system}_{split}"
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def collect_metrics(cfg: ProjectConfig) -> list[dict[str, Any]]:
    """All metrics.json of the run, flattened for a comparison table."""
    rows = []
    for path in sorted(cfg.run_reports_dir.glob("*_*/metrics.json")):
        m = read_json(path)
        rows.append(
            {
                "system": m["system"],
                "split": m["split"],
                "n": m["n"],
                "micro_f1": m["micro"]["f1"],
                "f1_ci95": tuple(m["micro_f1_ci95"]),
                "precision": m["micro"]["precision"],
                "recall": m["micro"]["recall"],
                "exact_match": m["exact_match"],
                "json_valid": m["json_valid_rate"],
                "strict_json": m["strict_json_rate"],
                "hallucination": m["hallucination_rate"],
                "s_per_example": m["seconds_per_example"],
            }
        )
    return rows
