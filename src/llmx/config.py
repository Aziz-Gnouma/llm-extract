"""Typed, immutable experiment configuration loaded from YAML.

Unknown keys raise an error so that a typo in the YAML can never be silently ignored.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    hf_dataset: str = "lhoestq/conll2003"
    hf_revision: str | None = "19edcb426bfd625c275c17b9a99b2239243f4377"
    field_map: dict[str, str] = field(
        default_factory=lambda: {"PER": "persons", "ORG": "organizations", "LOC": "locations"}
    )
    min_tokens: int = 4
    n_train: int = 1500
    n_val: int = 300
    n_test: int = 500

    @property
    def fields(self) -> list[str]:
        """JSON keys of the extraction schema, in a stable order."""
        return list(dict.fromkeys(self.field_map.values()))


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    max_seq_len: int = 512


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(frozen=True)
class TrainConfig:
    epochs: float = 2
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    eval_steps: int = 25
    logging_steps: int = 5
    save_total_limit: int = 2
    max_train_examples: int | None = None  # ablation: train on the first N (random) examples


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    batch_size: int = 16


@dataclass(frozen=True)
class BaselineConfig:
    few_shot_k: int = 3


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class ProjectConfig:
    run_name: str = "default"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ---- derived paths (relative to the current working directory = repo root) ----
    @property
    def processed_dir(self) -> Path:
        return Path(self.paths.data_dir) / "processed"

    @property
    def run_output_dir(self) -> Path:
        return Path(self.paths.output_dir) / self.run_name

    @property
    def adapter_dir(self) -> Path:
        return self.run_output_dir / "adapter"

    @property
    def run_reports_dir(self) -> Path:
        return Path(self.paths.reports_dir) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "lora": LoraConfig,
    "train": TrainConfig,
    "generation": GenerationConfig,
    "baseline": BaselineConfig,
    "paths": PathsConfig,
}


def _build(cls: type, raw: dict[str, Any], where: str) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in '{where}': {sorted(unknown)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    raw = dict(raw)
    sections = {name: _build(cls, raw.pop(name, {}) or {}, name) for name, cls in _SECTIONS.items()}
    return _build(ProjectConfig, {**raw, **sections}, "root")


def load_config(path: str | os.PathLike[str]) -> ProjectConfig:
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh) or {})


def set_seed(seed: int) -> None:
    """Seed every RNG we use. torch / transformers are optional (CPU-only tests)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)  # also seeds torch + CUDA
    except ImportError:
        pass
