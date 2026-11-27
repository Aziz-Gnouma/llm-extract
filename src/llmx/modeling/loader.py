"""Load the base model in 4-bit (QLoRA) and attach / reload LoRA adapters."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llmx.config import LoraConfig, ModelConfig


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only generation needs left padding
    return tokenizer


def load_base_model(cfg: ModelConfig):
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype(cfg.compute_dtype),
        )
    return AutoModelForCausalLM.from_pretrained(
        cfg.name,
        quantization_config=quant,
        torch_dtype=_dtype(cfg.compute_dtype),
        device_map="auto",
    )


def attach_lora(model, cfg: LoraConfig, gradient_checkpointing: bool = True):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_cfg)


def load_finetuned(cfg: ModelConfig, adapter_dir: Path):
    return PeftModel.from_pretrained(load_base_model(cfg), str(adapter_dir))


def count_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def release_gpu_memory() -> None:
    """Call after `del model` to return cached GPU memory between experiments (T4 = 16 GB)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
