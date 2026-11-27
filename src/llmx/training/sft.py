"""QLoRA supervised fine-tuning on top of ``transformers.Trainer``.

We tokenise ourselves (instead of a high-level SFT wrapper) so the label masking is explicit
and stable across library versions: prompt tokens get label -100, only the JSON answer
(plus the chat template's end-of-turn token) contributes to the loss.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from llmx.config import ProjectConfig
from llmx.data.conll import Record
from llmx.data.prompts import build_training_messages
from llmx.modeling.loader import attach_lora, count_parameters, load_base_model, load_tokenizer
from llmx.training.schedule import effective_eval_steps, total_optimizer_steps

IGNORE_INDEX = -100


def tokenize_example(
    record: Record, tokenizer, fields: Sequence[str], max_len: int
) -> dict[str, list[int]] | None:
    """Return input_ids / labels with the prompt masked, or None if too long."""
    messages = build_training_messages(record, fields)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template is not prefix-consistent; cannot isolate the answer tokens."
        )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full_text[len(prompt_text) :], add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_len:
        return None
    return {"input_ids": input_ids, "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids}


@dataclass
class CompletionCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in batch)
        ids, labels, mask = [], [], []
        for x in batch:
            pad = width - len(x["input_ids"])  # right padding for training
            ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            labels.append(x["labels"] + [IGNORE_INDEX] * pad)
            mask.append([1] * len(x["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def _tokenize_all(records, tokenizer, fields, max_len) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for rec in records:
        ex = tokenize_example(rec, tokenizer, fields, max_len)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


def train(
    cfg: ProjectConfig, train_records: Sequence[Record], val_records: Sequence[Record]
) -> dict[str, Any]:
    """Fine-tune, save the best adapter (lowest val loss) and return a training report."""
    fields = cfg.data.fields
    if torch.cuda.is_available():
        # max_memory_allocated is a process-wide high-water mark: reset it so the report
        # gives this run's peak, not the max over every run in the same kernel.
        torch.cuda.reset_peak_memory_stats()
    tokenizer = load_tokenizer(cfg.model)
    model = attach_lora(load_base_model(cfg.model), cfg.lora)
    params = count_parameters(model)

    train_ds, dropped_train = _tokenize_all(train_records, tokenizer, fields, cfg.model.max_seq_len)
    val_ds, dropped_val = _tokenize_all(val_records, tokenizer, fields, cfg.model.max_seq_len)

    t = cfg.train
    total_steps = total_optimizer_steps(len(train_ds), t.batch_size, t.grad_accum, t.epochs)
    eval_steps = effective_eval_steps(total_steps, t.eval_steps)
    args = TrainingArguments(
        output_dir=str(cfg.run_output_dir / "checkpoints"),
        num_train_epochs=t.epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.scheduler,
        warmup_ratio=t.warmup_ratio,
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.model.compute_dtype == "float16",
        bf16=cfg.model.compute_dtype == "bfloat16",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    trainer.save_model(str(cfg.adapter_dir))  # adapter weights only (a few MB)
    tokenizer.save_pretrained(str(cfg.adapter_dir))

    report = {
        "train_seconds": elapsed,
        "parameters": params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "total_steps": total_steps,
        "eval_steps": eval_steps,
        "dropped_too_long": {"train": dropped_train, "val": dropped_val},
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "peak_gpu_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        ),
        "log_history": trainer.state.log_history,
    }
    del trainer, model
    return report
