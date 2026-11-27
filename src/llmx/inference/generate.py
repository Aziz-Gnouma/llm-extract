"""Batched greedy generation (deterministic) for evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
from tqdm.auto import tqdm

from llmx.config import GenerationConfig
from llmx.data.conll import Record
from llmx.data.prompts import build_messages


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False).to(
        model.device
    )
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: reproducible outputs
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def predict(
    model,
    tokenizer,
    records: Sequence[Record],
    fields: Sequence[str],
    gen_cfg: GenerationConfig,
    shots: Sequence[Record] = (),
) -> tuple[list[str], float]:
    """Return raw generations and seconds per example."""
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(r["text"], fields, shots), tokenize=False, add_generation_prompt=True
        )
        for r in records
    ]
    outputs: list[str] = []
    start = time.perf_counter()
    bs = gen_cfg.batch_size
    for i in tqdm(range(0, len(prompts), bs), desc="generate"):
        outputs.extend(
            generate_batch(model, tokenizer, prompts[i : i + bs], gen_cfg.max_new_tokens)
        )
    return outputs, (time.perf_counter() - start) / max(1, len(records))
