"""End-to-end GPU smoke test on tiny data (a few minutes, any CUDA GPU >= 4 GB).

Exercises every GPU code path before spending real GPU hours:
4-bit loading -> zero-shot generation -> QLoRA training (with eval + checkpoint)
-> adapter reload -> generation -> scoring. Writes only to outputs/smoke and reports/smoke;
never touches the test split.

    python scripts/smoke_test.py [--n-eval 20] [--n-train 48] [--model Qwen/Qwen2.5-0.5B-Instruct]

--model swaps in a smaller model of the same family (same chat template and architecture) to
check the code paths on a small GPU / small disk; results are NOT comparable to the real run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

from llmx import pipeline
from llmx.config import load_config, set_seed
from llmx.data.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=48)
    parser.add_argument("--model", default=None, help="override model.name")
    args = parser.parse_args()

    base = load_config(args.config)
    cfg = replace(
        base,
        run_name="smoke",
        model=replace(base.model, name=args.model or base.model.name),
        train=replace(
            base.train,
            epochs=1,
            batch_size=2,
            grad_accum=2,
            eval_steps=4,
            logging_steps=2,
            max_train_examples=args.n_train,
        ),
        generation=replace(base.generation, batch_size=8),
    )
    set_seed(cfg.seed)
    pipeline.prepare_data(cfg)

    import torch

    from llmx.training.sft import train

    t0 = time.perf_counter()
    zero = pipeline.run_evaluation(cfg, "zero_shot", "val", limit=args.n_eval)

    splits = pipeline.load_splits(cfg)
    torch.cuda.reset_peak_memory_stats()
    report = train(cfg, splits["train"][: args.n_train], splits["val"][: args.n_eval])
    write_json(cfg.run_reports_dir / "train_report.json", report)

    tuned = pipeline.run_evaluation(cfg, "qlora", "val", use_adapter=True, limit=args.n_eval)

    summary = {
        "model": cfg.model.name,
        "gpu": torch.cuda.get_device_name(0),
        "total_minutes": round((time.perf_counter() - t0) / 60, 2),
        "train_seconds": round(report["train_seconds"], 1),
        "peak_train_gpu_gb": round(report["peak_gpu_mem_gb"] or 0, 2),
        "losses": [
            {k: round(v, 4) for k, v in e.items() if k in ("step", "loss", "eval_loss")}
            for e in report["log_history"]
            if "loss" in e or "eval_loss" in e
        ],
        "trainable_params": report["parameters"]["trainable"],
        "zero_shot": _short(zero),
        "qlora_tiny": _short(tuned),
    }
    print(json.dumps(summary, indent=2))


def _short(m: dict) -> dict:
    return {
        "n": m["n"],
        "micro_f1": round(m["micro"]["f1"], 3),
        "json_valid": m["json_valid_rate"],
        "strict_json": m["strict_json_rate"],
        "hallucination": round(m["hallucination_rate"], 3),
        "s_per_example": round(m["seconds_per_example"], 3),
    }


if __name__ == "__main__":
    main()
