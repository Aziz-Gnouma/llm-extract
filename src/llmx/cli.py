"""Command-line entry point:  python -m llmx <command> --config configs/base.yaml"""

from __future__ import annotations

import argparse
import json

from llmx import pipeline
from llmx.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llmx")
    parser.add_argument("--config", default="configs/base.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="download, convert and freeze the splits")
    prep.add_argument("--force", action="store_true")

    sub.add_parser("train", help="QLoRA fine-tuning")

    ev = sub.add_parser("evaluate", help="evaluate one system")
    ev.add_argument("--system", required=True, help="label, e.g. zero_shot, few_shot3, qlora")
    ev.add_argument("--split", choices=["val", "test"], default="val")
    ev.add_argument("--k-shots", type=int, default=0)
    ev.add_argument("--adapter", action="store_true", help="use the fine-tuned adapter")
    ev.add_argument("--limit", type=int, default=None, help="quick smoke test on N examples")

    sub.add_parser("report", help="print the comparison table of the run")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "prepare":
        result = pipeline.prepare_data(cfg, force=args.force)
    elif args.command == "train":
        result = {k: v for k, v in pipeline.run_training(cfg).items() if k != "log_history"}
    elif args.command == "evaluate":
        result = pipeline.run_evaluation(
            cfg, args.system, args.split, args.k_shots, args.adapter, args.limit
        )
    else:
        result = pipeline.collect_metrics(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
