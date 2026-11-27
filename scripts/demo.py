"""Try the fine-tuned model on your own sentences (GPU).

Prints, for each sentence, the answer of the base model (same prompt, adapter disabled) and the
answer of the QLoRA model, plus the parsed entities. Touches no split and writes nothing.

    python scripts/demo.py --adapter outputs/qwen1.5b-qlora-r16-e3/adapter \
        "Angela Merkel met Tim Cook at Apple's offices in Cupertino ."
    python scripts/demo.py --adapter ... --interactive      # type sentences, empty line to quit

Tip: put spaces around punctuation, like the CoNLL training data ("Apple 's", "Paris .").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmx.config import load_config
from llmx.data.prompts import build_messages
from llmx.inference.generate import generate_batch
from llmx.inference.parsing import parse_prediction
from llmx.modeling.loader import load_finetuned, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences", nargs="*")
    parser.add_argument(
        "--adapter", type=Path, default=Path("outputs/qwen1.5b-qlora-r16-e3/adapter")
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-base", action="store_true", help="skip the prompt-only answer")
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").exists():
        parser.error(f"no adapter found in {args.adapter}")
    cfg = load_config(args.config)
    fields = cfg.data.fields
    tokenizer = load_tokenizer(cfg.model)
    model = load_finetuned(cfg.model, args.adapter)
    model.eval()

    def answer(text: str) -> str:
        prompt = tokenizer.apply_chat_template(
            build_messages(text, fields), tokenize=False, add_generation_prompt=True
        )
        return generate_batch(model, tokenizer, [prompt], cfg.generation.max_new_tokens)[0]

    def show(text: str) -> None:
        print(f"\n>>> {text}")
        runs = [] if args.no_base else [("base (prompt only)", True)]
        for label, disabled in runs + [("QLoRA", False)]:
            if disabled:
                with model.disable_adapter():
                    raw = answer(text)
            else:
                raw = answer(text)
            parsed = parse_prediction(raw, fields)
            flags = f"strict JSON={parsed.is_strict_json}, schema OK={parsed.is_valid_schema}"
            print(f"--- {label}  ({flags})")
            print(f"raw    : {raw.strip()}")
            print(f"parsed : {json.dumps(parsed.entities, ensure_ascii=False)}")

    for s in args.sentences:
        show(s)
    if args.interactive or not args.sentences:
        while True:
            try:
                text = input("\nsentence (empty = quit) > ").strip()
            except EOFError:
                break
            if not text:
                break
            show(text)


if __name__ == "__main__":
    main()
