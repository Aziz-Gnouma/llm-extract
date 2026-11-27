# llm-extract

QLoRA fine-tuning of a small open LLM (Qwen2.5-1.5B-Instruct, 4-bit) to extract named entities
as JSON, compared to the same model with prompting only. Deep Learning module, project 9,
variant A. Specification: [docs/CAHIER_DES_CHARGES.md](docs/CAHIER_DES_CHARGES.md).

## Quick start

**On Colab / Kaggle (T4 GPU):** open `notebooks/main.ipynb` in a **fresh runtime** (a reused kernel
keeps an already-imported `llmx` in memory), then run all. The test locks of the submitted run are
committed in `reports/test_locks/`: re-running section 6 raises `HoldoutAlreadyUsedError` by design.
Delete that folder only to reproduce the whole protocol from scratch.

**From the command line:**

```bash
pip install -e ".[ml,dev]"
python -m pytest -q                                   # CPU-only unit tests
python -m llmx prepare                                # download + freeze splits
python -m llmx evaluate --system zero_shot            # baseline on validation
python -m llmx evaluate --system few_shot3 --k-shots 3
python -m llmx train                                  # QLoRA, best checkpoint on val loss
python -m llmx evaluate --system qlora --adapter
python -m llmx report                                 # comparison table
python -m llmx evaluate --system qlora --adapter --split test   # ONCE, at the very end
```

Add `--limit 20` to any validation run for a quick smoke test.

## Results

Test set: 500 CoNLL-2003 sentences, each system scored **once**. Final QLoRA configuration
(3 epochs) chosen beforehand on validation. Colab T4, commit `d2c2e78`.

| System | micro-F1 [95 % CI] | exact match | strict JSON | schema valid | hallucinations | s / sentence |
|---|---|---|---|---|---|---|
| zero-shot | 0.393 [0.354, 0.431] | 0.256 | 0 % | 96.8 % | 16.6 % | 0.18 |
| few-shot (3 examples) | 0.580 [0.545, 0.615] | 0.330 | 99.8 % | 65.2 % | 5.1 % | 0.20 |
| **QLoRA r=16, 3 epochs** | **0.902 [0.880, 0.921]** | **0.826** | **100 %** | **100 %** | **0.8 %** | 0.30 |

Training touches 1.2 % of the parameters (18.5 M), takes 20 min on a T4, and produces a 74 MB
adapter. Figures are in `reports/`; the critical analysis (in French) is section 7 of the notebook.

## Architecture

```
configs/base.yaml        every hyperparameter (typed + validated, unknown keys rejected)
src/llmx/
  config.py              frozen dataclasses, YAML loader, set_seed
  data/                  pure Python
    conll.py             BIO tags -> {"persons", "organizations", "locations"}
    splits.py            dedup, cross-split leak removal, seeded sampling
    prompts.py           one prompt shared by baseline and fine-tuned model
    io.py, loading.py    JSONL + SHA-256, Hugging Face download
  inference/
    parsing.py           pure: raw text -> validated JSON (3 strictness levels)
    generate.py          GPU: batched greedy generation
  evaluation/            pure
    metrics.py           set-based P/R/F1, exact match, hallucination rate, bootstrap CI
    errors.py            error taxonomy for the critical analysis
    guard.py             test set can be scored once per system
  modeling/loader.py     GPU: 4-bit loading, LoRA attach / reload
  training/sft.py        GPU: answer-only loss masking + HF Trainer
  pipeline.py            use cases called by both the CLI and the notebook
  cli.py
tests/                   41 unit tests (the tokenizer/collator ones are skipped without torch)
notebooks/main.ipynb     orchestration + figures + written analysis
```

Imports only point downwards: `pipeline` → GPU layers → pure layers → `config`. Everything that
decides a score (conversion, splitting, parsing, metrics, error taxonomy) is pure Python and unit
tested. Generation and scoring are separate steps, so saved outputs can be re-scored without a GPU.

## Reproducibility

* Dataset pinned to a commit; splits frozen in `data/processed/` with SHA-256 in `manifest.json`.
* Seed 42 everywhere, greedy decoding, library versions logged to `reports/environment.json`.
* Each run writes `reports/<run>/<system>_<split>/{metrics.json,predictions.jsonl}`,
  `train_report.json` (incl. full loss history) and a `config.json` snapshot.

## Evaluation protocol

| Metric | Meaning |
|---|---|
| `json_valid_rate` / `strict_json_rate` / `schema_valid_rate` | parsable / *only* JSON / exact keys and types |
| micro & per-field P/R/F1 (+ 95 % bootstrap CI) | set match after lower-casing and whitespace normalisation |
| `exact_match` | every field correct for the sentence |
| `hallucination_rate` | predicted values that do not occur (as whole tokens) in the input |

Invalid outputs are never repaired: they count as missed entities.
