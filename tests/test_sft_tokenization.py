"""Checks label masking with the real chat template. Needs network + transformers,
so it is skipped on a plain CPU dev machine and runs on Colab/Kaggle."""

import os

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmx.training.sft import IGNORE_INDEX, CompletionCollator, tokenize_example  # noqa: E402

MODEL = os.environ.get("LLMX_TOKENIZER", "Qwen/Qwen2.5-1.5B-Instruct")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except OSError:
        pytest.skip("tokenizer not downloadable (offline)")


def test_only_answer_tokens_are_trained(tokenizer):
    rec = {
        "id": "1",
        "text": "Ann lives in Rome",
        "entities": {"persons": ["Ann"], "locations": ["Rome"]},
    }
    ex = tokenize_example(rec, tokenizer, ["persons", "locations"], max_len=512)
    trained = [
        t for t, lab in zip(ex["input_ids"], ex["labels"], strict=True) if lab != IGNORE_INDEX
    ]
    decoded = tokenizer.decode(trained)
    assert decoded.startswith('{"persons": ["Ann"], "locations": ["Rome"]}')
    assert "Ann lives in Rome" not in decoded


def test_too_long_examples_are_dropped(tokenizer):
    rec = {"id": "1", "text": "word " * 600, "entities": {"persons": []}}
    assert tokenize_example(rec, tokenizer, ["persons"], max_len=128) is None


def test_collator_pads_labels_with_ignore_index():
    batch = CompletionCollator(pad_token_id=0)(
        [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}]
    )
    assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
