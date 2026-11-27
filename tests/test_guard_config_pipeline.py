import pytest

from llmx.config import config_from_dict, load_config
from llmx.data.prompts import build_messages, build_training_messages, select_shots
from llmx.evaluation.guard import HoldoutAlreadyUsedError, HoldoutGuard
from llmx.pipeline import score


def test_base_config_loads_and_fields_are_ordered():
    cfg = load_config("configs/base.yaml")
    assert cfg.data.fields == ["persons", "organizations", "locations"]
    assert cfg.model.compute_dtype == "float16"


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="lerning_rate"):
        config_from_dict({"train": {"lerning_rate": 1e-4}})


def test_guard_blocks_second_test_use(tmp_path):
    guard = HoldoutGuard(tmp_path)
    guard.acquire("run__qlora")
    with pytest.raises(HoldoutAlreadyUsedError):
        guard.check("run__qlora")
    guard.acquire("run__qlora", force=True)  # explicit override is allowed and logged
    assert guard.is_used("run__qlora")


def test_prompts_share_the_same_instruction():
    rec = {
        "id": "1",
        "text": "Ann in Rome",
        "entities": {"persons": ["Ann"], "locations": ["Rome"]},
    }
    fields = ["persons", "locations"]
    infer = build_messages(rec["text"], fields)
    train = build_training_messages(rec, fields)
    assert train[:-1] == infer
    assert train[-1] == {
        "role": "assistant",
        "content": '{"persons": ["Ann"], "locations": ["Rome"]}',
    }


def test_select_shots_only_uses_examples_with_entities():
    train = [
        {"id": str(i), "text": "t", "entities": {"persons": ["A"] if i % 2 else []}}
        for i in range(10)
    ]
    shots = select_shots(train, 3, seed=0)
    assert len(shots) == 3 and all(s["entities"]["persons"] for s in shots)
    assert select_shots(train, 0, seed=0) == []


def test_score_end_to_end_without_gpu():
    fields = ["persons", "locations"]
    records = [
        {"id": "1", "text": "Ann in Rome", "entities": {"persons": ["Ann"], "locations": ["Rome"]}}
    ]
    metrics, rows = score(records, ['{"persons": ["Ann"], "locations": []}'], fields, seed=0)
    assert metrics["micro"]["recall"] == 0.5
    assert metrics["error_summary"]["missed"] == 1
    assert rows[0]["pred"] == {"persons": ["Ann"], "locations": []}
