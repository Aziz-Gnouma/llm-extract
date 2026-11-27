import pytest

from llmx.data.splits import assert_no_overlap, build_splits, deduplicate, sample


def rec(i, text):
    return {"id": str(i), "text": text, "entities": {}}


def test_deduplicate_is_case_and_space_insensitive():
    out = deduplicate([rec(1, "Hello  World"), rec(2, "hello world"), rec(3, "other")])
    assert [r["id"] for r in out] == ["1", "3"]


def test_sample_is_deterministic():
    data = [rec(i, f"s{i}") for i in range(100)]
    assert sample(data, 10, seed=1) == sample(data, 10, seed=1)
    assert sample(data, 10, seed=1) != sample(data, 10, seed=2)


def test_build_splits_removes_leaks_with_test_priority():
    shared = "the same sentence"
    train = [rec("t1", shared), rec("t2", "train only")]
    val = [rec("v1", shared.upper()), rec("v2", "val only")]
    test = [rec("x1", shared)]
    splits = build_splits(train, val, test, n_train=10, n_val=10, n_test=10, seed=0)
    assert [r["id"] for r in splits["test"]] == ["x1"]
    assert [r["id"] for r in splits["val"]] == ["v2"]
    assert [r["id"] for r in splits["train"]] == ["t2"]


def test_assert_no_overlap_raises():
    with pytest.raises(AssertionError):
        assert_no_overlap({"train": [rec(1, "a b")], "test": [rec(2, "A  b")]})
