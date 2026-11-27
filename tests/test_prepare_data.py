"""prepare_data on a fresh clone: manifest committed, split files absent."""

import pytest

from llmx import pipeline
from llmx.config import config_from_dict
from llmx.data.io import read_json, write_json


def fake_records(prefix, n):
    return [
        {
            "id": f"{prefix}-{i}",
            "text": f"{prefix} sentence number {i} about Paris",
            "entities": {"persons": [], "organizations": [], "locations": ["Paris"]},
        }
        for i in range(n)
    ]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llmx.data.loading.load_conll_records",
        lambda _: {
            "train": fake_records("tr", 30),
            "val": fake_records("va", 10),
            "test": fake_records("te", 10),
        },
    )
    return config_from_dict(
        {
            "data": {"n_train": 20, "n_val": 5, "n_test": 5},
            "paths": {"data_dir": str(tmp_path / "data")},
        }
    )


def test_fresh_clone_regenerates_identical_splits(cfg):
    first = pipeline.prepare_data(cfg)
    for name in pipeline.SPLITS:  # simulate a clone: only manifest.json is versioned
        (cfg.processed_dir / f"{name}.jsonl").unlink()
    assert pipeline.prepare_data(cfg) == first
    assert len(pipeline.load_splits(cfg)["train"]) == 20


def test_mismatch_with_committed_manifest_raises(cfg):
    pipeline.prepare_data(cfg)
    manifest_path = cfg.processed_dir / "manifest.json"
    tampered = read_json(manifest_path)
    tampered["splits"]["test"]["sha256"] = "0" * 64
    write_json(manifest_path, tampered)
    with pytest.raises(pipeline.SplitMismatchError):
        pipeline.prepare_data(cfg)
    assert pipeline.prepare_data(cfg, force=True)["splits"]["test"]["sha256"] != "0" * 64
