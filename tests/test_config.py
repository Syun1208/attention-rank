from __future__ import annotations

from pathlib import Path

import pytest

from attnrank.data.classes import HotpotqaSettings, ShardSpec
from attnrank.data.sources import DatasetSpec, FieldMapping, hotpot_qa_adapter
from attnrank.utils.config import build_settings


def test_build_settings_converts_paths_and_tuples() -> None:
    settings = build_settings(
        HotpotqaSettings,
        values={"model": "m", "dataset": "data/x.jsonl", "output_dir": "out", "strategies": ["random", "attnrank"]},
        overrides={"questions": 5, "engine": {"device": 1}},
        source="test",
    )
    assert settings.dataset.path == Path("data/x.jsonl")
    assert settings.dataset.hub is None
    assert settings.strategies == ("random", "attnrank")
    assert settings.questions == 5
    assert settings.engine == {"device": 1}


def test_build_settings_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        build_settings(
            HotpotqaSettings,
            values={"model": "m", "dataset": "d", "output_dir": "o", "bogus": 1},
            overrides={},
            source="test",
        )


def test_dataset_spec_parses_hub_references() -> None:
    spec = DatasetSpec.parse(value="hf:hotpotqa/hotpot_qa:distractor@validation")
    assert (spec.hub, spec.config, spec.split) == ("hotpotqa/hotpot_qa", "distractor", "validation")
    assert spec.describe() == "hf:hotpotqa/hotpot_qa:distractor@validation"


def test_hotpot_qa_adapter_marks_supporting_titles_as_gold() -> None:
    adapt = hotpot_qa_adapter(fields=FieldMapping())
    row = {
        "id": "q1",
        "question": "who?",
        "answer": "x",
        "context": {"title": ["A", "B"], "sentences": [["a1 ", "a2"], ["b1"]]},
        "supporting_facts": {"title": ["B"], "sent_id": [0]},
    }
    adapted = adapt(row)
    assert adapted["documents"][0] == {"id": "doc-0", "title": "A", "text": "a1 a2", "is_gold": "false"}
    assert adapted["documents"][1]["is_gold"] == "true"


def test_shard_spec_selects_contiguous_slices() -> None:
    shard = ShardSpec.parse(text="1/2")
    assert shard.select(items=["a", "b", "c"]) == [(2, "c")]
    with pytest.raises(ValueError):
        ShardSpec.parse(text="2/2")
