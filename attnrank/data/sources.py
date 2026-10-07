from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from attnrank.data.loaders import iter_jsonl
from attnrank.interface.dataset import RowAdapter

logger = logging.getLogger(__name__)

HUB_PREFIX = "hf:"
DEFAULT_SPLIT = "train"
GOLD_FLAG = "is_gold"


@dataclass(frozen=True, slots=True)
class FieldMapping:
    id: str = "id"
    question: str = "question"
    answer: str = "answer"
    documents: str = "documents"
    document_title: str = "title"
    document_text: str = "text"
    document_gold: str = GOLD_FLAG


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    path: Path | None = None
    hub: str | None = None
    config: str | None = None
    split: str = DEFAULT_SPLIT
    adapter: str = "generic"
    fields: FieldMapping = field(default_factory=FieldMapping)

    @staticmethod
    def parse(*, value: Any) -> "DatasetSpec":
        if isinstance(value, DatasetSpec):
            return value
        if isinstance(value, (str, Path)):
            return DatasetSpec.from_text(text=str(value))
        if isinstance(value, Mapping):
            values = dict(value)
            fields = FieldMapping(**values.pop("fields", {}) or {})
            if "path" in values and values["path"] is not None:
                values["path"] = Path(values["path"])
            return DatasetSpec(fields=fields, **values)
        raise TypeError(f"cannot build a dataset spec from {type(value).__name__}")

    @staticmethod
    def from_text(*, text: str) -> "DatasetSpec":
        if not text.startswith(HUB_PREFIX):
            return DatasetSpec(path=Path(text))
        reference, _, split = text[len(HUB_PREFIX) :].partition("@")
        hub, _, config = reference.partition(":")
        return DatasetSpec(hub=hub, config=config or None, split=split or DEFAULT_SPLIT)

    def describe(self) -> str:
        if self.hub is not None:
            config = f":{self.config}" if self.config else ""
            return f"{HUB_PREFIX}{self.hub}{config}@{self.split}"
        return str(self.path)


def generic_adapter(*, fields: FieldMapping) -> RowAdapter:
    def adapt(row: Mapping[str, Any]) -> dict[str, Any]:
        documents = []
        for index, item in enumerate(row[fields.documents]):
            if isinstance(item, str):
                documents.append({"id": f"doc-{index}", "title": "", "text": item})
                continue
            documents.append(
                {
                    "id": str(item.get("id", f"doc-{index}")),
                    "title": str(item.get(fields.document_title, "")),
                    "text": str(item.get(fields.document_text, "")),
                    GOLD_FLAG: str(item.get(fields.document_gold, "")),
                }
            )
        return {
            "id": str(row.get(fields.id, "")),
            "question": row[fields.question],
            "answer": row.get(fields.answer, ""),
            "documents": documents,
        }

    return adapt


def hotpot_qa_adapter(*, fields: FieldMapping) -> RowAdapter:
    def adapt(row: Mapping[str, Any]) -> dict[str, Any]:
        context = row["context"]
        gold_titles = set(row.get("supporting_facts", {}).get("title", []))
        documents = [
            {
                "id": f"doc-{index}",
                "title": title,
                "text": "".join(sentences),
                GOLD_FLAG: str(title in gold_titles).lower(),
            }
            for index, (title, sentences) in enumerate(zip(context["title"], context["sentences"]))
        ]
        return {"id": str(row["id"]), "question": row["question"], "answer": row["answer"], "documents": documents}

    return adapt


def passthrough_adapter(*, fields: FieldMapping) -> RowAdapter:
    return lambda row: dict(row)


ADAPTERS: dict[str, Callable[..., RowAdapter]] = {
    "generic": generic_adapter,
    "hotpot_qa": hotpot_qa_adapter,
    "raw": passthrough_adapter,
}


def resolve_adapter(*, name: str, fields: FieldMapping) -> RowAdapter:
    if name in ADAPTERS:
        return ADAPTERS[name](fields=fields)
    module_name, _, attribute = name.partition(":")
    if not attribute:
        raise ValueError(f"unknown adapter '{name}', expected one of {sorted(ADAPTERS)} or 'module:function'")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(fields=fields)


class JsonlDatasetSource:
    def __init__(self, *, path: Path, adapter: RowAdapter) -> None:
        self._path = path
        self._adapter = adapter

    def rows(self) -> list[dict[str, Any]]:
        return [self._adapter(row) for row in iter_jsonl(path=self._path)]


class HuggingFaceDatasetSource:
    def __init__(self, *, repo_id: str, config: str | None, split: str, adapter: RowAdapter) -> None:
        self._repo_id = repo_id
        self._config = config
        self._split = split
        self._adapter = adapter

    def rows(self) -> list[dict[str, Any]]:
        from datasets import load_dataset

        dataset = load_dataset(self._repo_id, self._config, split=self._split)
        return [self._adapter(row) for row in dataset]


class MemoryDatasetSource:
    def __init__(self, *, rows: Iterable[Mapping[str, Any]], adapter: RowAdapter) -> None:
        self._rows = list(rows)
        self._adapter = adapter

    def rows(self) -> list[dict[str, Any]]:
        return [self._adapter(row) for row in self._rows]


def build_dataset_source(*, spec: DatasetSpec) -> JsonlDatasetSource | HuggingFaceDatasetSource:
    adapter = resolve_adapter(name=spec.adapter, fields=spec.fields)
    if spec.hub is not None:
        return HuggingFaceDatasetSource(repo_id=spec.hub, config=spec.config, split=spec.split, adapter=adapter)
    if spec.path is None:
        raise ValueError("dataset spec needs either 'path' or 'hub'")
    return JsonlDatasetSource(path=spec.path, adapter=adapter)


def load_rows(*, spec: DatasetSpec, offset: int = 0, limit: int = 0) -> list[dict[str, Any]]:
    rows = build_dataset_source(spec=spec).rows()
    selected = rows[offset:]
    if limit > 0:
        selected = selected[:limit]
    logger.info("dataset %s: %d rows, using %d", spec.describe(), len(rows), len(selected))
    return selected
