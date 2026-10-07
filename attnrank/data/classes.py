from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from attnrank.data.sources import DatasetSpec

DEFAULT_MAX_SEQUENCE = 4096
DEFAULT_PREFILL_CHUNK = 512
DEFAULT_MIN_EDGE_RATIO = 1.3
DEFAULT_PROBE_SAMPLES = 400
DEFAULT_SEED = 1234
DEFAULT_SYNTHETIC_SEED = 20250807
STRATEGIES = ("random", "descending", "ascending", "lim", "attnrank")


@dataclass(frozen=True, slots=True)
class EngineSettings:
    device: int = -1
    max_sequence: int = DEFAULT_MAX_SEQUENCE
    chunk: int = DEFAULT_PREFILL_CHUNK
    chat_format: str = ""
    system_prompt: str = ""
    instruction: str | None = None
    example: str | None = None
    max_document_tokens: int = 0
    include_titles: bool = False
    document_separator: str = "\n"
    document_label: str | None = None
    extra_stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SyntheticProbeSettings:
    document_slots: int = 5
    sample_count: int = 32
    words_per_document: int = 60
    seed: int = DEFAULT_SYNTHETIC_SEED


@dataclass(frozen=True, slots=True)
class LayerScanSettings:
    layer_begin: int = 0
    layer_end: int | None = None
    min_edge_ratio: float = DEFAULT_MIN_EDGE_RATIO
    normalize_by_length: bool = False
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    layer_index: int
    layer_span: tuple[int, int] | None = None
    average_all_layers: bool = False
    normalize_by_length: bool = False
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 0
    seed: int = DEFAULT_SEED
    extra_stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProfileTaskSettings:
    model: str
    output_dir: Path | None = None
    probes: DatasetSpec | None = None
    probe_samples: int = DEFAULT_PROBE_SAMPLES
    synthetic_slots: int = 5
    layer: int | None = None
    layer_begin: int = 0
    layer_end: int | None = None
    min_edge_ratio: float = DEFAULT_MIN_EDGE_RATIO
    normalize_by_length: bool = False
    profile_name: str = "profile.json"
    engine: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HotpotqaSettings:
    model: str
    dataset: DatasetSpec
    output_dir: Path | None = None
    profile: Path | None = None
    probes: DatasetSpec | None = None
    probe_samples: int = DEFAULT_PROBE_SAMPLES
    layer: int | None = None
    min_edge_ratio: float = DEFAULT_MIN_EDGE_RATIO
    ordering: str = "bm25"
    strategies: tuple[str, ...] = STRATEGIES
    offset: int = 0
    questions: int = 0
    max_new_tokens: int = 32
    seed: int = 0
    paper_row: str = "qwen2.5-7b"
    engine: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FinqaSettings:
    model: str
    dataset: DatasetSpec
    output_dir: Path | None = None
    profile: Path | None = None
    dataset_name: str | None = None
    examples: int | None = None
    chunks: int = 5
    strategy: str = "attnrank"
    probe_samples: int = 200
    min_edge_ratio: float = DEFAULT_MIN_EDGE_RATIO
    temperature: float = 0.0
    top_p: float = 0.8
    max_new_tokens: int = 512
    max_retries: int = 3
    seed: int = 0
    shard: str = "0/1"
    rerun: bool = False
    profile_only: bool = False
    finalize: bool = False
    engine: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    id: str
    strategy: str
    question: str
    reference: str
    prediction: str
    hit: bool
    gold_slots: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "question": self.question,
            "reference": self.reference,
            "prediction": self.prediction,
            "hit": self.hit,
            "gold_slots": list(self.gold_slots),
        }


@dataclass(frozen=True, slots=True)
class FinqaExample:
    id: str
    question: str
    principle: str
    gold_response: str
    rejected: str | None = None
    meta: Any = None

    @staticmethod
    def from_dict(*, row: dict[str, Any]) -> "FinqaExample":
        return FinqaExample(
            id=str(row["id"]),
            question=row["question"],
            principle=row["principle"],
            gold_response=row["gold_response"],
            rejected=row.get("rejected"),
            meta=row.get("meta"),
        )


@dataclass(frozen=True, slots=True)
class ShardSpec:
    index: int
    count: int

    @staticmethod
    def parse(*, text: str) -> "ShardSpec":
        index, count = (int(part) for part in text.split("/"))
        if count <= 0 or index < 0 or index >= count:
            raise ValueError(f"shard must be INDEX/COUNT with 0 <= INDEX < COUNT, got '{text}'")
        return ShardSpec(index=index, count=count)

    def select(self, *, items: Sequence[Any]) -> list[tuple[int, Any]]:
        size = (len(items) + self.count - 1) // self.count
        return list(enumerate(items))[self.index * size : (self.index + 1) * size]
