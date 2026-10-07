from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from attnrank.data.classes import (
    DEFAULT_MIN_EDGE_RATIO,
    DEFAULT_SEED,
    EngineSettings,
    GenerationSettings,
    LayerScanSettings,
    ProfileSettings,
    SyntheticProbeSettings,
)

CORE_MODULE_NAME = "attnrank._core"
CORE_FILE_PATTERN = "_core*.so"
PACKAGE_DIR = Path(__file__).resolve().parent
DEV_BUILD_DIRS = (PACKAGE_DIR.parent / "build" / "python", PACKAGE_DIR.parent / "build")
CHECKPOINT_FILE_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model.bin.index.json",
)
MODEL_CONFIG_FILE = "config.json"
WEIGHT_FILE_PATTERNS = ("*.safetensors", "pytorch_model*.bin")
MINIMUM_BASIN_SLOTS = 3

ProgressCallback = Callable[[int, int], None]


def _import_core() -> Any:
    try:
        return importlib.import_module(CORE_MODULE_NAME)
    except ImportError:
        pass
    for directory in DEV_BUILD_DIRS:
        for candidate in directory.glob(CORE_FILE_PATTERN):
            spec = importlib.util.spec_from_file_location(CORE_MODULE_NAME, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[CORE_MODULE_NAME] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        "attnrank._core extension not found; run 'pip install .' or "
        "'cmake -S . -B build && cmake --build build -j' from the AttnRank directory"
    )


core = _import_core()

Document = core.Document
ProbeSample = core.ProbeSample
AttentionProfile = core.AttentionProfile
Engine = core.Engine
AttnRankError = core.AttnRankError


def set_log_level(*, level: str = "info") -> None:
    core.set_log_level(level)


def is_model_directory(*, path: Path) -> bool:
    if not (path / MODEL_CONFIG_FILE).is_file():
        return False
    return any(any(path.glob(pattern)) for pattern in WEIGHT_FILE_PATTERNS)


def download_model(
    *,
    repo_id: str,
    out_dir: Path | None = None,
    revision: str | None = None,
    token: str | None = None,
    max_workers: int = 4,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("download_model needs huggingface_hub: pip install 'attnrank[hub]'") from exc
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=None if out_dir is None else str(out_dir),
        allow_patterns=list(CHECKPOINT_FILE_PATTERNS),
        max_workers=max_workers,
    )
    return Path(path)


def resolve_model_directory(
    *,
    model: str | Path,
    cache_dir: Path | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    candidate = Path(model)
    if is_model_directory(path=candidate):
        return candidate.resolve()
    if candidate.exists():
        raise FileNotFoundError(f"{model} exists but holds no {MODEL_CONFIG_FILE} and weight shards")
    out_dir = None if cache_dir is None else cache_dir / str(model).replace("/", "--")
    if out_dir is not None and is_model_directory(path=out_dir):
        return out_dir.resolve()
    return download_model(
        repo_id=str(model),
        out_dir=out_dir,
        revision=revision,
        token=token,
    ).resolve()


def model_config(*, model_dir: Path) -> dict[str, Any]:
    with (model_dir / MODEL_CONFIG_FILE).open(encoding="utf-8") as handle:
        return json.load(handle)


def detect_chat_format(*, model_dir: Path) -> str:
    return core.detect_chat_format(str(model_dir))


def chat_format_names() -> list[str]:
    return list(core.chat_format_names())


def reranker_names() -> list[str]:
    return list(core.reranker_names())


def load_model(*, model_dir: Path, settings: EngineSettings = EngineSettings()) -> Engine:
    options = core.RuntimeOptions(
        device_index=settings.device,
        max_sequence_length=settings.max_sequence,
        prefill_chunk_tokens=settings.chunk,
    )
    template = core.PromptTemplate()
    if settings.instruction is not None:
        template.instruction = settings.instruction
    if settings.example is not None:
        template.example = settings.example
    template.max_document_tokens = settings.max_document_tokens
    template.include_titles = settings.include_titles
    template.document_separator = settings.document_separator
    if settings.document_label is not None:
        template.document_label = settings.document_label
    template.extra_stop_sequences = list(settings.extra_stop_sequences)
    return Engine(str(model_dir), options, settings.chat_format, settings.system_prompt, template)


def from_pretrained(
    *,
    model: str | Path,
    settings: EngineSettings = EngineSettings(),
    cache_dir: Path | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Engine:
    model_dir = resolve_model_directory(
        model=model,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
    )
    return load_model(model_dir=model_dir, settings=settings)


def load_model_and_profile(
    *,
    model: str | Path,
    profile: Path | AttentionProfile,
    settings: EngineSettings = EngineSettings(),
) -> tuple[Engine, AttentionProfile]:
    engine = from_pretrained(model=model, settings=settings)
    loaded = profile if isinstance(profile, AttentionProfile) else load_attention_profile(path=profile)
    return engine, loaded


def make_document(*, item: Any, fallback_id: str = "") -> Document:
    if isinstance(item, Document):
        return item
    if isinstance(item, str):
        return Document(id=fallback_id, title="", text=item)
    if isinstance(item, Mapping):
        return Document(
            id=str(item.get("id", fallback_id)),
            title=str(item.get("title", "")),
            text=str(item.get("text", item.get("content", ""))),
        )
    raise TypeError(f"cannot convert {type(item).__name__} to Document")


def make_documents(*, items: Iterable[Any], prefix: str = "doc") -> list[Document]:
    return [make_document(item=item, fallback_id=f"{prefix}-{index}") for index, item in enumerate(items)]


def make_probe_sample(*, question: str, documents: Iterable[Any], sample_id: str = "sample") -> ProbeSample:
    return ProbeSample(question, make_documents(items=documents, prefix=sample_id))


def make_probe_samples(*, items: Iterable[Any]) -> list[ProbeSample]:
    samples = []
    for index, item in enumerate(items):
        if isinstance(item, ProbeSample):
            samples.append(item)
        elif isinstance(item, Mapping):
            samples.append(
                make_probe_sample(
                    question=item["question"],
                    documents=item["documents"],
                    sample_id=f"sample-{index}",
                )
            )
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            samples.append(
                make_probe_sample(
                    question=item[0],
                    documents=item[1],
                    sample_id=f"sample-{index}",
                )
            )
        else:
            raise TypeError(f"cannot convert item {index} ({type(item).__name__}) to ProbeSample")
    return samples


def load_probe_samples(*, path: Path, limit: int = 0) -> list[ProbeSample]:
    samples = core.load_probe_samples(str(path))
    return samples[:limit] if limit > 0 else samples


def synthetic_probe_samples(*, settings: SyntheticProbeSettings = SyntheticProbeSettings()) -> list[ProbeSample]:
    config = core.SyntheticProbeConfig()
    config.document_slots = settings.document_slots
    config.sample_count = settings.sample_count
    config.words_per_document = settings.words_per_document
    config.seed = settings.seed
    return core.synthetic_probe_samples(config)


def basin_statistics(*, attention: Sequence[float], min_edge_ratio: float = DEFAULT_MIN_EDGE_RATIO) -> Any:
    return core.basin_statistics(list(attention), core.BasinCriterion(min_edge_ratio))


def scan_attention_layers(
    *,
    engine: Engine,
    samples: Sequence[ProbeSample],
    settings: LayerScanSettings = LayerScanSettings(),
    progress: ProgressCallback | None = None,
) -> list[AttentionProfile]:
    config = core.LayerScanConfig()
    config.model_id = settings.model_id or engine.model_directory
    config.layer_begin = settings.layer_begin
    config.layer_end = -1 if settings.layer_end is None else settings.layer_end
    config.normalize_by_document_length = settings.normalize_by_length
    config.progress_interval = 0
    return core.scan_attention_layers(engine, list(samples), config, progress)


def select_shallowest_basin_layer(
    *,
    profiles: Sequence[AttentionProfile],
    min_edge_ratio: float = DEFAULT_MIN_EDGE_RATIO,
) -> int:
    return core.select_shallowest_basin_layer(list(profiles), core.BasinCriterion(min_edge_ratio))


@dataclass
class LayerScan:
    selected_layer: int
    min_edge_ratio: float
    profiles: list[AttentionProfile] = field(default_factory=list)

    def statistics(self) -> list[Any]:
        return [
            basin_statistics(attention=profile.attention, min_edge_ratio=self.min_edge_ratio)
            for profile in self.profiles
        ]

    def profile_for(self, *, layer_index: int) -> AttentionProfile:
        for profile in self.profiles:
            if profile.layer_index == layer_index:
                return profile
        raise KeyError(f"layer {layer_index} is not part of this scan")

    def highest_ratio_layer(self) -> int:
        ratios = [statistics.edge_ratio for statistics in self.statistics()]
        return self.profiles[max(range(len(ratios)), key=lambda index: ratios[index])].layer_index

    def table(self) -> str:
        lines = [f"{'layer':>5} {'first':>8} {'interior':>9} {'last':>8} {'ratio':>7}  basin"]
        for profile, stats in zip(self.profiles, self.statistics()):
            mark = "yes" if stats.is_basin else "-"
            if profile.layer_index == self.selected_layer:
                mark += "  <- selected"
            lines.append(
                f"{profile.layer_index:>5} {stats.first:>8.4f} {stats.interior_mean:>9.4f} "
                f"{stats.last:>8.4f} {stats.edge_ratio:>7.2f}  {mark}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_layer": self.selected_layer,
            "min_edge_ratio": self.min_edge_ratio,
            "profiles": [
                {
                    "layer_index": profile.layer_index,
                    "attention": list(profile.attention),
                    "position_order": list(profile.position_order),
                    "edge_ratio": stats.edge_ratio,
                    "is_basin": stats.is_basin,
                }
                for profile, stats in zip(self.profiles, self.statistics())
            ],
        }


def find_shallowest_attention_layer(
    *,
    engine: Engine,
    samples: Sequence[ProbeSample],
    settings: LayerScanSettings = LayerScanSettings(),
    progress: ProgressCallback | None = None,
) -> LayerScan:
    if len(samples) == 0:
        raise ValueError("at least one probe sample is required")
    if len(samples[0].documents) < MINIMUM_BASIN_SLOTS:
        raise ValueError(f"attention-basin detection needs at least {MINIMUM_BASIN_SLOTS} documents per sample")
    profiles = scan_attention_layers(
        engine=engine,
        samples=samples,
        settings=settings,
        progress=progress,
    )
    selected = select_shallowest_basin_layer(profiles=profiles, min_edge_ratio=settings.min_edge_ratio)
    return LayerScan(selected_layer=selected, min_edge_ratio=settings.min_edge_ratio, profiles=profiles)


def extract_attention_profile(
    *,
    engine: Engine,
    samples: Sequence[ProbeSample],
    settings: ProfileSettings,
    progress: ProgressCallback | None = None,
) -> AttentionProfile:
    if len(samples) == 0:
        raise ValueError("at least one probe sample is required")
    config = core.ProfilerConfig()
    config.model_id = settings.model_id or engine.model_directory
    config.layer_index = settings.layer_index
    config.average_all_layers = settings.average_all_layers
    if settings.layer_span is not None:
        config.layer_span_begin, config.layer_span_end = settings.layer_span[0], settings.layer_span[1] + 1
    config.normalize_by_document_length = settings.normalize_by_length
    config.progress_interval = 0
    return core.build_attention_profile(engine, list(samples), config, progress)


def save_attention_profile(*, profile: AttentionProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.save(str(path))


def load_attention_profile(*, path: Path) -> AttentionProfile:
    return AttentionProfile.load(str(path))


def profile_to_dict(*, profile: AttentionProfile) -> dict[str, Any]:
    return json.loads(profile.to_json_string())


def build_prompt(*, engine: Engine, question: str, documents: Iterable[Any]) -> Any:
    return engine.build_prompt(question, make_documents(items=documents))


def measure_document_attention(
    *,
    engine: Engine,
    question: str,
    documents: Iterable[Any],
    layer_index: int = 0,
    layer_span: tuple[int, int] | None = None,
    average_all_layers: bool = False,
    normalize_by_length: bool = False,
) -> list[float]:
    prompt = build_prompt(engine=engine, question=question, documents=documents)
    span_begin, span_end = (-1, -1) if layer_span is None else (layer_span[0], layer_span[1] + 1)
    return engine.measure_attention(
        prompt,
        layer_index=layer_index,
        layer_span_begin=span_begin,
        layer_span_end=span_end,
        average_all_layers=average_all_layers,
        normalize_by_length=normalize_by_length,
    )


def measure_attention_by_layer(
    *,
    engine: Engine,
    question: str,
    documents: Iterable[Any],
    layer_begin: int = 0,
    layer_end: int | None = None,
    normalize_by_length: bool = False,
) -> list[list[float]]:
    prompt = build_prompt(engine=engine, question=question, documents=documents)
    result = engine.measure_attention_by_layer(
        prompt,
        layer_begin=layer_begin,
        layer_end=-1 if layer_end is None else layer_end,
        normalize_by_length=normalize_by_length,
    )
    return [list(row) for row in result.attention_by_layer]


def attention_slot_order(*, position_order: Sequence[int], count: int) -> list[int]:
    return list(core.resolve_position_order(list(position_order), count))


def rerank_documents(*, documents_by_relevance: Sequence[Any], position_order: Sequence[int]) -> list[Any]:
    slots = attention_slot_order(position_order=position_order, count=len(documents_by_relevance))
    placed: list[Any] = [None] * len(documents_by_relevance)
    for rank, document in enumerate(documents_by_relevance):
        placed[slots[rank]] = document
    return placed


def rerank_with_profile(*, documents_by_relevance: Sequence[Any], profile: AttentionProfile) -> list[Any]:
    return rerank_documents(documents_by_relevance=documents_by_relevance, position_order=profile.position_order)


def make_reranker(
    *,
    strategy: str,
    position_order: Sequence[int] | None = None,
    seed: int = DEFAULT_SEED,
    lim_start_first: bool = False,
) -> Any:
    if strategy == "attnrank":
        if not position_order:
            raise ValueError("strategy 'attnrank' needs a position_order")
        return core.AttentionReranker(list(position_order))
    if strategy == "descending":
        return core.DescendingReranker()
    if strategy == "ascending":
        return core.AscendingReranker()
    if strategy == "random":
        return core.RandomReranker(seed)
    if strategy == "lim":
        return core.LostInTheMiddleReranker(lim_start_first)
    raise ValueError(f"unknown strategy '{strategy}'")


def rerank_baseline(
    *,
    documents_by_relevance: Sequence[Any],
    strategy: str,
    position_order: Sequence[int] | None = None,
    seed: int = DEFAULT_SEED,
    lim_start_first: bool = False,
) -> list[Any]:
    reranker = make_reranker(
        strategy=strategy,
        position_order=position_order,
        seed=seed,
        lim_start_first=lim_start_first,
    )
    count = len(documents_by_relevance)
    ranked = [core.ScoredDocument(index, float(count - index)) for index in range(count)]
    return [documents_by_relevance[scored.index] for scored in reranker.rerank(ranked)]


def generate_answer(
    *,
    engine: Engine,
    question: str,
    documents: Iterable[Any],
    settings: GenerationSettings = GenerationSettings(),
) -> Any:
    prompt = build_prompt(engine=engine, question=question, documents=documents)
    return generate_from_prompt(engine=engine, prompt=prompt, settings=settings)


def generate_from_prompt(*, engine: Engine, prompt: Any, settings: GenerationSettings = GenerationSettings()) -> Any:
    stops = list(engine.stop_sequences()) + list(settings.extra_stop_sequences)
    return engine.generate(
        prompt.tokens,
        max_new_tokens=settings.max_new_tokens,
        stop_sequences=stops,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        seed=settings.seed,
    )
