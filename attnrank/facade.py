from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from attnrank import engine as ar
from attnrank.data.classes import EngineSettings, GenerationSettings, LayerScanSettings, ProfileSettings
from attnrank.data.sources import DatasetSpec, load_rows

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class AttnRank:
    def __init__(self, *, profile: ar.AttentionProfile | None = None, engine: ar.Engine | None = None) -> None:
        self._profile = profile
        self._engine = engine

    @classmethod
    def from_profile(cls, *, path: Path) -> "AttnRank":
        return cls(profile=ar.load_attention_profile(path=path))

    @classmethod
    def from_pretrained(
        cls,
        *,
        model: str | Path,
        profile: Path | ar.AttentionProfile | None = None,
        settings: EngineSettings = EngineSettings(),
        cache_dir: Path | None = None,
    ) -> "AttnRank":
        engine = ar.from_pretrained(model=model, settings=settings, cache_dir=cache_dir)
        loaded = None
        if isinstance(profile, Path):
            loaded = ar.load_attention_profile(path=profile)
        elif profile is not None:
            loaded = profile
        return cls(profile=loaded, engine=engine)

    @property
    def profile(self) -> ar.AttentionProfile:
        if self._profile is None:
            raise RuntimeError("no attention profile yet: call build_profile() or pass profile=")
        return self._profile

    @property
    def engine(self) -> ar.Engine:
        if self._engine is None:
            raise RuntimeError("no model loaded: use AttnRank.from_pretrained(model=...)")
        return self._engine

    @property
    def position_order(self) -> list[int]:
        return list(self.profile.position_order)

    def build_profile(
        self,
        *,
        probes: DatasetSpec | Path | str | Sequence[Any],
        probe_samples: int = 0,
        layer: int | None = None,
        scan: LayerScanSettings = LayerScanSettings(),
        progress: ProgressCallback | None = None,
    ) -> ar.AttentionProfile:
        samples = self._probe_samples(probes=probes, limit=probe_samples)
        if layer is not None:
            profile = ar.extract_attention_profile(
                engine=self.engine,
                samples=samples,
                settings=ProfileSettings(layer_index=layer, normalize_by_length=scan.normalize_by_length),
                progress=progress,
            )
        else:
            result = ar.find_shallowest_attention_layer(
                engine=self.engine,
                samples=samples,
                settings=scan,
                progress=progress,
            )
            logger.info("layer scan:\n%s", result.table())
            if result.selected_layer < 0:
                raise RuntimeError("no attention-basin layer found, lower min_edge_ratio or add probe samples")
            profile = result.profile_for(layer_index=result.selected_layer)
        self._profile = profile
        return profile

    def save_profile(self, *, path: Path) -> None:
        ar.save_attention_profile(profile=self.profile, path=path)

    def rerank(self, *, documents_by_relevance: Sequence[Any]) -> list[Any]:
        return ar.rerank_with_profile(documents_by_relevance=documents_by_relevance, profile=self.profile)

    def slot_of_rank(self, *, count: int) -> list[int]:
        return ar.attention_slot_order(position_order=self.profile.position_order, count=count)

    def answer(
        self,
        *,
        question: str,
        documents_by_relevance: Sequence[Any],
        generation: GenerationSettings = GenerationSettings(),
    ) -> str:
        ordered = self.rerank(documents_by_relevance=documents_by_relevance)
        result = ar.generate_answer(engine=self.engine, question=question, documents=ordered, settings=generation)
        return result.text.strip()

    def prompt(self, *, question: str, documents_by_relevance: Sequence[Any]) -> str:
        ordered = self.rerank(documents_by_relevance=documents_by_relevance)
        return ar.build_prompt(engine=self.engine, question=question, documents=ordered).text

    def document_attention(self, *, question: str, documents: Iterable[Any]) -> list[float]:
        return ar.measure_document_attention(
            engine=self.engine,
            question=question,
            documents=documents,
            layer_index=self.profile.layer_index,
        )

    def _probe_samples(self, *, probes: DatasetSpec | Path | str | Sequence[Any], limit: int) -> list[Any]:
        if isinstance(probes, (DatasetSpec, Path, str)):
            spec = DatasetSpec.parse(value=probes)
            return ar.make_probe_samples(items=load_rows(spec=spec, limit=limit))
        items = list(probes)
        return ar.make_probe_samples(items=items[:limit] if limit > 0 else items)
