from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from attnrank import engine as ar
from attnrank.data.classes import (
    EngineSettings,
    LayerScanSettings,
    ProfileSettings,
    ProfileTaskSettings,
    SyntheticProbeSettings,
)
from attnrank.data.sources import DatasetSpec, load_rows
from attnrank.utils.config import write_settings_copy
from attnrank.utils.progress import PROBE, finish, progress

logger = logging.getLogger(__name__)


def build_engine_settings(*, overrides: dict[str, object]) -> EngineSettings:
    values = dict(overrides)
    if "extra_stop_sequences" in values:
        values["extra_stop_sequences"] = tuple(values["extra_stop_sequences"])
    return EngineSettings(**values)


def load_probes(*, spec: DatasetSpec | None, limit: int, synthetic_slots: int) -> list[ar.ProbeSample]:
    if spec is not None:
        samples = ar.make_probe_samples(items=load_rows(spec=spec, limit=limit))
        logger.info("loaded %d probe samples from %s", len(samples), spec.describe())
        return samples
    samples = ar.synthetic_probe_samples(
        settings=SyntheticProbeSettings(document_slots=synthetic_slots, sample_count=limit),
    )
    logger.info("built %d synthetic probe samples with %d slots", len(samples), synthetic_slots)
    return samples


class ProfileBuilder:
    def __init__(self, *, engine: ar.Engine) -> None:
        self._engine = engine

    def scan(self, *, samples: Sequence[ar.ProbeSample], settings: LayerScanSettings) -> ar.LayerScan:
        bar = progress(PROBE, total=len(samples), unit="sample")
        scan = ar.find_shallowest_attention_layer(
            engine=self._engine,
            samples=samples,
            settings=settings,
            progress=lambda completed, total: bar.update(completed - bar.n),
        )
        finish(bar)
        logger.info("layer scan:\n%s", scan.table())
        return scan

    def extract(self, *, samples: Sequence[ar.ProbeSample], settings: ProfileSettings) -> ar.AttentionProfile:
        bar = progress(PROBE, total=len(samples), unit="sample")
        profile = ar.extract_attention_profile(
            engine=self._engine,
            samples=samples,
            settings=settings,
            progress=lambda completed, total: bar.update(completed - bar.n),
        )
        finish(bar)
        return profile

    def select_or_extract(
        self,
        *,
        samples: Sequence[ar.ProbeSample],
        layer: int | None,
        scan_settings: LayerScanSettings,
        fallback_to_highest_ratio: bool,
    ) -> ar.AttentionProfile:
        if layer is not None:
            return self.extract(
                samples=samples,
                settings=ProfileSettings(layer_index=layer, normalize_by_length=scan_settings.normalize_by_length),
            )
        scan = self.scan(samples=samples, settings=scan_settings)
        selected = scan.selected_layer
        if selected < 0 and fallback_to_highest_ratio:
            selected = scan.highest_ratio_layer()
            logger.warning(
                "no layer reaches edge ratio %.2f, using layer %d with the highest ratio",
                scan_settings.min_edge_ratio,
                selected,
            )
        if selected < 0:
            raise RuntimeError("no attention-basin layer found, lower min_edge_ratio or use more probe samples")
        return scan.profile_for(layer_index=selected)


def describe_profile(*, profile: ar.AttentionProfile) -> str:
    attention = [round(value, 4) for value in profile.attention]
    return f"layer {profile.layer_index}, attention {attention}, position_order {list(profile.position_order)}"


def run_profile_task(*, settings: ProfileTaskSettings) -> Path:
    if settings.output_dir is None:
        raise ValueError("profile needs output_dir")
    engine_settings = build_engine_settings(overrides=settings.engine)
    engine = ar.from_pretrained(model=settings.model, settings=engine_settings)
    samples = load_probes(
        spec=settings.probes,
        limit=settings.probe_samples,
        synthetic_slots=settings.synthetic_slots,
    )
    builder = ProfileBuilder(engine=engine)
    profile = builder.select_or_extract(
        samples=samples,
        layer=settings.layer,
        scan_settings=LayerScanSettings(
            layer_begin=settings.layer_begin,
            layer_end=settings.layer_end,
            min_edge_ratio=settings.min_edge_ratio,
            normalize_by_length=settings.normalize_by_length,
        ),
        fallback_to_highest_ratio=False,
    )

    target = settings.output_dir / settings.profile_name
    ar.save_attention_profile(profile=profile, path=target)
    write_settings_copy(settings=settings, output_dir=settings.output_dir)
    logger.info("profile: %s", describe_profile(profile=profile))
    logger.info("profile written to %s", target)
    return target
