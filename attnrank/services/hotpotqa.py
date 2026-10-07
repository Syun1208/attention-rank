from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from attnrank import engine as ar
from attnrank.data.classes import (
    STRATEGIES,
    EvaluationRecord,
    GenerationSettings,
    HotpotqaSettings,
    LayerScanSettings,
)
from attnrank.data.loaders import JsonlRecordStore
from attnrank.data.sources import load_rows
from attnrank.interface.ordering import AnswerScorer, RelevanceOrdering
from attnrank.services.profile import ProfileBuilder, build_engine_settings, describe_profile, load_probes
from attnrank.utils.config import write_settings_copy
from attnrank.utils.progress import GENERATE, finish, progress
from attnrank.utils.text import bm25_scores, substring_match

logger = logging.getLogger(__name__)

PAPER_TABLE1 = {
    "qwen2.5-7b": {"random": 52.32, "descending": 53.31, "ascending": 54.64, "lim": 52.18, "attnrank": 54.55},
    "qwen2.5-3b": {"random": 45.77, "descending": 46.42, "ascending": 47.45, "lim": 45.62, "attnrank": 47.54},
    "qwen2.5-1.5b": {"random": 40.62, "descending": 41.23, "ascending": 43.56, "lim": 39.48, "attnrank": 44.14},
}
ANSWER_STOP_SEQUENCES = ("\n\n",)
PROGRESS_LOG_INTERVAL = 25
RECORDS_FILE_NAME = "records.jsonl"
PROFILE_FILE_NAME = "profile.json"
GOLD_FLAG = "is_gold"
ORDERINGS = ("bm25", "gold-first", "given")


def is_gold(*, document: Mapping[str, Any]) -> bool:
    return str(document.get(GOLD_FLAG, "")).lower() == "true"


class GivenOrdering:
    def order(self, *, question: str, documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return list(documents)


class GoldFirstOrdering:
    def order(self, *, question: str, documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        gold = [document for document in documents if is_gold(document=document)]
        rest = [document for document in documents if not is_gold(document=document)]
        return gold + rest


class Bm25Ordering:
    def order(self, *, question: str, documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        scores = bm25_scores(question=question, documents=documents)
        ranking = sorted(range(len(documents)), key=lambda index: -scores[index])
        return [documents[index] for index in ranking]


ORDERING_STRATEGIES: dict[str, type[RelevanceOrdering]] = {
    "given": GivenOrdering,
    "gold-first": GoldFirstOrdering,
    "bm25": Bm25Ordering,
}


def build_ordering(*, name: str) -> RelevanceOrdering:
    if name not in ORDERING_STRATEGIES:
        raise ValueError(f"unknown ordering '{name}', expected one of {ORDERINGS}")
    return ORDERING_STRATEGIES[name]()


def relevance_order(
    *,
    question: str,
    documents: Sequence[Mapping[str, Any]],
    ordering: str,
) -> list[Mapping[str, Any]]:
    return build_ordering(name=ordering).order(question=question, documents=documents)


class SubstringScorer:
    def score(self, *, prediction: str, reference: str) -> bool:
        return substring_match(prediction=prediction, gold=reference)


class StrategyPlacement:
    def __init__(self, *, position_order: Sequence[int], seed: int) -> None:
        self._position_order = list(position_order)
        self._seed = seed

    def place(self, *, ranked: Sequence[Any], strategy: str, question_index: int) -> list[Any]:
        if strategy == "random":
            ordered = list(ranked)
            random.Random(self._seed + question_index).shuffle(ordered)
            return ordered
        return ar.rerank_baseline(
            documents_by_relevance=ranked,
            strategy=strategy,
            position_order=self._position_order,
        )


class HotpotqaEvaluation:
    def __init__(
        self,
        *,
        engine: ar.Engine,
        ordering: RelevanceOrdering,
        placement: StrategyPlacement,
        scorer: AnswerScorer,
        store: JsonlRecordStore,
        strategies: Sequence[str],
        generation: GenerationSettings,
    ) -> None:
        self._engine = engine
        self._ordering = ordering
        self._placement = placement
        self._scorer = scorer
        self._store = store
        self._strategies = list(strategies)
        self._generation = generation

    def run(self, *, rows: Sequence[Mapping[str, Any]], offset: int) -> dict[str, float]:
        hits = {strategy: 0 for strategy in self._strategies}
        bar = progress(GENERATE, total=len(rows) * len(self._strategies), unit="answer")
        for index, row in enumerate(rows):
            ranked = self._ordering.order(question=row["question"], documents=row["documents"])
            for strategy in self._strategies:
                record = self._evaluate(
                    row=row,
                    ranked=ranked,
                    strategy=strategy,
                    question_index=offset + index,
                )
                hits[strategy] += int(record.hit)
                self._store.append(record=record.to_dict())
                bar.update(1)
            if (index + 1) % PROGRESS_LOG_INTERVAL == 0:
                done = index + 1
                bar.set_postfix_str(" ".join(f"{name}={hits[name] / done * 100:.1f}" for name in self._strategies))
        finish(bar)
        return {strategy: hits[strategy] / max(1, len(rows)) * 100 for strategy in self._strategies}

    def _evaluate(
        self,
        *,
        row: Mapping[str, Any],
        ranked: Sequence[Mapping[str, Any]],
        strategy: str,
        question_index: int,
    ) -> EvaluationRecord:
        ordered = self._placement.place(ranked=ranked, strategy=strategy, question_index=question_index)
        gold_slots = tuple(slot for slot, document in enumerate(ordered) if is_gold(document=document))
        result = ar.generate_answer(
            engine=self._engine,
            question=row["question"],
            documents=ordered,
            settings=self._generation,
        )
        prediction = result.text.strip()
        return EvaluationRecord(
            id=str(row.get("id", question_index)),
            strategy=strategy,
            question=row["question"],
            reference=row["answer"],
            prediction=prediction,
            hit=self._scorer.score(prediction=prediction, reference=row["answer"]),
            gold_slots=gold_slots,
        )


class HotpotqaReport:
    def __init__(self, *, paper_row: str | None) -> None:
        self._paper = PAPER_TABLE1.get(paper_row or "", {})

    def render(self, *, records: Sequence[Mapping[str, Any]]) -> str:
        hits: dict[str, int] = {}
        counts: dict[str, int] = {}
        slots: dict[str, Counter] = {}
        seen: set[tuple[str, str]] = set()
        for record in records:
            key = (str(record["id"]), record["strategy"])
            if key in seen:
                continue
            seen.add(key)
            strategy = record["strategy"]
            counts[strategy] = counts.get(strategy, 0) + 1
            hits[strategy] = hits.get(strategy, 0) + int(record["hit"])
            slots.setdefault(strategy, Counter()).update(record["gold_slots"])

        header = (
            f"{'strategy':>10} {'n':>5} {'measured':>9} {'paper':>7} {'delta':>7} {'vs random':>10} "
            f"{'paper vs rnd':>12}  gold slots"
        )
        lines = [header, "-" * len(header)]
        base = hits.get("random", 0) / counts["random"] * 100 if counts.get("random") else float("nan")
        paper_base = self._paper.get("random", float("nan"))
        for strategy in STRATEGIES:
            if strategy not in counts:
                continue
            measured = hits[strategy] / counts[strategy] * 100
            paper = self._paper.get(strategy, float("nan"))
            histogram = [slots[strategy].get(index, 0) for index in range(max(slots[strategy]) + 1)]
            lines.append(
                f"{strategy:>10} {counts[strategy]:>5} {measured:>9.2f} {paper:>7.2f} {measured - paper:>+7.2f} "
                f"{measured - base:>+10.2f} {paper - paper_base:>+12.2f}  {histogram}"
            )
        return "\n".join(lines)


def resolve_profile(
    *,
    engine: ar.Engine,
    settings: HotpotqaSettings,
) -> ar.AttentionProfile:
    if settings.profile is not None:
        profile = ar.load_attention_profile(path=settings.profile)
        logger.info("loaded profile %s: %s", settings.profile, describe_profile(profile=profile))
        return profile
    if settings.probes is None:
        raise ValueError("hotpotqa needs either 'profile' or 'probes'")
    samples = load_probes(spec=settings.probes, limit=settings.probe_samples, synthetic_slots=0)
    profile = ProfileBuilder(engine=engine).select_or_extract(
        samples=samples,
        layer=settings.layer,
        scan_settings=LayerScanSettings(min_edge_ratio=settings.min_edge_ratio),
        fallback_to_highest_ratio=False,
    )
    ar.save_attention_profile(profile=profile, path=Path(settings.output_dir or ".") / PROFILE_FILE_NAME)
    logger.info("profile: %s", describe_profile(profile=profile))
    return profile


def run_hotpotqa_task(*, settings: HotpotqaSettings) -> dict[str, float]:
    if settings.output_dir is None:
        raise ValueError("hotpotqa needs output_dir")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_settings_copy(settings=settings, output_dir=settings.output_dir)
    engine = ar.from_pretrained(model=settings.model, settings=build_engine_settings(overrides=settings.engine))
    profile = resolve_profile(engine=engine, settings=settings)
    rows = load_rows(spec=settings.dataset, offset=settings.offset, limit=settings.questions)
    logger.info("evaluating %d questions with %s", len(rows), list(settings.strategies))

    evaluation = HotpotqaEvaluation(
        engine=engine,
        ordering=build_ordering(name=settings.ordering),
        placement=StrategyPlacement(position_order=profile.position_order, seed=settings.seed),
        scorer=SubstringScorer(),
        store=JsonlRecordStore(path=settings.output_dir / RECORDS_FILE_NAME),
        strategies=settings.strategies,
        generation=GenerationSettings(
            max_new_tokens=settings.max_new_tokens,
            extra_stop_sequences=ANSWER_STOP_SEQUENCES,
        ),
    )
    accuracy = evaluation.run(rows=rows, offset=settings.offset)
    for strategy, value in accuracy.items():
        logger.info("%10s accuracy %.2f", strategy, value)
    return accuracy


def run_hotpotqa_report(*, records: Sequence[Path], paper_row: str | None) -> str:
    rows: list[Mapping[str, Any]] = []
    for path in records:
        rows.extend(JsonlRecordStore(path=path).load())
    return HotpotqaReport(paper_row=paper_row).render(records=rows)
