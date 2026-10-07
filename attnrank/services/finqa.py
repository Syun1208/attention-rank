from __future__ import annotations

import logging
import random
import shutil
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from attnrank import engine as ar
from attnrank.data.classes import FinqaExample, FinqaSettings, GenerationSettings, LayerScanSettings, ShardSpec
from attnrank.data.loaders import JsonlRecordStore, read_json, write_json
from attnrank.data.sources import load_rows
from attnrank.services.profile import ProfileBuilder, build_engine_settings, describe_profile
from attnrank.utils.chunking import chunk_context
from attnrank.utils.config import write_settings_copy
from attnrank.utils.numeric import FULL_UNIT, extract_answer, is_correct
from attnrank.utils.progress import GENERATE, finish, progress
from attnrank.utils.text import bm25_scores

logger = logging.getLogger(__name__)

METHOD = "attnrank"
FORMULA = (
    "context split into k chunks; chunks ranked by BM25 against the question; "
    "rank i placed at the slot with the i-th highest profiled attention"
)
GUIDANCE = (
    "Use only the provided documents. Show the calculation briefly, then write the final answer "
    "on the last line in the form 'Final answer: <value>'."
)
PROFILE_INSTRUCTION = "Answer the question according to the given reference time interval."
SCORER_TOLERANCE = (
    "accuracy: half a unit of the last gold decimal or 0.5% relative; "
    "accuracy_truncation: one unit of the last gold decimal or 0.5% relative"
)
QUESTION_MARKER = "Question:"
EXAMPLES_FILE_NAME = "examples.jsonl"
RUN_FILE_NAME = "run.json"
PROFILE_FILE_NAME = "profile.json"
SESSION_GLOB = "session.shard*.json"
EXAMPLES_GLOB = "examples*.jsonl"
SHARD_GLOB = "examples.shard*.jsonl"
PROGRESS_LOG_INTERVAL = 20
ERROR_PREVIEW_CHARACTERS = 200
DOCUMENT_SEPARATOR = "\n"
RAW_ADAPTER = "raw"


@dataclass(frozen=True, slots=True)
class PreparedExample:
    question: str
    documents: list[dict[str, str]]
    ranking: list[int]


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    prompt: str
    prompt_len: int
    response: str
    generated_ids: list[int]
    stop_reason: str
    seconds: float


def load_finqa_examples(*, rows: Iterable[Mapping[str, Any]], limit: int | None) -> list[FinqaExample]:
    examples = [FinqaExample.from_dict(row=dict(row)) for row in rows]
    return examples if limit is None or limit < 0 else examples[:limit]


def split_question(*, field: str) -> tuple[str, str]:
    context, _, question = field.rpartition(QUESTION_MARKER)
    return context.strip(), question.strip()


def build_instruction(*, principle: str) -> str:
    return f"{principle.strip()}\n{GUIDANCE}" if principle.strip() else GUIDANCE


def prepare_example(*, example: FinqaExample, chunks: int) -> PreparedExample:
    context, question = split_question(field=example.question)
    documents = [
        {"id": f"chunk-{index}", "title": "", "text": text}
        for index, text in enumerate(chunk_context(context=context, count=chunks))
    ]
    scores = bm25_scores(question=question, documents=documents)
    ranking = sorted(range(len(documents)), key=lambda index: -scores[index])
    return PreparedExample(question=question, documents=documents, ranking=ranking)


class ChunkPlacement:
    def __init__(self, *, strategy: str, position_order: Sequence[int], seed: int) -> None:
        self._strategy = strategy
        self._position_order = list(position_order)
        self._seed = seed

    def order(self, *, prepared: PreparedExample, position: int) -> list[int]:
        if self._strategy == "original":
            return list(range(len(prepared.documents)))
        if self._strategy == "random":
            order = list(prepared.ranking)
            random.Random(self._seed + position).shuffle(order)
            return order
        return ar.rerank_baseline(
            documents_by_relevance=prepared.ranking,
            strategy=self._strategy,
            position_order=self._position_order,
        )


class GenerationCache:
    def __init__(self, *, engine: ar.Engine, generation: GenerationSettings) -> None:
        self._engine = engine
        self._generation = generation
        self._cache: dict[tuple[str, str, tuple[int, ...]], GenerationOutcome] = {}
        self._reused: set[tuple[str, str, tuple[int, ...]]] = set()

    def generate(
        self,
        *,
        example: FinqaExample,
        question: str,
        ordered: Sequence[Mapping[str, str]],
        order: Sequence[int],
    ) -> tuple[GenerationOutcome, bool]:
        key = (example.question, example.principle, tuple(order))
        if key in self._cache:
            self._reused.add(key)
            return self._cache[key], True
        outcome = self._run(principle=example.principle, question=question, ordered=ordered)
        self._cache[key] = outcome
        return outcome, False

    def _run(self, *, principle: str, question: str, ordered: Sequence[Mapping[str, str]]) -> GenerationOutcome:
        template = self._engine.prompt_template
        template.instruction = build_instruction(principle=principle)
        self._engine.prompt_template = template
        prompt = ar.build_prompt(engine=self._engine, question=question, documents=ordered)

        started = time.perf_counter()
        result = ar.generate_from_prompt(engine=self._engine, prompt=prompt, settings=self._generation)
        elapsed = time.perf_counter() - started
        generated = list(result.generated_tokens)
        if result.hit_stop_sequence:
            stop_reason = "stop_sequence"
        elif len(generated) >= self._generation.max_new_tokens:
            stop_reason = "max_new_tokens"
        else:
            stop_reason = "eos"
        return GenerationOutcome(
            prompt=prompt.text,
            prompt_len=len(prompt.tokens),
            response=result.text.strip(),
            generated_ids=generated,
            stop_reason=stop_reason,
            seconds=elapsed,
        )


def record_stats(*, records: Sequence[Mapping[str, Any]], n_error: int) -> dict[str, Any]:
    ok = [record for record in records if record.get("error") is None]
    scored = [record for record in ok if record.get("correct") is not None]
    correct = sum(1 for record in scored if record["correct"])
    truncation = sum(1 for record in scored if record.get("correct_truncation"))
    generated = [record["n_generated"] for record in ok]
    return {
        "n_ok": len(ok),
        "n_error": n_error,
        "n_scored": len(scored),
        "n_correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "n_correct_truncation": truncation,
        "accuracy_truncation": truncation / len(scored) if scored else None,
        "n_context_overflow": sum(1 for record in ok if record["context_overflow"]),
        "n_stop_max_new_tokens": sum(1 for record in ok if record["stop_reason"] == "max_new_tokens"),
        "mean_generated_tokens": sum(generated) / len(generated) if generated else None,
    }


class FinqaRunStore:
    def __init__(self, *, run_dir: Path) -> None:
        self._run_dir = run_dir

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def shard_store(self, *, shard: ShardSpec) -> JsonlRecordStore:
        name = EXAMPLES_FILE_NAME if shard.count == 1 else f"examples.shard{shard.index}of{shard.count}.jsonl"
        return JsonlRecordStore(path=self._run_dir / name)

    def merged_store(self) -> JsonlRecordStore:
        return JsonlRecordStore(path=self._run_dir / EXAMPLES_FILE_NAME)

    def completed(self, *, store: JsonlRecordStore) -> dict[str, dict[str, Any]]:
        return {record["id"]: record for record in store.load() if record.get("error") is None}

    def reset(self) -> None:
        if self._run_dir.exists():
            shutil.rmtree(self._run_dir)

    def write_session(self, *, shard: ShardSpec, started: datetime, session: Mapping[str, Any]) -> None:
        name = f"session.shard{shard.index}of{shard.count}.{started:%H%M%S}.json"
        write_json(path=self._run_dir / name, payload=session)

    def load_sessions(self) -> list[dict[str, Any]]:
        return [read_json(path=path) for path in sorted(self._run_dir.glob(SESSION_GLOB))]

    def merge_shards(self) -> tuple[list[dict[str, Any]], int]:
        merged: dict[str, dict[str, Any]] = {}
        n_error = 0
        for path in sorted(self._run_dir.glob(EXAMPLES_GLOB)):
            for record in JsonlRecordStore(path=path).load():
                if record.get("error") is None:
                    merged[record["id"]] = record
                elif record["id"] not in merged:
                    n_error += 1
        ordered = sorted(merged.values(), key=lambda record: record["position"])
        self.merged_store().write_all(records=ordered)
        for path in list(self._run_dir.glob(SHARD_GLOB)) + list(self._run_dir.glob(SESSION_GLOB)):
            path.unlink()
        return ordered, n_error

    def write_run(self, *, payload: Mapping[str, Any]) -> None:
        write_json(path=self._run_dir / RUN_FILE_NAME, payload=payload)


class FinqaPipeline:
    def __init__(
        self,
        *,
        settings: FinqaSettings,
        profile: ar.AttentionProfile,
        placement: ChunkPlacement,
        generation: GenerationCache,
        store: FinqaRunStore,
        device: int,
    ) -> None:
        self._settings = settings
        self._profile = profile
        self._placement = placement
        self._generation = generation
        self._store = store
        self._device = device

    def run_shard(self, *, examples: Sequence[FinqaExample], shard: ShardSpec, run_id: str) -> None:
        selected = shard.select(items=examples)
        if self._settings.rerun and shard.count == 1:
            self._store.reset()
        self._store.run_dir.mkdir(parents=True, exist_ok=True)
        shard_store = self._store.shard_store(shard=shard)
        if self._settings.rerun:
            shard_store.delete()

        done = self._store.completed(store=shard_store)
        if shard.count > 1:
            done.update(self._store.completed(store=self._store.merged_store()))
        resumed = sum(1 for _, example in selected if example.id in done)
        logger.info("%s shard %d/%d: %d examples, %d resumed", run_id, shard.index, shard.count, len(selected), resumed)

        started = datetime.now()
        n_run = n_error = 0
        bar = progress(GENERATE, total=len(selected), unit="example")
        bar.update(resumed)
        for position, example in selected:
            if example.id in done:
                continue
            record = self._run_with_retries(position=position, example=example)
            if record["error"] is not None:
                n_error += 1
                logger.warning("failed %s: %s", example.id, record["error"][:ERROR_PREVIEW_CHARACTERS])
            else:
                done[example.id] = record
            shard_store.append(record=record)
            n_run += 1
            bar.update(1)
            if n_run % PROGRESS_LOG_INTERVAL == 0:
                stats = record_stats(records=[done[e.id] for _, e in selected if e.id in done], n_error=n_error)
                accuracy = stats["accuracy"]
                shown = accuracy * 100 if accuracy is not None else float("nan")
                bar.set_postfix_str(f"accuracy {shown:.1f}% errors {n_error}")
        finish(bar)

        finished = datetime.now()
        self._store.write_session(
            shard=shard,
            started=started,
            session={
                "shard": shard.index,
                "device": f"cuda:{self._device}",
                "started_at": started.isoformat(timespec="seconds"),
                "finished_at": finished.isoformat(timespec="seconds"),
                "total_time_s": (finished - started).total_seconds(),
                "n_run": n_run,
                "n_resumed": len(selected) - n_run,
            },
        )

    def _run_with_retries(self, *, position: int, example: FinqaExample) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for attempt in range(self._settings.max_retries):
            try:
                return self._run_example(position=position, example=example)
            except Exception as error:
                logger.debug("attempt %d for %s failed", attempt + 1, example.id, exc_info=True)
                record = {
                    "id": example.id,
                    "position": position,
                    "method": METHOD,
                    "error": f"{type(error).__name__}: {error}",
                    "attempt": attempt + 1,
                }
        return record

    def _run_example(self, *, position: int, example: FinqaExample) -> dict[str, Any]:
        prepared = prepare_example(example=example, chunks=self._settings.chunks)
        order = self._placement.order(prepared=prepared, position=position)
        ordered = [prepared.documents[index] for index in order]
        outcome, reused = self._generation.generate(
            example=example,
            question=prepared.question,
            ordered=ordered,
            order=order,
        )
        extracted = extract_answer(response=outcome.response)
        latency = outcome.seconds / len(outcome.generated_ids) * 1000 if outcome.generated_ids else None
        return {
            "id": example.id,
            "position": position,
            "method": METHOD,
            "strategy": self._settings.strategy,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "question": example.question,
            "query": prepared.question,
            "principle": example.principle,
            "ground_truth": example.gold_response,
            "rejected": example.rejected,
            "meta": example.meta,
            "n_chunks": len(prepared.documents),
            "relevance_ranking": prepared.ranking,
            "chunk_order": order,
            "profile_layer": self._profile.layer_index,
            "position_order": list(self._profile.position_order),
            "prompt": outcome.prompt,
            "prompt_len": outcome.prompt_len,
            "context_overflow": outcome.prompt_len + self._settings.max_new_tokens > self._max_sequence(),
            "response": outcome.response,
            "generated_ids": outcome.generated_ids,
            "n_generated": len(outcome.generated_ids),
            "stop_reason": outcome.stop_reason,
            "extracted": extracted,
            "correct": is_correct(extracted=extracted, gold=example.gold_response),
            "correct_truncation": is_correct(extracted=extracted, gold=example.gold_response, units=FULL_UNIT),
            "reused_generation": reused,
            "t_total_s": outcome.seconds,
            "latency_per_token_ms": latency,
            "error": None,
        }

    def _max_sequence(self) -> int:
        return int(self._settings.engine.get("max_sequence", ar.EngineSettings().max_sequence))


class FinqaFinalizer:
    def __init__(self, *, settings: FinqaSettings, store: FinqaRunStore, profile_path: Path) -> None:
        self._settings = settings
        self._store = store
        self._profile_path = profile_path

    def finalize(self, *, examples: Sequence[FinqaExample], run_id: str) -> dict[str, Any]:
        sessions = self._store.load_sessions()
        ordered, n_error = self._store.merge_shards()
        for record in ordered:
            record["correct"] = is_correct(extracted=record["extracted"], gold=record["ground_truth"])
            record["correct_truncation"] = is_correct(
                extracted=record["extracted"],
                gold=record["ground_truth"],
                units=FULL_UNIT,
            )
        self._store.merged_store().write_all(records=ordered)

        profile = read_json(path=self._profile_path)
        distinct: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
        for record in ordered:
            distinct.setdefault((record["question"], record["principle"], tuple(record["chunk_order"])), record)
        fresh = list(distinct.values())
        unique_stats = record_stats(records=fresh, n_error=0)
        tokens = sum(record["n_generated"] for record in fresh)
        decode = sum(record["t_total_s"] for record in fresh)
        started = min((session["started_at"] for session in sessions), default=None)
        finished = max((session["finished_at"] for session in sessions), default=None)
        wall = (
            (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
            if started and finished
            else None
        )
        run_info = {
            "run_id": run_id,
            "method": METHOD,
            "formula": FORMULA,
            "model": self._settings.model,
            "dtype": "fp16 weights, fp32 activations",
            "device": sorted({session["device"] for session in sessions}),
            "engine": "AttnRank C++/CUDA",
            "prompt_style": self._settings.engine.get("chat_format", ""),
            "strategy": self._settings.strategy,
            "chunks": self._settings.chunks,
            "relevance": "bm25",
            "profile": str(self._profile_path),
            "profile_layer": profile["layer_index"],
            "profile_attention": profile["attention"],
            "position_order": profile["position_order"],
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_new_tokens": self._settings.max_new_tokens,
            "max_sequence": self._settings.engine.get("max_sequence"),
            "max_retries": self._settings.max_retries,
            "scorer": "numeric",
            "scorer_tolerance": SCORER_TOLERANCE,
            "n_examples": len(examples),
            "n_unique_prompts": len(fresh),
            "n_shards": len({session["shard"] for session in sessions}),
            "n_sessions": len(sessions),
            "n_run_this_session": sum(session["n_run"] for session in sessions),
            "n_resumed": sum(session["n_resumed"] for session in sessions),
            "rerun": self._settings.rerun,
            "trace_steps": "none",
            "started_at": started,
            "finished_at": finished,
            "total_time_s": wall,
            "total_generated_tokens": tokens,
            "throughput_tok_per_s": tokens / decode if decode else None,
            "examples_per_min": len(ordered) / wall * 60 if wall else None,
            **record_stats(records=ordered, n_error=n_error),
            "accuracy_unique_prompts": unique_stats["accuracy"],
            "accuracy_truncation_unique_prompts": unique_stats["accuracy_truncation"],
        }
        self._store.write_run(payload=run_info)
        return run_info


def build_finqa_profile(
    *,
    engine: ar.Engine,
    examples: Sequence[FinqaExample],
    settings: FinqaSettings,
    path: Path,
) -> ar.AttentionProfile:
    seen: set[tuple[str, str]] = set()
    samples = []
    shuffler = random.Random(settings.seed)
    for example in examples:
        key = (example.question, example.principle)
        if key in seen:
            continue
        seen.add(key)
        prepared = prepare_example(example=example, chunks=settings.chunks)
        documents = list(prepared.documents)
        shuffler.shuffle(documents)
        samples.append({"question": prepared.question, "documents": documents})
        if len(samples) >= settings.probe_samples:
            break

    profile = ProfileBuilder(engine=engine).select_or_extract(
        samples=ar.make_probe_samples(items=samples),
        layer=None,
        scan_settings=LayerScanSettings(min_edge_ratio=settings.min_edge_ratio),
        fallback_to_highest_ratio=True,
    )
    ar.save_attention_profile(profile=profile, path=path)
    logger.info("profile: %s", describe_profile(profile=profile))
    return profile


def dataset_label(*, settings: FinqaSettings) -> str:
    if settings.dataset.path is not None:
        return settings.dataset.path.parent.name
    return str(settings.dataset.hub).split("/")[-1]


def run_identifier(*, settings: FinqaSettings, dataset_name: str) -> str:
    model_name = Path(str(settings.model)).name
    method = METHOD if settings.strategy == "attnrank" else settings.strategy
    return f"{dataset_name}_{method}_{model_name}_k{settings.chunks}_t{settings.temperature}"


def run_finqa_task(*, settings: FinqaSettings) -> dict[str, Any] | None:
    raw_rows = load_rows(spec=replace(settings.dataset, adapter=RAW_ADAPTER))
    examples = load_finqa_examples(rows=raw_rows, limit=settings.examples)
    dataset_name = settings.dataset_name or dataset_label(settings=settings)
    run_id = run_identifier(settings=settings, dataset_name=dataset_name)
    if settings.output_dir is None:
        raise ValueError("finqa needs output_dir")
    run_dir = settings.output_dir / run_id
    profile_path = settings.profile or run_dir / PROFILE_FILE_NAME
    store = FinqaRunStore(run_dir=run_dir)
    finalizer = FinqaFinalizer(settings=settings, store=store, profile_path=profile_path)

    if settings.finalize:
        info = finalizer.finalize(examples=examples, run_id=run_id)
        logger.info("finalized %s: accuracy %s", run_id, info["accuracy"])
        return info

    run_dir.mkdir(parents=True, exist_ok=True)
    write_settings_copy(settings=settings, output_dir=run_dir)
    engine_settings = build_engine_settings(
        overrides={"instruction": build_instruction(principle=PROFILE_INSTRUCTION), **settings.engine},
    )
    engine_settings = replace(engine_settings, include_titles=False, document_separator=DOCUMENT_SEPARATOR)
    engine = ar.from_pretrained(model=settings.model, settings=engine_settings)

    if settings.profile_only or not profile_path.exists():
        profile = build_finqa_profile(engine=engine, examples=examples, settings=settings, path=profile_path)
        if settings.profile_only:
            return None
    else:
        profile = ar.load_attention_profile(path=profile_path)
        logger.info("loaded profile %s: %s", profile_path, describe_profile(profile=profile))

    shard = ShardSpec.parse(text=settings.shard)
    pipeline = FinqaPipeline(
        settings=settings,
        profile=profile,
        placement=ChunkPlacement(
            strategy=settings.strategy,
            position_order=profile.position_order,
            seed=settings.seed,
        ),
        generation=GenerationCache(
            engine=engine,
            generation=GenerationSettings(
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                seed=settings.seed,
            ),
        ),
        store=store,
        device=engine_settings.device,
    )
    pipeline.run_shard(examples=examples, shard=shard, run_id=run_id)
    if shard.count == 1:
        return finalizer.finalize(examples=examples, run_id=run_id)
    return None
