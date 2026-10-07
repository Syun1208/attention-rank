from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from attnrank import engine as ar
from attnrank.data.classes import EngineSettings, GenerationSettings
from attnrank.data.loaders import iter_jsonl, read_json

logger = logging.getLogger(__name__)


def rerank_top_k(
    *,
    documents_by_relevance: Sequence[Any],
    profile: ar.AttentionProfile | Path,
) -> list[Any]:
    loaded = profile if isinstance(profile, ar.AttentionProfile) else ar.load_attention_profile(path=profile)
    return ar.rerank_with_profile(documents_by_relevance=documents_by_relevance, profile=loaded)


def answer_with_top_k(
    *,
    engine: ar.Engine,
    question: str,
    documents_by_relevance: Sequence[Any],
    profile: ar.AttentionProfile | Path,
    generation: GenerationSettings = GenerationSettings(),
) -> tuple[list[Any], str]:
    ordered = rerank_top_k(documents_by_relevance=documents_by_relevance, profile=profile)
    result = ar.generate_answer(engine=engine, question=question, documents=ordered, settings=generation)
    return ordered, result.text.strip()


def load_documents_file(*, path: Path) -> list[Mapping[str, Any]]:
    if path.suffix == ".jsonl":
        return [row for row in iter_jsonl(path=path)]
    payload = read_json(path=path)
    if isinstance(payload, Mapping) and "documents" in payload:
        payload = payload["documents"]
    if not isinstance(payload, list):
        raise ValueError(f"{path} must hold a JSON list of documents or an object with a 'documents' list")
    return [{"text": item} if isinstance(item, str) else item for item in payload]


def run_rerank_task(
    *,
    documents: Path,
    profile: Path,
    question: str | None,
    model: str | None,
    engine_overrides: Mapping[str, Any],
    generation: GenerationSettings,
) -> dict[str, Any]:
    top_k = load_documents_file(path=documents)
    loaded = ar.load_attention_profile(path=profile)
    ordered = rerank_top_k(documents_by_relevance=top_k, profile=loaded)
    slots = ar.attention_slot_order(position_order=loaded.position_order, count=len(top_k))
    result: dict[str, Any] = {
        "profile_layer": loaded.layer_index,
        "position_order": list(loaded.position_order),
        "slot_of_rank": slots,
        "documents": ordered,
    }
    logger.info("rank -> slot: %s", {rank: slot for rank, slot in enumerate(slots)})

    if question is not None and model is not None:
        engine = ar.from_pretrained(model=model, settings=EngineSettings(**dict(engine_overrides)))
        answer = ar.generate_answer(engine=engine, question=question, documents=ordered, settings=generation)
        result["question"] = question
        result["answer"] = answer.text.strip()
        logger.info("answer: %s", result["answer"])
    return result
