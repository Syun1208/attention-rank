import argparse
import asyncio
import json
import logging
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router
from tqdm.auto import tqdm
from attnrank.utils.workspace import workspace_path

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parents[3]
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "FinQA_ALIGNMENT_JUDGE.md"
LABELS_PATH = PROJECT_DIR / "data" / "finqa" / "dataset.jsonl"

HALLUCINATION_TYPES = {
    "none",
    "fabricated_number",
    "fabricated_fact",
    "invalid_step",
    "fabricated_reference",
    "misattributed_value",
}

DEFECT_TYPES = {
    "degenerate_text",
    "meta_commentary",
    "unjustified_refusal",
    "factual_error",
}

PRINCIPLE_WEIGHT = 0.6
FOLLOWS_THRESHOLD = 7
IGNORED_PRINCIPLE_CAP = 3
UNHELPFUL_THRESHOLD = 3
UNHELPFUL_CAP = 5

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeConfig:
    model_name: str = "azure/gpt-4o"
    temperature: float = 0.0
    concurrency: int = 8
    max_retries: int = 3
    max_question_chars: int = 12000
    max_response_chars: int = 6000


@dataclass
class JudgeStats:
    total: int = 0
    parsed: int = 0
    failed: int = 0
    scores: list[int] = field(default_factory=list)
    hallucinations: int = 0
    invalid_derivations: int = 0
    gold_suspect: int = 0
    principle_scores: list[int] = field(default_factory=list)
    helpfulness_scores: list[int] = field(default_factory=list)
    follows_principle: int = 0
    truncated: int = 0
    defective: int = 0
    defects: dict[str, int] = field(default_factory=dict)


def get_model(model_name: str = "azure/gpt-4o", temperature: float = 0.0) -> ChatLiteLLMRouter:
    config = [
        {
            "model_name": "azure/gpt-4o",
            "litellm_params": {
                "model": "azure/gpt-4o",
                "base_model": "gpt-4o",
                "api_key": os.environ.get("AZURE_API_KEY"),
                "api_version": os.environ.get("AZURE_API_VERSION"),
                "api_base": os.environ.get("AZURE_API_BASE"),
                "timeout": 60 * 10,
                "stream": False,
            },
        }
    ]
    router = Router(model_list=config)
    return ChatLiteLLMRouter(router=router, model_name=model_name, temperature=temperature)


def load_prompt_sections(path: Path = PROMPT_PATH) -> tuple[str, str]:
    text = path.read_text()
    system = text.split("## System", 1)[1].split("## User", 1)[0].strip()
    user = text.split("## User", 1)[1].strip().strip("`").strip()
    return system, user


def load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open() as handle:
        return {row["id"]: row["gold_response"] for row in map(json.loads, handle)}


def load_records(run_dir: Path, labels: dict[str, str]) -> list[dict]:
    records = []
    relabelled = 0
    with (run_dir / "examples.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("error"):
                continue
            gold = labels.get(record["id"])
            if gold is not None and str(record["ground_truth"]) != gold:
                record["ground_truth"] = gold
                relabelled += 1
            records.append(record)
    if labels:
        print(f"   🏷️  {relabelled} gold labels replaced from the labels file")
    return records


def sample_records(records: list[dict], n: int | None, seed: int, unique_questions: bool) -> list[dict]:
    pool = records
    if unique_questions:
        seen: dict[str, dict] = {}
        for record in sorted(records, key=lambda r: r["id"]):
            seen.setdefault(record["question"], record)
        pool = list(seen.values())
    pool = sorted(pool, key=lambda r: r["id"])
    if n is None or n >= len(pool):
        return pool
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n], key=lambda r: r["id"])


def done_ids(output_path: Path, golds: dict[str, str]) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    kept = []
    stale = 0
    with output_path.open() as handle:
        for line in handle:
            try:
                verdict = json.loads(line)
                verdict_id = verdict["id"]
            except (json.JSONDecodeError, KeyError):
                kept.append(line)
                continue
            if verdict_id in golds and str(verdict.get("gold")) != golds[verdict_id]:
                stale += 1
                continue
            ids.add(verdict_id)
            kept.append(line)
    if stale:
        output_path.write_text("".join(kept))
        print(f"   🧹 dropped {stale} verdicts judged against an outdated gold")
    return ids


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[... truncated ...]\n" + text[-half:]


def build_messages(system: str, user_template: str, record: dict, config: JudgeConfig) -> list[tuple[str, str]]:
    user = (
        user_template.replace("{question}", truncate(record["question"], config.max_question_chars))
        .replace("{principle}", record.get("principle") or "(none)")
        .replace("{gold}", str(record["ground_truth"]))
        .replace("{response}", truncate(record.get("response") or "(empty)", config.max_response_chars))
    )
    return [("system", system), ("human", user)]


def clamp_score(value: object, low: int) -> int:
    return max(low, min(10, int(value)))


def combine_psoups_score(principle: int, helpfulness: int) -> int:
    score = int(PRINCIPLE_WEIGHT * principle + (1 - PRINCIPLE_WEIGHT) * helpfulness + 0.5)
    if principle <= IGNORED_PRINCIPLE_CAP:
        score = min(score, IGNORED_PRINCIPLE_CAP)
    if helpfulness <= UNHELPFUL_THRESHOLD:
        score = min(score, UNHELPFUL_CAP)
    return score


def parse_psoups_verdict(verdict: dict) -> dict:
    raw_defects = verdict.get("defects") or []
    defects = [str(name) for name in (raw_defects if isinstance(raw_defects, list) else [raw_defects])]
    principle = clamp_score(verdict["principle_score"], 1)
    helpfulness = clamp_score(verdict["helpfulness_score"], 1)
    return {
        "score": combine_psoups_score(principle, helpfulness),
        "principle_score": principle,
        "helpfulness_score": helpfulness,
        "follows_principle": principle >= FOLLOWS_THRESHOLD,
        "truncated": bool(verdict.get("truncated", False)),
        "defects": list(dict.fromkeys(name for name in defects if name in DEFECT_TYPES)),
        "reason": str(verdict.get("reason", ""))[:400],
    }


def parse_verdict(text: str) -> dict:
    match = JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object in judge output")
    verdict = json.loads(match.group())
    if "principle_score" in verdict:
        return parse_psoups_verdict(verdict)
    score = int(verdict.get("score", 0))
    hallucination_type = str(verdict.get("hallucination_type", "none"))
    return {
        "final_value": verdict.get("final_value"),
        "derivation_valid": bool(verdict.get("derivation_valid", False)),
        "score": max(0, min(10, score)),
        "hallucination": bool(verdict.get("hallucination", False)),
        "hallucination_type": hallucination_type if hallucination_type in HALLUCINATION_TYPES else "none",
        "gold_suspect": bool(verdict.get("gold_suspect", False)),
        "reason": str(verdict.get("reason", ""))[:400],
    }


async def judge_one(
    model: ChatLiteLLMRouter,
    system: str,
    user_template: str,
    record: dict,
    config: JudgeConfig,
    semaphore: asyncio.Semaphore,
) -> dict:
    messages = build_messages(system, user_template, record, config)
    async with semaphore:
        for attempt in range(config.max_retries):
            try:
                reply = await model.ainvoke(messages)
                verdict = parse_verdict(reply.content)
                return {"id": record["id"], "gold": record["ground_truth"], **verdict}
            except Exception as error:
                if attempt == config.max_retries - 1:
                    return {
                        "id": record["id"],
                        "gold": record["ground_truth"],
                        "error": f"{type(error).__name__}: {error}"[:300],
                    }
                await asyncio.sleep(min(60.0, 2.0 * 2**attempt) + random.uniform(0.0, 1.0))
    return {"id": record["id"], "error": "unreachable"}


async def rerun_errors(
    rerun_file: Path,
    run_dir: Path,
    config: JudgeConfig,
    labels: dict[str, str],
) -> JudgeStats:
    print(f"\n\033[1m🔁 rerun {rerun_file.name}\033[0m   📂 examples from {run_dir}")
    lines = rerun_file.read_text().splitlines()
    verdicts: list[dict | None] = []
    for line in lines:
        try:
            verdicts.append(json.loads(line))
        except json.JSONDecodeError:
            verdicts.append(None)
    judged = {v["id"] for v in verdicts if v and "id" in v and "error" not in v}
    records = {record["id"]: record for record in load_records(run_dir, labels)}

    drop: set[int] = set()
    pending: dict[int, dict] = {}
    missing = 0
    for index, verdict in enumerate(verdicts):
        if not verdict or "error" not in verdict:
            continue
        if verdict.get("id") in judged or verdict.get("id") in {r["id"] for r in pending.values()}:
            drop.add(index)
        elif verdict.get("id") in records:
            pending[index] = records[verdict["id"]]
        else:
            missing += 1
    print(f"   ❌ errored {len(pending) + len(drop) + missing}   🎯 to rerun {len(pending)}   "
          f"🧹 already judged elsewhere {len(drop)}   ❓ not in examples {missing}")
    if not pending and not drop:
        print("   ✨ nothing to rerun")
        return summarise(rerun_file)

    fixed = broken = 0
    try:
        if pending:
            print(f"   🤖 {config.model_name}   🧵 concurrency {config.concurrency}   🔄 retries {config.max_retries}")
            system, user_template = load_prompt_sections()
            model = get_model(config.model_name, config.temperature)
            semaphore = asyncio.Semaphore(config.concurrency)

            async def judge_at(index: int, record: dict) -> tuple[int, dict]:
                return index, await judge_one(model, system, user_template, record, config, semaphore)

            tasks = [asyncio.create_task(judge_at(index, record)) for index, record in pending.items()]
            bar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="   🔁 rerunning", unit="ex", ncols=100, colour="yellow")
            for task in bar:
                index, verdict = await task
                verdicts[index] = verdict
                if "error" in verdict:
                    broken += 1
                else:
                    fixed += 1
                bar.set_postfix_str(f"✅ {fixed}  ❌ {broken}", refresh=False)
            bar.close()
    finally:
        kept = [
            json.dumps(verdict, ensure_ascii=False) if verdict is not None else lines[index]
            for index, verdict in enumerate(verdicts)
            if index not in drop
        ]
        tmp_path = rerun_file.with_suffix(rerun_file.suffix + ".tmp")
        tmp_path.write_text("\n".join(kept) + "\n")
        os.replace(tmp_path, rerun_file)
        print(f"   💾 updated in place: ✅ fixed {fixed}   ❌ still failing {broken}   🧹 dropped {len(drop)}")
    return summarise(rerun_file)


async def run_judge(
    run_dir: Path,
    output_path: Path,
    config: JudgeConfig,
    n: int | None,
    seed: int,
    unique_questions: bool,
    resume: bool,
    labels: dict[str, str],
) -> JudgeStats:
    system, user_template = load_prompt_sections()
    records = sample_records(load_records(run_dir, labels), n, seed, unique_questions)
    skip = done_ids(output_path, {r["id"]: str(r["ground_truth"]) for r in records}) if resume else set()
    pending = [record for record in records if record["id"] not in skip]
    print(f"\n\033[1m⚖️  {run_dir.name}\033[0m")
    print(f"   📂 selected {len(records)}   ✅ already judged {len(skip)}   🎯 to judge {len(pending)}")
    if not pending:
        print("   ✨ nothing to do, everything is already judged")
        return summarise(output_path)
    print(f"   🤖 {config.model_name}   🧵 concurrency {config.concurrency}   🌡️  temp {config.temperature}")
    model = get_model(config.model_name, config.temperature)
    semaphore = asyncio.Semaphore(config.concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    running: list[int] = []
    worst = 0
    broken = 0
    with output_path.open("a") as handle:
        tasks = [
            asyncio.create_task(judge_one(model, system, user_template, record, config, semaphore))
            for record in pending
        ]
        bar = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="   🧑‍⚖️ judging",
            unit="ex",
            ncols=100,
            colour="green",
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            leave=True,
        )
        for task in bar:
            verdict = await task
            handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
            handle.flush()
            if "error" in verdict:
                broken += 1
            else:
                running.append(verdict["score"])
                worst += int(verdict["score"] <= (3 if "principle_score" in verdict else 0))
            mean = sum(running) / len(running) if running else 0.0
            bar.set_postfix_str(f"⭐ {mean:.2f}  💀 {worst}  ❌ {broken}", refresh=False)
        bar.close()
    return summarise(output_path)


def summarise(output_path: Path) -> JudgeStats:
    stats = JudgeStats()
    if not output_path.exists():
        return stats
    with output_path.open() as handle:
        for line in handle:
            try:
                verdict = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats.total += 1
            if "error" in verdict:
                stats.failed += 1
                continue
            stats.parsed += 1
            stats.scores.append(verdict["score"])
            if "principle_score" in verdict:
                stats.principle_scores.append(verdict["principle_score"])
                stats.helpfulness_scores.append(verdict["helpfulness_score"])
                stats.follows_principle += int(verdict["follows_principle"])
                stats.truncated += int(verdict["truncated"])
                stats.defective += int(bool(verdict["defects"]))
                for name in verdict["defects"]:
                    stats.defects[name] = stats.defects.get(name, 0) + 1
                continue
            stats.hallucinations += int(verdict["hallucination"])
            stats.invalid_derivations += int(not verdict["derivation_valid"])
            stats.gold_suspect += int(verdict["gold_suspect"])
    return stats


def mean_of(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def histogram(scores: list[int], width: int = 24) -> list[str]:
    counts = [sum(1 for s in scores if s == value) for value in range(11)]
    peak = max(counts) or 1
    lines = []
    for value, count in enumerate(counts):
        bar = "█" * round(width * count / peak)
        share = count / len(scores) if scores else 0.0
        lines.append(f"      {value:2d} │{bar:<{width}}│ {count:5d}  {share:5.1%}")
    return lines


def print_stats(stats: JudgeStats, label: str) -> None:
    print(f"\n\033[1m📊 {label}\033[0m")
    print(f"   ✅ judged {stats.parsed}   ❌ failed {stats.failed}")
    if not stats.scores:
        print("   🫥 nothing scored")
        return
    scores = sorted(stats.scores)
    mean = sum(scores) / len(scores)
    good = sum(1 for s in scores if s >= 8)
    zero = sum(1 for s in scores if s == 0)
    print(f"   ⭐ mean {mean:.2f}/10      📈 median {scores[len(scores) // 2]}")
    print(f"   🏆 score ≥ 8      {good:5d}  {good / len(scores):5.1%}")
    if stats.principle_scores:
        low = sum(1 for s in scores if s <= 3)
        print(f"   💀 score ≤ 3      {low:5d}  {low / len(scores):5.1%}")
        print(f"   🎭 principle      {mean_of(stats.principle_scores):5.2f}/10")
        print(f"   🤝 helpfulness    {mean_of(stats.helpfulness_scores):5.2f}/10")
        print(f"   ✅ follows        {stats.follows_principle:5d}  {stats.follows_principle / stats.parsed:5.1%}")
        print(f"   ✂️  truncated      {stats.truncated:5d}  {stats.truncated / stats.parsed:5.1%}")
        print(f"   🧨 any defect     {stats.defective:5d}  {stats.defective / stats.parsed:5.1%}")
        for name, count in sorted(stats.defects.items(), key=lambda item: -item[1]):
            print(f"      · {name:<20}{count:5d}  {count / stats.parsed:5.1%}")
    else:
        print(f"   💀 score = 0      {zero:5d}  {zero / len(scores):5.1%}")
        print(f"   🧮 bad derivation {stats.invalid_derivations:5d}  {stats.invalid_derivations / stats.parsed:5.1%}")
        print(f"   🌀 hallucination  {stats.hallucinations:5d}  {stats.hallucinations / stats.parsed:5.1%}")
        print(f"   🚩 gold suspect   {stats.gold_suspect:5d}  {stats.gold_suspect / stats.parsed:5.1%}")
    print("   📉 score histogram")
    for line in histogram(scores):
        print(line)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge scoring for FinQA runs")
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=workspace_path(relative="outputs/judge"))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", type=str, default="azure/gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--unique-questions", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="jsonl with id/gold_response overriding ground_truth")
    parser.add_argument("--no-labels", action="store_true", help="use ground_truth from examples.jsonl as is")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rerun", action="store_true", help="re-judge only the errored verdicts and update the judge file in place")
    parser.add_argument(
        "--rerun-file",
        type=Path,
        action="append",
        default=None,
        help="judge jsonl to rerun (implies --rerun); its run dir defaults to <output-dir>/../<file stem>, or pass --run-dir in the same order",
    )
    args = parser.parse_args(argv)
    if args.rerun_file:
        args.rerun = True
    if not args.run_dir and not args.rerun_file:
        parser.error("--run-dir is required unless --rerun-file is given")
    if args.rerun_file and args.run_dir and len(args.run_dir) != len(args.rerun_file):
        parser.error("when both are given, pass one --run-dir per --rerun-file")
    return args


def print_comparison(collected: dict[str, JudgeStats]) -> None:
    if len(collected) < 2:
        return
    print("\n\033[1m🥊 head to head\033[0m")
    psoups = any(stats.principle_scores for stats in collected.values())
    if psoups:
        print(f"   {'run':<46}{'⭐ mean':>9}{'🏆 ≥8':>9}{'🎭 prin':>9}{'🤝 help':>9}{'✅ follow':>10}{'🧨 defect':>10}")
    else:
        print(f"   {'run':<46}{'⭐ mean':>9}{'🏆 ≥8':>9}{'💀 =0':>9}{'🌀 hall':>9}")
    for label, stats in collected.items():
        if not stats.scores:
            continue
        mean = mean_of(stats.scores)
        good = sum(1 for s in stats.scores if s >= 8) / len(stats.scores)
        if psoups:
            follow = stats.follows_principle / stats.parsed
            defect = stats.defective / stats.parsed
            print(f"   {label[:45]:<46}{mean:>9.2f}{good:>9.1%}{mean_of(stats.principle_scores):>9.2f}"
                  f"{mean_of(stats.helpfulness_scores):>9.2f}{follow:>10.1%}{defect:>10.1%}")
            continue
        zero = sum(1 for s in stats.scores if s == 0) / len(stats.scores)
        hall = stats.hallucinations / stats.parsed
        print(f"   {label[:45]:<46}{mean:>9.2f}{good:>9.1%}{zero:>9.1%}{hall:>9.1%}")
    best = max(collected.items(), key=lambda kv: sum(kv[1].scores) / len(kv[1].scores) if kv[1].scores else -1)
    print(f"   🎉 highest mean score: {best[0]}")


async def rerun_all(
    pairs: list[tuple[Path, Path]],
    config: JudgeConfig,
    labels: dict[str, str],
) -> dict[str, JudgeStats]:
    collected: dict[str, JudgeStats] = {}
    for rerun_file, run_dir in pairs:
        stats = await rerun_errors(rerun_file, run_dir, config, labels)
        print_stats(stats, rerun_file.stem)
        print(f"   💾 saved → {rerun_file}")
        collected[rerun_file.stem] = stats
    return collected


async def judge_all(
    run_dirs: list[Path],
    output_dir: Path,
    config: JudgeConfig,
    n: int | None,
    seed: int,
    unique_questions: bool,
    resume: bool,
    labels: dict[str, str],
) -> dict[str, JudgeStats]:
    collected: dict[str, JudgeStats] = {}
    for run_dir in run_dirs:
        output_path = output_dir / f"{run_dir.name}.jsonl"
        stats = await run_judge(
            run_dir=run_dir,
            output_path=output_path,
            config=config,
            n=n,
            seed=seed,
            unique_questions=unique_questions,
            resume=resume,
            labels=labels,
        )
        print_stats(stats, run_dir.name)
        print(f"   💾 saved → {output_path}")
        collected[run_dir.name] = stats
    return collected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = JudgeConfig(
        model_name=args.model,
        temperature=args.temperature,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )
    print("\n\033[1m✨ LLM-as-a-judge for FinQA ✨\033[0m")
    labels = load_labels(None if args.no_labels else args.labels)
    if args.rerun:
        if args.rerun_file:
            run_dirs = args.run_dir or [rerun_file.resolve().parent.parent / rerun_file.stem for rerun_file in args.rerun_file]
            pairs = list(zip(args.rerun_file, run_dirs))
        else:
            pairs = [(args.output_dir / f"{run_dir.name}.jsonl", run_dir) for run_dir in args.run_dir]
        collected = asyncio.run(rerun_all(pairs, config, labels))
        print_comparison(collected)
        print("\n   🏁 done\n")
        return 0
    collected = asyncio.run(
        judge_all(
            run_dirs=args.run_dir,
            output_dir=args.output_dir,
            config=config,
            n=args.n,
            seed=args.seed,
            unique_questions=args.unique_questions,
            resume=not args.no_resume,
            labels=labels,
        )
    )
    print_comparison(collected)
    print("\n   🏁 done\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
