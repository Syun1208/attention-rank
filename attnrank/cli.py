from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from attnrank.data.classes import (
    STRATEGIES,
    FinqaSettings,
    GenerationSettings,
    HotpotqaSettings,
    ProfileTaskSettings,
)
from attnrank.utils.config import load_env, load_settings
from attnrank.utils.logging import setup_logging
from attnrank.utils.workspace import Workspace, resolve_workspace

logger = logging.getLogger(__name__)

APP_NAME = "attnrank"
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERRUPTED = 130
RESEARCH_PACKAGE = "attnrank.services.research"
ENGINE_OVERRIDE_KEYS = ("device", "max_sequence", "chat_format")


def existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return path


def comma_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def engine_overrides(*, args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in ENGINE_OVERRIDE_KEYS if getattr(args, key, None) is not None}


def merge_engine_section(*, settings_values: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    section = dict(settings_values.get("engine") or {})
    section.update(engine_overrides(args=args))
    return section


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=existing_path, default=None, help="YAML file with the task settings")
    parser.add_argument("--model", default=None, help="model directory or Hugging Face repo id")
    parser.add_argument("--output-dir", type=Path, default=None, help="where results and the config copy go")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--max-sequence", type=int, default=None, help="context window in tokens")
    parser.add_argument("--chat-format", default=None, help="plain, chatml, vicuna, llama2 or mistral")


def overrides_from(*, args: argparse.Namespace, keys: Sequence[str]) -> dict[str, Any]:
    values = {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}
    values["engine"] = None
    return values


def run_name(*, args: argparse.Namespace) -> str:
    stem = args.config.stem if args.config is not None else "run"
    return f"{args.command}.{stem}"


def build_task_settings(settings_type: type, *, args: argparse.Namespace, keys: Sequence[str]) -> Any:
    from dataclasses import replace

    from attnrank.utils.config import read_yaml

    file_values = read_yaml(path=args.config) if args.config is not None else {}
    overrides = overrides_from(args=args, keys=keys)
    overrides["engine"] = merge_engine_section(settings_values=file_values, args=args)
    settings = load_settings(settings_type, config_path=args.config, overrides=overrides)
    workspace: Workspace = args.workspace_paths
    changes: dict[str, Any] = {}
    if settings.output_dir is None:
        changes["output_dir"] = workspace.run_output_dir(name=run_name(args=args))
    if getattr(settings, "profile", None) is not None:
        changes["profile"] = workspace.resolve_profile(path=settings.profile)
    for key in ("dataset", "probes"):
        spec = getattr(settings, key, None)
        if spec is not None and spec.path is not None:
            changes[key] = replace(spec, path=workspace.resolve_data(path=spec.path))
    settings = replace(settings, **changes)
    logger.info("output dir: %s", settings.output_dir)
    return settings


def run_profile(args: argparse.Namespace) -> int:
    from attnrank.services.profile import run_profile_task

    settings = build_task_settings(
        ProfileTaskSettings,
        args=args,
        keys=("model", "output_dir", "probes", "probe_samples", "layer", "min_edge_ratio"),
    )
    run_profile_task(settings=settings)
    return EXIT_OK


def run_rerank(args: argparse.Namespace) -> int:
    from attnrank.services.rerank import run_rerank_task

    result = run_rerank_task(
        documents=args.documents,
        profile=args.workspace_paths.resolve_profile(path=args.profile),
        question=args.question,
        model=args.model,
        engine_overrides=engine_overrides(args=args),
        generation=GenerationSettings(max_new_tokens=args.max_new_tokens),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return EXIT_OK


def run_hotpotqa(args: argparse.Namespace) -> int:
    from attnrank.services.hotpotqa import run_hotpotqa_task

    settings = build_task_settings(
        HotpotqaSettings,
        args=args,
        keys=("model", "output_dir", "dataset", "profile", "probes", "ordering", "strategies", "offset", "questions"),
    )
    run_hotpotqa_task(settings=settings)
    return EXIT_OK


def run_hotpotqa_report(args: argparse.Namespace) -> int:
    from attnrank.services.hotpotqa import run_hotpotqa_report

    print(run_hotpotqa_report(records=args.records, paper_row=args.paper_row))
    return EXIT_OK


def run_finqa(args: argparse.Namespace) -> int:
    from attnrank.services.finqa import run_finqa_task

    settings = build_task_settings(
        FinqaSettings,
        args=args,
        keys=(
            "model",
            "output_dir",
            "dataset",
            "profile",
            "strategy",
            "chunks",
            "examples",
            "shard",
            "rerun",
            "profile_only",
            "finalize",
        ),
    )
    run_finqa_task(settings=settings)
    return EXIT_OK


def run_research(args: argparse.Namespace) -> int:
    module = importlib.import_module(f"{RESEARCH_PACKAGE}.{args.name}")
    return int(module.main(args.arguments))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=APP_NAME, description="AttnRank: attention-basin layer, profile, rerank")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging on the console")
    parser.add_argument("--workspace", type=Path, default=None, help="root holding logs/, outputs/, profiles/")
    parser.add_argument("--log-dir", type=Path, default=None, help="where log files go (default <workspace>/logs)")
    parser.add_argument("--profiles-dir", type=Path, default=None, help="where profile names are looked up")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="parent of per-run output folders")
    parser.add_argument("--data-dir", type=Path, default=None, help="where relative dataset paths are looked up")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="scan layers on probe prompts and write an attention profile")
    add_common_arguments(profile)
    profile.add_argument("--probes", default=None, help="probe samples: JSONL path or hf:repo[:config]@split")
    profile.add_argument("--probe-samples", type=int, default=None, help="number of probe samples to use")
    profile.add_argument("--layer", type=int, default=None, help="profile this layer instead of scanning")
    profile.add_argument("--min-edge-ratio", type=float, default=None, help="basin criterion")
    profile.set_defaults(handler=run_profile)

    rerank = subparsers.add_parser("rerank", help="place top-k documents (relevance order) into profiled slots")
    rerank.add_argument("--documents", type=existing_path, required=True, help="JSON list or JSONL of top-k documents")
    rerank.add_argument("--profile", type=Path, required=True, help="attention profile JSON (path or name)")
    rerank.add_argument("--question", default=None, help="generate an answer for this question")
    rerank.add_argument("--model", default=None, help="model directory or repo id, needed with --question")
    rerank.add_argument("--device", type=int, default=None)
    rerank.add_argument("--max-sequence", type=int, default=None)
    rerank.add_argument("--chat-format", default=None)
    rerank.add_argument("--max-new-tokens", type=int, default=GenerationSettings().max_new_tokens)
    rerank.set_defaults(handler=run_rerank)

    hotpotqa = subparsers.add_parser("hotpotqa", help="evaluate ordering strategies on HotpotQA")
    add_common_arguments(hotpotqa)
    hotpotqa.add_argument("--dataset", default=None, help="JSONL path or hf:repo[:config]@split")
    hotpotqa.add_argument("--profile", type=Path, default=None, help="profile JSON path or name in profiles dir")
    hotpotqa.add_argument("--probes", default=None, help="JSONL path or hf:repo[:config]@split")
    hotpotqa.add_argument("--ordering", choices=("bm25", "gold-first", "given"), default=None)
    hotpotqa.add_argument("--strategies", type=comma_list, default=None, help=f"subset of {','.join(STRATEGIES)}")
    hotpotqa.add_argument("--offset", type=int, default=None)
    hotpotqa.add_argument("--questions", type=int, default=None, help="0 means all")
    hotpotqa.set_defaults(handler=run_hotpotqa)

    report = subparsers.add_parser("hotpotqa-report", help="summarise HotpotQA record files")
    report.add_argument("--records", type=existing_path, nargs="+", required=True)
    report.add_argument("--paper-row", default="qwen2.5-7b")
    report.set_defaults(handler=run_hotpotqa_report)

    finqa = subparsers.add_parser("finqa", help="AttnRank over chunks of a long FinQA context")
    add_common_arguments(finqa)
    finqa.add_argument("--dataset", default=None, help="JSONL path or hf:repo[:config]@split")
    finqa.add_argument("--profile", type=Path, default=None)
    finqa.add_argument("--strategy", choices=("attnrank", "descending", "ascending", "lim", "random", "original"))
    finqa.add_argument("--chunks", type=int, default=None)
    finqa.add_argument("--examples", type=int, default=None)
    finqa.add_argument("--shard", default=None, help="INDEX/COUNT")
    finqa.add_argument("--rerun", action="store_const", const=True, default=None)
    finqa.add_argument("--profile-only", action="store_const", const=True, default=None)
    finqa.add_argument("--finalize", action="store_const", const=True, default=None)
    finqa.set_defaults(handler=run_finqa)

    research = subparsers.add_parser("research", help="run a research script: attention_trace, placement_trace, ...")
    research.add_argument("name")
    research.add_argument("arguments", nargs=argparse.REMAINDER)
    research.set_defaults(handler=run_research)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env(start=Path.cwd())
    load_env(start=Path(__file__).resolve().parent)
    args.workspace_paths = resolve_workspace(
        root=args.workspace,
        logs_dir=args.log_dir,
        outputs_dir=args.outputs_dir,
        profiles_dir=args.profiles_dir,
        data_dir=args.data_dir,
    )
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(APP_NAME, log_dir=args.workspace_paths.logs_dir, level=level)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_INTERRUPTED
    except (ValueError, FileNotFoundError) as error:
        logger.exception("input error: %s", error)
        return EXIT_USAGE_ERROR
    except Exception as error:
        logger.exception("failed: %s", error)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
