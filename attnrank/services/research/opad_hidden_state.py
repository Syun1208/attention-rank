from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from attnrank.services.hotpotqa import relevance_order
from attnrank.services.research.correctness_trace import figure_before_after, load_outcomes
from attnrank.services.research.placement_trace import is_gold, place, resolve_order
from attnrank.utils.workspace import workspace_path

PLACEMENTS = ("attnrank", "random", "descending")
PROJECT_DIR = Path(__file__).resolve().parents[3]
OPAD_ROOT = Path("/home/acm_llm/longpm/opad/OPAD")


def load_opad_module(opad_root: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, opad_root / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found under {opad_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def opad_prompts(opad_root: Path, principle_id: int):
    Llama2ConversationAdapter = load_opad_module(opad_root, "conversation").Llama2ConversationAdapter
    Principle = load_opad_module(opad_root, "dataset").Principle

    adapter = Llama2ConversationAdapter()
    principle = Principle().principle_list_hh[principle_id]

    def body(question: str, documents: list[dict]) -> str:
        lines = "\n".join(f"Document [{i + 1}] {d['text']}" for i, d in enumerate(documents))
        return f"{lines}\n\nQuestion: {question}"

    def with_principle(question: str, documents: list[dict]) -> str:
        return adapter.format_dialogue(principle, ["Human", body(question, documents)])

    def without_principle(question: str, documents: list[dict]) -> str:
        return adapter.format_dialogue("", ["Human", body(question, documents)])

    return with_principle, without_principle, principle


class LastState:
    def __init__(self, model_dir: str, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()

    def __call__(self, text: str) -> np.ndarray:
        torch = self.torch
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.model.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        return output.hidden_states[-1][0, -1].float().cpu().numpy().astype(np.float16)


def collect(args) -> Path:
    outcomes = load_outcomes(sorted(workspace_path(relative="outputs").glob("qwen7b-fig5-bm25-shard*.jsonl")), "attnrank")
    rows = []
    with open(args.dataset) as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row["id"] in outcomes and len(row["documents"]) == 5:
                rows.append((index, row))
    rows = rows[: args.limit] if args.limit else rows
    slots = resolve_order([int(v) for v in json.loads(Path(args.profile).read_text())["position_order"]], 5)
    with_principle, without_principle, principle = opad_prompts(Path(args.opad_root), args.principle_id)
    state = LastState(args.model, args.device, args.dtype)
    store: dict[str, list] = {k: [] for k in ("index", "correct", "gold", "reference_h", "h_without_principle", *(f"h_{p}" for p in PLACEMENTS))}
    for done, (index, row) in enumerate(rows):
        ranked = relevance_order(question=row["question"], documents=row["documents"], ordering="bm25")
        placed = {"attnrank": place(ranked, slots), "descending": list(ranked)}
        shuffled = list(ranked)
        random.Random(index).shuffle(shuffled)
        placed["random"] = shuffled
        store["index"].append(index)
        store["correct"].append(bool(outcomes[row["id"]]["hit"]))
        store["gold"].append([is_gold(d) for d in placed["attnrank"]])
        store["reference_h"].append(np.stack([state(with_principle(row["question"], [d])) for d in placed["attnrank"]]))
        store["h_without_principle"].append(state(without_principle(row["question"], placed["attnrank"])))
        for name in PLACEMENTS:
            store[f"h_{name}"].append(state(with_principle(row["question"], placed[name])))
        if (done + 1) % 50 == 0:
            print(f"{done + 1}/{len(rows)}", file=sys.stderr, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, principle=principle, **{k: np.asarray(v) for k, v in store.items()})
    return out


def add_without_gold(args) -> None:
    arrays = dict(np.load(args.out))
    with open(args.dataset) as handle:
        rows = [json.loads(line) for line in handle]
    slots = resolve_order([int(v) for v in json.loads(Path(args.profile).read_text())["position_order"]], 5)
    with_principle, _, _ = opad_prompts(Path(args.opad_root), args.principle_id)
    state = LastState(args.model, args.device, args.dtype)
    states = []
    for done, index in enumerate(arrays["index"]):
        row = rows[int(index)]
        ranked = relevance_order(question=row["question"], documents=row["documents"], ordering="bm25")
        placed = place(ranked, slots)
        states.append(state(with_principle(row["question"], [d for d in placed if not is_gold(d)])))
        if (done + 1) % 100 == 0:
            print(f"{done + 1}/{len(arrays['index'])}", file=sys.stderr, flush=True)
    arrays["h_without_gold"] = np.stack(states)
    np.savez(args.out, **arrays)


def render(args) -> dict:
    from attnrank.services.research.plot_attention_trace import apply_style

    apply_style()
    arrays = dict(np.load(args.out))
    n = len(arrays["index"])
    after = {
        "index": arrays["index"],
        "correct": arrays["correct"],
        "gold": arrays["gold"],
        "h": arrays["h_attnrank"][:, None, :],
        "reference_h": arrays["reference_h"][:, :, None, :],
        "slot_match": np.ones(n, dtype=bool),
    }
    summary: dict = {"questions_collected": int(n), "answered_correctly_with_attnrank": int(arrays["correct"].sum())}
    for name in ("random", "descending"):
        before = {
            "index": arrays["index"],
            "correct": arrays["correct"],
            "gold": arrays["gold"],
            "h": arrays[f"h_{name}"][:, None, :],
            "slot_match": np.zeros(n, dtype=bool),
        }
        figure = f"fig_hlast_before_after_opad" if name == args.before else None
        if figure is None:
            report = figure_before_after(
                f"_tmp_{name}", "", after, before, 0, Path(args.figures), r"$h_{\mathrm{last}}$", "h", "reference_h",
                f"Before AttnRank ({name} order)", "After AttnRank",
            )
            (Path(args.figures) / f"_tmp_{name}.png").unlink(missing_ok=True)
        else:
            report = figure_before_after(
                figure, r"$h_{\mathrm{last}}$ (last layer), OPAD prompt", after, before, 0, Path(args.figures),
                r"$h_{\mathrm{last}}$", "h", "reference_h", f"Without AttnRank ({name} order)", "With AttnRank",
            )
        summary[name] = report
    without = {
        "index": arrays["index"],
        "correct": arrays["correct"],
        "gold": arrays["gold"],
        "h": arrays["h_without_principle"][:, None, :],
        "slot_match": np.zeros(n, dtype=bool),
    }
    report = figure_before_after(
        "_tmp_without", "", after, without, 0, Path(args.figures), r"$h_{\mathrm{last}}$", "h", "reference_h",
        "Without principle", "With principle",
    )
    (Path(args.figures) / "_tmp_without.png").unlink(missing_ok=True)
    summary["without_principle_vs_with_principle_attnrank"] = report
    if "h_without_gold" in arrays:
        without_gold = {
            "index": arrays["index"],
            "correct": arrays["correct"],
            "gold": arrays["gold"],
            "h": arrays["h_without_gold"][:, None, :],
            "slot_match": np.zeros(n, dtype=bool),
        }
        summary["without_gold_vs_attnrank"] = figure_before_after(
            "fig_hlast_without_gold_before_after_vs_opad", r"$h_{\mathrm{last}}$ (last layer), OPAD prompt", after, without_gold, 0, Path(args.figures),
            r"$h_{\mathrm{last}}$", "h", "reference_h", "Without AttnRank", "With AttnRank",
        )
    (Path(args.figures) / "fig_hlast_before_after_opad_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hidden state of the OPAD prompt before and after AttnRank on the HotpotQA questions")
    parser.add_argument("--opad-root", default=str(OPAD_ROOT))
    parser.add_argument("--model", default=str(workspace_path(relative="models/qwen2.5-7b-instruct")))
    parser.add_argument("--profile", default=str(workspace_path(relative="profiles/profile-qwen7b-hotpotqa-fig5.json")))
    parser.add_argument("--dataset", default=str(PROJECT_DIR / "data/hotpotqa-attnrank/dataset.jsonl"))
    parser.add_argument("--principle-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--before", choices=("random", "descending"), default="random")
    parser.add_argument("--out", default=str(workspace_path(relative="outputs/opad_hidden_state.npz")))
    parser.add_argument("--figures", default=str(PROJECT_DIR / "docs/figures"))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--add-without-gold", action="store_true")
    args = parser.parse_args(argv)
    if args.add_without_gold:
        add_without_gold(args)
    elif not args.plot_only:
        collect(args)
    summary = render(args)
    for key in ("random", "descending", "without_principle_vs_with_principle_attnrank", "without_gold_vs_attnrank"):
        if key not in summary:
            continue
        report = summary[key]
        print(key, "questions", report["questions"], "| before: cos gold", round(report["before_vs_gold"]["mean_cosine"], 4), "cos false", round(report["before_vs_false"]["mean_cosine"], 4),
              "| after: cos gold", round(report["after_vs_gold"]["mean_cosine"], 4), "cos false", round(report["after_vs_false"]["mean_cosine"], 4),
              "| margin", round(report["before_margin_gold_minus_false"], 4), "->", round(report["after_margin_gold_minus_false"], 4),
              "| cos(before, after)", round(report["cosine_between_before_and_after_state"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
