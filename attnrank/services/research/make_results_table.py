from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Sequence
from attnrank.utils.workspace import workspace_path

PROJECT_DIR = Path(__file__).resolve().parents[3]
STRATEGIES = ("random", "descending", "ascending", "lim", "attnrank")
PAPER_QWEN_7B = (52.32, 53.31, 54.64, 52.18, 54.55)


def aggregate(paths: list[str]) -> tuple[int, tuple[float, ...]]:
    hits: Counter = Counter()
    counts: Counter = Counter()
    seen = set()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                key = (record["id"], record["strategy"])
                if key in seen:
                    continue
                seen.add(key)
                counts[record["strategy"]] += 1
                hits[record["strategy"]] += int(record["hit"])
    if not counts:
        raise SystemExit("no evaluation records found")
    n = min(counts[s] for s in STRATEGIES)
    return n, tuple(hits[s] / counts[s] * 100 for s in STRATEGIES)


def ranks(values: tuple[float, ...]) -> list[int | None]:
    order = sorted(set(values), reverse=True)
    return [order.index(v) if order.index(v) < 2 else None for v in values]


def cell(value: float, rank: int | None) -> str:
    text = f"{value:.2f}"
    if rank == 0:
        return r"\textbf{" + text + "}"
    if rank == 1:
        return r"\underline{" + text + "}"
    return text


def row(method: str, layer: str, questions: int, values: tuple[float, ...], shade: bool) -> str:
    marks = ranks(values)
    cells = [cell(v, marks[i]) for i, v in enumerate(values)]
    prefix = r"\rowcolor{oursrow} " if shade else ""
    return prefix + " & ".join([method, "Qwen2.5-7B", layer, str(questions), *cells]) + r" \\"


def delta_row(ours: tuple[float, ...], paper: tuple[float, ...]) -> str:
    cells = [f"{a - b:+.2f}" for a, b in zip(ours, paper)]
    return " & ".join([r"$\Delta$ vs paper", "", "", "", *cells]) + r" \\"


def judge_summary(path: Path) -> dict:
    verdicts = {}
    failed = 0
    with path.open() as handle:
        for line in handle:
            verdict = json.loads(line)
            if "error" in verdict:
                failed += 1
                continue
            verdicts[verdict["id"]] = verdict
    scores = [v["score"] for v in verdicts.values()]
    total = len(scores)
    return {
        "judged": total,
        "failed": failed,
        "mean": sum(scores) / total,
        "high": sum(1 for v in scores if v >= 8) / total * 100,
        "zero": sum(1 for v in scores if v == 0) / total * 100,
        "invalid": sum(1 for v in verdicts.values() if not v["derivation_valid"]) / total * 100,
        "hallucination": sum(1 for v in verdicts.values() if v["hallucination"]) / total * 100,
        "gold_suspect": sum(1 for v in verdicts.values() if v["gold_suspect"]) / total * 100,
    }


def finqa_table(run: dict, judge: dict, judge_model: str) -> list[str]:
    cells = [
        r"\rowcolor{oursrow} AttnRank (C++/CUDA)",
        "Qwen2.5-7B",
        str(run["profile_layer"]),
        str(judge["judged"]),
        f"{judge['mean']:.2f}",
        f"{judge['high']:.2f}",
        f"{judge['zero']:.2f}",
        f"{judge['invalid']:.2f}",
        f"{judge['hallucination']:.2f}",
        f"{judge['gold_suspect']:.2f}",
        f"{run['accuracy'] * 100:.2f}",
    ]
    caption = (
        r"\caption{FinQA with AttnRank and Qwen2.5-7B-Instruct, " + str(run["n_examples"]) + " examples. Each context is split into "
        + str(run["chunks"]) + r" contiguous chunks, the chunks are ranked by BM25 against the question and placed by the attention profile. "
        r"Shallowest layer is the probe layer of the FinQA attention profile, numbered from 0. "
        r"Judge columns come from " + judge_model + r" with the FinQA alignment prompt: the score is 0 to 10 and is set to 0 when the derivation is invalid, "
        r"even if the final value matches the reference. Numeric accuracy compares the extracted final value with the reference, "
        r"within half a unit of the last reference decimal.}"
    )
    return [
        r"\begin{table*}[t]",
        r"\centering",
        caption,
        r"\label{tab:finqa}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll|c|r|cccccc|c}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Model} & \textbf{Shallowest layer} & \textbf{Questions} & \textbf{Judge score (0--10)} & \textbf{Score $\geq 8$ (\%)} & \textbf{Score $= 0$ (\%)} & \textbf{Invalid derivation (\%)} & \textbf{Hallucination (\%)} & \textbf{Gold suspect (\%)} & \textbf{Numeric accuracy (\%)} \\",
        r"\midrule",
        " & ".join(cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]


def build(ours_n: int, ours_values: tuple[float, ...], ours_layer: int, finqa: list[str]) -> str:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{oursrow}{HTML}{E8F5E9}",
        r"\begin{document}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{HotpotQA answer accuracy (\%) of Qwen2.5-7B-Instruct under five document ordering strategies, five documents per question. Top: Table~1 of Yi et al.\ (2025). Highlighted: our C++/CUDA engine with the prompt of Figure~5 and the relevance order given by BM25 over the five candidates. Shallowest layer is the probe layer of the attention profile, numbered from 0: the paper uses the first layer, ours is the shallowest layer showing the attention basin over 397 probe prompts. $\Delta$ vs paper is ours minus the paper. Bold: best strategy in each row; underlined: second best.}",
        r"\label{tab:reproduce}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll|c|r|ccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Model} & \textbf{Shallowest layer} & \textbf{Questions} & \textbf{Random} & \textbf{Descending} & \textbf{Ascending} & \textbf{LIM} & \textbf{AttnRank} \\",
        r"\midrule",
        row("Yi et al. (2025)", "0", 7405, PAPER_QWEN_7B, shade=False),
        row("Ours (C++/CUDA)", str(ours_layer), ours_n, ours_values, shade=True),
        r"\midrule",
        delta_row(ours_values, PAPER_QWEN_7B),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        *finqa,
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render docs/reproduce_results.tex: paper vs our BM25-ordered run for Qwen2.5-7B")
    parser.add_argument("--root", default=str(PROJECT_DIR))
    parser.add_argument("--records", default=str(workspace_path(relative="outputs/qwen7b-fig5-bm25-shard*.jsonl")))
    parser.add_argument("--profile", default=str(workspace_path(relative="profiles/profile-qwen7b-hotpotqa-fig5.json")))
    parser.add_argument("--out", default="docs/reproduce_results.tex")
    parser.add_argument("--finqa-run", default=str(workspace_path(relative="outputs/finqa_attnrank_qwen2.5-7b-instruct_k5_t0.0")))
    parser.add_argument("--judge-dir", default=str(workspace_path(relative="outputs/judge")))
    parser.add_argument("--judge-model", default="GPT-4o")
    args = parser.parse_args(argv)
    root = Path(args.root)
    n, values = aggregate(sorted(glob.glob(str(root / args.records))))
    output = root / args.out
    layer = json.loads((root / args.profile).read_text())["layer_index"]
    run_dir = root / args.finqa_run
    verdicts = root / args.judge_dir / f"{run_dir.name}.jsonl"
    finqa: list[str] = []
    if verdicts.exists() and (run_dir / "run.json").exists():
        judge = judge_summary(verdicts)
        finqa = finqa_table(json.loads((run_dir / "run.json").read_text()), judge, args.judge_model)
        print("finqa judge", {k: round(v, 2) for k, v in judge.items()})
    output.write_text(build(n, values, layer, finqa), encoding="utf-8")
    print(output)
    print("ours", n, [round(v, 2) for v in values])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
