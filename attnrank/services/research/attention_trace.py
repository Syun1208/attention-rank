from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import attnrank as ar
from attnrank.services.research.plot_attention_trace import render
from attnrank.utils.workspace import workspace_path


def is_gold(document: dict) -> bool:
    return str(document.get("is_gold", "")).lower() == "true"


def gold_first(documents: list[dict]) -> list[dict]:
    return [d for d in documents if is_gold(d)] + [d for d in documents if not is_gold(d)]


def load_records(pattern: str) -> dict[str, dict[str, dict]]:
    table: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                table[record["id"]][record["strategy"]] = record
    return table


def orderings(row: dict, index: int, profile, seed: int) -> dict[str, list[dict]]:
    ranked = gold_first(row["documents"])
    shuffled = list(ranked)
    random.Random(seed + index).shuffle(shuffled)
    return {
        "descending": ar.rerank_baseline(documents_by_relevance=ranked, strategy="descending"),
        "ascending": ar.rerank_baseline(documents_by_relevance=ranked, strategy="ascending"),
        "lim": ar.rerank_baseline(documents_by_relevance=ranked, strategy="lim"),
        "attnrank": ar.rerank_baseline(
            documents_by_relevance=ranked, strategy="attnrank", position_order=profile.position_order
        ),
        "random": shuffled,
    }


def slot_pair_accuracy(mask: np.ndarray, hits: np.ndarray) -> dict[str, list]:
    cell: dict[tuple, list] = defaultdict(lambda: [0, 0])
    for row_mask, hit in zip(mask, hits):
        key = tuple(int(i) for i in np.flatnonzero(row_mask))
        cell[key][0] += int(hit)
        cell[key][1] += 1
    return {str(k): [v[0] / v[1] * 100, v[1]] for k, v in sorted(cell.items())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace per-layer attention on gold vs competing documents for each ordering strategy")
    parser.add_argument("--model", default=str(workspace_path(relative="models/qwen2.5-7b-instruct")))
    parser.add_argument("--profile", default=str(workspace_path(relative="profiles/profile-qwen7b-hotpotqa-fig5.json")))
    parser.add_argument("--dataset", default="data/hotpotqa-attnrank/dataset.jsonl")
    parser.add_argument("--records", default=str(workspace_path(relative="outputs/qwen7b-fig5-shard*.jsonl")))
    parser.add_argument("--questions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ar.set_log_level(level="warn")
    settings = ar.EngineSettings(
        device=args.device,
        max_sequence=4096,
        chat_format=args.chat_format,
        include_titles=False,
        document_separator="\n",
    )
    engine = ar.load_model(model_dir=Path(args.model), settings=settings)
    profile = ar.load_attention_profile(path=Path(args.profile))
    records = load_records(args.records)

    rows = []
    with open(args.dataset) as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row["id"] in records and len(records[row["id"]]) == 5 and len(row["documents"]) == 5:
                rows.append((index, row))
            if len(rows) >= args.questions:
                break
    print(f"tracing {len(rows)} questions", file=sys.stderr)

    strategies = ("descending", "ascending", "lim", "attnrank", "random")
    layers = engine.num_layers
    attention = {s: np.zeros((len(rows), layers, 5)) for s in strategies}
    gold_mask = {s: np.zeros((len(rows), 5), dtype=bool) for s in strategies}
    hits = {s: np.zeros(len(rows), dtype=bool) for s in strategies}

    for position, (index, row) in enumerate(rows):
        for strategy, ordered in orderings(row, index, profile, args.seed).items():
            prompt = ar.build_prompt(engine=engine, question=row["question"], documents=ordered)
            result = engine.measure_attention_by_layer(prompt, layer_begin=0, layer_end=-1)
            attention[strategy][position] = np.asarray(result.attention_by_layer)
            gold_mask[strategy][position] = [is_gold(d) for d in ordered]
            hits[strategy][position] = bool(records[row["id"]][strategy]["hit"])
        if (position + 1) % 50 == 0:
            print(f"{position + 1}/{len(rows)}", file=sys.stderr)

    summary: dict = {
        "model": args.model,
        "profile": args.profile,
        "profile_layer": profile.layer_index,
        "profile_position_order": list(profile.position_order),
        "n_questions": len(rows),
        "layers": layers,
        "position_attention_random_order": attention["random"].mean(axis=0).tolist(),
    }
    gold_share = {}
    for strategy in strategies:
        mask = gold_mask[strategy][:, None, :]
        gold_mean = (attention[strategy] * mask).sum(axis=2) / mask.sum(axis=2)
        noise_mean = (attention[strategy] * ~mask).sum(axis=2) / (~mask).sum(axis=2)
        gold_share[strategy] = (attention[strategy] * mask).sum(axis=2)
        summary[f"mean_attention_{strategy}"] = attention[strategy].mean(axis=0).tolist()
        summary[f"gold_minus_noise_{strategy}"] = (gold_mean - noise_mean).mean(axis=0).tolist()
        summary[f"accuracy_{strategy}"] = float(hits[strategy].mean() * 100)
        summary[f"gold_slots_{strategy}"] = [int(s) for s in np.flatnonzero(gold_mask[strategy].mean(axis=0) > 0.5)]

    pooled_share = np.concatenate([gold_share[s] for s in strategies], axis=0)
    pooled_hits = np.concatenate([hits[s] for s in strategies], axis=0).astype(float)
    summary["corr_gold_share_vs_hit_by_layer"] = [float(np.corrcoef(pooled_share[:, layer], pooled_hits)[0, 1]) for layer in range(layers)]

    quintile_layers = sorted({profile.layer_index, 7, 13, 20, layers - 1})
    quintile_table = {}
    for layer in quintile_layers:
        share = pooled_share[:, layer]
        edges = np.quantile(share, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(share, edges)
        quintile_table[str(layer)] = [float(pooled_hits[bins == b].mean() * 100) for b in range(5)]
    summary["accuracy_by_gold_share_quintile"] = quintile_table

    summary["accuracy_by_gold_slots"] = slot_pair_accuracy(np.concatenate([gold_mask[s] for s in strategies]), np.concatenate([hits[s] for s in strategies]))
    summary["accuracy_by_gold_slots_random"] = slot_pair_accuracy(gold_mask["random"], hits["random"])

    summary_path = out_dir / "attention_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    for name in render(summary_path, out_dir):
        print(name)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("accuracy_") and not k.startswith("accuracy_by")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
