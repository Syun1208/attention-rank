from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from attnrank.services.hotpotqa import relevance_order
from attnrank.services.research.placement_trace import build_prompt, is_gold, place, resolve_order, token_span
from attnrank.utils.workspace import workspace_path

LABELS = {"correct": "Factual response (AttnRank answer correct)", "wrong": "False response (AttnRank answer wrong)"}


def load_record(paths: list[Path], strategy: str, identifier: str) -> dict:
    for path in paths:
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["strategy"] == strategy and record["id"] == identifier:
                    return record
    raise KeyError(identifier)


def load_row(dataset: Path, identifier: str) -> dict:
    with dataset.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] == identifier:
                return row
    raise KeyError(identifier)


def first_sentence(text: str) -> str:
    head, separator, _ = text.partition(". ")
    return head + "." if separator else text


def trace(model, tokenizer, row: dict, record: dict, slots: list[int], step: int, max_tokens: int) -> dict:
    import torch

    ranked = relevance_order(question=row["question"], documents=row["documents"], ordering="bm25")
    placed = place(ranked, slots)
    gold = np.array([is_gold(d) for d in placed])
    response = first_sentence(record["prediction"].strip())
    prompt_text, char_spans = build_prompt(row["question"], placed)
    encoded = tokenizer(prompt_text + " " + response, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"][0].tolist()]
    prompt_tokens = sum(1 for _, b in offsets if b <= len(prompt_text))
    if offsets[prompt_tokens - 1][1] != len(prompt_text):
        raise RuntimeError("response tokens merged with the prompt boundary")
    input_ids = encoded["input_ids"][:, : prompt_tokens + max_tokens].to(model.device)
    count = input_ids.shape[1] - prompt_tokens
    rows = torch.arange(prompt_tokens - 1, prompt_tokens - 1 + count, device=model.device)
    targets = input_ids[0, prompt_tokens:]
    spans = [token_span(offsets, a, b) for a, b in char_spans]
    with torch.no_grad():
        output = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True, use_cache=False)
        layers = list(range(step - 1, len(output.attentions), step))
        direct = model.lm_head(output.hidden_states[-1][0, rows]).float()
        print(f"last hidden state is already normalised: {bool(torch.allclose(direct, output.logits[0, rows].float(), atol=1e-2))}", file=sys.stderr)
        probability = np.zeros((len(layers), count), dtype=np.float32)
        agreement = np.zeros((len(layers), count), dtype=bool)
        share = np.zeros((len(layers), count), dtype=np.float32)
        slot_share = np.zeros((len(layers), count, len(spans)), dtype=np.float32)
        document_share = np.zeros((len(layers), count), dtype=np.float32)
        for position, layer in enumerate(layers):
            hidden = output.hidden_states[layer + 1][0, rows]
            final = layer + 1 == len(output.attentions)
            logits = output.logits[0, rows].float() if final else model.lm_head(model.model.norm(hidden)).float()
            probabilities = torch.softmax(logits, dim=-1)
            probability[position] = probabilities[torch.arange(count), targets].cpu().numpy()
            agreement[position] = (probabilities.argmax(dim=-1) == targets).cpu().numpy()
            attention = output.attentions[layer][0, :, rows, :].float().mean(dim=0)
            mass = torch.stack([attention[:, begin:end].sum(dim=1) for begin, end in spans], dim=1).cpu().numpy()
            share[position] = mass[:, gold].sum(axis=1) / mass.sum(axis=1)
            slot_share[position] = mass / mass.sum(axis=1, keepdims=True)
            document_share[position] = mass.sum(axis=1)
    tokens = [tokenizer.decode([int(t)]) for t in targets]
    start = len(prompt_text) + 1
    found = response.lower().find(row["answer"].lower())
    window = (start + found, start + found + len(row["answer"])) if found >= 0 else (0, 0)
    reference_tokens = [i for i in range(count) if offsets[prompt_tokens + i][1] > window[0] and offsets[prompt_tokens + i][0] < window[1]]
    return {
        "id": row["id"],
        "question": row["question"],
        "reference": row["answer"],
        "response": response,
        "hit": bool(record["hit"]),
        "gold_slots": [int(i) + 1 for i in np.flatnonzero(gold)],
        "layers": [layer + 1 for layer in layers],
        "tokens": tokens,
        "reference_tokens": reference_tokens,
        "probability": probability.tolist(),
        "agreement": agreement.tolist(),
        "gold_attention_share": share.tolist(),
        "slot_attention_share": slot_share.tolist(),
        "document_attention_mass": document_share.tolist(),
        "documents": [{"slot": i + 1, "title": d["title"], "gold": bool(is_gold(d)), "contains_reference": row["answer"].lower() in d["text"].lower()} for i, d in enumerate(placed)],
    }


class TokenAttentionProbe:
    def __init__(self, model) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.rows = None
        self.spans: list[tuple[int, int]] = []
        self.mass: list = []
        for layer in model.model.layers:
            layer.self_attn.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        torch = self.torch
        weights = None
        for candidate in output if isinstance(output, (tuple, list)) else (output,):
            if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                weights = candidate
        if weights is None:
            raise RuntimeError("attention weights unavailable; eager attention is required")
        attention = weights[0][:, self.rows, :].float().mean(dim=0)
        self.mass.append(torch.stack([attention[:, begin:end].sum(dim=1) for begin, end in self.spans], dim=1))

    def measure(self, tokenizer, row: dict, record: dict, slots: list[int], max_tokens: int) -> dict:
        torch = self.torch
        ranked = relevance_order(question=row["question"], documents=row["documents"], ordering="bm25")
        placed = place(ranked, slots)
        response = first_sentence(record["prediction"].strip()) or row["answer"]
        prompt_text, char_spans = build_prompt(row["question"], placed)
        encoded = tokenizer(prompt_text + " " + response, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"][0].tolist()]
        prompt_tokens = sum(1 for _, b in offsets if b <= len(prompt_text))
        if offsets[prompt_tokens - 1][1] != len(prompt_text):
            raise RuntimeError("response tokens merged with the prompt boundary")
        input_ids = encoded["input_ids"][:, : prompt_tokens + max_tokens].to(self.model.device)
        count = input_ids.shape[1] - prompt_tokens
        self.rows = torch.arange(prompt_tokens - 1, prompt_tokens - 1 + count, device=self.model.device)
        self.spans = [token_span(offsets, a, b) for a, b in char_spans]
        self.mass = []
        with torch.no_grad():
            output = self.model(input_ids=input_ids, use_cache=False)
        mass = torch.stack(self.mass).cpu().numpy()
        log_probs = torch.log_softmax(output.logits[0, self.rows].float(), dim=-1)
        targets = input_ids[0, prompt_tokens:]
        found = response.lower().find(row["answer"].lower())
        start = len(prompt_text) + 1 + found
        reference = [i for i in range(count) if found >= 0 and offsets[prompt_tokens + i][1] > start and offsets[prompt_tokens + i][0] < start + len(row["answer"])]
        answer = row["answer"].lower()
        return {
            "mass": mass,
            "entropy": (-(log_probs.exp() * log_probs).sum(dim=-1)).cpu().numpy(),
            "probability": log_probs.exp()[torch.arange(count), targets].cpu().numpy(),
            "tokens": count,
            "reference_start": reference[0] if reference else -1,
            "gold": np.array([is_gold(d) for d in placed]),
            "bearing": np.array([is_gold(d) and answer in d["text"].lower() for d in placed]),
        }


def collect_population(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = [Path(p) for p in args.outcomes] if args.outcomes else sorted(workspace_path(relative="outputs").glob("qwen7b-fig5-bm25-shard*.jsonl"))
    outcomes = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["strategy"] == args.strategy:
                    outcomes[record["id"]] = record
    rows = []
    with open(args.dataset) as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row["id"] in outcomes and len(row["documents"]) == 5:
                rows.append((index, row))
    size = 250
    blocks = [rows[start : start + size] for start in range(0, len(rows), size)]
    shard, count = (int(v) for v in args.shard.split("/"))
    out_dir = Path(args.population_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [(number, block) for number, block in enumerate(blocks) if number % count == shard and not (out_dir / f"part-{number:04d}.npz").exists()]
    print(f"shard {shard}/{count}: {len(pending)} blocks pending of {len(blocks)}", file=sys.stderr, flush=True)
    if not pending:
        return
    order = [int(v) for v in json.loads(Path(args.profile).read_text())["position_order"]]
    slots = resolve_order(order, 5)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map=args.device, attn_implementation="eager")
    model.eval()
    probe = TokenAttentionProbe(model)
    layers = len(model.model.layers)
    width = args.max_tokens
    for done, (number, block) in enumerate(pending):
        store: dict[str, list] = {k: [] for k in ("index", "correct", "gold", "bearing", "mass", "entropy", "probability", "tokens", "reference_start")}
        for index, row in block:
            try:
                item = probe.measure(tokenizer, row, outcomes[row["id"]], slots, width)
            except RuntimeError as error:
                print(f"skipped {index}: {error}", file=sys.stderr, flush=True)
                continue
            mass = np.full((layers, width, 5), np.nan, dtype=np.float32)
            mass[:, : item["tokens"]] = item["mass"]
            store["mass"].append(mass)
            for key in ("entropy", "probability"):
                values = np.full(width, np.nan, dtype=np.float32)
                values[: item["tokens"]] = item[key]
                store[key].append(values)
            store["index"].append(index)
            store["correct"].append(bool(outcomes[row["id"]]["hit"]))
            for key in ("gold", "bearing", "tokens", "reference_start"):
                store[key].append(item[key])
        temporary = out_dir / f"part-{number:04d}.tmp.npz"
        np.savez(temporary, **{key: np.asarray(values) for key, values in store.items()})
        temporary.replace(out_dir / f"part-{number:04d}.npz")
        print(f"shard {shard}: block {done + 1}/{len(pending)} questions {len(store['index'])} correct {int(np.sum(store['correct']))}", file=sys.stderr, flush=True)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy import stats

    ranks = stats.rankdata(scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def load_population(directory: Path, step: int) -> dict:
    parts = [np.load(p) for p in sorted(directory.glob("part-*.npz")) if ".tmp" not in p.name]
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files}
    mass = arrays["mass"].astype(np.float64)
    gold = arrays["gold"]
    bearing = arrays["bearing"]
    bridge = gold & ~bearing
    total = mass.sum(axis=3)

    def share(mask: np.ndarray) -> np.ndarray:
        return np.where(mask[:, None, None, :], mass, 0.0).sum(axis=3) / total

    return {
        "correct": arrays["correct"],
        "two_hop": bearing.any(axis=1) & bridge.any(axis=1),
        "everyone": np.ones(len(gold), dtype=bool),
        "layers": list(range(step - 1, mass.shape[1], step)),
        "positions": np.arange(1, mass.shape[2] + 1),
        "gold": share(gold),
        "answer": share(bearing),
        "bridge": share(bridge),
    }


def figure_average(population: dict, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, save

    correct = population["correct"]
    layers = population["layers"]
    positions = population["positions"]
    mask = population["two_hop"]
    colors = {True: "#2166AC", False: "#B2182B"}
    styles = {True: "-", False: "--"}
    names = {True: "factual responses", False: "false responses"}
    summary: dict = {
        "questions": int(len(correct)),
        "factual": int(correct.sum()),
        "false": int((~correct).sum()),
        "two_hop_questions": int(mask.sum()),
        "layers_averaged": [layer + 1 for layer in layers],
    }
    for key, group in (("gold", "everyone"), ("answer", "two_hop"), ("bridge", "two_hop")):
        selected = population[group]
        response = np.nanmean(np.nanmean(population[key][:, layers], axis=1), axis=1) * 100.0
        summary[key] = {
            "factual_percent": float(response[selected & correct].mean()),
            "false_percent": float(response[selected & ~correct].mean()),
            "auroc_predicts_factual": auroc(response[selected], correct[selected]),
        }

    fig = plt.figure(figsize=(DOUBLE_WIDTH, 6.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.7], hspace=0.38)
    axis = fig.add_subplot(grid[0])
    averaged = np.nanmean(population["answer"][:, layers], axis=1) * 100.0
    for outcome in (True, False):
        block = averaged[mask & (correct == outcome)]
        mean = np.nanmean(block, axis=0)
        error = 1.96 * np.nanstd(block, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(block), axis=0))
        axis.plot(positions, mean, color=colors[outcome], linestyle=styles[outcome], marker="o", markersize=2.6, label=f"{names[outcome]} (n = {len(block)})")
        axis.fill_between(positions, mean - error, mean + error, color=colors[outcome], alpha=0.18, linewidth=0)
        summary["answer"][f"{'factual' if outcome else 'false'}_by_position"] = [float(v) for v in mean]
    axis.axhline(20.0, color="0.5", linewidth=0.7, linestyle=":", label="equal attention on all 5 documents")
    axis.set_xticks(positions)
    axis.set_xlim(0.5, positions[-1] + 0.5)
    axis.set_xlabel("Position of the token in the response")
    axis.set_ylabel("Attention on the\nanswer document (%)")
    axis.set_title(f"(a) Attention on the answer document, mean over layers {layers[0] + 1}, {layers[1] + 1}, ..., {layers[-1] + 1}", fontsize=8.2, loc="left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(loc="upper left", fontsize=6.8)

    axis = fig.add_subplot(grid[1])
    values = population["answer"][:, layers] * 100.0
    difference = np.nanmean(values[mask & correct], axis=0) - np.nanmean(values[mask & ~correct], axis=0)
    limit = float(np.abs(difference).max())
    image = axis.imshow(difference, aspect="auto", cmap="RdBu_r", origin="lower", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), interpolation="nearest")
    for row in range(difference.shape[0]):
        for column in range(difference.shape[1]):
            value = difference[row, column]
            axis.text(column, row, f"{value:+.0f}", ha="center", va="center", fontsize=5.4, color="white" if abs(value) > 0.72 * limit else "black")
    axis.set_xticks(range(len(positions)), [str(v) for v in positions])
    axis.set_yticks(range(len(layers)), [str(layer + 1) for layer in layers])
    axis.set_xlabel("Position of the token in the response")
    axis.set_ylabel("Layer")
    axis.set_title("(b) Attention on the answer document: factual minus false responses, in percentage points", fontsize=8.2, loc="left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(True)
    bar = fig.colorbar(image, cax=axis.inset_axes([1.012, 0.0, 0.018, 1.0]))
    bar.set_label("Difference (percentage points)", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    save(fig, out_dir, "fig_response_layers_average")
    summary["difference_answer_document"] = {"layers": [layer + 1 for layer in layers], "percentage_points": difference.tolist()}
    return summary


def figure_mean_heatmaps(population: dict, out_dir: Path, key: str, group: str, uniform: float, label: str, name: str) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, save

    correct = population["correct"]
    layers = population["layers"]
    positions = population["positions"]
    mask = population[group]
    values = population[key][:, layers]
    means = {True: np.nanmean(values[mask & correct], axis=0), False: np.nanmean(values[mask & ~correct], axis=0)}
    counts = {True: int((mask & correct).sum()), False: int((mask & ~correct).sum())}
    titles = {True: "Factual responses (AttnRank answer correct)", False: "False responses (AttnRank answer wrong)"}
    top = max(float(means[True].max()), float(means[False].max()))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=uniform, vmax=max(1.0 if uniform >= 0.4 else 0.6, top))
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_WIDTH, 8.4))
    image = None
    for index, (axis, outcome) in enumerate(zip(axes, (True, False))):
        block = means[outcome]
        image = axis.imshow(block, aspect="auto", cmap="RdBu_r", origin="lower", norm=norm, interpolation="nearest")
        for row in range(block.shape[0]):
            for column in range(block.shape[1]):
                value = block[row, column]
                shade = norm(value)
                axis.text(column, row, f"{value:.2f}"[1:], ha="center", va="center", fontsize=5.2, color="white" if shade >= 0.85 or shade <= 0.12 else "black")
        axis.set_xticks(range(len(positions)), [str(v) for v in positions])
        axis.set_yticks(range(len(layers)), [str(layer + 1) for layer in layers])
        axis.set_xlabel("Position of the token in the response")
        axis.set_ylabel("Layer")
        axis.set_title(f"({chr(ord('a') + index)}) {titles[outcome]}, mean over {counts[outcome]} responses", fontsize=8.2, loc="left")
        for side in ("top", "right"):
            axis.spines[side].set_visible(True)
    fig.tight_layout(h_pad=1.6)
    bar = fig.colorbar(image, ax=axes.tolist(), fraction=0.022, pad=0.015)
    bar.set_label(f"{label} (equal attention = {uniform:.1f})", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    save(fig, out_dir, name)
    return {
        "factual_responses": counts[True],
        "false_responses": counts[False],
        "layers": [layer + 1 for layer in layers],
        "factual": means[True].tolist(),
        "false": means[False].tolist(),
        "largest_difference": float(np.abs(means[True] - means[False]).max()),
        "mean_difference": float((means[True] - means[False]).mean()),
    }


def render_population(directory: Path, out_dir: Path, step: int) -> dict:
    import warnings

    from attnrank.services.research.plot_attention_trace import apply_style

    warnings.simplefilter("ignore", category=RuntimeWarning)
    apply_style()
    population = load_population(directory, step)
    summary = figure_average(population, out_dir)
    summary["fig_response_layers_mean_gold"] = figure_mean_heatmaps(
        population, out_dir, "gold", "everyone", 0.4, "Share of document attention on the gold documents", "fig_response_layers_mean_gold"
    )
    summary["fig_response_layers_mean_answer"] = figure_mean_heatmaps(
        population, out_dir, "answer", "two_hop", 0.2, "Share of document attention on the answer document", "fig_response_layers_mean_answer"
    )
    (out_dir / "response_layers_average_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def display(token: str) -> str:
    text = token.replace("\n", "\\n").strip() or "(space)"
    return text.replace("$", r"\$").replace("_", r"\_")


def figure(items: dict, out_dir: Path) -> str:
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, save

    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.4, vmax=1.0)
    columns = max(len(item["tokens"]) for item in items.values())
    fig, axes = plt.subplots(len(items), 1, figsize=(DOUBLE_WIDTH, 4.3 * len(items)))
    image = None
    for index, (axis, (kind, item)) in enumerate(zip(np.atleast_1d(axes), items.items())):
        layers = item["layers"]
        tokens = item["tokens"]
        share = np.asarray(item["gold_attention_share"])
        image = axis.imshow(share, aspect="auto", cmap="RdBu_r", origin="lower", norm=norm, interpolation="nearest", extent=(-0.5, len(tokens) - 0.5, -0.5, len(layers) - 0.5))
        for row in range(share.shape[0]):
            for column in range(share.shape[1]):
                value = share[row, column]
                color = "white" if value >= 0.78 or value <= 0.12 else "black"
                axis.text(column, row, f"{value:.2f}"[1:] if value < 0.995 else "1.0", ha="center", va="center", fontsize=5.2, color=color)
        axis.set_xlim(-0.5, columns - 0.5)
        axis.set_xticks(range(len(tokens)), [display(t) for t in tokens], rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
        marked = set(item["reference_tokens"])
        for position, label in enumerate(axis.get_xticklabels()):
            if position in marked:
                label.set_fontweight("bold")
                label.set_color("#1B7837")
        axis.set_yticks(range(len(layers)), [str(v) for v in layers])
        axis.set_ylabel("Layer")
        axis.set_xlabel("Response token (bold green: reference answer)" if marked else "Response token (the reference answer does not appear in the response)")
        for side in ("top", "right"):
            axis.spines[side].set_visible(len(tokens) == columns)
        question = textwrap.fill(f"Question: {item['question']}", 110)
        axis.set_title(f"({chr(ord('a') + index)}) {LABELS[kind]}\n{question}\nReference answer: {item['reference']}   |   gold documents at slots {item['gold_slots']}", fontsize=7.6, loc="left")
    fig.tight_layout(h_pad=1.6)
    bar = fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), fraction=0.022, pad=0.015)
    bar.set_label("Share of document attention on the gold documents (uniform = 0.4)", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    name = "fig_response_layers"
    save(fig, out_dir, name)
    return name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Token-by-layer view of one correct and one wrong AttnRank response")
    parser.add_argument("--model", default=str(workspace_path(relative="models/qwen2.5-7b-instruct")))
    parser.add_argument("--profile", default=str(workspace_path(relative="profiles/profile-qwen7b-hotpotqa-fig5.json")))
    parser.add_argument("--dataset", default=str(workspace_path(relative="data/hotpotqa-attnrank/dataset.jsonl")))
    parser.add_argument("--outcomes", nargs="+", default=None)
    parser.add_argument("--strategy", default="attnrank")
    parser.add_argument("--correct-id", default="5ae12aa6554299422ee99617")
    parser.add_argument("--wrong-id", default="5ab613985542992aa134a418")
    parser.add_argument("--layer-step", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--cache", default="docs/figures/response_layer_trace.json")
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--population", action="store_true")
    parser.add_argument("--plot-population", action="store_true")
    parser.add_argument("--population-dir", default=str(workspace_path(relative="outputs/response_layers")))
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(argv)

    if args.population:
        collect_population(args)
        return 0
    if args.plot_population:
        summary = render_population(Path(args.population_dir), Path(args.out_dir), args.layer_step)
        for key in ("questions", "factual", "false", "two_hop_questions"):
            print(key, summary[key])
        for key in ("gold", "answer", "bridge"):
            print(key, {k: round(v, 3) for k, v in summary[key].items() if not k.endswith("by_position")})
        for key in ("fig_response_layers_mean_gold", "fig_response_layers_mean_answer"):
            print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary[key].items() if k not in ("factual", "false", "layers")})
        return 0

    cache = Path(args.cache)
    if args.plot_only:
        items = json.loads(cache.read_text())
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        paths = [Path(p) for p in args.outcomes] if args.outcomes else sorted(workspace_path(relative="outputs").glob("qwen7b-fig5-bm25-shard*.jsonl"))
        order = [int(v) for v in json.loads(Path(args.profile).read_text())["position_order"]]
        slots = resolve_order(order, 5)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map=args.device, attn_implementation="eager")
        model.eval()
        items = {}
        for kind, identifier in (("correct", args.correct_id), ("wrong", args.wrong_id)):
            record = load_record(paths, args.strategy, identifier)
            if record["hit"] != (kind == "correct"):
                raise ValueError(f"{identifier} is not a {kind} response")
            items[kind] = trace(model, tokenizer, load_row(Path(args.dataset), identifier), record, slots, args.layer_step, args.max_tokens)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(items, indent=1))

    from attnrank.services.research.plot_attention_trace import apply_style

    apply_style()
    name = figure(items, Path(args.out_dir))
    for kind, item in items.items():
        share = np.asarray(item["gold_attention_share"])
        print(f"{name} {kind}: {len(item['tokens'])} tokens, layers {item['layers']}, mean gold share {share.mean():.3f}")
        print("  response:", item["response"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
