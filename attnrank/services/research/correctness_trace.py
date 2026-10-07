from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from attnrank.services.hotpotqa import relevance_order
from attnrank.services.research.placement_trace import LayerTracer, build_prompt, is_gold, place, resolve_order
from attnrank.utils.workspace import workspace_path

TOKENS = 16
BLOCK = 100


def load_outcomes(paths: list[Path], strategy: str) -> dict[str, dict]:
    outcomes = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["strategy"] == strategy:
                    outcomes[record["id"]] = record
    return outcomes


def padded(values: np.ndarray) -> np.ndarray:
    row = np.full(TOKENS, np.nan, dtype=np.float32)
    count = min(TOKENS, len(values))
    row[:count] = values[:count]
    return row


class HiddenTracer:
    def __init__(self, model_dir: str, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()

    def trace(self, question: str, documents: list[dict], answer: str) -> dict:
        torch = self.torch
        text, _ = build_prompt(question, documents)
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        return {"residual": torch.stack([state[0, -1].float() for state in output.hidden_states[1:]]).cpu().numpy()}


def arrange(row: dict, index: int, slots: list[int], placement: str) -> list[dict]:
    ranked = relevance_order(question=row["question"], documents=row["documents"], ordering="bm25")
    if placement == "random":
        ordered = list(ranked)
        random.Random(index).shuffle(ordered)
        return ordered
    placed = place(ranked, slots)
    if placement == "without-gold":
        return [document for document in placed if not is_gold(document)]
    return placed


def trace_block(tracer, rows: list[tuple[int, dict]], outcomes: dict, slots: list[int], layers: list[int], states_only: bool, placement: str) -> dict:
    hidden_only = isinstance(tracer, HiddenTracer)
    keys = [
        "index", "correct", "gold", "h", "reference_h", "value", "reference_value",
        "entropy", "top1", "token_entropy", "token_log_prob", "tokens", "attention_share", "slot_match",
    ]
    if hidden_only:
        keys = ["index", "correct", "gold", "h", "slot_match"]
    store: dict[str, list] = {k: [] for k in keys if not (states_only and k.startswith("reference"))}
    for index, row in rows:
        record = outcomes[row["id"]]
        placed = arrange(row, index, slots, placement)
        gold = np.array([is_gold(d) for d in placed])
        prediction = record["prediction"].strip() or row["answer"]
        try:
            full = tracer.trace(row["question"], placed, prediction)
            singles = [] if states_only else [tracer.trace(row["question"], [document], prediction) for document in placed]
        except RuntimeError as error:
            print(f"skipped {index}: {error}", file=sys.stderr, flush=True)
            continue
        store["index"].append(index)
        store["correct"].append(bool(record["hit"]))
        store["gold"].append(gold)
        store["slot_match"].append([int(i) for i in np.flatnonzero(gold)] == [int(i) for i in record["gold_slots"]])
        store["h"].append(full["residual"][layers].astype(np.float16))
        if hidden_only:
            continue
        if states_only:
            store["value"].append(full["value_total"].sum(axis=0, keepdims=True).astype(np.float16))
        else:
            store["value"].append(full["value_total"].astype(np.float16))
            store["reference_h"].append(np.stack([s["residual"][layers] for s in singles]).astype(np.float16))
            store["reference_value"].append(np.stack([s["value_total"][0] for s in singles]).astype(np.float16))
        store["entropy"].append(full["entropy"])
        store["top1"].append(float(full["top_probs"][0]))
        store["token_entropy"].append(padded(full["token_entropy"]))
        store["token_log_prob"].append(padded(full["token_log_prob"]))
        store["tokens"].append(len(full["token_entropy"]))
        store["attention_share"].append((full["attention"][:, gold].sum(axis=1) / full["attention"].sum(axis=1)).astype(np.float32))
    return {key: np.asarray(values) for key, values in store.items()}


GROUPS = ("correct", "wrong")
GROUP_LABELS = {"correct": "output correct", "wrong": "output wrong"}
GROUP_COLORS = {"correct": "#2166AC", "wrong": "#B2182B"}
KINDS = ("gold", "false")
KIND_LABELS = {"gold": "gold documents", "false": "false documents"}
KIND_COLORS = {"gold": "#1B7837", "false": "#8C6D31"}


def load_parts(directory: Path) -> dict:
    parts = [np.load(path) for path in sorted(directory.glob("part-*.npz")) if ".tmp" not in path.name]
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files if key != "layers"}
    arrays["layers"] = parts[0]["layers"]
    return arrays


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy import stats

    ranks = stats.rankdata(scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def document_plane(references: np.ndarray, gold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = references.mean(axis=1, keepdims=True)
    centered = references - center
    weight = gold[..., None].astype(np.float32)
    gold_mean = (centered * weight).sum(axis=1) / weight.sum(axis=1)
    false_mean = (centered * (1 - weight)).sum(axis=1) / (1 - weight).sum(axis=1)
    first = unit((gold_mean - false_mean).mean(axis=0))
    flat = centered.reshape(-1, centered.shape[-1])
    flat = flat - np.outer(flat @ first, first)
    covariance = flat.T @ flat
    second = np.linalg.eigh(covariance)[1][:, -1]
    second = unit(second - (second @ first) * first)
    return first, second


def density(axis, points: np.ndarray, color: str, extent: tuple[float, float, float, float], rng) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde

    sample = points[rng.choice(len(points), size=min(len(points), 4000), replace=False)]
    xs = np.linspace(extent[0], extent[1], 140)
    ys = np.linspace(extent[2], extent[3], 140)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    values = gaussian_kde(sample.T)(grid.T).reshape(len(ys), len(xs))
    levels = np.quantile(values[values > values.max() * 0.02], [0.35, 0.6, 0.8, 0.93])
    levels = np.unique(np.concatenate([levels, [values.max()]]))
    shades = LinearSegmentedColormap.from_list("shade", ["#FFFFFF", color])
    axis.contourf(xs, ys, values, levels=levels, cmap=shades, alpha=0.55, vmin=0.0, vmax=values.max())
    axis.contour(xs, ys, values, levels=levels[:-1], colors=color, linewidths=0.5, alpha=0.9)


def figure_alignment(name: str, title: str, source: np.ndarray, references: np.ndarray, arrays: dict, out_dir: Path, summary: dict, source_label: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, panel_label, save

    rng = np.random.default_rng(0)
    correct = arrays["correct"]
    gold = arrays["gold"]
    source = unit(source.astype(np.float32))
    references = unit(references.astype(np.float32))
    first, second = document_plane(references, gold)
    center = references.mean(axis=1)
    plane = np.stack([first, second], axis=1)
    documents = (references - center[:, None, :]) @ plane
    states = (source - center) @ plane
    cosines = np.einsum("nd,nkd->nk", source, references)
    masks = {"correct": correct, "wrong": ~correct}
    kinds = {"gold": gold, "false": ~gold}
    everything = np.concatenate([documents.reshape(-1, 2), states])
    low = np.quantile(everything, 0.002, axis=0)
    high = np.quantile(everything, 0.998, axis=0)
    pad = 0.08 * (high - low)
    extent = (low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1])

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 5.6), sharex=True, sharey=True)
    report: dict = {}
    for row, group in enumerate(GROUPS):
        for column, kind in enumerate(KINDS):
            axis = axes[row, column]
            selected = masks[group]
            points = documents[selected][kinds[kind][selected]]
            state = states[selected]
            density(axis, points, KIND_COLORS[kind], extent, rng)
            shown = state[rng.choice(len(state), size=min(len(state), 1200), replace=False)]
            axis.scatter(shown[:, 0], shown[:, 1], s=2.2, color=GROUP_COLORS[group], alpha=0.35, linewidths=0, rasterized=True)
            axis.scatter(*points.mean(axis=0), marker="X", s=46, color=KIND_COLORS[kind], edgecolors="white", linewidths=0.7, zorder=5)
            axis.scatter(*state.mean(axis=0), marker="o", s=38, color=GROUP_COLORS[group], edgecolors="white", linewidths=0.7, zorder=6)
            similarity = np.where(kinds[kind][selected], cosines[selected], np.nan)
            mean_cosine = float(np.nanmean(similarity))
            distance = float(np.linalg.norm(points.mean(axis=0) - state.mean(axis=0)))
            report[f"{group}_vs_{kind}"] = {"mean_cosine": mean_cosine, "centroid_distance_in_plane": distance, "states": int(len(state)), "documents": int(len(points))}
            axis.text(0.03, 0.97, f"cos = {mean_cosine:.3f}\ncentroid distance = {distance:.3f}", transform=axis.transAxes, va="top", ha="left", fontsize=7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.5))
            axis.set_title(f"{source_label} ({GROUP_LABELS[group]}, n = {len(state)}) vs. {KIND_LABELS[kind]}", fontsize=7.8)
            axis.axhline(0, color="0.6", linewidth=0.4, zorder=0)
            axis.axvline(0, color="0.6", linewidth=0.4, zorder=0)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            if row == 1:
                axis.set_xlabel("Gold $-$ false document direction")
            if column == 0:
                axis.set_ylabel("First orthogonal principal component")
            panel_label(axis, f"({chr(ord('a') + row * 2 + column)})")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=4, color=GROUP_COLORS["correct"], label=f"{source_label}, output correct"),
        Line2D([], [], marker="o", linestyle="", markersize=4, color=GROUP_COLORS["wrong"], label=f"{source_label}, output wrong"),
        Patch(facecolor=KIND_COLORS["gold"], alpha=0.55, label="gold documents (density)"),
        Patch(facecolor=KIND_COLORS["false"], alpha=0.55, label="false documents (density)"),
        Line2D([], [], marker="X", linestyle="", markersize=5, color="0.3", label="document centroid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.6, bbox_to_anchor=(0.5, -0.005), handletextpad=0.3, columnspacing=1.0)
    fig.suptitle(title, fontsize=9, y=0.992)
    fig.tight_layout(rect=(0, 0.035, 1, 1.0), h_pad=1.2, w_pad=1.0)
    save(fig, out_dir, name)

    gold_cosine = np.nanmean(np.where(gold, cosines, np.nan), axis=1)
    false_cosine = np.nanmean(np.where(~gold, cosines, np.nan), axis=1)
    nearest = gold[np.arange(len(gold)), cosines.argmax(axis=1)]
    report["margin_gold_minus_false"] = {g: float((gold_cosine - false_cosine)[masks[g]].mean()) for g in GROUPS}
    report["auroc_margin_predicts_correct"] = auroc(gold_cosine - false_cosine, correct)
    report["auroc_direction_predicts_correct"] = auroc(states[:, 0], correct)
    report["nearest_document_is_gold"] = {g: float(nearest[masks[g]].mean()) for g in GROUPS}
    report["auroc_direction_separates_gold_from_false_documents"] = auroc(documents[..., 0].reshape(-1), gold.reshape(-1))
    summary[name] = report


def figure_before_after(name: str, title: str, after: dict, before: dict, position: int, out_dir: Path, source_label: str, source_key: str, reference_key: str, before_title: str, after_title: str) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, panel_label, save

    rng = np.random.default_rng(0)
    lookup = {int(index): row for row, index in enumerate(before["index"])}
    rows_after = np.array([row for row, index in enumerate(after["index"]) if int(index) in lookup and after["correct"][row]])
    rows_before = np.array([lookup[int(after["index"][row])] for row in rows_after])

    def select(arrays: dict, key: str, rows: np.ndarray) -> np.ndarray:
        values = arrays[key][rows]
        if key == "value":
            return values.astype(np.float32).sum(axis=1)
        return values[:, position].astype(np.float32) if key == "h" else values.astype(np.float32)

    gold = after["gold"][rows_after]
    references = after[reference_key][rows_after]
    references = unit((references[:, :, position] if reference_key == "reference_h" else references).astype(np.float32))
    states = {"before": unit(select(before, source_key, rows_before)), "after": unit(select(after, source_key, rows_after))}
    first, second = document_plane(references, gold)
    plane = np.stack([first, second], axis=1)
    center = references.mean(axis=1)
    documents = (references - center[:, None, :]) @ plane
    projected = {key: (value - center) @ plane for key, value in states.items()}
    cosines = {key: np.einsum("nd,nkd->nk", value, references) for key, value in states.items()}
    kinds = {"gold": gold, "false": ~gold}
    everything = np.concatenate([documents.reshape(-1, 2), projected["before"], projected["after"]])
    low = np.quantile(everything, 0.002, axis=0)
    high = np.quantile(everything, 0.998, axis=0)
    pad = 0.08 * (high - low)
    extent = (low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1])
    columns = (("before", before_title, "#6A3D9A"), ("after", after_title, GROUP_COLORS["correct"]))

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 5.6), sharex=True, sharey=True)
    report: dict = {"questions": int(len(rows_after))}
    for row, kind in enumerate(KINDS):
        for column, (stage, label, color) in enumerate(columns):
            axis = axes[row, column]
            points = documents[kinds[kind]]
            state = projected[stage]
            density(axis, points, KIND_COLORS[kind], extent, rng)
            shown = state[rng.choice(len(state), size=min(len(state), 1200), replace=False)]
            axis.scatter(shown[:, 0], shown[:, 1], s=2.2, color=color, alpha=0.35, linewidths=0, rasterized=True)
            axis.scatter(*points.mean(axis=0), marker="X", s=46, color=KIND_COLORS[kind], edgecolors="white", linewidths=0.7, zorder=5)
            axis.scatter(*state.mean(axis=0), marker="o", s=38, color=color, edgecolors="white", linewidths=0.7, zorder=6)
            mean_cosine = float(np.nanmean(np.where(kinds[kind], cosines[stage], np.nan)))
            distance = float(np.linalg.norm(points.mean(axis=0) - state.mean(axis=0)))
            report[f"{stage}_vs_{kind}"] = {"mean_cosine": mean_cosine, "centroid_distance_in_plane": distance}
            axis.text(0.03, 0.97, f"cos = {mean_cosine:.3f}\ncentroid distance = {distance:.3f}", transform=axis.transAxes, va="top", ha="left", fontsize=7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.5))
            axis.set_title(f"{source_label} vs. {KIND_LABELS[kind]}", fontsize=7.8)
            axis.axhline(0, color="0.6", linewidth=0.4, zorder=0)
            axis.axvline(0, color="0.6", linewidth=0.4, zorder=0)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            if row == 1:
                axis.set_xlabel("Gold $-$ false document direction")
            if column == 0:
                axis.set_ylabel("First orthogonal principal component")
            panel_label(axis, f"({chr(ord('a') + row * 2 + column)})")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=4, color=columns[0][2], label=f"{source_label}, {before_title[0].lower()}{before_title[1:].split(' (')[0]}"),
        Line2D([], [], marker="o", linestyle="", markersize=4, color=columns[1][2], label=f"{source_label}, {after_title[0].lower()}{after_title[1:]}"),
        Patch(facecolor=KIND_COLORS["gold"], alpha=0.55, label="gold documents (density)"),
        Patch(facecolor=KIND_COLORS["false"], alpha=0.55, label="false documents (density)"),
        Line2D([], [], marker="X", linestyle="", markersize=5, color="0.3", label="document centroid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.6, bbox_to_anchor=(0.5, -0.005), handletextpad=0.3, columnspacing=1.0)
    fig.suptitle(f"{title}, {len(rows_after)} questions answered correctly with AttnRank", fontsize=9, y=0.992)
    fig.tight_layout(rect=(0, 0.035, 1, 1.0), h_pad=1.2, w_pad=1.0)
    save(fig, out_dir, name)

    from scipy import stats

    margins = {}
    for stage in ("before", "after"):
        gold_cosine = np.nanmean(np.where(gold, cosines[stage], np.nan), axis=1)
        false_cosine = np.nanmean(np.where(~gold, cosines[stage], np.nan), axis=1)
        margins[stage] = gold_cosine - false_cosine
        report[f"{stage}_margin_gold_minus_false"] = float(margins[stage].mean())
        report[f"{stage}_direction"] = float(projected[stage][:, 0].mean())
    difference = margins["after"] - margins["before"]
    report["margin_change"] = {"mean": float(difference.mean()), "share_of_questions_increasing": float((difference > 0).mean()), "wilcoxon_p": float(stats.wilcoxon(difference).pvalue)}
    report["cosine_between_before_and_after_state"] = float((states["before"] * states["after"]).sum(axis=1).mean())
    if before["slot_match"].all():
        report["accuracy_before_on_these_questions"] = float(before["correct"][rows_before].mean())
    return report


def render_before_after(after: dict, before: dict, out_dir: Path, layer: int, before_title: str, after_title: str, name: str) -> dict:
    from attnrank.services.research.plot_attention_trace import apply_style

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    position = [int(v) for v in after["layers"]].index(layer)
    summary = {
        "layer": layer,
        "left_column": before_title,
        "right_column": after_title,
        name: figure_before_after(
            name, rf"$h_{{\mathrm{{last}}}}$ (layer {layer})",
            after, before, position, out_dir, r"$h_{\mathrm{last}}$", "h", "reference_h", before_title, after_title,
        ),
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def figure_entropy(arrays: dict, out_dir: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats
    from attnrank.services.research.placement_trace import kde
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, panel_label, save

    correct = arrays["correct"]
    masks = {"correct": correct, "wrong": ~correct}
    styles = {"correct": "-", "wrong": "--"}
    first = arrays["entropy"].astype(np.float64)
    tokens = arrays["token_entropy"].astype(np.float64)
    window = 8
    mean = np.nanmean(tokens[:, :window], axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_WIDTH, 2.6))
    report: dict = {}
    for axis, values, label, key in ((axes[0], first, "Entropy at the first answer token (nats)", "first_token"), (axes[1], mean, f"Mean entropy over the first {window} answer tokens (nats)", f"mean_first_{window}_tokens")):
        logs = np.log10(np.clip(values, 1e-4, None))
        grid = np.linspace(logs.min(), logs.max() + 0.1, 300)
        for group in GROUPS:
            axis.plot(grid, kde(logs[masks[group]], grid), color=GROUP_COLORS[group], linestyle=styles[group], label=f"{GROUP_LABELS[group]} (n = {int(masks[group].sum())})")
            axis.fill_between(grid, kde(logs[masks[group]], grid), color=GROUP_COLORS[group], alpha=0.12, linewidth=0)
            axis.axvline(np.median(logs[masks[group]]), color=GROUP_COLORS[group], linestyle=":", linewidth=0.8)
        ticks = [t for t in (-4, -3, -2, -1, 0, 1) if grid[0] <= t <= grid[-1]]
        axis.set_xticks(ticks, [f"$10^{{{t}}}$" for t in ticks])
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        test = stats.mannwhitneyu(values[masks["wrong"]], values[masks["correct"]], alternative="greater")
        score = auroc(-values, correct)
        report[key] = {
            "mean": {g: float(values[masks[g]].mean()) for g in GROUPS},
            "median": {g: float(np.median(values[masks[g]])) for g in GROUPS},
            "auroc_low_entropy_predicts_correct": score,
            "mann_whitney_p": float(test.pvalue),
        }
        axis.text(0.03, 0.97, f"AUROC = {score:.3f}\nmedian {np.median(values[masks['correct']]):.3f} vs. {np.median(values[masks['wrong']]):.3f}", transform=axis.transAxes, va="top", fontsize=6.8, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.5))
    axes[0].set_title("First-token uncertainty")
    axes[1].set_title("Uncertainty over the answer")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.6, ncol=1)

    axis = axes[2]
    positions = np.arange(1, tokens.shape[1] + 1)
    for group in GROUPS:
        block = tokens[masks[group]]
        average = np.nanmean(block, axis=0)
        error = 1.96 * np.nanstd(block, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(block), axis=0))
        axis.plot(positions, average, color=GROUP_COLORS[group], linestyle=styles[group], marker="o", markersize=2.4, label=GROUP_LABELS[group])
        axis.fill_between(positions, average - error, average + error, color=GROUP_COLORS[group], alpha=0.18, linewidth=0)
    axis.set_xticks([1, 4, 8, 12, 16])
    axis.set_xlabel("Answer token position")
    axis.set_ylabel("Mean entropy (nats)")
    axis.set_title("Entropy along the generated answer")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(loc="upper right", fontsize=6.6)
    for index, axis in enumerate(axes):
        panel_label(axis, f"({chr(ord('a') + index)})")
    fig.tight_layout(w_pad=1.2)
    save(fig, out_dir, "fig_correct_entropy")
    report["top1_probability"] = {**{g: float(arrays["top1"][masks[g]].mean()) for g in GROUPS}, "auroc": auroc(arrays["top1"], correct)}
    summary["fig_correct_entropy"] = report


def answer_bearing(arrays: dict, dataset: Path, slots: list[int]) -> np.ndarray:
    from attnrank.utils.text import normalize

    with dataset.open() as handle:
        rows = [json.loads(line) for line in handle]
    bearing = np.zeros_like(arrays["gold"])
    for position, index in enumerate(arrays["index"]):
        row = rows[int(index)]
        ranked = relevance_order(question=row["question"], documents=row["documents"], ordering="bm25")
        placed = place(ranked, slots)
        answer = normalize(text=row["answer"])
        bearing[position] = [is_gold(d) and answer in normalize(text=d["text"]) for d in placed]
    return bearing


def figure_separability(arrays: dict, bearing: np.ndarray, position: int, out_dir: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt
    from attnrank.services.research.plot_attention_trace import COLUMN_WIDTH, save

    rng = np.random.default_rng(0)
    correct = arrays["correct"]
    gold = arrays["gold"]
    bridge = gold & ~bearing
    two_hop = bearing.any(axis=1) & bridge.any(axis=1)

    def cosines(source: np.ndarray, references: np.ndarray) -> np.ndarray:
        return np.einsum("nd,nkd->nk", unit(source.astype(np.float32)), unit(references.astype(np.float32)))

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, values, 0.0).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)

    hidden = cosines(arrays["h"][:, position], arrays["reference_h"][:, :, position])
    value = cosines(arrays["value"].astype(np.float32).sum(axis=1), arrays["reference_value"])
    everyone = np.ones(len(correct), dtype=bool)
    measures = [
        (r"$h_{\mathrm{last}}$: cos(gold) $-$ cos(false)", masked_mean(hidden, gold) - masked_mean(hidden, ~gold), everyone, "alignment"),
        (r"value: cos(gold) $-$ cos(false)", masked_mean(value, gold) - masked_mean(value, ~gold), everyone, "alignment"),
        (r"$h_{\mathrm{last}}$: cos(answer doc) $-$ cos(bridge doc)", masked_mean(hidden, bearing) - masked_mean(hidden, bridge), two_hop, "alignment"),
        (r"value: cos(answer doc) $-$ cos(bridge doc)", masked_mean(value, bearing) - masked_mean(value, bridge), two_hop, "alignment"),
        ("attention share on gold (mean over layers)", arrays["attention_share"].mean(axis=1), everyone, "attention"),
        ("top-1 probability, first token", arrays["top1"], everyone, "output"),
        ("$-$ entropy, first token", -arrays["entropy"], everyone, "output"),
        ("$-$ mean entropy, first 8 tokens", -np.nanmean(arrays["token_entropy"][:, :8], axis=1), everyone, "output"),
    ]
    palette = {"alignment": "#4C72B0", "attention": "#8172B3", "output": "#DD8452"}
    report = {}
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH * 1.35, 2.9))
    for row, (label, scores, mask, family) in enumerate(measures):
        scores = np.asarray(scores, dtype=np.float64)[mask]
        labels = correct[mask]
        point = auroc(scores, labels)
        samples = []
        for _ in range(300):
            draw = rng.integers(0, len(scores), len(scores))
            samples.append(auroc(scores[draw], labels[draw]))
        low, high = np.quantile(samples, [0.025, 0.975])
        report[label] = {"auroc": point, "ci95": [float(low), float(high)], "questions": int(mask.sum())}
        y = len(measures) - 1 - row
        axis.errorbar(point, y, xerr=[[point - low], [high - point]], fmt="o", color=palette[family], markersize=3.5, capsize=2, linewidth=0.9)
        axis.text(max(high, 0.5) + 0.004, y, f"{point:.3f}", va="center", fontsize=6.6)
    axis.axvline(0.5, color="0.4", linewidth=0.7, linestyle="--")
    axis.set_yticks(range(len(measures)), [m[0] for m in measures][::-1], fontsize=6.8)
    axis.set_xlabel("AUROC for predicting a correct output (0.5 = chance)")
    axis.set_xlim(0.44, 0.62)
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    axis.set_title("Which signal separates correct from wrong outputs")
    save(fig, out_dir, "fig_correct_separability")
    summary["fig_correct_separability"] = report


def render(arrays: dict, out_dir: Path, layer: int, dataset: Path, slots: list[int]) -> dict:
    from attnrank.services.research.plot_attention_trace import apply_style

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(v) for v in arrays["layers"]]
    position = layers.index(layer)
    correct = arrays["correct"]
    summary: dict = {
        "questions": int(len(correct)),
        "output_correct": int(correct.sum()),
        "output_wrong": int((~correct).sum()),
        "layer": layer,
        "gold_slots_match_evaluated_run": float(arrays["slot_match"].mean()),
    }
    figure_alignment(
        "fig_correct_hlast_vs_documents", rf"Hidden state at the answer position, $h_{{\mathrm{{last}}}}$ (layer {layer})",
        arrays["h"][:, position], arrays["reference_h"][:, :, position], arrays, out_dir, summary, r"$h_{\mathrm{last}}$",
    )
    figure_alignment(
        "fig_correct_value_vs_documents", r"Attention value output written to the answer position, $\sum_{\ell}\sum_{j\in D}\alpha_{j}\,W_{O}v_{j}$",
        arrays["value"].astype(np.float32).sum(axis=1), arrays["reference_value"], arrays, out_dir, summary, "value output",
    )
    figure_entropy(arrays, out_dir, summary)
    figure_separability(arrays, answer_bearing(arrays, dataset, slots), position, out_dir, summary)
    (out_dir / "correctness_trace_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hidden states, value outputs and output entropy of HotpotQA prompts, split by whether the recorded output was correct")
    parser.add_argument("--model", default=str(workspace_path(relative="models/qwen2.5-7b-instruct")))
    parser.add_argument("--profile", default=str(workspace_path(relative="profiles/profile-qwen7b-hotpotqa-fig5.json")))
    parser.add_argument("--dataset", default="data/hotpotqa-attnrank/dataset.jsonl")
    parser.add_argument("--outcomes", nargs="+", default=None)
    parser.add_argument("--strategy", default="attnrank")
    parser.add_argument("--layers", default="20,27")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default=str(workspace_path(relative="outputs/correctness_trace")))
    parser.add_argument("--figures", default="docs/figures")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--placement", choices=("attnrank", "descending", "random", "without-gold"), default="attnrank")
    parser.add_argument("--before-title", default="Without AttnRank (random order)")
    parser.add_argument("--after-title", default="With AttnRank")
    parser.add_argument("--figure-name", default="fig_hlast_before_after_attnrank")
    parser.add_argument("--states-only", action="store_true")
    parser.add_argument("--hidden-only", action="store_true")
    parser.add_argument("--before-dir", default=str(workspace_path(relative="outputs/correctness_trace_random")))
    parser.add_argument("--plot-before-after", action="store_true")
    args = parser.parse_args(argv)

    if args.plot_before_after:
        summary = render_before_after(load_parts(Path(args.out_dir)), load_parts(Path(args.before_dir)), Path(args.figures), args.layer, args.before_title, args.after_title, args.figure_name)
        print(json.dumps(summary, indent=1))
        return 0
    if args.plot_only:
        order = [int(v) for v in json.loads(Path(args.profile).read_text())["position_order"]]
        summary = render(load_parts(Path(args.out_dir)), Path(args.figures), args.layer, Path(args.dataset), resolve_order(order, 5))
        print(json.dumps(summary, indent=1))
        return 0

    paths = [Path(p) for p in args.outcomes] if args.outcomes else sorted(workspace_path(relative="outputs").glob("qwen7b-fig5-bm25-shard*.jsonl"))
    outcomes = load_outcomes(paths, args.strategy)
    rows = []
    with open(args.dataset) as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row["id"] in outcomes and len(row["documents"]) == 5:
                rows.append((index, row))
    if args.limit:
        rows = rows[: args.limit]
    blocks = [rows[start : start + BLOCK] for start in range(0, len(rows), BLOCK)]
    shard, count = (int(v) for v in args.shard.split("/"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [(number, block) for number, block in enumerate(blocks) if number % count == shard and not (out_dir / f"part-{number:04d}.npz").exists()]
    print(f"shard {shard}/{count}: {len(pending)} blocks pending of {len(blocks)}", file=sys.stderr, flush=True)
    if not pending:
        return 0

    profile = json.loads(Path(args.profile).read_text())
    slots = resolve_order([int(v) for v in profile["position_order"]], 5) if args.placement == "attnrank" else list(range(5))
    layers = [int(v) for v in args.layers.split(",")]
    tracer = HiddenTracer(args.model, args.device, args.dtype) if args.hidden_only else LayerTracer(args.model, args.device, args.dtype)
    for done, (number, block) in enumerate(pending):
        arrays = trace_block(tracer, block, outcomes, slots, layers, args.states_only or args.hidden_only, args.placement)
        arrays["layers"] = np.asarray(layers)
        temporary = out_dir / f"part-{number:04d}.tmp.npz"
        np.savez(temporary, **arrays)
        temporary.replace(out_dir / f"part-{number:04d}.npz")
        print(f"shard {shard}: block {done + 1}/{len(pending)} questions {len(arrays['index'])} correct {int(arrays['correct'].sum())} slot match {int(arrays['slot_match'].sum())}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
