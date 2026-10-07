from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

STRATEGIES = ("descending", "ascending", "lim", "attnrank", "random")
LABELS = {"descending": "Descending", "ascending": "Ascending", "lim": "LIM", "attnrank": "AttnRank", "random": "Random"}
COLORS = {"descending": "#4C72B0", "ascending": "#DD8452", "lim": "#8172B3", "attnrank": "#55A868", "random": "#7F7F7F"}
MARKERS = {"descending": "o", "ascending": "s", "lim": "D", "attnrank": "^", "random": "x"}
LINESTYLES = {"descending": "-", "ascending": "--", "lim": "-.", "attnrank": "-", "random": ":"}
COLUMN_WIDTH = 3.4
SEQUENTIAL = "YlGnBu"
DOUBLE_WIDTH = 7.0


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(axis, text: str) -> None:
    axis.text(-0.12, 1.06, text, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(figure, out_dir: Path, name: str) -> None:
    figure.savefig(out_dir / f"{name}.png")
    plt.close(figure)


def layer_ticks(axis, layers: int, step: int = 5) -> None:
    ticks = list(range(0, layers, step))
    if ticks[-1] != layers - 1:
        ticks.append(layers - 1)
    axis.set_yticks(ticks)


def figure_position_basin(summary: dict, out_dir: Path) -> None:
    position = np.asarray(summary["position_attention_random_order"])
    layers, slots = position.shape
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    im = ax.imshow(position, aspect="auto", cmap=SEQUENTIAL, origin="lower", interpolation="nearest")
    ax.set_xticks(range(slots), [str(i + 1) for i in range(slots)])
    layer_ticks(ax, layers)
    ax.set_xlabel("Document slot")
    ax.set_ylabel("Layer")
    ax.set_title("Attention by slot (random order)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Normalized attention", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    ax = axes[1]
    shallow = position[: max(1, layers // 4)].mean(axis=0)
    middle = position[layers // 4 : 3 * layers // 4].mean(axis=0)
    deep = position[3 * layers // 4 :].mean(axis=0)
    x = np.arange(1, slots + 1)
    ax.plot(x, shallow, marker="o", color="#4C72B0", label=f"layers 0–{layers // 4 - 1}")
    ax.plot(x, middle, marker="s", linestyle="--", color="#55A868", label=f"layers {layers // 4}–{3 * layers // 4 - 1}")
    ax.plot(x, deep, marker="^", linestyle="-.", color="#C44E52", label=f"layers {3 * layers // 4}–{layers - 1}")
    ax.axhline(1.0 / slots, color="0.5", linewidth=0.7, linestyle=":", label=f"uniform (1/{slots})")
    ax.set_xticks(x)
    ax.set_xlabel("Document slot")
    ax.set_ylabel("Normalized attention")
    ax.set_title("Attention basin per depth band")
    ax.legend(loc="upper center")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, out_dir, "fig_attention_basin")


def figure_attention_vs_accuracy(summary: dict, out_dir: Path) -> None:
    correlation = np.asarray(summary["corr_gold_share_vs_hit_by_layer"])
    layers = correlation.shape[0]
    quintiles = {int(k): v for k, v in summary["accuracy_by_gold_share_quintile"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.6))
    ax = axes[0]
    ax.bar(range(layers), correlation, color=np.where(correlation < 0, "#C44E52", "#4C72B0"), width=0.8, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson $r$")
    ax.set_title("Attention share on gold vs. correctness")
    ax.set_xticks(list(range(0, layers, 5)) + [layers - 1])

    ax = axes[1]
    palette = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"][: len(quintiles)]
    for color, (layer, values) in zip(palette, sorted(quintiles.items())):
        ax.plot(range(1, 6), values, marker="o", color=color, label=f"layer {layer}")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Quintile of attention share on gold documents")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_title("Accuracy vs. attention on gold")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28), handlelength=1.6, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, out_dir, "fig_attention_vs_accuracy")


def figure_heatmap_by_strategy(summary: dict, out_dir: Path) -> None:
    strategies = [s for s in STRATEGIES if f"mean_attention_{s}" in summary]
    if not strategies:
        return
    maps = [np.asarray(summary[f"mean_attention_{s}"]) for s in strategies]
    layers, slots = maps[0].shape
    vmin = min(m.min() for m in maps)
    vmax = max(m.max() for m in maps)
    fig, axes = plt.subplots(1, len(strategies), figsize=(DOUBLE_WIDTH, 2.9), sharey=True)
    for index, (ax, strategy, block) in enumerate(zip(axes, strategies, maps)):
        im = ax.imshow(block, aspect="auto", cmap=SEQUENTIAL, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        gold = summary.get(f"gold_slots_{strategy}", [])
        for slot in gold:
            ax.add_patch(Rectangle((slot - 0.5, -0.5), 1, layers, fill=False, edgecolor="#C44E52", linewidth=1.2, linestyle="--"))
        ax.set_xticks(range(slots), [str(i + 1) for i in range(slots)])
        ax.set_xlabel("Document slot")
        ax.set_title(LABELS[strategy], pad=4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        ax.text(0.0, 1.06, f"({chr(ord('a') + index)})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    layer_ticks(axes[0], layers)
    axes[0].set_ylabel("Layer")
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("Normalized attention", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    save(fig, out_dir, "fig_attention_by_strategy")


def render(summary_path: Path, out_dir: Path) -> list[str]:
    apply_style()
    summary = json.loads(summary_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_position_basin(summary, out_dir)
    figure_attention_vs_accuracy(summary, out_dir)
    figure_heatmap_by_strategy(summary, out_dir)
    return sorted(p.name for p in out_dir.glob("fig_*.png"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render publication-style figures from attention_trace_summary.json")
    parser.add_argument("--summary", default="docs/figures/attention_trace_summary.json")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args(argv)
    for name in render(Path(args.summary), Path(args.out_dir)):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
