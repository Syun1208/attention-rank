from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from attnrank.utils.workspace import workspace_path

INSTRUCTION = (
    "Write a high-quality answer for the given question using only the provided search "
    "results (some of which might be irrelevant)."
)
CONDITIONS = ("factual", "false")
LABELS = {"factual": "gold at AttnRank slots", "false": "gold at least-attended slots"}
COLORS = {"factual": "#4C72B0", "false": "#C44E52"}
STYLES = {"factual": "-", "false": "--"}
TOP_K = 20


def is_gold(document: dict) -> bool:
    return str(document.get("is_gold", "")).lower() == "true"


def resolve_order(preferred: list[int], count: int) -> list[int]:
    taken = [False] * count
    order = []
    for slot in preferred:
        if 0 <= slot < count and not taken[slot]:
            taken[slot] = True
            order.append(slot)
    order.extend(slot for slot in range(count) if not taken[slot])
    return order


def place(ranked: list[dict], slots: list[int]) -> list[dict]:
    placed: list = [None] * len(ranked)
    for rank, slot in enumerate(slots):
        placed[slot] = ranked[rank]
    return placed


def build_prompt(question: str, documents: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    text = INSTRUCTION + "\n\n"
    spans = []
    for index, document in enumerate(documents):
        segment = f"Document [{index + 1}] {document['text']}"
        spans.append((len(text), len(text) + len(segment)))
        text += segment + ("\n" if index + 1 < len(documents) else "\n\n")
    text += f"Question: {question}\nAnswer:"
    return text, spans


def token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]
    return indices[0], indices[-1] + 1


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


class LayerTracer:
    def __init__(self, model_dir: str, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=getattr(torch, dtype), device_map=device, attn_implementation="eager"
        )
        self.model.eval()
        self.layers = self.model.model.layers
        config = self.model.config
        self.hidden = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.group = self.heads // self.kv_heads
        self.device = next(self.model.parameters()).device
        self.values: dict[int, object] = {}
        self.row = 0
        self.spans: list[tuple[int, int]] = []
        self.target = None
        self.attention = None
        self.value_norm = None
        self.value_logit = None
        self.value_total = None
        self.final_input = None
        for index, layer in enumerate(self.layers):
            layer.self_attn.v_proj.register_forward_hook(self._value_hook(index))
            layer.self_attn.register_forward_hook(self._attention_hook(index))
        self.model.model.norm.register_forward_pre_hook(self._norm_hook)

    def _norm_hook(self, module, inputs):
        self.final_input = inputs[0][0, self.row].detach().float()

    def _value_hook(self, index: int):
        def hook(module, inputs, output):
            self.values[index] = output[0].detach()

        return hook

    def _attention_hook(self, index: int):
        torch = self.torch

        def hook(module, inputs, output):
            weights = None
            candidates = output if isinstance(output, (tuple, list)) else (output,)
            for candidate in candidates:
                if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                    weights = candidate
            if weights is None:
                raise RuntimeError("attention weights unavailable; eager attention is required")
            alpha = weights[0, :, self.row, :].float()
            values = self.values.pop(index).float()
            values = values.view(values.shape[0], self.kv_heads, self.head_dim).repeat_interleave(self.group, dim=1)
            for slot, (begin, end) in enumerate(self.spans):
                mixed = torch.einsum("hj,jhd->hd", alpha[:, begin:end], values[begin:end])
                projected = module.o_proj(mixed.reshape(1, -1).to(module.o_proj.weight.dtype))[0].float()
                self.attention[index, slot] = alpha[:, begin:end].sum(dim=1).mean()
                self.value_norm[index, slot] = projected.norm()
                self.value_logit[index, slot] = projected @ self.target
                self.value_total[slot] += projected

        return hook

    def trace(self, question: str, documents: list[dict], answer: str) -> dict:
        torch = self.torch
        prompt_text, char_spans = build_prompt(question, documents)
        full_text = prompt_text + " " + answer
        encoded = self.tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"][0].tolist()]
        prompt_tokens = sum(1 for _, b in offsets if b <= len(prompt_text))
        if offsets[prompt_tokens - 1][1] != len(prompt_text):
            raise RuntimeError("answer tokens merged with the prompt boundary")
        input_ids = encoded["input_ids"].to(self.device)
        answer_ids = input_ids[0, prompt_tokens:]
        self.row = prompt_tokens - 1
        self.spans = [token_span(offsets, a, b) for a, b in char_spans]
        norm = self.model.model.norm
        self.target = (norm.weight.float() * self.model.lm_head.weight[answer_ids[0]].float()).detach()
        layers, count = len(self.layers), len(documents)
        self.attention = torch.zeros(layers, count, device=self.device)
        self.value_norm = torch.zeros(layers, count, device=self.device)
        self.value_logit = torch.zeros(layers, count, device=self.device)
        self.value_total = torch.zeros(count, self.hidden, device=self.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        scale = torch.sqrt(self.final_input.pow(2).mean() + getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
        residual = torch.stack([state[0, self.row].float() for state in output.hidden_states[1:]])
        logits = output.logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        positions = torch.arange(self.row, self.row + answer_ids.shape[0], device=self.device)
        answer_log_probs = log_probs[positions, answer_ids]
        token_entropy = -(torch.exp(log_probs[positions]) * log_probs[positions]).sum(dim=-1)
        first = torch.softmax(logits[self.row], dim=-1)
        return {
            "residual": residual.cpu().numpy(),
            "value_total": self.value_total.cpu().numpy(),
            "attention": self.attention.cpu().numpy(),
            "value_norm": self.value_norm.cpu().numpy(),
            "value_logit": (self.value_logit / scale).cpu().numpy(),
            "gold": np.array([is_gold(d) for d in documents]),
            "answer_log_prob": float(answer_log_probs.mean()),
            "token_log_prob": answer_log_probs.cpu().numpy(),
            "token_entropy": token_entropy.cpu().numpy(),
            "entropy": float(-(first * torch.log(first.clamp_min(1e-12))).sum()),
            "top_probs": torch.sort(first, descending=True).values[:TOP_K].cpu().numpy(),
        }


def gold_margin(source: np.ndarray, references: np.ndarray, gold: np.ndarray) -> np.ndarray:
    references = unit(references)
    center = references.mean(axis=0)
    gold_direction = (references[gold] - center).mean(axis=0)
    false_direction = (references[~gold] - center).mean(axis=0)
    centered = unit(source) - center
    return cosine(centered, gold_direction) - cosine(centered, false_direction)


def collect(args) -> dict:
    profile = json.loads(Path(args.profile).read_text())
    order = [int(v) for v in profile["position_order"]]
    tracer = LayerTracer(args.model, args.device, args.dtype)
    rows = []
    with open(args.dataset) as handle:
        for line in handle:
            row = json.loads(line)
            if len(row["documents"]) == 5 and sum(is_gold(d) for d in row["documents"]) == 2:
                rows.append(row)
            if len(rows) >= args.questions:
                break
    slots = {"factual": resolve_order(order, 5), "false": resolve_order(order[::-1], 5)}
    keys = ("attention_share", "value_share", "logit_gold", "logit_false", "residual_margin", "value_margin", "answer_log_prob", "entropy", "top_probs")
    store: dict = {c: {k: [] for k in keys} for c in CONDITIONS}
    for index, row in enumerate(rows):
        ranked = [d for d in row["documents"] if is_gold(d)] + [d for d in row["documents"] if not is_gold(d)]
        ranked_gold = np.array([is_gold(d) for d in ranked])
        singles = [tracer.trace(row["question"], [document], row["answer"]) for document in ranked]
        reference_residual = np.stack([s["residual"] for s in singles])
        reference_value = np.stack([s["value_total"][0] for s in singles])
        for condition in CONDITIONS:
            traced = tracer.trace(row["question"], place(ranked, slots[condition]), row["answer"])
            gold = traced["gold"]
            bucket = store[condition]
            bucket["attention_share"].append(traced["attention"][:, gold].sum(axis=1) / traced["attention"].sum(axis=1))
            bucket["value_share"].append(traced["value_norm"][:, gold].sum(axis=1) / traced["value_norm"].sum(axis=1))
            bucket["logit_gold"].append(traced["value_logit"][:, gold].sum(axis=1))
            bucket["logit_false"].append(traced["value_logit"][:, ~gold].sum(axis=1))
            bucket["residual_margin"].append(gold_margin(traced["residual"], reference_residual, ranked_gold))
            bucket["value_margin"].append(float(gold_margin(traced["value_total"].sum(axis=0), reference_value, ranked_gold)))
            for key in ("answer_log_prob", "entropy", "top_probs"):
                bucket[key].append(traced[key])
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(rows)}", file=sys.stderr, flush=True)
    arrays = {f"{c}_{k}": np.asarray(v) for c in CONDITIONS for k, v in store[c].items()}
    arrays["slots_factual"] = np.asarray(slots["factual"])
    arrays["slots_false"] = np.asarray(slots["false"])
    return arrays


def effect_size(difference: np.ndarray) -> np.ndarray:
    return difference.mean(axis=0) / (difference.std(axis=0, ddof=1) + 1e-12)


def bootstrap_interval(difference: np.ndarray, rng, draws: int = 1000) -> tuple[float, float]:
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    samples = difference[indices]
    values = samples.mean(axis=1) / (samples.std(axis=1, ddof=1) + 1e-12)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def figure_effect(arrays: dict, out_dir: Path, summary: dict, profile_layer: int) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats
    from attnrank.services.research.plot_attention_trace import DOUBLE_WIDTH, panel_label, save

    rng = np.random.default_rng(0)

    def paired(key: str) -> np.ndarray:
        return arrays[f"factual_{key}"] - arrays[f"false_{key}"]

    layers = arrays["factual_attention_share"].shape[1]
    curves = (
        ("attention_share", "attention on gold documents", "#4C72B0", "-"),
        ("value_share", "value output from gold documents", "#55A868", "--"),
        ("residual_margin", "hidden state alignment with gold", "#8172B3", "-."),
    )
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 3.1), gridspec_kw={"width_ratios": [1.0, 1.25]})
    axis = axes[0]
    for key, label, color, style in curves:
        difference = paired(key)
        axis.plot(np.arange(layers), effect_size(difference), color=color, linestyle=style, label=label)
        tests = [stats.wilcoxon(difference[:, layer]).pvalue for layer in range(layers)]
        summary[f"by_layer:{key}"] = {
            "mean_difference": [float(v) for v in difference.mean(axis=0)],
            "effect_size": [float(v) for v in effect_size(difference)],
            "significant_layers_p<0.01": [int(layer) for layer in range(layers) if tests[layer] < 0.01 and difference[:, layer].mean() > 0],
        }
    axis.axhline(0.0, color="black", linewidth=0.5)
    axis.set_xlabel("Layer")
    axis.set_xticks(list(range(0, layers, 5)) + [layers - 1])
    axis.set_ylabel("Effect of placement (standardised, $d_z$)")
    axis.set_title("Effect by layer")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(loc="upper right", fontsize=6.5)

    stages = (
        (f"attention on gold, layer {profile_layer}", paired("attention_share")[:, profile_layer], "input"),
        ("attention on gold, all layers", paired("attention_share").mean(axis=1), "input"),
        ("value output from gold, all layers", paired("value_share").mean(axis=1), "input"),
        ("hidden state alignment with gold, last layer", paired("residual_margin")[:, -1], "state"),
        ("logit attribution of gold to $y^{*}$", paired("logit_gold").sum(axis=1), "state"),
        ("top-1 probability, first token", arrays["factual_top_probs"][:, 0] - arrays["false_top_probs"][:, 0], "output"),
        ("certainty ($-$entropy), first token", -(paired("entropy")), "output"),
        (r"$\log P(y^{*}\mid x, D)$", paired("answer_log_prob"), "output"),
    )
    palette = {"input": "#4C72B0", "state": "#8172B3", "output": "#DD8452"}
    names = {"input": "what the model reads", "state": "what it computes", "output": "what it outputs"}
    axis = axes[1]
    report = {}
    for row, (label, difference, family) in enumerate(stages):
        difference = np.asarray(difference, dtype=np.float64)
        point = float(effect_size(difference))
        low, high = bootstrap_interval(difference, rng)
        y = len(stages) - 1 - row
        axis.barh(y, point, color=palette[family], alpha=0.85, height=0.62, xerr=[[point - low], [high - point]], error_kw={"linewidth": 0.8, "capsize": 2, "ecolor": "0.2"})
        axis.text(max(high, 0.0) + 0.12, y, f"{point:+.2f}", va="center", fontsize=6.8)
        report[label] = {"effect_size": point, "ci95": [low, high], "mean_difference": float(difference.mean()), "wilcoxon_p": float(stats.wilcoxon(difference).pvalue)}
    axis.axvline(0.0, color="black", linewidth=0.6)
    axis.set_yticks(range(len(stages)), [stage[0] for stage in stages][::-1], fontsize=6.8)
    axis.set_xlabel("Effect of placement (standardised, $d_z$)")
    axis.set_title("Effect along the pipeline")
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    limit = axis.get_xlim()
    axis.set_xlim(min(limit[0], -0.3), limit[1] + 0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k], alpha=0.85) for k in palette]
    axis.legend(handles, [names[k] for k in palette], loc="lower right", fontsize=6.5)
    panel_label(axes[0], "(a)")
    axis.text(-0.62, 1.06, "(b)", transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    fig.suptitle("Gold documents at the AttnRank slots versus the least-attended slots (paired, 500 questions)".replace("500", str(len(arrays["factual_entropy"]))), fontsize=8.5, y=0.99)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 1.02))
    save(fig, out_dir, "fig_placement_effect")
    summary["pipeline_effect"] = report


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = values.std(ddof=1)
    bandwidth = 1.06 * spread * len(values) ** (-0.2) if spread > 0 else 1.0
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))


def figure_output_distribution(arrays: dict, out_dir: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats
    from attnrank.services.research.plot_attention_trace import panel_label, save

    log_prob = {c: arrays[f"{c}_answer_log_prob"] for c in CONDITIONS}
    entropy = {c: arrays[f"{c}_entropy"] for c in CONDITIONS}
    top = {c: arrays[f"{c}_top_probs"] for c in CONDITIONS}
    summary["entropy"] = {**{c: float(entropy[c].mean()) for c in CONDITIONS}, "wilcoxon_p": float(stats.wilcoxon(entropy["factual"], entropy["false"]).pvalue)}
    summary["top1_probability"] = {**{c: float(top[c][:, 0].mean()) for c in CONDITIONS}, "wilcoxon_p": float(stats.wilcoxon(top["factual"][:, 0], top["false"][:, 0]).pvalue)}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax = axes[0]
    grid = np.linspace(min(v.min() for v in log_prob.values()), 0.0, 300)
    for condition in CONDITIONS:
        ax.plot(grid, kde(log_prob[condition], grid), color=COLORS[condition], linestyle=STYLES[condition], label=LABELS[condition])
        ax.axvline(np.median(log_prob[condition]), color=COLORS[condition], linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"mean token $\log P(y^{*}\mid x, D)$")
    ax.set_ylabel("Density")
    ax.set_title("Likelihood of the correct answer")
    ax.legend(loc="upper left", fontsize=6.3)

    ax = axes[1]
    ranks = np.arange(1, top["factual"].shape[1] + 1)
    for condition in CONDITIONS:
        logs = np.log10(np.clip(top[condition], 1e-12, None))
        mean = logs.mean(axis=0)
        error = logs.std(axis=0) / np.sqrt(logs.shape[0])
        ax.plot(ranks, 10**mean, color=COLORS[condition], linestyle=STYLES[condition], marker="o", markersize=2.5, label=LABELS[condition])
        ax.fill_between(ranks, 10 ** (mean - 1.96 * error), 10 ** (mean + 1.96 * error), color=COLORS[condition], alpha=0.2, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Token rank at the first answer step")
    ax.set_ylabel(r"$P(y\mid x, D)$")
    ax.set_title("Sorted softmax probabilities")

    ax = axes[2]
    grid = np.linspace(0.0, max(v.max() for v in entropy.values()) * 1.05, 300)
    for condition in CONDITIONS:
        ax.plot(grid, kde(entropy[condition], grid), color=COLORS[condition], linestyle=STYLES[condition], label=LABELS[condition])
        ax.axvline(np.median(entropy[condition]), color=COLORS[condition], linestyle=":", linewidth=0.8)
    ax.set_xlabel("Entropy of the softmax (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty at the first answer step")

    for index, axis in enumerate(axes):
        panel_label(axis, f"({chr(ord('a') + index)})")
    fig.tight_layout(w_pad=1.2)
    save(fig, out_dir, "fig_output_distribution")


def render(arrays: dict, out_dir: Path, profile_layer: int) -> dict:
    from attnrank.services.research.plot_attention_trace import apply_style

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "questions": int(arrays["factual_answer_log_prob"].shape[0]),
        "slots_factual": arrays["slots_factual"].tolist(),
        "slots_false": arrays["slots_false"].tolist(),
    }
    figure_effect(arrays, out_dir, summary, profile_layer)
    figure_output_distribution(arrays, out_dir, summary)
    (out_dir / "placement_trace_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer-wise effect of placing the gold documents at the AttnRank slots versus the least-attended slots")
    parser.add_argument("--model", default=str(workspace_path(relative="models/qwen2.5-7b-instruct")))
    parser.add_argument("--profile", default=str(workspace_path(relative="profiles/profile-qwen7b-hotpotqa-fig5.json")))
    parser.add_argument("--dataset", default=str(workspace_path(relative="data/hotpotqa-attnrank/dataset.jsonl")))
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--cache", default=str(workspace_path(relative="outputs/placement_trace.npz")))
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)

    if args.plot_only:
        arrays = dict(np.load(args.cache))
    else:
        arrays = collect(args)
        np.savez_compressed(args.cache, **arrays)
    summary = render(arrays, Path(args.out_dir), int(json.loads(Path(args.profile).read_text())["layer_index"]))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_layer")}, indent=1))
    for key, value in summary.items():
        if key.startswith("by_layer"):
            print(key, "layers with factual > false at p<0.01:", value["significant_layers_p<0.01"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
