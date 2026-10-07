# AttnRank

## Overview

A from-scratch C++/CUDA implementation of AttnRank, the two-stage, training-free reranking method of *Attention Basin: Why Contextual Position Matters in Large Language Models* (Yi et al., 2025), packaged as the Python library `attnrank`. The engine reads Hugging Face checkpoints directly (safetensors and PyTorch `.bin`), runs the Llama-family forward pass on cuBLAS, measures query-to-document attention per layer, picks the shallowest attention-basin layer, and places your top-k documents into the slots the model attends to most.

## Highlights

- No ML framework at runtime: CUDA, cuBLAS and a C++17 compiler, with a pybind11 module `attnrank._core`.
- `pip install attnrank` builds the engine and installs the `attnrank` package, the C++ CLI `attnrank` and the Python entry point `attnrank-run`.
- Works on top-k documents you already have (any retriever or none): `rerank_top_k` only needs the saved attention profile, the model is needed only to answer.
- Models load from a local directory or a Hugging Face repo id (`from_pretrained`), datasets from JSONL or the Hugging Face Hub with a column mapping or a custom adapter.
- One entry point `main.py` with subcommands `profile`, `rerank`, `hotpotqa`, `hotpotqa-report`, `finqa` and `research`, YAML configs in `configs/`, logs in `logs/`, tqdm progress bars.

## Installation

Requirements: Linux, Python 3.10+, CUDA 12 or 13 with cuBLAS, CMake 3.24+, a C++17 compiler, one NVIDIA GPU.

From PyPI (https://pypi.org/project/attnrank/):

```bash
conda create -n attnrank python=3.12 -y && conda activate attnrank
pip install attnrank
pip install 'attnrank[hub]'
pip install 'attnrank[research]'
python -c "import attnrank as ar; print(ar.__version__)"
```

From source, for development:

```bash
git clone https://github.com/Syun1208/attention-rank.git AttnRank && cd AttnRank
pip install '.[hub,dev]'
pytest -q
```

The PyPI package is a source distribution, so `pip install` compiles the engine for the GPU found at build time (`CMAKE_CUDA_ARCHITECTURES=native`), which takes a few minutes. Override with `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" pip install attnrank`. In a `uv` managed venv use `uv pip install --python /path/.venv/bin/python attnrank`. The `hub` extra adds `huggingface_hub` and `python-dotenv`, `research` adds torch, transformers, numpy, scipy and matplotlib for the scripts under `attnrank/services/research/`, C++ and CUDA sources sit in `attnrank/src/` with public headers in `include/attnrank/`. `pip wheel attnrank -w dist` builds a redistributable wheel for the same CUDA version and architecture.

Development build without pip:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

This writes `build/attnrank` and `build/python/_core*.so`, and `import attnrank` from the repository root finds that `.so` when the package is not installed.

Environment variables are read from a `.env` file found by walking up from the package directory (see `.env.example`): `HF_TOKEN` for gated checkpoints and `OPENAI_API_KEY` for the LLM judge.

## Dataset Preparation

Datasets live in the workspace `data/` folder (on this machine `/mnt/HDD4/longpm/AttnRank/data`, never committed): relative paths such as `data/finqa/dataset.jsonl` are looked up there, or set `ATTNRANK_DATA_DIR` / `--data-dir`. Every task accepts either a JSONL file or a Hugging Face dataset written as `hf:<repo>[:<config>]@<split>`.

```
data/
├── hotpotqa-attnrank-profile/dataset.jsonl   397 probe questions, 5 documents each
├── hotpotqa-attnrank/dataset.jsonl           7405 HotpotQA questions, 5 documents each, gold flagged
└── finqa/dataset.jsonl                       1549 FinQA rows: question (context + question), principle, gold_response
```

Row format for probes and HotpotQA (the `generic` adapter):

```json
{"id": "q1", "question": "...", "answer": "...", "documents": [{"title": "t", "text": "...", "is_gold": "true"}, {"text": "..."}]}
```

Other column names are mapped in the config, and the `hotpot_qa` adapter reads the official `hotpotqa/hotpot_qa` layout directly:

```yaml
dataset:
  hub: hotpotqa/hotpot_qa
  config: distractor
  split: validation
  adapter: hotpot_qa
probes:
  path: data/my_probes.jsonl
  fields: {question: query, documents: passages, document_text: body, document_gold: gold}
```

A custom adapter is any factory `my_module:make_adapter` returning a callable that maps one raw row to the format above (`adapter: my_module:make_adapter`). FinQA rows are read as is (`question`, `principle`, `gold_response`, optional `rejected`, `meta`), and `data/finqa/fix_labels.py` documents the label corrections applied to the source dataset.

## Evaluation / Inference

Top-k documents in relevance order go in, a slot order comes out. From Python:

```python
from pathlib import Path
import attnrank as ar

profile = ar.load_attention_profile(path=Path("profiles/profile-qwen7b-hotpotqa-fig5.json"))
ordered = ar.rerank_with_profile(documents_by_relevance=top_k_documents, profile=profile)

engine = ar.from_pretrained(model="Qwen/Qwen2.5-7B-Instruct", settings=ar.EngineSettings(device=0, chat_format="plain"))
answer = ar.generate_answer(engine=engine, question=question, documents=ordered).text
```

Building a profile for a new model or prompt format:

```python
engine = ar.from_pretrained(model="Qwen/Qwen2.5-7B-Instruct", settings=ar.EngineSettings(device=0, max_sequence=4096))
samples = ar.load_probe_samples(path=Path("data/hotpotqa-attnrank-profile/dataset.jsonl"), limit=400)
scan = ar.find_shallowest_attention_layer(engine=engine, samples=samples, settings=ar.LayerScanSettings(min_edge_ratio=1.3))
print(scan.table())
ar.save_attention_profile(profile=scan.profile_for(layer_index=scan.selected_layer), path=Path("profiles/my-profile.json"))
```

Serving with vLLM (or any OpenAI-compatible server): the profile does the reordering, `format_prompt` renders the same prompt the engine uses without loading a model, and `chat_messages` wraps it for chat endpoints. Verified against the vLLM quickstart with `LLM.generate`, `LLM.chat`, `vllm serve` plus `/v1/completions` and `/v1/chat/completions`.

```python
from vllm import LLM, SamplingParams

ranker = ar.AttnRank.from_profile(path=Path("profiles/profile-qwen7b-hotpotqa-fig5.json"))
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
params = SamplingParams(temperature=0.0, max_tokens=32)
llm.generate([ranker.format_prompt(question=question, documents_by_relevance=top_k_documents)], params)
llm.chat([ranker.chat_messages(question=question, documents_by_relevance=top_k_documents)], params)
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=ranker.chat_messages(question=question, documents_by_relevance=top_k_documents),
)
```

All functions take keyword arguments, and configuration lives in frozen dataclasses (`EngineSettings`, `LayerScanSettings`, `ProfileSettings`, `GenerationSettings`). `ar.measure_document_attention` and `ar.measure_attention_by_layer` probe a single prompt, `ar.rerank_baseline` gives the `descending`, `ascending`, `random` and `lim` orders, `ar.detect_chat_format` and `ar.model_config` inspect a checkpoint.

Command line: every task reads `--config configs/<task>/<name>.yaml`, flags override single values, and a copy of the resolved config is written next to the outputs.

| Task | Command | Output |
|---|---|---|
| Attention profile (layer scan) | `bash scripts/profile_qwen7b_fig5.sh` | `outputs/<date>_profile.qwen7b.hotpotqa.fig5/profile-qwen7b-hotpotqa-fig5.json` |
| Rerank top-k documents, optional answer | `bash scripts/rerank_example.sh` | JSON on stdout: `slot_of_rank`, `documents`, `answer` |
| HotpotQA, five ordering strategies | `bash scripts/hotpotqa_qwen7b_fig5.sh` | `outputs/<date>_hotpotqa.qwen7b.fig5/records.jsonl` and the report table |
| HotpotQA report from record files | `python main.py hotpotqa-report --records outputs/*/records.jsonl` | table measured vs Table 1 of the paper |
| FinQA, long context split into chunks | `bash scripts/finqa_qwen7b_k5.sh` | `outputs/<date>_finqa.qwen7b.k5/<run_id>/{examples.jsonl,run.json}` |
| Research scripts (traces, figures, judge) | `python main.py research placement_trace --help` | `docs/figures/`, `outputs/` |

Rerank without a model, from a JSON file holding the top-k list (`{"documents": [...]}` or a bare list):

```bash
python main.py rerank --documents docs/examples/top_k_documents.json --profile profiles/profile-qwen7b-hotpotqa-fig5.json
```

Paths: logs go to `<workspace>/logs/attnrank_<timestamp>.log`, run folders to `<workspace>/outputs/<date>_<task>.<config>/` and profile names are looked up in `<workspace>/profiles/`. The workspace defaults to the repository root (on this machine `/mnt/HDD4/longpm/AttnRank`, holding `logs/`, `outputs/`, `profiles/`, `data/` and `models/`) and is changed with `--workspace`, `--log-dir`, `--outputs-dir`, `--profiles-dir`, or the variables `ATTNRANK_WORKSPACE`, `ATTNRANK_LOG_DIR`, `ATTNRANK_OUTPUT_DIR`, `ATTNRANK_PROFILES_DIR`, `ATTNRANK_DATA_DIR` in the environment or `.env`. `output_dir` in a config or `--output-dir` fixes one run's folder.

Shards for several GPUs: `python main.py hotpotqa --config ... --offset 0 --questions 3703` and `--offset 3703`, then `hotpotqa-report` over both record files. FinQA uses `--shard 0/2`, `--shard 1/2` and `--finalize`. Scripts pin `CUDA_VISIBLE_DEVICES`, so edit the `.sh` file to change the GPU and pass extra flags through `"$@"`.

The C++ CLI is installed as `attnrank` (`attnrank inspect --model DIR`, `attnrank profile --model DIR --samples-file FILE --auto-layer --out FILE`), and `attnrank --help` lists all options.

Prompt formats: the chat template is detected from `config.json` (`model_type` `qwen*` gives `chatml`, `mistral` gives `mistral`, a numeric `sliding_window` gives `llama2`, `tokenizer.model` gives `vicuna`, otherwise `chatml`), and `chat_format: plain` reproduces the paper's Figure 5 prompt without a chat template.

## Results

HotpotQA answer accuracy (%), Qwen2.5-7B-Instruct, 7405 questions, five documents each, relevance order from BM25 over the five candidates, Figure 5 prompt, substring match. Paper row from Table 1 of Yi et al. (2025). Our profile is `profiles/profile-qwen7b-hotpotqa-fig5.json` (shallowest basin layer 2 over 397 probes). Source: `docs/reproduce_results.tex`.

| Method | Random | Descending | Ascending | LIM | AttnRank |
|---|---|---|---|---|---|
| Yi et al. (2025) | 52.32 | 53.31 | **54.64** | 52.18 | 54.55 |
| Ours (C++/CUDA) | 62.34 | 62.13 | 62.80 | 62.70 | **63.16** |

AttnRank beats random by +0.82 (p = 0.07) and descending by +1.03 (p = 0.03), and is within noise of ascending and LIM. With the gold documents placed first (ideal retriever) all structured orders beat random but are within noise of each other.

FinQA, Qwen2.5-7B-Instruct, 1549 examples, context split into 5 chunks ranked by BM25 and placed by the profile `profiles/profile-qwen2.5-7b-instruct-finqa-k5.json` (layer 4): numeric accuracy 78.11%, GPT-4o judge score 7.61/10, 81.47% of answers scored 8 or more.

## Pretrained Models

Any Llama-architecture checkpoint with `config.json`, a tokenizer and safetensors or `pytorch_model*.bin` shards: Qwen2.5 (0.5B to 7B tested), Vicuna 7B, Llama 2 and Mistral 7B formats. Saved attention profiles live in the workspace `profiles/` folder (file name gives model, dataset and chunk count). `profile-qwen1.5b-attnrank.json` was made with unknown prompt flags and does not reproduce.

## Citation

```bibtex
@article{yi2025attentionbasin,
  title   = {Attention Basin: Why Contextual Position Matters in Large Language Models},
  author  = {Yi, Zihao and Ouyang, Yuqi and Wang, Yinlong and Yang, Lanxing and Zhuo, Bingjie and Zhou, Jiahang and Wang, Yan and Huang, Xin and Qin, Shuo},
  journal = {arXiv preprint arXiv:2508.05128},
  year    = {2025}
}
```

## Acknowledgements / License

- [Attention Basin: Why Contextual Position Matters in Large Language Models](https://arxiv.org/abs/2508.05128), Yi et al., 2025, the method implemented here (paper PDF in `docs/AttentionBasin.pdf`).
- Checkpoints and datasets are fetched through `huggingface_hub` and `datasets`, and the Python module is built with pybind11 and scikit-build-core.
- MIT license.
