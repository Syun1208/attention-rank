#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT_ROOT}"
"${PYTHON}" main.py rerank \
    --documents docs/examples/top_k_documents.json \
    --profile profile-qwen7b-hotpotqa-fig5.json \
    --question "Which country does the composer of the song Nhu Mot Loi Chia Tay come from?" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device 0 \
    --chat-format plain \
    "$@"
