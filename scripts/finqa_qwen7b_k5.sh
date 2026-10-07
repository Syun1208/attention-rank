#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=python
RUN_DATE=$(date +%F)
OUTPUT_DIR=outputs/${RUN_DATE}_finqa.qwen7b.k5
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT_ROOT}"
"${PYTHON}" main.py finqa \
    --config configs/finqa/qwen7b.k5.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --device 0 \
    "$@"
