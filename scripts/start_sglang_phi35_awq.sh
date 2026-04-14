#!/usr/bin/env bash
set -euo pipefail

: "${SGLANG_PORT:=30000}"
: "${SGLANG_MODEL:=HuggingFaceTB/SmolLM2-360M-Instruct}"
: "${SGLANG_MEM_FRACTION:=0.45}"

python -m sglang.launch_server \
  --model-path "${SGLANG_MODEL}" \
  --port "${SGLANG_PORT}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION}"
